import json
import os
import logging
import threading
import time
import uuid
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import StreamingHttpResponse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from django.db import transaction
from django.utils import timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from redis import Redis
from redis.exceptions import LockError

# Import your other modules here
from .llm_query import query_openrouter, query_openai
from .utils import calculate_mention_rate
from .local_rate_limiter import shared_limiter

from .models import Job

from celery import shared_task  

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# This lock ensures that only one job is processed at a time
# Global processing lock
processing_lock = Lock()
current_processing_job = None

# Constants
MODEL_IDS = {
    "gemini": "google/gemini-2.5-flash",
    "openai": "openai/o4-mini",
    "perplexity": "perplexity/sonar-pro",
    "deepseek": "deepseek/deepseek-r1-0528",
    "claude": "anthropic/claude-3.5-haiku-20241022:beta"
}


# Worker configuration here
NUM_OUTER_WORKERS = int(os.getenv("NUM_OUTER_WORKERS", 15))  
PROMPT_WORKERS = int(os.getenv("PROMPT_WORKERS", 5))  

PYTHON_ENVIRONMENT = os.getenv('PYTHON_ENVIRONMENT', 'development').lower()
IS_DEVELOPMENT = (PYTHON_ENVIRONMENT == 'development')


if not IS_DEVELOPMENT:
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('http.client').setLevel(logging.WARNING)


def log_conditionally(level, msg, *args, **kwargs):
    """
    Logs a message only if in development environment or if the log level
    is WARNING, ERROR, or CRITICAL.
    """
    if IS_DEVELOPMENT or level >= logging.WARNING:
        logger.log(level, msg, *args, **kwargs)

def print_env_variables():
    print("➡️  NUM_OUTER_WORKERS:", os.getenv('NUM_OUTER_WORKERS', 'Not Set'))
    print("➡️  PROMPT_WORKERS:", os.getenv('PROMPT_WORKERS', 'Not Set'))
    print("➡️  RATE_INTERVAL_S:", os.getenv('RATE_INTERVAL_S', 'Not Set'))
    print("➡️  RATE_LIMIT_MAX:", os.getenv('RATE_LIMIT_MAX', 'Not Set'))
    print("➡️  NUMBER_OF_PROMPTS:", os.getenv('NUMBER_OF_PROMPTS', 'Not Set'))
    print("➡️  IS_CELERY_WORKER_ON:", os.getenv('IS_CELERY_WORKER_ON', 'Not Set'))
    
            

def query_openrouter_limited(prompt: str, model_id: str) -> str:
    shared_limiter.wait_for_slot()
    return query_openrouter(prompt, model_id)

def process_prompt(prompt):
    """Process a single prompt with all models in parallel"""
    try: 
        with ThreadPoolExecutor(max_workers=5) as inner_executor:
            futures = {
                model: inner_executor.submit(query_openrouter_limited, prompt, model_id)
                for model, model_id in MODEL_IDS.items()
            }
            results = {model: future.result() for model, future in futures.items()}
            return results
    except Exception as e:
        logger.error(f"process_prompt failed: {str(e)}")
        return {model: "" for model in MODEL_IDS}


# @shared_task(bind=True, max_retries=3)
# def process_brand_mention(self, job_id: str):
    """
    Processes a single job with strict sequential execution
    Only one job runs at a time - others wait in queue
    Minimal Redis connections used
    """
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 Starting processing for job: {job_id}")
    
    # raise Exception("THIS TASK IS RUNNING SYNCHRONOUSLY! Traceback will show where.")
    # >>> REMOVE THIS LINE AFTER TESTING <<<

    
    # Minimal Redis connection - created only when needed
    redis_client = None
    lock = None
    job = None
    
    try:

        try:
            job = Job.objects.get(job_id=job_id)
            logger.info(f"🔍 Found job: {job_id} with status: {job.status}")
        except Job.DoesNotExist:
            logger.error(f"❌ Job {job_id} does not exist in database")
            return

        # 2. Skip if not queued
        if job.status != Job.STATUS_QUEUED:
            logger.warning(f"⏩ Job {job_id} already processed (status: {job.status})")
            return


        # 3. Create Redis connection (only when needed)
        redis_url = os.getenv('REDIS_URL')
        
        if not redis_url:
            logger.error("❌ Could not connect to Redis")
            raise ValueError("❌ Could not connect to Redis. Please set the REDIS_URL environment variable.")
        

        redis_client = Redis.from_url(redis_url, max_connections=1)
        
        # 4. Acquire global lock (ensures only one job runs at a time)
        lock = redis_client.lock("global_job_processing_lock", timeout=3600)
        acquired = lock.acquire(blocking=True, blocking_timeout=30)
        
        if not acquired:
            logger.warning(f"⏳ Could not acquire lock for job {job_id}")
            # Retry after short delay without holding connection
            raise self.retry(countdown=5)
        
        logger.info(f"🔒 Lock acquired for job {job_id}")
        
        # 5. Get and update job status
        with transaction.atomic():
            job = Job.objects.select_for_update().get(
                job_id=job_id,
                status=Job.STATUS_QUEUED
            )
            job.status = Job.STATUS_PROCESSING
            job.started_at = timezone.now()
            job.progress = 0.0
            job.save()
        
        logger.info(f"🧵 Started processing job: {job.job_id}")
        
        # 6. Process prompts (original logic)
        prompts = job.prompts or []
        total = len(prompts)
        responses = {model: [] for model in MODEL_IDS}
        
        for idx, prompt in enumerate(prompts, start=1):
            
            # Process all models for this prompt
            model_outputs = process_prompt(prompt)
            
            for model, output in model_outputs.items():
                responses[model].append({
                    "prompt": prompt,
                    "response": output
                })
            
            # Update progress
            job.progress = (idx / total) * 100.0
            job.save(update_fields=["progress"])
        
        # 6. Final calculations
        final = {}
        for model, reps in responses.items():
            rate = calculate_mention_rate(reps, job.brand)
            yes, no = segregate_prompts_by_mention(reps, job.brand)
            final[model] = {
                "mention_rate": rate,
                "mentioned": yes[:3],
                "not_mentioned": no[:3],
            }
        
        # 7. Save results
        job.result = {
            "brand": job.brand,
            "total_prompts": total,
            **{f"{m}_mention_rate": final[m]["mention_rate"] for m in MODEL_IDS},
            "segregated_prompts": final,
        }
        job.status = Job.STATUS_COMPLETED
        job.completed_at = timezone.now()
        job.save()
        logger.info(f"✅ Completed job: {job.job_id}")
        logger.info(f"Sleeping for 120 seconds..")
        print(f"✅ Job {job_id} completed and sleeping for 120 seconds")
        logger.warning(f"Sleeping for 120 seconds..")
        time.sleep(10)
        
    except Job.DoesNotExist:
        logger.warning(f"⏩ Job {job_id} already processed or doesn't exist")
    except Exception as e:
        logger.exception(f"❌ Job processing failed: {str(e)}")
        if job:
            job.status = Job.STATUS_FAILED
            job.error = str(e)
            job.completed_at = timezone.now()
            job.save()
        # Retry after delay
        raise self.retry(exc=e, countdown=60)
    finally:
        # 7. Always release resources
        try:
            if lock and lock.locked():
                lock.release()
                logger.info(f"🔓 Lock released for job {job_id}")
        except LockError:
            pass
            
        if redis_client:
            try:
                redis_client.close()
            except Exception:
                pass

# @shared_task(bind=True, max_retries=3)
# def process_brand_mention(self, job_id: str):
#     """
#     Processes a single job by running its prompts in parallel using a ThreadPoolExecutor.
#     """
#     logger.info(f"🚀 Starting processing for job: {job_id}")

#     redis_client = None
#     lock = None
#     job = None

#     try:
#         try:
#             job = Job.objects.get(job_id=job_id)
#             logger.info(f"🔍 Found job: {job_id} with status: {job.status}")
#         except Job.DoesNotExist:
#             logger.error(f"❌ Job {job_id} does not exist in database")
#             return

#         if job.status != Job.STATUS_QUEUED:
#             logger.warning(f"⏩ Job {job_id} already processed (status: {job.status})")
#             return

#         redis_url = os.getenv('REDIS_URL')
#         if not redis_url:
#             logger.error("❌ REDIS_URL environment variable not set.")
#             raise ValueError("❌ Could not connect to Redis. Please set the REDIS_URL environment variable.")

#         redis_client = Redis.from_url(redis_url, max_connections=1)

#         lock = redis_client.lock("global_job_processing_lock", timeout=3600)
#         acquired = lock.acquire(blocking=True, blocking_timeout=30)

#         if not acquired:
#             logger.warning(f"⏳ Could not acquire lock for job {job_id}")
#             raise self.retry(countdown=5)

#         logger.info(f"🔒 Lock acquired for job {job_id}")

#         with transaction.atomic():
#             job = Job.objects.select_for_update().get(
#                 job_id=job_id,
#                 status=Job.STATUS_QUEUED
#             )
#             job.status = Job.STATUS_PROCESSING
#             job.started_at = timezone.now()
#             job.progress = 0.0
#             job.save()

#         logger.info(f"🧵 Started processing job: {job.job_id}")

#         prompts = job.prompts or []
#         total_prompts = len(prompts)
        
#         prompt_workers_count = int(os.getenv('PROMPT_WORKERS', '5'))
#         if prompt_workers_count < 10:
#             print(f"❗ Invalid PROMPT_WORKERS value : {prompt_workers_count}, defaulting to 10 worker.")
#             prompt_workers_count = 10
        
#         logger.info(f"🚀 Processing {total_prompts} prompts with {prompt_workers_count} parallel workers.")

#         responses = {model: [] for model in MODEL_IDS}
#         processed_count = 0

#         with ThreadPoolExecutor(max_workers=prompt_workers_count) as executor:
            
#             future_to_prompt = {executor.submit(process_prompt, prompt): prompt for prompt in prompts}
            
#             for future in as_completed(future_to_prompt):
#                 original_prompt = future_to_prompt[future]
#                 try:
#                     model_outputs = future.result() 
                    
#                     for model, output in model_outputs.items():
#                         responses[model].append({
#                             "prompt": original_prompt,
#                             "response": output
#                         })
                    
#                     processed_count += 1
                    
#                     with transaction.atomic():
#                         job.progress = (processed_count / total_prompts) * 100.0
#                         job.save(update_fields=["progress"])
#                     logger.info(f"Progress for job {job_id}: {job.progress:.2f}%")

#                 except Exception as exc:
#                     logger.error(f"❌ Prompt '{original_prompt}' generated an exception: {exc}")
#                     # - Continue processing other prompts, log the error for this one.

#         # Final calculations after all prompts are processed
#         final = {}
#         for model, reps in responses.items():
#             rate = calculate_mention_rate(reps, job.brand)
#             yes, no = segregate_prompts_by_mention(reps, job.brand)
#             final[model] = {
#                 "mention_rate": rate,
#                 "mentioned": yes[:3],
#                 "not_mentioned": no[:3],
#             }

#         # Save results
#         job.result = {
#             "brand": job.brand,
#             "total_prompts": total_prompts,
#             **{f"{m}_mention_rate": final[m]["mention_rate"] for m in MODEL_IDS},
#             "segregated_prompts": final,
#         }
#         job.status = Job.STATUS_COMPLETED
#         job.completed_at = timezone.now()
#         job.save()
#         logger.info(f"✅ Completed job: {job.job_id}")
#         logger.info(f"Sleeping for 10 seconds..") # Changed from 120
#         print(f"✅ Job {job_id} completed and sleeping for 10 seconds")
#         logger.warning(f"Sleeping for 10 seconds..") # Changed from 120
#         time.sleep(10) # Changed from 120

#     except Job.DoesNotExist:
#         logger.warning(f"⏩ Job {job_id} already processed or doesn't exist")
#     except Exception as e:
#         logger.exception(f"❌ Job processing failed: {str(e)}")
#         if job:
#             job.status = Job.STATUS_FAILED
#             job.error = str(e)
#             job.completed_at = timezone.now()
#             job.save()
#         raise self.retry(exc=e, countdown=60)
#     finally:
#         try:
#             if lock and lock.locked():
#                 lock.release()
#                 logger.info(f"🔓 Lock released for job {job_id}")
#         except LockError:
#             pass # Lock might have expired or released by another process (shouldn't happen with global lock)

#         if redis_client:
#             try:
#                 redis_client.close()
#             except Exception:
#                 pass

@shared_task(bind=True, max_retries=3)
def process_brand_mention(self, job_id: str):
    """
    Processes a single job by running its prompts in parallel using a ThreadPoolExecutor.
    """
    log_conditionally(logging.INFO, f"🚀 Starting processing for job: {job_id}")

    redis_client = None
    lock = None
    job = None

    try:
        try:
            job = Job.objects.get(job_id=job_id)
            log_conditionally(logging.INFO, f"🔍 Found job: {job_id} with status: {job.status}")
        except Job.DoesNotExist:
            log_conditionally(logging.ERROR, f"❌ Job {job_id} does not exist in database")
            return

        if job.status != Job.STATUS_QUEUED:
            log_conditionally(logging.WARNING, f"⏩ Job {job_id} already processed (status: {job.status})")
            return

        redis_url = os.getenv('REDIS_URL')
        if not redis_url:
            logger.error("❌ REDIS_URL environment variable not set.")
            raise ValueError("❌ Could not connect to Redis. Please set the REDIS_URL environment variable.")

        redis_client = Redis.from_url(redis_url, max_connections=1)

        lock = redis_client.lock("global_job_processing_lock", timeout=3600)
        acquired = lock.acquire(blocking=True, blocking_timeout=30)

        if not acquired:
            logger.warning(f"⏳ Could not acquire lock for job {job_id}")
            raise self.retry(countdown=5)

        log_conditionally(logging.INFO, f"🔒 Lock acquired for job {job_id}")

        with transaction.atomic():
            job = Job.objects.select_for_update().get(
                job_id=job_id,
                status=Job.STATUS_QUEUED
            )
            job.status = Job.STATUS_PROCESSING
            job.started_at = timezone.now()
            job.progress = 0.0
            job.save()

        logger.info(f"🧵 Started processing job: {job.job_id}")

        prompts = job.prompts or []
        total_prompts = len(prompts)
        
        prompt_workers_count = int(os.getenv('PROMPT_WORKERS', '10'))
        
        log_conditionally(logging.INFO, f"🚀 Processing {total_prompts} prompts with {prompt_workers_count} parallel workers.")

        responses = {model: [] for model in MODEL_IDS}
        processed_count = 0

        with ThreadPoolExecutor(max_workers=prompt_workers_count) as executor:
            future_to_prompt = {executor.submit(process_prompt, prompt): prompt for prompt in prompts}
            
            for future in as_completed(future_to_prompt):
                original_prompt = future_to_prompt[future]
                try:
                    model_outputs = future.result() 
                    
                    for model, output in model_outputs.items():
                        responses[model].append({
                            "prompt": original_prompt,
                            "response": output
                        })
                    
                    processed_count += 1
                    
                    with transaction.atomic():
                        job.progress = (processed_count / total_prompts) * 100.0
                        job.save(update_fields=["progress"])
                    log_conditionally(logging.INFO, f"Progress for job {job_id}: {job.progress:.2f}%")

                except Exception as exc:
                    log_conditionally(logging.ERROR, f"❌ Prompt '{original_prompt}' generated an exception: {exc}")
                    # Continue processing other prompts

        # Final calculations after all prompts are processed
        final = {}
        for model, reps in responses.items():
            rate = calculate_mention_rate(reps, job.brand)
            yes, no = segregate_prompts_by_mention(reps, job.brand)
            final[model] = {
                "mention_rate": rate,
                "mentioned": yes[:3],
                "not_mentioned": no[:3],
            }

        job.result = {
            "brand": job.brand,
            "total_prompts": total_prompts,
            **{f"{m}_mention_rate": final[m]["mention_rate"] for m in MODEL_IDS},
            "segregated_prompts": final,
        }
        job.status = Job.STATUS_COMPLETED
        job.completed_at = timezone.now()
        job.save()

        logger.info(f"✅ Completed job: {job.job_id}")
        

    except Job.DoesNotExist:
        log_conditionally(logging.WARNING, f"⏩ Job {job_id} already processed or doesn't exist")
    except Exception as e:
        logger.exception(f"❌ Job processing failed: {str(e)}")
        if job:
            job.status = Job.STATUS_FAILED
            job.error = str(e)
            job.completed_at = timezone.now()
            job.save()
        raise self.retry(exc=e, countdown=60)
    finally:
        try:
            if lock and lock.locked():
                lock.release()
                log_conditionally(logging.INFO, f"🔓 Lock released for job {job_id}")
        except LockError:
            pass

        if redis_client:
            try:
                redis_client.close()
            except Exception:
                pass






@api_view(['POST'])
def brand_mention_score(request):
    logger.info(f"Handling request on worker: {os.getpid()}-{threading.get_ident()}")
    brand = request.data.get("brand")
    prompts = request.data.get("prompts", [])

    if not brand or not prompts:
        return Response({"error": "brand and prompts are required"}, status=400)

    job = Job.objects.create(
        brand=brand,
        prompts=prompts,
        status=Job.STATUS_QUEUED  
    )
    job.save()
    
    print(f"➡️  Created job: {job.job_id} with status: {Job.STATUS_QUEUED}")
    
    # Enqueue in Celery
    process_brand_mention.delay(str(job.job_id))

    position = Job.objects.filter(
        status=Job.STATUS_QUEUED,
        created_at__lte=job.created_at
    ).count()

    wait_seconds = position * 200  # Assuming each job takes ~200 seconds to process

    return Response({
        "status": "queued",
        "job_id": str(job.job_id),
        "created_at": job.created_at.isoformat(),
        "position_in_queue": position,
        "estimated_wait_seconds": wait_seconds
    }, status=202)


@api_view(['POST'])
def job_status(request):
    job_id = request.data.get("job_id")
    if not job_id:
        return Response({"error": "job_id is required"}, status=400)

    try:
        job = Job.objects.get(job_id=job_id)
    except Job.DoesNotExist:
        return Response({"error": "Job not found"}, status=404)

    response_data = {
        "job_id": job_id,
        "status": job.status,
        "created_at": job.created_at.isoformat(),
    }

    if job.status == Job.STATUS_QUEUED:
        # count how many queued jobs were created before or at this one
        position = Job.objects.filter(
            status=Job.STATUS_QUEUED,
            created_at__lte=job.created_at
        ).count()
        response_data.update({
            "position_in_queue": position,
            "estimated_wait_seconds": position * 10
        })

    elif job.status == Job.STATUS_PROCESSING:
        response_data.update({
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "progress": f"{job.progress:.1f}%",
            "total_prompts": len(job.prompts or [])
        })

    elif job.status == Job.STATUS_COMPLETED:
        response_data.update({
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "data": job.result or {}
        })

    elif job.status == Job.STATUS_FAILED:
        response_data.update({
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "error": job.error or "Unknown error"
        })

    return Response(response_data)

        
def segregate_prompts_by_mention(responses, brand_name):
    """
    Splits prompts into those where the given brand_name
    *actually appears* in its response vs. those where it does not.
    """
    mentioned, not_mentioned = [], []
    
    for item in responses:
        prompt_text = item.get("prompt", "")
        resp       = item.get("response", "") or ""
        if brand_name.lower() in resp.lower():
            mentioned.append(prompt_text)
        else:
            not_mentioned.append(prompt_text)
    
    return mentioned, not_mentioned


@api_view(['GET'])
def health_check(request):
    port = os.getenv("PORT", "8000")
    return Response({"status": "ok", "message": f"Server running on PORT {port}"}, status=200)

  
@api_view(['POST'])
def generate_prompts(request):
    brand = request.data.get('brand')
    website = request.data.get('website')
    custom_comments = request.data.get('custom_comments', "")
   
    if not brand or not website:
        return Response({"error": "Brand and Website are required"}, status=status.HTTP_400_BAD_REQUEST)
    
    NUMBER_OF_PROMPTS = int(os.getenv("NUMBER_OF_PROMPTS", "5"))
    
    
    prompt_template = (
        f"I have a brand/product/application known as {brand}."
        f"It has a website at {website}. "
        f"{('Some custom comments about my platform are: ' + custom_comments + '. ') if custom_comments else ''}"
        f"Use the information provided above to generate a list of {NUMBER_OF_PROMPTS} prompts which would potentially mention my platform in their response if a user searches over the web for platforms similar to mine or for platforms in the same category. Give the prompts imagining that you're a random user, who does not know about my platform, but is looking for a platform which has the same features and use cases as mine. "
        f"(In your response , I only need the prompts separated by semicolons, in a txt format, not markdown, and no extra text with it. Keep the prompts short and concise. Do not include any brand names or specific product names in the prompts, just the specific use cases that the user might be looking for.)"
    )

    
    prompt = prompt_template.format(brand=brand, website=website, custom_comments=custom_comments or "")

    print(f"Generated Prompt: {prompt}")

    try:
        OPENAI_MODEL = os.getenv("OPENAI_MODEL", "openai/o4-mini")
        o_response = query_openai(prompt, OPENAI_MODEL)
        o_response_array = [p.strip() for p in o_response.split(';') if p.strip()]
        results = {
            "openai": o_response_array
        }

        return Response({"results": results})

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    
def check_background_workers():
    """Temporary function to verify no background workers are running"""
    for thread in threading.enumerate():
        if "worker" in thread.name.lower():
            logger.warning(f"⚠️ Background worker still active: {thread.name}")
            return True
    return False

@api_view(['GET'])
def worker_check(request):
    has_workers = check_background_workers()
    return Response({
        "has_background_workers": has_workers,
        "active_threads": [t.name for t in threading.enumerate()]
    })