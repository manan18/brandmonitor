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


# Import your other modules here
from .llm_query import query_openrouter, query_openai
from .utils import calculate_mention_rate
from .local_rate_limiter import InMemoryRateLimiter

from .models import Job

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

def print_env_variables():
    print("➡️  NUM_OUTER_WORKERS:", os.getenv('NUM_OUTER_WORKERS', 'Not Set'))
    print("➡️  PROMPT_WORKERS:", os.getenv('PROMPT_WORKERS', 'Not Set'))
    print("➡️  RATE_INTERVAL_S:", os.getenv('RATE_INTERVAL_S', 'Not Set'))
    print("➡️  RATE_LIMIT_MAX:", os.getenv('RATE_LIMIT_MAX', 'Not Set'))
    print("➡️  DATABASE_URL:", os.getenv('DATABASE_URL', 'Not Set'))
            


RATE_LIMITER = InMemoryRateLimiter(
    rate_per_sec = float(os.getenv("RATE_LIMIT_MAX", 100)) / float(os.getenv("RATE_INTERVAL_S", 60)),
    burst        = int(os.getenv("RATE_LIMIT_MAX", 100)),
)


def query_openrouter_limited(prompt: str, model_id: str) -> str:
    RATE_LIMITER.wait_for_slot()
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



def worker_thread():
    # give each outer worker its own prompt executor
    prompt_executor = ThreadPoolExecutor(max_workers=PROMPT_WORKERS)

    while True:
        job = None
        try:
            # 1) Atomically grab one queued job
            with transaction.atomic():
                job = (
                    Job.objects
                       .select_for_update(skip_locked=True)
                       .filter(status=Job.STATUS_QUEUED)
                       .order_by("created_at")
                       .first()
                )
                if not job:
                    continue

                job.status     = Job.STATUS_PROCESSING
                job.started_at = timezone.now()
                job.progress   = 0.0
                job.save(update_fields=["status","started_at","progress"])

            prompts        = job.prompts
            total_prompts  = len(prompts)
            completed      = 0
            lock           = threading.Lock()
            responses      = {m: [] for m in MODEL_IDS}

            # 2) Submit all prompts into this worker’s prompt_executor
            future_to_prompt = {
                prompt_executor.submit(process_prompt, p): p
                for p in prompts
            }

            # 3) As each prompt finishes, record results and update progress
            for future in as_completed(future_to_prompt):
                prompt_text = future_to_prompt[future]
                try:
                    model_results = future.result()
                    for m, resp in model_results.items():
                        responses[m].append({
                            "prompt":  prompt_text,
                            "response": resp
                        })
                except Exception as e:
                    err = str(e)
                    for m in MODEL_IDS:
                        responses[m].append({
                            "prompt":  prompt_text,
                            "response": f"Error: {err}"
                        })

                with lock:
                    completed += 1
                    pct = (completed / total_prompts) * 100.0
                    job.progress = pct
                    job.save(update_fields=["progress"])

            # 4) Compute final results
            final = {}
            for m in MODEL_IDS:
                rate = calculate_mention_rate(responses[m], job.brand)
                yes, no = segregate_prompts_by_mention(responses[m], job.brand)
                final[m] = {
                    "mention_rate":    rate,
                    "mentioned":       yes[:3],
                    "not_mentioned":   no[:3]
                }

            # 5) Mark job completed
            job.result       = {
                "brand":           job.brand,
                "total_prompts":   total_prompts,
                **{f"{m}_mention_rate": final[m]["mention_rate"] for m in MODEL_IDS},
                "segregated":      final
            }
            job.status        = Job.STATUS_COMPLETED
            job.completed_at  = timezone.now()
            job.save(update_fields=["result","status","completed_at"])

        except Exception as exc:
            # Mark job failed if we grabbed one
            if job:
                job.status       = Job.STATUS_FAILED
                job.error        = str(exc)
                job.completed_at = timezone.now()
                job.save(update_fields=["status","error","completed_at"])
                  

if os.environ.get("RUN_MAIN") == "true":
    for _ in range(NUM_OUTER_WORKERS):  
        threading.Thread(target=worker_thread, daemon=True).start()
    logger.info(f"Started {NUM_OUTER_WORKERS} worker threads")  


@api_view(['POST'])
def brand_mention_score(request):
    logger.info(f"Handling request on worker: {os.getpid()}-{threading.get_ident()}")
    brand = request.data.get("brand")
    prompts = request.data.get("prompts", [])

    if not brand or not prompts:
        return Response({"error": "brand and prompts are required"}, status=400)

    # Enqueue into Postgres
    job = Job.objects.create(
        brand=brand,
        prompts=prompts
    )

    # Position in queue = number still waiting (including this one)
    position = Job.objects.filter(status=Job.STATUS_QUEUED).count()

    # Estimate wait (e.g. 10s per position)
    wait_seconds = position * 10

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