import redis
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

# Import your other modules here
from .llm_query import query_openrouter, query_openai
from .utils import calculate_mention_rate

load_dotenv()

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
r = redis.Redis.from_url(REDIS_URL)

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

# Redis keys
JOB_QUEUE_KEY = "job_queue"
JOB_TRACKER_PREFIX = "job_tracker:"
WORKER_BUSY_KEY = "worker_busy"
RATE_LIMITER_KEY = "rate_limiter"

class RedisRateLimiter:
    def __init__(self, max_requests: int, interval_s: float):
        self.max_requests = max_requests
        self.interval = interval_s

    def wait_for_slot(self):
        while True:
            current_time = time.time()
            # Remove old timestamps
            r.zremrangebyscore(RATE_LIMITER_KEY, "-inf", current_time - self.interval)
            
            # Count current requests
            count = r.zcount(RATE_LIMITER_KEY, current_time - self.interval, current_time)
            
            if count < self.max_requests:
                r.zadd(RATE_LIMITER_KEY, {str(uuid.uuid4()): current_time})
                return
            time.sleep(0.05)
            
    def can_request(self) -> bool:
        current_time = time.time()
        r.zremrangebyscore(RATE_LIMITER_KEY, "-inf", current_time - self.interval)
        return r.zcount(RATE_LIMITER_KEY, current_time - self.interval, current_time) < self.max_requests

# Initialize rate limiter
RATE_LIMITER = RedisRateLimiter(max_requests=100, interval_s=60.0)
# RATE_LIMITER = RedisRateLimiter(max_requests=1, interval_s=10.0)

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
    """Background worker that processes queued requests from Redis"""
    while True:
        try:
            # Blocking pop from Redis queue (wait up to 10 seconds)
            job_data = r.blpop(JOB_QUEUE_KEY, timeout=10)
            
            if not job_data:
                continue
                
            _, data = job_data
            job = json.loads(data)
            job_id = job["job_id"]
            brand = job["brand"]
            prompts = job["prompts"]
            
            # Set worker as busy with expiration
            r.set(WORKER_BUSY_KEY, "processing", ex=3600)
            
            logger.info(f"Starting job {job_id} with {len(prompts)} prompts")
            
            # Update job status to processing
            job_tracker = {
                "status": "processing",
                "started_at": time.time(),
                "progress": 0,
                "total_prompts": len(prompts)
            }
            r.set(f"{JOB_TRACKER_PREFIX}{job_id}", json.dumps(job_tracker))
            
            # Process all prompts
            responses = {model: [] for model in MODEL_IDS}
            total_prompts = len(prompts)
            completed = 0
            
            for idx, prompt in enumerate(prompts):
                try:
                    model_results = process_prompt(prompt)
                    for model, response in model_results.items():
                        responses[model].append({
                            "prompt": prompt,
                            "response": response
                        })
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    for model in MODEL_IDS:
                        responses[model].append({
                            "prompt": prompt,
                            "response": error_msg
                        })
                    logger.error(f"Error processing prompt: {str(e)}")
                
                # Update progress
                completed += 1
                progress = (completed / total_prompts) * 100
                job_tracker["progress"] = progress
                r.set(f"{JOB_TRACKER_PREFIX}{job_id}", json.dumps(job_tracker))
            
            # Calculate results
            results = {}
            for model in MODEL_IDS:
                mention_rate = calculate_mention_rate(responses[model], brand)
                mentioned, not_mentioned = segregate_prompts_by_mention(responses[model], brand)
                results[model] = {
                    "mention_rate": mention_rate,
                    "mentioned": mentioned[:3],
                    "not_mentioned": not_mentioned[:3]
                }
            
            # Mark job as complete
            job_tracker = {
                "status": "completed",
                "completed_at": time.time(),
                "result": {
                    "brand": brand,
                    "total_prompts": total_prompts,
                    "openAi_mention_rate": results["openai"]["mention_rate"],
                    "gemini_mention_rate": results["gemini"]["mention_rate"],
                    "perplexity_mention_rate": results["perplexity"]["mention_rate"],
                    "deepseek_mention_rate": results["deepseek"]["mention_rate"],
                    "claude_mention_rate": results["claude"]["mention_rate"],
                    "segregated_prompts": {
                        "openai": {
                            "mentioned": results["openai"]["mentioned"],
                            "not_mentioned": results["openai"]["not_mentioned"]
                        },
                        "gemini": {
                            "mentioned": results["gemini"]["mentioned"],
                            "not_mentioned": results["gemini"]["not_mentioned"]
                        },
                        "perplexity": {
                            "mentioned": results["perplexity"]["mentioned"],
                            "not_mentioned": results["perplexity"]["not_mentioned"]
                        },
                        "deepseek": {
                            "mentioned": results["deepseek"]["mentioned"],
                            "not_mentioned": results["deepseek"]["not_mentioned"]
                        },
                        "claude": {
                            "mentioned": results["claude"]["mentioned"],
                            "not_mentioned": results["claude"]["not_mentioned"]
                        }
                    }
                }
            }
            r.set(f"{JOB_TRACKER_PREFIX}{job_id}", json.dumps(job_tracker))
            
            logger.info(f"Completed job {job_id}")
            
        except Exception as e:
            logger.error(f"Worker error: {str(e)}")
            if job_id:
                job_tracker = {
                    "status": "failed",
                    "completed_at": time.time(),
                    "error": str(e)
                }
                r.set(f"{JOB_TRACKER_PREFIX}{job_id}", json.dumps(job_tracker))
        finally:
            # Clear worker busy status
            r.delete(WORKER_BUSY_KEY)
            # Update queue positions for remaining jobs
            update_queue_positions()

# Start worker thread
def start_worker():
    # Only start if no worker is active
    if not r.get(WORKER_BUSY_KEY):
        worker = threading.Thread(target=worker_thread, daemon=True)
        worker.start()
        logger.info("Started worker thread")

@api_view(['POST'])
def brand_mention_score(request):
    worker_id = f"{os.getpid()}-{threading.get_ident()}"
    logger.info(f"Handling request on worker: {worker_id}")
    
    start_worker()  # Ensure worker is running
    
    brand = request.data.get("brand")
    prompts = request.data.get("prompts", [])
    if not brand or not prompts:
        return Response({"error": "brand and prompts are required"}, status=400)
    
    # ALWAYS QUEUE THE JOB
    job_id = str(uuid.uuid4())
    job_data = {
        "job_id": job_id,
        "brand": brand,
        "prompts": prompts
    }
    
    # Add to Redis queue
    r.rpush(JOB_QUEUE_KEY, json.dumps(job_data))
    
    # Get queue position
    queue_size = r.llen(JOB_QUEUE_KEY)
    position = queue_size  # Since we just added to end
    
    # Create job tracker entry
    job_tracker = {
        "status": "queued",
        "created_at": time.time(),
        "brand": brand,
        "prompt_count": len(prompts),
        "position_in_queue": position
    }
    r.set(f"{JOB_TRACKER_PREFIX}{job_id}", json.dumps(job_tracker))
    
    # Calculate wait time estimate
    wait_seconds = position * 10  # Simplified estimation
    
    response_data = {
        "status": "queued",
        "message": "Request queued",
        "job_id": job_id,
        "position_in_queue": position,
        "estimated_wait_seconds": wait_seconds
    }
    
    return Response(response_data, status=202)

@api_view(['POST'])
def job_status(request):
    job_id = request.data.get("job_id")
    job_data = r.get(f"{JOB_TRACKER_PREFIX}{job_id}")
    
    if not job_data:
        return Response({"error": "Job not found"}, status=404)
    
    job = json.loads(job_data)
    response_data = {
        "job_id": job_id,
        "status": job["status"],
        "created_at": job.get("created_at"),
    }
    
    if job["status"] == "queued":
        response_data["position_in_queue"] = job.get("position_in_queue", 0)
        response_data["estimated_wait_seconds"] = job.get("position_in_queue", 0) * 10
    
    elif job["status"] == "processing":
        response_data["started_at"] = job.get("started_at")
        response_data["progress"] = f"{job.get('progress', 0):.1f}%"
        response_data["total_prompts"] = job.get("total_prompts", 0)
    
    elif job["status"] == "completed":
        response_data["completed_at"] = job.get("completed_at")
        response_data["data"] = job.get("result", {})
    
    elif job["status"] == "failed":
        response_data["completed_at"] = job.get("completed_at")
        response_data["error"] = job.get("error", "Unknown error")
    
    return Response(response_data)

# Keep all other functions (run_query, generate_prompts, health_check) unchanged     
        
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


def update_queue_positions():
    """Update queue positions for all queued jobs"""
    queued_jobs = r.lrange(JOB_QUEUE_KEY, 0, -1)
    
    for position, job_data in enumerate(queued_jobs, start=1):
        try:
            job = json.loads(job_data)
            job_id = job["job_id"]
            tracker_key = f"{JOB_TRACKER_PREFIX}{job_id}"
            
            job_tracker_data = r.get(tracker_key)
            if job_tracker_data:
                job_tracker = json.loads(job_tracker_data)
                if job_tracker.get("status") == "queued":
                    job_tracker["position_in_queue"] = position
                    r.set(tracker_key, json.dumps(job_tracker))
        except Exception as e:
            logger.error(f"Error updating queue position: {str(e)}")
            
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

    
    prompt =prompt_template.format(brand=brand, website=website, custom_comments=custom_comments or "")

    print(f"Generated Prompt: {prompt}")

    try:
        OPENAI_MODEL = os.getenv("OPENAI_MODEL", "openai/o4-mini")
        o_response = query_openai(prompt, OPENAI_MODEL)
        # g_response_array = [p.strip() for p in g_response.split(';') if p.strip()]
        o_response_array = [p.strip() for p in o_response.split(';') if p.strip()]
        results = {
            # "gemini": g_response_array,
            "openai": o_response_array
        }

        return Response({"results": results})

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
 