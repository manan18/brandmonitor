from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from .llm_query import query_openai, query_openrouter
from .sentiment import get_sentiment
from .theme_extraction import extract_themes
import os
from dotenv import load_dotenv
from .utils import calculate_mention_rate
from concurrent.futures import ThreadPoolExecutor, as_completed
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import StreamingHttpResponse

import threading
import time
import uuid
import queue
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from rest_framework.decorators import api_view
from rest_framework.response import Response
import json

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "")
MODEL_IDS = {
        "gemini": "google/gemini-2.5-flash",
        "openai": "openai/o4-mini",
        "perplexity": "perplexity/sonar-pro",
        "deepseek": "deepseek/deepseek-r1-0528",
        "claude": "anthropic/claude-3.5-haiku-20241022:beta"
    }
# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Global request queue and job tracker
request_queue = queue.Queue()
job_tracker = {}
job_lock = threading.Lock()
worker_busy = False
worker_busy_lock = threading.Lock()


class RateLimiter:
    def __init__(self, max_requests: int, interval_s: float):
        self.max_requests = max_requests
        self.interval = interval_s
        self.request_timestamps = []
        self.lock = threading.Lock()

    def _prune(self):
        cutoff = time.time() - self.interval
        while self.request_timestamps and self.request_timestamps[0] < cutoff:
            self.request_timestamps.pop(0)

    def wait_for_slot(self):
        while True:
            with self.lock:
                self._prune()
                if len(self.request_timestamps) < self.max_requests:
                    self.request_timestamps.append(time.time())
                    return
            time.sleep(0.05)
            
    def can_request(self) -> bool:
        with self.lock:
            self._prune()
            return len(self.request_timestamps) < self.max_requests


# instantiate with your OpenRouter limits
# RATE_LIMITER = RateLimiter(max_requests=100, interval_s=60.0)
RATE_LIMITER = RateLimiter(max_requests=1, interval_s=10.0)

def query_openrouter_limited(prompt: str, model_id: str) -> str:
    RATE_LIMITER.wait_for_slot()
    max_tokens = 500
    if model_id == "openai/o4-mini":
        max_tokens = 1000
    return query_openrouter(prompt, model_id, max_tokens)

def process_prompt(prompt):
    """Process a single prompt with all models in parallel"""
    try : 
        with ThreadPoolExecutor(max_workers=5) as inner_executor:
        
            gemini_future = inner_executor.submit(query_openrouter_limited, prompt, MODEL_IDS["gemini"])
            openai_future = inner_executor.submit(query_openrouter_limited, prompt, MODEL_IDS["openai"])
            perplexity_future = inner_executor.submit(query_openrouter_limited, prompt, MODEL_IDS["perplexity"])
            deepseek_future = inner_executor.submit(query_openrouter_limited, prompt, MODEL_IDS["deepseek"])
            claude_future = inner_executor.submit(query_openrouter_limited, prompt, MODEL_IDS["claude"])
            return gemini_future.result(), openai_future.result(), perplexity_future.result(), deepseek_future.result(), claude_future.result()

    except Exception as e:
        logger.error(f"process_prompt failed: {str(e)}")
        return "", "", "", "", ""  # Return empty strings on failure

def worker_thread():
    """Background worker that processes queued requests"""
    global worker_busy
    while True:
        # Get next job from queue
        job_id, brand, prompts = request_queue.get()
        logger.info(f"Starting job {job_id} with {len(prompts)} prompts")
        with worker_busy_lock:
            worker_busy = True 
        
        try:
            # Update job status to processing
            with job_lock:
                if job_id in job_tracker:
                    job_tracker[job_id].update({
                        "status": "processing",
                        "started_at": time.time(),
                        "progress": 0,
                        "total_prompts": len(prompts)
                    })
                else:
                    job_tracker[job_id] = {
                        "status": "processing",
                        "started_at": time.time(),
                        "progress": 0,
                        "total_prompts": len(prompts)
                    }
            
            # Process all prompts in parallel
            openAi_responses = []
            gemini_responses = []
            perplexity_responses = []
            deepseek_responses = []
            claude_responses = []
            total_prompts = len(prompts)
            completed = 0
            
            with ThreadPoolExecutor(max_workers=5) as outer_executor:
                # Create futures for all prompts
                futures = {
                    outer_executor.submit(process_prompt, prompt): idx
                    for idx, prompt in enumerate(prompts)
                }
                
                # Process completed futures as they come in
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        g_response, o_response, p_response, d_future, c_future = future.result()
                        openAi_responses.append({"prompt": prompts[idx], "response": o_response})
                        gemini_responses.append({"prompt": prompts[idx], "response": g_response})
                        perplexity_responses.append({"prompt": prompts[idx], "response": p_response})
                        deepseek_responses.append({"prompt": prompts[idx], "response": d_future})
                        claude_responses.append({"prompt": prompts[idx], "response": c_future})
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        openAi_responses.append({"prompt": prompts[idx], "response": error_msg})
                        gemini_responses.append({"prompt": prompts[idx], "response": error_msg})
                        perplexity_responses.append({"prompt": prompts[idx], "response": error_msg})
                        deepseek_responses.append({"prompt": prompts[idx], "response": error_msg})
                        claude_responses.append({"prompt": prompts[idx], "response": error_msg})
                        logger.error(f"Error processing prompt: {str(e)}")
                    
                    # Update progress
                    completed += 1
                    with job_lock:
                        if job_id in job_tracker:
                            job_tracker[job_id]["progress"] = (completed / total_prompts) * 100
            
            # Calculate results
            openAi_mention_rate = calculate_mention_rate(openAi_responses, brand)
            gemini_mention_rate = calculate_mention_rate(gemini_responses, brand)
            perplexity_mention_rate = calculate_mention_rate(perplexity_responses, brand)
            deepseek_mention_rate = calculate_mention_rate(deepseek_responses, brand)
            claude_mention_rate = calculate_mention_rate(claude_responses, brand)
            
            # Segregate the prompts
            openai_mentioned, openai_not_mentioned = segregate_prompts_by_mention(openAi_responses, brand)
            gemini_mentioned, gemini_not_mentioned = segregate_prompts_by_mention(gemini_responses, brand)
            perplexity_mentioned, perplexity_not_mentioned = segregate_prompts_by_mention(perplexity_responses, brand)
            deepseek_mentioned, deepseek_not_mentioned = segregate_prompts_by_mention(deepseek_responses, brand)
            claude_mentioned, claude_not_mentioned = segregate_prompts_by_mention(claude_responses, brand)

            
            # Mark job as complete
            with job_lock:
                if job_id in job_tracker:
                    job_tracker[job_id].update({
                        "status": "completed",
                        "completed_at": time.time(),
                        "result": {
                            "brand": brand,
                            "total_prompts": total_prompts,
                            "openAi_mention_rate": openAi_mention_rate,
                            "gemini_mention_rate": gemini_mention_rate,
                            "perplexity_mention_rate": perplexity_mention_rate,
                            "deepseek_mention_rate": deepseek_mention_rate,
                            "claude_mention_rate": claude_mention_rate,
                            "segregated_prompts": {
                                "openai": {
                                    "mentioned": openai_mentioned[:3],
                                    "not_mentioned": openai_not_mentioned[:3]
                                },
                                "gemini": {
                                    "mentioned": gemini_mentioned[:3],
                                    "not_mentioned": gemini_not_mentioned[:3]
                                },
                                "perplexity": {
                                    "mentioned": perplexity_mentioned[:3],
                                    "not_mentioned": perplexity_not_mentioned[:3]
                                },
                                "deepseek": {
                                    "mentioned": deepseek_mentioned[:3],
                                    "not_mentioned": deepseek_not_mentioned[:3]
                                },
                                "claude": {
                                    "mentioned": claude_mentioned[:3],
                                    "not_mentioned": claude_not_mentioned[:3]
                                }
                            }

                        }
                    })
                else:
                    job_tracker[job_id] = {
                        "status": "completed",
                        "completed_at": time.time(),
                        "result": {
                            "brand": brand,
                            "total_prompts": total_prompts,
                            "openAi_mention_rate": openAi_mention_rate,
                            "gemini_mention_rate": gemini_mention_rate,
                            "perplexity_mention_rate": perplexity_mention_rate,
                            "deepseek_mention_rate": deepseek_mention_rate,
                            "claude_mention_rate": claude_mention_rate,
                            "segregated_prompts": {
                                "openai": {
                                    "mentioned": openai_mentioned[:3],
                                    "not_mentioned": openai_not_mentioned[:3]
                                },
                                "gemini": {
                                    "mentioned": gemini_mentioned[:3],
                                    "not_mentioned": gemini_not_mentioned[:3]
                                },
                                "perplexity": {
                                    "mentioned": perplexity_mentioned[:3],
                                    "not_mentioned": perplexity_not_mentioned[:3]
                                },
                                "deepseek": {
                                    "mentioned": deepseek_mentioned[:3],
                                    "not_mentioned": deepseek_not_mentioned[:3]
                                },
                                "claude": {
                                    "mentioned": claude_mentioned[:3],
                                    "not_mentioned": claude_not_mentioned[:3]
                                }
                            }

                        }
                    }
                
        except Exception as e:
            logger.error(f"Job {job_id} failed: {str(e)}")
            with job_lock:
                if job_id in job_tracker:
                    job_tracker[job_id].update({
                        "status": "failed",
                        "completed_at": time.time(),
                        "error": str(e)
                    })
                else:
                    job_tracker[job_id] = {
                        "status": "failed",
                        "completed_at": time.time(),
                        "error": str(e)
                    }
        
        finally:
            with worker_busy_lock:
                worker_busy = False
            request_queue.task_done()
            logger.info(f"Finished job {job_id}")

# Start worker thread on first request
worker = None
def start_worker():
    global worker
    if worker is None or not worker.is_alive():
        worker = threading.Thread(target=worker_thread, daemon=True)
        worker.start()
        logger.info("Started worker thread")



@api_view(['POST'])
def run_query(request):
    brand = request.data.get('brand')
    competitor = request.data.get('competitor')
    prompt_template = request.data.get('prompt')

    if not brand or not prompt_template:
        return Response({"error": "brand and prompt are required"}, status=status.HTTP_400_BAD_REQUEST)

    prompt = prompt_template.format(brand=brand, competitor=competitor or "")

    try:
        o_response = query_openai(prompt, OPENAI_MODEL)

        results = [
            {
                "brand": brand,
                "competitor": competitor,
                "prompt": prompt,
                "response": o_response,
                "sentiment": get_sentiment(o_response),
                "themes": extract_themes(o_response),
                "ai": "openai"
            }
        ]

        return Response({"results": results})

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

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
        f"(Strict instructions : In your response , I only need the prompts separated by semicolons, in a txt format, not markdown, and no extra text with it. Keep the prompts short and simple)"

    )
    
    prompt =prompt_template.format(brand=brand, website=website, custom_comments=custom_comments or "")

    try:
        o_response = query_openai(prompt, OPENAI_MODEL)
        # g_response_array = [p.strip() for p in g_response.split(';') if p.strip()]
        o_response_array = [p.strip() for p in o_response.split(';') if p.strip()]
        results = {
            "openai": o_response_array
        }

        return Response({"results": results})

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(['POST'])
def job_status(request):
    job_id = request.data.get("job_id")
    print(f"Job ID: {job_id}")
    with job_lock:
        job = job_tracker.get(job_id)
    
    if not job:
        return Response({"error": "Job not found"}, status=404)
    
    response_data = {
        "job_id": job_id,
        "status": job["status"],
        "created_at": job.get("created_at"),
    }
    
    if job["status"] == "queued":
        response_data["position_in_queue"] = job.get("position_in_queue", 0)
        # Recalculate estimated wait time
        if "position_in_queue" in job:
            # Estimate wait: next‐slot wait + 5s per job
            now = time.time()
            elapsed = now - (RATE_LIMITER.request_timestamps[0] if RATE_LIMITER.request_timestamps else now)
            position = job.get("position_in_queue", 0)
            wait_seconds = max(0, RATE_LIMITER.interval - elapsed) + (position * 5)

            response_data["estimated_wait_seconds"] = wait_seconds
    
    elif job["status"] == "processing":
        response_data["started_at"] = job.get("started_at")
        response_data["progress"] = f"{job.get('progress', 0):.1f}%"
        response_data["processed"] = int(job.get('progress', 0) * job.get("total_prompts", 0) / 100)
        response_data["total_prompts"] = job.get("total_prompts", 0)
    
    elif job["status"] == "completed":
        response_data["completed_at"] = job.get("completed_at")
        response_data["data"] = job.get("result", {})
    
    elif job["status"] == "failed":
        response_data["completed_at"] = job.get("completed_at")
        response_data["error"] = job.get("error", "Unknown error")
    
    return Response(response_data)

@api_view(['POST'])
def brand_mention_score(request):
    start_worker()  # Ensure worker is running
    
    brand = request.data.get("brand")
    prompts = request.data.get("prompts", [])
    if not brand or not prompts:
        return Response({"error": "brand and prompts are required"}, status=400)
    
    
    # Check if we can process immediately
    if RATE_LIMITER.can_request() and not worker_busy and request_queue.empty():
        try:
            
            # Process immediately with parallel execution
            openAi_responses = []
            gemini_responses = []
            perplexity_responses = []
            deepseek_responses = []
            claude_responses = []
            with ThreadPoolExecutor(max_workers=8) as outer_executor:
                futures = {
                    outer_executor.submit(process_prompt, prompt): prompt
                    for prompt in prompts
                }
                
                for future in as_completed(futures):
                    try:
                        g_response, o_response, p_response, d_response, c_response = future.result()
                        openAi_responses.append({"prompt": futures[future], "response": o_response})
                        gemini_responses.append({"prompt": futures[future], "response": g_response})
                        perplexity_responses.append({"prompt": futures[future], "response": p_response})
                        deepseek_responses.append({"prompt": futures[future], "response": d_response})
                        claude_responses.append({"prompt": futures[future], "response": c_response})
                    except Exception as e:
                        error_resp = {"prompt": futures[future], "response": f"Error: {str(e)}"}
                        openAi_responses.append(error_resp)
                        gemini_responses.append(error_resp)
                        perplexity_responses.append(error_resp)
                        deepseek_responses.append(error_resp)
                        claude_responses.append(error_resp)
            
            # Calculate results
            openAi_mention_rate = calculate_mention_rate(openAi_responses, brand)
            gemini_mention_rate = calculate_mention_rate(gemini_responses, brand)
            perplexity_mention_rate = calculate_mention_rate(perplexity_responses, brand)
            deepseek_mention_rate = calculate_mention_rate(deepseek_responses, brand)
            claude_mention_rate = calculate_mention_rate(claude_responses, brand)
            
           
            # Segregate the prompts on the basis of mention rate
            openai_mentioned, openai_not_mentioned = segregate_prompts_by_mention(openAi_responses, brand)
            gemini_mentioned, gemini_not_mentioned = segregate_prompts_by_mention(gemini_responses, brand)
            perplexity_mentioned, perplexity_not_mentioned = segregate_prompts_by_mention(perplexity_responses, brand)
            deepseek_mentioned, deepseek_not_mentioned = segregate_prompts_by_mention(deepseek_responses, brand)
            claude_mentioned, claude_not_mentioned = segregate_prompts_by_mention(claude_responses, brand)


            # Segregate the prompts on the basis of mention rate
            openai_mentioned, openai_not_mentioned = segregate_prompts_by_mention(openAi_responses, brand)
            gemini_mentioned, gemini_not_mentioned = segregate_prompts_by_mention(gemini_responses, brand)
            perplexity_mentioned, perplexity_not_mentioned = segregate_prompts_by_mention(perplexity_responses, brand)
            deepseek_mentioned, deepseek_not_mentioned = segregate_prompts_by_mention(deepseek_responses, brand)
            claude_mentioned, claude_not_mentioned = segregate_prompts_by_mention(claude_responses, brand)

            return Response({
                "status": "completed",
                "brand": brand,
                "total_prompts": len(prompts),
                "openAi_mention_rate": openAi_mention_rate,
                "gemini_mention_rate": gemini_mention_rate,
                "perplexity_mention_rate": perplexity_mention_rate,
                "deepseek_mention_rate": deepseek_mention_rate,
                "claude_mention_rate": claude_mention_rate,
                "segregated_prompts": {
                    "openai": {
                        "mentioned": openai_mentioned[:3],
                        "not_mentioned": openai_not_mentioned[:3]
                    },
                    "gemini": {
                        "mentioned": gemini_mentioned[:3],
                        "not_mentioned": gemini_not_mentioned[:3]
                    },
                    "perplexity": {
                        "mentioned": perplexity_mentioned[:3],
                        "not_mentioned": perplexity_not_mentioned[:3]
                    },
                    "deepseek": {
                        "mentioned": deepseek_mentioned[:3],
                        "not_mentioned": deepseek_not_mentioned[:3]
                    },
                    "claude": {
                        "mentioned": claude_mentioned[:3],
                        "not_mentioned": claude_not_mentioned[:3]
                    }
                }
            })
        
        except Exception as e:
            return Response({"status": "failed", "error": str(e)}, status=500)
    
    else:
        # Create job and add to queue
        job_id = str(uuid.uuid4())
        
        with worker_busy_lock:
            is_busy = worker_busy
            queue_size = request_queue.qsize()
            
        with job_lock:
            position = queue_size + (1 if is_busy else 0)
            job_tracker[job_id] = {
                "status": "queued",
                "created_at": time.time(),
                "brand": brand,
                "prompt_count": len(prompts),
                "position_in_queue": position
            }
        
        request_queue.put((job_id, brand, prompts))
        
        # Calculate wait time estimate (5 seconds per queued job + rate limit wait)
        now = time.time()
        oldest = RATE_LIMITER.request_timestamps[0] if RATE_LIMITER.request_timestamps else now
        elapsed = now - oldest
        wait_seconds = max(0, RATE_LIMITER.interval - elapsed) + (position * 5)
        
         # Create response data
        response_data = {
            "status": "queued",
            "message": "Server is busy, request queued",
            "job_id": job_id,
            "position_in_queue": position,
            "estimated_wait_seconds": wait_seconds
        }
        
        def generate():
            yield json.dumps(response_data)
        
        return StreamingHttpResponse(
            generate(),
            status=202,
            content_type='application/json'
        )
        
        
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

