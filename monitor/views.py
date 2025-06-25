from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from .llm_query import query_openai, query_gemini, query_openrouter
from .sentiment import get_sentiment
from .theme_extraction import extract_themes
import os
from dotenv import load_dotenv
from .utils import calculate_mention_rate
from concurrent.futures import ThreadPoolExecutor, as_completed
from rest_framework.decorators import api_view
from rest_framework.response import Response

import threading
import time
import uuid
import queue
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from rest_framework.decorators import api_view
from rest_framework.response import Response

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

# Rate limiter class for Gemini
class GeminiRateLimiter:
    def __init__(self, rpm_limit=10, tpm_limit=950000):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.request_timestamps = []
        self.token_count = 0
        self.lock = threading.Lock()
        self.last_reset = time.time()

    def can_process(self, estimated_tokens=100):
        now = time.time()
        with self.lock:
            # Reset counters every minute
            if now - self.last_reset >= 60:
                self.request_timestamps = []
                self.token_count = 0
                self.last_reset = now
                
            # Check RPM limit
            if len(self.request_timestamps) >= self.rpm_limit:
                return False
                
            # Check TPM limit
            if self.token_count + estimated_tokens >= self.tpm_limit:
                return False
                
            return True

    def add_request(self, estimated_tokens=100):
        now = time.time()
        with self.lock:
            if now - self.last_reset >= 60:
                self.request_timestamps = []
                self.token_count = 0
                self.last_reset = now
                
            self.request_timestamps.append(now)
            self.token_count += estimated_tokens

    def next_available_time(self):
        now = time.time()
        with self.lock:
            reset_in = 60 - (now - self.last_reset)
            
            if self.request_timestamps:
                oldest_request = self.request_timestamps[0]
                rpm_wait = max(0, 60 - (now - oldest_request))
            else:
                rpm_wait = 0
                
            if self.token_count >= self.tpm_limit:
                tpm_wait = reset_in
            else:
                tpm_wait = 0
                
            return max(reset_in, rpm_wait, tpm_wait)

# Initialize rate limiter with 90% of Gemini Flash limits
GEMINI_LIMITER = GeminiRateLimiter(rpm_limit=10, tpm_limit=950000)

def process_prompt(prompt):
    """Process a single prompt with both models in parallel"""
    with ThreadPoolExecutor(max_workers=5) as inner_executor:
        # gemini_future = inner_executor.submit(query_gemini, prompt, GEMINI_API_KEY)
        # openai_future = inner_executor.submit(query_openai, prompt, OPENAI_MODEL, OPENAI_API_KEY)
        gemini_future = inner_executor.submit(query_openrouter, prompt, MODEL_IDS["gemini"])
        openai_future = inner_executor.submit(query_openrouter, prompt, MODEL_IDS["openai"])
        perplexity_future = inner_executor.submit(query_openrouter, prompt, MODEL_IDS["perplexity"])
        deepseek_future = inner_executor.submit(query_openrouter, prompt, MODEL_IDS["deepseek"])
        claude_future = inner_executor.submit(query_openrouter, prompt, MODEL_IDS["claude"])
        return gemini_future.result(), openai_future.result(), perplexity_future.result(), deepseek_future.result(), claude_future.result()


def worker_thread():
    """Background worker that processes queued requests"""
    global worker_busy
    while True:
        # Get next job from queue
        job_id, brand, prompts = request_queue.get()
        logger.info(f"Starting job {job_id} with {len(prompts)} prompts")
        
        try:
            # Update job status to processing
            with job_lock:
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
            
            with ThreadPoolExecutor(max_workers=15) as outer_executor:
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
            # Mark job as complete
            with job_lock:
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
                    }
                }
                
        except Exception as e:
            logger.error(f"Job {job_id} failed: {str(e)}")
            with job_lock:
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
        g_response = query_gemini(prompt, GEMINI_API_KEY)
        o_response = query_openai(prompt, OPENAI_MODEL, OPENAI_API_KEY)

        results = [
            {
                "brand": brand,
                "competitor": competitor,
                "prompt": prompt,
                "response": g_response,
                "sentiment": get_sentiment(g_response),
                "themes": extract_themes(g_response),
                "ai": "gemini"
            },
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
    category = request.data.get('category')
    core_features= request.data.get('core_features')
    primary_use_case = request.data.get('primary_use_case')
    target_audience = request.data.get('target_audience')
    differentiators = request.data.get('differentiators')
    integrations = request.data.get('integrations')
    deployment = request.data.get('deployment')
    geographic_locations = request.data.get('geographic_locations')
    keywords = request.data.get('keywords')

    if not brand or not core_features or not primary_use_case or not category or not target_audience or not category:
        return Response({"error": "brand and prompt are required"}, status=status.HTTP_400_BAD_REQUEST)
    
    prompt_template = (
        f"I have a brand/product/application known as {brand}, which falls under the category of {category}. "
        f"It's core features are {core_features}, and the primary use case is {primary_use_case}. "
        f"My target audience is {target_audience}. "
        f"{'My key differentiating points are ' + differentiators + '. ' if differentiators else ''}"
        f"{'Some other platforms/technologies my tool connects with are: ' + integrations + '. ' if integrations else ''}"
        f"{'My deployment and pricing models are ' + deployment + ' respectively. ' if deployment else ''}"
        f"{'My geographic and/or language focuses on ' + geographic_locations + '. ' if geographic_locations else ''}"
        f"{'Some common keywords which people use to describe my tool/product are ' + keywords + '. ' if keywords else ''}"
        "Use the information provided above to generate a list of 100 prompts which would potentially mention my platform in their response if a user searches over the web for platforms similar to mine or for platforms in the same category. Give the prompts imagining that you're a random user, who does not know about my platform, but is looking for a platform which has the same features and use cases as mine. "
        "(In your response , I only need the prompts separated by semicolons, in a txt format, not markdown, and no extra text with it.)"
    )

    prompt =prompt_template.format(brand=brand, category=category, core_features=core_features, primary_use_case=primary_use_case, target_audience=target_audience, differentiators=differentiators or "", integrations=integrations or "", deployment=deployment or "", geographic_locations=geographic_locations or "", keywords=keywords or "")

    # print(f"Generated Prompt: {prompt}")

    try:
        # g_response = query_gemini(prompt, GEMINI_API_KEY)
        o_response = query_openai(prompt, OPENAI_MODEL, OPENAI_API_KEY)
        # g_response_array = [p.strip() for p in g_response.split(';') if p.strip()]
        o_response_array = [p.strip() for p in o_response.split(';') if p.strip()]
        results = {
            # "gemini": g_response_array,
            "openai": o_response_array
        }

        return Response({"results": results})

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(['POST'])
def job_status(request):
    job_id = request.data.get("job_id")
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
            wait_seconds = GEMINI_LIMITER.next_available_time() + (job["position_in_queue"] * 5)
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
    
    # Estimate tokens for the entire request
    estimated_tokens = sum(len(prompt) // 4 + 150 for prompt in prompts)
    
    # Check if we can process immediately
    if GEMINI_LIMITER.can_process(estimated_tokens) and not worker_busy and request_queue.empty():
        try:
            # Record the request
            GEMINI_LIMITER.add_request(estimated_tokens)
            
            # Process immediately with parallel execution
            openAi_responses = []
            gemini_responses = []
            perplexity_responses = []
            deepseek_responses = []
            claude_responses = []
            with ThreadPoolExecutor(max_workers=15) as outer_executor:
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

            return Response({
                "status": "completed",
                "brand": brand,
                "total_prompts": len(prompts),
                "openAi_mention_rate": openAi_mention_rate,
                "gemini_mention_rate": gemini_mention_rate,
                "perplexity_mention_rate": perplexity_mention_rate,
                "deepseek_mention_rate": deepseek_mention_rate,
                "claude_mention_rate": claude_mention_rate,
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
        wait_seconds = GEMINI_LIMITER.next_available_time() + (position * 5)
        
        return Response({
            "status": "queued",
            "message": "Server is busy, request queued",
            "job_id": job_id,
            "position_in_queue": position,
            "estimated_wait_seconds": wait_seconds
        }, status=202)  # 202 Accepted