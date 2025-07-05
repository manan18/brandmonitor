import redis
import threading
import time
import uuid
import json
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import StreamingHttpResponse
import os
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Redis connection for shared state
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    password=os.getenv('REDIS_PASSWORD', None),
    decode_responses=True
)

class DistributedRateLimiter:
    def __init__(self, max_requests: int, interval_s: float, redis_client):
        self.max_requests = max_requests
        self.interval = interval_s
        self.redis = redis_client
        self.key_prefix = "rate_limiter"
        self.lock_key = f"{self.key_prefix}:lock"
        self.timestamps_key = f"{self.key_prefix}:timestamps"
        self.active_key = f"{self.key_prefix}:active"
        
    def _get_lock(self, timeout=5):
        """Acquire distributed lock"""
        lock_id = str(uuid.uuid4())
        end_time = time.time() + timeout
        
        while time.time() < end_time:
            if self.redis.set(self.lock_key, lock_id, nx=True, ex=timeout):
                return lock_id
            time.sleep(0.01)
        return None
    
    def _release_lock(self, lock_id):
        """Release distributed lock"""
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        self.redis.eval(lua_script, 1, self.lock_key, lock_id)
    
    def _prune_timestamps(self):
        """Remove old timestamps"""
        cutoff = time.time() - self.interval
        self.redis.zremrangebyscore(self.timestamps_key, 0, cutoff)
    
    def can_process_immediately(self):
        """Check if request can be processed immediately"""
        lock_id = self._get_lock()
        if not lock_id:
            return False
        
        try:
            self._prune_timestamps()
            
            # Count active requests and recent timestamps
            active_count = len(self.redis.smembers(self.active_key))
            timestamp_count = self.redis.zcard(self.timestamps_key)
            
            total_requests = active_count + timestamp_count
            
            logger.info(f"Rate limiter check - Active: {active_count}, History: {timestamp_count}, Total: {total_requests}")
            
            return total_requests < self.max_requests
            
        finally:
            self._release_lock(lock_id)
    
    def acquire_slot(self, request_id):
        """Acquire a processing slot"""
        lock_id = self._get_lock()
        if not lock_id:
            raise Exception("Could not acquire lock")
        
        try:
            self._prune_timestamps()
            
            # Check if slot is available
            active_count = len(self.redis.smembers(self.active_key))
            timestamp_count = self.redis.zcard(self.timestamps_key)
            
            if active_count + timestamp_count >= self.max_requests:
                # Wait for slot to become available
                return False
            
            # Acquire slot
            self.redis.sadd(self.active_key, request_id)
            logger.info(f"Acquired slot for {request_id}")
            return True
            
        finally:
            self._release_lock(lock_id)
    
    def release_slot(self, request_id):
        """Release a processing slot"""
        lock_id = self._get_lock()
        if not lock_id:
            return
        
        try:
            # Remove from active set and add timestamp
            self.redis.srem(self.active_key, request_id)
            self.redis.zadd(self.timestamps_key, {str(time.time()): time.time()})
            logger.info(f"Released slot for {request_id}")
            
        finally:
            self._release_lock(lock_id)

class DistributedJobTracker:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.key_prefix = "job_tracker"
        self.queue_key = f"{self.key_prefix}:queue"
    
    def create_job(self, job_id, brand, prompts):
        """Create a new job"""
        job_data = {
            "job_id": job_id,
            "brand": brand,
            "prompts": json.dumps(prompts),
            "status": "queued",
            "created_at": time.time(),
            "prompt_count": len(prompts)
        }
        
        # Store job data
        self.redis.hset(f"{self.key_prefix}:jobs", job_id, json.dumps(job_data))
        
        # Add to queue
        self.redis.lpush(self.queue_key, job_id)
        
        # Calculate position
        position = self.redis.llen(self.queue_key)
        job_data["position_in_queue"] = position
        
        self.redis.hset(f"{self.key_prefix}:jobs", job_id, json.dumps(job_data))
        
        return position
    
    def get_job(self, job_id):
        """Get job data"""
        job_data = self.redis.hget(f"{self.key_prefix}:jobs", job_id)
        if job_data:
            return json.loads(job_data)
        return None
    
    def update_job(self, job_id, updates):
        """Update job data"""
        job_data = self.get_job(job_id)
        if job_data:
            job_data.update(updates)
            self.redis.hset(f"{self.key_prefix}:jobs", job_id, json.dumps(job_data))
    
    def get_next_job(self):
        """Get next job from queue"""
        job_id = self.redis.rpop(self.queue_key)
        if job_id:
            job_data = self.get_job(job_id)
            if job_data:
                return job_id, job_data
        return None, None
    
    def get_queue_length(self):
        """Get current queue length"""
        return self.redis.llen(self.queue_key)

# Initialize distributed components
rate_limiter = DistributedRateLimiter(max_requests=1, interval_s=10.0, redis_client=redis_client)
job_tracker = DistributedJobTracker(redis_client)

def process_job_worker():
    """Background worker for processing jobs"""
    while True:
        try:
            job_id, job_data = job_tracker.get_next_job()
            
            if not job_id:
                time.sleep(1)  # Wait before checking again
                continue
            
            logger.info(f"Processing job {job_id}")
            
            # Update job status
            job_tracker.update_job(job_id, {
                "status": "processing",
                "started_at": time.time()
            })
            
            # Acquire rate limiter slot
            if not rate_limiter.acquire_slot(job_id):
                # Put job back in queue if can't acquire slot
                job_tracker.redis.rpush(job_tracker.queue_key, job_id)
                job_tracker.update_job(job_id, {"status": "queued"})
                time.sleep(1)
                continue
            
            try:
                # Process the job
                brand = job_data["brand"]
                prompts = json.loads(job_data["prompts"])
                
                # Your existing processing logic here
                result = process_brand_mention_job(brand, prompts, job_id)
                
                # Update job with results
                job_tracker.update_job(job_id, {
                    "status": "completed",
                    "completed_at": time.time(),
                    "result": result
                })
                
            except Exception as e:
                logger.error(f"Error processing job {job_id}: {str(e)}")
                job_tracker.update_job(job_id, {
                    "status": "failed",
                    "completed_at": time.time(),
                    "error": str(e)
                })
            
            finally:
                # Always release the slot
                rate_limiter.release_slot(job_id)
                
        except Exception as e:
            logger.error(f"Worker error: {str(e)}")
            time.sleep(1)

def process_brand_mention_job(brand, prompts, job_id):
    """Process brand mention job with progress tracking"""
    from .llm_query import query_openrouter
    from .utils import calculate_mention_rate
    
    MODEL_IDS = {
        "gemini": "google/gemini-2.5-flash",
        "openai": "openai/o4-mini",
        "perplexity": "perplexity/sonar-pro",
        "deepseek": "deepseek/deepseek-r1-0528",
        "claude": "anthropic/claude-3.5-haiku-20241022:beta"
    }
    
    responses = {model: [] for model in MODEL_IDS.keys()}
    
    for idx, prompt in enumerate(prompts):
        try:
            # Process each model sequentially
            for model_name, model_id in MODEL_IDS.items():
                response = query_openrouter(prompt, model_id)
                responses[model_name].append({
                    "prompt": prompt,
                    "response": response
                })
                
                # Small delay to respect rate limits
                time.sleep(0.5)
            
            # Update progress
            progress = ((idx + 1) / len(prompts)) * 100
            job_tracker.update_job(job_id, {"progress": progress})
            
        except Exception as e:
            logger.error(f"Error processing prompt {idx}: {str(e)}")
            # Add error responses
            for model_name in MODEL_IDS.keys():
                responses[model_name].append({
                    "prompt": prompt,
                    "response": f"Error: {str(e)}"
                })
    
    # Calculate mention rates
    mention_rates = {}
    segregated_prompts = {}
    
    for model_name, model_responses in responses.items():
        mention_rates[f"{model_name}_mention_rate"] = calculate_mention_rate(model_responses, brand)
        mentioned, not_mentioned = segregate_prompts_by_mention(model_responses, brand)
        segregated_prompts[model_name] = {
            "mentioned": mentioned[:3],
            "not_mentioned": not_mentioned[:3]
        }
    
    return {
        "brand": brand,
        "total_prompts": len(prompts),
        **mention_rates,
        "segregated_prompts": segregated_prompts
    }

def segregate_prompts_by_mention(responses, brand_name):
    """Segregate prompts by brand mention"""
    mentioned, not_mentioned = [], []
    
    for item in responses:
        prompt_text = item.get("prompt", "")
        resp = item.get("response", "") or ""
        if brand_name.lower() in resp.lower():
            mentioned.append(prompt_text)
        else:
            not_mentioned.append(prompt_text)
    
    return mentioned, not_mentioned

# Start worker thread
worker_thread = threading.Thread(target=process_job_worker, daemon=True)
worker_thread.start()
logger.info("Distributed worker started")

@api_view(['POST'])
def brand_mention_score(request):
    """Handle brand mention score requests with distributed rate limiting"""
    
    brand = request.data.get("brand")
    prompts = request.data.get("prompts", [])
    
    if not brand or not prompts:
        return Response({"error": "brand and prompts are required"}, status=400)
    
    # Check if can process immediately
    can_process = rate_limiter.can_process_immediately()
    queue_length = job_tracker.get_queue_length()
    
    logger.info(f"Request check - Can process: {can_process}, Queue length: {queue_length}")
    
    if can_process and queue_length == 0:
        # Try to acquire slot immediately
        request_id = str(uuid.uuid4())
        if rate_limiter.acquire_slot(request_id):
            try:
                # Process immediately
                logger.info(f"Processing request {request_id} immediately")
                result = process_brand_mention_job(brand, prompts, request_id)
                
                return Response({
                    "status": "completed",
                    **result
                })
            
            except Exception as e:
                logger.error(f"Immediate processing failed: {str(e)}")
                return Response({"status": "failed", "error": str(e)}, status=500)
            
            finally:
                rate_limiter.release_slot(request_id)
    
    # Queue the request
    job_id = str(uuid.uuid4())
    position = job_tracker.create_job(job_id, brand, prompts)
    
    # Calculate estimated wait time
    estimated_wait = position * len(prompts) * 50  # 50 seconds per prompt
    
    response_data = {
        "status": "queued",
        "message": "Request queued for processing",
        "job_id": job_id,
        "position_in_queue": position,
        "estimated_wait_seconds": estimated_wait,
        "estimated_wait_minutes": estimated_wait / 60
    }
    
    logger.info(f"Job {job_id} queued at position {position}")
    
    return StreamingHttpResponse(
        [json.dumps(response_data)],
        status=202,
        content_type='application/json'
    )

@api_view(['POST'])
def job_status(request):
    """Get job status"""
    job_id = request.data.get("job_id")
    
    if not job_id:
        return Response({"error": "job_id is required"}, status=400)
    
    job_data = job_tracker.get_job(job_id)
    
    if not job_data:
        return Response({"error": "Job not found"}, status=404)
    
    return Response(job_data)

@api_view(['GET'])
def health_check(request):
    """Health check endpoint"""
    try:
        # Test Redis connection
        redis_client.ping()
        redis_status = "connected"
    except:
        redis_status = "disconnected"
    
    return Response({
        "status": "ok",
        "redis": redis_status,
        "queue_length": job_tracker.get_queue_length()
    })