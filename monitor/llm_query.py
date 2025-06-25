from openai import OpenAI
import google.generativeai as genai
import yaml
import os
import time  # For rate limiting

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def query_openrouter(prompt, model_id):
    """Query any model through OpenRouter API"""
    client = OpenAI(
        api_key=os.getenv('OPENROUTER_API_KEY'),
        base_url="https://openrouter.ai/api/v1"
    )
    
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

def run_queries():
    config = load_config()
    responses = []
    
    MODEL_IDS = {
        "gemini": "google/gemini-1.5-flash",
        "openai": "openai/gpt-4o"
    }

    for brand in config["brands"]:
        for prompt_template in config["prompts"]:
            if "{competitor}" in prompt_template:
                for comp in config["competitors"]:
                    prompt = prompt_template.format(brand=brand, competitor=comp)
                    
                    result = query_openrouter(prompt, MODEL_IDS["gemini"])
                    responses.append({"brand": brand, "prompt": prompt, "response": result, "ai": "gemini"})
                    
                    result1 = query_openrouter(prompt, MODEL_IDS["openai"])
                    responses.append({"brand": brand, "prompt": prompt, "response": result1, "ai": "openai"})
                    
                    # time.sleep(1) 
            else:
                prompt = prompt_template.format(brand=brand)
                
                result = query_openrouter(prompt, MODEL_IDS["gemini"])
                responses.append({"brand": brand, "prompt": prompt, "response": result, "ai": "gemini"})
                
                result1 = query_openrouter(prompt, MODEL_IDS["openai"])
                responses.append({"brand": brand, "prompt": prompt, "response": result1, "ai": "openai"})
                
                # time.sleep(1)  
    return responses