from openai import OpenAI
import google.generativeai as genai
import yaml
import os
import tiktoken

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def query_gemini(prompt, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)
    return response.text

def query_openai(prompt, model):
    
    client = OpenAI(
        api_key=os.getenv('OPENROUTER_API_KEY'),
        base_url="https://openrouter.ai/api/v1"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    return response.choices[0].message.content

def count_openai_tokens(prompt: str, model: str = "gpt-4o-mini"):
    """Uses tiktoken to estimate token count for OpenAI models."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(prompt))

def log_model_cost(model_id, input_tokens, output_tokens, input_cost, output_cost, total_cost, prompt):
    """Log token usage, cost, and prompt to a CSV file."""
    import csv
    from datetime import datetime

    log_file = "model_costs.csv"
    file_exists = os.path.exists(log_file)

    with open(log_file, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "Timestamp", "Model", "Input Tokens", "Output Tokens",
                "Input Cost ($)", "Output Cost ($)", "Total Cost ($)", "Prompt"
            ])
        writer.writerow([
            datetime.now().isoformat(),
            model_id,
            input_tokens,
            output_tokens,
            f"{input_cost:.6f}",
            f"{output_cost:.6f}",
            f"{total_cost:.6f}",
            prompt.replace("\n", " ").strip()
        ])



def print_model_cost(model_id, input_tokens, output_tokens, prompt):
    """Calculate and log the estimated cost for the given model and token usage."""
    pricing = {
        "google/gemini-2.5-flash": {"input": 0.30 / 1_000_000, "output": 2.50 / 1_000_000},
        "openai/o4-mini": {"input": 1.10 / 1_000_000, "output": 4.40 / 1_000_000},
        "perplexity/sonar-pro": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
        "deepseek/deepseek-r1-0528": {"input": 0.50 / 1_000_000, "output": 2.15 / 1_000_000},
        "anthropic/claude-3.5-haiku-20241022:beta": {"input": 0.80 / 1_000_000, "output": 4.00 / 1_000_000},
    }

    model_pricing = pricing.get(model_id)
    if not model_pricing:
        return

    input_cost = input_tokens * model_pricing["input"]
    output_cost = output_tokens * model_pricing["output"]
    total_cost = input_cost + output_cost

    # Log the calculated costs
    log_model_cost(model_id, input_tokens, output_tokens, input_cost, output_cost, total_cost, prompt)

    # print(f"Cost: ${total_cost:.6f} (Input: ${input_cost:.6f}, Output: ${output_cost:.6f})")


def query_openrouter(prompt, model_id, max_tokens=500):
    """Query any model through OpenRouter API"""
    from openai import OpenAI
    import os

    client = OpenAI(
        api_key=os.getenv('OPENROUTER_API_KEY'),
        base_url="https://openrouter.ai/api/v1"
    )
   
   
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=max_tokens
        )
        
        if model_id == "openai/o4-mini":
            input_tokens = count_openai_tokens(prompt, model="gpt-4o-mini") 
            output_tokens = count_openai_tokens(response.choices[0].message.content, model="gpt-4o-mini")
        else:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

        # print(f"🤖Model: {model_id}")
        # print(f"Input tokens: {input_tokens}, Output tokens: {output_tokens}")
        # print(f"Max token: {max_tokens}")
        # print(response.choices[0].message.content)

        print_model_cost(model_id, input_tokens, output_tokens, prompt)

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error in query: {str(e)}")
        return None
