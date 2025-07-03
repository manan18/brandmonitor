from openai import OpenAI
import google.generativeai as genai
import yaml
import os
def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def query_openai(prompt, model):
    
    client = OpenAI(
        api_key=os.getenv('OPENROUTER_API_KEY'),
        base_url="https://openrouter.ai/api/v1",
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content


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

def print_model_cost(model_id, input_tokens, output_tokens):
    """Print the estimated cost for the given model and token usage"""
    pricing = {
        "google/gemini-2.5-flash": {"input": 0.30 / 1_000_000, "output": 2.50 / 1_000_000},
        "openai/o4-mini": {"input": 1.10 / 1_000_000, "output": 4.40 / 1_000_000},
        "perplexity/sonar-pro": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
        "deepseek/deepseek-r1-0528": {"input": 0.50 / 1_000_000, "output": 2.15 / 1_000_000},
        "anthropic/claude-3.5-haiku-20241022:beta": {"input": 0.80 / 1_000_000, "output": 4.00 / 1_000_000},
    }

    model_pricing = pricing.get(model_id)
    if not model_pricing:
        print(f"No pricing available for model: {model_id}")
        return

    input_cost = input_tokens * model_pricing["input"]
    output_cost = output_tokens * model_pricing["output"]
    total_cost = input_cost + output_cost

    print(f"Cost: ${total_cost:.6f} (Input: ${input_cost:.6f}, Output: ${output_cost:.6f})")


def query_openrouter(prompt, model_id):
    """Query any model through OpenRouter API"""
    from openai import OpenAI
    import os

    client = OpenAI(
        api_key=os.getenv('OPENROUTER_API_KEY'),
        base_url="https://openrouter.ai/api/v1"
    )
    
    max_tokens = 500
    if model_id == "openai/o4-mini" :
        max_tokens = 1000   

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=max_tokens
        )

        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        # print(f"Model: {model_id}")
        # print(f"Input tokens: {input_tokens}, Output tokens: {output_tokens}")

        # Call the cost display function
        # print_model_cost(model_id, input_tokens, output_tokens)

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error in query: {str(e)}")
        return None
