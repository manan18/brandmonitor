from openai import OpenAI
import google.generativeai as genai
import yaml

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def query_gemini(prompt, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)
    return response.text

def query_openai(prompt, model, api_key):
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content


def run_queries():
    config = load_config()
    responses = []

    for brand in config["brands"]:
        for prompt_template in config["prompts"]:
            if "{competitor}" in prompt_template:
                for comp in config["competitors"]:
                    prompt = prompt_template.format(brand=brand, competitor=comp)
                    result = query_gemini(prompt, config["llms"][0]["api_key"])
                    result1 = query_openai(prompt, config["llms"][1]["model"], config["llms"][1]["api_key"])
                    responses.append({"brand": brand, "prompt": prompt, "response": result, "ai": "gemini"})
                    responses.append({"brand": brand, "prompt": prompt, "response": result1, "ai": "openai"})
            else:
                prompt = prompt_template.format(brand=brand)
                result = query_gemini(prompt, config["llms"][0]["api_key"])
                result1 = query_openai(prompt, config["llms"][1]["model"], config["llms"][1]["api_key"])
                responses.append({"brand": brand, "prompt": prompt, "response": result, "ai": "gemini"})
                responses.append({"brand": brand, "prompt": prompt, "response": result1, "ai": "openai"})
    return responses
