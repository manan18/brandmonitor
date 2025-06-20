from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from .llm_query import query_openai, query_gemini
from .sentiment import get_sentiment
from .theme_extraction import extract_themes
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "")

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

    print(f"Generated Prompt: {prompt}")

    try:
        g_response = query_gemini(prompt, GEMINI_API_KEY)
        o_response = query_openai(prompt, OPENAI_MODEL, OPENAI_API_KEY)
        g_response_array = [p.strip() for p in g_response.split(';') if p.strip()]
        o_response_array = [p.strip() for p in o_response.split(';') if p.strip()]
        results = {
            "gemini": g_response_array,
            "openai": o_response_array
        }

        return Response({"results": results})

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)