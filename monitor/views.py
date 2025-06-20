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
