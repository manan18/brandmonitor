def calculate_mention_rate(responses, brand_name):
    total = len(responses)
    mentions = sum(
        1 for r in responses if brand_name.lower() in r.get("response", "").lower()
    )
    rate = (mentions / total) * 100 if total > 0 else 0
    return round(rate, 2)
