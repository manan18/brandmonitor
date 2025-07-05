def calculate_mention_rate(responses, brand_name):
    total = len(responses)
    if total == 0:
        return 0.0
    
    mentions = 0
    for r in responses:
        response_text = r.get("response", "")  # Default to empty string
        
        # Handle all unexpected types
        if not isinstance(response_text, str):
            try:
                response_text = str(response_text)
            except:
                response_text = ""
        
        # Check for brand mention
        if brand_name.lower() in response_text.lower():
            mentions += 1
    
    rate = (mentions / total) * 100
    return round(rate, 2)