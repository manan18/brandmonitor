from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    return {
        "positive": scores["pos"],
        "neutral": scores["neu"],
        "negative": scores["neg"],
        "compound": scores["compound"]
    }
