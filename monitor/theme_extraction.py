import spacy

nlp = spacy.load("en_core_web_sm")

def extract_themes(text):
    doc = nlp(text)
    return list(set(chunk.text.strip().lower() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2))
