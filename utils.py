"""
utils.py - Utility functions for the Fake News Detection System
"""

import re
import string


def clean_text(text: str) -> str:
    """
    Clean and preprocess raw news text before feeding it to the model.
    Steps: lowercase, remove URLs, remove HTML, remove punctuation, strip whitespace.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)   # remove URLs
    text = re.sub(r"<.*?>", "", text)                       # strip HTML tags
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()               # collapse whitespace
    return text


def format_confidence(probability: float) -> str:
    """Convert probability (0-1) to percentage string."""
    return f"{probability * 100:.1f}%"


def get_prediction_label(prediction: int) -> str:
    """Map integer prediction to human-readable label."""
    label_map = {0: "REAL NEWS", 1: "FAKE NEWS"}
    return label_map.get(prediction, "UNKNOWN")


def get_confidence_level(confidence: float) -> str:
    """Categorise confidence into descriptive tiers."""
    if confidence >= 0.90:
        return "Very High"
    elif confidence >= 0.75:
        return "High"
    elif confidence >= 0.60:
        return "Moderate"
    else:
        return "Low"


SAMPLE_REAL_NEWS = """The Federal Reserve raised interest rates by 25 basis points on Wednesday, its tenth increase since March 2022, as policymakers continue to battle persistent inflation. Fed Chair Jerome Powell stated that the central bank remains committed to returning inflation to its 2% target. Markets responded positively, with the S&P 500 gaining 0.9% following the announcement."""

SAMPLE_FAKE_NEWS = """SHOCKING: Scientists discover that drinking bleach cures all known diseases! The mainstream media is hiding this from you because Big Pharma doesn't want you to know the truth. A secret report leaked from the CDC confirms that the government has been suppressing this miraculous cure for decades. Share this before they delete it! The globalists are terrified!"""
def get_sample_news():
    return [
        {
            "label": "📰 Real News",
            "text": SAMPLE_REAL_NEWS
        },
        {
            "label": "🚨 Fake News",
            "text": SAMPLE_FAKE_NEWS
        }
    ]
