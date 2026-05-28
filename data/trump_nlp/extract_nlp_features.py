import pandas as pd
import re
import os
import math
import numpy as np
import torch

from tqdm import tqdm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline


# ============================================================
# Device
# ============================================================

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return 0
    return -1


# ============================================================
# Keyword Extraction
# ============================================================

def extract_keywords(text):
    text_lower = text.lower()

    return {

        'kw_china': int(bool(re.search(
            r'\bchina\b|\bchinese\b|\bccp\b|\bbeijing\b|'
            r'\bxi jinping\b|\bhuawei\b|\btiktok\b',
            text_lower
        ))),

        'kw_taiwan': int(bool(re.search(
            r'\btaiwan\b|\btsmc\b|\btaipei\b',
            text_lower
        ))),

        'kw_tariffs': int(bool(re.search(
            r'\btariffs?\b|\btrade war\b|'
            r'\bduties\b|\bsection 301\b',
            text_lower
        ))),

        'kw_sanctions': int(bool(re.search(
            r'\bsanctions?\b|\bexport controls?\b|'
            r'\bentity list\b|\bblacklist\b',
            text_lower
        ))),

        'kw_chips': int(bool(re.search(
            r'\bchips?\b|\bsemiconductors?\b|'
            r'\bgpu\b|\bcpu\b|\bnvidia\b|'
            r'\bamd\b|\bintel\b|\btsmc\b',
            text_lower
        ))),

        'kw_tech': int(bool(re.search(
            r'\btech\b|\btechnology\b|\bsoftware\b|'
            r'\bcloud\b|\bcyber\b|\bapple\b|'
            r'\bgoogle\b|\bmicrosoft\b',
            text_lower
        ))),

        'kw_ai': int(bool(re.search(
            r'\bai\b|\bartificial intelligence\b|'
            r'\bmachine learning\b|\bdeep learning\b|'
            r'\bchatgpt\b|\bopenai\b',
            text_lower
        ))),

        'kw_market': int(bool(re.search(
            r'\bstocks?\b|\bnasdaq\b|\bwall street\b|'
            r'\brecession\b|\binflation\b|\bfed\b',
            text_lower
        ))),

        'kw_supply_chain': int(bool(re.search(
            r'\bsupply chain\b|\bmanufacturing\b|'
            r'\blogistics\b|\bshipping\b',
            text_lower
        ))),

        'kw_military': int(bool(re.search(
            r'\bmilitary\b|\bmissile\b|\bnavy\b|'
            r'\btaiwan strait\b|\bsouth china sea\b',
            text_lower
        )))
    }


# ============================================================
# Event Features
# ============================================================

def extract_event_features(text):

    text_str = str(text)

    words = text_str.split()

    # ----------------------------
    # Text Length
    # ----------------------------
    char_count = len(text_str)

    word_count = len(words)

    avg_word_length = (
        np.mean([len(w) for w in words])
        if len(words) > 0 else 0
    )

    # ----------------------------
    # ALL CAPS intensity
    # ----------------------------
    caps_words = [
        w for w in words
        if len(w) >= 2 and w.isupper()
    ]

    caps_ratio = (
        len(caps_words) / len(words)
        if len(words) > 0 else 0
    )

    # ----------------------------
    # Punctuation intensity
    # ----------------------------
    exclamation_count = text_str.count('!')
    question_count = text_str.count('?')

    exclamation_ratio = (
        exclamation_count / max(1, word_count)
    )

    question_ratio = (
        question_count / max(1, word_count)
    )

    # ----------------------------
    # Number intensity
    # ----------------------------
    number_count = len(re.findall(r'\d+', text_str))

    # ----------------------------
    # Hashtags / mentions
    # ----------------------------
    hashtag_count = len(re.findall(r'#\w+', text_str))

    mention_count = len(re.findall(r'@\w+', text_str))

    # ----------------------------
    # URL count
    # ----------------------------
    url_count = len(re.findall(r'http\S+', text_str))

    # ----------------------------
    # Repeated punctuation
    # ----------------------------
    repeated_exclamation = int("!!" in text_str)
    repeated_question = int("??" in text_str)

    # ----------------------------
    # Emotional escalation
    # ----------------------------
    emotional_intensity = (
        exclamation_count
        + len(caps_words)
        + repeated_exclamation * 3
    )

    return {

        "char_count": char_count,
        "word_count": word_count,
        "avg_word_length": avg_word_length,

        "caps_ratio": caps_ratio,

        "exclamation_count": exclamation_count,
        "question_count": question_count,

        "exclamation_ratio": exclamation_ratio,
        "question_ratio": question_ratio,

        "number_count": number_count,

        "hashtag_count": hashtag_count,
        "mention_count": mention_count,
        "url_count": url_count,

        "repeated_exclamation": repeated_exclamation,
        "repeated_question": repeated_question,

        "emotional_intensity": emotional_intensity,
    }


# ============================================================
# Time Features
# ============================================================

def extract_time_features(timestamp):

    ts = pd.to_datetime(timestamp)

    hour = ts.hour

    return {

        "post_hour": hour,

        "post_dayofweek": ts.dayofweek,

        "is_weekend": int(ts.dayofweek >= 5),

        "is_night_post": int(
            hour >= 0 and hour <= 5
        ),

        "is_market_hours": int(
            9 <= hour <= 16
        ),

        "is_after_market": int(
            16 < hour <= 20
        )
    }


# ============================================================
# Main
# ============================================================

def main():

    input_file = "merged_trump_posts.csv"
    output_file = "trump_posts_features_2017_2026.csv"

    checkpoint_interval = 500

    print(f"Loading {input_file}")

    df = pd.read_csv(input_file)

    df["Content"] = df["Content"].astype(str)

    # ========================================================
    # Resume checkpoint
    # ========================================================

    if os.path.exists(output_file):

        print(f"Found checkpoint: {output_file}")

        processed_df = pd.read_csv(output_file)

        start_idx = len(processed_df)

        df_out = processed_df.to_dict("records")

        print(f"Resuming from {start_idx}")

    else:

        start_idx = 0

        df_out = []

    if start_idx >= len(df):

        print("Already complete")

        return

    # ========================================================
    # Models
    # ========================================================

    print(f"Loading models on device: {get_device()}")

    analyzer = SentimentIntensityAnalyzer()

    emotion_classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        device=get_device(),
        truncation=True,
        max_length=512
    )

    # ========================================================
    # Feature extraction
    # ========================================================

    for i in tqdm(range(start_idx, len(df))):

        row = df.iloc[i].to_dict()

        text = row["Content"]

        # ====================================================
        # Keyword Features
        # ====================================================

        kws = extract_keywords(text)

        row.update(kws)

        # ====================================================
        # Event Features
        # ====================================================

        event_features = extract_event_features(text)

        row.update(event_features)

        # ====================================================
        # Time Features
        # ====================================================

        time_features = extract_time_features(
            row["Timestamp"]
        )

        row.update(time_features)

        # ====================================================
        # VADER Sentiment
        # ====================================================

        scores = analyzer.polarity_scores(text)

        row["vader_compound"] = scores["compound"]

        row["vader_pos"] = scores["pos"]

        row["vader_neg"] = scores["neg"]

        row["vader_neu"] = scores["neu"]

        # ====================================================
        # Emotion Classification
        # ====================================================

        try:

            emotion_res = emotion_classifier(text)

            row["emotion_label"] = emotion_res[0]["label"]

            row["emotion_score"] = emotion_res[0]["score"]

        except Exception:

            row["emotion_label"] = "unknown"

            row["emotion_score"] = 0.0

        # ====================================================
        # Engagement Features
        # ====================================================

        likes = row.get("Likes", 0)

        retweets = row.get("Retweets", 0)

        if pd.isna(likes):
            likes = 0

        if pd.isna(retweets):
            retweets = 0

        likes = max(0, float(likes))

        retweets = max(0, float(retweets))

        # log normalization
        row["log_likes"] = math.log1p(likes)

        row["log_retweets"] = math.log1p(retweets)

        row["engagement_score"] = (
            row["log_likes"]
            + row["log_retweets"]
        )

        # weighted sentiment
        row["weighted_vader"] = (
            scores["compound"]
            * row["engagement_score"]
        )

        # ====================================================
        # Viral / anomaly signals
        # ====================================================

        row["viral_score"] = (
            row["engagement_score"]
            * (
                1
                + row["caps_ratio"]
                + row["exclamation_ratio"]
            )
        )

        # ====================================================
        # Keyword density
        # ====================================================

        keyword_sum = sum([
            v for k, v in kws.items()
        ])

        row["keyword_density"] = (
            keyword_sum / max(1, row["word_count"])
        )

        # ====================================================
        # Composite event score
        # ====================================================

        row["event_score"] = (
            row["weighted_vader"]
            + row["emotional_intensity"]
            + row["viral_score"]
            + keyword_sum
        )

        df_out.append(row)

        # ====================================================
        # Checkpoint save
        # ====================================================

        if (
            (i + 1) % checkpoint_interval == 0
            or (i + 1) == len(df)
        ):

            pd.DataFrame(df_out).to_csv(
                output_file,
                index=False
            )

    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
