import argparse
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None

try:
    from transformers import pipeline
except ImportError:
    pipeline = None


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

def _has(pattern, text_lower):
    return int(bool(re.search(pattern, text_lower)))


def extract_keywords(text):
    text_lower = text.lower()

    keywords = {

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

    # Trump-code style event families. These are deliberately broad lexical
    # hooks; the neural model learns whether a hook is useful for each asset.
    keywords.update({
        "tc_tariff": _has(r"\btariffs?\b|\bdut(?:y|ies)\b|\btrade war\b|\bsection 301\b", text_lower),
        "tc_deal": _has(r"\bdeal\b|\bagreement\b|\bsigned\b|\bnegotiate|negotiation\b|\bframework\b", text_lower),
        "tc_relief": _has(r"\bpause\b|\bexempt|exemption\b|\bsuspend\b|\bdelay\b|\bextend\b|\bwaiver\b", text_lower),
        "tc_action": _has(r"\bimmediately\b|\bhereby\b|\bexecutive order\b|\bjust signed\b|\bordered\b", text_lower),
        "tc_attack": _has(r"\bfake news\b|\bcorrupt\b|\bfraud\b|\bwitch hunt\b|\bterrible\b|\bdisaster\b", text_lower),
        "tc_positive": _has(r"\bgreat\b|\btremendous\b|\bincredible\b|\bhistoric\b|\bbeautiful\b|\bperfect\b|\bstrong\b", text_lower),
        "tc_market_brag": _has(r"\bstock market\b|\ball[- ]time high\b|\brecord high\b|\bdow\b|\bs&p\b|\bnasdaq\b", text_lower),
        "tc_iran": _has(r"\biran\b|\biranian\b", text_lower),
        "tc_russia": _has(r"\brussia\b|\bputin\b|\bukraine\b", text_lower),
        "tc_fed": _has(r"\bfed\b|\bfederal reserve\b|\bpowell\b|\brate cut\b|\binterest rates?\b", text_lower),
        "tc_energy": _has(r"\boil\b|\bgas\b|\benergy\b|\bopec\b|\bdrill\b", text_lower),
    })

    return keywords


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

    alpha_count = sum(1 for ch in text_str if ch.isalpha())
    upper_count = sum(1 for ch in text_str if ch.isupper())
    uppercase_char_ratio = upper_count / max(1, alpha_count)

    return {

        "char_count": char_count,
        "word_count": word_count,
        "avg_word_length": avg_word_length,

        "caps_ratio": caps_ratio,
        "uppercase_char_ratio": uppercase_char_ratio,

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

    ts = pd.to_datetime(timestamp, utc=True)
    ts_et = ts.tz_convert("America/New_York")
    ts_tw = ts.tz_convert("Asia/Taipei")

    hour = ts.hour
    et_hour = ts_et.hour
    et_minute = ts_et.minute

    return {

        "post_hour": hour,

        "post_dayofweek": ts.dayofweek,
        "post_hour_et": et_hour,
        "post_dayofweek_et": ts_et.dayofweek,
        "post_hour_tw": ts_tw.hour,

        "is_weekend": int(ts.dayofweek >= 5),

        "is_night_post": int(
            et_hour <= 5 or et_hour >= 23
        ),

        "is_market_hours": int(
            (et_hour > 9 or (et_hour == 9 and et_minute >= 30)) and et_hour < 16
        ),
        "is_pre_market": int(et_hour < 9 or (et_hour == 9 and et_minute < 30)),

        "is_after_market": int(
            16 <= et_hour <= 20
        )
    }


def extract_signature_features(text):
    text_str = str(text)
    return {
        "sig_djt": int("President DJT" in text_str),
        "sig_potus": int("PRESIDENT OF THE UNITED STATES" in text_str),
        "sig_tyfa": int("Thank you for your attention" in text_str),
        "is_retweet_text": int(text_str.strip().lower().startswith("rt ")),
    }


def build_trumpcode_scores(row):
    tariff = row["tc_tariff"]
    deal = row["tc_deal"]
    relief = row["tc_relief"]
    action = row["tc_action"]
    positive = row["tc_positive"]
    attack = row["tc_attack"]
    market_brag = row["tc_market_brag"]

    row["tc_pre_tariff"] = int(row["is_pre_market"] and tariff)
    row["tc_pre_deal"] = int(row["is_pre_market"] and deal)
    row["tc_pre_relief"] = int(row["is_pre_market"] and relief)
    row["tc_pre_action"] = int(row["is_pre_market"] and action)
    row["tc_open_tariff"] = int(row["is_market_hours"] and tariff)
    row["tc_open_deal"] = int(row["is_market_hours"] and deal)
    row["tc_night_tariff"] = int(row["is_night_post"] and tariff)

    row["tc_deal_over_tariff_post"] = int(deal and not tariff)
    row["tc_tariff_only_post"] = int(tariff and not deal)
    row["tc_relief_positive_post"] = int(relief and positive)
    row["tc_attack_market_post"] = int(attack and market_brag)

    row["tc_directional_pressure"] = (
        1.5 * relief
        + 1.0 * deal
        + 0.8 * action
        + 0.6 * positive
        - 1.4 * tariff
        - 0.7 * attack
        - 0.6 * row["tc_night_tariff"]
    )
    row["tc_event_intensity"] = (
        row["tc_tariff"]
        + row["tc_deal"]
        + row["tc_relief"]
        + row["tc_action"]
        + row["tc_attack"]
        + row["tc_market_brag"]
        + row.get("kw_china", 0)
        + row.get("kw_taiwan", 0)
        + row.get("kw_chips", 0)
    )
    return row


def resolve_paths(input_file, output_file):
    base = Path(__file__).resolve().parent
    input_path = Path(input_file)
    output_path = Path(output_file)
    if not input_path.is_absolute():
        candidates = [
            Path.cwd() / input_path,
            base / input_path,
            base / "trump_posts_features_2017_2026.csv",
            base / "merged_trump_posts.csv",
            base / "trump_truth_social_posts.csv",
            base.parent / "text" / "trump_posts_features_2017_2026.csv",
        ]
        input_path = next((p for p in candidates if p.exists()), candidates[0])
    if not output_path.is_absolute():
        output_path = base / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return input_path, output_path


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="merged_trump_posts.csv")
    parser.add_argument("--output", default="trump_posts_features_2017_2026.csv")
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--skip-emotion-model", action="store_true")
    parser.add_argument("--reuse-existing-nlp", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_file, output_file = resolve_paths(args.input, args.output)

    checkpoint_interval = args.checkpoint_interval

    print(f"Loading {input_file}")

    df = pd.read_csv(input_file)

    df["Content"] = df["Content"].astype(str)

    # ========================================================
    # Resume checkpoint
    # ========================================================

    if os.path.exists(output_file) and not args.force:

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

    if args.reuse_existing_nlp:
        analyzer = None
    else:
        if SentimentIntensityAnalyzer is None:
            raise ImportError("nltk is required unless --reuse-existing-nlp is used.")
        analyzer = SentimentIntensityAnalyzer()

    emotion_classifier = None
    if not args.skip_emotion_model and not args.reuse_existing_nlp:
        if pipeline is None:
            raise ImportError("transformers is required unless --skip-emotion-model or --reuse-existing-nlp is used.")
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

        row.update(extract_signature_features(text))

        # ====================================================
        # VADER Sentiment
        # ====================================================

        if analyzer is None and "vader_compound" in row:
            scores = {
                "compound": float(row.get("vader_compound", 0.0) or 0.0),
                "pos": float(row.get("vader_pos", 0.0) or 0.0),
                "neg": float(row.get("vader_neg", 0.0) or 0.0),
                "neu": float(row.get("vader_neu", 0.0) or 0.0),
            }
        else:
            scores = analyzer.polarity_scores(text)

        row["vader_compound"] = scores["compound"]

        row["vader_pos"] = scores["pos"]

        row["vader_neg"] = scores["neg"]

        row["vader_neu"] = scores["neu"]

        # ====================================================
        # Emotion Classification
        # ====================================================

        try:

            if args.reuse_existing_nlp and "emotion_label" in row and "emotion_score" in row:
                row["emotion_label"] = row.get("emotion_label", "unknown")
                row["emotion_score"] = float(row.get("emotion_score", 0.0) or 0.0)
                raise StopIteration

            if emotion_classifier is None:
                raise RuntimeError("emotion model skipped")

            emotion_res = emotion_classifier(text)

            row["emotion_label"] = emotion_res[0]["label"]

            row["emotion_score"] = emotion_res[0]["score"]

        except StopIteration:
            pass

        except Exception:

            if not args.reuse_existing_nlp:
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
        if args.reuse_existing_nlp and "weighted_vader" in row and not pd.isna(row["weighted_vader"]):
            row["weighted_vader"] = float(row["weighted_vader"])
        else:
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

        row = build_trumpcode_scores(row)

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
