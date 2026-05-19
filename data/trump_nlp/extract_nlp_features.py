import pandas as pd
import re
import os
import torch
from tqdm import tqdm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

def extract_keywords(text):
    text_lower = text.lower()
    return {
        'kw_china': int(bool(re.search(r'\bchina\b|\bchinese\b', text_lower))),
        'kw_taiwan': int(bool(re.search(r'\btaiwan\b|\btaiwanese\b', text_lower))),
        'kw_tariffs': int(bool(re.search(r'\btariffs?\b|\btax\b', text_lower))),
        'kw_sanctions': int(bool(re.search(r'\bsanctions?\b', text_lower))),
        'kw_chips': int(bool(re.search(r'\bchips?\b|\bsemiconductors?\b', text_lower))),
        'kw_tech': int(bool(re.search(r'\btech\b|\btechnology\b', text_lower))),
        'kw_ai': int(bool(re.search(r'\bai\b|\bartificial intelligence\b', text_lower)))
    }

def main():
    input_file = 'merged_trump_posts.csv'
    output_file = 'trump_posts_features_2017_2026.csv'
    checkpoint_interval = 500
    
    print(f"Loading input data: {input_file}")
    df = pd.read_csv(input_file)
    # Ensure Content is string
    df['Content'] = df['Content'].astype(str)
    
    # Initialize Checkpoint
    if os.path.exists(output_file):
        print(f"Found existing checkpoint: {output_file}")
        processed_df = pd.read_csv(output_file)
        start_idx = len(processed_df)
        print(f"Resuming from index {start_idx} out of {len(df)}")
        df_out = processed_df.to_dict('records')
    else:
        start_idx = 0
        df_out = []
    
    if start_idx >= len(df):
        print("All rows already processed!")
        return

    print(f"Setting up models... (Using device: {get_device()})")
    # VADER
    analyzer = SentimentIntensityAnalyzer()
    
    # Transformers Emotion
    # We use j-hartmann/emotion-english-distilroberta-base
    emotion_classifier = pipeline("text-classification", 
                                  model="j-hartmann/emotion-english-distilroberta-base", 
                                  device=get_device(),
                                  truncation=True,
                                  max_length=512)

    print("Starting feature extraction...")
    for i in tqdm(range(start_idx, len(df))):
        row = df.iloc[i].to_dict()
        text = row['Content']
        
        # 1. Keywords
        kws = extract_keywords(text)
        row.update(kws)
        
        # 2. VADER Sentiment
        scores = analyzer.polarity_scores(text)
        row['vader_compound'] = scores['compound']
        
        # 3. Emotion (Transformers)
        try:
            # Output format: [{'label': 'joy', 'score': 0.99}]
            emotion_res = emotion_classifier(text)
            row['emotion_label'] = emotion_res[0]['label']
            row['emotion_score'] = emotion_res[0]['score']
        except Exception as e:
            row['emotion_label'] = 'unknown'
            row['emotion_score'] = 0.0

        # 4. Engagement Weighted Sentiment (if likes > 0)
        import math
        likes = row.get('Likes', 0)
        if pd.isna(likes): likes = 0
        weight = math.log1p(max(0, float(likes)))
        row['weighted_vader'] = scores['compound'] * weight

        df_out.append(row)
        
        # Checkpoint save
        if (i + 1) % checkpoint_interval == 0 or (i + 1) == len(df):
            pd.DataFrame(df_out).to_csv(output_file, index=False)
    
    print(f"\nCompleted! Saved features to {output_file}")

if __name__ == "__main__":
    main()
