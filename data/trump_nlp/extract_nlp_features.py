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

import re

def extract_keywords(text):
    text_lower = text.lower()

    return {

        # =====================================================
        # China / CCP / US-China Relations
        # =====================================================
        'kw_china': int(bool(re.search(
            r'\bchina\b|\bchinese\b|\bccp\b|\bcommunist party\b|'
            r'\bbeijing\b|\bxi jinping\b|\bprc\b|'
            r'\bmainland\b|\bred china\b|\bcommunists?\b|'
            r'\bmade in china\b|\bchina deal\b|'
            r'\bchina trade\b|\bchina virus\b|'
            r'\bhuawei\b|\btiktok\b|\balibaba\b|\btencent\b',
            text_lower
        ))),

        # =====================================================
        # Taiwan / Cross-Strait
        # =====================================================
        'kw_taiwan': int(bool(re.search(
            r'\btaiwan\b|\btaiwanese\b|'
            r'\brepublic of china\b|\broc\b|'
            r'\bformosa\b|\btaipei\b|'
            r'\btsmc\b|\bfoxconn\b|'
            r'\bcross[- ]strait\b|\btaiwan strait\b',
            text_lower
        ))),

        # =====================================================
        # Tariffs / Trade War / Trade Policy
        # =====================================================
        'kw_tariffs': int(bool(re.search(
            r'\btariffs?\b|\btrade war\b|'
            r'\bimport tax\b|\bduties\b|'
            r'\bsection 301\b|\bsection 232\b|'
            r'\btrade deficit\b|\btrade surplus\b|'
            r'\bprotectionism\b|\btrade agreement\b|'
            r'\bfree trade\b|\bdeal with china\b|'
            r'\bmade in america\b|\bamerica first\b',
            text_lower
        ))),

        # =====================================================
        # Sanctions / Export Controls / Restrictions
        # =====================================================
        'kw_sanctions': int(bool(re.search(
            r'\bsanctions?\b|\bexport ban\b|'
            r'\bexport controls?\b|\bblacklist\b|'
            r'\bentity list\b|\bembargo\b|'
            r'\brestrictions?\b|\bbanned\b|'
            r'\bdecoupling\b|\bnational security\b|'
            r'\bcfius\b|\bforced delisting\b|'
            r'\bchip ban\b|\btechnology ban\b',
            text_lower
        ))),

        # =====================================================
        # Semiconductor / Chips / Hardware
        # =====================================================
        'kw_chips': int(bool(re.search(
            r'\bchips?\b|\bsemiconductors?\b|'
            r'\bwafer\b|\bfoundry\b|\bfab\b|'
            r'\bintegrated circuits?\b|\bic\b|'
            r'\bgpu\b|\bcpu\b|\baccelerator\b|'
            r'\bchipmaking\b|\bchipmaker\b|'
            r'\badvanced node\b|\b3nm\b|\b5nm\b|\b7nm\b|'
            r'\btsmc\b|\bintel\b|\bnvidia\b|'
            r'\bamd\b|\bqualcomm\b|\bmicron\b|'
            r'\basml\b|\bbroadcom\b|\bmediatek\b|'
            r'\barm\b|\bsamsung electronics\b',
            text_lower
        ))),

        # =====================================================
        # Technology / Big Tech / Digital Economy
        # =====================================================
        'kw_tech': int(bool(re.search(
            r'\btech\b|\btechnology\b|\bbig tech\b|'
            r'\bsilicon valley\b|\bsoftware\b|'
            r'\bcloud\b|\bcyber\b|\bdigital\b|'
            r'\bcybersecurity\b|\bdata center\b|'
            r'\binternet\b|\bplatform\b|'
            r'\b5g\b|\b6g\b|\bcomputing\b|'
            r'\bsmartphone\b|\biphone\b|'
            r'\bmeta\b|\bgoogle\b|\bmicrosoft\b|'
            r'\bapple\b|\bamazon\b|\bfacebook\b',
            text_lower
        ))),

        # =====================================================
        # AI / Machine Learning / Automation
        # =====================================================
        'kw_ai': int(bool(re.search(
            r'\bai\b|\bartificial intelligence\b|'
            r'\bgenerative ai\b|\bmachine learning\b|'
            r'\bdeep learning\b|\bneural network\b|'
            r'\bllm\b|\blarge language model\b|'
            r'\bautomation\b|\brobotics\b|'
            r'\bopenai\b|\bchatgpt\b|'
            r'\bgpt[- ]?4\b|\bgpt[- ]?5\b|'
            r'\bcopilot\b|\bautonomous\b',
            text_lower
        ))),

        # =====================================================
        # Market / Economy / Stocks / Financial Risk
        # =====================================================
        'kw_market': int(bool(re.search(
            r'\bstock market\b|\bstocks?\b|'
            r'\bshares?\b|\bequities\b|'
            r'\bwall street\b|\bnasdaq\b|'
            r'\bdow jones\b|\bsp500\b|\bs&p\b|'
            r'\bbull market\b|\bbear market\b|'
            r'\bmarket crash\b|\bsell[- ]off\b|'
            r'\brecession\b|\binflation\b|'
            r'\binterest rates?\b|\bfed\b|'
            r'\bfederal reserve\b|\beconomy\b|'
            r'\bgdp\b|\bunemployment\b',
            text_lower
        ))),

        # =====================================================
        # Supply Chain / Manufacturing / Logistics
        # =====================================================
        'kw_supply_chain': int(bool(re.search(
            r'\bsupply chain\b|\bmanufacturing\b|'
            r'\bfactory\b|\bproduction\b|'
            r'\bassembly\b|\bindustrial\b|'
            r'\boutsourcing\b|\bonshoring\b|'
            r'\breshoring\b|\blogistics\b|'
            r'\bshipping\b|\bcontainer\b|'
            r'\bport congestion\b|\bfreight\b|'
            r'\bglobal supply\b|\bprocurement\b',
            text_lower
        ))),

        # =====================================================
        # Military / Taiwan Strait / Security
        # =====================================================
        'kw_military': int(bool(re.search(
            r'\bmilitary\b|\bnavy\b|\bmissile\b|'
            r'\bwarships?\b|\bair force\b|'
            r'\barmy\b|\bdefense\b|\bdefence\b|'
            r'\bweapons?\b|\bdrone\b|'
            r'\binvasion\b|\bconflict\b|'
            r'\bsecurity\b|\bnational defense\b|'
            r'\bsouth china sea\b|\btaiwan strait\b|'
            r'\bpla\b|\bpeople\'s liberation army\b',
            text_lower
        )))
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
