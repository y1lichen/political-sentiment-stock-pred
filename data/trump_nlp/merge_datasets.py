import pandas as pd
import json
from datetime import datetime

print("Loading Kaggle tweets.csv...")
df_tweets = pd.read_csv('tweets.csv')
df_tweets['Timestamp'] = pd.to_datetime(df_tweets['date'], errors='coerce')

# Filter for 2017-01-01 ~ 2021-01-09
df_tweets = df_tweets[(df_tweets['Timestamp'] >= '2017-01-01') & (df_tweets['Timestamp'] <= '2021-01-09')]
df_tweets = df_tweets[['Timestamp', 'text', 'favorites', 'retweets']].rename(columns={'text': 'Content', 'favorites': 'Likes', 'retweets': 'Retweets'})
df_tweets['Platform'] = 'Twitter_Legacy'

print("Loading Truth Social posts...")
try:
    with open('trump-code/data/trump_posts_all.json', 'r', encoding='utf-8') as f:
        ts_data = json.load(f)
        
    ts_rows = []
    for post in ts_data.get('posts', []):
        ts_rows.append({
            'Timestamp': post.get('created_at'),
            'Content': post.get('content', ''),
            'Platform': 'Truth_Social',
            'Likes': 0, # Not available in this dataset
            'Retweets': 0
        })
    df_ts = pd.DataFrame(ts_rows)
    df_ts['Timestamp'] = pd.to_datetime(df_ts['Timestamp'], errors='coerce')
except Exception as e:
    print("Error loading Truth Social data:", e)
    df_ts = pd.DataFrame()

print("Loading X (Twitter) new posts...")
try:
    with open('trump-code/data/x_posts_full.json', 'r', encoding='utf-8') as f:
        x_data = json.load(f)

    x_rows = []
    for post in x_data.get('tweets', []):
        metrics = post.get('public_metrics', {})
        x_rows.append({
            'Timestamp': post.get('created_at'),
            'Content': post.get('text', ''),
            'Platform': 'X',
            'Likes': metrics.get('like_count', 0),
            'Retweets': metrics.get('retweet_count', 0)
        })
    df_x = pd.DataFrame(x_rows)
    df_x['Timestamp'] = pd.to_datetime(df_x['Timestamp'], errors='coerce')
except Exception as e:
    print("Error loading X data:", e)
    df_x = pd.DataFrame()

print("Merging datasets...")
df_all = pd.concat([df_tweets, df_ts, df_x], ignore_index=True)

# Sort and filter to 2017/01/01 ~ 2026/04/30
df_all = df_all.dropna(subset=['Timestamp', 'Content'])
df_all['Timestamp'] = pd.to_datetime(df_all['Timestamp'], errors='coerce', utc=True)
df_all['Timestamp_naive'] = df_all['Timestamp'].dt.tz_localize(None)
df_all = df_all[(df_all['Timestamp_naive'] >= '2017-01-01') & (df_all['Timestamp_naive'] <= '2026-04-30')]
df_all = df_all.drop(columns=['Timestamp_naive'])
df_all = df_all.sort_values(by='Timestamp')

df_all['Content'] = df_all['Content'].astype(str)

print("\nData Shape:", df_all.shape)
print("Data Distribution by Platform:")
print(df_all['Platform'].value_counts())

print("\nSaving to merged_trump_posts.csv...")
df_all.to_csv('merged_trump_posts.csv', index=False)
print("Merge Complete!")
