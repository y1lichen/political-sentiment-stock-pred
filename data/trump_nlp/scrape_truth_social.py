import csv
import time
import random
import sys
import os
from bs4 import BeautifulSoup
from curl_cffi import requests

def clean_html(raw_html):
    if not raw_html:
        return ""
    return BeautifulSoup(raw_html, "html.parser").get_text(separator=' ', strip=True)

def main():
    base_url = "https://truthsocial.com/api/v1/accounts/107780257626128497/statuses"
    limit = 40
    csv_filename = "trump_truth_social_posts.csv"
    
    # Check for last max_id
    max_id = None
    file_exists = os.path.isfile(csv_filename)
    if file_exists:
        try:
            with open(csv_filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    last_line = lines[-1]
                    reader = csv.reader([last_line])
                    last_row = list(reader)[0]
                    if len(last_row) > 0:
                        last_url = last_row[-1]
                        if last_url.startswith('http'):
                            max_id = last_url.rstrip('/').split('/')[-1]
                            print(f"Resuming from max_id: {max_id}")
        except Exception as e:
            print("Could not read last max_id. Starting from beginning.")
    
    csv_file = open(csv_filename, mode='a' if file_exists else 'w', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(["Timestamp", "Type", "Content", "URL"])
    
    count = 0
    consecutive_429 = 0
    params = {"limit": limit}

    try:
        while True:
            if max_id:
                params["max_id"] = max_id
                
            try:
                response = requests.get(base_url, params=params, impersonate="chrome", timeout=15)
            except Exception as e:
                print(f"Request failed: {e}. Retrying in 10 seconds...")
                time.sleep(10)
                continue
                
            if response.status_code != 200:
                print(f"Error {response.status_code}")
                if response.status_code == 429 or response.status_code == 403:
                    consecutive_429 += 1
                    sleep_time = 30 * consecutive_429
                    print(f"Cloudflare/Rate limit hit. Sleeping for {sleep_time}s...")
                    time.sleep(sleep_time)
                    continue
                else:
                    print("Failed with unrecoverable status:", response.text[:200])
                    time.sleep(10)
                    continue
                    
            consecutive_429 = 0
            data = response.json()
            if not data:
                print("No more data returned. Probably reached the end.")
                break
                
            for post in data:
                created_at = post.get("created_at")
                post_url = post.get("url")
                
                if post.get("reblog"):
                    post_type = "Re-Truth"
                    content = clean_html(post["reblog"].get("content"))
                else:
                    post_type = "Original"
                    content = clean_html(post.get("content"))
                    
                writer.writerow([created_at, post_type, content, post_url])
                max_id = post.get("id")
                count += 1
                
            print(f"Scraped {count} posts in this run... (latest max_id: {max_id})", flush=True)
            csv_file.flush()
            
            # Use a longer, randomized wait time to avoid being flagged as a bot by Cloudflare easily
            wait_time = random.uniform(2.5, 5.0)
            time.sleep(wait_time)
            
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        csv_file.close()
        print(f"Finished. Scraped a total of {count} posts this session.")

if __name__ == "__main__":
    main()
