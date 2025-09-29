import requests
import uuid
from urllib.parse import urljoin, urlencode, urlparse, parse_qs, urlunparse
import time
from itertools import islice, tee
import json
import ast
import pandas as pd
from pathlib import Path

# === URL SETUP === #

BASE_URL = "https://www.taptap.cn/webapiv2/"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)

PLATFORM = {
    "ios": "iOS",
    "android": "Android"
}

# === SUPPORTING FUNCTIONS & CLASSES === #

def beautify_json_string(json_str: str) -> str:
    try:
        obj = json.loads(json_str)
        pretty = json.dumps(obj, indent=2, ensure_ascii=False)
        return pretty
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"
    
def beautify_dict_string(input_str: str) -> str:
    try:
        obj = ast.literal_eval(input_str)  # Convert Python-style dict string to actual dict
        pretty = json.dumps(obj, indent=2, ensure_ascii=False)
        return pretty
    except (ValueError, SyntaxError) as e:
        return f"Invalid input: {e}"
    
def categorize_ratings_by_value(ratings: list) -> dict:
    result = {'up': [], 'down': []}

    for rating in ratings:
        value = rating.get('value')
        rating_type = rating.get('type')
        if value in result:
            result[value].append(rating_type)

    return result

def convert_and_stream_reviews(raw_reviews, review_count, out_file):
    ensure_parent_dir(out_file)
    print(f"▶ Converting and writing {review_count} reviews to {out_file}...")
    with open(out_file, "w", encoding="utf-8") as f:
        for index, review in enumerate(raw_reviews, start=1):
            if index % 50 == 0 or index == review_count:
                print(f"   • Processed {index}/{review_count} reviews")

            r = review.get('review', {})
            a = review.get('author', {}).get('user', {})
            app = review.get('app', {})
            stat = review.get('stat', {})

            concise_review = {
                'app_id': app.get('id'),
                'app_title': app.get('title'),
                'review_id': r.get('id'),
                'review_publish_time': review.get('publish_time'),
                'review_score': r.get('score'),
                'review_author_user_id': a.get('id'),
                'review_author_user_name': a.get('name'),
                'device': review.get('device'),
                'review_played_spent': r.get('played_spent'),
                'up_rating_aspects': categorize_ratings_by_value(r.get('ratings', []))['up'],
                'down_rating_aspects': categorize_ratings_by_value(r.get('ratings', []))['down'],
                'review_content_raw_text': r.get('contents', {}).get('raw_text'),
                'comments': stat.get('comments'),
                'review_up_votes': stat.get('ups'),
                'review_down_votes': stat.get('downs')
            }

            f.write(json.dumps(concise_review, ensure_ascii=False) + "\n")
    print("✔ Finished writing reviews")

def tee_and_len(iterable):
    a, b = tee(iterable)
    count = sum(1 for _ in a)
    return b, count

def ensure_parent_dir(path_str: str) -> None:
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)

class TapTapClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "user-agent": USER_AGENT
        })

    def _build_url(self, path: str, params: dict, platform: str = "android") -> str:
        full_url = urljoin(BASE_URL, path)
        parsed = urlparse(full_url)
        query = parse_qs(parsed.query)

        # Merge params into query
        for k, v in params.items():
            query[k] = v

        # Handle platform override
        platform = platform.lower()
        device_platform = PLATFORM.get(platform, "Android")
        query["X-UA"] = (
            f"V=1&PN=WebApp&LANG=zh_CN&VN_CODE=102&LOC=CN&PLT=PC&"
            f"DS={device_platform}&UID={uuid.uuid4()}&OS=Mac+OS&OSV=10.15.7&DT=PC"
        )

        encoded_query = urlencode(query, doseq=True)
        final_url = urlunparse(parsed._replace(query=encoded_query))
        return final_url

    def get(self, path: str, params: dict = None, platform: str = "android"):
        if params is None:
            params = {}

        url = self._build_url(path, params, platform)
        print(f"GET {url}")
        response = self.session.get(url)
        response.raise_for_status()

        data = response.json()
        if not data.get("success", False):
            raise Exception("Request failed")

        return data.get("data", {})

    def list(self, path: str, params: dict = None, platform: str = "android"):
        if params is None:
            params = {}
        params.setdefault("from", 0)
        params.setdefault("limit", 10)

        page_num = 0
        total = None
        while True:
            page_num += 1
            print(f"▶ Fetching page {page_num} (from={params['from']}, limit={params['limit']})")
            data = self.get(path, params, platform)
            total = data.get("total", 0)
            items = data.get("list", [])
            next_page = data.get("next_page")

            print(f"   • Got {len(items)} items (total so far: {params['from']+len(items)}/{total})")

            for item in items:
                yield item

            if not next_page or params["from"] + len(items) >= total:
                print("✔ Reached end of list")
                break

            params["from"] += len(items)
            time.sleep(1)

    def get_app(self, app_id, platform: str = "android", **params):
        print(f"▶ Fetching app detail for {app_id}")
        params["id"] = app_id
        return self.get("app/v4/detail", params, platform)

    def list_apps(self, type_name="reserve", platform: str = "android", **params):
        print(f"▶ Listing apps of type={type_name}")
        params.setdefault("type_name", type_name)
        for row in self.list("app-top/v2/hits", params, platform):
            if not row.get("is_add") and row.get("type") == "app":
                count += 1
                if count % 20 == 0:
                    print(f"   • Processed {count} apps")
                yield row["app"]
        print(f"✔ Done listing apps (total {count})")

    def list_reviews(self, app_id, sort="new", platform: str = "android", **params):
        print(f"▶ Listing reviews for app_id={app_id}, sort={sort}")
        params.update({
            "app_id": app_id,
            "sort": sort
        })
        count = 0
        for row in self.list("review/v2/list-by-app", params, platform):
            if row.get("type") == "moment":
                count += 1
                if count % 50 == 0:
                    print(f"   • Processed {count} reviews")
                yield row["moment"]
        print(f"✔ Done listing reviews (total {count})")