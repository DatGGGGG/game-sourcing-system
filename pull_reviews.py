import pandas as pd
import re
from pathlib import Path
import sys
import taptap_data_helpers as tdh


def fetch_reviews(game_url: str, output_root: str = "output") -> None:
    """
    Fetch reviews for a given TapTap game URL and save them to JSONL + CSV.
    Args:
        game_url: Full TapTap game URL, e.g. https://www.taptap.cn/app/209601?os=android
        output_root: Root output directory (default: "output")
    """
    # Extract app id
    match = re.search(r"/app/(\d+)", game_url)
    if not match:
        raise ValueError(f"No TapTap app id found in URL: {game_url}")
    app_id = match.group(1)
    print("TapTap App ID:", app_id)

    # Setup output paths
    out_dir = Path(output_root) / app_id
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"reviews_{app_id}.jsonl"
    csv_path = out_dir / f"reviews_{app_id}.csv"

    print(f"All relevant data for the app will be saved inside {out_dir}/")

    # Fetch reviews
    taptap_client = tdh.TapTapClient()
    print("Getting App Reviews (combined iOS + Android feed)")
    reviews_iter = taptap_client.list_reviews(app_id)
    reviews_iter, review_count = tdh.tee_and_len(reviews_iter)

    tdh.convert_and_stream_reviews(reviews_iter, review_count, out_file=str(jsonl_path))
    print(f"Review JSONL saved → {jsonl_path}")

    # Convert to DataFrame and save CSV
    df_reviews = pd.read_json(jsonl_path, lines=True)
    print(f"Exporting the reviews to: {csv_path}")
    df_reviews.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("✔ Done with getting the reviews!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pull_reviews.py <game_url>")
        sys.exit(1)
    game_url = sys.argv[1]
    fetch_reviews(game_url)