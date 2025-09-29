import pandas as pd
import openai
import tiktoken
import json
import os
from datetime import datetime
from pathlib import Path
from jsonschema import validate, ValidationError
import re
import math
import time
import csv
import sys
import chatgpt_helpers as ch
from openai import OpenAI
from typing import List, Any, Dict, Tuple

# === CONFIGURATION ===

# Accept game_url from command line
if len(sys.argv) < 2:
    print("Usage: python analyze_reviews.py <game_url>")
    sys.exit(1)

game_url = sys.argv[1]

# Extract app id
match = re.search(r"/app/(\d+)", game_url)
if not match:
    raise ValueError(f"No TapTap app id found in URL: {game_url}")
app_id = match.group(1)

print("TapTap App ID:", app_id)

# Get the path to the review csv file
review_csv_path = "output/{}/reviews_{}.csv".format(app_id, app_id)


# === HELPER FUNCTIONS ===

def ask_yes_no(question: str, default: str = "n") -> bool:
    """Prompt the user for y/n. Returns True for yes."""
    default = default.lower()
    prompt = " [Y/n]: " if default == "y" else " [y/N]: "
    while True:
        ans = input(question + prompt).strip().lower()
        if not ans:
            return default == "y"
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please answer y or n.")

# === EXECUTION ===

# Load review data
df = pd.read_csv(review_csv_path)
reviews = df[['review_id', 'review_content_raw_text']].dropna().to_dict(orient="records")

# Prepare prompt for analyzing the reviews

with open("input/prompt_prefix_1.txt", "r", encoding="utf-8") as f:
    PROMPT_PREFIX_1 = f.read().strip()

with open("input/prompt_prefix_2.txt", "r", encoding="utf-8") as f:
    PROMPT_PREFIX_2 = f.read().strip()

PROMPT_PREFIX = PROMPT_PREFIX_1 + " " + game_url + "\n\n" + PROMPT_PREFIX_2

# Break the tokens into batches
print(f"\nBreaking the reviews into batches to meet token limit per call")
batches = ch.batch_reviews(reviews)
print(f"Number of batches: {len(batches)}")
# print(f"First batch sneekpeak: {batches[0]}")

# --- Cost estimation ---
try:
    encoding = tiktoken.encoding_for_model(ch.MODEL)
except KeyError:
    encoding = tiktoken.get_encoding("cl100k_base")

base_token_count = ch.count_tokens(PROMPT_PREFIX)
total_review_tokens = sum(ch.count_tokens(str(r['review_content_raw_text'])) for r in reviews)
# we send the prefix once per batch
total_tokens_input = base_token_count * len(batches) + total_review_tokens

print(f"\n📊 Review analysis: Estimated token usage:")
print(f"- Prompt prefix (per batch): {base_token_count} tokens")
print(f"- Total reviews: {len(reviews)}")
print(f"- Total review content: {total_review_tokens} tokens")
print(f"- Estimated total input tokens: {total_tokens_input}")

model_costs = ch.MODEL_COSTS.get(ch.MODEL)
if model_costs:
    # Rough output estimate:
    #   - ~CHUNK_OUTPUT_MAX per batch (chunk summaries)
    #   - +REDUCE_OUTPUT_MAX once (merge)
    #   - +~1200 for the final general summary
    approx_output_per_batch = ch.CHUNK_OUTPUT_MAX
    approx_reduce = ch.REDUCE_OUTPUT_MAX
    approx_general_summary = 1200

    est_output_tokens = approx_output_per_batch * len(batches) + approx_reduce + approx_general_summary

    cost_in  = total_tokens_input  * model_costs["input"]
    cost_out = est_output_tokens    * model_costs["output"]
    cost_tot = cost_in + cost_out

    print(f"- Estimated output tokens (rough): {est_output_tokens}")
    print(f"- Approx cost IN:  ${cost_in:.4f} ({ch.MODEL})")
    print(f"- Approx cost OUT: ${cost_out:.4f} ({ch.MODEL})")
    print(f"- ≈ Total cost:    ${cost_tot:.4f}\n")
else:
    print(f"\n⚠️ No pricing info found for model {ch.MODEL}")
    cost_tot = None

# --- Ask for confirmation before any AI calls ---
if not ask_yes_no("Proceed with AI-powered analysis?", default="n"):
    print("Aborted before calling OpenAI. ✅")
    sys.exit(0)
    
# Calling AI API to analyze the reviews batch by batch
print(f"\nAnalyzing the reviews, batch by batch...")

for i in range(0, len(batches)):

    batch = batches[i]
    
    print(f"\n🚀 Sending batch {i+1}/{len(batches)}...")
    
    # Build prompt
    batch_prompt = "\n\nINPUT REVIEWS TO ANALYZE (NOTE THAT THERE COULD BE MULTIPLE REVIEWS)\n" + json.dumps(batch, ensure_ascii=False, indent=2)

    # Call API
    try:
        response = ch.client.responses.create(
            model=ch.MODEL,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": PROMPT_PREFIX}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": batch_prompt}],
                },
            ],
            # temperature=0.3,
            # max_output_tokens=ch.CHUNK_OUTPUT_MAX,  # optional cap
        )
        content = response.output_text.strip()
    except Exception as e:
        print(f"❌ OpenAI API error in batch {i+1}: {e}")
        continue

    # Save raw content for backup/debug
    out_dir = Path(f"output/{app_id}/review_analysis_output")
    out_dir.mkdir(parents=True, exist_ok=True)    
    batch_output_path = out_dir / f"batch_{i+1:03d}_raw.txt"
    with open(batch_output_path, "w", encoding="utf-8") as f:
        f.write(content)

print(f"Done with pulling all analysis batches!")

# Load all the batches
print(f"\nCompiling the reviews...")
out_dir = Path(f"output/{app_id}/review_analysis_output")
out_dir.mkdir(parents=True, exist_ok=True) 
review_analysis_result = ch.load_all_batches_from_folder(out_dir)
df_review_analysis_results_all = pd.DataFrame(review_analysis_result)

# Ensure every question column q0..q41 exists, even if some are missing in model outputs
column_order = [f"q{i}" for i in range(0, ch.ANALYSIS_QUESTION_COUNT + 1)]
missing = [c for c in column_order if c not in df_review_analysis_results_all.columns]
for c in missing:
    df_review_analysis_results_all[c] = None

# Reorder columns consistently
df_review_analysis_results_all = df_review_analysis_results_all[column_order]
print("Review analysis compiling done! (added missing columns:", missing, ")")

# Export detailed review analysis dataframe to csv
print("Export detailed review analysis dataframe to csv...")
out_dir = Path(f"output/{app_id}/report")
out_dir.mkdir(parents=True, exist_ok=True)    
detailed_review_analysis_output_path = out_dir / f"detailed_review_analysis_{app_id}.csv"
df_review_analysis_results_all.to_csv(detailed_review_analysis_output_path, index=False)
print("Detailed review analysis exported successfully to csv!")

# Generate analysis_summary report
print("Generating analysis summary...")
df_review_analysis_results_for_synthesis = pd.read_csv(detailed_review_analysis_output_path)
df_reviews = pd.DataFrame(reviews)
df_review_analysis_results_for_synthesis = pd.merge(
    df_review_analysis_results_for_synthesis,
    df_reviews,
    left_on="q0",
    right_on="review_id"
)

synthesis_prompts = dict()

for q, p in ch.synthesis_prompts_core_parts.items():
    current_full_prompt = p + "\n \n" + ch.rules
    synthesis_prompts[q] = current_full_prompt


synthesized_answers = ch.synthesize_answers(
    df_review_analysis_results_for_synthesis,
    "review_id",
    "review_content_raw_text",
    ch.question_types,
    synthesis_prompts
)

synthesized_answers_with_question_content = []

for question_code, answer_content in synthesized_answers.items():
    current_data_piece = {
        'question_code':question_code,
        'question_content': ch.question_content[question_code],
        'synthesis_answer': answer_content
    }
    synthesized_answers_with_question_content.append(current_data_piece)

out_dir = Path(f"output/{app_id}/report")
out_dir.mkdir(parents=True, exist_ok=True)

analysis_summary_json_path = out_dir / f"analysis_summary_{app_id}.json"
analysis_summary_csv_path = out_dir / f"analysis_summary_{app_id}.csv"

with open(analysis_summary_json_path, "w", encoding="utf-8") as f:
    json.dump(synthesized_answers_with_question_content, f, ensure_ascii=False, indent=4)
print("Analysis summary exported successfully to json!")

df_synthesized_answers = pd.DataFrame(synthesized_answers_with_question_content)
df_synthesized_answers.to_csv(analysis_summary_csv_path, index=False, encoding="utf-8")
print("Analysis summary exported successfully to csv!")

# Generate general_summary report
print("Generating general summary report...")

with open(analysis_summary_json_path, "r", encoding="utf-8") as f:
    synthesized_answers_with_question_content = json.load(f)

context_string = ""

for qna in synthesized_answers_with_question_content:
    context_string = context_string + "\n\n" + qna["question_code"] + ": " + qna["question_content"] + "\n" + str(qna["synthesis_answer"])

general_summary = ch.get_commentary_from_text_detailed_to_general_summary(game_url, context_string)

out_dir = Path(f"output/{app_id}/report")
out_dir.mkdir(parents=True, exist_ok=True)
general_summary_csv_path = out_dir / f"general_summary_{app_id}.csv"

with open(general_summary_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["game_link", "commentary"])  # header
    writer.writerow([game_url, general_summary])

print("General summary exported successfully to csv!")

# Compile everything into a final xlsx report

print("Compiling everything into a final report...")

csv_files = [
(general_summary_csv_path, "general_summary"),
(analysis_summary_csv_path, "analysis_summary"),
(detailed_review_analysis_output_path, "detailed_review_analysis"),
(review_csv_path, "reviews"),
]

out_dir = Path(f"output/{app_id}/report")
out_dir.mkdir(parents=True, exist_ok=True)

final_report_path = out_dir / f"final_report_{app_id}.xlsx"

with pd.ExcelWriter(final_report_path, engine="xlsxwriter") as writer:
    for path, sheet_name in csv_files:
        df_excel = pd.read_csv(path)
        df_excel.to_excel(writer, sheet_name=sheet_name[:31], index=False)

print("Final report exported successfully to xlsx!")
