import pandas as pd
import tiktoken
import json
import os
import re
import time
from pathlib import Path
from typing import List, Any, Dict, Tuple
from openai import OpenAI

# === CONFIG ===
MODEL = "gpt-5"                 # default model
MODEL_CONTEXT = 128_000
OUTPUT_BUFFER = 2_500
SAFETY_MARGIN = 1_000
CHUNK_OUTPUT_MAX = 2500
REDUCE_OUTPUT_MAX = 1_200
TEMP = 0.2
RETRY = 3
RETRY_SLEEP = 2.0
MAX_TOKENS_PER_BATCH = 120_000
OUTPUT_TOKEN_BUFFER_PER_REVIEW = 1_200

MODEL_DETAILED_TO_GENERAL_SUMMARY = "gpt-5"
SYSTEM_INSTRUCTIONS_DETAILED_TO_GENERAL_SUMMARY = (
    "You are a senior game analyst. Write concise, English-only commentary."
)

MODEL_COSTS = {
    # GPT-5 family
    "gpt-5":       {"input": 0.00000125, "output": 0.00001000},
    "gpt-5-mini":  {"input": 0.00000025, "output": 0.00000200},
    "gpt-5-nano":  {"input": 0.00000005, "output": 0.00000040},

    # GPT-4.1 family
    "gpt-4.1":       {"input": 0.00000200, "output": 0.00000800},
    "gpt-4.1-mini":  {"input": 0.00000040, "output": 0.00000160},
    "gpt-4.1-nano":  {"input": 0.00000010, "output": 0.00000040},

    # GPT-4o family
    "gpt-4o":       {"input": 0.00000250, "output": 0.00001000},
    "gpt-4o-mini":  {"input": 0.00000015, "output": 0.00000060},

    # o-series
    "o4-mini": {"input": 0.00000110, "output": 0.00000440},
    "o3-mini": {"input": 0.00000110, "output": 0.00000440},
    "o3":      {"input": 0.00000200, "output": 0.00000800},
}

# Analysis question config

ANALYSIS_QUESTION_COUNT = 41

question_content = {
    "q1": "Does the game contain any sensitive content (Sexual Content/Nudity, Violence/Gore, Drugs/Alcohol/Tobacco, Religious/Political)? (If not mentioned, answer “N/A”)",
    "q2": "List the sensitive content mentioned, if any.",
    "q3": "Summarize the core combat mechanics (or “N/A” if not mentioned).",
    "q4": "Rate combat satisfaction: -2 (very negative) to 2 (very positive) or “N/A”",
    "q5": "Reason for previous question’s rating",
    "q6": "List the strategic/tactical features mentioned, if any.",
    "q7": "Rate strategic/tactical depth: -2 to 2 or “N/A”",
    "q8": "Reason for previous question’s rating",
    "q9": "Rate game progression: -2 to 2 or “N/A”",
    "q10": "Reason for previous question’s rating",
    "q11": "Rate hero balance: -2 to 2 or “N/A”",
    "q12": "Reason for previous question’s rating",
    "q13": "Rate hero/team build diversity: -2 to 2 or “N/A”",
    "q14": "Reason for previous question’s rating",
    "q15": "Describe the secondary core loop (or “N/A”)",
    "q16": "Does the secondary loop match tycoon/crafting (e.g. Township, Hay Day)? (True/False/“N/A”)",
    "q17": "Rate simplicity of the secondary loop: -2 to 2 or “N/A”",
    "q18": "Reason for previous question’s rating",
    "q19": "Rate resources earned from the secondary loop: -2 to 2 or “N/A”",
    "q20": "Reason for previous question’s rating",
    "q21": "Rate frequency of content/meta updates: -2 to 2 or “N/A”",
    "q22": "Reason for previous question’s rating",
    "q23": "Does the game have a gacha guarantee system? (True/False/“N/A”)",
    "q24": "Briefly describe the gacha guarantee system if answer to previous question is True, else answer “N/A”",
    "q25": "Rate reasonableness of gacha pull price: -2 to 2 or “N/A”",
    "q26": "Reason for previous question’s rating",
    "q27": "Rate major feature access for non-paying users: -2 to 2 or “N/A”",
    "q28": "Reason for previous question’s rating",
    "q29": "Rate spending pressure: -2 (heavy/annoying) to 2 (elegant/subtle), or “N/A”",
    "q30": "Reason for previous question’s rating",
    "q31": "Rate quality/quantity of free rewards: -2 to 2 or “N/A”",
    "q32": "Reason for previous question’s rating",
    "q33": "Does the game integrate IP? (True/False if says none/“N/A” if not mentioned). Briefly describe.",
    "q34": "Briefly describe the IP if answer to previous question is True, else answer “N/A”",
    "q35": "Rate IP integration depth: -2 to 2 or “N/A”",
    "q36": "Reason for previous question’s rating",
    "q37": "Rate lightness of installation file: -2 to 2 or “N/A”",
    "q38": "Reason for previous question’s rating",
    "q39": "Rate in-game download experience: -2 to 2 or “N/A”",
    "q40": "Reason for previous question’s rating",
    "q41": "Briefly list other issues (not captured by the questions above)",
}

question_types = {
    "q1": "true_false",
    "q2": "long_form",
    "q3": "long_form",
    "q4": "integer_rating",
    "q5": "long_form",
    "q6": "long_form",
    "q7": "integer_rating",
    "q8": "long_form",
    "q9": "integer_rating",
    "q10": "long_form",
    "q11": "integer_rating",
    "q12": "long_form",
    "q13": "integer_rating",
    "q14": "long_form",
    "q15": "long_form",
    "q16": "true_false",
    "q17": "integer_rating",
    "q18": "long_form",
    "q19": "integer_rating",
    "q20": "long_form",
    "q21": "integer_rating",
    "q22": "long_form",
    "q23": "true_false",
    "q24": "long_form",
    "q25": "integer_rating",
    "q26": "long_form",
    "q27": "integer_rating",
    "q28": "long_form",
    "q29": "integer_rating",
    "q30": "long_form",
    "q31": "integer_rating",
    "q32": "long_form",
    "q33": "true_false",
    "q34": "long_form",
    "q35": "integer_rating",
    "q36": "long_form",
    "q37": "integer_rating",
    "q38": "long_form",
    "q39": "integer_rating",
    "q40": "long_form",
    "q41": "long_form",
}

# Analysis Summary Config

rules = """Rules:
- Output bulleted points only (no paragraphs, no JSON).
- Group points into short, factual insights (≤20 words each).
- Each bullet must end with review IDs in square brackets, e.g. [37988997, 35631039].
- Include ≤1 short quote (≤12 words) if useful for clarity.
- Output must strictly be in English
- If INPUT is empty, output: • No reviews matched.
"""

synthesis_prompts_core_parts = {
    "q2":"""You are an analyst. ONLY use the reviews in INPUT. Summarize what players say about sensitive content.
Sensitive categories: Sexual Content/Nudity, Violence/Gore, Drugs/Alcohol/Tobacco, Religious/Political.""",
    "q3":"""You are an analyst. ONLY use the reviews provided in INPUT.
Summarize what players say about the game's core combat mechanics.
Definition: Core combat mechanics = primary battle systems, controls, pacing, balance, skill systems, roles/classes, resource usage in combat, and related player strategy.""",
    "q5":"""You are an analyst. ONLY use the reviews provided in INPUT.
Summarize the players' opinion on combat satisfaction rating.""",
    "q6":"""You are an analyst. ONLY use the reviews provided in INPUT.
List the strategic/tactical features mentioned by players.""",
    "q8":"""You are an analyst. ONLY use the reviews provided in INPUT.
Summarize the players' opinion on strategic/tactical depth.""",
    "q10":"""You are an analyst. ONLY use the reviews provided in INPUT.
Summarize the players' opinion on progression system.""",
    "q12":"""You are an analyst. ONLY use the reviews provided in INPUT.
Summarize the players' opinion on hero balance.""",
    "q14":"""You are an analyst. ONLY use the reviews provided in INPUT.
Summarize the players' opinion on hero/team build diversity.""",
    "q15":"""You are an analyst. ONLY use the reviews provided in INPUT.
Describe the secondary core loop of the game as mentioned by players.""",
    "q18":"""You are an analyst. ONLY use the reviews provided in INPUT.
Summarize the players' opinion on the simplicity rating of the secondary loop.""",
    "q20":"""You are an analyst. ONLY use the reviews provided in INPUT.
Summarize the players' opinion on the resources earned from the secondary loop.""",
    "q22":"""You are an analyst. ONLY use the reviews provided in INPUT.
Summarize the players' opinion on the frequency of content/meta updates.""",
    "q24":"""You are an analyst. ONLY use the reviews provided in INPUT.
Briefly describe the gacha guarantee system as mentioned by players.""",
    "q26":"""You are an analyst. ONLY use the reviews provided in INPUT.
Summarize the players' opinion on the gacha pull price reasonableness.""",
    "q28":"""You are an analyst. ONLY use the reviews provided in INPUT.
Summarize the players' opinion on major feature access for non-paying users.""",
    "q30":"""You are an analyst. ONLY use the reviews provided in INPUT.
Summarize the players' opinion on spending pressure.""",
    "q32":"""You are an analyst. ONLY use the reviews provided in INPUT.
Summarize the players' opinion on free rewards quality/quantity.""",
    "q34":"""You are an analyst. ONLY use the reviews provided in INPUT.
Briefly describe the IP integrated into the game as mentioned by players.""",
    "q36":"""You are an analyst. ONLY use the reviews provided in INPUT.
Summarize the players' opinion on IP integration depth.""",
    "q38":"""You are an analyst. ONLY use the reviews provided in INPUT.
Summarize the players' opinion on lightness of installation file.""",
    "q40":"""You are an analyst. ONLY use the reviews provided in INPUT.
Summarize the players' opinion on in-game download experience.""",
    "q41":"""You are an analyst. ONLY use the reviews provided in INPUT.
List issues mentioned by the players other than the following issues: sensitive content, core combat mechanics, strategic/tactical features, game progression, hero balance, hero/team build diversity, secondary game loop, frequency of content/meta updates, gacha guarantee system, gacha pull price, feature access for non-paying users, quality/quantity of free rewards, IP integration, lightness of installation file, in-game download experience"""
}

# === OpenAI client ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === SUPPORTING FUNCTIONS ===

def extract_fields_from_list_of_dicts(list_of_dicts, fields_to_extract):
    return [
        {field: dicti.get(field, None) for field in fields_to_extract}
        for dicti in list_of_dicts
    ]

def filter_keys_by_value(d, target_value):
    return [k for k, v in d.items() if v == target_value]

def count_tokens(text: str, model: str = MODEL) -> int:
    """
    Estimate token count for a user message with the given model.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    # Keep compatibility for gpt-4o / gpt-5
    if model in {"gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-5"}:
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"Token counting not supported for model: {model}")

    messages = [{"role": "user", "content": text}]
    total_tokens = 0
    for message in messages:
        total_tokens += tokens_per_message
        for key, value in message.items():
            total_tokens += len(encoding.encode(str(value)))
            if key == "name":
                total_tokens += tokens_per_name

    total_tokens += 3  # Priming reply
    return total_tokens

def summarize_and_count_tokens(reviews, review_id_field_name="review_id"):
    token_summaries = []
    for review in reviews:
        review_id = review.get(review_id_field_name, "unknown")
        review_string = json.dumps(review, ensure_ascii=False)
        token_count = count_tokens(review_string)
        token_summaries.append({"review_id": review_id, "token_count": token_count})
    return token_summaries

def batch_reviews(
    reviews,
    max_tokens=MAX_TOKENS_PER_BATCH,
    output_token_buffer_per_review=OUTPUT_TOKEN_BUFFER_PER_REVIEW
):
    batches = []
    current_batch = []
    current_tokens = 0
    for review in reviews:
        review_text = f"{review}"
        input_tokens = count_tokens(review_text)
        total_estimated_tokens = input_tokens + output_token_buffer_per_review
        if current_tokens + total_estimated_tokens > max_tokens and current_batch:
            batches.append(current_batch)
            current_batch = [review]
            current_tokens = total_estimated_tokens
        else:
            current_batch.append(review)
            current_tokens += total_estimated_tokens
    if current_batch:
        batches.append(current_batch)
    return batches

def remap_keys(data, key_mapping):
    return [
        {key_mapping.get(k, k): v for k, v in entry.items()}
        for entry in data
    ]

def count_column_values(df, column_name):
    counts = df[column_name].value_counts(dropna=False).to_dict()
    cleaned_counts = {}
    for key, value in counts.items():
        if pd.isna(key):
            cleaned_counts[None] = value
        else:
            cleaned_counts[key] = value
    return cleaned_counts

def calc_weighted_average(df, column_name):
    d = count_column_values(df, column_name)
    total = count = 0
    for key, value in d.items():
        if key is not None:
            total += key * value
            count += value
    return total / count if count > 0 else None

def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def _chunk_items_by_tokens(
    items: List[Dict],
    prompt: str,
    model_context: int = MODEL_CONTEXT,
    output_buffer: int = OUTPUT_BUFFER,
    safety_margin: int = SAFETY_MARGIN,
) -> List[List[Dict]]:
    """
    Pack items into chunks so that:
      tokens(system+user prompt + items JSON) + safety_margin + output_buffer <= model_context
    """
    # Token budget for the *input side*
    input_budget = model_context - output_buffer - safety_margin
    if input_budget <= 0:
        raise ValueError("Token budget is non-positive. Lower OUTPUT_BUFFER/SAFETY_MARGIN or increase context.")

    # Base tokens from messages without items
    base_user_prefix = "INPUT:\n"  # matches your format
    base_tokens = count_tokens(prompt) + count_tokens(base_user_prefix)  # system + "INPUT:\n"

    batches: List[List[Dict]] = []
    current: List[Dict] = []
    current_tokens = base_tokens + count_tokens('{"items":[]}')  # minimal structure

    for it in items:
        # Estimate tokens if we add this item
        item_json = _json(it)
        # extra 1-2 chars for comma when appending; add small cushion
        add_tokens = count_tokens(item_json) + 2

        if current and (current_tokens + add_tokens) > input_budget:
            batches.append(current)
            current = [it]
            current_tokens = base_tokens + count_tokens('{"items":[' + item_json + "]}") + 2
        else:
            current.append(it)
            current_tokens += add_tokens

    if current:
        batches.append(current)

    return batches

def _call_response(model: str, system_prompt: str, user_payload: str, max_tokens: int) -> str:
    """
    Call OpenAI Responses API with retries.
    """
    for attempt in range(1, RETRY + 1):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                    {"role": "user", "content": [{"type": "input_text", "text": user_payload}]},
                ],
                # temperature=TEMP,
                max_output_tokens=max_tokens,
            )
            return resp.output_text
        except Exception as e:
            if attempt == RETRY:
                raise
            time.sleep(RETRY_SLEEP * attempt)

def load_all_batches_from_folder(folder_path, file_prefix="batch_", file_suffix="_raw", extension=".txt"):
    """
    Loads and parses all .txt batch files from a folder into a single list,
    handling markdown ```json fences and trailing ``` if present.

    Args:
        folder_path (str): Path to the folder containing the batch files.
        file_prefix (str): Common prefix for batch files (default: "batch_").
        file_suffix (str): Suffix after batch number (default: "_raw").
        extension (str): File extension (default: ".txt").

    Returns:
        list[dict]: Combined list of all parsed review entries from all batches.
    """
    all_data = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.startswith(file_prefix) and filename.endswith(file_suffix + extension):
            full_path = os.path.join(folder_path, filename)
            print(f"📂 Reading: {filename}")
            with open(full_path, "r", encoding="utf-8") as f:
                raw = f.read().strip()

                # ✅ Clean markdown-style wrapping if present
                if raw.startswith("```json"):
                    raw = raw[len("```json"):].strip()
                elif raw.startswith("```"):
                    raw = raw[len("```"):].strip()

                if raw.endswith("```"):
                    raw = raw[:-3].strip()
                    
                # Fix TRUE/FALSE casing
                raw = raw.replace("TRUE", "true").replace("FALSE", "false")

                # ✅ Parse JSON safely
                try:
                    data = json.loads(raw)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        print(f"⚠️ Skipped (not a list): {filename}")
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse JSON from {filename}: {e}")
    
    print(f"\n✅ Total reviews loaded: {len(all_data)}")
    return all_data

def synthesize_long_form_answers_with_ai(items: List[Dict], prompt: str) -> str:
    """
    Summarize a potentially large list of items by:
      1) chunking to fit the context window,
      2) summarizing each chunk,
      3) reducing the partial summaries into a final answer.

    Returns the final synthesized text.
    """
    # 1) Chunk
    chunks = _chunk_items_by_tokens(items, prompt)
    print(f"⚙️  Synthesizing in {len(chunks)} chunk(s)...")

    # 2) Map: summarize each chunk
    partial_summaries: List[str] = []
    for idx, chunk in enumerate(chunks, 1):
        user_content = "INPUT:\n" + _json({"items": chunk})
        summary = _call_response(MODEL, prompt, user_content, CHUNK_OUTPUT_MAX)
        partial_summaries.append(summary)
        print(f"   ✅ Chunk {idx}/{len(chunks)} summarized (≈{count_tokens(summary)} tokens).")

    # If only one chunk, we’re done
    if len(partial_summaries) == 1:
        final = partial_summaries[0]
        # Optional: print(final)
        return final

    # 3) Reduce: merge partial summaries
    reduce_instructions = (
        "You will receive several partial summaries produced from different chunks of the same dataset. "
        "Merge them into a single, cohesive answer that follows the **same instructions** as the original system prompt. "
        "Avoid duplication; keep structure consistent; resolve conflicts sensibly. Be concise and complete."
    )
    reduce_user_content = "PARTIAL_SUMMARIES:\n" + _json({"summaries": partial_summaries})

    final_summary = _call_response(
        MODEL,
        system_prompt=f"{prompt}\n\n[Reducer instructions]\n{reduce_instructions}",
        user_payload=reduce_user_content,
        max_tokens=REDUCE_OUTPUT_MAX,
    )
    # Optional: print(final_summary)
    return final_summary

def synthesize_answers(df_all_answers, review_id_column_name, review_content_column_name, question_types, synthesis_prompts):

    synthesized_answers = dict()
    
    for q in question_types.keys():
        print("Synthesizing answers for question: {} - question type: {}".format(q, question_types[q]))
        
        if question_types[q] == "true_false":
            current_synthesized_answer = count_column_values(df_all_answers, q)
            print("Synthesizing Successful!")
            
        elif question_types[q] == "integer_rating":
            current_synthesized_answer = {
                "overall_score": calc_weighted_average(df_all_answers, q),
                "detailed_scoring": count_column_values(df_all_answers, q)
            }
            print("Synthesizing Successful!")
            
        else:
            # synthesizing long-form answers
            
            # prepare the table of review ids and review content for the reviews that touch on the topic of the question
            df_rows_with_nonna_values_for_current_question = df_all_answers[
                df_all_answers[q].notna()
            ][[review_id_column_name, review_content_column_name]]

            relevant_reviews = df_rows_with_nonna_values_for_current_question.to_dict(orient="records")

            # synthesize the answers with AI but using the review content itself rather than the current answers => go directly again to the review content for better reliability
            current_synthesized_answer = synthesize_long_form_answers_with_ai(relevant_reviews, synthesis_prompts[q])
            
            # notification
            print("Synthesizing Successful!")

        synthesized_answers[q] = current_synthesized_answer

    return synthesized_answers

# functions for generating general report from detailed reports

def build_prompt_from_text_detailed_to_general_summary(game_link: str, qa_block_text: str) -> str:
    """
    Builds the exact prompt string you provided, inserting the game link and the
    raw 'THE ANALYSIS ANSWERS:' text block (already formatted with q1..qN).
    """
    return (
        "Below are the answers to questions used to analyze user reviews on "
        f"taptap.cn for this game: {game_link} "
        "Please help to summarize these analysis answers into a brief commentary of the game. "
        "The commentary should entirely be in English and its structure should be:\n"
        "- Overall conclusion\n"
        "- Breakdown:\n"
        "   + Sensitive Content (q1 - q2)\n"
        "   + Gameplay (q3 - q20)\n"
        "   + LiveOps & Retention Systems (q21 - q22)\n"
        "   + Monetization & Game Economy (q23 - q30)\n"
        "   + Anti-P2W Mechanism (q31 - q32)\n"
        "   + IP Integration (q33 - q36)\n"
        "   + System Requirement (q37 - q40)\n"
        "   + Other Issues (q41)\n\n"
        "THE ANALYSIS ANSWERS:\n"
        f"{qa_block_text}".strip()
    )

def get_commentary_from_text_detailed_to_general_summary(game_link: str, qa_block_text: str) -> str:
    """
    Sends one request to the OpenAI Responses API and returns the model's final concatenated text.
    """

    prompt_text = build_prompt_from_text_detailed_to_general_summary(game_link, qa_block_text)

    resp = client.responses.create(
        model=MODEL_DETAILED_TO_GENERAL_SUMMARY,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": SYSTEM_INSTRUCTIONS_DETAILED_TO_GENERAL_SUMMARY }],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt_text}],
            },
        ],
    )

    return resp.output_text  # convenient final text