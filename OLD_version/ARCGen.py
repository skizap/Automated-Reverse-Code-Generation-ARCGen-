#!/usr/bin/env python3
"""
DeepSeek Chunk-by-Chunk GMod Addon Rewriter

This script scans a Garry's Mod addon folder, immediately rewrites each text-based file
in smaller chunks, and saves the rewritten results to a new directory. Non-text files
are copied as-is.

Key Features:
1. Splits each text-based file into smaller chunks (configurable chunk size).
2. For each chunk, calls DeepSeek ("deepseek-coder" model) to rewrite/optimize code
   in smaller pieces, minimizing risk of exceeding token limits.
3. Concatenates chunk rewrites into a final file, preserving overall file structure.
4. Copies all binary/unreadable files (e.g. models, textures) directly to the output folder.
5. Includes concurrency (via ThreadPoolExecutor) to handle multiple files in parallel.
6  Its not perfet this is only V1
Usage:
  python deepseek.py <addon_path> --output-path <output_dir> [--verbose]

Example:
  python ARCGen.py "C:/some/addon" ^
    --output-path "C:/some/rewritten_addon" ^
    --verbose
"""

import os
import sys
import json
import argparse
import requests
import shutil
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

###############################################################################
#                              CONFIGURATION                                  #
###############################################################################

DEEPSEEK_API_URL = "https://api.deepseek.com"
DEEPSEEK_CHAT_COMPLETIONS_ENDPOINT = "/chat/completions"
DEEPSEEK_MODEL_ID = "deepseek-coder"
DEEPSEEK_API_KEY = "API-KEY-HERE"  # <--- Replace with your actual key

# Max tokens per request (DeepSeek can often do up to 8k tokens).
MAX_TOKENS = 8000

# Chunk size in characters. Increase/decrease based on your usage. If you still
# risk hitting token limits, consider smaller chunks or token-based chunking.
CHUNK_SIZE = 3000

# Number of retries for transient errors or rate limits.
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2

###############################################################################
#                          DEEPSEEK API CALL FUNCTION                         #
###############################################################################

def deepseek_call(prompt: str, max_tokens: int = MAX_TOKENS, retry_count: int = 0) -> str:
    """
    Makes a request to DeepSeek's 'deepseek-coder' model to rewrite or summarize
    code in smaller chunks. Automatically retries on transient failures or rate limits.

    :param prompt: The text prompt to send to DeepSeek.
    :param max_tokens: Upper limit on tokens in the response.
    :param retry_count: For internal retry tracking.
    :return: The response text from the model (string).
    """
    url = f"{DEEPSEEK_API_URL}{DEEPSEEK_CHAT_COMPLETIONS_ENDPOINT}"
    payload = {
        "model": DEEPSEEK_MODEL_ID,
        "messages": [
            {"role": "system", "content": "You are a helpful code rewriting assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 1
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        # Typically we expect: {"choices": [ { "message": { "content": "..."} } ]}
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        else:
            return "[DeepSeek Error] API returned no content."
    except requests.exceptions.RequestException as e:
        # Retry with exponential backoff
        if retry_count < MAX_RETRIES:
            sleep_time = (RETRY_BACKOFF_BASE ** retry_count)
            time.sleep(sleep_time)
            return deepseek_call(prompt, max_tokens, retry_count=retry_count + 1)
        else:
            return f"[DeepSeek Error] Request failed after {MAX_RETRIES} retries: {e}"

###############################################################################
#                              CHUNKING HELPERS                               #
###############################################################################

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list:
    """
    Splits `text` into a list of smaller chunks, each up to `chunk_size` characters.
    """
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

###############################################################################
#                         FILE REWRITING WORKFLOW                             #
###############################################################################

def rewrite_file_in_chunks(file_path: Path, verbose: bool = False) -> str:
    """
    Reads a text-based file, breaks it into chunks, rewrites each chunk individually
    via the DeepSeek API, and then concatenates them into a single final string.

    :param file_path: Path to the original file on disk.
    :param verbose: If True, prints debug information.
    :return: The final rewritten file content (str).
    """
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            full_text = f.read()
    except (UnicodeDecodeError, OSError) as exc:
        if verbose:
            print(f"[WARN] Could not read file {file_path}: {exc}")
        return ""

    if not full_text.strip():
        if verbose:
            print(f"[INFO] Skipping empty file: {file_path}")
        return ""

    # Split the file text into smaller chunks.
    file_chunks = chunk_text(full_text, chunk_size=CHUNK_SIZE)
    final_rewritten = []

    for idx, chunk_data in enumerate(file_chunks):
        # Build a rewriting prompt
        rewrite_prompt = (
            "Rewrite and optimize this Garry's Mod code chunk. Preserve functionality, "
            "improve clarity, and ensure it still runs as intended:\n\n"
            f"{chunk_data}\n\n"
            "Return ONLY the updated code for this chunk."
        )

        chunk_rewritten = deepseek_call(rewrite_prompt)
        final_rewritten.append(chunk_rewritten)

        if verbose:
            print(f"[DEBUG] Rewrote chunk {idx+1}/{len(file_chunks)} for {file_path}")

    # Combine all rewritten chunks
    return "\n".join(final_rewritten)


def process_single_file(
    source_file: Path,
    addon_root: Path,
    output_root: Path,
    verbose: bool
) -> dict:
    """
    Processes a single file. If it's text-based (Lua, etc.), rewrites it in chunks.
    Otherwise, copies the file directly.

    Returns a dict with metadata about how the file was handled, used for logging/analysis.
    """
    relative_path = source_file.relative_to(addon_root)
    target_path = output_root.joinpath(relative_path)
    result = {
        "file": str(source_file),
        "relative": str(relative_path),
        "rewritten": False,
        "copied": False,
        "error": None
    }

    # Ensure target subdirectory exists
    if not target_path.parent.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)

    # Attempt rewriting as text
    try:
        # Check if it's a known text-based extension (e.g. .lua, .txt, .cfg, .json, etc.)
        # or do a quick sniff of the file content. For simplicity, let's just
        # check extension. You can expand this logic as needed.
        text_extensions = {".lua", ".txt", ".cfg", ".json", ".vmt", ".vmf"}
        if source_file.suffix.lower() in text_extensions:
            # Rewriting in chunks
            rewritten_code = rewrite_file_in_chunks(source_file, verbose=verbose)
            if rewritten_code.strip():
                with target_path.open("w", encoding="utf-8", errors="ignore") as nf:
                    nf.write(rewritten_code)
                result["rewritten"] = True
            else:
                # If rewriting returned empty, let's just copy
                shutil.copy2(source_file, target_path)
                result["copied"] = True

        else:
            # By default, treat as non-text, so just copy
            shutil.copy2(source_file, target_path)
            result["copied"] = True

    except Exception as e:
        # If anything failed, record error, attempt direct copy as fallback.
        result["error"] = str(e)
        if verbose:
            print(f"[ERROR] {source_file} rewriting error: {e}. Attempting direct copy.")
        try:
            shutil.copy2(source_file, target_path)
            result["copied"] = True
        except Exception as copy_exc:
            result["error"] += f" [Copy failed: {copy_exc}]"

    return result

###############################################################################
#                             MAIN SCRIPT LOGIC                               #
###############################################################################

def run_analysis(
    addon_path: Path,
    output_path: Path,
    verbose: bool = False
):
    """
    Main function that scans the addon_path, processes each file in parallel,
    and logs rewriting results. Produces JSON & optional Markdown for analysis.
    """
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # Collect all files
    all_files = list(addon_path.rglob("*"))
    target_files = [f for f in all_files if f.is_file()]

    if verbose:
        print(f"[INFO] Found {len(target_files)} files in {addon_path}")

    # Process files in parallel
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {
            executor.submit(process_single_file, f, addon_path, output_path, verbose): f
            for f in target_files
        }
        for future in as_completed(future_map):
            outcome = future.result()
            results.append(outcome)

    # Build a dictionary to store results
    analysis_data = {
        "addon_name": addon_path.name,
        "addon_path": str(addon_path),
        "output_path": str(output_path),
        "files_processed": len(results),
        "results": results
    }

    # Save a JSON summary (you can also produce a Markdown if desired)
    analysis_json = addon_path / "addon_analysis.json"
    try:
        with analysis_json.open("w", encoding="utf-8") as f:
            json.dump(analysis_data, f, indent=2)
        if verbose:
            print(f"[INFO] Wrote analysis JSON to {analysis_json}")
    except Exception as e:
        if verbose:
            print(f"[ERROR] Failed to write JSON analysis: {e}")

    if verbose:
        print("[INFO] Chunk-by-chunk rewriting complete!")

def main():
    parser = argparse.ArgumentParser(
        description="Chunk-by-chunk rewriting of a GMod addon with DeepSeek v3 (deepseek-coder)."
    )
    parser.add_argument(
        "addon_path",
        help="Path to the GMod addon directory, e.g. homigrad."
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Where to store the newly rewritten addon (defaults to <addon_path>_remade)."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging."
    )
    args = parser.parse_args()

    addon_path = Path(args.addon_path).resolve()
    if not addon_path.is_dir():
        print(f"[FATAL] Provided addon path is not a directory: {addon_path}")
        sys.exit(1)

    # If no output path is specified, default to "<addon_name>_remade"
    if args.output_path:
        output_path = Path(args.output_path).resolve()
    else:
        output_path = addon_path.parent.joinpath(addon_path.name + "_remade")

    if args.verbose:
        print(f"[INFO] Addon: {addon_path}")
        print(f"[INFO] Output: {output_path}")

    run_analysis(addon_path, output_path, verbose=args.verbose)

if __name__ == "__main__":
    main()
