#!/usr/bin/env python3
"""
Convert LLM API request logs (CSV) to genai-bench compatible dataset format.

Usage:
    python scripts/convert_logs_to_dataset.py <input_csv> <output_dir> [--format txt|csv|json]

Example:
    python scripts/convert_logs_to_dataset.py /path/to/logs.csv /path/to/output --format csv

Output formats:
    - txt: One prompt per line (simple text file)
    - csv: CSV with 'prompt' column (for --dataset-path with --prompt-column)
    - json: JSON array of prompts (for --dataset-path)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


def parse_request_body(raw: str) -> dict | None:
    """Parse the double-escaped JSON from CSV export."""
    if pd.isna(raw):
        return None
    try:
        # First, handle the CSV escaping: \" -> "
        fixed = raw.replace('\\"', '"')
        # Handle nested escaped quotes in prompt content: \\" -> \"
        fixed = fixed.replace('\\\\"', '\\"')
        return json.loads(fixed)
    except Exception:
        return None


def convert_logs_to_dataset(
    input_csv: str,
    output_dir: str,
    output_format: str = "csv",
    deduplicate: bool = True,
) -> dict:
    """Convert LLM API logs to genai-bench dataset format.

    Args:
        input_csv: Path to input CSV file
        output_dir: Directory to save output files
        output_format: Output format (txt, csv, or json)
        deduplicate: Whether to remove duplicate prompts

    Returns:
        dict with conversion statistics
    """
    print(f"Loading CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Total rows: {len(df)}")

    # Parse request bodies and extract prompts
    prompts = []
    metadata = []
    parse_errors = 0

    for i, row in df.iterrows():
        body = parse_request_body(row.get("request_body", ""))
        if body and body.get("prompt"):
            prompts.append(body["prompt"])
            metadata.append({
                "index": i,
                "model": body.get("model"),
                "max_tokens": body.get("max_tokens"),
                "temperature": body.get("temperature"),
                "e2e_latency_ms": row.get("e2e_latency_ms"),
                "ttft_ms": row.get("ttft_ms"),
                "tpot_us": row.get("tpot_us"),
            })
        else:
            parse_errors += 1

    print(f"Parsed: {len(prompts)} prompts ({parse_errors} errors)")

    # Deduplicate if requested
    original_count = len(prompts)
    if deduplicate:
        seen = set()
        unique_prompts = []
        unique_metadata = []
        for p, m in zip(prompts, metadata):
            if p not in seen:
                seen.add(p)
                unique_prompts.append(p)
                unique_metadata.append(m)
        prompts = unique_prompts
        metadata = unique_metadata
        print(f"After deduplication: {len(prompts)} unique prompts")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Calculate prompt length statistics
    prompt_lengths = [len(p) for p in prompts]
    stats = {
        "total_rows": len(df),
        "parsed_prompts": original_count,
        "unique_prompts": len(prompts),
        "parse_errors": parse_errors,
        "prompt_length_min": min(prompt_lengths) if prompt_lengths else 0,
        "prompt_length_max": max(prompt_lengths) if prompt_lengths else 0,
        "prompt_length_avg": sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0,
    }

    # Save in requested format
    if output_format == "txt":
        output_file = output_path / "prompts.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for prompt in prompts:
                # Replace newlines with space for single-line format
                clean_prompt = prompt.replace("\n", " ").replace("\r", "")
                f.write(clean_prompt + "\n")
        print(f"Saved: {output_file}")

    elif output_format == "csv":
        output_file = output_path / "prompts.csv"
        prompt_df = pd.DataFrame({"prompt": prompts})
        prompt_df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

        # Also save metadata
        meta_file = output_path / "metadata.csv"
        meta_df = pd.DataFrame(metadata)
        meta_df.to_csv(meta_file, index=False)
        print(f"Saved: {meta_file}")

    elif output_format == "json":
        output_file = output_path / "prompts.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)
        print(f"Saved: {output_file}")

        # Also save full data with metadata
        full_file = output_path / "dataset.json"
        full_data = [{"prompt": p, **m} for p, m in zip(prompts, metadata)]
        with open(full_file, "w", encoding="utf-8") as f:
            json.dump(full_data, f, ensure_ascii=False, indent=2)
        print(f"Saved: {full_file}")

    # Save statistics
    stats_file = output_path / "stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {stats_file}")

    # Generate genai-bench dataset config file
    config = {
        "source": {
            "type": "file",
            "path": str(output_file),
            "file_format": output_format,
        },
        "prompt_column": "prompt" if output_format == "csv" else None,
    }
    config_file = output_path / "dataset_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved: {config_file}")

    print(f"\n=== Conversion Complete ===")
    print(f"Output directory: {output_path}")
    print(f"Prompts: {len(prompts)}")
    print(f"Avg length: {stats['prompt_length_avg']:.0f} chars")

    print(f"\n=== Usage with genai-bench ===")
    if output_format == "csv":
        print(f"  genai-bench benchmark --dataset-path {output_file} --prompt-column prompt ...")
    elif output_format == "txt":
        print(f"  genai-bench benchmark --dataset-path {output_file} ...")
    else:  # json
        print(f"  genai-bench benchmark --dataset-config {config_file} ...")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert LLM API logs to genai-bench dataset format"
    )
    parser.add_argument("input_csv", help="Input CSV file path")
    parser.add_argument("output_dir", help="Output directory path")
    parser.add_argument(
        "--format",
        choices=["txt", "csv", "json"],
        default="csv",
        help="Output format (default: csv)",
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Keep duplicate prompts",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"Error: Input file not found: {args.input_csv}")
        sys.exit(1)

    stats = convert_logs_to_dataset(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        output_format=args.format,
        deduplicate=not args.no_deduplicate,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
