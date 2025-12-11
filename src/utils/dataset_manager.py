"""
Dataset Manager for Benchmark

Handles downloading, extracting, converting, and caching of datasets
from remote URLs for use with genai-bench.
"""

import gzip
import hashlib
import json
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import pandas as pd
import requests


class DatasetManager:
    """Manages dataset download, extraction, conversion, and caching."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize dataset manager.

        Args:
            cache_dir: Directory for caching datasets. Defaults to
                       ~/.local/share/inference-autotuner/datasets/
        """
        if cache_dir is None:
            data_home = Path.home() / ".local" / "share" / "inference-autotuner"
            cache_dir = data_home / "datasets"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_dataset(
        self,
        url: str,
        deduplicate: bool = True,
        force_download: bool = False,
    ) -> Path:
        """Download and convert dataset, return path to genai-bench format CSV.

        Args:
            url: Remote URL to dataset file (CSV, JSONL, or archive)
            deduplicate: Whether to remove duplicate prompts
            force_download: Force re-download even if cached

        Returns:
            Path to converted prompts.csv file

        Raises:
            ValueError: If dataset format is unsupported or conversion fails
            requests.RequestException: If download fails
        """
        cache_key = self._get_cache_key(url)
        cache_path = self.cache_dir / cache_key
        converted_file = cache_path / "converted" / "prompts.csv"

        # Check cache
        if not force_download and converted_file.exists():
            print(f"[DatasetManager] Using cached dataset: {cache_key[:12]}...")
            return converted_file

        print(f"[DatasetManager] Processing dataset from URL: {url}")

        # Create cache directories
        original_dir = cache_path / "original"
        extracted_dir = cache_path / "extracted"
        converted_dir = cache_path / "converted"
        for d in [original_dir, extracted_dir, converted_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Save cache info
        cache_info = {"url": url, "cache_key": cache_key}
        with open(cache_path / "cache_info.json", "w") as f:
            json.dump(cache_info, f, indent=2)

        # Download file
        downloaded_file = self._download_file(url, original_dir)

        # Extract if archive
        data_file = self._extract_archive(downloaded_file, extracted_dir)

        # Convert to genai-bench format
        output_file = self._convert_to_genai_format(
            data_file, converted_dir, deduplicate=deduplicate
        )

        print(f"[DatasetManager] Dataset ready: {output_file}")
        return output_file

    def _get_cache_key(self, url: str) -> str:
        """Generate cache key from URL using SHA256 hash."""
        return hashlib.sha256(url.encode()).hexdigest()[:16]

    def _download_file(self, url: str, dest_dir: Path) -> Path:
        """Download file from URL or copy local file.

        Args:
            url: Remote URL or local file path (file:// or absolute path)
            dest_dir: Directory to save downloaded file

        Returns:
            Path to downloaded file
        """
        # Parse URL to get filename
        parsed = urlparse(url)

        # Handle local files (file:// URL or absolute path)
        if parsed.scheme == "file" or parsed.scheme == "":
            local_path = Path(parsed.path) if parsed.scheme == "file" else Path(url)
            if local_path.exists():
                filename = local_path.name
                dest_path = dest_dir / filename
                print(f"[DatasetManager] Copying local file: {local_path}")
                shutil.copy2(local_path, dest_path)
                print(f"[DatasetManager] Copied: {dest_path.name} ({dest_path.stat().st_size} bytes)")
                return dest_path
            else:
                raise FileNotFoundError(f"Local file not found: {local_path}")

        filename = Path(parsed.path).name
        if not filename:
            filename = "dataset"

        dest_path = dest_dir / filename

        print(f"[DatasetManager] Downloading: {url}")

        # Get proxy from environment
        proxies = {}
        https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
        if https_proxy:
            proxies["https"] = https_proxy
        if http_proxy:
            proxies["http"] = http_proxy

        # Download with retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    url,
                    stream=True,
                    timeout=300,
                    proxies=proxies if proxies else None,
                )
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0

                with open(dest_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = downloaded / total_size * 100
                            print(
                                f"\r[DatasetManager] Progress: {progress:.1f}%",
                                end="",
                                flush=True,
                            )

                print()  # New line after progress
                print(f"[DatasetManager] Downloaded: {dest_path.name} ({downloaded} bytes)")
                return dest_path

            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"[DatasetManager] Download failed, retrying in {wait}s: {e}")
                    import time
                    time.sleep(wait)
                else:
                    raise

    def _extract_archive(self, archive_path: Path, extract_dir: Path) -> Path:
        """Extract archive and find CSV/JSONL file.

        Args:
            archive_path: Path to archive file
            extract_dir: Directory to extract to

        Returns:
            Path to data file (CSV or JSONL)
        """
        suffix = archive_path.suffix.lower()
        name = archive_path.name.lower()

        # Handle different archive formats
        if suffix == ".gz" and not name.endswith(".tar.gz"):
            # Single gzip file
            output_name = archive_path.stem  # Remove .gz
            output_path = extract_dir / output_name
            print(f"[DatasetManager] Extracting gzip: {archive_path.name}")
            with gzip.open(archive_path, "rb") as f_in:
                with open(output_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return output_path

        elif suffix == ".zip":
            # ZIP archive
            print(f"[DatasetManager] Extracting zip: {archive_path.name}")
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(extract_dir)
            return self._find_data_file(extract_dir)

        elif name.endswith(".tar.gz") or name.endswith(".tgz"):
            # Tarball
            print(f"[DatasetManager] Extracting tarball: {archive_path.name}")
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(extract_dir)
            return self._find_data_file(extract_dir)

        elif suffix in [".csv", ".jsonl"]:
            # Not an archive, use directly
            return archive_path

        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported: .csv, .jsonl, .gz, .zip, .tar.gz"
            )

    def _find_data_file(self, directory: Path) -> Path:
        """Find CSV or JSONL file in directory.

        Args:
            directory: Directory to search

        Returns:
            Path to data file

        Raises:
            ValueError: If no data file found
        """
        # Look for CSV files first
        csv_files = list(directory.rglob("*.csv"))
        if csv_files:
            # Prefer larger files (more data)
            csv_files.sort(key=lambda p: p.stat().st_size, reverse=True)
            print(f"[DatasetManager] Found CSV: {csv_files[0].name}")
            return csv_files[0]

        # Look for JSONL files
        jsonl_files = list(directory.rglob("*.jsonl"))
        if jsonl_files:
            jsonl_files.sort(key=lambda p: p.stat().st_size, reverse=True)
            print(f"[DatasetManager] Found JSONL: {jsonl_files[0].name}")
            return jsonl_files[0]

        raise ValueError(
            f"No CSV or JSONL file found in extracted archive. "
            f"Contents: {list(directory.rglob('*'))}"
        )

    def _convert_to_genai_format(
        self,
        input_file: Path,
        output_dir: Path,
        deduplicate: bool = True,
    ) -> Path:
        """Convert API logs to genai-bench CSV format.

        Args:
            input_file: Path to input CSV or JSONL file
            output_dir: Directory to save converted file
            deduplicate: Whether to remove duplicate prompts

        Returns:
            Path to converted prompts.csv file
        """
        print(f"[DatasetManager] Converting: {input_file.name}")

        suffix = input_file.suffix.lower()

        if suffix == ".csv":
            prompts = self._parse_csv_logs(input_file)
        elif suffix == ".jsonl":
            prompts = self._parse_jsonl_logs(input_file)
        else:
            raise ValueError(f"Unsupported data format: {suffix}")

        if not prompts:
            raise ValueError("No prompts extracted from dataset")

        print(f"[DatasetManager] Extracted {len(prompts)} prompts")

        # Deduplicate
        if deduplicate:
            original_count = len(prompts)
            prompts = list(dict.fromkeys(prompts))  # Preserve order
            print(f"[DatasetManager] After deduplication: {len(prompts)} unique prompts")

        # Save as CSV
        output_file = output_dir / "prompts.csv"
        df = pd.DataFrame({"prompt": prompts})
        df.to_csv(output_file, index=False)

        # Save stats
        stats = {
            "total_prompts": len(prompts),
            "source_file": input_file.name,
        }
        with open(output_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        return output_file

    def _parse_csv_logs(self, csv_file: Path) -> list[str]:
        """Parse prompts from CSV API logs.

        Expects CSV with 'request_body' column containing JSON with 'prompt' field.
        """
        df = pd.read_csv(csv_file)

        if "request_body" not in df.columns:
            raise ValueError(
                f"CSV must have 'request_body' column. "
                f"Found columns: {list(df.columns)}"
            )

        prompts = []
        parse_errors = 0

        for _, row in df.iterrows():
            prompt = self._extract_prompt_from_request_body(row.get("request_body", ""))
            if prompt:
                prompts.append(prompt)
            else:
                parse_errors += 1

        if parse_errors > 0:
            print(f"[DatasetManager] Warning: {parse_errors} rows failed to parse")

        return prompts

    def _parse_jsonl_logs(self, jsonl_file: Path) -> list[str]:
        """Parse prompts from JSONL API logs.

        Expects JSONL where each line has 'request_body' with 'prompt' field,
        or directly has 'prompt' field.
        """
        prompts = []
        parse_errors = 0

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)

                    # Try direct prompt field first
                    if "prompt" in data:
                        prompts.append(data["prompt"])
                    # Try request_body JSON
                    elif "request_body" in data:
                        prompt = self._extract_prompt_from_request_body(
                            data["request_body"]
                        )
                        if prompt:
                            prompts.append(prompt)
                        else:
                            parse_errors += 1
                    else:
                        parse_errors += 1
                except json.JSONDecodeError:
                    parse_errors += 1

        if parse_errors > 0:
            print(f"[DatasetManager] Warning: {parse_errors} lines failed to parse")

        return prompts

    def _extract_prompt_from_request_body(self, raw: str) -> Optional[str]:
        """Extract prompt from request_body JSON string.

        Handles double-escaped JSON from CSV export.
        Supports both text completions (prompt field) and chat completions (messages array).
        """
        if pd.isna(raw) or not raw:
            return None

        try:
            # Handle string input
            if isinstance(raw, str):
                # First, handle CSV escaping: \" -> "
                fixed = raw.replace('\\"', '"')
                # Handle nested escaped quotes: \\" -> \"
                fixed = fixed.replace('\\\\"', '\\"')
                body = json.loads(fixed)
            else:
                # Already a dict
                body = raw

            # Try text completions format first (prompt field)
            if "prompt" in body:
                return body.get("prompt")

            # Try chat completions format (messages array)
            if "messages" in body:
                messages = body["messages"]
                if not messages:
                    return None
                # Extract the last user message content as the prompt
                # This is most representative of the actual input for benchmarking
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if content:
                            # Handle multi-modal content format (list of content parts)
                            # e.g., [{"type": "text", "text": "..."}]
                            if isinstance(content, list):
                                text_parts = []
                                for part in content:
                                    if isinstance(part, dict) and part.get("type") == "text":
                                        text_parts.append(part.get("text", ""))
                                return " ".join(text_parts) if text_parts else None
                            return content
                # Fallback: use last message content regardless of role
                last_msg = messages[-1]
                content = last_msg.get("content")
                # Handle list content in fallback too
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    return " ".join(text_parts) if text_parts else None
                return content

            return None
        except (json.JSONDecodeError, AttributeError):
            return None

    def clear_cache(self, url: Optional[str] = None):
        """Clear cached datasets.

        Args:
            url: If provided, only clear cache for this URL.
                 Otherwise, clear all cached datasets.
        """
        if url:
            cache_key = self._get_cache_key(url)
            cache_path = self.cache_dir / cache_key
            if cache_path.exists():
                shutil.rmtree(cache_path)
                print(f"[DatasetManager] Cleared cache for: {url}")
        else:
            for item in self.cache_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
            print(f"[DatasetManager] Cleared all cached datasets")
