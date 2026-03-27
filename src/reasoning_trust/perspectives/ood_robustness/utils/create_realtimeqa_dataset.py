"""
Create JSONL dataset from RealTimeQA GitHub repository.

This script fetches data from the RealTimeQA GitHub repository and converts it
to JSONL format compatible with the DecodingTrust OOD dataset structure (like knowledge.jsonl).

Expected output columns in JSONL:
- category: "qa_2023" or "qa_2025"
- split: "test"
- id: "qa_2023_{index}" or "qa_2025_{index}" (1-indexed)
- question_sentence: The question text
- choices: List of 4 choice options
- question_date: Date string
- answer: Integer index of the correct answer (0-3)
"""

import json
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import Dataset
import argparse


# GitHub repository base URLs
GITHUB_BASE_URL = "https://raw.githubusercontent.com/realtimeqa/realtimeqa_public/main"
GITHUB_API_BASE = "https://api.github.com/repos/realtimeqa/realtimeqa_public/contents"


def fetch_file_list(year: int) -> List[str]:
    """
    Fetch list of _qa.jsonl files from GitHub repository for a given year.
    
    Args:
        year: The year (2023 or 2025)
        
    Returns:
        List of file paths relative to the repository root
    """
    year_path = f"past/{year}"
    url = f"{GITHUB_API_BASE}/{year_path}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        files = response.json()
        
        qa_files = []
        for file_info in files:
            if isinstance(file_info, dict) and file_info.get("name", "").endswith("_qa.jsonl"):
                qa_files.append(f"{year_path}/{file_info['name']}")
        
        return sorted(qa_files)
    except Exception as e:
        print(f"Error fetching file list for {year}: {e}")
        return []


def fetch_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Fetch and parse a JSONL file from GitHub.
    
    Args:
        file_path: Path to the file relative to repository root
        
    Returns:
        List of parsed JSON objects
    """
    url = f"{GITHUB_BASE_URL}/{file_path}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        data = []
        for line in response.text.strip().split('\n'):
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line in {file_path}: {e}")
                    continue
        
        return data
    except Exception as e:
        print(f"Error fetching {file_path}: {e}")
        return []


def parse_date(date_str: str) -> Optional[datetime]:
    """
    Parse date string to datetime object.
    
    Supports common date formats from RealTimeQA dataset.
    """
    # Try common date formats from RealTimeQA
    formats = [
        "%Y/%m/%d",  # RealTimeQA format: "2023/01/05"
        "%Y-%m-%d",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
        "%m/%d/%Y",
        "%d/%m/%Y",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    
    return None


def filter_by_date(data: List[Dict[str, Any]], year: int, cutoff_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """
    Filter data by question_date.
    
    For 2023: Include all data
    For 2025: Include data from cutoff_date onwards (default: August 6, 2025)
    
    Args:
        data: List of data entries
        year: The year (2023 or 2025)
        cutoff_date: Cutoff date for filtering (default: August 6, 2025 for 2025)
        
    Returns:
        Filtered list of data entries
    """
    if year == 2023:
        # Include all 2023 data
        return data
    
    if year == 2025:
        if cutoff_date is None:
            cutoff_date = datetime(2025, 8, 6)
        
        filtered = []
        for entry in data:
            question_date_str = entry.get("question_date", "") or entry.get("date", "")
            if not question_date_str:
                continue
            
            parsed_date = parse_date(question_date_str)
            if parsed_date and parsed_date >= cutoff_date:
                filtered.append(entry)
        
        return filtered
    
    return []


def convert_to_hf_format(data: List[Dict[str, Any]], category: str, start_idx: int = 1) -> List[Dict[str, Any]]:
    """
    Convert RealTimeQA data format to HuggingFace dataset format.
    
    Args:
        data: List of data entries from JSONL files
        category: Category name ("qa_2023" or "qa_2025")
        start_idx: Starting index for IDs (default: 1)
        
    Returns:
        List of converted entries in HuggingFace format
    """
    converted = []
    
    for idx, entry in enumerate(data, start=start_idx):
        # RealTimeQA format: question_sentence, choices, question_date, answer (as list)
        question = entry.get("question_sentence", "") or entry.get("question", "")
        choices = entry.get("choices", [])
        question_date = entry.get("question_date", "") or entry.get("date", "")
        answer_raw = entry.get("answer", ["0"])
        
        # Handle answer - RealTimeQA stores it as a list with one string element
        if isinstance(answer_raw, list) and len(answer_raw) > 0:
            answer_str = str(answer_raw[0])
        elif isinstance(answer_raw, str):
            answer_str = answer_raw
        else:
            answer_str = "0"
        
        # Convert answer string to integer (0-3)
        try:
            answer = int(answer_str)
        except (ValueError, TypeError):
            answer = 0
        
        # Ensure answer is between 0-3
        answer = max(0, min(3, answer))
        
        # Ensure choices is a list with exactly 4 elements
        if not isinstance(choices, list):
            choices = []
        while len(choices) < 4:
            choices.append("")
        choices = choices[:4]
        
        converted_entry = {
            "category": category,
            "split": "test",
            "id": f"{category}_{idx}",
            "question_sentence": question,
            "choices": choices,
            "question_date": question_date,
            "answer": answer,
        }
        
        converted.append(converted_entry)
    
    return converted


def create_dataset(
    year_2023: bool = True,
    year_2025: bool = True,
    cutoff_date_2025: Optional[str] = None,
    output_path: Optional[str] = None,
    save_to_disk: bool = False,
) -> Dataset:
    """
    Create HuggingFace dataset from RealTimeQA GitHub repository.
    
    Args:
        year_2023: Whether to include 2023 data
        year_2025: Whether to include 2025 data
        cutoff_date_2025: Cutoff date for 2025 data (format: "YYYY-MM-DD", default: "2025-08-06")
        output_path: Path to save the dataset (optional)
        save_to_disk: Whether to save dataset to disk
        
    Returns:
        HuggingFace Dataset object
    """
    all_data = []
    current_idx_2023 = 1
    current_idx_2025 = 1
    
    # Parse cutoff date for 2025
    cutoff_datetime = None
    if cutoff_date_2025:
        cutoff_datetime = parse_date(cutoff_date_2025)
        if not cutoff_datetime:
            print(f"Warning: Could not parse cutoff date {cutoff_date_2025}, using default (2025-08-06)")
            cutoff_datetime = datetime(2025, 8, 6)
    
    # Fetch 2023 data
    if year_2023:
        print("Fetching 2023 data...")
        files_2023 = fetch_file_list(2023)
        print(f"Found {len(files_2023)} files for 2023")
        
        for file_path in files_2023:
            print(f"  Processing {file_path}...")
            data_2023 = fetch_jsonl_file(file_path)
            filtered_2023 = filter_by_date(data_2023, 2023)
            converted_2023 = convert_to_hf_format(filtered_2023, "qa_2023", start_idx=current_idx_2023)
            all_data.extend(converted_2023)
            current_idx_2023 += len(converted_2023)
            print(f"    Added {len(converted_2023)} entries")
    
    # Fetch 2025 data
    if year_2025:
        print("\nFetching 2025 data...")
        files_2025 = fetch_file_list(2025)
        print(f"Found {len(files_2025)} files for 2025")
        
        for file_path in files_2025:
            print(f"  Processing {file_path}...")
            data_2025 = fetch_jsonl_file(file_path)
            filtered_2025 = filter_by_date(data_2025, 2025, cutoff_datetime)
            converted_2025 = convert_to_hf_format(filtered_2025, "qa_2025", start_idx=current_idx_2025)
            all_data.extend(converted_2025)
            current_idx_2025 += len(converted_2025)
            print(f"    Added {len(converted_2025)} entries")
    
    print(f"\nTotal entries: {len(all_data)}")
    
  
    dataset = Dataset.from_list(all_data)
    
    # Save as JSONL file
    if save_to_disk:
        if output_path:
            output_path_obj = Path(output_path)
            if output_path_obj.is_dir() or not output_path_obj.suffix:
                jsonl_path = output_path_obj / "knowledge.jsonl"
            else:
                jsonl_path = output_path_obj
        else:
            jsonl_path = Path("knowledge.jsonl")
        
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving JSONL file to: {jsonl_path}")
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for entry in all_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(all_data)} entries to {jsonl_path}")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Create JSONL dataset from RealTimeQA GitHub repository. "
                    "Includes all 2023 data and 2025 data from August 6th onwards by default. "
                    "Saves as knowledge.jsonl file (similar to DecodingTrust format)."
    )
    parser.add_argument(
        "--cutoff-date-2025",
        type=str,
        default="2025-08-06",
        help="Cutoff date for 2025 data (format: YYYY-MM-DD, default: 2025-08-06)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save the JSONL file (directory or full path to knowledge.jsonl)"
    )
    parser.add_argument(
        "--save-to-disk",
        dest="save_to_disk",
        action="store_true",
        help="Save dataset as JSONL file (enabled by default)"
    )
    parser.add_argument(
        "--no-save",
        dest="save_to_disk",
        action="store_false",
        help="Don't save JSONL file"
    )
    
    parser.set_defaults(save_to_disk=True)
    
    args = parser.parse_args()
    
    dataset = create_dataset(
        year_2023=True,  # Always include 2023 data
        year_2025=True,  # Always include 2025 data
        cutoff_date_2025=args.cutoff_date_2025,
        output_path=args.output_path,
        save_to_disk=args.save_to_disk,
    )
    
    print(f"\nDataset created successfully!")
    print(f"Dataset size: {len(dataset)}")
    print(f"Features: {dataset.features}")
    
    if len(dataset) > 0:
        print(f"\nSample entry:")
        print(json.dumps(dataset[0], indent=2))
    
    return dataset


if __name__ == "__main__":
    main()

