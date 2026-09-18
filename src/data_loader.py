"""
Data loading and dataset construction module for DrivebuddyAI Pav Bhaji Challenge.
Extracts and parses JSON metadata, maps image filenames to ground truth labels,
and constructs a clean tabular dataset without using any image pixels.
"""

import os
import re
import json
import zipfile
import logging
from typing import Optional, Tuple, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)


def extract_dataset_if_needed(
    zip_path: str = "dataset.zip",
    extract_to: str = "data"
) -> str:
    """Extract dataset.zip to the target directory if not already extracted."""
    target_dir = os.path.join(extract_to, "dataset")
    if os.path.exists(target_dir) and os.path.exists(os.path.join(target_dir, "pavbhaji.json")):
        logger.info(f"Dataset already extracted at: {target_dir}")
        return target_dir

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"dataset.zip not found at {zip_path}")

    logger.info(f"Extracting {zip_path} to {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    logger.info(f"Extraction complete. Target directory: {target_dir}")
    return target_dir


def parse_instagram_json(json_path: str) -> list:
    """Safely parse pavbhaji.json without assuming top-level structure."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found at {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            data = data["data"]
        else:
            data = list(data.values())

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list or dict, got {type(data)}")

    logger.info(f"Parsed {len(data)} raw Instagram posts from {json_path}")
    return data


def extract_caption_text(item: Dict[str, Any]) -> str:
    """Extract caption text safely from Instagram GraphQL structure."""
    edge_caption = item.get("edge_media_to_caption")
    if isinstance(edge_caption, dict):
        edges = edge_caption.get("edges", [])
        if edges and isinstance(edges[0], dict):
            node = edges[0].get("node", {})
            return str(node.get("text", "") or "")
    elif isinstance(edge_caption, str):
        return edge_caption
    return ""


def extract_filename_from_url(display_url: str) -> str:
    """Extract the clean image filename from Instagram CDN display URL."""
    if not display_url:
        return ""
    # Strip URL query parameters and get basename
    clean_url = display_url.split("?")[0]
    return clean_url.split("/")[-1]


def build_dataset(
    dataset_dir: str = "data/dataset",
    include_unlabeled: bool = False
) -> pd.DataFrame:
    """
    Build a clean tabular DataFrame from Instagram JSON and image directory structure.
    Ground truth labels:
      1: Pav Bhaji (image in images/1/)
      0: Not Pav Bhaji (image in images/0/)
      None: Unlabeled (no image provided in dataset)
    """
    json_path = os.path.join(dataset_dir, "pavbhaji.json")
    images_dir = os.path.join(dataset_dir, "images")
    dir_0 = os.path.join(images_dir, "0")
    dir_1 = os.path.join(images_dir, "1")

    raw_posts = parse_instagram_json(json_path)

    # Gather image filenames in ground-truth folders
    images_0 = set(os.listdir(dir_0)) if os.path.exists(dir_0) else set()
    images_1 = set(os.listdir(dir_1)) if os.path.exists(dir_1) else set()
    # Filter out hidden or macOS metadata files
    images_0 = {f for f in images_0 if not f.startswith(".") and not f.startswith("_")}
    images_1 = {f for f in images_1 if not f.startswith(".") and not f.startswith("_")}

    rows = []
    for item in raw_posts:
        post_id = str(item.get("id", ""))
        shortcode = str(item.get("shortcode", ""))
        display_url = str(item.get("display_url", "") or "")
        filename = extract_filename_from_url(display_url)

        # Map to label based on image presence in ground-truth folders
        label = None
        image_path = None
        if filename in images_0:
            label = 0
            image_path = os.path.join("images", "0", filename)
        elif filename in images_1:
            label = 1
            image_path = os.path.join("images", "1", filename)

        if label is None and not include_unlabeled:
            continue

        caption = extract_caption_text(item)
        raw_tags = item.get("tags")
        if isinstance(raw_tags, list):
            tags_list = [str(t).strip() for t in raw_tags if t]
        elif isinstance(raw_tags, str):
            tags_list = [t.strip() for t in raw_tags.split() if t.strip()]
        else:
            tags_list = []

        tags_str = " ".join(f"#{t}" if not t.startswith("#") else t for t in tags_list)

        # Combine relevant textual fields
        combined_text = f"{caption} {tags_str}".strip()

        likes_count = 0
        liked_by = item.get("edge_liked_by")
        if isinstance(liked_by, dict):
            likes_count = int(liked_by.get("count", 0) or 0)

        comments_count = 0
        comment_edge = item.get("edge_media_to_comment")
        if isinstance(comment_edge, dict):
            comments_count = int(comment_edge.get("count", 0) or 0)

        taken_at = item.get("taken_at_timestamp")
        is_video = bool(item.get("is_video", False))

        rows.append({
            "post_id": post_id,
            "shortcode": shortcode,
            "image_filename": filename,
            "image_path": image_path,
            "label": label,
            "description": caption,
            "hashtags": tags_str,
            "tags_list": tags_list,
            "combined_text": combined_text,
            "likes_count": likes_count,
            "comments_count": comments_count,
            "taken_at_timestamp": taken_at,
            "is_video": is_video,
        })

    df = pd.DataFrame(rows)
    logger.info(f"Constructed DataFrame with {len(df)} rows. Labels: {df['label'].value_counts().to_dict()}")
    return df


def get_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate and return comprehensive data quality statistics."""
    total_samples = len(df)
    pos_samples = int((df["label"] == 1).sum())
    neg_samples = int((df["label"] == 0).sum())
    unlabeled_samples = int(df["label"].isna().sum())

    missing_desc = int((df["description"].str.strip() == "").sum())
    missing_tags = int((df["hashtags"].str.strip() == "").sum())
    duplicate_posts = int(df.duplicated(subset=["post_id"]).sum())
    duplicate_desc = int(df.duplicated(subset=["description"]).sum())
    duplicate_images = int(df.duplicated(subset=["image_filename"]).sum())

    return {
        "total_samples": total_samples,
        "positive_samples": pos_samples,
        "negative_samples": neg_samples,
        "unlabeled_samples": unlabeled_samples,
        "positive_percentage": round(pos_samples / total_samples * 100, 2) if total_samples > 0 else 0,
        "negative_percentage": round(neg_samples / total_samples * 100, 2) if total_samples > 0 else 0,
        "missing_descriptions": missing_desc,
        "missing_hashtags": missing_tags,
        "duplicate_post_ids": duplicate_posts,
        "duplicate_descriptions": duplicate_desc,
        "duplicate_images": duplicate_images,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataset_dir = extract_dataset_if_needed()
    df = build_dataset(dataset_dir=dataset_dir, include_unlabeled=False)
    quality = get_data_quality_report(df)
    print("=== Data Quality Report ===")
    for k, v in quality.items():
        print(f"{k}: {v}")
