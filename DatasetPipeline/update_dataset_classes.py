#!/usr/bin/env python3
"""
Dataset Class Label Update Script for Project Prayag BEV Datasets.

This script updates all CSV files across the three dataset variants to include
the OBB class labels extracted from their corresponding JSON files.

Class Mapping:
    0 - HPE (Human-Pedestrian Entity)
    1 - LVE (Large Vehicle Entity)
    2 - SVE (Small Vehicle Entity)

Datasets Updated:
    1. ChunkedProjectPrayagBEVDataset (30 FPS)
    2. ChunkedProjectPrayagBEVDataset10Hz (10 FPS)
    3. ProjectPrayagTopDownDataset (Original)

Author: Senior ML Engineer
Date: December 2024
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Constants and Configuration
# ============================================================================

CLASS_MAPPING = {
    0: "HPE",  # Human-Pedestrian Entity
    1: "LVE",  # Large Vehicle Entity
    2: "SVE",  # Small Vehicle Entity
}

CLASS_DESCRIPTIONS = {
    0: "Human-Pedestrian Entity (HPE) - Pedestrians, cyclists, and other human-powered entities",
    1: "Large Vehicle Entity (LVE) - Buses, trucks, large commercial vehicles",
    2: "Small Vehicle Entity (SVE) - Cars, motorcycles, auto-rickshaws, small vehicles",
}


@dataclass
class DatasetConfig:
    """Configuration for each dataset variant."""
    name: str
    base_path: Path
    has_chunks: bool
    splits: List[str]
    annotation_subdir: str
    intermediate_subdir: Optional[str]


# ============================================================================
# Core Processing Functions
# ============================================================================

def load_json_class_mapping(json_path: Path) -> Dict[Tuple[int, int], int]:
    """
    Load JSON tracking file and extract class mapping.
    
    Args:
        json_path: Path to the JSON tracking file.
        
    Returns:
        Dictionary mapping (frame_id, track_id) -> class_id
    """
    class_map = {}
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # JSON structure: {frame_id: {track_id: {center, obb, class}, ...}, ...}
        for frame_id_str, tracks in data.items():
            frame_id = int(frame_id_str)
            for track_id_str, track_data in tracks.items():
                track_id = int(track_id_str)
                if 'class' in track_data:
                    class_map[(frame_id, track_id)] = track_data['class']
                    
    except Exception as e:
        logger.error(f"Error loading JSON file {json_path}: {e}")
        
    return class_map


def update_csv_with_classes(
    csv_path: Path,
    json_path: Path,
    output_path: Optional[Path] = None
) -> Tuple[int, int, Dict[int, int]]:
    """
    Update a CSV file with class information from the corresponding JSON.
    
    Args:
        csv_path: Path to the input CSV file.
        json_path: Path to the JSON file containing class info.
        output_path: Path for the output CSV. If None, overwrites input.
        
    Returns:
        Tuple of (rows_processed, rows_with_class, class_counts)
    """
    if output_path is None:
        output_path = csv_path
    
    # Load class mapping from JSON
    class_map = load_json_class_mapping(json_path)
    
    if not class_map:
        logger.warning(f"No class data found in {json_path}")
        return 0, 0, {}
    
    rows_processed = 0
    rows_with_class = 0
    class_counts = defaultdict(int)
    updated_rows = []
    
    # Read existing CSV
    try:
        with open(csv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames)
            
            # Check if 'class' column already exists
            has_class_column = 'class' in fieldnames
            if not has_class_column:
                # Insert 'class' after 'track_id' or at the end
                if 'track_id' in fieldnames:
                    idx = fieldnames.index('track_id') + 1
                    fieldnames.insert(idx, 'class')
                else:
                    fieldnames.append('class')
            
            # Also add class_name column if not present
            if 'class_name' not in fieldnames:
                if 'class' in fieldnames:
                    idx = fieldnames.index('class') + 1
                    fieldnames.insert(idx, 'class_name')
                else:
                    fieldnames.append('class_name')
            
            for row in reader:
                rows_processed += 1
                frame_id = int(row['frame_id'])
                track_id = int(row['track_id'])
                
                key = (frame_id, track_id)
                if key in class_map:
                    class_id = class_map[key]
                    row['class'] = class_id
                    row['class_name'] = CLASS_MAPPING.get(class_id, 'Unknown')
                    rows_with_class += 1
                    class_counts[class_id] += 1
                else:
                    # Default to SVE (most common) if not found
                    row['class'] = 2
                    row['class_name'] = CLASS_MAPPING[2]
                    class_counts[2] += 1
                
                updated_rows.append(row)
                
    except Exception as e:
        logger.error(f"Error reading CSV file {csv_path}: {e}")
        return 0, 0, {}
    
    # Write updated CSV
    try:
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)
            
    except Exception as e:
        logger.error(f"Error writing CSV file {output_path}: {e}")
        return 0, 0, {}
    
    return rows_processed, rows_with_class, dict(class_counts)


def process_chunked_dataset(config: DatasetConfig) -> Dict:
    """
    Process a chunked dataset (ChunkedProjectPrayagBEVDataset variants).
    
    Args:
        config: Dataset configuration.
        
    Returns:
        Statistics dictionary.
    """
    stats = {
        'dataset': config.name,
        'files_processed': 0,
        'total_rows': 0,
        'rows_with_class': 0,
        'class_distribution': defaultdict(int),
        'errors': []
    }
    
    for split in config.splits:
        split_dir = config.base_path / split / config.annotation_subdir
        
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            continue
        
        # Find all CSV files in this split
        csv_files = list(split_dir.glob("*_tracks.csv"))
        
        for csv_path in csv_files:
            # Construct corresponding JSON path
            json_path = csv_path.with_suffix('.json')
            
            if not json_path.exists():
                logger.warning(f"JSON file not found: {json_path}")
                stats['errors'].append(f"Missing JSON: {json_path}")
                continue
            
            logger.info(f"Processing: {csv_path.name}")
            
            rows, with_class, class_counts = update_csv_with_classes(
                csv_path, json_path
            )
            
            stats['files_processed'] += 1
            stats['total_rows'] += rows
            stats['rows_with_class'] += with_class
            
            for cls, count in class_counts.items():
                stats['class_distribution'][cls] += count
    
    return stats


def process_original_dataset(config: DatasetConfig) -> Dict:
    """
    Process the original ProjectPrayagTopDownDataset.
    
    Args:
        config: Dataset configuration.
        
    Returns:
        Statistics dictionary.
    """
    stats = {
        'dataset': config.name,
        'files_processed': 0,
        'total_rows': 0,
        'rows_with_class': 0,
        'class_distribution': defaultdict(int),
        'errors': []
    }
    
    if config.intermediate_subdir:
        intermediate_dir = config.base_path / config.intermediate_subdir
        
        if not intermediate_dir.exists():
            logger.error(f"Intermediate directory not found: {intermediate_dir}")
            return stats
        
        # Process each scene directory
        for scene_dir in intermediate_dir.iterdir():
            if not scene_dir.is_dir():
                continue
            
            csv_files = list(scene_dir.glob("*_tracks.csv"))
            
            for csv_path in csv_files:
                json_path = csv_path.with_suffix('.json')
                
                if not json_path.exists():
                    logger.warning(f"JSON file not found: {json_path}")
                    stats['errors'].append(f"Missing JSON: {json_path}")
                    continue
                
                logger.info(f"Processing: {csv_path.name}")
                
                rows, with_class, class_counts = update_csv_with_classes(
                    csv_path, json_path
                )
                
                stats['files_processed'] += 1
                stats['total_rows'] += rows
                stats['rows_with_class'] += with_class
                
                for cls, count in class_counts.items():
                    stats['class_distribution'][cls] += count
    
    return stats


def regenerate_unified_csv(config: DatasetConfig) -> bool:
    """
    Regenerate the unified_tracking_data.csv from individual track files.
    
    Args:
        config: Dataset configuration.
        
    Returns:
        True if successful, False otherwise.
    """
    unified_csv_path = config.base_path / "unified_tracking_data.csv"
    
    logger.info(f"Regenerating unified CSV for {config.name}")
    
    all_rows = []
    fieldnames = None
    
    if config.has_chunks:
        # Process chunked datasets
        for split in config.splits:
            split_dir = config.base_path / split / config.annotation_subdir
            
            if not split_dir.exists():
                continue
            
            csv_files = sorted(split_dir.glob("*_tracks.csv"))
            
            for csv_path in csv_files:
                try:
                    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
                        reader = csv.DictReader(f)
                        if fieldnames is None:
                            fieldnames = reader.fieldnames
                        for row in reader:
                            all_rows.append(row)
                except Exception as e:
                    logger.error(f"Error reading {csv_path}: {e}")
                    
    else:
        # Process original dataset
        if config.intermediate_subdir:
            intermediate_dir = config.base_path / config.intermediate_subdir
            
            for scene_dir in sorted(intermediate_dir.iterdir()):
                if not scene_dir.is_dir():
                    continue
                
                csv_files = sorted(scene_dir.glob("*_tracks.csv"))
                
                for csv_path in csv_files:
                    try:
                        with open(csv_path, 'r', encoding='utf-8', newline='') as f:
                            reader = csv.DictReader(f)
                            if fieldnames is None:
                                fieldnames = reader.fieldnames
                            for row in reader:
                                all_rows.append(row)
                    except Exception as e:
                        logger.error(f"Error reading {csv_path}: {e}")
    
    if not all_rows or fieldnames is None:
        logger.error(f"No data found for unified CSV generation in {config.name}")
        return False
    
    # Write unified CSV
    try:
        with open(unified_csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        
        logger.info(f"Unified CSV written: {unified_csv_path} ({len(all_rows)} rows)")
        return True
        
    except Exception as e:
        logger.error(f"Error writing unified CSV: {e}")
        return False


def regenerate_unified_json(config: DatasetConfig) -> bool:
    """
    Regenerate the unified_tracking_data.json from individual track files.
    
    Args:
        config: Dataset configuration.
        
    Returns:
        True if successful, False otherwise.
    """
    unified_json_path = config.base_path / "unified_tracking_data.json"
    
    logger.info(f"Regenerating unified JSON for {config.name}")
    
    all_data = {}
    
    if config.has_chunks:
        # Process chunked datasets
        for split in config.splits:
            split_dir = config.base_path / split / config.annotation_subdir
            
            if not split_dir.exists():
                continue
            
            json_files = sorted(split_dir.glob("*_tracks.json"))
            
            for json_path in json_files:
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Extract scene_id from filename
                    scene_id = json_path.stem.replace('_tracks', '')
                    all_data[scene_id] = data
                    
                except Exception as e:
                    logger.error(f"Error reading {json_path}: {e}")
                    
    else:
        # Process original dataset
        if config.intermediate_subdir:
            intermediate_dir = config.base_path / config.intermediate_subdir
            
            for scene_dir in sorted(intermediate_dir.iterdir()):
                if not scene_dir.is_dir():
                    continue
                
                json_files = sorted(scene_dir.glob("*_tracks.json"))
                
                for json_path in json_files:
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        scene_id = json_path.stem.replace('_tracks', '')
                        all_data[scene_id] = data
                        
                    except Exception as e:
                        logger.error(f"Error reading {json_path}: {e}")
    
    if not all_data:
        logger.error(f"No data found for unified JSON generation in {config.name}")
        return False
    
    # Write unified JSON
    try:
        with open(unified_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, separators=(',', ':'))
        
        logger.info(f"Unified JSON written: {unified_json_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error writing unified JSON: {e}")
        return False


def compute_class_statistics(config: DatasetConfig) -> Dict:
    """
    Compute class distribution statistics for the dataset.
    
    Args:
        config: Dataset configuration.
        
    Returns:
        Statistics dictionary with class distributions.
    """
    class_counts = defaultdict(int)
    per_scene_stats = {}
    
    if config.has_chunks:
        for split in config.splits:
            split_dir = config.base_path / split / config.annotation_subdir
            
            if not split_dir.exists():
                continue
            
            json_files = sorted(split_dir.glob("*_tracks.json"))
            
            for json_path in json_files:
                scene_id = json_path.stem.replace('_tracks', '')
                scene_counts = defaultdict(int)
                
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    for frame_id, tracks in data.items():
                        for track_id, track_data in tracks.items():
                            if 'class' in track_data:
                                cls = track_data['class']
                                class_counts[cls] += 1
                                scene_counts[cls] += 1
                    
                    per_scene_stats[scene_id] = dict(scene_counts)
                    
                except Exception as e:
                    logger.error(f"Error processing {json_path}: {e}")
    else:
        if config.intermediate_subdir:
            intermediate_dir = config.base_path / config.intermediate_subdir
            
            for scene_dir in sorted(intermediate_dir.iterdir()):
                if not scene_dir.is_dir():
                    continue
                
                json_files = sorted(scene_dir.glob("*_tracks.json"))
                
                for json_path in json_files:
                    scene_id = json_path.stem.replace('_tracks', '')
                    scene_counts = defaultdict(int)
                    
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        for frame_id, tracks in data.items():
                            for track_id, track_data in tracks.items():
                                if 'class' in track_data:
                                    cls = track_data['class']
                                    class_counts[cls] += 1
                                    scene_counts[cls] += 1
                        
                        per_scene_stats[scene_id] = dict(scene_counts)
                        
                    except Exception as e:
                        logger.error(f"Error processing {json_path}: {e}")
    
    total = sum(class_counts.values())
    
    return {
        'total_annotations': total,
        'class_counts': dict(class_counts),
        'class_percentages': {
            cls: (count / total * 100) if total > 0 else 0
            for cls, count in class_counts.items()
        },
        'per_scene': per_scene_stats
    }


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    start_time = time.time()
    
    # Define base path
    base_path = Path(__file__).parent
    
    # Define dataset configurations
    datasets = [
        DatasetConfig(
            name="ChunkedProjectPrayagBEVDataset",
            base_path=base_path / "ChunkedProjectPrayagBEVDataset",
            has_chunks=True,
            splits=["train", "val", "test"],
            annotation_subdir="annotations",
            intermediate_subdir=None
        ),
        DatasetConfig(
            name="ChunkedProjectPrayagBEVDataset10Hz",
            base_path=base_path / "ChunkedProjectPrayagBEVDataset10Hz",
            has_chunks=True,
            splits=["train", "val", "test"],
            annotation_subdir="annotations",
            intermediate_subdir=None
        ),
        DatasetConfig(
            name="ProjectPrayagTopDownDataset",
            base_path=base_path / "ProjectPrayagTopDownDataset",
            has_chunks=False,
            splits=[],
            annotation_subdir="",
            intermediate_subdir="intermediate_files"
        ),
    ]
    
    all_stats = []
    
    print("=" * 80)
    print("Project Prayag Dataset Class Label Update Script")
    print("=" * 80)
    print()
    print("Class Mapping:")
    for cls_id, cls_name in CLASS_MAPPING.items():
        print(f"  {cls_id} - {cls_name}: {CLASS_DESCRIPTIONS[cls_id]}")
    print()
    
    for config in datasets:
        print("-" * 80)
        print(f"Processing: {config.name}")
        print("-" * 80)
        
        if not config.base_path.exists():
            logger.warning(f"Dataset path not found: {config.base_path}")
            continue
        
        # Process the dataset
        if config.has_chunks:
            stats = process_chunked_dataset(config)
        else:
            stats = process_original_dataset(config)
        
        all_stats.append(stats)
        
        # Print statistics
        print(f"\n  Files processed: {stats['files_processed']}")
        print(f"  Total rows: {stats['total_rows']:,}")
        print(f"  Rows with class: {stats['rows_with_class']:,}")
        print("  Class distribution:")
        for cls_id, count in sorted(stats['class_distribution'].items()):
            pct = (count / stats['total_rows'] * 100) if stats['total_rows'] > 0 else 0
            print(f"    {cls_id} ({CLASS_MAPPING.get(cls_id, 'Unknown')}): {count:,} ({pct:.2f}%)")
        
        if stats['errors']:
            print(f"  Errors: {len(stats['errors'])}")
            for err in stats['errors'][:5]:
                print(f"    - {err}")
        
        # Regenerate unified files
        print("\n  Regenerating unified files...")
        csv_success = regenerate_unified_csv(config)
        json_success = regenerate_unified_json(config)
        
        print(f"  Unified CSV: {'✓' if csv_success else '✗'}")
        print(f"  Unified JSON: {'✓' if json_success else '✗'}")
        
        print()
    
    # Final summary
    elapsed = time.time() - start_time
    
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    
    total_files = sum(s['files_processed'] for s in all_stats)
    total_rows = sum(s['total_rows'] for s in all_stats)
    
    print(f"Total datasets processed: {len(all_stats)}")
    print(f"Total files updated: {total_files}")
    print(f"Total rows processed: {total_rows:,}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print()
    
    # Combined class distribution
    combined_dist = defaultdict(int)
    for stats in all_stats:
        for cls_id, count in stats['class_distribution'].items():
            combined_dist[cls_id] += count
    
    print("Combined Class Distribution:")
    for cls_id, count in sorted(combined_dist.items()):
        pct = (count / total_rows * 100) if total_rows > 0 else 0
        print(f"  {cls_id} ({CLASS_MAPPING.get(cls_id, 'Unknown')}): {count:,} ({pct:.2f}%)")
    
    print()
    print("Class Label Legend:")
    print("  0 - HPE: Human-Pedestrian Entity")
    print("  1 - LVE: Large Vehicle Entity") 
    print("  2 - SVE: Small Vehicle Entity")
    print()
    print("Script completed successfully!")
    
    return all_stats


if __name__ == "__main__":
    main()
