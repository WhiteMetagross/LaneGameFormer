#!/usr/bin/env python3
"""
Calculate average OBB dimensions per class for the Prayag BEV datasets.
"""

import json
import math
from pathlib import Path
from collections import defaultdict


def calculate_obb_dimensions(obb):
    """Calculate length and width from OBB corners."""
    # OBB corners are 4 points forming a rectangle
    p0, p1, p2, p3 = obb
    
    # Distance between p0-p1 and p1-p2 (two adjacent edges)
    d01 = math.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)
    d12 = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    
    # Length is max, width is min
    length = max(d01, d12)
    width = min(d01, d12)
    return length, width


def analyze_dataset(base_path, is_chunked=True):
    """Analyze a dataset and return class statistics."""
    class_stats = defaultdict(lambda: {'lengths': [], 'widths': [], 'count': 0})
    
    if is_chunked:
        for split in ['train', 'val', 'test']:
            annot_dir = base_path / split / 'annotations'
            if not annot_dir.exists():
                continue
            for json_file in annot_dir.glob('*_tracks.json'):
                process_json_file(json_file, class_stats)
    else:
        intermediate_dir = base_path / 'intermediate_files'
        if intermediate_dir.exists():
            for scene_dir in intermediate_dir.iterdir():
                if scene_dir.is_dir():
                    for json_file in scene_dir.glob('*_tracks.json'):
                        process_json_file(json_file, class_stats)
    
    return class_stats


def process_json_file(json_file, class_stats):
    """Process a single JSON file and update class_stats."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for frame_id, tracks in data.items():
            for track_id, track_data in tracks.items():
                if 'class' in track_data and 'obb' in track_data:
                    cls = track_data['class']
                    obb = track_data['obb']
                    try:
                        length, width = calculate_obb_dimensions(obb)
                        class_stats[cls]['lengths'].append(length)
                        class_stats[cls]['widths'].append(width)
                        class_stats[cls]['count'] += 1
                    except Exception:
                        pass
    except Exception as e:
        print(f"Error processing {json_file}: {e}")


def compute_statistics(class_stats):
    """Compute summary statistics from raw data."""
    results = {}
    for cls in sorted(class_stats.keys()):
        stats = class_stats[cls]
        if stats['count'] > 0:
            avg_len = sum(stats['lengths']) / len(stats['lengths'])
            avg_wid = sum(stats['widths']) / len(stats['widths'])
            results[cls] = {
                'count': stats['count'],
                'avg_length': avg_len,
                'avg_width': avg_wid,
                'aspect_ratio': avg_len / avg_wid if avg_wid > 0 else 0
            }
    return results


CLASS_NAMES = {
    0: 'HPE (Human-Pedestrian Entity)',
    1: 'LVE (Large Vehicle Entity)',
    2: 'SVE (Small Vehicle Entity)'
}


def main():
    base = Path('.')
    
    datasets = [
        ('ChunkedProjectPrayagBEVDataset', True),
        ('ChunkedProjectPrayagBEVDataset10Hz', True),
        ('ProjectPrayagTopDownDataset', False),
    ]
    
    for name, is_chunked in datasets:
        dataset_path = base / name
        if not dataset_path.exists():
            print(f"Dataset not found: {name}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Analyzing: {name}")
        print('='*80)
        
        class_stats = analyze_dataset(dataset_path, is_chunked)
        results = compute_statistics(class_stats)
        
        print(f"\n{'Class':<10} {'Name':<35} {'Count':>12} {'Avg Length':>12} {'Avg Width':>12} {'Aspect Ratio':>14}")
        print('-'*95)
        
        for cls, stats in results.items():
            cls_name = CLASS_NAMES.get(cls, 'Unknown')
            print(f"{cls:<10} {cls_name:<35} {stats['count']:>12,} {stats['avg_length']:>12.2f} {stats['avg_width']:>12.2f} {stats['aspect_ratio']:>14.2f}")
        
        # Print markdown table format
        print("\n\nMarkdown Table Format:")
        print("| Class ID | Class Name | Annotations | Avg Length (px) | Avg Width (px) | Aspect Ratio |")
        print("|----------|------------|-------------|-----------------|----------------|--------------|")
        for cls, stats in results.items():
            cls_name = CLASS_NAMES.get(cls, 'Unknown')
            print(f"| {cls} | {cls_name} | {stats['count']:,} | {stats['avg_length']:.2f} | {stats['avg_width']:.2f} | {stats['aspect_ratio']:.2f} |")


if __name__ == '__main__':
    main()
