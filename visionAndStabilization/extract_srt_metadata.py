#This script extracts GPS, altitude, time, focal length and other metadata from DJI drone SRT files.
#This script processes metadata at 30 FPS to synchronize with converted video files.
#This script saves extracted data as CSV files with the same name as the source SRT files.

import os
import re
import csv
import glob
from datetime import datetime, timedelta

SRT_DIR = r'C:\Users\Xeron\OneDrive\Documents\Programs\PrayagProjectv1.5\ProjectPrayagTopDownDataset\CIRAerialDroneIndianIntersectionsVideoes'
TARGET_FPS = 30.0


def parse_timestamp(ts_str):
    #This function parses SRT timestamp format (HH:MM:SS,mmm) to total milliseconds.
    parts = ts_str.strip().split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    sec_ms = parts[2].split(',')
    seconds = int(sec_ms[0])
    milliseconds = int(sec_ms[1])
    total_ms = (hours * 3600 + minutes * 60 + seconds) * 1000 + milliseconds
    return total_ms


def parse_srt_entry(entry_text):
    #This function parses a single SRT entry and extracts all metadata fields.
    lines = entry_text.strip().split('\n')
    if len(lines) < 3:
        return None
    
    #Parse timestamp line (line index 1).
    timestamp_line = lines[1]
    timestamp_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', timestamp_line)
    if not timestamp_match:
        return None
    
    start_time_ms = parse_timestamp(timestamp_match.group(1))
    end_time_ms = parse_timestamp(timestamp_match.group(2))
    
    #Join remaining lines for content parsing.
    content = ' '.join(lines[2:])
    
    #Remove font tags.
    content = re.sub(r'<[^>]+>', '', content)
    
    #Extract SrtCnt and DiffTime.
    srt_cnt_match = re.search(r'SrtCnt\s*:\s*(\d+)', content)
    diff_time_match = re.search(r'DiffTime\s*:\s*(\d+)ms', content)
    
    #Extract datetime.
    datetime_match = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3},\d{3})', content)
    if not datetime_match:
        datetime_match = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', content)
    
    #Extract camera settings.
    iso_match = re.search(r'\[iso\s*:\s*(\d+)\]', content)
    shutter_match = re.search(r'\[shutter\s*:\s*([^\]]+)\]', content)
    fnum_match = re.search(r'\[fnum\s*:\s*(\d+)\]', content)
    ev_match = re.search(r'\[ev\s*:\s*([^\]]+)\]', content)
    ct_match = re.search(r'\[ct\s*:\s*(\d+)\]', content)
    color_md_match = re.search(r'\[color_md\s*:\s*([^\]]+)\]', content)
    focal_len_match = re.search(r'\[focal_len\s*:\s*(\d+)\]', content)
    dzoom_match = re.search(r'\[dzoom_ratio\s*:\s*(\d+)', content)
    
    #Extract GPS data.
    latitude_match = re.search(r'\[latitude\s*:\s*([^\]]+)\]', content)
    longitude_match = re.search(r'\[longitude\s*:\s*([^\]]+)\]', content)
    altitude_match = re.search(r'\[altitude\s*:\s*([^\]]+)\]', content)
    
    entry_data = {
        'srt_index': int(srt_cnt_match.group(1)) if srt_cnt_match else None,
        'start_time_ms': start_time_ms,
        'end_time_ms': end_time_ms,
        'diff_time_ms': int(diff_time_match.group(1)) if diff_time_match else None,
        'datetime': datetime_match.group(1).strip() if datetime_match else None,
        'iso': int(iso_match.group(1)) if iso_match else None,
        'shutter': shutter_match.group(1).strip() if shutter_match else None,
        'fnum': int(fnum_match.group(1)) if fnum_match else None,
        'ev': float(ev_match.group(1)) if ev_match else None,
        'color_temp': int(ct_match.group(1)) if ct_match else None,
        'color_mode': color_md_match.group(1).strip() if color_md_match else None,
        'focal_length': int(focal_len_match.group(1)) if focal_len_match else None,
        'dzoom_ratio': int(dzoom_match.group(1)) if dzoom_match else None,
        'latitude': float(latitude_match.group(1)) if latitude_match else None,
        'longitude': float(longitude_match.group(1)) if longitude_match else None,
        'altitude': float(altitude_match.group(1)) if altitude_match else None,
    }
    
    return entry_data


def parse_srt_file(srt_path):
    #This function parses an entire SRT file and returns list of metadata entries.
    with open(srt_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    #Split by double newline to get individual entries.
    entries_raw = re.split(r'\n\s*\n', content)
    
    entries = []
    for entry_text in entries_raw:
        entry_text = entry_text.strip()
        if not entry_text:
            continue
        #Check if first line is a number (SRT index).
        lines = entry_text.split('\n')
        if lines and lines[0].strip().isdigit():
            parsed = parse_srt_entry(entry_text)
            if parsed:
                entries.append(parsed)
    
    return entries


def resample_to_30fps(entries, original_fps=None):
    #This function resamples metadata entries to 30 FPS to match video conversion.
    if not entries:
        return []
    
    #Estimate original FPS from diff_time if not provided.
    if original_fps is None:
        diff_times = [e['diff_time_ms'] for e in entries if e['diff_time_ms'] is not None]
        if diff_times:
            avg_diff = sum(diff_times) / len(diff_times)
            original_fps = 1000.0 / avg_diff if avg_diff > 0 else 60.0
        else:
            original_fps = 60.0
    
    #Calculate total duration and number of output frames at 30 FPS.
    total_duration_ms = entries[-1]['end_time_ms'] if entries else 0
    total_frames_30fps = int(total_duration_ms * TARGET_FPS / 1000.0)
    
    if total_frames_30fps <= 0:
        return entries
    
    #Frame skip ratio for resampling.
    frame_skip = original_fps / TARGET_FPS
    
    resampled = []
    frame_duration_ms = 1000.0 / TARGET_FPS
    
    for frame_idx in range(total_frames_30fps):
        #Calculate source frame index.
        src_frame_idx = int(frame_idx * frame_skip)
        if src_frame_idx >= len(entries):
            src_frame_idx = len(entries) - 1
        
        #Get source entry and create resampled entry.
        src_entry = entries[src_frame_idx]
        
        new_start_ms = int(frame_idx * frame_duration_ms)
        new_end_ms = int((frame_idx + 1) * frame_duration_ms)
        
        resampled_entry = src_entry.copy()
        resampled_entry['frame_id'] = frame_idx
        resampled_entry['start_time_ms'] = new_start_ms
        resampled_entry['end_time_ms'] = new_end_ms
        resampled_entry['original_srt_index'] = src_entry['srt_index']
        
        resampled.append(resampled_entry)
    
    return resampled


def save_to_csv(entries, output_path):
    #This function saves metadata entries to a CSV file.
    if not entries:
        print(f'  No entries to save.')
        return False
    
    fieldnames = [
        'frame_id',
        'start_time_ms',
        'end_time_ms',
        'datetime',
        'latitude',
        'longitude',
        'altitude',
        'focal_length',
        'iso',
        'shutter',
        'fnum',
        'ev',
        'color_temp',
        'color_mode',
        'dzoom_ratio',
        'original_srt_index'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(entries)
    
    return True


def process_srt_file(srt_path):
    #This function processes a single SRT file and saves metadata to CSV.
    print(f'Processing: {srt_path}')
    
    #Parse SRT file.
    entries = parse_srt_file(srt_path)
    if not entries:
        print(f'  Warning: No valid entries found.')
        return False
    
    print(f'  Parsed {len(entries)} SRT entries.')
    
    #Estimate original FPS.
    diff_times = [e['diff_time_ms'] for e in entries if e['diff_time_ms'] is not None]
    if diff_times:
        avg_diff = sum(diff_times) / len(diff_times)
        original_fps = 1000.0 / avg_diff if avg_diff > 0 else 60.0
        print(f'  Estimated original FPS: {original_fps:.2f}')
    else:
        original_fps = 60.0
        print(f'  Using default FPS: {original_fps:.2f}')
    
    #Resample to 30 FPS.
    resampled = resample_to_30fps(entries, original_fps)
    print(f'  Resampled to {len(resampled)} frames at 30 FPS.')
    
    #Save to CSV with same name.
    base_name = os.path.splitext(os.path.basename(srt_path))[0]
    csv_path = os.path.join(os.path.dirname(srt_path), f'{base_name}_metadata.csv')
    
    if save_to_csv(resampled, csv_path):
        print(f'  Saved: {csv_path}')
        return True
    
    return False


def find_srt_files(folder):
    #This function finds all SRT files in the given folder.
    srt_files = glob.glob(os.path.join(folder, '*.SRT'))
    srt_files.extend(glob.glob(os.path.join(folder, '*.srt')))
    return sorted(set(srt_files))


if __name__ == '__main__':
    #This script processes all SRT files in the dataset folder.
    if not os.path.isdir(SRT_DIR):
        print(f'Error: folder not found: {SRT_DIR}')
        exit(1)
    
    srt_files = find_srt_files(SRT_DIR)
    if not srt_files:
        print('No SRT files found.')
        exit(0)
    
    print(f'Found {len(srt_files)} SRT file(s) to process.')
    print('=' * 60)
    
    success_count = 0
    for srt_path in srt_files:
        if process_srt_file(srt_path):
            success_count += 1
        print()
    
    print('=' * 60)
    print(f'Finished. Successfully processed: {success_count}/{len(srt_files)}.')
