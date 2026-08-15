#This file performs object tracking on videos using YOLO-OBB models with BoTSORT and ReID.

import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import config
import utils
from botsort import BoTSORT

#This function extracts tracking data from a video file using a three-model cascade detection pipeline.
def extract_tracking_data(video_path, scene_id, road_mask_filter=None):
    # Load all three models
    if not os.path.exists(config.YOLO_OBB_MODEL_1):
        print(f"  Error: Primary OBB model not found at {config.YOLO_OBB_MODEL_1}")
        return None
    if not os.path.exists(config.YOLO_BBOX_MODEL):
        print(f"  Error: Classification model not found at {config.YOLO_BBOX_MODEL}")
        return None
    if not os.path.exists(config.YOLO_OBB_MODEL_2):
        print(f"  Error: Validation model not found at {config.YOLO_OBB_MODEL_2}")
        return None

    model_obb1 = YOLO(config.YOLO_OBB_MODEL_1)  # Primary OBB detection
    model_bbox = YOLO(config.YOLO_BBOX_MODEL)   # Class classification
    model_obb2 = YOLO(config.YOLO_OBB_MODEL_2)  # Validation
    
    print(f"  Loaded three-model cascade: OBB1 -> BBox -> OBB2")
    
    # Initialize BoTSORT tracker with ReID
    try:
        tracker = BoTSORT(
            reid_model_path=config.REID_MODEL_PATH,
            reid_config_path=config.REID_CONFIG_PATH,
            max_disappeared=config.BOTSORT_MAX_DISAPPEARED,
            min_hits=config.BOTSORT_MIN_HITS,
            iou_threshold=config.BOTSORT_IOU_THRESHOLD,
            device=config.DEVICE
        )
        print(f"  BoTSORT tracker initialized with ReID")
    except Exception as e:
        print(f"  Error initializing BoTSORT tracker: {e}")
        return None

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Error: Could not open video file {video_path}")
        return None
    
    # Get original FPS and check if conversion is needed
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = 30.0
    frame_skip = 1
    
    if abs(original_fps - target_fps) > 0.1:
        frame_skip = max(1, int(round(original_fps / target_fps)))
        print(f"  Converting from {original_fps:.2f} FPS to ~{target_fps} FPS (processing every {frame_skip} frame(s))")
    else:
        print(f"  Video already at {original_fps:.2f} FPS")
    
    per_frame_data = defaultdict(dict)
    frame_count = 0
    processed_frame_count = 0
    total_detections = 0
    total_filtered = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            processed_frame_count += 1
            
            # Step 1: Run primary OBB detection (RoadVehiclesYOLOOBB.pt)
            results_obb1 = model_obb1(frame, verbose=False, conf=config.YOLO_CONF)
            
            # Step 2: Run classification model (RoadVehiclesYOLO11m.pt)
            results_bbox = model_bbox(frame, verbose=False, conf=config.YOLO_CONF)
            
            # Step 3: Run validation model (PrayagProjectBEVYOLO11OBB.pt)
            results_obb2 = model_obb2(frame, verbose=False, conf=config.YOLO_CONF)
            
            # Collect all detections from all models
            all_detections = []
            
            # Process Model 1: RoadVehiclesYOLOOBB.pt (all class 1/HVE)
            for result_obb1 in results_obb1:
                if hasattr(result_obb1, 'obb') and result_obb1.obb is not None:
                    for box in result_obb1.obb:
                        corners = utils.extract_obb_corners(box)
                        if road_mask_filter is not None:
                            if not road_mask_filter.filter_detection(corners):
                                total_filtered += 1
                                continue
                        
                        corners_array = np.array(corners)
                        x1 = corners_array[:, 0].min()
                        y1 = corners_array[:, 1].min()
                        x2 = corners_array[:, 0].max()
                        y2 = corners_array[:, 1].max()
                        conf = box.conf[0].cpu().numpy()
                        
                        all_detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'obb': corners,
                            'conf': conf,
                            'class': 1,  # HVE
                            'model': 1
                        })
            
            # Process Model 2: RoadVehiclesYOLO11m.pt (RBB with class mapping)
            for result_bbox in results_bbox:
                if hasattr(result_bbox, 'boxes') and result_bbox.boxes is not None:
                    for box in result_bbox.boxes:
                        bbox_coords = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = bbox_coords
                        conf = box.conf[0].cpu().numpy()
                        bbox_class = int(box.cls[0].cpu().numpy())
                        
                        # Convert RBB to OBB (simple rectangle)
                        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                        
                        if road_mask_filter is not None:
                            if not road_mask_filter.filter_detection(corners):
                                total_filtered += 1
                                continue
                        
                        # Map classes: 0->2(LVE), 1->1(HVE), 2->0(HPE)
                        if bbox_class == 0:
                            final_class = 2  # LVE
                        elif bbox_class == 1:
                            final_class = 1  # HVE
                        elif bbox_class == 2:
                            final_class = 0  # HPE
                        else:
                            final_class = 1  # Default HVE
                        
                        all_detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'obb': corners,
                            'conf': conf,
                            'class': final_class,
                            'model': 2
                        })
            
            # Process Model 3: PrayagProjectBEVYOLO11OBB.pt
            for result_obb2 in results_obb2:
                if hasattr(result_obb2, 'obb') and result_obb2.obb is not None:
                    for box in result_obb2.obb:
                        corners = utils.extract_obb_corners(box)
                        if road_mask_filter is not None:
                            if not road_mask_filter.filter_detection(corners):
                                total_filtered += 1
                                continue
                        
                        corners_array = np.array(corners)
                        x1 = corners_array[:, 0].min()
                        y1 = corners_array[:, 1].min()
                        x2 = corners_array[:, 0].max()
                        y2 = corners_array[:, 1].max()
                        conf = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        all_detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'obb': corners,
                            'conf': conf,
                            'class': class_id,
                            'model': 3
                        })
            
            # Deduplicate detections: merge overlapping detections (>=25% IoU)
            # Priority hierarchy: Model 1 > Model 2 > Model 3
            final_detections = []
            used = set()
            
            for i, det1 in enumerate(all_detections):
                if i in used:
                    continue
                
                # Find all detections that overlap with this one
                overlapping = [i]
                for j, det2 in enumerate(all_detections):
                    if j <= i or j in used:
                        continue
                    
                    iou = calculate_iou(det1['bbox'], det2['bbox'])
                    if iou >= config.MODEL_OVERLAP_THRESHOLD:
                        overlapping.append(j)
                
                # Keep detection based on model priority: 1 > 2 > 3
                best_idx = min(overlapping, key=lambda idx: all_detections[idx]['model'])
                best_det = all_detections[best_idx]
                
                final_detections.append(best_det)
                used.update(overlapping)
            
            # Convert to BoTSORT format
            detections = []
            obb_corners_list = []
            class_ids_list = []
            
            for det in final_detections:
                x1, y1, x2, y2 = det['bbox']
                detections.append([x1, y1, x2, y2, det['conf'], det['class']])
                obb_corners_list.append(det['obb'])
                class_ids_list.append(det['class'])
            
            # Update BoTSORT tracker
            if detections:
                tracks = tracker.update(detections, frame)
            else:
                tracks = tracker.update([], frame)
            
            # Store tracking results with OBB information
            for i, track in enumerate(tracks):
                if len(track) >= 5:
                    x1, y1, x2, y2, track_id = track[:5]
                    track_id = int(track_id)
                    
                    # Find the corresponding OBB corners
                    # Match track to detection by bbox IoU
                    best_match_idx = -1
                    best_iou = 0.0
                    track_bbox = [x1, y1, x2, y2]
                    
                    for det_idx, det in enumerate(detections):
                        det_bbox = det[:4]
                        iou = calculate_iou(track_bbox, det_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_match_idx = det_idx
                    
                    # Use matched OBB or convert bbox to OBB
                    if best_match_idx >= 0 and best_iou > 0.5:
                        corners = obb_corners_list[best_match_idx]
                        track_class_id = class_ids_list[best_match_idx]
                    else:
                        # Fallback: create OBB from bbox
                        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                        track_class_id = 1  # Default to HVE
                    
                    center = [np.mean([c[0] for c in corners]), np.mean([c[1] for c in corners])]
                    per_frame_data[processed_frame_count][track_id] = {
                        "center": [float(center[0]), float(center[1])],
                        "obb": [[float(c[0]), float(c[1])] for c in corners],
                        "class": int(track_class_id)
                    }
                    total_detections += 1
            
            if processed_frame_count % 100 == 0:
                print(f"  ... processed frame {processed_frame_count} (original frame {frame_count})")
    
    except Exception as e:
        print(f"  Error during tracking: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()

    print(f"  Finished processing {processed_frame_count} frames (from {frame_count} original frames).")
    print(f"  Total detections: {total_detections}, Filtered out: {total_filtered}")

    if not total_detections:
        print("  Diagnostic: No valid tracks were detected.")
        return None

    return per_frame_data

#This function calculates the Intersection over Union between two bounding boxes.
def calculate_iou(bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Calculate intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    inter_area = inter_width * inter_height
    
    # Calculate union
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area