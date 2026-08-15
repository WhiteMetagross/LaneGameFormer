#This file implements the BoTSORT algorithm for multi-object tracking with ReID.

import numpy as np
from collections import OrderedDict
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

import fastreid
from fastreid.config import get_cfg
from fastreid.modeling import build_model
from fastreid.utils.checkpoint import Checkpointer

#This class uses Kalman Filter to track bounding boxes over time.
class KalmanBoxTracker:
    
    count = 0

    #This function initializes a tracker using an initial bounding box.
    def __init__(self, bbox, feature=None):
        
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
        self.features = []
        if feature is not None:
            self.features.append(feature)

    #This function updates the state vector with an observed bounding box.
    def update(self, bbox, feature=None):
        
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox_to_z(bbox))
        
        if feature is not None:
            self.features.append(feature)
            if len(self.features) > 100:
                self.features = self.features[-100:]

    #This function advances the state vector and returns the predicted bounding box estimate.
    def predict(self):
        
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    #This function returns the current bounding box estimate.
    def get_state(self):
        return self._convert_x_to_bbox(self.kf.x)
    
    #This function returns the most recent feature.
    def get_feature(self):
        if self.features:
            return self.features[-1]
        return None

    #This function converts bounding box from [x1,y1,x2,y2] to [x,y,s,r] format.
    @staticmethod
    def _convert_bbox_to_z(bbox):
        
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    #This function converts bounding box from [x,y,s,r] to [x1,y1,x2,y2] format.
    @staticmethod
    def _convert_x_to_bbox(x, score=None):
        
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if score is None:
            return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
        else:
            return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))

#This class extracts ReID features using a pretrained VeRiWild model.
class FastReIDFeatureExtractor:
    
    #This function initializes the feature extractor with model and config paths.
    def __init__(self, model_path, config_path, device='cuda'):
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_path)
        self.cfg.MODEL.DEVICE = self.device
        self.cfg.freeze()
        
        self.model = build_model(self.cfg)
        self.model.eval()
        self.model = self.model.to(self.device)
        
        checkpointer = Checkpointer(self.model)
        checkpointer.load(model_path)
        
        print(f"[FastReID] Loaded VeRiWild model on {self.device}")
    
    #This function extracts features from a list of image crops.
    def extract_features(self, crops):
        
        if not crops:
            return np.empty((0, 2048), dtype=np.float32)
        
        processed_crops = []
        for crop in crops:
            import cv2
            crop_resized = cv2.resize(crop, (128, 256))
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
            
            crop_tensor = torch.from_numpy(crop_rgb.transpose(2, 0, 1)).float()
            processed_crops.append(crop_tensor)
        
        batch = torch.stack(processed_crops).to(self.device)
        
        with torch.no_grad():
            features = self.model(batch)
            features = F.normalize(features, p=2, dim=1)
        
        return features.cpu().numpy()

#This function computes the Intersection over Union between two sets of bounding boxes.
def iou_batch(bb_test, bb_gt):
    
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o

#This function associates detections to existing trackers using IoU and ReID features.
def associate_detections_to_trackers(detections, trackers, reid_features=None, track_features=None, 
                                   iou_threshold=0.3, reid_threshold=0.7):
    
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)
    
    if reid_features is not None and track_features is not None:
        reid_sim = np.dot(reid_features, track_features.T)
        cost_matrix = 1 - (0.7 * iou_matrix + 0.3 * reid_sim)
        print(f"[Association] Using IoU + ReID: IoU shape {iou_matrix.shape}, ReID shape {reid_sim.shape}")
    else:
        cost_matrix = 1 - iou_matrix
        print(f"[Association] Using IoU only: shape {iou_matrix.shape}")

    if min(cost_matrix.shape) > 0:
        a = (cost_matrix > 1 - iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_sum_assignment(cost_matrix)
            matched_indices = np.array(list(zip(*matched_indices)))
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if cost_matrix[m[0], m[1]] > 1 - iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

#This class integrates detection tracking and ReID feature extraction.
class BoTSORT:
    
    #This function initializes the BoTSORT tracker with parameters and ReID model.
    def __init__(self, reid_model_path, reid_config_path, max_disappeared=30, min_hits=3, 
                 iou_threshold=0.3, device='cuda'):
        
        self.max_disappeared = max_disappeared
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.reid_extractor = FastReIDFeatureExtractor(
            reid_model_path, reid_config_path, device
        )
        
        self.trackers = []
        self.frame_count = 0
    
    #This function updates the tracker with detections and the current frame for ReID.
    def update(self, detections, frame=None):
        
        self.frame_count += 1
        
        reid_features = None
        if frame is not None and detections:
            crops = self._extract_crops(frame, detections)
            reid_features = self.reid_extractor.extract_features(crops)
            print(f"[BoTSORT] Frame {self.frame_count}: Extracted {len(reid_features)} ReID features for {len(detections)} detections")
        
        trks = np.zeros((len(self.trackers), 5))
        track_features = []
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                track_features.append(self.trackers[t].get_feature())
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
            track_features.pop(t)
        
        if track_features and all(f is not None for f in track_features):
            track_features = np.array(track_features)
            print(f"[BoTSORT] Using {len(track_features)} track features for association")
        else:
            track_features = None
            print(f"[BoTSORT] No track features available - using IoU only")
        
        if detections:
            dets = np.array([[d[0], d[1], d[2], d[3], d[4]] for d in detections])
        else:
            dets = np.empty((0, 5))
            
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, reid_features, track_features, self.iou_threshold
        )
        
        for m in matched:
            feature = reid_features[m[0]] if reid_features is not None else None
            self.trackers[m[1]].update(dets[m[0], :4], feature)
        
        for i in unmatched_dets:
            feature = reid_features[i] if reid_features is not None else None
            trk = KalmanBoxTracker(dets[i, :4], feature)
            self.trackers.append(trk)
        
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_disappeared:
                self.trackers.pop(i)
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
    
    #This function extracts image crops from the frame for each detection.
    def _extract_crops(self, frame, detections):
        
        crops = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
            else:
                crops.append(np.zeros((64, 64, 3), dtype=np.uint8))
        return crops