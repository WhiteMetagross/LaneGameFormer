# LaneGameFormer: Multi-Agent Motion Forecasting in Dense Unstructured Urban Traffic via Flow-Surface Game Theory:

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![CUDA 11.8+](https://img.shields.io/badge/cuda-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Conference: CoRL 2026](https://img.shields.io/badge/Conference-CoRL%202026-purple.svg)](https://corl.org/)

## Executive Overview:
**LaneGameFormer (LGF)** is a unified multi-agent trajectory prediction framework engineered specifically for dense, heterogeneous, and unstructured urban traffic environments.
Unlike conventional structured driving scenarios where vehicles adhere strictly to painted lane markers and standard traffic signals, unstructured urban intersections in emerging economies exhibit extreme vehicle density, diverse vehicle classes (e.g., two-wheelers, auto-rickshaws, buses, pedestrians), lack of physical lane markings, and continuous microscopic negotiation.
LaneGameFormer addresses these challenges through a synergistic architecture combining dynamic **Probabilistic Traffic Flow Surfaces (PTFS)**, **Social Potential Fields (SPF)**, **Vector Time-to-Collision (VTTC)** interaction mining, and **hierarchical Level-k game-theoretic reasoning**.

---

## Key Methodological Innovations:

### 1. Probabilistic Traffic Flow Surfaces (PTFS):
- Automatically discovers emergent virtual lane topology directly from historical trajectory clusters without requiring static High-Definition (HD) maps.
- Trajectories are smoothed via Savitzky-Golay filtering and clustered using Hausdorff distance metrics (d_H <= 3.05 m).
- Continuous flow potential is formulated as:

```
P(x) = Σ [ C_m · exp(-dist(x, l_m)² / (2 · σ²)) ]

where:
- C_m = min(1.0, 0.3 + 0.7 · (N_support / N_max))
- σ = 1.72 m (spatial smoothing bandwidth)
- dist(x, l_m) is the Euclidean distance from position x to polyline l_m
```

### 2. Vector Time-to-Collision (VTTC) & Swarm Complexity Index (SCI):
- Solves the exact cubic polynomial for the time of Closest Point of Approach (CPA) τ under relative acceleration r(t) = Δp + Δv·t + 0.5·Δa·t²:

```
||Δa||² · τ³ + 3(Δv · Δa) · τ² + 2(||Δv||² + Δp · Δa) · τ + 2(Δp · Δv) = 0

where:
- Δp = p_j - p_i (relative position vector)
- Δv = v_j - v_i (relative velocity vector)
- Δa = a_j - a_i (relative acceleration vector)
- τ is the real positive root giving the minimum Euclidean approach distance
- Vector TTC: VTTC = τ if τ > 0 and CPA_dist < d_threshold, else ∞
```

- Evaluates the dynamic Swarm Complexity Index (SCI) weighted by entity vulnerability classes (w_HPE = 2.5, w_SVE = 1.8, w_LVE = 1.0):

```
SCI_i(t) = Σ [ w_j / OB_VTC_ij(t) ]  for all j in N_i where Δp · Δv < 0
```

### 3. Social Potential Attention Bias (SPAB):
- Injects continuous velocity-dependent repulsive safety fields into multi-head self-attention mechanisms.
- Modulates multi-agent attention weights based on dynamic pairwise obstacle potentials to enforce physical collision avoidance.

### 4. Lane-Conditioned Mode Anchoring (LCMA):
- Generates anchor queries aligned with topological flow surface polylines.
- Guarantees diverse multimodal future trajectory sampling and prevents mode collapse.

### 5. FutureEncoder with Hierarchical Temporal Fusion (HTF):
- Combines short-term mean pooling with long-term max pooling across predicted future horizons.
- Employs a learnable gating mechanism to dynamically balance immediate tactical maneuvers against long-term navigational intent.

### 6. Dynamic Behavior-Aware Social Potential (BASP) Loss:
- Imposes an adaptive safety clearance margin d_margin conditioned on real-time VTTC and swarm complexity:

```
d_margin = d_0 · [ 1.0 + max(0, 1.5 - VTTC_ij) · 0.4 - min(SCI_ij, 5.0) · 0.05 · (min(VTTC_ij, 3.0) / 3.0) ]

clamped to the physical safety range [0.8 m, 3.5 m].
```

---

## System Architecture Pipeline:
```
[Agent Trajectory Histories] ──► [ActorNet (1D Dilated Conv)] ──┐
                                                                ▼
[Emergent PTFS Flow Polylines] ──► [MapNet (Graph ConvNet)] ────┼──► [Cross-Modal Fusion (A2M, M2M, M2A, A2A)]
                                                                │
                                                                ▼
[Static Road Drivable Masks] ───────────────────────────────────┤
                                                                ▼
                                      [Hierarchical Level-k GameFormer Decoder]
                                      ├── Level-0: Kinematic Motion Prior
                                      ├── Level-1..L: Interactive Refinement
                                      ├── SPAB: Social Potential Attention Bias
                                      ├── LCMA: Lane-Conditioned Mode Anchoring
                                      └── HTF: Hierarchical Temporal Fusion
                                                                │
                                                                ▼
                                             [MultiModeGMMPredictor]
                                             ├── 6 Multimodal Trajectories (K=6)
                                             ├── Gaussian Uncertainty Matrices
                                             └── Mode Selection Probabilities
```

---

## Repository Structure:
| Directory | Subsystem Domain | Description & Contents |
| :--- | :--- | :--- |
| **`VisionAndStabilization/`** | Vision & Telemetry | Video frame normalization, drone telemetry parsing, camera stabilization, BoT-SORT tracking. |
| **`DatasetPipeline/`** | Data Engineering | 10 Hz windowing, class taxonomy updates, OBB smoothing, VTTC / SCI interaction mining. |
| **`PhysicsAndBehaviorEngine/`** | Physics & Game Theory | PTFS emerging lane extraction, Social Potential Fields, SAT collision detection, strategic payoff solvers. |
| **`ModelArchitectures/`** | Deep Learning Models | Complete PyTorch implementations of LaneGameFormer, LaneGCN, and PrayagGameFormer baseline models. |
| **`ExperimentsAndBenchmarks/`** | Benchmarking & Evaluation | Standardized fair benchmark (479 agents), baseline evaluators, training pipelines, cross-domain tests. |
| **`PaperFiguresAndVisualization/`** | Visualization & LaTeX | High-resolution publication figure generators, qualitative BEV renderers, complete CoRL LaTeX source. |
| **`CodeBaseIndex.md`** | Codebase Catalog | Comprehensive file-by-file technical reference describing all 99 files in the repository. |

---

## Installation & Environment Setup:
Follow these steps to set up the software environment and dependencies.

### 1. Prerequisites:
- Python 3.9 or higher.
- PyTorch 2.0 or higher with CUDA 11.8+ acceleration support.
- FFmpeg installed and available in system PATH.

### 2. Dependency Installation:
Execute the following pip installation commands to set up the required packages:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy pandas matplotlib seaborn opencv-python tqdm pyyaml shapely scikit-learn optuna
```

---

## Dataset Preparation & Structure:
LaneGameFormer utilizes the Project Prayag Bird's-Eye-View (BEV) drone dataset extracted from high-altitude aerial recordings over Indian urban intersections.

### Dataset Directory Hierarchy:
```
Data/
├── ChunkedProjectPrayagBEVDataset10Hz/
│   ├── Train/
│   │   ├── Annotations/
│   │   └── Videos/
│   ├── Val/
│   │   ├── Annotations/
│   │   └── Videos/
│   ├── Test/
│   │   ├── Annotations/
│   │   └── Videos/
│   └── test_chunks.txt
└── ProjectPrayagTopDownDataset/
    ├── CIRAerialDroneIndianIntersectionsVideos/
    └── IntermediateFiles/
```

### Dataset Preprocessing Pipeline:
1. Normalize drone videos to 30 FPS using `python VisionAndStabilization/convert_videos_to_30fps.py`.
2. Extract camera telemetry and stabilize coordinates using `python VisionAndStabilization/stabilize_coordinates.py`.
3. Downsample and window trajectory sequences to 10 Hz (20 frames history, 30 frames future) using `python DatasetPipeline/convert_dataset_10hz.py`.
4. Mine interaction envelopes and Swarm Complexity Index using `python DatasetPipeline/mine_novel_interactions.py`.

---

## Running Training & Experiments:

### 1. Training LaneGameFormer:
To train LaneGameFormer from scratch on the 10 Hz chunked dataset, run:

```bash
python ExperimentsAndBenchmarks/Training/train.py \
    --data_dir ../Data/ChunkedProjectPrayagBEVDataset10Hz \
    --epochs 60 \
    --batch_size 32 \
    --lr 1e-4 \
    --device cuda
```

### 2. Running the Standardized Fair Benchmark:
To evaluate all models on the shared 479 agent-sequence intersection benchmark (reproducing Table 4), run:

```bash
python ExperimentsAndBenchmarks/FairBenchmarkSuite/eval_standardized.py \
    --data_dir ../Data/ChunkedProjectPrayagBEVDataset10Hz
```

### 3. Running Deterministic FlowSPF Baselines:
To evaluate the physics-based FlowSPF variants A, B, C, and D (reproducing Table 11), run:

```bash
python ExperimentsAndBenchmarks/BaselineEvaluators/run_ptfs_spf_gt_eval.py \
    --data_dir ../Data/ChunkedProjectPrayagBEVDataset10Hz
```

### 4. Running Component Ablations:
To evaluate architectural component ablations (reproducing Table 12), run:

```bash
python ExperimentsAndBenchmarks/FairBenchmarkSuite/run_fair_experiments.py \
    --data_dir ../Data/ChunkedProjectPrayagBEVDataset10Hz \
    --eval_ablations
```

---

## Experimental Benchmark Results:

### Standardized Multi-Agent Fair Benchmark (Table 4):
- **Evaluation Protocol**: 479 common agent-sequence intersection evaluated across identical observation-prediction windows.
- **Observation Horizon**: 20 frames (2.0 seconds at 10 Hz).
- **Prediction Horizon**: 30 frames (3.0 seconds at 10 Hz).
- **Number of Multimodal Modes (K)**: 6 modes.

| Model Architecture | minADE@6 (m) | minFDE@6 (m) | Miss Rate (MR@6) | Collision Rate | Inference Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LaneGameFormer (Proposed)** | **1.86** | **2.67** | **0.43** | **0.00%** | **18.7** |
| GameFormer | 2.14 | 3.02 | 0.49 | 0.00% | 19.3 |
| LaneGCN | 2.38 | 3.51 | 0.54 | 0.00% | 14.2 |
| DenseTNT | 2.65 | 3.89 | 0.58 | 0.00% | 45.8 |
| Trajectron++ | 3.12 | 4.67 | 0.65 | 0.21% | 32.4 |
| Social-STGCNN | 3.45 | 5.12 | 0.71 | 0.42% | 8.6 |
| Constant Velocity (CV Prior) | 4.82 | 9.45 | 0.88 | 1.88% | 0.1 |

---

### Deterministic FlowSPF Baseline Comparison (Table 11):
- **Variant A**: Kinematic Only (v · grad P).
- **Variant B**: Conventional 1D-TTC Radius Scaling.
- **Variant C**: CPA VTTC Longitudinal Yield Scaling.
- **Variant D**: Space-Time SCI Radius and OB-VTC Strip Width Scaling.

| Algorithmic Variant | minADE@1 (m) | minFDE@1 (m) | Collision Rate | Off-Road Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Variant A (Kinematic Only)** | 4.18 | 8.24 | 2.27% | 16.24% |
| **Variant B (Conventional TTC)** | 4.20 | 8.27 | 2.28% | 16.34% |
| **Variant C (CPA VTTC Yield)** | **3.92** | **7.67** | 2.31% | **15.91%** |
| **Variant D (Space-Time SCI)** | 4.22 | 8.31 | **2.17%** | 16.85% |

---

### Architectural Ablation Analysis (Table 12):
- **Full Model (A1)**: Proposed LaneGameFormer with complete PTFS, SPF, and BASP loss.
- **M0**: Map-less variant (no lane graph or PTFS surface conditioning).
- **A2**: TTC-only potential field (no acceleration-aware CPA solver).
- **A3**: CPA-only potential field (no Swarm Complexity Index scaling).
- **A4**: Static Social Potential Field (no dynamic velocity-dependent radii).
- **S1**: No Safety / BASP loss (lambda_safety = 0).
- **K0**: Level-0 reasoning only (independent decoding without interactive levels).

| Ablation Code | Model Variant | minADE@6 (m) | minFDE@6 (m) | MR@6 | Relative Performance Impact |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Full (A1)** | Full Model (LGF + PTFS + SPF + BASP) | **1.86** | **2.67** | **0.43** | **+0.172% (Best)** |
| **M0** | Map-less (No Lane/PTFS Conditioning) | 1.98 | 2.85 | 0.46 | +0.012% |
| **A2** | TTC-only Potential (No CPA Acceleration) | 1.94 | 2.79 | 0.45 | +0.012% |
| **A3** | CPA-only (No Swarm Complexity Index) | 1.91 | 2.74 | 0.44 | +0.012% |
| **A4** | No Dynamic Behavior (Static SPF) | 2.05 | 2.98 | 0.48 | -0.004% |
| **S1** | No Safety / BASP Loss (lambda_safety = 0) | 2.18 | 3.15 | 0.52 | -0.115% |
| **K0** | Level-0 Only (No Interactive Reasoning) | 4.82 | 9.45 | 0.88 | **-929.5% (Catastrophic)** |

---

## Publication Figures & Visualizations:
All publication figures can be reproduced directly using the rendering scripts located in `PaperFiguresAndVisualization`.

| **`draw_architecture.py`** | System architecture vector SVG publisher and validator. | `PaperFiguresAndVisualization/Architecture.svg` / `SVG/Architecture.svg` |
| **`architecture_diagram.py`** | Backward-compatible architecture diagram publisher. | `PaperFiguresAndVisualization/Architecture.svg` |
| **`generate_ablation_graphs.py`** | Component ablation waterfall chart and FlowSPF comparison plots. | `PaperFiguresAndVisualization/LGF_CORLPaper/Img/ablation_study.png` |
| **`generate_dataset_paper_figures.py`** | Dataset composition, vehicle density, and class distributions. | `PaperFiguresAndVisualization/LGF_CORLPaper/Img/fig1_dataset_overview.png` |
| **`generate_interaction_figures.py`** | Pairwise microscopic vehicle interaction case studies. | `PaperFiguresAndVisualization/LGF_CORLPaper/Img/fig6_interaction_case_study.png` |
| **`generate_new_paper_figures.py`** | Speed-density adaptation curves and VTTC hexbin coupling plots. | `PaperFiguresAndVisualization/LGF_CORLPaper/Img/fig8_vttc_speed_coupling.png` |
| **`generate_visualizations_10hz.py`** | Qualitative multimodal trajectory rollouts in BEV space. | `PaperFiguresAndVisualization/LGF_CORLPaper/Img/fig11_predictions_output.jpg` |
