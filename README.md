# LaneGameFormer: A Unified Motion Forecasting Framework for Dense Unstructured Traffic:

## Project Overview:
LaneGameFormer (LGF) is a motion forecasting framework designed for dense, heterogeneous, and unstructured traffic environments.
Unlike conventional structured highways where vehicles adhere strictly to marked lanes and standardized traffic rules, unstructured urban intersections in emerging economies exhibit high agent density, diverse vehicle classes, lack of physical lane markings, and continuous microscopic negotiation.
LaneGameFormer addresses these challenges through a synergistic architecture combining dynamic Probabilistic Traffic Flow Surfaces (PTFS), Social Potential Fields (SPF), Vector Time-to-Collision (VTTC) interaction mining, and hierarchical Level-$k$ game-theoretic reasoning.

---

## Key Methodological Highlights:
- **Probabilistic Traffic Flow Surfaces (PTFS)**: Discovers emergent virtual lane structures directly from unstructured trajectory clusters without requiring static High-Definition (HD) maps.
- **Vector Time-to-Collision (VTTC) & Swarm Complexity Index (SCI)**: Solves the exact cubic polynomial for Closest Point of Approach (CPA) under acceleration to quantify interaction urgency across heterogeneous agent classes.
- **Social Potential Attention Bias (SPAB)**: Injects continuous velocity-dependent repulsive safety fields into multi-head self-attention mechanisms to enforce physical collision avoidance.
- **Lane-Conditioned Mode Anchoring (LCMA)**: Anchors multimodal prediction queries to topological flow polylines to prevent mode collapse.
- **Hierarchical Temporal Fusion (HTF)**: Fuses short-term and long-term future temporal encodings through learnable gated embeddings during Level-$k$ interactive reasoning.
- **Behavior-Aware Social Potential (BASP) Loss**: Imposes a dynamic safety clearance margin conditioned on real-time VTTC and swarm complexity.

---

## Repository Structure:
- `VisionAndStabilization`: Contains video stabilization, telemetry parsing, coordinate calibration, and BoT-SORT multi-object tracking pipelines.
- `DatasetPipeline`: Contains dataset chunking, frequency resampling to 10 Hz, class taxonomy mapping, and interaction mining routines.
- `PhysicsAndBehaviorEngine`: Implements emerging lane discovery, social potential fields, 2D-OBB Separating Axis Theorem (SAT) collision detection, and game-theoretic payoff modeling.
- `ModelArchitectures`: Contains PyTorch implementations of LaneGameFormer, LaneGCN, and PrayagGameFormer baseline models.
- `ExperimentsAndBenchmarks`: Contains standardized fair evaluation suites, baseline evaluators, multi-seed training scripts, and zero-shot cross-domain generalization benchmarks.
- `PaperFiguresAndVisualization`: Contains visualization scripts, figure rendering pipelines, and the complete LaTeX research paper manuscript.
- `CodeBaseIndex.md`: Complete directory-by-directory and file-by-file technical index.

---

## Installation & Environment Setup:
Follow these steps to set up the software environment and dependencies.

### 1. Prerequisites:
- Python 3.9 or higher.
- PyTorch 2.0 or higher with CUDA acceleration support.
- FFmpeg installed and available in system PATH.

### 2. Dependency Installation:
Execute the following pip installation commands to set up the required packages:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy pandas matplotlib seaborn opencv-python tqdm pyyaml shapely scikit-learn optuna
```.

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
```.

### Dataset Preprocessing Pipeline:
1. Normalize drone videos to 30 FPS using `VisionAndStabilization/convert_videos_to_30fps.py`.
2. Extract camera telemetry and stabilize coordinates using `VisionAndStabilization/stabilize_coordinates.py`.
3. Downsample and window trajectory sequences to 10 Hz (20 frames history, 30 frames future) using `DatasetPipeline/convert_dataset_10hz.py`.
4. Mine interaction envelopes and Swarm Complexity Index using `DatasetPipeline/mine_novel_interactions.py`.

---

## Model Architecture Details:
The LaneGameFormer architecture consists of an integrated encoder-decoder pipeline.

### 1. LaneGCN Encoder:
- **ActorNet**: Processes multi-agent past trajectory sequences using multi-scale 1D dilated convolutions.
- **MapNet**: Processes emergent traffic flow polylines using spatial graph convolutions.
- **Cross-Modal Fusion**: Performs multi-scale information sharing across Actor-to-Map (A2M), Map-to-Map (M2M), Map-to-Actor (M2A), and Actor-to-Actor (A2A) fusion layers.

### 2. GameFormer Interactive Decoder:
- **Level-0 Prior**: Computes kinematic trajectory priors from fused agent and map representations.
- **Level-1 to Level-L Interaction Decoding**: Iteratively refines multi-agent predictions through game-theoretic cross-attention layers.
- **Social Potential Attention Bias (SPAB)**: Modulates attention weights based on dynamic pairwise obstacle potentials.
- **MultiModeGMMPredictor**: Predicts 6 multimodal trajectory modes parametrized as Gaussian Mixture Models with learned mode probabilities.

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
```.

### 2. Running the Standardized Fair Benchmark:
To evaluate all models on the shared 479 agent-sequence intersection benchmark (reproducing Table 4), run:
```bash
python ExperimentsAndBenchmarks/FairBenchmarkSuite/eval_standardized.py \
    --data_dir ../Data/ChunkedProjectPrayagBEVDataset10Hz
```.

### 3. Running Deterministic FlowSPF Baselines:
To evaluate the physics-based FlowSPF variants A, B, C, and D (reproducing Table 11), run:
```bash
python ExperimentsAndBenchmarks/BaselineEvaluators/run_ptfs_spf_gt_eval.py \
    --data_dir ../Data/ChunkedProjectPrayagBEVDataset10Hz
```.

### 4. Running Component Ablations:
To evaluate architectural component ablations (reproducing Table 12), run:
```bash
python ExperimentsAndBenchmarks/FairBenchmarkSuite/run_fair_experiments.py \
    --data_dir ../Data/ChunkedProjectPrayagBEVDataset10Hz \
    --eval_ablations
```.

---

## Experimental Benchmark Results:

### Standardized Multi-Agent Fair Benchmark (Table 4):
- **Evaluation Subset**: 479 shared agent-sequence intersection samples evaluated across identical observation-prediction windows.
- **Observation Horizon**: 20 frames (2.0 seconds at 10 Hz).
- **Prediction Horizon**: 30 frames (3.0 seconds at 10 Hz).
- **Number of Multimodal Modes ($K$)**: 6 modes.

| Model Architecture | minADE@6 (m) | minFDE@6 (m) | Miss Rate (MR@6) | Collision Rate | Inference Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LaneGameFormer (Proposed)** | **1.86** | **2.67** | **0.43** | **0.00%** | **18.7** |
| GameFormer | 2.14 | 3.02 | 0.49 | 0.00% | 19.3 |
| LaneGCN | 2.38 | 3.51 | 0.54 | 0.00% | 14.2 |
| DenseTNT | 2.65 | 3.89 | 0.58 | 0.00% | 45.8 |
| Trajectron++ | 3.12 | 4.67 | 0.65 | 0.21% | 32.4 |
| Social-STGCNN | 3.45 | 5.12 | 0.71 | 0.42% | 8.6 |
| Constant Velocity (CV Prior) | 4.82 | 9.45 | 0.88 | 1.88% | 0.1 |

.

---

### Deterministic FlowSPF Baseline Comparison (Table 11):
- **Variant A**: Kinematic Only ($\mathbf{v} \odot \nabla P$).
- **Variant B**: Conventional 1D-TTC Radius Scaling.
- **Variant C**: CPA VTTC Longitudinal Yield Scaling.
- **Variant D**: Space-Time SCI Radius and OB-VTC Strip Width Scaling.

| Algorithmic Variant | minADE@1 (m) | minFDE@1 (m) | Collision Rate | Off-Road Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Variant A (Kinematic Only)** | 4.18 | 8.24 | 2.27% | 16.24% |
| **Variant B (Conventional TTC)** | 4.20 | 8.27 | 2.28% | 16.34% |
| **Variant C (CPA VTTC Yield)** | **3.92** | **7.67** | 2.31% | **15.91%** |
| **Variant D (Space-Time SCI)** | 4.22 | 8.31 | **2.17%** | 16.85% |

.

---

### Architectural Ablation Analysis (Table 12):
- **Full Model (A1)**: Proposed LaneGameFormer with complete PTFS, SPF, and BASP loss.
- **M0**: Map-less variant (no lane graph or PTFS surface conditioning).
- **A2**: TTC-only potential field (no acceleration-aware CPA solver).
- **A3**: CPA-only potential field (no Swarm Complexity Index scaling).
- **A4**: Static Social Potential Field (no dynamic velocity-dependent radii).
- **S1**: No Safety / BASP loss ($\lambda_{\text{safety}} = 0$).
- **K0**: Level-0 reasoning only (independent decoding without interactive levels).

| Ablation Code | Model Variant | minADE@6 (m) | minFDE@6 (m) | MR@6 | Relative Performance Impact |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Full (A1)** | Full Model (LGF + PTFS + SPF + BASP) | **1.86** | **2.67** | **0.43** | **+0.172% (Best)** |
| **M0** | Map-less (No Lane/PTFS Conditioning) | 1.98 | 2.85 | 0.46 | +0.012% |
| **A2** | TTC-only Potential (No CPA Acceleration) | 1.94 | 2.79 | 0.45 | +0.012% |
| **A3** | CPA-only (No Swarm Complexity Index) | 1.91 | 2.74 | 0.44 | +0.012% |
| **A4** | No Dynamic Behavior (Static SPF) | 2.05 | 2.98 | 0.48 | -0.004% |
| **S1** | No Safety / BASP Loss ($\lambda_{\text{safety}} = 0$) | 2.18 | 3.15 | 0.52 | -0.115% |
| **K0** | Level-0 Only (No Interactive Reasoning) | 4.82 | 9.45 | 0.88 | **-929.5% (Catastrophic)** |

.

---

## Generating Figures & Visualizations:
All figures presented in the research paper can be generated using the scripts in `PaperFiguresAndVisualization`.

### Execution Commands:
- To generate the system architecture diagram, run `python PaperFiguresAndVisualization/draw_architecture.py`.
- To generate the ablation study and FlowSPF comparison graphs, run `python PaperFiguresAndVisualization/generate_ablation_graphs.py`.
- To generate dataset distribution and density figures, run `python PaperFiguresAndVisualization/generate_dataset_paper_figures.py`.
- To generate microscopic interaction case studies, run `python PaperFiguresAndVisualization/generate_interaction_figures.py`.
- To generate 10 Hz trajectory stream overlays, run `python PaperFiguresAndVisualization/generate_visualizations_10hz.py`.

---

## Citation:
If you find this codebase or research useful in your work, please cite the research paper as follows:
```bibtex
@inproceedings{mandal2026lanegameformer,
  title={LaneGameFormer: Multi-Agent Motion Forecasting in Dense Unstructured Urban Traffic via Flow-Surface Game Theory},
  author={Mandal, Mridankan and Collaborators},
  booktitle={Proceedings of the Conference on Robot Learning (CoRL)},
  year={2026}
}
```.
