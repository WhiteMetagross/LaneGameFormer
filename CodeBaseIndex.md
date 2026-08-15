# LaneGameFormer Codebase Technical Index:

## Overview & Architecture Index:
This document provides an exhaustive, structured index of all directories, subpackages, source files, configuration assets, and research artifacts in the **LaneGameFormer (LGF)** repository.
LaneGameFormer is a motion forecasting framework designed for dense, heterogeneous, and unstructured traffic environments.
The codebase is modularized into 6 major subpackages spanning video stabilization, dataset synthesis, physics engines, neural architectures, benchmarking suites, and paper visualization tools.

---

## Repository Module Quick-Reference:
| Module Directory | Primary Domain | Core Functions / Capabilities | File Count |
| :--- | :--- | :--- | :---: |
| **`VisionAndStabilization/`** | Vision & Telemetry | Video frame normalization, drone telemetry parsing, camera stabilization, BoT-SORT tracking. | 9 |
| **`DatasetPipeline/`** | Data Engineering | 10 Hz windowing, class taxonomy updates, OBB smoothing, VTTC / SCI interaction mining. | 12 |
| **`PhysicsAndBehaviorEngine/`** | Physics & Game Theory | PTFS emerging lane extraction, Social Potential Fields, SAT collision detection, strategic payoff solvers. | 7 |
| **`ModelArchitectures/`** | Deep Learning Models | LaneGameFormer (Proposed), LaneGCN, and PrayagGameFormer baseline architectures. | 21 |
| **`ExperimentsAndBenchmarks/`** | Benchmarking & Evaluation | Standardized fair benchmark (479 agents), baseline evaluators, training pipelines, cross-domain tests. | 17 |
| **`PaperFiguresAndVisualization/`** | Visualization & LaTeX | High-resolution publication figure generators, qualitative BEV renderers, complete CoRL LaTeX source. | 23 |

.

---

## Directory Hierarchy Tree:
```
Code/
├── VisionAndStabilization/
│   ├── botsort.py
│   ├── botsort.yaml
│   ├── convert_videos_to_30fps.py
│   ├── extract_srt_metadata.py
│   ├── map_pixels_to_meters.py
│   ├── stabilize_coordinates.py
│   ├── stabilize_video_files.py
│   ├── telemetry_aligner.py
│   └── tracker.py
├── DatasetPipeline/
│   ├── calculate_class_sizes.py
│   ├── chunk_dataset.py
│   ├── chunk_interactions.py
│   ├── cleanup_datasets.py
│   ├── convert_dataset_10hz.py
│   ├── create_manifest.py
│   ├── create_stratified_dataset.py
│   ├── interaction_miner.py
│   ├── mine_conventional_ttc.py
│   ├── mine_novel_interactions.py
│   ├── smooth_obb_trajectories.py
│   └── update_dataset_classes.py
├── PhysicsAndBehaviorEngine/
│   ├── emerging_lane_extractor.py
│   ├── game_theory_predictor.py
│   ├── lane_manager.py
│   ├── road_mask_manager.py
│   ├── social_potential_field.py
│   ├── track_data_loader.py
│   └── trajectory_predictor.py
├── ModelArchitectures/
│   ├── LaneGameFormer/
│   │   ├── gameformer_decoder.py
│   │   ├── lane_game_former.py
│   │   ├── lanegcn_encoder.py
│   │   └── layers.py
│   ├── LaneGCN/
│   │   ├── dataset_generator.py
│   │   ├── graph_processor.py
│   │   ├── lanegcn.py
│   │   └── layers.py
│   └── GameFormer/
│       └── PrayagGameFormer/
│           ├── Checkpoints/
│           │   └── MapCache/
│           ├── Data/
│           │   ├── dataset.py
│           │   └── mapEncoder.py
│           ├── Model/
│           │   ├── gameformer.py
│           │   └── modules.py
│           ├── Utils/
│           │   ├── helpers.py
│           │   ├── metrics.py
│           │   └── training_utils.py
│           ├── config.py
│           ├── evaluate.py
│           ├── requirements.txt
│           ├── train.py
│           └── tune.py
├── ExperimentsAndBenchmarks/
│   ├── BaselineEvaluators/
│   │   ├── eval_cv_baseline.py
│   │   ├── eval_fixed_cr.py
│   │   ├── eval_trivial_baselines.py
│   │   ├── evaluate_all_models.py
│   │   └── run_ptfs_spf_gt_eval.py
│   ├── FairBenchmarkSuite/
│   │   ├── eval_all_agents.py
│   │   ├── eval_standardized.py
│   │   ├── evaluate_system.py
│   │   ├── evaluation_metrics.py
│   │   └── run_fair_experiments.py
│   ├── Training/
│   │   ├── inference.py
│   │   ├── shared_training_utils.py
│   │   └── train.py
│   └── ZeroShotCrossDomain/
│       ├── compare_models.py
│       ├── test_models.py
│       └── test_models_on_largedatasets.py
└── PaperFiguresAndVisualization/
    ├── draw_architecture.py
    ├── generate_ablation_graphs.py
    ├── generate_dataset_paper_figures.py
    ├── generate_interaction_figures.py
    ├── generate_new_paper_figures.py
    ├── generate_visualizations_10hz.py
    ├── run_visualization.py
    ├── visualizer.py
    └── LGF_CORLPaper/
        ├── corl_2026.sty
        ├── corlabbrvnat.bst
        ├── paper.bib
        ├── paper.tex
        └── Img/
            ├── Architecture.png
            ├── ablation_study.png
            ├── fig1_dataset_overview.png
            ├── fig1_dataset_overview_10hz.png
            ├── fig2_traffic_density.png
            ├── fig3_entity_classes.png
            ├── fig4_annotation_statistics.png
            ├── fig5_spatial_patterns.png
            ├── fig6_interaction_case_study.png
            ├── fig7_speed_density_adaptation.png
            ├── fig8_vttc_speed_coupling.png
            ├── fig9_trajectory_flow_road_mask.png
            ├── fig10_flow_potential_surface.jpg
            ├── fig11_predictions_output.jpg
            └── flow_spf_comparison.png
```.

---

## 1. VisionAndStabilization Module:
The `VisionAndStabilization` module handles raw aerial drone telemetry ingestion, homography stabilization, metric coordinate scaling, and multi-object tracking.

| File | Subsystem | Mathematical & Algorithmic Summary | Key Exports & Dependencies |
| :--- | :--- | :--- | :--- |
| **`botsort.py`** | Multi-Object Tracking | Implements the BoT-SORT tracker with Kalman filtering, Global Motion Compensation (GMC), and ReID cosine distance matching. | `BoTSORT`, `KalmanFilter` |
| **`botsort.yaml`** | Tracking Configuration | Hyperparameters defining high/low detection thresholds, track buffers (30 frames), and proximity weights. | Configuration dictionary |
| **`convert_videos_to_30fps.py`** | Video Preprocessing | Enforces uniform 30.0 FPS sampling via FFmpeg to guarantee strict temporal step intervals ($\Delta t = 33.3\text{ ms}$). | CLI script |
| **`extract_srt_metadata.py`** | Telemetry Parsing | Parses embedded SRT streams for drone altitude, latitude, longitude, and 3-axis gimbal angles. | Metadata CSV generator |
| **`map_pixels_to_meters.py`** | Spatial Calibration | Computes Ground Sampling Distance (GSD $= 0.16\text{ m/px}$) and perspective homography matrices. | Homography transform tools |
| **`stabilize_coordinates.py`** | Coordinate Stabilization | Applies affine compensation matrices to raw pixel trajectories to remove drone drift. | Stabilized CSV generator |
| **`stabilize_video_files.py`** | Image Stabilization | Computes inter-frame optical flow and warps raw video frames to produce stable top-down videos. | Stabilized MP4 generator |
| **`telemetry_aligner.py`** | Temporal Synchronization | Interpolates telemetry timestamps with video frame indices to ensure sensor alignment. | Telemetry alignment pipeline |
| **`tracker.py`** | Detection & Tracking Runner | End-to-end wrapper combining YOLO object detection with BoT-SORT state estimation. | Trajectory tracking pipeline |

.

### Detailed Functionality:
- **`botsort.py`**: Solves tracking across severe occlusion and dense clustering by fusing appearance features with motion priors.
- **`botsort.yaml`**: Provides fine-tuned tracking parameters for high-altitude aerial drone perspectives.
- **`convert_videos_to_30fps.py`**: Removes temporal frame drops and variable frame rate distortions from drone video recordings.
- **`extract_srt_metadata.py`**: Decodes flight log telemetry to recover physical camera pitch, roll, and elevation parameters.
- **`map_pixels_to_meters.py`**: Provides calibrated metric coordinate transformations for accurate physical velocity estimation.
- **`stabilize_coordinates.py`**: Converts raw pixel tracks into globally stabilized metric Cartesian coordinate trajectories.
- **`stabilize_video_files.py`**: Generates stabilized visualization video files for qualitative assessment.
- **`telemetry_aligner.py`**: Resolves time offsets between drone IMU logs and video frames.
- **`tracker.py`**: Ingests raw drone video sequences and exports raw multi-agent track CSV files.

---

## 2. DatasetPipeline Module:
The `DatasetPipeline` module handles trajectory cleaning, frequency conversion to 10 Hz, sliding-window chunking, class harmonization, and interaction mining.

| File | Domain | Core Algorithmic Logic | Output / Purpose |
| :--- | :--- | :--- | :--- |
| **`calculate_class_sizes.py`** | Entity Geometry | Computes mean length, width, and bounding box aspect ratios across HPE, SVE, and LVE classes. | Geometric class statistics |
| **`chunk_dataset.py`** | Temporal Chunking | Segments long trajectory recordings into standardized sliding windows (20 obs + 30 pred frames). | Chunked dataset splits |
| **`chunk_interactions.py`** | Interaction Slicing | Filters and isolates multi-agent encounter windows with high interaction density. | Curated interaction chunks |
| **`cleanup_datasets.py`** | Quality Assurance | Detects broken track IDs, interpolates missing bounding boxes, and removes spurious single-frame detections. | Cleaned annotation records |
| **`convert_dataset_10hz.py`** | Temporal Resampling | Downsamples 30 Hz raw trajectories to 10 Hz with anti-aliasing smoothing and trajectory interpolation. | 10 Hz benchmark dataset |
| **`create_manifest.py`** | Manifest Generation | Generates cryptographic checksums, sequence lengths, and metadata catalogs for fast data loading. | Split manifest catalogs |
| **`create_stratified_dataset.py`** | Stratified Sampling | Partitions trajectory sequences into stratified subsets by vehicle density, speed, and time of day. | Stratified HPO datasets |
| **`interaction_miner.py`** | Conflict Identification | Identifies conflict points, merging bottlenecks, and multi-vehicle spatial encounter zones. | Spatial conflict catalog |
| **`mine_conventional_ttc.py`** | 1D Physics Mining | Computes conventional 1D longitudinal Time-to-Collision along heading vectors for comparison. | 1D TTC distributions |
| **`mine_novel_interactions.py`** | Acceleration-Aware VTTC | Implements exact cubic CPA polynomial solving and vulnerability-weighted Swarm Complexity Index calculation. | VTTC & SCI metadata |
| **`smooth_obb_trajectories.py`** | Trajectory Smoothing | Applies Savitzky-Golay filtering and spline interpolation to smooth Oriented Bounding Box centerlines and headings. | Smoothed trajectory tracks |
| **`update_dataset_classes.py`** | Taxonomy Mapping | Harmonizes annotation labels into Small Vehicle Entities (SVE), Large Vehicle Entities (LVE), and Pedestrians (HPE). | Standardized annotations |

.

### Detailed Functionality:
- **`calculate_class_sizes.py`**: Validates physical dimensions ($\mu = 2.60\text{ m} \times 4.80\text{ m}$) across all annotated entity classes.
- **`chunk_dataset.py`**: Partitions continuous video trajectory recordings into standardized 5.0-second chunk sequences.
- **`chunk_interactions.py`**: Extracts interactive multi-agent scenarios where minimum inter-agent distance drops below critical thresholds.
- **`cleanup_datasets.py`**: Enforces strict temporal continuity and spatial validity across all track records.
- **`convert_dataset_10hz.py`**: Constructs the canonical 10 Hz dataset partition matching standard autonomous driving benchmarks.
- **`create_manifest.py`**: Creates indexing catalogs to enable random-access batch loading during neural network training.
- **`create_stratified_dataset.py`**: Generates balanced training and validation splits for hyperparameter tuning.
- **`interaction_miner.py`**: Discovers microscopic interaction active zones across complex intersection geometries.
- **`mine_conventional_ttc.py`**: Quantifies the limitations of classical 1D TTC in non-lane-based traffic environments.
- **`mine_novel_interactions.py`**: Implements the 3rd-degree polynomial Closest Point of Approach solver under acceleration.
- **`smooth_obb_trajectories.py`**: Suppresses measurement noise while preserving rapid steering and braking maneuvers.
- **`update_dataset_classes.py`**: Unifies heterogeneous entity labels into standard SVE, LVE, and HPE taxonomies.

---

## 3. PhysicsAndBehaviorEngine Module:
The `PhysicsAndBehaviorEngine` module extracts emergent flow surfaces, constructs dynamic potential fields, evaluates polygon collisions, and computes game-theoretic payoffs.

| File | Algorithmic Core | Mathematical Formulation | Role in Framework |
| :--- | :--- | :--- | :--- |
| **`emerging_lane_extractor.py`** | Probabilistic Flow Surface | Hausdorff clustering ($d_H \le 3.05\text{ m}$) and Gaussian potential field fitting: $P(\mathbf{x}) = \sum C_{l_m} \exp(-\frac{\text{dist}^2}{2\sigma^2})$. | PTFS surface discovery |
| **`game_theory_predictor.py`** | Strategic Game Theory | Cooperative and non-cooperative game formulation solving for equilibrium yielding payoffs $J(s_i, s_j)$. | Strategic interaction modeling |
| **`lane_manager.py`** | Spatial Map Graph | Spatial kd-tree indexing, Frenet coordinate conversion, and lane centerline polyline queries. | Vector map manager |
| **`road_mask_manager.py`** | Drivable Area Constraints | Binary road mask processing and off-road boundary violation checking. | Spatial feasibility checker |
| **`social_potential_field.py`** | Dynamic Potential Fields | Multi-band repulsive social potentials $U_j(\mathbf{x}, t)$ and 2D-OBB Separating Axis Theorem (SAT) collision detection. | Safety field engine |
| **`track_data_loader.py`** | High-Throughput IO | Fast trajectory parsing, coordinate transforms, and batched NumPy tensor synthesis. | Data loading backend |
| **`trajectory_predictor.py`** | Deterministic Rollout | Kinematic and potential field gradient descent rollout engine for FlowSPF baseline variants A, B, C, and D. | Physics baseline engine |

.

### Detailed Functionality:
- **`emerging_lane_extractor.py`**: Automatically constructs virtual lane graphs in unlaned traffic environments without static HD maps.
- **`game_theory_predictor.py`**: Computes strategic right-of-way priorities and mutual yielding payoffs between interacting vehicles.
- **`lane_manager.py`**: Provides spatial querying and longitudinal/lateral offset calculations for lane centerlines.
- **`road_mask_manager.py`**: Computes off-road rates to evaluate whether predicted paths violate physical road boundaries.
- **`social_potential_field.py`**: Formulates dynamic velocity-dependent repulsive safety buffers around surrounding agents.
- **`track_data_loader.py`**: High-performance data ingestion backend serving raw trajectory data to simulation and training pipelines.
- **`trajectory_predictor.py`**: Simulates deterministic agent trajectories using combined kinematic extrapolation and potential field forces.

---

## 4. ModelArchitectures Module:
The `ModelArchitectures` module implements the complete neural network architectures for LaneGameFormer, LaneGCN, and GameFormer.

### 4.1 LaneGameFormer (Proposed Architecture):
| File | Component | Architecture & Mechanism | Key Classes |
| :--- | :--- | :--- | :--- |
| **`lane_game_former.py`** | Top-Level Model | Integrates LaneGCN graph encoder with GameFormer interactive decoder and GMM prediction heads. | `LaneGameFormer` |
| **`lanegcn_encoder.py`** | Graph Encoder | ActorNet temporal 1D CNN + MapNet graph CNN with multi-scale A2M, M2M, M2A, and A2A fusion. | `LaneGCNEncoder`, `ActorNet`, `MapNet` |
| **`gameformer_decoder.py`** | Level-$k$ Decoder | Hierarchical interactive decoder with SPAB attention bias, LCMA mode anchoring, and BASP safety loss. | `GameFormerDecoder`, `InteractionDecoderLevel` |
| **`layers.py`** | Neural Primitives | Fourier positional embeddings, multi-head cross-attention, MLP blocks, and layer normalization. | `FourierEmbedding`, `MultiHeadAttention` |

.

### 4.2 LaneGCN (Baseline Architecture):
| File | Component | Architecture & Mechanism | Key Classes |
| :--- | :--- | :--- | :--- |
| **`lanegcn.py`** | Baseline Model | Full standalone PyTorch implementation of the LaneGCN motion forecasting model. | `LaneGCN` |
| **`graph_processor.py`** | Graph Construction | Computes spatial adjacency matrices, lane orientations, and successor/predecessor graph edges. | `GraphProcessor` |
| **`dataset_generator.py`** | Tensor Collation | Converts trajectory CSVs into LaneGCN graph tensor dictionaries with disk caching. | `DatasetGenerator` |
| **`layers.py`** | Graph Convolutions | Specialized multi-scale spatial graph convolutions and dilated 1D temporal convolution operators. | `GraphConv`, `DilatedConv1D` |

.

### 4.3 GameFormer / PrayagGameFormer (Baseline Architecture):
| File | Component | Architecture & Mechanism | Key Classes |
| :--- | :--- | :--- | :--- |
| **`Model/gameformer.py`** | Baseline Decoder | Level-$k$ interactive decoder adapted for BEV drone trajectory prediction. | `GameFormer` |
| **`Model/modules.py`** | Attention Blocks | Multi-agent self-attention, cross-attention, and iterative query refinement layers. | `CrossAttention`, `SelfAttention` |
| **`Data/dataset.py`** | Dataset Loader | PyTorch Dataset class for loading chunked trajectory files with coordinate normalization. | `BEVDataset` |
| **`Data/mapEncoder.py`** | Map Feature Encoder | Raster and vector processing for static road geometry. | `MapEncoder` |
| **`Utils/helpers.py`** | Geometry Utilities | Rotation matrices, bounding box geometry, and Frenet frame coordinate conversions. | Helper functions |
| **`Utils/metrics.py`** | Metric Library | Implementation of minADE, minFDE, Miss Rate, and Collision Rate metrics. | `MetricEvaluator` |
| **`Utils/training_utils.py`** | Training Harness | AdamW optimization, Cosine Annealing schedulers, gradient clipping, and checkpoint management. | `Trainer`, `CheckpointManager` |
| **`config.py`** | Configuration | Central hyperparameter and dataset directory path specifications. | `Config` |
| **`train.py`** | Training Entrypoint | Dedicated training script for PrayagGameFormer. | CLI training script |
| **`tune.py`** | HPO Pipeline | Automated hyperparameter optimization using the Optuna framework. | HPO study runner |
| **`evaluate.py`** | Evaluation Runner | Standalone evaluation script testing trained GameFormer checkpoints. | CLI evaluation script |

.

---

## 5. ExperimentsAndBenchmarks Module:
The `ExperimentsAndBenchmarks` module contains standardized fair evaluation harnesses, baseline evaluators, multi-seed training scripts, and zero-shot cross-domain testing tools.

### 5.1 BaselineEvaluators:
| File | Evaluator | Target Models & Protocols | Key Output |
| :--- | :--- | :--- | :--- |
| **`eval_cv_baseline.py`** | Constant Velocity | Linear extrapolation using final observed velocity vector. | Kinematic baseline metrics |
| **`eval_fixed_cr.py`** | Collision Rate Evaluator | Exact 2D Oriented Bounding Box geometric polygon intersection checks across 30 future frames. | Quantitative collision rates |
| **`eval_trivial_baselines.py`** | Non-Learning Baselines | Evaluates Static, Constant Acceleration, and Mean Historical Velocity baselines. | Empirical performance floor |
| **`evaluate_all_models.py`** | Multi-Model Benchmark | Aggregates comparative benchmark metrics across all trained neural and kinematic models. | Unified comparison tables |
| **`run_ptfs_spf_gt_eval.py`** | FlowSPF Physics Evaluation | Evaluates deterministic FlowSPF baseline variants A, B, C, and D (reproducing Table 11). | FlowSPF benchmark table |

.

### 5.2 FairBenchmarkSuite:
| File | Evaluation Suite | Description & Protocol | Target Benchmark |
| :--- | :--- | :--- | :--- |
| **`eval_standardized.py`** | Shared Fair Benchmark | Evaluates models on the 479 common agent-sequence intersection set with identical 20-obs + 30-pred windows. | Reproduces Table 4 |
| **`eval_all_agents.py`** | Global Evaluation | Evaluates models across all 8,412 visible agents per scene with class-wise performance breakdowns. | Reproduces Table 5 |
| **`evaluate_system.py`** | Flexible Harness | General-purpose evaluation runner supporting multiple export formats (JSON, CSV, LaTeX). | Comprehensive reports |
| **`evaluation_metrics.py`** | Metric Library | Mathematical definitions of minADE@K, minFDE@K, MR@K, Brier-FDE, CR, and Off-Road Rate. | Core metric engine |
| **`run_fair_experiments.py`** | Master Experiment Runner | Automates Optuna HPO, 3-seed model training, and architectural ablation studies. | Reproduces Table 12 |

.

### 5.3 Training:
| File | Pipeline | Core Capabilities | Key Interfaces |
| :--- | :--- | :--- | :--- |
| **`train.py`** | Model Training | Multi-GPU mixed-precision training loop for LaneGameFormer with dynamic loss scheduling. | CLI training entrypoint |
| **`inference.py`** | Trajectory Rollout | Generates multimodal trajectory predictions and mode probabilities from trained weights. | Inference API |
| **`shared_training_utils.py`** | Training Utilities | Composite loss functions, gradient scalers, and master seed initialization routines. | Utility library |

.

### 5.4 ZeroShotCrossDomain:
| File | Generalization Test | Protocol & Scope | Target Analysis |
| :--- | :--- | :--- | :--- |
| **`compare_models.py`** | Cross-Scene Comparison | Compares generalization performance of models evaluated on unseen intersection topologies. | Cross-domain robustness |
| **`test_models.py`** | Single-Scene Transfer | Tests pretrained models on individual unseen intersection scenes. | Transfer error metrics |
| **`test_models_on_largedatasets.py`** | Large-Scale Validation | Evaluates models across multi-flight aerial datasets to test real-world deployment robustness. | Large-scale summary |

.

---

## 6. PaperFiguresAndVisualization Module:
The `PaperFiguresAndVisualization` module contains figure generators, visualization tools, and the complete LaTeX publication source.

| File / Directory | Visual Asset / Script | Figure Description & Paper Reference | Output Format |
| :--- | :--- | :--- | :---: |
| **`draw_architecture.py`** | Architecture Diagram | High-resolution schematic showing LaneGCN encoder and GameFormer decoder dataflow. | Vector / PNG |
| **`generate_ablation_graphs.py`** | Ablation & FlowSPF Charts | Generates the ablation waterfall plot and FlowSPF comparison bar charts. | High-res PNG |
| **`generate_dataset_paper_figures.py`** | Dataset Figures | Generates dataset composition, vehicle density, and entity class breakdown figures. | Multi-panel PNG |
| **`generate_interaction_figures.py`** | Interaction Case Studies | Plots pairwise trajectory encounters, CPA distance profiles, and VTTC timelines. | High-res PNG |
| **`generate_new_paper_figures.py`** | Coupling & Speed-Density | Plots VTTC vs. speed hexbin coupling and speed-density adaptation curves. | High-res PNG |
| **`generate_visualizations_10hz.py`** | 10 Hz Trajectory Overlays | Overlays multimodal trajectory prediction ribbons onto top-down BEV drone frames. | Color-coded PNG |
| **`run_visualization.py`** | Visualization CLI | Master CLI tool for generating qualitative rollouts and animation sequences. | PNG / MP4 |
| **`visualizer.py`** | Core Graphics Library | Matplotlib and OpenCV rendering routines for bounding boxes, trajectories, and heatmaps. | Rendering backend |
| **`LGF_CORLPaper/paper.tex`** | Conference Manuscript | Complete LaTeX source text for the conference paper submission. | LaTeX document |
| **`LGF_CORLPaper/paper.bib`** | Bibliography | Complete BibTeX bibliography with all literature citations. | BibTeX database |
| **`LGF_CORLPaper/corl_2026.sty`** | Style Package | Conference stylesheet defining document layout, fonts, and geometry. | Style file |
| **`LGF_CORLPaper/corlabbrvnat.bst`** | BibTeX Style | Bibliography formatting style implementing abbreviated author-year references. | BST style file |
| **`LGF_CORLPaper/Img/`** | Publication Figures | Contains all 15 publication figures and high-resolution visual diagrams referenced in the paper. | PNG / JPG |

.

---

## Verification & Integrity Assurance:
All file paths, directory references, and class definitions in this index correspond directly to the physical files in the repository.
All source code files and benchmarking scripts are validated and ready for experimental execution and reproduction.
