# LaneGameFormer Codebase Index:

## Overview of the Repository:
This index provides a comprehensive directory and file catalog for the **LaneGameFormer (LGF)** codebase.
LaneGameFormer is a motion forecasting framework designed for dense and unstructured multi-agent traffic environments.
The codebase integrates computer vision, trajectory stabilization, physics and potential field estimation, game-theoretic reasoning, neural model architectures, and experimental benchmarking suites.

---

## Directory Tree Overview:
```
Code/
├── VisionAndStabilization/
├── DatasetPipeline/
├── PhysicsAndBehaviorEngine/
├── ModelArchitectures/
│   ├── LaneGameFormer/
│   ├── LaneGCN/
│   └── GameFormer/
│       └── PrayagGameFormer/
│           ├── Checkpoints/
│           │   └── MapCache/
│           ├── Data/
│           ├── Model/
│           └── Utils/
├── ExperimentsAndBenchmarks/
│   ├── BaselineEvaluators/
│   ├── FairBenchmarkSuite/
│   ├── Training/
│   └── ZeroShotCrossDomain/
└── PaperFiguresAndVisualization/
    └── LGF_CORLPaper/
        └── Img/
```.

---

## 1. VisionAndStabilization Directory:
The `VisionAndStabilization` directory contains scripts and configurations for raw drone video processing, camera telemetry extraction, homography-based coordinate stabilization, and multi-object tracking.

### botsort.py:
- Implements the BoT-SORT multi-object tracking algorithm adapted for high-altitude drone aerial footage.
- Combines Kalman filter state estimation, camera motion compensation via global motion estimation, and visual feature re-identification.
- Produces persistent tracking IDs across dense vehicle clusters under severe occlusion.

### botsort.yaml:
- Defines hyperparameter configurations for the BoT-SORT tracking pipeline.
- Specifies track buffer lengths, high and low detection confidence thresholds, and proximity association weights.
- Tunes ReID feature distance thresholds for heterogeneous traffic entities.

### convert_videos_to_30fps.py:
- Normalizes raw aerial drone recordings to a standard 30 frames-per-second temporal rate.
- Uses FFmpeg bindings to resample variable frame rate video captures into constant frame rate sequences.
- Ensures uniform delta-time intervals across all subsequent downstream tracking steps.

### extract_srt_metadata.py:
- Parses embedded subtitle metadata streams from DJI drone video recordings.
- Extracts per-frame telemetry including altitude, latitude, longitude, gimbal pitch, roll, and yaw angles.
- Serializes extracted telemetry records to structured CSV formats for camera motion calibration.

### map_pixels_to_meters.py:
- Computes Ground Sampling Distance (GSD) and camera homography matrices.
- Translates image pixel coordinates into metric distance coordinates based on optical parameters and drone altitude.
- Calibrates camera scale factors for accurate physical velocity and acceleration estimation.

### stabilize_coordinates.py:
- Applies affine transformation matrices to raw pixel trajectory coordinates.
- Compensates for drone drift, wind buffeting, and altitude variations across consecutive frames.
- Outputs stabilized Bird's-Eye-View (BEV) Cartesian coordinates in metric space.

### stabilize_video_files.py:
- Performs direct image-space video frame stabilization using feature point tracking and perspective warping.
- Computes optical flow and homography between successive frames to synthesize a rock-steady top-down camera view.
- Saves stabilized MP4 video files used for qualitative evaluation and visual rendering.

### telemetry_aligner.py:
- Aligns asynchronous telemetry streams with video frame indices using timestamp interpolation.
- Synchronizes IMU orientation, barometric altitude, and GPS positioning with image capture timestamps.
- Guarantees temporal consistency between physical telemetry and computer vision detections.

### tracker.py:
- High-level orchestration pipeline combining YOLO object detection with BoT-SORT state estimation.
- Ingests raw video files and outputs frame-by-frame multi-agent bounding boxes with persistent tracking identifiers.
- Exports structured raw tracking CSV logs for the dataset pipeline.

---

## 2. DatasetPipeline Directory:
The `DatasetPipeline` directory handles dataset generation, chunking, filtering, coordinate transformation, traffic entity taxonomy mapping, and microscopic interaction mining.

### calculate_class_sizes.py:
- Computes geometric bounding box dimension statistics across all annotated entities.
- Calculates mean length, width, aspect ratio, and variance for Small Vehicle Entities (SVE), Large Vehicle Entities (LVE), and Human-Pedestrian Entities (HPE).
- Validates physical bounding box consistency against real-world vehicular dimensions.

### chunk_dataset.py:
- Segments continuous multi-minute drone trajectory recordings into standardized temporal windows.
- Extracts sliding window observation-prediction samples with configurable history and horizon lengths.
- Organizes generated chunk files into standardized train, validation, and test partition directories.

### chunk_interactions.py:
- Filters and extracts trajectory chunks exhibiting dense multi-agent interactions.
- Identifies critical encounters based on inter-agent proximity, approaching angles, and velocity gradients.
- Generates curated interaction evaluation benchmarks for interactive motion forecasting.

### cleanup_datasets.py:
- Performs data hygiene and integrity verification on raw trajectory logs.
- Detects and repairs broken track IDs, interpolates missing bounding box frames, and removes spurious single-frame detections.
- Ensures all exported dataset records contain valid continuous time sequences.

### convert_dataset_10hz.py:
- Downsamples 30 Hz raw trajectory data to the canonical 10 Hz sampling rate used in autonomous driving benchmarks.
- Performs anti-aliasing smoothing and trajectory interpolation to preserve kinematic fidelity.
- Produces the standardized 10 Hz dataset partition with 20-frame observation and 30-frame prediction horizons.

### create_manifest.py:
- Builds index catalogs, metadata manifests, and cryptographic checksums for all dataset splits.
- Records scene identifiers, sequence lengths, agent counts, and file locations for fast random-access data loading.
- Generates split manifests for training, validation, and testing pipelines.

### create_stratified_dataset.py:
- Creates stratified dataset subsets balanced across vehicle density levels, speed profiles, and time-of-day conditions.
- Partitions data into low-density, medium-density, high-density, and congested regimes.
- Provides balanced splits for hyperparameter optimization and cross-domain generalization studies.

### interaction_miner.py:
- Scans trajectory datasets to identify spatial conflict points, merging bottlenecks, and multi-vehicle intersections.
- Calculates geometric proximity envelopes and spatial interaction densities across intersection scenes.
- Catalogs microscopic interaction events for qualitative case studies.

### mine_conventional_ttc.py:
- Implements classical 1D longitudinal Time-to-Collision (TTC) mining along vehicle heading vectors.
- Serves as an empirical baseline to compare against vector-based collision estimation methods.
- Computes distribution histograms and interaction active zones under standard TTC metrics.

### mine_novel_interactions.py:
- Implements the acceleration-aware Vector Time-to-Collision (VTTC) polynomial solver for Closest Point of Approach (CPA).
- Computes the dynamic Swarm Complexity Index (SCI) weighted by entity vulnerability classes (HPE, SVE, LVE).
- Extracts safety-critical interaction events characterized by dynamic spatial negotiation.

### smooth_obb_trajectories.py:
- Applies Savitzky-Golay filtering and cubic smoothing splines to raw Oriented Bounding Box (OBB) trajectories.
- Eliminates sensor jitter while preserving high-frequency steering maneuvers and braking dynamics.
- Computes smooth first-order (velocity) and second-order (acceleration) time derivatives.

### update_dataset_classes.py:
- Harmonizes heterogeneous annotation taxonomies into standardized three-tier entity classes.
- Maps diverse Indian traffic agents into Small Vehicle Entities (two-wheelers, auto-rickshaws), Large Vehicle Entities (cars, buses, trucks), and Human-Pedestrian Entities.
- Updates annotation CSV and JSON schemas across all dataset directories.

---

## 3. PhysicsAndBehaviorEngine Directory:
The `PhysicsAndBehaviorEngine` directory implements physics models, emerging lane extraction, potential fields, and game-theoretic behavior rules.

### emerging_lane_extractor.py:
- Discovers virtual lane topology directly from unstructured multi-agent trajectory clusters without static HD maps.
- Computes pairwise Hausdorff distances between smoothed trajectories and clusters paths into representative polylines.
- Evaluates support confidence scores and fits continuous B-spline representations of emergent traffic corridors.

### game_theory_predictor.py:
- Formulates multi-agent trajectory prediction as an interactive non-cooperative and cooperative game.
- Solves for equilibrium yielding decisions, right-of-way prioritization, and strategic payoff matrices.
- Computes interactive trajectory modifications based on mutual utility optimization.

### lane_manager.py:
- Provides spatial querying, polyline sampling, and graph indexing for lane centerlines and boundaries.
- Computes nearest-lane projections, Frenet frame transformations, and longitudinal/lateral offsets.
- Interfaces between extracted lane geometries and neural network map encoders.

### road_mask_manager.py:
- Loads and processes high-resolution binary drivable area road masks.
- Validates trajectory feasibility by checking whether predicted coordinates remain within legal drivable boundaries.
- Computes off-road violation rates for model evaluation and safety auditing.

### social_potential_field.py:
- Implements dynamic Social Potential Fields (SPF) with velocity-dependent repulsive potential functions.
- Formulates multi-band safety envelopes and dynamic clearance margins.
- Implements the 2D Oriented Bounding Box Separating Axis Theorem (SAT) for differentiable polygon collision checking.

### track_data_loader.py:
- Provides high-throughput loading and caching of raw and chunked trajectory datasets.
- Parses CSV track annotations, applies coordinate transformations, and constructs batched NumPy data structures.
- Supplies formatted input records to downstream physics engines and visualization routines.

### trajectory_predictor.py:
- Implements physics-based deterministic trajectory rollout algorithms.
- Integrates kinematic motion models with potential field gradient descent to simulate agent trajectories.
- Implements the deterministic FlowSPF algorithmic baseline variants A, B, C, and D.

---

## 4. ModelArchitectures Directory:
The `ModelArchitectures` directory contains the neural network implementations for LaneGameFormer, LaneGCN, and GameFormer.

### 4.1 LaneGameFormer:
#### lane_game_former.py:
- Top-level PyTorch module for the complete LaneGameFormer (LGF) motion forecasting architecture.
- Integrates the LaneGCN graph convolution encoder with the multi-level GameFormer interaction decoder.
- Combines Constant Velocity (CV) kinematic priors with learned multimodal Gaussian Mixture Model (GMM) prediction heads.

#### lanegcn_encoder.py:
- Implements the LaneGCN feature encoding network tailored for top-down BEV traffic scenes.
- Contains ActorNet for temporal trajectory encoding and MapNet for lane graph convolution.
- Executes multi-scale fusion across Actor-to-Map (A2M), Map-to-Map (M2M), Map-to-Actor (M2A), and Actor-to-Actor (A2A) pathways.

#### gameformer_decoder.py:
- Implements the hierarchical Level-$k$ interactive game-theoretic reasoning decoder.
- Features Social Potential Attention Bias (SPAB) injecting repulsive safety fields into multi-head attention mechanisms.
- Incorporates Lane-Conditioned Mode Anchoring (LCMA), FutureEncoder with Hierarchical Temporal Fusion (HTF), and Gated Residual connections.
- Implements composite training losses including GMM negative log-likelihood, mode classification loss, and Behavior-Aware Social Potential (BASP) loss.

#### layers.py:
- Provides foundational neural building blocks including sinusoidal Fourier positional embeddings, multi-head self-attention, and cross-attention blocks.
- Implements multi-layer perceptron (MLP) submodules, residual projections, and layer normalization layers.
- Supports flexible dimension projections across encoder and decoder representations.

### 4.2 LaneGCN:
#### lanegcn.py:
- Standalone PyTorch implementation of the LaneGCN baseline architecture.
- Encodes agent trajectories and vectorized lane polylines via multi-scale graph convolutions.
- Outputs multimodal future trajectory predictions with confidence scoring.

#### graph_processor.py:
- Constructs spatial graph adjacency matrices and lane connectivity graphs from raw vectorized coordinates.
- Computes spatial edge attributes, relative orientation angles, and predecessor/successor connections.
- Converts raw polyline geometries into LaneGCN graph tensor inputs.

#### dataset_generator.py:
- Converts trajectory datasets into graph tensor dictionaries compatible with LaneGCN.
- Handles coordinate normalization, agent centering, and graph batch collation.
- Manages disk caching of preprocessed graph structures for accelerated training.

#### layers.py:
- Implements specialized 1D dilated temporal convolutions and spatial graph convolution operators for LaneGCN.
- Formulates multi-scale feature aggregation across spatial neighbor nodes.
- Provides non-linear activation and normalization layers.

### 4.3 GameFormer (PrayagGameFormer):
#### PrayagGameFormer/Model/gameformer.py:
- Implementation of the GameFormer baseline architecture adapted for unstructured BEV drone datasets.
- Implements hierarchical level-$k$ interactive decoding over agent and map queries.
- Predicts multimodal trajectory distributions with level-wise trajectory refinement.

#### PrayagGameFormer/Model/modules.py:
- Contains cross-attention and self-attention interaction layers for GameFormer decoders.
- Formulates iterative query updating across sequential reasoning levels.
- Implements multi-agent attention masks and spatial relative position encodings.

#### PrayagGameFormer/Data/dataset.py:
- PyTorch Dataset class for loading chunked trajectory files for GameFormer.
- Performs agent coordinate canonicalization, past trajectory normalization, and future ground truth extraction.
- Generates batch tensors for training, validation, and evaluation loops.

#### PrayagGameFormer/Data/mapEncoder.py:
- Encodes road geometry and vectorized map elements into spatial feature embeddings.
- Provides raster and vector processing for static environmental context.
- Interacts with attention query modules in the GameFormer decoder.

#### PrayagGameFormer/Utils/helpers.py:
- Geometry helper routines including rotation matrices, box intersections, and coordinate transforms.
- Implements trajectory smoothing, heading angle calculation, and Frenet frame projections.
- Formulates bounding box overlap and distance calculations.

#### PrayagGameFormer/Utils/metrics.py:
- Implements standardized motion forecasting metrics including ADE, FDE, Miss Rate, and Collision Rate.
- Computes minimum displacement error across top-$K$ multimodal trajectory predictions.
- Implements collision detection between predicted agent bounding boxes.

#### PrayagGameFormer/Utils/training_utils.py:
- Training utility functions including AdamW optimization routines, Cosine Annealing learning rate schedulers, and gradient clipping.
- Manages model checkpoint saving, resume loading, and validation loss tracking.
- Logs training metrics to TensorBoard.

#### PrayagGameFormer/config.py:
- Centralized configuration file specifying model hyperparameters, batch sizes, learning rates, and dataset paths.
- Configures observation lengths, prediction horizons, and number of multimodal modes.
- Manages device placement and distributed training flags.

#### PrayagGameFormer/train.py:
- Dedicated training script for training PrayagGameFormer on chunked trajectory datasets.
- Executes full training and validation loops with metric evaluation after each epoch.
- Exports best-performing model weights based on validation displacement error.

#### PrayagGameFormer/tune.py:
- Automated hyperparameter optimization pipeline using the Optuna framework.
- Explores learning rate, hidden dimension, dropout, and loss weighting search spaces.
- Identifies optimal hyperparameter configurations for competitive benchmarking.

#### PrayagGameFormer/evaluate.py:
- Standalone evaluation script evaluating trained GameFormer checkpoints against test sets.
- Computes minADE, minFDE, Miss Rate, and Collision Rate metrics.
- Outputs detailed evaluation reports and metric breakdown summaries.

---

## 5. ExperimentsAndBenchmarks Directory:
The `ExperimentsAndBenchmarks` directory contains evaluation harnesses, fair benchmarking suites, baseline evaluators, training pipelines, and zero-shot cross-domain testing scripts.

### 5.1 BaselineEvaluators:
#### eval_cv_baseline.py:
- Evaluates the Constant Velocity (CV) kinematic baseline across all benchmark partitions.
- Extrapolates future positions linearly using the last observed velocity vector.
- Provides a reference performance floor to quantify the value of learned neural representations.

#### eval_fixed_cr.py:
- Evaluates multi-agent Collision Rates (CR) using exact 2D Oriented Bounding Box geometric polygon intersections.
- Analyzes predicted inter-agent spatial overlaps across full 30-frame prediction horizons.
- Measures safety violations across all baseline models and proposed architectures.

#### eval_trivial_baselines.py:
- Implements and evaluates simple non-learning baselines including Static Position, Constant Acceleration, and Mean Historical Velocity.
- Quantifies trajectory prediction errors across standard displacement metrics.
- Validates the complexity of the underlying unstructured traffic dataset.

#### evaluate_all_models.py:
- Master evaluation suite running comparative benchmarks across LaneGameFormer, GameFormer, LaneGCN, and kinematic baselines.
- Aggregates metric outputs across all test splits and generates unified comparative markdown tables.
- Evaluates model latency, parameter counts, and inference throughput.

#### run_ptfs_spf_gt_eval.py:
- Evaluates deterministic FlowSPF baseline variants A, B, C, and D on ground-truth trajectory test splits.
- Quantifies displacement error, collision rate, and off-road violation rates for each physics formulation.
- Validates the empirical necessity of acceleration-aware CPA VTTC and dynamic Swarm Complexity Index scaling.

### 5.2 FairBenchmarkSuite:
#### eval_standardized.py:
- Implements the standardized fair evaluation protocol on the shared 479 agent-sequence intersection set.
- Enforces identical 20-frame observation and 30-frame prediction horizons across all competing models.
- Reports minADE@6, minFDE@6, Miss Rate (MR@6), Collision Rate, and inference latency.

#### eval_all_agents.py:
- Evaluates models on the complete all-agents dataset partition comprising 8,412 multi-agent instances.
- Computes global trajectory error metrics across dense, heterogeneous traffic scenes.
- Analyzes performance breakdowns across SVE, LVE, and HPE agent classes.

#### evaluate_system.py:
- General-purpose evaluation runner supporting multiple dataset formats, metric definitions, and export options.
- Generates structured JSON reports, metric summary logs, and LaTeX table snippets.
- Provides CLI flags for selective model and dataset evaluation.

#### evaluation_metrics.py:
- Core mathematical library implementing motion forecasting metrics.
- Implements minADE@K, minFDE@K, Miss Rate (MR@K), Brier-FDE, Collision Rate, and Off-Road Rate.
- Supports weighted metrics conditioned on agent velocity, heading variance, and traffic density.

#### run_fair_experiments.py:
- Master automated experimental suite managing hyperparameter optimization (HPO) and 3-seed model training.
- Coordinates multi-seed training across master seeds to compute mean and standard deviation performance.
- Executes full architectural ablation experiments across all subcomponent variations.

### 5.3 Training:
#### train.py:
- Main training script for LaneGameFormer supporting multi-GPU execution and mixed-precision training.
- Implements customized training loops with dynamic loss scheduling and validation monitoring.
- Saves best-checkpoint weights and exports evaluation summaries.

#### inference.py:
- Standalone inference engine generating multimodal trajectory predictions from trained model weights.
- Accepts raw trajectory tensors or preprocessed chunk files and outputs multimodal rollout paths.
- Exports predicted trajectory coordinates and mode probabilities for visualization.

#### shared_training_utils.py:
- Shared training utilities including loss function builders, gradient scalers, and metric accumulators.
- Provides reproducible random seed initialization across PyTorch, NumPy, and CUDA environments.
- Manages logging hooks and checkpoint serialization formats.

### 5.4 ZeroShotCrossDomain:
#### compare_models.py:
- Compares cross-scene generalization performance across LaneGameFormer, LaneGCN, and GameFormer.
- Analyzes prediction degradation when models trained on specific intersection scenes are evaluated on unseen topologies.
- Evaluates topological adaptability in the absence of HD maps.

#### test_models.py:
- Evaluates pretrained model checkpoints on independent single-scene test partitions.
- Computes zero-shot transfer error metrics across differing intersection geometries.
- Quantifies domain adaptation robustness across diverse Indian urban intersection environments.

#### test_models_on_largedatasets.py:
- Executes large-scale cross-domain evaluation over the full top-down aerial dataset repository.
- Processes continuous multi-camera flight logs to evaluate real-world model deployment robustness.
- Outputs comprehensive cross-domain generalization summary reports.

---

## 6. PaperFiguresAndVisualization Directory:
The `PaperFiguresAndVisualization` directory contains visualization tools, figure generators, rendering scripts, and the complete LaTeX paper manuscript.

### draw_architecture.py:
- Generates high-resolution schematic diagrams illustrating the LaneGameFormer system architecture.
- Visualizes data flow from agent histories and map polylines through LaneGCN graph convolutions to GameFormer interactive decoders.
- Exports publication-ready vector and raster architecture figures.

### generate_ablation_graphs.py:
- Generates the component ablation waterfall charts and FlowSPF comparison bar charts.
- Formulates comparative visualizations showing relative performance gains and safety improvements.
- Exports high-resolution PDF and PNG figures for the paper manuscript.

### generate_dataset_paper_figures.py:
- Generates publication figures detailing dataset statistics, spatial occupancy distributions, and split compositions.
- Plots annotation counts, vehicle density heatmaps, and time-of-day distributions.
- Produces multi-panel figure layouts formatted for conference paper standards.

### generate_interaction_figures.py:
- Produces microscopic multi-agent interaction case study plots.
- Visualizes pairwise vehicle trajectories, Closest Point of Approach (CPA) timelines, and VTTC profiles.
- Demonstrates dynamic safety margin adaptation during close-proximity vehicle negotiation.

### generate_new_paper_figures.py:
- Generates 2D hexbin density plots coupling ego vehicle speed with Vector Time-to-Collision (VTTC).
- Plots speed-density adaptation curves demonstrating traffic slowdown under congestion.
- Formulates spatial occupancy heatmaps across irregular intersection geometries.

### generate_visualizations_10hz.py:
- Generates 10 Hz Bird's-Eye-View trajectory stream visualizations overlaid on aerial drone frames.
- Renders multi-agent historical paths, ground truth futures, and multimodal prediction corridors.
- Exports color-coded trajectory stream visualizations.

### run_visualization.py:
- Command-line interface for generating qualitative prediction rollouts and visual inspection assets.
- Allows selective rendering of specific scene chunks, time frames, and agent subsets.
- Outputs high-resolution annotated image sequences and video clips.

### visualizer.py:
- Core visualization rendering engine with Matplotlib and OpenCV graphics routines.
- Draws Oriented Bounding Boxes, multi-mode trajectory ribbons, uncertainty ellipses, and road masks.
- Manages color maps, font sizing, and layout formatting for paper figures.

### LGF_CORLPaper Subdirectory:
#### paper.tex:
- Complete LaTeX source manuscript for the conference research paper.
- Contains the formal mathematical formulations, architectural descriptions, experimental tables, and qualitative analyses.
- Formatted in accordance with official publication guidelines.

#### paper.bib:
- Comprehensive BibTeX bibliography file containing references for autonomous driving, motion forecasting, and game theory.
- Includes citations for foundational baselines, datasets, graph neural networks, and interactive transformer models.
- Maintains standard bibliographic formatting.

#### corl_2026.sty:
- Conference LaTeX style package defining page geometry, font configurations, header styles, and formatting constraints.
- Enforces official publication visual standards.

#### corlabbrvnat.bst:
- Bibliography formatting style file implementing abbreviated author-year citation formatting.
- Manages reference section numbering, sorting, and inline citation appearance.

#### Img Subdirectory:
- Contains all high-resolution diagram assets, experimental graphs, and qualitative visualization figures referenced in the paper manuscript.
- Includes `Architecture.png`, `fig1_dataset_overview.png`, `fig2_traffic_density.png`, `fig3_entity_classes.png`, `fig4_annotation_statistics.png`, `fig5_spatial_patterns.png`, `fig6_interaction_case_study.png`, `fig7_speed_density_adaptation.png`, `fig8_vttc_speed_coupling.png`, `fig9_trajectory_flow_road_mask.png`, `fig10_flow_potential_surface.jpg`, `fig11_predictions_output.jpg`, `flow_spf_comparison.png`, and `ablation_study.png`.
