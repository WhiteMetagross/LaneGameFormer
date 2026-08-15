# PrayagGameFormer

## Game-Theoretic Transformer Networks for Trajectory Prediction in Unstructured Indian Traffic

---

**An Adaptation of GameFormer for Multi-Agent Motion Forecasting on the ChunkedProjectPrayagBEVDataset**

---

## Abstract

PrayagGameFormer implements the GameFormer architecture for trajectory prediction on the ChunkedProjectPrayagBEVDataset, comprising aerial drone footage of Indian intersections. This implementation follows the comparative evaluation protocol established in the LaneGameFormer Strategic Plan, serving as a baseline to quantify the performance of game-theoretic interaction modeling approaches in unstructured traffic environments.

The key contribution of GameFormer is the hierarchical Level-k reasoning mechanism, which models strategic interactions between agents through iterative refinement of trajectory predictions based on anticipated behaviors of other agents.

---

## Table of Contents

1. [Methodology](#methodology)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Hyperparameter Optimization](#hyperparameter-optimization)
8. [Configuration](#configuration)
9. [Dataset](#dataset)
10. [Evaluation Metrics](#evaluation-metrics)
11. [Linux Server Deployment](#linux-server-deployment)
12. [References](#references)

---

## Methodology

### Problem Formulation

Multi-agent trajectory prediction is formulated as a hierarchical game where each agent $i$ iteratively refines its prediction by reasoning about other agents' anticipated behaviors. Given observed trajectories $\mathbf{X} = \{\mathbf{x}_i\}_{i=1}^N$ and map context $\mathcal{M}$, we predict:

$$\{\hat{\tau}_k^i, p_k^i, \sigma_k^i\}_{k=1}^K \quad \forall i \in \{1, ..., N\}$$

where $\hat{\tau}_k^i$ is the $k$-th trajectory mode, $p_k^i$ is the mode probability, and $\sigma_k^i$ represents uncertainty via GMM variance parameters.

### Level-k Game-Theoretic Reasoning

The core innovation of GameFormer is modeling trajectory prediction as a cognitive hierarchy game based on Camerer et al.'s Level-k theory:

**Level-0 (Physics Prior)**: Agents predict independently without considering interactions:

$$\tau_i^{(0)} = f_0(\mathbf{h}_i, \mathcal{M})$$

where $\mathbf{h}_i$ is the agent's encoded state and $\mathcal{M}$ is map context.

**Level-k (Strategic Refinement)**: Each agent refines predictions by attending to other agents' Level-(k-1) predictions:

$$\tau_i^{(k)} = f_k\left(\tau_i^{(k-1)}, \{\tau_j^{(k-1)}\}_{j \neq i}, \mathcal{M}\right)$$

This recursive formulation captures how human drivers reason: "Given that they will likely do X, I should do Y."

### Self-Masking Mechanism

A critical component is **self-masking** during Level-k attention. When agent $i$ attends to encoded futures at Level-k:

$$\text{Attention}_i^{(k)} = \text{softmax}\left(\frac{\mathbf{Q}_i (\mathbf{K}_{-i})^T}{\sqrt{d}}\right) \mathbf{V}_{-i}$$

The subscript $-i$ indicates that agent $i$'s own Level-(k-1) prediction is **masked out**. This prevents trivial self-copying and forces genuine strategic reasoning about other agents.

### Agent Trajectory Encoding

Past trajectories are encoded via LSTM:

$$\mathbf{h}_i^{enc} = \text{LSTM}(\{(x_t, y_t, v_x^t, v_y^t, \theta_t)\}_{t=1}^{T_{obs}})$$

The encoded state captures:
- Position $(x, y)$: Absolute coordinates
- Velocity $(v_x, v_y)$: Motion direction and speed
- Heading $\theta$: Agent orientation

### Lane Context Encoding

Unlike HD map-based approaches, PrayagGameFormer uses simplified lane encoding from skeletonized road masks:

**PointNet-style Lane Encoding**:

$$\mathbf{l}_j = \text{MLP}\left(\frac{1}{|L_j|} \sum_{p \in L_j} \text{Linear}(p)\right) + \text{PE}(j)$$

where $L_j$ is the $j$-th lane polyline and PE is positional encoding.

### Future Trajectory Encoding

At each Level-k, predicted trajectories from Level-(k-1) are encoded for attention:

$$\mathbf{e}_i^{(k)} = \text{FutureEnc}(\tau_i^{(k-1)}) + \mathbf{e}_i^{(k-1)}$$

The FutureEncoder processes 8-dimensional trajectory features:

| Dimension | Description |
|-----------|-------------|
| $x, y$ | Position coordinates |
| $\theta$ | Heading angle |
| $v_x, v_y$ | Velocity components |
| $w, l$ | Vehicle width and length |
| $h$ | Vehicle height (optional) |

### Gaussian Mixture Model Output

The GMMPredictor outputs distributional predictions:

$$\hat{\tau}_k^i(t) = (\mu_x^{k,t}, \mu_y^{k,t}, \log\sigma_x^{k,t}, \log\sigma_y^{k,t})$$

The log-variance parameterization ensures $\sigma > 0$ and enables stable gradient flow.

### Training Objective

The total loss combines multiple components with level-wise weighting:

$$\mathcal{L}_{total} = \sum_{l=0}^{L} w_l \cdot \left(\mathcal{L}_{nll}^{(l)} + \lambda_{cls} \mathcal{L}_{cls}^{(l)}\right)$$

**Level Weights**: Later levels receive higher weights as they produce more refined predictions:

| Level | Weight | Interpretation |
|-------|--------|----------------|
| 0 | 0.50 | Physics prior (no interaction) |
| 1 | 0.75 | First strategic refinement |
| 2 | 1.00 | Mature game-theoretic reasoning |
| 3 | 1.00 | Equilibrium approximation |

**GMM Negative Log-Likelihood**:

$$\mathcal{L}_{nll} = \frac{1}{NT} \sum_{i,t} \left[\frac{(y_t - \mu_t)^2}{2\sigma_t^2} + \log\sigma_t + \frac{1}{2}\log(2\pi)\right]$$

**Classification Loss** (Winner-Takes-All):

$$\mathcal{L}_{cls} = -\frac{1}{N} \sum_{i=1}^{N} \log p_{k^*}^i$$

where $k^* = \arg\min_k \text{FDE}(\hat{\tau}_k^i, \tau_i^{gt})$.

### Inference

At inference, the model performs $L$ forward passes through the decoder:

1. **Level-0**: Generate initial predictions from encoded states
2. **Level-1 to L**: Iteratively refine using self-masked attention
3. **Mode Selection**: Either argmax (deterministic) or sampling (stochastic)

The final prediction uses Level-$L$ outputs with mode probabilities for ranking.

---

## Architecture

PrayagGameFormer comprises an encoder-decoder architecture with game-theoretic interaction modeling:

### Encoder

**AgentEncoder**: LSTM-based trajectory encoding
- Processes past agent trajectories as sequences
- Captures temporal dynamics and motion patterns
- Output dimension: dim (default 256)

**LaneEncoder**: PointNet-based lane encoding
- Encodes lane polylines from skeletonized road masks
- Positional encoding for spatial relationships
- Output dimension: dim (default 256)

**CrosswalkEncoder**: MLP-based context encoding
- Encodes additional map context features
- Simplified for unstructured environments

**Transformer Fusion**: Self-attention context aggregation
- Multi-head self-attention across all encoded features
- Produces unified context representation

### Decoder

**InitialDecoder (Level-0)**: Physics prior predictions
- Learnable modal query embeddings (K=6 modes)
- Agent-specific query embeddings
- Cross-attention to map context
- No interaction modeling at this level

**InteractionDecoder (Level-k)**: Game-theoretic refinement
- FutureEncoder: Encodes previous-level trajectory predictions
- Self-attention with self-masking (agents attend to others' predictions)
- Cross-attention to map context
- Iterative refinement through L decoder levels

**GMMPredictor**: Gaussian Mixture Model output
- Outputs mean positions (mu_x, mu_y)
- Outputs log standard deviations (log_sigma_x, log_sigma_y)
- Mode probability classification head

### Level-k Reasoning

The hierarchical game-theoretic reasoning follows:

- **Level-0**: Independent predictions based on physics prior
- **Level-1**: Each agent refines predictions based on Level-0 of other agents
- **Level-k**: Each agent refines based on Level-(k-1) of other agents

This models how human drivers reason about "what will they do, and therefore what should I do."

---

## Installation

### Prerequisites

- Python 3.11 or higher
- CUDA 12.1+ (for GPU acceleration)
- NVIDIA GPU with 8GB+ VRAM (RTX 4060 minimum, RTX 6000 recommended)

### Setup

```bash
cd PrayagGameFormer

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Validation Run (Test Mode)

```bash
# Quick test with 1 data chunk, 2 epochs
python train.py --datasetType 10Hz --testChunks 1 --batchSize 4 --numEpochs 2
```

### Full Training

```bash
# Standard training on 10Hz dataset
python train.py --datasetType 10Hz --batchSize 16 --numEpochs 50
```

---

## Training

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--datasetType` | str | 10Hz | Dataset variant (10Hz or 30Hz) |
| `--batchSize` | int | 16 | Training batch size |
| `--numEpochs` | int | 100 | Number of training epochs |
| `--lr` | float | 1e-4 | Initial learning rate |
| `--patience` | int | 10 | Early stopping patience |
| `--resume` | str | None | Checkpoint path to resume from |
| `--seed` | int | None | Random seed for reproducibility |
| `--multiSeed` | flag | False | Run with 3 seeds [42, 1337, 2024] |
| `--hpoParams` | str | None | Path to best_params.json from HPO |
| `--workers` | int | 8/4 | DataLoader workers (Linux/Windows) |
| `--saveDir` | str | None | Directory to save checkpoints |

### Training Commands

```bash
# === STANDARD TRAINING (with default hyperparameters) ===

# Train on 10Hz dataset with multi-seed for fair comparison
python train.py --datasetType 10Hz --multiSeed

# Train on 30Hz dataset with multi-seed
python train.py --datasetType 30Hz --multiSeed

# Single seed training
python train.py --datasetType 10Hz --seed 42


# === TRAINING WITH HPO BEST PARAMETERS ===

# Train on 10Hz using HPO-tuned parameters
python train.py --datasetType 10Hz --hpoParams outputs/hpo_optuna/best_params.json --multiSeed

# Train on 30Hz using HPO-tuned parameters
python train.py --datasetType 30Hz --hpoParams outputs/hpo_optuna/best_params.json --multiSeed


# === RESUME/CUSTOM TRAINING ===

# Resume from checkpoint
python train.py --datasetType 10Hz --resume checkpoints/model_epoch_50.pth

# Custom batch size for 8GB VRAM
python train.py --datasetType 10Hz --batchSize 8 --multiSeed
```

### Training Output

Training artifacts are saved to `outputs/`:
- `checkpoints/`: Model checkpoints (best and periodic)
- `logs/`: Training logs with loss and metric history
- `config.json`: Training configuration snapshot

---

## Evaluation

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint` | str | Required | Path to model checkpoint |
| `--datasetType` | str | 10Hz | Dataset variant |
| `--split` | str | test | Evaluation split (val/test) |
| `--batchSize` | int | 16 | Evaluation batch size |
| `--visualize` | flag | False | Generate prediction visualizations |
| `--outputDir` | str | outputs/eval | Output directory |

### Example Commands

```bash
# Standard evaluation on test set
python evaluate.py --checkpoint outputs/checkpoints/best_model.pth --split test

# Evaluation with visualization
python evaluate.py --checkpoint outputs/checkpoints/best_model.pth --visualize

# Evaluation on validation set
python evaluate.py --checkpoint outputs/checkpoints/best_model.pth --split val
```

### Evaluation Output

Results are saved as JSON and CSV files containing:
- Per-agent metrics (minADE, minFDE, MR, NormFDE)
- Aggregate statistics with confidence intervals
- Prediction visualizations (if enabled)

---

## Hyperparameter Optimization

PrayagGameFormer supports Bayesian hyperparameter optimization via Optuna with optional Ray Tune integration for distributed search. HPO uses the **Stratified dataset (20% sample)** for faster optimization.

### HPO Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--datasetType` | str | 10Hz | Dataset variant (10Hz or 30Hz) |
| `--numTrials` | int | 30 | Number of HPO trials |
| `--numEpochs` | int | 10 | Epochs per trial |
| `--patience` | int | 4 | Early stopping patience per trial |
| `--backend` | str | optuna | HPO backend (optuna or ray) |
| `--seed` | int | 17 | HPO seed for reproducibility |

### HPO Commands

```bash
# === HPO ON 10Hz STRATIFIED DATASET ===
python tune.py --datasetType 10Hz --numTrials 30 --numEpochs 10 --patience 4

# === HPO ON 30Hz STRATIFIED DATASET ===
python tune.py --datasetType 30Hz --numTrials 30 --numEpochs 10 --patience 4

# === QUICK TEST RUN ===
python tune.py --datasetType 10Hz --numTrials 3 --numEpochs 2

# === DISTRIBUTED HPO (Ray Tune) ===
python tune.py --backend ray --datasetType 10Hz --numTrials 30 --numEpochs 10
```

### Dataset Configuration

| Mode | Dataset | Description |
|------|---------|-------------|
| **HPO** | StratifiedProjectPrayagBEVDataset(10Hz) | 20% sample for fast HPO |
| **Training** | ChunkedProjectPrayagBEVDataset(10Hz) | Full dataset for training |

### HPO Output

Results are saved with SQLite persistence for resume capability:
- `outputs/hpo_optuna/optuna_study.db`: Optuna study database
- `outputs/hpo_optuna/best_params.json`: Best hyperparameters (use with `--hpoParams`)
- `outputs/hpo_optuna/hpo_history.csv`: Trial history

---

## Configuration

Key configuration parameters in `config.py`:

### Temporal Settings

| Parameter | 10Hz Dataset | 30Hz Dataset | Description |
|-----------|--------------|--------------|-------------|
| obsHorizon | 10 | 30 | Observation frames (1 second) |
| predHorizon | 30 | 90 | Prediction frames (3 seconds) |
| sampleRate | 10 | 30 | Frames per second |

### Model Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| dim | 256 | Hidden dimension |
| heads | 8 | Multi-head attention heads |
| encoderLayers | 6 | Transformer encoder layers |
| decoderLevels | 3 | Number of Level-k reasoning levels |
| numModes | 6 | Number of prediction modes |
| maxAgents | 100 | Maximum agents per scene |
| neighborsToPredict | 31 | Agents to predict (excluding ego) |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| lr | 1e-4 | Learning rate |
| weightDecay | 1e-4 | AdamW weight decay |
| epochs | 50 | Training epochs |
| valInterval | 1 | Validation frequency |

---

## Dataset

### ChunkedProjectPrayagBEVDataset

| Property | 10Hz | 30Hz |
|----------|------|------|
| Resolution | 1920 x 1080 | 1920 x 1080 |
| Frame Rate | 10 FPS | 30 FPS |
| Observation | 1.0 seconds | 1.0 seconds |
| Prediction | 3.0 seconds | 3.0 seconds |
| Scenes | 5 | 5 |
| Splits | train/val/test | train/val/test |

### Data Pipeline

1. **Trajectory Loading**: JSON annotation files with per-frame bounding boxes
2. **Map Encoding**: Lane polyline extraction from skeletonized road masks
3. **Agent Selection**: Priority-based selection of agents to predict
4. **Feature Extraction**: Position, velocity, heading, and size encoding

---

## Evaluation Metrics

### Displacement Metrics

| Metric | Description |
|--------|-------------|
| minADE@K | Minimum Average Displacement Error over K modes |
| minFDE@K | Minimum Final Displacement Error over K modes |
| MR@K | Miss Rate (FDE > 2.0m threshold) |
| NormFDE | FDE normalized by prediction horizon |

### Safety Metrics

| Metric | Description |
|--------|-------------|
| CR | Collision Rate between predicted trajectories |
| ORR | Off-Road Rate (predictions outside valid regions) |

### Uncertainty Metrics

| Metric | Description |
|--------|-------------|
| NLL | Negative Log-Likelihood of GMM predictions |
| APD | Average Path Diversity across modes |

### Model Selection Score (MSS)

Unified scoring metric combining displacement accuracy with safety:

```
MSS = (minADE/100) + (minFDE/100) + 10 * (CR + ORR)
```

Lower MSS indicates better overall performance.

---

## Linux Server Deployment

### Environment Setup

```bash
# Clone repository to server
git clone <repository_url> ~/LaneGameFormer
cd ~/LaneGameFormer

# Create conda environment from YAML
conda env create -f environment.yml
conda activate lanegameformer

# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

### Data Transfer

```bash
# Transfer dataset to server (from local machine)
rsync -avz --progress ChunkedProjectPrayagBEVDataset10Hz/ user@server:~/LaneGameFormer/ChunkedProjectPrayagBEVDataset10Hz/

# Or use SCP for smaller transfers
scp -r ChunkedProjectPrayagBEVDataset10Hz/ user@server:~/LaneGameFormer/
```

### Running Training

```bash
# Navigate to model directory
cd ~/LaneGameFormer/PrayagGameFormer

# Run training with nohup for persistence
nohup python train.py --datasetType 10Hz --batchSize 16 --numEpochs 50 > train.log 2>&1 &

# Monitor training
tail -f train.log

# Or use tmux for interactive sessions
tmux new -s gameformer_train
python train.py --datasetType 10Hz --batchSize 16 --numEpochs 50
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t gameformer_train
```

### Running HPO

```bash
# Full HPO study (background)
nohup python tune.py --numTrials 70 --numEpochs 10 > hpo.log 2>&1 &

# Monitor HPO progress
tail -f hpo.log
```

### SLURM Job Submission (HPC Clusters)

```bash
#!/bin/bash
#SBATCH --job-name=prayag_gameformer
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module load cuda/12.1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lanegameformer

cd ~/LaneGameFormer/PrayagGameFormer
python train.py --datasetType 10Hz --batchSize 16 --numEpochs 50
```

### GPU Memory Management

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Clear GPU memory if needed
python -c "import torch; torch.cuda.empty_cache()"

# For larger models (384 dim), use gradient checkpointing
python train.py --datasetType 10Hz --batchSize 8 --gradientCheckpoint
```

---

## References

[1] Huang, Z., Liu, H., Wu, J., and Lv, C. **GameFormer: Game-theoretic Modeling and Learning of Transformer-based Interactive Prediction and Planning for Autonomous Driving**. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[2] Liang, M., Yang, B., Hu, R., Chen, Y., Liao, R., Feng, S., and Urtasun, R. **Learning Lane Graph Representations for Motion Forecasting**. In Proceedings of the European Conference on Computer Vision (ECCV), 2020.

[3] Camerer, C. F., Ho, T.-H., and Chong, J.-K. **A Cognitive Hierarchy Model of Games**. Quarterly Journal of Economics, 119(3), 2004.

---

## License

This implementation is provided for research purposes under the terms of the original GameFormer license.
