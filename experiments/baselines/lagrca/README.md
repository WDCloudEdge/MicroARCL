# Bridging the Delay: Lag-Aware Spatio-Temporal Causal Inference for Microservice Root Cause Analysis

## 📂 Project Structure

The repository is organized as follows:

```
LagRCA/
├── config.py               # Configuration settings and dataset mappings
├── main.py                 # Entry point for training and evaluation
├── model/                  # Core model definitions
│   ├── LagRCA.py           # Main LagRCA model architecture
│   ├── LagCrossNodeAttention.py # Attention mechanism implementation
│   └── MultiLagCausalLearner.py # Causal learning module
├── module/                 # Utility modules
│   ├── DataProcessor.py    # Data preprocessing and loading
│   ├── RootCauseScorer.py  # Root cause scoring and ranking logic
│   ├── NodeDecoder.py      # Decoder module for node metrics
│   ├── InstanceEmbedding.py     # Aggregates per-metric features into pod-level embeddings
│   └── TemporalEncoder.py  # Transformer-based temporal encoder for metric sequences
├── data/                   # Dataset directory
│   ├── D1/                 # Dataset D1
│   └── D2/                 # Dataset D2
└── README.md               # This file
```

## 📝 Dataset

D1 is derived from a production-grade, high-fidelity microservice system deployed on the cloud infrastructure of a top-tier commercial bank.
The system consists of 46 microservice instances running across multiple virtual machines and follows realistic e-commerce traffic patterns observed in daily operations.
The dataset covers five representative infrastructure failure categories frequently encountered in practice, including CPU Anomaly, Memory Exhaustion, Service Interruption, Storage Capacity, and I/O Contention.
Each collected record is annotated with the corresponding root-cause instances and failure categories.

D2 is built on Online Boutique, a widely used open-source microservice benchmark deployed on a Kubernetes cluster. It contains 41 microservice instances and represents a standard cloud-native architecture. To obtain precise ground truth and controlled failure scenarios, we inject faults spanning the entire stack, comprehensively covering infrastructure-level issues (Network, Storage, Resource Stress, and Pod Lifecycle) as well as application-level anomalies (JVM Runtime and Application Logic errors).

We have preprocessed two datasets and placed them in the following folder.

D1: LagRCA/data/D1

D2: LagRCA/data/D2

The preprocessed dataset contains the following files:

- `normal_data.pkl`: Multivariate time-series data under normal (non-failure) conditions for all microservice instances.
- `case_data.pkl`: Multivariate time-series data for failure cases, typically covering a 20-minute time window around each incident, including metrics and corresponding labels.
- `adj.pkl`: Adjacency matrix describing the call graph / dependency structure among microservice instances, used to construct the graph for LagRCA.




## 🛠️ Requirements

- Python 3.9+
- PyTorch >= 2.1
- PyTorch Geometric
- Pandas
- NumPy
- Tqdm
- Scikit-learn


## 🚀 Usage

### 1. Data Preparation
Please unzip D1.zip and D2.zip.

### 2. Training and Evaluation
To train the model and evaluate root cause analysis performance, use `main.py`.

#### Basic Usage:
```bash
python main.py --dataset D1
```

#### Full Command with Arguments:
```bash
python main.py \
  --dataset D1 \
  --epochs 100 \
  --batch_size 16 \
  --lr 0.001 \
  --window_size 10 \
  --stride 1
```

### Arguments:
- `--dataset`: Dataset name (`D1`, `D2`). Default: `D1`.
- `--epochs`: Number of training epochs. Default: `100`.
- `--batch_size`: Batch size for training. Default: `16`.
- `--lr`: Learning rate. Default: `0.001`.
- `--window_size`: Size of the sliding time window. Default: `10`.
- `--stride`: Step size for the sliding window. Default: `1`.
- `--limit`: Maximum number of cases to evaluate. By default, evaluate all cases.
