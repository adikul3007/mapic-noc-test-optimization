# MAPIC: RL-Based Test-Time Optimization for NoC Systems

This repository contains the official implementation of **MAPIC (Machine Learning Assisted Pairing of IO Pairs to IP Cores)**, a reinforcement learning–based framework for optimizing **test-time in Network-on-Chip (NoC) systems**.

MAPIC formulates the **core-to-IO mapping problem** as a sequence prediction task and solves it using a **Pointer Network trained with Reinforcement Learning (REINFORCE)**.

---

## Paper

[**Reinforcement Learning for Testtime Optimization in the Network-on-Chip based Systems**](https://doi.org/10.1109/ISQED65160.2025.11014406) - 
*ISQED 2025*

---

## Key Contributions

* Pointer Network for **core-to-IO mapping**
* Reinforcement Learning for **test-time minimization**
* Integration with **Shortest Test Duration First (STF)** scheduling
* Outperforms Genetic Algorithm (GA) baseline in both:

  * Test time
  * Runtime efficiency

---

## Problem Overview

In NoC-based systems:

* Multiple IP cores must be tested
* Limited IO pairs are available
* Routing paths may conflict

Goal:

> Find an optimal mapping and schedule that **minimizes total test time**

This is an **NP-hard optimization problem**.

---

## Method Overview

### 1. Pointer Network

* Learns mapping: **cores → IO pairs**
* Encoder: LSTM over core features
* Decoder: Attention-based selection

### 2. Reinforcement Learning

* Policy: ( p(m | s, \theta) )
* Reward: negative total test time
* Optimization: REINFORCE

### 3. Scheduling

* Non-preemptive testing
* STF (Shortest Test First)
* Conflict-aware routing

---

## Project Structure

```bash
.
├── README.md
├── requirements.txt
│
├── baselines/
│   └── genetic_algorithm.py     # GA baseline
│
├── scripts/
│   ├── data.py                 # Data loading / preprocessing
│   ├── gen.py                  # Synthetic NoC generation
│   ├── train.py                # Supervised pretraining
│   ├── rl.py                   # RL fine-tuning (MAPIC)
│   └── inference.py            # Evaluation / testing
│
├── src/
│   ├── main.py                 # Entry point (train / rl / test)
│   │
│   ├── models/
│   │   ├── pointer_net.py      # Pointer Network
│   │   └── attention.py        # Attention mechanism
│   │
│   └── utils/
│       ├── obj_funct.py        # Test-time calculation (objective)
│       ├── tensor_utils.py
│       └── modules.py
```

---

## Installation

```bash
pip install -r requirements.txt
```

### (Optional) GPU Support

Install CUDA-enabled PyTorch from:
https://pytorch.org/get-started/locally/

---

## Usage

All modes are controlled via `main.py`.

### Training (Supervised Pretraining)

```bash
python src/main.py --mode train --num_cores 32 --batch_size 64 --lr 1e-4
```

---

### Reinforcement Learning (MAPIC Optimization)

```bash
python src/main.py --mode rl --num_cores 14 --alpha 0.9 --epoch 5000
```

---

### Testing / Inference

```bash
python src/main.py --mode test --num_cores 32
```

---

## Arguments

| Argument           | Description                                      | Type   | Default Value                          |
|--------------------|--------------------------------------------------|--------|----------------------------------------|
| `--mode`           | Mode of operation: `train`, `rl`, or `test`      | str    | `test`                                 |
| `--model_path`     | Path to the trained model file                   | str    | `ordered_active_search_48cores.pt`      |
| `--num_samples`    | Number of samples during decoding                | int    | `1`                                    |
| `--batch_size`     | Batch size for training/RL                       | int    | `64`                                   |
| `--lr`             | Learning rate                                    | float  | `1e-8`                                 |
| `--wt_decay`       | Weight decay for regularization                  | float  | `1e-5`                                 |
| `--factor`         | Learning rate scheduler factor                   | float  | `0.001`                                |
| `--patience`       | Patience for scheduler/early stopping            | float  | `100`                                  |
| `--dropout`        | Dropout probability in the network               | float  | `0.1`                                  |
| `--n_layers`       | Number of layers in encoder/decoder              | int    | `2`                                    |
| `--input_dim`      | Input dimension for encoder                      | int    | `2`                                    |
| `--hidden_dim`     | Hidden dimension size                            | int    | `2`                                    |
| `--epoch`          | Maximum number of training epochs                | int    | `10000`                                |
| `--num_cores`      | Number of cores in the NoC system                | int    | `7`                                    |
| `--num_test_iter`  | Number of iterations during testing              | int    | `50`                                   |
| `--alpha`          | Discount factor for reinforcement learning       | float  | `0.99`                                 |

---

## Workflow

### 1. Data Generation

* Synthetic NoC benchmarks created using ITC’02 data
* Controlled via `scripts/gen.py`

---

### 2. Pretraining

* Train Pointer Network to learn initial mappings
* Output: trained model weights (`.pt` files)

---

### 3. RL Optimization (MAPIC)

* Fine-tune model using REINFORCE
* Objective: minimize total test time

---

### 4. Inference

* Generate mapping solutions
* Evaluate test-time performance

---

### 5. Baseline Comparison

* Compare with GA implementation:

```bash
python baselines/genetic_algorithm.py
```

---

## Outputs

* Trained models (`.pt`)
* Test-time values (objective function)
* Best mapping solutions
* Performance logs (min/max costs)

---

## Results

MAPIC achieves:

* Lower test-time compared to GA baseline
* Faster convergence during optimization
* Better scalability with increasing core counts

---

## Acknowledgements
This work was supported in part by the [Center for Research Excellence
in Semiconductor Technologies](https://crest.bits-pilani.ac.in/), Birla Institute
of Technology and Sciences (BITS) Pilani, India, under Grant Refencence:
SC/07/23/125.
