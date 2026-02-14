# MVAN

**MVAN: Towards Interpretable Joint Prediction of Multiple Flight Safety Events with Multi-View Attention Network**

**Authors**:  Chengxiang Li, Xu Li, et al.

This repository contains the source code for the paper:

> **MVAN: Towards Interpretable Joint Prediction of Multiple Flight Safety Events with Multi-View Attention Network**

---

## 1. Project Overview

Flight safety risk analysis often involves **multiple correlated safety events** and **heterogeneous flight data views**.  
To address this challenge, we propose **MVAN**, a **Multi-View Attention Network** that jointly predicts multiple flight safety events while providing **interpretable attention-based insights**.

Due to the sensitive nature of **Quick Access Recorder (QAR)** data, **the full raw dataset cannot be publicly released**.  
However, this repository provides:

- The **complete implementation** of the proposed MVAN model  
- **All baseline models** used in both multi-task and single-task settings  
- The **full data preprocessing pipeline**  
- A **limited set of anonymized QAR samples** (several dozen flights) for demonstration and reproducibility

This allows researchers to fully understand, reproduce, and extend our work.

---

## 2. Repository Structure


### 2.1 `baselines/`

This folder contains implementations of **baseline methods** used in the experiments, including:

- Single-task learning models  
- Multi-task learning baselines  
- Traditional machine learning and deep learning approaches  

All baselines follow the same input/output protocol as MVAN for fair comparison.

---

### 2.2 `data/`

This folder stores:

- A **small set of anonymized QAR samples**  
- Example data files used to demonstrate the data format  

⚠️ **Note**  
The complete QAR dataset used in the paper is **not publicly available** due to safety and privacy constraints.

---

### 2.3 `data_process/`

This folder contains the **entire data preprocessing pipeline**, including:

- Raw QAR signal parsing  
- Feature extraction and normalization  
- Multi-view feature construction  
- Label generation for multiple flight safety events  

The provided code reflects exactly the preprocessing procedure described in the paper.

---

## 3. Python Virtual Environment

This project uses a **Conda + pip** based Python virtual environment, designed primarily for **deep learning, graph neural networks (GNNs), and Transformer-based research**.  
The environment has been tested on **Linux (x86_64)**.

---

## 4. System & Environment Overview

- **Operating System**: Linux (x86_64)
- **Python Version**: 3.7.12
- **Package Manager**: Conda (conda-forge channel) + pip
- **CUDA Version**: CUDA 11.x
- **PyTorch Version**: 1.13.0 (CUDA 11)

> ⚠️ **Important**  
> Python 3.7 is officially deprecated.  
> This environment is intended **only for reproducing legacy experiments** reported in the paper.

---

## 5. Requirements

### 5.1 Deep Learning & GPU

- torch==1.13.0  
- torchvision==0.14.0  
- torchaudio==0.13.0  
- nvidia-cuda-runtime-cu11  
- nvidia-cudnn-cu11  

Used for GPU-accelerated training and inference.

---

### 5.2 Graph Learning & GNN

- dgl-cu101==0.6.0.post1  
- torch-geometric==2.3.1  
- networkx==2.6.3  
- python-louvain==0.16  

Supports graph construction, community detection, and graph neural networks.

---

### 5.3 Transformer & Attention Mechanisms

- reformer-pytorch==1.4.4  
- axial-positional-embedding==0.2.1  
- local-attention==1.9.3  
- colt5-attention==0.10.20  
- product-key-memory==0.2.10  
- einops==0.6.1  

Used for efficient attention mechanisms and long-sequence modeling.

---

### 5.4 Scientific Computing & Machine Learning

- numpy==1.21.6  
- scipy==1.7.3  
- pandas==1.3.5  
- scikit-learn==1.0.2  
- sympy==1.10.1  
- mpmath==1.3.0  

---

### 5.5 Visualization & Development Tools

- matplotlib==3.5.3  
- seaborn==0.12.2  
- tqdm==4.66.4  
- ipython==7.33.0  
- ipykernel==6.16.2  
- jupyter-client==7.4.9  
- jupyter-core==4.12.0  
- debugpy==1.7.0  

---

## 6. Environment Setup

```bash
conda create -n MVAN python=3.7
conda activate MVAN
pip install -r requirements.txt
---

## 7. Contact
bruceli.cx@foxmail.com(Chengxiang Li)
