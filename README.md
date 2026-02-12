# MVAN
MVAN: Towards Interpretable Joint Prediction of Multiple Flight Safety Events with Multi-View Attention Network

Authors: Jiaxing Shang, Chengxiang Li, Xu Li, et al.

This project contains the source code for the paper:

**MVAN: Towards Interpretable Joint Prediction of Multiple Flight Safety Events with Multi-View Attention Network**


Due to the sensitive nature of the QAR data, we provide the complete implementation of our proposed approach and all baseline models used in both multi-task and single-task experiments, along with the full data-processing code and a limited set of several dozen anonymized QAR samples


# Python Virtual Environment

This project uses a **Conda + pip** based Python virtual environment, designed primarily for **deep learning, graph neural networks (GNNs), and Transformer-based research**.  
The environment has been tested and validated on **Linux (x86_64)**.

---

## 1. System & Environment Overview

- **Operating System**: Linux (x86_64)
- **Python Version**: 3.7.12
- **Package Manager**: Conda (conda-forge channel) + pip
- **CUDA Version**: CUDA 11.x
- **PyTorch Version**: 1.13.0 (CUDA 11)

> ⚠️ **Important**  
> This environment relies on **Python 3.7**, which is officially deprecated.  
> It is recommended **only for reproducing legacy experiments or maintaining existing projects**.

---

## Requirements

### Deep Learning & GPU

- torch==1.13.0  
- torchvision==0.14.0  
- torchaudio==0.13.0  
- nvidia-cuda-runtime-cu11  
- nvidia-cudnn-cu11  

Used for GPU-accelerated training and inference.

---

###Graph Learning & GNN

- dgl-cu101==0.6.0.post1  
- torch-geometric==2.3.1  
- networkx==2.6.3  
- python-louvain==0.16  

Provides support for graph construction, community detection, and graph neural networks.

---

### 2.3 Transformer & Attention Mechanisms

- reformer-pytorch==1.4.4  
- axial-positional-embedding==0.2.1  
- local-attention==1.9.3  
- colt5-attention==0.10.20  
- product-key-memory==0.2.10  
- einops==0.6.1  

Used for efficient attention mechanisms and long-sequence Transformer models.

---

###  Scientific Computing & Machine Learning

- numpy==1.21.6  
- scipy==1.7.3  
- pandas==1.3.5  
- scikit-learn==1.0.2  
- sympy==1.10.1  
- mpmath==1.3.0  

Supports numerical computation, classical machine learning, and mathematical modeling.

---

###  Visualization & Analysis

- matplotlib==3.5.3  
- seaborn==0.12.2  
- tqdm==4.66.4  

Used for experiment visualization and progress tracking.

--
### Jupyter & Development Tools

- ipython==7.33.0  
- ipykernel==6.16.2  
- jupyter-client==7.4.9  
- jupyter-core==4.12.0  
- debugpy==1.7.0  



