# Low-Rank Selective Aggregation in Federated Learning (Global SVD)

**Author:** Yichen Zhou  
**Supervisor:** Prof. Long Feng  
**Affiliation:** Department of Statistics and Actuarial Science / School of Computing and Data Science, The University of Hong Kong  

---

## Overview

Fine-tuning large pre-trained models in federated learning (FL) is both resource- and time-intensive.  
This project explores a **low-rank SVD-based aggregation strategy** — called **Global SVD** — to enhance the efficiency of federated fine-tuning by reducing parameters while preserving client-specific information.

The work is inspired by **FedSA-LoRA** (*Guo et al., 2024*), which selectively aggregates low-rank components during federated adaptation.

---

## Experiment Setup

| **Component** | **Details** |
|----------------|-------------|
| **Model** | CNN: 2 Conv layers + 2 FC layers |
| **Dataset** | CIFAR-10 (Airplane vs Automobile) |
| **Clients** | 3 |
| **Batch Size** | 16 |
| **Noise** | 0.1 |
| **Non-IID Splits** | Airplane ratio (0.9, 0.1, 0.5) |
| **Ranks Tested** | k = 3, 4, 5, 6 |

**Comparison Methods**

- **Global SVD:** Server-side SVD aggregation  
- **Local SVD:** Client-side SVD with A-matrix averaging (FedSA-like)

---

## Methodology

### 1. Global SVD Aggregation

The **Global SVD** approach performs SVD decomposition on the **server side** after collecting client updates:

**Mathematical Formulation:**

$$
\text{Global SVD:}
$$
$$
A = U_k \sqrt{\Sigma_k} \quad (\text{Shared})
$$
$$
B_i^T = \sqrt{\Sigma_k} V_k^T[:, \hat{i}:(\hat{i}+1)p] \quad (\text{Client-Specific})
$$

**Weight Update:**
$$
AB_i^T + W
$$

Where:
- $A \in \mathbb{R}^{d \times k}$: Shared low-rank left singular vectors (aggregated across clients)
- $B_i^T \in \mathbb{R}^{k \times p}$: Client-specific right singular vectors for client $i$
- $k$: Low-rank approximation rank
- $p$: Output dimension
- $W$: Original pre-trained weights

### 2. Local SVD Aggregation 

The **Local SVD** approach performs SVD decomposition on the **client side** and averages the $A$ matrices:

**Mathematical Formulation:**

$$
\text{Local SVD: Average of } A \text{ matrices}
$$
$$
A_{\text{global}} = \frac{1}{N} \sum_{i=1}^N A_i
$$

**Weight Update:**
$$
A_{\text{global}} B_i + W
$$

Where:
- $A_{\text{global}}$: Averaged left singular vectors across all $N$ clients
- $B_i$: Client-specific right singular vectors

---


