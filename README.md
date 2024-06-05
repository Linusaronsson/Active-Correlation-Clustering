# Active Correlation Clustering
This repository contains code for the paper: [Correlation Clustering with Active Learning of Pairwise Similarities](https://openreview.net/forum?id=Ryf1TVCjBz) (TMLR), 2024.

# Installation

1. **Clone the repository**
```
  git clone https://github.com/Linusaronsson/Active-Correlation-Clustering.git
```

2. **Clone COBRAS repository (for COBRAS/nCOBRAS baselines)**

```
  git clone https://github.com/Linusaronsson/noise_robust_cobras.git
```

3. Optionally setup conda environment
```
  conda create --name acc python=3.9
  conda activate acc
```

4. Install cloned repositories
```
cd Active-Correlation-Clustering
pip install -r .
cd noise_robust_cobras
pip install -r .
```

5. Install other packages

```
  pip install -r requirements.txt
```

# Running experiments

1. See [this](https://github.com/Linusaronsson/Active-Correlation-Clustering/blob/main/notebooks/datasets.ipynb) notebook for details about how datasets were preprocessed. No need to run this code as all datasets have already been generated [here](https://github.com/Linusaronsson/Active-Correlation-Clustering/tree/main/datasets).
2. See [this](https://github.com/Linusaronsson/Active-Correlation-Clustering/blob/main/notebooks/experiments.ipynb) notebook for details about how to run experiments and generate plots.

# Acknowledgements

This work was partially supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation. The computations and data handling were enabled by resources provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS) and the Swedish National Infrastructure for Computing (SNIC) at Chalmers Centre for Computational Science and Engineering (C3SE), High Performance Computing Center North (HPC2N) and Uppsala Multidisciplinary Center for Advanced Computational Science (UPPMAX) partially funded by the Swedish Research Council through grant agreements no. 2022-06725 and no. 2018-05973.
  

  
