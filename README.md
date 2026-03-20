# SpatioEv

Spatial feature extraction and spatial evolution framework for multiplexed imaging.

SpatioEv is a modular Python package for:
- Segmentation quality control (QC)
- Imaging artifacts removal (QC)
- ECM-Cell interaction
- Per-cell spatial feature extraction
- Spatial neighbourhood modelling
- Trajectory-ready feature engineering

The package is designed to integrate with the scverse ecosystem (AnnData / Scanpy) while remaining modular and lightweight.

---

## 🚀 Installation

### 1️⃣ Create a Dedicated Environment (Recommended)

We strongly recommend installing SpatioEv in a clean conda environment to avoid dependency conflicts.

```bash
conda create -n spatioev_env python=3.11
conda activate spatioev_env
```

### 2️⃣ Navigate to the Project Root

```bash
git clone https://github.com/Bashford-Rogers-lab/SpatioEv.git
cd SpatioEv
```

### 3️⃣ Install
Core Installation

```bash
pip install -e .
```

This installs core dependencies:
	•	numpy
	•	pandas
	•	scipy
	•	matplotlib
	•	seaborn
	•	anndata
	•	scikit-image
    •	scanpy

Full scverse / Spatial Installation (Optional)
If you plan to use Scanpy, Squidpy, or SpatialData functionality:

```bash
pip install -e ".[scverse]"
```
### ✅ Verify Installation

```python
import spatioev
import spatioev.qc
import spatioev.plot
```

If using spatial ecosystem features:
```python
import scanpy
import squidpy
```

