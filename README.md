# ComCon: Community Connectivity Package

**Version 1.0.0** | Laura Bachmaier | University of Plymouth

ComCon is a Python package for multi-species marine larval dispersal modelling using Lagrangian particle tracking with OceanParcels and NEMO hydrodynamic data.

## Quick Start

### 1. Install Miniconda

Download and install Miniconda from: https://docs.conda.io/en/latest/miniconda.html

See the User Guide (`docs/ComCon_User_Guide_V1.html`) for detailed installation instructions.

### 2. Create the Environment

Open a terminal (Anaconda Prompt on Windows) and run these commands one after the other:

```bash
# Navigate to the comcon_distribution folder (adjust path as needed)
cd path/to/comcon_distribution

# Create environment with all dependencies
conda conda create -n comcon -c conda-forge python=3.10 parcels cartopy geopandas jupyter ipywidgets folium ipyleaflet psutil tqdm networkx python-louvain ffmpeg scikit-learn seaborn scipy

# Activate the environment
conda activate comcon

# Install ComCon
pip install -e .
```

### 3. Launch Jupyter Lab

```bash
jupyter lab
```

### 4. Start Using ComCon

In a new notebook cell:

```python
import comcon
```

Select "Interactive Mode" and follow the prompts!

## Documentation

See the **ComCon User Guide** (`docs/ComCon_User_Guide_V1.html`) for comprehensive instructions including:

- Detailed installation walkthrough with screenshots
- Step-by-step workflow guide
- Data preparation instructions
- Troubleshooting common issues
- Glossary of terms

## Test Your Installation

Open `examples/test_installation.ipynb` in Jupyter Lab and run all cells to verify everything is working.

## Three Operating Modes

| Mode | Best For | Description |
|------|----------|-------------|
| **Interactive** | First-time users | Step-by-step guidance with explanations |
| **Automatic** | Repeat analyses | Collect parameters upfront, run automatically |
| **Manual** | Custom workflows | Access individual functions for scripting |

## Requirements

- Python 3.10
- OceanParcels >= 3.0
- Copernicus Marine Service account (free)
- 8GB+ RAM recommended

## Support

For questions or issues, contact: laura.bachmaier@plymouth.ac.uk
