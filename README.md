# ComCon: Community Connectivity Package

**Version 1.0.0** | Laura Bachmaier | University of Plymouth

ComCon is a Python package for multi-species marine larval dispersal modelling using Lagrangian particle tracking with OceanParcels and NEMO hydrodynamic data. It offers two workflow pathways: a quick **passive-dispersal** overview (following James et al. 2025) and a full **trait-defined** workflow that uses species-specific planktonic larval durations, spawning windows and habitat preferences.

## Quick Start

### 1. Install Miniconda

Download and install Miniconda from: https://docs.conda.io/en/latest/miniconda.html

See the User Guide (`docs/ComCon_User_Guide_V2.html`) for detailed installation instructions.

### 2. Create the Environment

Open a terminal (Anaconda Prompt on Windows) and run these commands one after the other:

```bash
# Navigate to the comcon_distribution folder (adjust path as needed)
cd path/to/comcon_distribution

# Create environment with all dependencies
conda create -n comcon -c conda-forge python=3.10 parcels cartopy geopandas jupyter ipywidgets folium ipyleaflet psutil tqdm networkx python-louvain ffmpeg scikit-learn seaborn scipy

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

This shows a welcome screen and prompts you to choose a pathway (Passive or Trait-defined) and, for the trait-defined pathway, a mode (Interactive, Automatic or Manual). Follow the prompts from there.

## Two Pathways

| Pathway | Best For | Description |
|---------|----------|-------------|
| **Passive dispersal** | A quick overview; data-poor settings | Releases particles from all ocean cells and extracts connectivity at fixed timepoints (7, 14, 21 and 28 days) for a summer and a winter run. No species traits required. Follows James et al. (2025). |
| **Trait-defined** | Species-specific connectivity | Constrains release, dispersal and settlement using per-species planktonic larval duration, spawning window and habitat preference for a biologically-informed analysis. |

## Trait-Defined Pathway: Three Operating Modes

| Mode | Best For | Description |
|------|----------|-------------|
| **Interactive** | First-time users | Step-by-step guidance with explanations |
| **Automatic** | Repeat analyses | Collect parameters upfront, run end-to-end |
| **Manual** | Custom workflows | Access individual functions for scripting |

## Documentation

See the **ComCon User Guide** (`docs/ComCon_User_Guide_V2.html`) for comprehensive instructions including:

- Detailed installation walkthrough with screenshots
- Step-by-step workflow guide for both pathways
- Data preparation instructions
- Troubleshooting common issues
- Glossary of terms

Worked examples are provided as Jupyter notebooks in `docs/vignettes/`: single-species quickstarts for each mode, a multi-species connectivity example, and a trait-uncertainty hypothesis-testing example.

## Requirements

- Python 3.10
- OceanParcels >= 3.0
- Copernicus Marine Service account (free)
- 8GB+ RAM recommended

## Support

For questions or issues, contact: laura.bachmaier@plymouth.ac.uk
