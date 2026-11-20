# ComCon: Community Connectivity Package

**Version 1.0.0** | Laura Bachmaier | University of Plymouth

ComCon is a Python package for multi-species marine larval dispersal modelling using Lagrangian particle tracking with OceanParcels and NEMO hydrodynamic data. It offers two workflow pathways: a quick **passive-dispersal** overview (following James et al. 2025) and a full **trait-defined** workflow that uses species-specific planktonic larval durations, spawning windows and habitat preferences.

## Quick Start

### 1. Download ComCon

On this GitHub page, click the green **Code** button and then **Download ZIP**. Unzip it somewhere on your computer (not in a OneDrive or Dropbox folder). The folder will be called `comcon-main`.

You can also download a fixed version from the **Releases** section on the right of this page. Use a release if you want to cite the exact version you used. If you use Git, you can run `git clone https://github.com/LauraBachmaier/comcon.git` instead.

### 2. Install Miniconda

Download and install Miniconda from: https://docs.conda.io/en/latest/miniconda.html

See the User Guide (`docs/ComCon_User_Guide_V2.html`) for detailed installation instructions.

### 3. Create the Environment

Open a terminal (Anaconda Prompt on Windows) and run these commands one after the other:

```bash
# Go to the ComCon folder (adjust the path to where you unzipped it)
cd path/to/comcon-main

# Create the environment with all dependencies (reads environment.yml in this folder)
conda env create -f environment.yml

# Activate the environment
conda activate comcon

# Install ComCon
pip install -e .
```

### 4. Launch Jupyter Lab

```bash
jupyter lab
```

### 5. Start Using ComCon

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

- Python 3.11
- OceanParcels >= 3.0
- Copernicus Marine Service account (free)
- 8GB+ RAM recommended

## Trait database

The built-in trait database (`comcon/data/trait_database.csv`) was compiled from the published literature. A full table of sources for each trait value will be added in a later release (planned for early 2027).

## Citing ComCon

If you use ComCon, please cite the version you used:

> Bachmaier, L. (2026). ComCon: Community Connectivity Package (v1.0.0). Zenodo. DOI to follow.

The DOI for each release will be listed here and on the Releases page. GitHub's **Cite this repository** button (on the right of this page) gives the citation in other formats.

## Acknowledgements

ComCon was developed as part of my PhD at the University of Plymouth, supervised by Antony Knights, Pete Cotton and Louise Firth. The Louvain community detection is based on code by Antony Knights. The reflective boundary (unbeaching) method is adapted from Clubley et al. (2024) (https://github.com/cclubley/Multi-generational-dispersal). The passive dispersal pathway follows James et al. (2025). Thank you to Giada Riva for testing ComCon on Mediterranean data.

## Use of AI tools

I designed ComCon and developed the science and the workflow. AI tools (Claude and Claude Code, by Anthropic) were used to help write, debug and reorganise parts of the code and to draft parts of the documentation. I reviewed and tested these changes. A full manual review of the code is under way for the next release.

## Support

Please report bugs or ask questions by opening an issue on the **Issues** tab of this GitHub page. You can also email laura.bachmaier@plymouth.ac.uk.

## Licence

ComCon is released under the MIT licence (see `LICENSE`).
