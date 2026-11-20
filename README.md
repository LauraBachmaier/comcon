# ComCon: Community Connectivity Package

[![DOI](https://zenodo.org/badge/1465528245.svg)](https://doi.org/10.5281/zenodo.24404291)

**Version 1.0.0** | Laura Bachmaier | University of Plymouth

ComCon is a Python package for multi-species marine larval dispersal modelling using Lagrangian particle tracking with OceanParcels and NEMO hydrodynamic data. It offers two workflow pathways: a quick **passive-dispersal** overview (following James et al. 2025) and a full **trait-defined** workflow that uses species-specific planktonic larval durations, spawning windows and habitat preferences.

## Quick Start

### 1. Download ComCon

On this GitHub page, click the green **Code** button and then **Download ZIP**. Unzip it somewhere on your computer. The unzipped folder is automatically called `comcon-main` and you do not need to rename it. We recommend a local folder rather than a cloud-synced one such as OneDrive or Dropbox, as syncing can interfere with the simulation output files (see the User Guide).

The **Releases** section on the right of this page has fixed versions of ComCon, each with its own DOI. The Download ZIP above always gives the latest version, which may include small fixes made since the last release. Use a release if you want to cite the exact version you used. If you use Git, you can run `git clone https://github.com/LauraBachmaier/comcon.git` instead.

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
| **Passive dispersal** | A quick overview; data-poor scenarios | Releases particles from all ocean cells and extracts connectivity at fixed timepoints (7, 14, 21 and 28 days) for a summer and a winter run. No species traits required. Follows James et al. (2025). |
| **Trait-defined** | Species-specific connectivity | Constrains release, dispersal and settlement using per-species planktonic larval duration, spawning window and habitat preference for a biologically-informed analysis. |

## Trait-Defined Pathway: Three Operating Modes

| Mode | Best For | Description |
|------|----------|-------------|
| **Interactive** | First-time users | Step-by-step guidance with explanations |
| **Automatic** | Repeat analyses | Collect parameters upfront, run end-to-end |
| **Manual** | Custom workflows | Access individual functions for more flexibility |

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

The built-in trait database (`comcon/data/trait_database.csv`) gives spawning windows and planktonic larval durations for 28 rocky shore species. Most values come from BIOTIC, the Biological Traits Information Catalogue (MarLIN, 2006). The other values come from the published literature. A full table of sources for each species and value will be added in a later release (planned for early 2027).

## Citing ComCon

If you use ComCon, please cite the version you used:

> Bachmaier, L. (2026). ComCon: Community Connectivity Package (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.24404292

To cite ComCon in general (all versions), use https://doi.org/10.5281/zenodo.24404291. Each new release gets its own DOI, listed on Zenodo and the Releases page. GitHub's **Cite this repository** button (on the right of this page) gives the citation in other formats.

## Acknowledgements

ComCon was developed as part of my PhD at the University of Plymouth, supervised by Pete Cotton, Louise Firth, David Bilton, Kerry Howell and Antony Knights. The Louvain community detection is based on code by Antony Knights. The reflective boundary (unbeaching) method is adapted from Clubley et al. (2024) and their code (https://github.com/cclubley/Multi-generational-dispersal). The passive dispersal pathway follows James et al. (2025). Thank you to Giada Riva for testing ComCon on Mediterranean data.

## Use of AI tools

Generative AI tools were used during the development of ComCon and its documentation. Between 2024 and mid 2025, OpenAI ChatGPT (GPT-4o to GPT-4.1) was used as a coding aid, in a similar way to an online programming forum, to suggest and troubleshoot short code snippets, which I assembled into the working functions. From May 2025, Anthropic Claude (Sonnet 4 to Claude Opus 5.5, through claude.ai and Claude Code) was used to restructure this code into a Python package, including refactoring, debugging, memory optimisation and packaging and to generate some parts of the analysis and visualisation functions. Claude was also used to draft parts of the user guide and docstrings. The scientific design of ComCon, including the trait-based post-processing approach, the architectural workflow, the choice and implementation of analyses and visualisations and the parameter defaults, was developed by me. All AI-generated code and text were reviewed, modified and tested by me, and I take full responsibility for the software.

## Support

Please report bugs or ask questions by opening an issue on the **Issues** tab of this GitHub page. You can also email laura.bachmaier@plymouth.ac.uk.

## Licence

ComCon is released under the MIT licence (see `LICENSE`).

## References

Clubley, C.H., Silva, T.A.M., Wood, L.E., Firth, L.B., Bilton, D.T., O'Dea, E. and Knights, A.M. (2024) Multi-generational dispersal and dynamic patch occupancy reveals spatial and temporal stability of seascapes. *Science of the Total Environment*, 952, 175762. https://doi.org/10.1016/j.scitotenv.2024.175762

James, M.K., Lessin, G., Al Azhar, M., Bedington, M., Clubley, C.H., Somerfield, P.J. and Knights, A.M. (2025) The 'everything is everywhere' framework: holistic network analysis as a marine spatial management tool. *Ecological Informatics*, 87, 103105. https://doi.org/10.1016/j.ecoinf.2025.103105

MarLIN (2006) BIOTIC - Biological Traits Information Catalogue. Marine Life Information Network. Plymouth: Marine Biological Association of the United Kingdom. Available from www.marlin.ac.uk/biotic
