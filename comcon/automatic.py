# region Automatic Mode Functions
"""
ComCon Automatic Mode
=====================

This module provides automatic workflow execution for the trait-defined pathway of ComCon.
All parameters are collected upfront, then the entire workflow runs with progress updates only (no interactive prompts during execution).

Note:
    Automatic mode applies only to the trait-defined pathway.
    The passive dispersal pathway (Path A) is already non-interactive by design and has its own entry point: `passive_dispersal_pathway()` in `comcon.passive`.

Usage:
    from comcon.automatic import start_automatic_mode
    start_automatic_mode()

How It Works:
-------------
1. start_automatic_mode() asks if this is a new analysis or re-run
2. For new analysis: collects ALL parameters upfront via prompts
3. For re-run: loads parameters from a previously saved config file
4. The workflow then runs automatically with progress updates only
5. No further prompts during execution - just progress messages

Comparison with Interactive Mode:
---------------------------------
- Interactive: Prompts at each step, can pause and edit between steps
- Automatic: All prompts upfront, then runs start-to-finish unattended
"""

# region Import Modules
import os
import re
import json
import gc
import math
import builtins
from unittest.mock import patch
import csv
import time
import warnings
import traceback
import getpass
import copy
import pickle
import itertools
from datetime import timedelta, datetime
from operator import attrgetter
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import geopandas as gpd
from shapely.geometry import Point, Polygon
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm
import psutil
import contextlib
from scipy.spatial import cKDTree
import io
from IPython.display import display, HTML
import networkx as nx
from parcels import (
    AdvectionRK4, FieldSet, JITParticle, ParticleSet, Variable,
    Field, ParcelsRandom, VectorField, DiffusionUniformKh, StatusCode
)

warnings.filterwarnings('ignore')
import logging
logging.getLogger("urllib3").setLevel(logging.ERROR)
# endregion

# region Standard Formatting
# Dynamic Width Formatting
width = 92  # Fallback for non-HTML contexts

from comcon.interactive import (
    display_line, section_header, subsection_header,
    info_box, data_box, status_message, code_box,
    centered_text, completion_message, setup_project_directory,
    load_builtin_traits, get_valid_path, create_map,
    plot_spp_occ, plot_masks, create_custom_kernels,
)

# endregion

# region Helper Functions
## Standardise species names
def standardise_species_name(name):
    """Standardise species names to use underscores consistently."""
    if name is None:
        return None
    return str(name).replace(' ', '_')

## Displaying species names
def display_species_name(name):
    """Format species name for display (replace underscores with spaces)."""
    if name is None:
        return None
    return str(name).replace('_', ' ')

## Memory monitor
class MemoryMonitor:
    """Context manager for monitoring memory usage during operations."""

    def __init__(self, operation_name):
        self.operation_name = operation_name

    def __enter__(self):
        self.start_mem = psutil.Process().memory_info().rss / 1024 ** 3
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Silent in automatic mode
        return False

## Checking system memory
def check_system_memory():
    """Check available system memory and return in GB."""
    return psutil.virtual_memory().available / (1024 ** 3)

## Getting the memory configuration setup
def get_memory_config(available_gb, runtime_days=None, output_hours=None):
    """Determine memory configuration based on available RAM."""
    if available_gb >= 16:
        return {'system_type': 'high', 'chunk_size': 100000}
    elif available_gb >= 8:
        return {'system_type': 'medium', 'chunk_size': 50000}
    else:
        return {'system_type': 'limited', 'chunk_size': 25000}

## Get validation report
def get_validated_input(prompt, valid_options, default=None):
    """Get validated user input with error handling."""
    while True:
        if default:
            user_input = input(f"{prompt} [default: {default}]: ").strip() or default
        else:
            user_input = input(f"{prompt}: ").strip()

        if user_input.lower() in [str(v).lower() for v in valid_options]:
            return user_input
        else:
            print(f"  Invalid input. Please enter one of: {', '.join(str(v) for v in valid_options)}")

## Error handling for inputs
## Analyses error handling function
def get_yes_no(prompt, default='no'):
    """
    Get yes/no input with validation and re-prompting.

    Parameters
    ----------
    prompt : str
        The prompt to display
    default : str
        Default value if user presses Enter ('yes' or 'no')

    Returns
    -------
    bool
        True for yes, False for no
    """
    while True:
        response = input(f"{prompt} [default: {default}]: ").strip().lower()
        if response == '':
            return default.lower() in ['yes', 'y']
        if response in ['yes', 'y']:
            return True
        if response in ['no', 'n']:
            return False
        print("  Invalid input. Please enter 'yes' or 'no'.")

# region Main Entry Point
def start_automatic_mode():
    """
    Start the automatic workflow mode.

    This is the main entry point for automatic mode. It guides the user
    through setup and then runs the entire workflow automatically.

    Usage:
        from comcon.automatic import start_automatic_mode
        start_automatic_mode()
    """
    section_header("Automatic Mode")

    print("""
        Automatic mode runs the entire workflow with parameters collected upfront.
        After setup, the workflow runs automatically with progress updates only.""")

    info_box([
        "Before You Begin:",
        "",
        "Automatic mode requires the following to be ready:",
        "  1. Copernicus Marine Service account (username + password)",
        "  2. A study area polygon in GeoJSON format",
        "",
        "If you don't have a GeoJSON polygon yet, interrupt the kernel and create one first using the following function:",
        "  create_map()",
        "This opens an interactive map to draw and export your study area.",
    ])

    # Step 1: Project folder (always required, regardless of which path below)
    project_dirs = setup_project_directory()

    # Step 2: Decide whether to collect new params or load a saved config
    subsection_header("Choose an Option")
    print("\n  1. New analysis (collect all parameters now)")
    print("  2. Re-run with saved settings (load from a config file)")

    while True:
        choice = input("\nEnter choice (1/2) [default: 1]: ").strip() or '1'

        if choice == '1':
            # New analysis - collect parameters then run
            subsection_header("Starting Parameter Collection")
            print("You will be prompted for all workflow parameters.")
            print("Once complete, the workflow will run automatically.\n")

            config = collect_all_parameters(project_dirs=project_dirs)

            subsection_header("Parameter Collection Complete.")

            print("\nOptions:")
            print("  1. Run workflow now")
            print("  2. Review and edit parameters first")
            print("  3. Save configuration and exit (run later)")

            action = input("\nSelect option (1/2/3) [default: 1]: ").strip() or '1'

            if action == '1':
                results = run_workflow(config)
                return results
            elif action == '2':
                # Review and edit parameters
                edited_config = review_and_edit_config(config)

                if edited_config is None:
                    print("\nEditing cancelled.")
                    return config

                config = edited_config

                # Run workflow after editing
                proceed = get_yes_no("\nRun workflow now?", default='yes')
                if proceed:
                    results = run_workflow(config)
                    return results
                else:
                    print("\nWorkflow not started. Configuration returned.")
                    print("To run later, use: run_workflow(config)")
                    return config
            else:
                print("\nWorkflow cancelled. Your config has been saved.")
                print("To run later, load the config file and use: run_workflow(config)")
                return config

        elif choice == '2':
            # Load existing config - auto-detect like hydro params
            config_file = "comcon_config.json"

            if not os.path.exists(config_file):
                print(f"\nNo saved configuration found ({config_file}).")
                print("Please run a new analysis first (option 1) to create a configuration file.")
                continue

            subsection_header("Load Saved Configuration")
            print(f"\nFound saved configuration: {config_file}")

            while True:
                config_path = config_file

                # Allow user to cancel and go back to main menu
                if config_path.lower() == 'cancel':
                    break

                # Auto-add .json extension if not present
                if not config_path.endswith('.json'):
                    config_path += '.json'

                if not os.path.exists(config_path):
                    print(f"\nError: File not found: {config_path}")
                    print("Please check the path and try again, or type 'cancel' to go back.")
                    continue

                try:
                    config = load_config(config_path)
                    print("\nConfiguration loaded successfully!")
                    print(f"  Species traits: {config['traits']['source']}")
                    print(f"  Occurrence data: {config['occurrence']['data_source']}")
                    print(f"  Output directory: {config['output']['directory']}")

                    # Ask if user wants to review/edit
                    display_line("─")
                    print("Options:")
                    print("  1. Run workflow with these settings")
                    print("  2. Review and edit parameters first")
                    print("  3. Cancel")

                    action = input("\nSelect option (1/2/3) [default: 1]: ").strip() or '1'

                    if action == '1':
                        results = run_workflow(config)
                        return results
                    elif action == '2':
                        # Review and edit parameters
                        edited_config = review_and_edit_config(config)

                        if edited_config is None:
                            print("\nEditing cancelled.")
                            return config

                        config = edited_config

                        # Save the edited config
                        save_edited = get_yes_no("\nSave edited configuration to file?", default='yes')
                        if save_edited:
                            print(f"\nDefault save path: {config_path}")
                            print("Press Enter to use default or enter a new path:")
                            save_path = input("New path: ").strip() or config_path
                            if not save_path.endswith('.json'):
                                save_path += '.json'
                            with open(save_path, 'w') as f:
                                json.dump(config, f, indent=2, default=str)
                            print(f"Configuration saved to: {save_path}")

                        # Now run workflow
                        proceed = get_yes_no("\nRun workflow now?", default='yes')
                        if proceed:
                            results = run_workflow(config)
                            return results
                        else:
                            print("\nWorkflow not started. Configuration returned.")
                            return config
                    else:
                        print("\nWorkflow cancelled.")
                        return config

                except Exception as e:
                    print(f"\nError loading config: {e}")
                    print("Please check the file format and try again, or type 'cancel' to go back.")
                    continue
        else:
            print("Please enter 1 or 2")


# endregion

# region Configuration Collection Functions
def collect_all_parameters(project_dirs=None):
    """
    Collect ALL workflow parameters upfront via interactive prompts.

    This function guides the user through configuring every aspect of the
    workflow before any processing begins. Parameters are returned as a
    dictionary that can be saved to JSON for reproducibility.

    Returns
    -------
    dict
        Complete configuration dictionary containing all workflow parameters
    """
    config = {}

    section_header("Parameter Collection")
    print("\nThis will collect all parameters needed for the automatic workflow.")
    print("Progress will be shown as: [Section X/9]\n")

    # Section 1: Project Setup
    subsection_header("[Section 1/9] Project Setup")

    # The project folder is created once at the start of the run via setup_project_directory,
    # which offers the full options (new folder, your own directory, or continue an existing project).
    # We reuse that here rather than prompting again.
    if project_dirs is None:
        # Direct call to collect_all_parameters without start_automatic_mode: set the project up now.
        project_dirs = setup_project_directory()
    else:
        print("\nUsing the project folder created at the start of this run.")

    config['project'] = project_dirs

    # Section 2: Hydrodynamic Data
    subsection_header("[Section 2/9] Hydrodynamic Data")
    print("\n  Data source:")
    print("    1. Download from Copernicus Marine Service (requires internet)")
    print("    2. Load from a local file (no internet needed)")
    hydro_source = input("\n  Select data source (1/2) [default: 1]: ").strip() or '1'
    config['copernicus'] = {}
    config['copernicus']['source'] = 'local' if hydro_source == '2' else 'copernicus'

    if config['copernicus']['source'] == 'local':
        local_path = get_valid_path(
            "Path to your hydrodynamic data file",
            must_exist=True,
            extensions=['.nc', '.zarr']
        )
        config['copernicus']['local_path'] = local_path
        config['copernicus']['export'] = False
        # Skip to traits section
    else:
        config['copernicus'] = {}
        print("\nCopernicus Marine Service credentials:")
        config['copernicus']['username'] = input("  Username: ").strip()
        config['copernicus']['password'] = getpass.getpass("  Password: ").strip()
        info_box([
            "Select a hydrodynamic dataset. The most commonly used datasets are listed below.",
            "You can also enter a custom dataset ID if your region is not covered.",
            "",
            "  1. Global Ocean (~9km resolution, 2022–present)",
            "     cmems_mod_glo_phy_anfc_0.083deg_PT1H-m",
            "",
            "  2. NW European Shelf (~1.5km resolution, 2024–present)",
            "     cmems_mod_nws_phy-cur_anfc_1.5km-2D_PT1H-i",
            "",
            "  3. Atlantic–Iberian–Mediterranean (~3km resolution, 2022–present)",
            "     cmems_mod_ibi_phy_anfc_0.027deg-2D_PT1H-m",
            "",
            "  4. Mediterranean Sea (~4km resolution, 2022–present)",
            "     cmems_mod_med_phy-cur_anfc_4.2km-2D_PT1H-m",
            "",
            "  5. Enter a custom dataset ID",
            "",
            "  6. Browse the Copernicus Marine Data Store"
        ])

        dataset_options = {
            '1': 'cmems_mod_glo_phy_anfc_0.083deg_PT1H-m',
            '2': 'cmems_mod_nws_phy-cur_anfc_1.5km-2D_PT1H-i',
            '3': 'cmems_mod_ibi_phy_anfc_0.027deg-2D_PT1H-m',
            '4': 'cmems_mod_med_phy-cur_anfc_4.2km-2D_PT1H-m',
        }

        while True:
            choice = input("Select dataset (1-6) [default: 1]: ").strip() or '1'
            if choice in dataset_options:
                config['copernicus']['dataset_id'] = dataset_options[choice]
                print(f"\n  Selected: {config['copernicus']['dataset_id']}")
                break
            elif choice == '5':
                custom_id = input("Enter the dataset ID: ").strip()
                if not custom_id:
                    print("Error: Dataset ID cannot be empty.")
                    continue
                config['copernicus']['dataset_id'] = custom_id
                print(f"\n  Selected: {custom_id}")
                break
            elif choice == '6':
                import webbrowser
                webbrowser.open("https://data.marine.copernicus.eu/products")
                print("  Opening Copernicus Marine Data Store in your browser...")
                print("  Find your dataset, then select option 5 to enter its ID.\n")
                continue
            else:
                print("  Invalid choice. Enter 1-6.")
                continue

        print("\nHydrodynamic domain extent (must cover your study area + buffer):")
        while True:
            try:
                config['copernicus']['min_lat'] = float(input("  Minimum latitude (e.g., 49.0): "))
                config['copernicus']['max_lat'] = float(input("  Maximum latitude (e.g., 52.0): "))
                if config['copernicus']['min_lat'] >= config['copernicus']['max_lat']:
                    print("  Error: min_lat must be less than max_lat")
                    continue
                if not (-90 <= config['copernicus']['min_lat'] <= 90) or not (
                        -90 <= config['copernicus']['max_lat'] <= 90):
                    print("  Error: Latitude must be between -90 and 90.")
                    continue
                break
            except ValueError:
                print("  Error: Please enter valid numbers")

        while True:
            try:
                config['copernicus']['min_lon'] = float(input("  Minimum longitude (e.g., -6.0): "))
                config['copernicus']['max_lon'] = float(input("  Maximum longitude (e.g., 2.0): "))
                if config['copernicus']['min_lon'] >= config['copernicus']['max_lon']:
                    print("  Error: min_lon must be less than max_lon")
                    continue
                if not (-180 <= config['copernicus']['min_lon'] <= 180) or not (
                        -180 <= config['copernicus']['max_lon'] <= 180):
                    print("  Error: Longitude must be between -180 and 180.")
                    continue
                break
            except ValueError:
                print("  Error: Please enter valid numbers")

        print("\n  Note: ComCon streams hydrodynamic data directly from Copernicus (using lazy loading).")
        print("  You do not need to download the data to run your analysis.")
        print("  Downloading is only useful if you want a local copy for offline use or to avoid re-downloading in future sessions.")
        print("  Warning: Downloaded files are often large (multiple GB) and take a long time to download (several hours).\n")

        # Fix: default-to-NO logic was inverted. "yes" should export, "no" should skip.
        export = input("  Save hydrodynamic data to file? (yes/no) [default: no]: ").strip().lower() or 'no'
        config['copernicus']['export'] = export in ['yes', 'y']

        if config['copernicus']['export']:
            print("\n  Export format:")
            print("    1. NetCDF (.nc) Standard scientific data format")
            print("    2. Zarr (.zarr) Smaller, faster, recommended for large datasets")
            export_fmt = input("  Choose format (1/2) [default: 1]: ").strip() or '1'
            config['copernicus']['export_format'] = 'zarr' if export_fmt == '2' else 'netcdf'
            file_ext = '.zarr' if config['copernicus']['export_format'] == 'zarr' else '.nc'

            default_hydro_path = os.path.join(config['project']['data'], f'hydro_data{file_ext}')
            hydro_path = input(f"  Output path [default: {default_hydro_path}]: ").strip()
            config['copernicus']['output_path'] = hydro_path if hydro_path else default_hydro_path
            # Ensure correct extension
            if config['copernicus']['export_format'] == 'zarr' and not config['copernicus']['output_path'].endswith(
                    '.zarr'):
                config['copernicus']['output_path'] = config['copernicus']['output_path'].rsplit('.', 1)[0] + '.zarr'
            elif config['copernicus']['export_format'] == 'netcdf' and not config['copernicus']['output_path'].endswith(
                    '.nc'):
                config['copernicus']['output_path'] += '.nc'
        else:
            config['copernicus']['export_format'] = 'netcdf'
            config['copernicus']['output_path'] = None

        # Grid Resolution
        print("\nGrid Resolution for Release and Settlement")
        print("\nThe grid resolution only controls the resolution of the analysis (release and settlement sites).")
        print("Particle advection always runs on the full native-resolution hydrodynamic grid.")
        print("A coarser grid means fewer sites, fewer particles and faster simulations, but less spatial detail in the connectivity results.")
        print("\nOptions:")
        print("  1. Recommended (auto-calculated from your domain size)")
        print("  2. Native (the hydrodynamic grid's own resolution).")
        print("     On large domains this can produce many thousands of sites and may require significant working memory and computation time.")
        print("     On very large domains it can exhaust available RAM.")
        print("  3. Custom (specify your own resolution in km)")

        while True:
            res_choice = input("\nSelect option (1/2/3) [default: 1]: ").strip() or '1'
            if res_choice in ['1', '2', '3']:
                break
            print("  Invalid input. Please enter 1, 2, or 3.")

        if res_choice == '1':
            config['grid'] = {'resolution_mode': 'recommended', 'resolution_km': None}
        elif res_choice == '2':
            config['grid'] = {'resolution_mode': 'native', 'resolution_km': None}
        else:
            while True:
                try:
                    custom_km = int(input("  Enter resolution in km (e.g. 15): ").strip())
                    if custom_km < 1:
                        print("  Resolution must be at least 1 km.")
                        continue
                    config['grid'] = {'resolution_mode': 'custom', 'resolution_km': custom_km}
                    break
                except ValueError:
                    print("  Please enter a whole number.")

        # Section 2: Species Trait Data
        subsection_header("[Section 3/9] Species Traits Data")

        config['traits'] = {}
        print("\nTrait data source:")
        print("  1. Use built-in trait database")
        print("  2. Provide your own CSV file")

        trait_source = input("\nSelect option (1/2) [1]: ").strip() or '1'
        config['traits']['source'] = 'builtin' if trait_source == '1' else 'custom'

        if config['traits']['source'] == 'custom':
            print("\n  Your CSV file must contain the following columns:")
            print("    Required: species_name, min_PLD, max_PLD, spawning_start, spawning_end")
            print("    Optional: aphia_id, lat, lon, polygon_id")
            print("  Date format for spawning: DD/MM (e.g., 01/10 for 1st October)")
            print("  PLD values should be in days (integers)")
            print("  Any additional columns are permitted and will be ignored.\n")
            config['traits']['custom_path'] = get_valid_path(
                "Path to your trait CSV file",
                must_exist=True,
                extensions=['.csv']
            )
            config['traits']['species_indices'] = None
        else:
            config['traits']['custom_path'] = None
            print("\nSpecies selection:")
            print("  1. Select specific species by number")
            print("  2. Use all species in database")

            species_choice = input("\nSelect option (1/2) [default: 1]: ").strip() or '1'

            if species_choice == '2':
                config['traits']['species_indices'] = None
                print("  Using all species in database.")
            else:
                # Show available species from database
                all_traits = load_builtin_traits()

                if all_traits is None or len(all_traits) == 0:
                    print("Error: Could not load trait database. Using all species.")
                    config['traits']['species_indices'] = None
                else:
                    display_line("-")
                    print("Available Species in Database:")
                    display_line("-")

                    for i, row in all_traits.iterrows():
                        spp_name = row['species_name']
                        min_pld = row['min_PLD']
                        max_pld = row['max_PLD']
                        spawn_start_raw = str(row.get('spawning_start', 'N/A'))
                        spawn_end_raw = str(row.get('spawning_end', 'N/A'))
                        spawn_start = '/'.join(
                            spawn_start_raw.split('/')[:2]) if '/' in spawn_start_raw else spawn_start_raw
                        spawn_end = '/'.join(spawn_end_raw.split('/')[:2]) if '/' in spawn_end_raw else spawn_end_raw
                        print(
                            f"  {i + 1:2}. {spp_name:<30} PLD: {min_pld}-{max_pld} days, Spawning: {spawn_start} to {spawn_end}")

                    display_line("-")

                    print("\n  Enter species numbers separated by commas (e.g., 1,3,5)")
                    indices_str = input("  Species numbers: ").strip()

                    try:
                        selected_indices = [int(x.strip()) - 1 for x in indices_str.split(',')]
                        config['traits']['species_indices'] = selected_indices

                        print("\n  Selected species:")
                        for idx in selected_indices:
                            if 0 <= idx < len(all_traits):
                                row = all_traits.iloc[idx]
                                spp_name = row['species_name']
                                min_pld = row['min_PLD']
                                max_pld = row['max_PLD']
                                spawn_start_raw = str(row['spawning_start'])
                                spawn_end_raw = str(row['spawning_end'])
                                spawn_start = '/'.join(
                                    spawn_start_raw.split('/')[:2]) if '/' in spawn_start_raw else spawn_start_raw
                                spawn_end = '/'.join(
                                    spawn_end_raw.split('/')[:2]) if '/' in spawn_end_raw else spawn_end_raw
                                print(
                                    f"    - {spp_name}: PLD {min_pld}-{max_pld} days, Spawning {spawn_start} to {spawn_end}")

                        confirm = input("\n  Confirm selection? (yes/no) [default: yes]: ").strip().lower()
                        if confirm in ['no', 'n']:
                            print("  Restarting species selection...")
                            # Reset and let them try again - for simplicity, use all species
                            config['traits']['species_indices'] = None
                            print("  Using all species. You can re-select after restarting.")
                    except ValueError:
                        print("  Invalid input. Using all species.")
                        config['traits']['species_indices'] = None
        _default_year = datetime.now().year - 1
        while True:
            try:
                year_input = input(f"\nStart year for dispersal season [default: {_default_year}]: ").strip()
                config['traits']['start_year'] = int(year_input) if year_input else _default_year
                break
            except ValueError:
                print("  Error: Please enter a valid year")

        print("\nTemporal buffer (extends the hydrodynamic data period):")
        add_buffer = input("  Add buffer days for trait sensitivity analysis? (yes/no) [default: no]: ").strip().lower()
        if add_buffer in ['yes', 'y']:
            config['traits']['buffer_before'] = int(input("  Days before earliest spawning (e.g. 14): ").strip() or "14")
            config['traits']['buffer_after'] = int(input("  Days after latest dispersal end (e.g. 14): ").strip() or "14")
        else:
            config['traits']['buffer_before'] = 0
            config['traits']['buffer_after'] = 0

        print("\nSave selected traits to CSV?")
        print("  Note: This is optional. Your species selection is stored in the setup file")
        print("  and will be automatically re-selected when you reload the setup.")
        print("  Saving to CSV is useful only for your records.")
        save_traits = input("\n  Save traits to CSV? (yes/no) [default: no]: ").strip().lower()
        config['traits']['save_csv'] = save_traits not in ['no', 'n']
        if config['traits']['save_csv']:
            default_trait_dir = config['project']['data']
            config['traits']['output_dir'] = input(
                f"  Output directory [default: {default_trait_dir}]: ").strip() or default_trait_dir
            config['traits']['filename'] = input("  Filename [default: selected_traits]: ").strip() or "selected_traits"
        else:
            config['traits']['output_dir'] = config['project']['data']
            config['traits']['filename'] = "selected_traits"

        # Section 3: Species Occurrence Data
        subsection_header("[Section 4/9] Species Occurrence Data")

        config['occurrence'] = {}

        print("\nOccurrence data source:")
        print("  1. Download from GBIF (requires internet)")
        print("  2. Use your own CSV file")

        data_source = input("\nSelect option (1/2) [default: 1]: ").strip() or '1'
        config['occurrence']['data_source'] = {'1': 'gbif', '2': 'user'}.get(data_source, 'gbif')

        config['occurrence']['user_data_path'] = None

        if config['occurrence']['data_source'] == 'user':
            config['occurrence']['user_data_path'] = get_valid_path(
                "Path to occurrence CSV file",
                must_exist=True,
                extensions=['.csv']
            )

        print("\nStudy area polygon (GeoJSON format):")
        print("  Option 1: Provide a path to an existing GeoJSON file (press Enter to browse).")
        print("  Option 2: Type 'draw' to open the interactive map widget and draw one now.\n")

        while True:
            user_input = input("Path to GeoJSON file (or 'draw'): ").strip()

            if user_input.lower() == 'draw':
                print("\n  Opening interactive map. Draw your study area polygon, then click Export.")
                print("  When the file is saved, return here and enter the file path it gave you.\n")
                _ = create_map()
                continue  # ask again for the path once user finishes drawing

            if not user_input:
                # Empty → open a file dialog via get_valid_path
                geojson_path = get_valid_path(
                    "Path to GeoJSON file",
                    must_exist=True,
                    extensions=['.geojson', '.json']
                )
            else:
                geojson_path = user_input
                if not geojson_path.lower().endswith(('.geojson', '.json')):
                    geojson_path += '.geojson'
                if not os.path.exists(geojson_path):
                    print(f"  File not found: {geojson_path}. Please try again.")
                    continue

            config['occurrence']['geojson_path'] = geojson_path
            print(f"  File found: {geojson_path}")
            break

        print("\nHabitat filtering:")
        print("  1. Coastal (recommended for littoral/intertidal species)")
        print("  2. Ocean (coastal + offshore)")
        print("  3. None (use data as-is)")

        habitat_choice = input("\nSelect option (1/2/3) [default: 1]: ").strip() or '1'
        habitat_map = {'1': 'coastal', '2': 'ocean', '3': 'none'}
        config['occurrence']['habitat_filtering'] = habitat_map.get(habitat_choice, 'coastal')

        save_occ = input("\nSave cleaned occurrence data to CSV? (yes/no) [default: no]: ").strip().lower()
        config['occurrence']['save_csv'] = save_occ in ['yes', 'y']
        if config['occurrence']['save_csv']:
            default_occ_path = os.path.join(config['project']['occurrence'], 'cleaned_occurrences.csv')
            config['occurrence']['csv_path'] = input(
                f"  Output path [default: {default_occ_path}]: ").strip() or default_occ_path
            if not config['occurrence']['csv_path'].endswith('.csv'):
                config['occurrence']['csv_path'] += '.csv'
        else:
            config['occurrence']['csv_path'] = None

        # Section 4: Release & Settlement Strategy
        subsection_header("[Section 5/9] Release and Settlement Strategy")

        config['grids'] = {}

        # Get the habitat type from earlier selection
        habitat_type = config['occurrence']['habitat_filtering']
        habitat_label = {'coastal': 'coastal', 'ocean': 'ocean', 'none': 'all'}.get(habitat_type, 'coastal')

        print(f"\nBased on your habitat filtering selection: {habitat_label}")

        print("\nRelease strategy (where particles are released from):")
        print(f"  1. Everywhere in your study area (all {habitat_label} sites)")
        print(f"  2. All sites in your occurrence data (non-species-specific)")
        print(f"  3. Species-specific habitat in your study area (uses your occurrence data)")

        release_choice = input("\nSelect option (1/2/3) [default: 3]: ").strip() or '3'
        release_map = {'1': 'all_sites', '2': 'pooled_sites', '3': 'species_habitat'}
        config['grids']['release_strategy'] = release_map.get(release_choice, 'species_habitat')

        print("\nSettlement strategy (where particles can settle):")
        print(f"  1. Everywhere in your study area (all {habitat_label} sites)")
        print(f"  2. All sites in your occurrence data (non-species-specific)")
        print(f"  3. Species-specific habitat in your study area (uses your occurrence data)")
        print(f"  4. A different occurrence dataset (separate CSV)")

        settle_choice = input("\nSelect option (1/2/3/4) [default: 3]: ").strip() or '3'
        settle_map = {'1': 'all_sites', '2': 'pooled_sites', '3': 'species_habitat', '4': 'custom_habitat'}
        config['grids']['settlement_strategy'] = settle_map.get(settle_choice, 'species_habitat')

        if config['grids']['settlement_strategy'] == 'custom_habitat':
            config['grids']['custom_settlement_path'] = input("  Path to custom settlement CSV: ").strip()
            if not config['grids']['custom_settlement_path'].endswith('.csv'):
                config['grids']['custom_settlement_path'] += '.csv'
            print("\nHow should this dataset be used?")
            print("  1. All sites in your dataset (not species-specific)")
            print("  2. Species-specific habitat in your study area (uses your occurrence data)")
            _csm = input("\nSelect option (1/2) [default: 2]: ").strip() or '2'
            config['grids']['custom_settlement_mode'] = 'pooled' if _csm == '1' else 'species_specific'
        else:
            config['grids']['custom_settlement_path'] = None
            config['grids']['custom_settlement_mode'] = None

        save_grids = input("\nSave release/settlement grids to CSV? (yes/no) [default: no]: ").strip().lower()
        config['grids']['save_grids'] = save_grids in ['yes', 'y']
        if config['grids']['save_grids']:
            default_grid_dir = config['project']['grids']
            config['grids']['output_dir'] = input(
                f"  Output directory [default: {default_grid_dir}]: ").strip() or default_grid_dir
        else:
            config['grids']['output_dir'] = config['project']['grids']

        # Section 5: Simulation Parameters
        subsection_header("[Section 6/9] Particle Simulation")

        config['simulation'] = {}

        print("\nRelease interval (how often particles are released):")
        print("  Options: daily, weekly, biweekly, monthly")
        print("  Or enter a number for custom days (e.g., 5)")
        interval_input = input("\nRelease interval [default: weekly]: ").strip().lower() or 'weekly'
        config['simulation']['release_interval'] = interval_input

        config['simulation']['particles_per_location'] = int(
            input("\nParticles per location per release [default: 100]: ").strip() or "100"
        )

        print("\nHorizontal diffusivity (Kh) represents small-scale random motion caused by turbulence")
        print("and applies evenly to all particles in the simulation.")
        config['simulation']['horizontal_diffusivity'] = float(
            input("  Kh value in m²/s: [default: 1.0]").strip() or "1.0"
        )

        default_sim_path = os.path.join(config['project']['simulation'], 'simulation.zarr')
        sim_path = input(f"\nSimulation output path [default: {default_sim_path}]: ").strip()
        config['simulation']['output_path'] = sim_path if sim_path else default_sim_path
        if not config['simulation']['output_path'].endswith('.zarr'):
            config['simulation']['output_path'] += '.zarr'

        config['simulation']['output_dt_hours'] = int(
            input("\nOutput frequency in hours [default: 24]: ").strip() or "24"
        )

        config['simulation']['dt_minutes'] = int(
            input("Simulation timestep in minutes [default: 60]: ").strip() or "60"
        )

        # Section 6: Post-Processing Parameters
        subsection_header("[Section 7/9] Post-Processing")

        config['postprocessing'] = {}

        print("\nSettlement buffer (how close particles must be to suitable habitat to settle):")
        print("  1 = adjacent cell only, 2 = two cells away, etc.")
        config['postprocessing']['settlement_buffer'] = int(
            input("  Buffer size in grid cells [default: 1]: ").strip() or "1"
        )

        # Section 7: Analysis Configuration
        subsection_header("[Section 8/9] Connectivity Analyses")

        config['analysis'] = {}

        # Analysis scope selection
        print("\nAnalysis scope:")
        print("  1. Community level only (all species combined)")
        print("  2. Per-species only (each species analysed separately)")
        print("  3. Both community and per-species")
        print("  4. Custom species selection (select specific species)")

        scope_choice = input("\nSelect scope (1/2/3/4) [default: 3]: ").strip() or '3'

        if scope_choice == '4':
            # Custom species selection
            config['analysis']['scope'] = 'custom'

            # Show available species
            print("\nAvailable species:")
            if config['traits']['source'] == 'builtin' and config['traits']['species_indices']:
                # Already selected specific species
                all_traits = load_builtin_traits()
                selected_traits = all_traits.iloc[config['traits']['species_indices']]
                for i, (_, row) in enumerate(selected_traits.iterrows()):
                    print(f"  {i + 1}. {row['species_name']}")
                max_species = len(selected_traits)
            else:
                # Show all species from config
                print("  (Species will be numbered when traits are loaded)")
                print(
                    "  Enter species numbers after seeing the full list, or enter 'all' to analyse all species individually.")
                max_species = 999  # Will be validated later

            print(
                "\nEnter species numbers to analyse (comma-separated, e.g. 1,3,5), or enter 'all' to run per-species analysis on all.")
            species_input = input("Species numbers: ").strip()

            if species_input.lower() == 'all':
                config['analysis']['selected_species_indices'] = None  # All species
                config['analysis']['scope'] = 'per_species'  # Treat as per-species
            else:
                try:
                    selected_indices = [int(x.strip()) - 1 for x in species_input.split(',')]
                    config['analysis']['selected_species_indices'] = selected_indices
                except ValueError:
                    print("  Invalid input. Using all species.")
                    config['analysis']['selected_species_indices'] = None
                    config['analysis']['scope'] = 'per_species'
        else:
            scope_map = {'1': 'community', '2': 'per_species', '3': 'both'}
            config['analysis']['scope'] = scope_map.get(scope_choice, 'both')
            config['analysis']['selected_species_indices'] = None

        print("\nAvailable analyses:")
        print("  Type 1: Connectivity metrics (self-recruitment, local retention, etc.)")
        print("  Type 2: Network analysis (identify connectivity hotspots)")
        print("  Type 3: Trait sensitivity (modify a trait and compare to the baseline)")

        run_type1 = input("\nRun Type 1: Connectivity Metrics? (yes/no) [default: yes]: ").strip().lower()
        config['analysis']['run_type1'] = run_type1 not in ['no', 'n']

        run_type2 = input("Run Type 2: Network Analysis? (yes/no) [default: yes]: ").strip().lower()
        config['analysis']['run_type2'] = run_type2 not in ['no', 'n']

        run_type3 = input("Run Type 3: Trait Sensitivity Analysis? (yes/no) [default: no]: ").strip().lower()
        config['analysis']['run_type3'] = run_type3 in ['yes', 'y']

        # Match interactive: ask whether to exclude self-recruitment from source/sink metrics.
        # Used by both Type 2 and Type 3 network analysis so the modes agree.
        if config['analysis']['run_type2'] or config['analysis'].get('run_type3', False):
            excl = input(
                "\nExclude self-recruitment (particles returning to their release site) "
                "from source/sink metrics? (yes/no) [default: yes]: "
            ).strip().lower()
            config['analysis']['exclude_self_recruitment'] = excl not in ['no', 'n']

        if config['analysis']['run_type2']:
            print("\nSpatial Scale Selection")
            print("\nDistance thresholds filter which site-to-site connections are included.")
            print("Only connections where larvae travel less than the threshold are kept.")
            print("  - Smaller thresholds reveal local stepping-stone connectivity")
            print("  - Larger thresholds include long-distance dispersal events")
            print("\nOptions:")
            print("  1. Full domain (include all connections regardless of distance)")
            print("  2. Single scale (set one distance threshold)")
            print("  3. Multi-scale (compare local, regional and seascape patterns)")

            scale_choice = input("\nSelect option (1/2/3) [default: 1]: ").strip() or '1'

            if scale_choice == '1':
                config['analysis']['type2_scales'] = [None]  # None means no filtering
                config['analysis']['scale_type'] = 'full_domain'
                print("\nUsing full domain. All connections will be included.")
            elif scale_choice == '3':
                print("\nMulti-scale analysis:")
                local_km = int(input("  Local scale threshold (km) [default: 10]: ").strip() or "10")
                regional_km = int(input("  Regional scale threshold (km) [default: 30]: ").strip() or "30")
                seascape_km = int(input("  Seascape scale threshold (km) [default: 100]: ").strip() or "100")
                config['analysis']['type2_scales'] = [local_km, regional_km, seascape_km]
                config['analysis']['scale_type'] = 'multi'
            else:
                print("\nSingle scale analysis:")
                scale_km = int(input("  Distance threshold in km [default: 30]: ").strip() or "30")
                config['analysis']['type2_scales'] = [scale_km]
                config['analysis']['scale_type'] = 'single'

            print("\nNetwork analysis connectivity metrics selection:")
            print("  1. Composite measure of functional roles (source strength, sink strength, self-recruitment)")
            print("  2. Composite measure of centrality measures (betweenness, closeness, degree)")
            print("  3. Overall composite (all metrics combined)")
            print("  4. Custom selection of one or several metrics")
            print("\n A composite measure detects overall connectivity hotspots across several key metrics.")

            composite_choice = input("\nSelect option (1/2/3/4) [default: 3]: ").strip() or '3'
            if composite_choice == '4':
                print("\nAvailable metrics:")
                print("  1. source")
                print("  2. sink")
                print("  3. self-recruitment")
                print("  4. betweenness")
                print("  5. closeness")
                print("  6. degree")
                _metric_map = {
                    '1': 'source', '2': 'sink', '3': 'self_recruitment',
                    '4': 'betweenness', '5': 'closeness', '6': 'degree'
                }
                while True:
                    metric_input = input("\nEnter metric numbers to combine (comma-separated, e.g. 1,2,4): ").strip()
                    if not metric_input:
                        print("  Please enter at least one metric number.")
                        continue
                    try:
                        chosen = [m.strip() for m in metric_input.split(',') if m.strip()]
                        invalid = [m for m in chosen if m not in _metric_map]
                        if invalid:
                            print(f"  Invalid number(s): {', '.join(invalid)}. Please use 1-6.")
                            continue
                        config['analysis']['custom_metrics'] = [_metric_map[m] for m in chosen]
                        config['analysis']['type2_composite'] = 'custom'
                        print(f"  Selected: {', '.join(config['analysis']['custom_metrics'])}")
                        break
                    except Exception:
                        print("  Could not parse input. Please enter comma-separated numbers (e.g. 1,2,4).")
            else:
                composite_map = {'1': 'functional', '2': 'centrality', '3': 'overall'}
                config['analysis']['type2_composite'] = composite_map.get(composite_choice, 'overall')

                # Save figures (asked for every composite choice, including custom)
            save_figs = input("\nSave Type 2 figures to file? (yes/no) [default: no]: ").strip().lower()
            config['analysis']['type2_visualizations'] = 'all' if save_figs in ['yes', 'y'] else 'skip'
        else:
            config['analysis']['type2_scales'] = [None]
            config['analysis']['scale_type'] = 'full_domain'
            config['analysis']['type2_composite'] = 'overall'
            config['analysis']['type2_visualizations'] = 'skip'

        if config['analysis']['run_type3']:
            # Trait sensitivity is the only Type 3 option in V1
            # (the previous null-model option has been removed).
            config['analysis']['type3_option'] = 'trait_sensitivity'
            print("\nType 3 will run a trait sensitivity comparison:")
            print("  Modifies one trait, re-runs network analysis, compares to baseline.")

            print("\nNetwork analysis scale:")
            print("  1. Full domain (no distance filtering, all connections included)")
            print("  2. Specify distance threshold (km)")

            scale_opt = input("\nSelect option (1/2) [default: 1]: ").strip() or '1'

            if scale_opt == '1':
                config['analysis']['type3_scale'] = None  # None means full domain
                print("  Using full domain (all connections)")
            else:
                config['analysis']['type3_scale'] = int(
                    input("  Distance threshold in km [default: 30]: ").strip() or "30"
                )
                print(f"  Using {config['analysis']['type3_scale']} km threshold")

            print("\nTrait modification type:")
            print("  1. Modify PLD (pelagic larval duration)")
            print("  2. Shift spawning window")

            mod_choice = input("\nSelect option (1/2) [default: 1]: ").strip() or '1'
            config['analysis']['trait_mod_type'] = 'pld' if mod_choice == '1' else 'spawning'

            if config['analysis']['trait_mod_type'] == 'pld':
                print("\nPLD modification options:")
                print("  1. Same change to both min and max PLD")
                print("  2. Different changes to min and max PLD")
                pld_option = input("\nSelect (1/2) [default: 1]: ").strip() or '1'

                if pld_option == '2':
                    config['analysis']['min_pld_change'] = int(
                        input("  Days to add/subtract from min_PLD [default: 0]: ").strip() or "0"
                    )
                    config['analysis']['max_pld_change'] = int(
                        input("  Days to add/subtract from max_PLD [default: 0]: ").strip() or "0"
                    )
                else:
                    pld_change = int(
                        input("  Days to add/subtract from both min and max PLD [default: 0]: ").strip() or "0"
                    )
                    config['analysis']['min_pld_change'] = pld_change
                    config['analysis']['max_pld_change'] = pld_change
                config['analysis']['spawning_start_shift'] = 0
                config['analysis']['spawning_end_shift'] = 0
            else:
                print("\nSpawning window modification options:")
                print("  1. Shift entire window (same change to start and end)")
                print("  2. Modify start and end separately")
                spawn_option = input("\nSelect (1/2) [default: 1]: ").strip() or '1'

                if spawn_option == '2':
                    config['analysis']['spawning_start_shift'] = int(
                        input(
                            "  Days to shift start (positive = later, negative = earlier) [default: 0]: ").strip() or "0"
                    )
                    config['analysis']['spawning_end_shift'] = int(
                        input(
                            "  Days to shift end (positive = later, negative = earlier) [default: 0]: ").strip() or "0"
                    )
                else:
                    shift = int(
                        input(
                            "  Days to shift entire window (positive = later, negative = earlier) [default: 0]: ").strip() or "0"
                    )
                    config['analysis']['spawning_start_shift'] = shift
                    config['analysis']['spawning_end_shift'] = shift
                config['analysis']['min_pld_change'] = 0
                config['analysis']['max_pld_change'] = 0

        # Section 8: Output Settings
        subsection_header("[Section 9/9] Output Settings")

        config['output'] = {}

        default_output_dir = config['project']['root']
        config['output']['directory'] = input(
            f"\nOutput directory for all files [default: {default_output_dir}]: ").strip() or default_output_dir
        config['output']['figures_dir'] = config['project']['figures']
        config['output']['results_dir'] = config['project']['results']

        display_figs = input("\nDisplay figures during workflow? (yes/no) [default: yes]: ").strip().lower()
        config['output']['display_figures'] = display_figs not in ['no', 'n']

        save_figs = input("Save figures to files? (yes/no) [default: yes]: ").strip().lower()
        config['output']['save_figures'] = save_figs not in ['no', 'n']

        if config['output']['save_figures']:
            print("\nFigure format options: png, pdf, tif, jpeg, svg")
            config['output']['format'] = input("  Format [default: tif]: ").strip().lower() or 'tif'
            config['output']['dpi'] = int(input("  Resolution DPI [default: 300]: ").strip() or "300")
        else:
            config['output']['format'] = 'tif'
            config['output']['dpi'] = 300

        # Save Configuration
        section_header("Parameter Collection Complete")

        save_config = input("\nSave setup to file for future re-runs? (yes/no) [default: yes]: ").strip().lower()
        if save_config not in ['no', 'n']:
            default_config_path = os.path.join(config['project']['config'], 'comcon_setup.json')
            print(f"\nDefault setup path: {default_config_path}")
            print("Press Enter to use the default or enter a new output path and file name:")
            config_path = input("  Setup file path: ").strip()
            config_path = config_path if config_path else default_config_path
            if not config_path.endswith('.json'):
                config_path += '.json'

            # Create directory if needed
            config_dir = os.path.dirname(config_path)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir, exist_ok=True)

            # Remove password before saving
            config_to_save = copy.deepcopy(config)
            config_to_save['copernicus']['password'] = "YOUR_PASSWORD_HERE"

            with open(config_path, 'w') as f:
                json.dump(config_to_save, f, indent=4)
            print(f"\n  Configuration saved to: {config_path}")
            print("  Note: Password replaced with placeholder for security.")
            print("  You will be prompted for password when loading this config.")

    return config

## Review and edit the setup parameters
def review_and_edit_config(config):
    """Allow user to review and edit specific parameters."""

    def build_param_list(cfg):
        """Build a flat list of all editable parameters."""
        params = []
        num = 1

        # Copernicus section (8 params)
        params.append((num, "Copernicus Username", cfg.get('copernicus', {}).get('username', 'Not set'), 'copernicus', 'username'))
        num += 1
        params.append((num, "Copernicus Password", "***hidden***", 'copernicus', 'password'))
        num += 1
        params.append((num, "Dataset ID", cfg.get('copernicus', {}).get('dataset_id', 'Not set'), 'copernicus', 'dataset_id'))
        num += 1
        params.append((num, "Min Longitude", cfg.get('copernicus', {}).get('min_lon', 'Not set'), 'copernicus', 'min_lon'))
        num += 1
        params.append((num, "Max Longitude", cfg.get('copernicus', {}).get('max_lon', 'Not set'), 'copernicus', 'max_lon'))
        num += 1
        params.append((num, "Min Latitude", cfg.get('copernicus', {}).get('min_lat', 'Not set'), 'copernicus', 'min_lat'))
        num += 1
        params.append((num, "Max Latitude", cfg.get('copernicus', {}).get('max_lat', 'Not set'), 'copernicus', 'max_lat'))
        num += 1
        params.append((num, "Export Hydro Data", cfg.get('copernicus', {}).get('export', False), 'copernicus', 'export'))
        num += 1

        # Traits section (6 params)
        params.append((num, "Trait Source", cfg.get('traits', {}).get('source', 'builtin'), 'traits', 'source'))
        num += 1
        params.append((num, "Custom Trait Path", cfg.get('traits', {}).get('custom_path', 'None'), 'traits', 'custom_path'))
        num += 1
        params.append((num, "Species Indices", cfg.get('traits', {}).get('species_indices', 'all'), 'traits', 'species_indices'))
        num += 1
        params.append((num, "Start Year", cfg.get('traits', {}).get('start_year', 2024), 'traits', 'start_year'))
        num += 1
        params.append((num, "Buffer Before (days)", cfg.get('traits', {}).get('buffer_before', 0), 'traits', 'buffer_before'))
        num += 1
        params.append((num, "Buffer After (days)", cfg.get('traits', {}).get('buffer_after', 0), 'traits', 'buffer_after'))
        num += 1

        # Occurrence section (3 params)
        params.append((num, "Data Source", cfg.get('occurrence', {}).get('data_source', 'gbif'), 'occurrence', 'data_source'))
        num += 1
        params.append((num, "GeoJSON Path", cfg.get('occurrence', {}).get('geojson_path', 'None'), 'occurrence', 'geojson_path'))
        num += 1
        params.append((num, "Habitat Filtering", cfg.get('occurrence', {}).get('habitat_filtering', 'coastal'), 'occurrence', 'habitat_filtering'))
        num += 1

        # Grids section (2 params)
        params.append(
            (num, "Release Strategy", cfg.get('grids', {}).get('release_strategy', 'species_habitat'), 'grids','release_strategy'))
        num += 1
        params.append((num, "Settlement Strategy", cfg.get('grids', {}).get('settlement_strategy', 'species_habitat'), 'grids', 'settlement_strategy'))
        num += 1

        # Simulation section (6 params)
        params.append((num, "Release Interval", cfg.get('simulation', {}).get('release_interval', 'weekly'), 'simulation', 'release_interval'))
        num += 1
        params.append((num, "Particles/Location", cfg.get('simulation', {}).get('particles_per_location', 100), 'simulation', 'particles_per_location'))
        num += 1
        params.append((num, "Timestep (minutes)", cfg.get('simulation', {}).get('dt_minutes', 60), 'simulation', 'dt_minutes'))
        num += 1
        params.append((num, "Output Interval (hours)", cfg.get('simulation', {}).get('output_dt_hours', 24), 'simulation', 'output_dt_hours'))
        num += 1
        params.append((num, "Diffusivity (Kh)", cfg.get('simulation', {}).get('horizontal_diffusivity', 1.0), 'simulation', 'horizontal_diffusivity'))
        num += 1
        params.append((num, "Output Path", cfg.get('simulation', {}).get('output_path', './outputs/simulation.zarr'), 'simulation', 'output_path'))
        num += 1

        # Post-processing section (1 param)
        params.append((num, "Settlement Buffer (cells)", cfg.get('postprocessing', {}).get('settlement_buffer', 1), 'postprocessing', 'settlement_buffer'))
        num += 1

        # Analysis section (8 params)
        params.append((num, "Analysis Scope", cfg.get('analysis', {}).get('scope', 'both'), 'analysis', 'scope'))
        num += 1
        params.append((num, "Run Type 1 (Metrics)", cfg.get('analysis', {}).get('run_type1', True), 'analysis', 'run_type1'))
        num += 1
        params.append((num, "Run Type 2 (Network)", cfg.get('analysis', {}).get('run_type2', True), 'analysis', 'run_type2'))
        num += 1
        params.append((num, "Type 2 Scale Type", cfg.get('analysis', {}).get('scale_type', 'full_domain'), 'analysis', 'scale_type'))
        num += 1
        params.append((num, "Type 2 Scales (km)", cfg.get('analysis', {}).get('type2_scales', [None]), 'analysis', 'type2_scales'))
        num += 1
        params.append((num, "Type 2 Composite", cfg.get('analysis', {}).get('type2_composite', 'overall'), 'analysis', 'type2_composite'))
        num += 1
        params.append((num, "Run Type 3 (Sensitivity)", cfg.get('analysis', {}).get('run_type3', False), 'analysis', 'run_type3'))
        num += 1
        params.append((num, "Type 3 Scale Type", cfg.get('analysis', {}).get('type3_scale_type', 'full_domain'), 'analysis', 'type3_scale_type'))
        num += 1

        # Output section (4 params)
        params.append((num, "Output Directory", cfg.get('output', {}).get('directory', './outputs'), 'output', 'directory'))
        num += 1
        params.append((num, "Figure Format", cfg.get('output', {}).get('format', 'tif'), 'output', 'format'))
        num += 1
        params.append((num, "Figure DPI", cfg.get('output', {}).get('dpi', 300), 'output', 'dpi'))
        num += 1
        params.append((num, "Save Figures", cfg.get('output', {}).get('save_figures', True), 'output', 'save_figures'))
        num += 1

        return params

    # Build and display parameters
    params = build_param_list(config)

    section_header("Configuration Review")

    sections = {
        'copernicus': 'Hydrodynamic Data (Copernicus)',
        'traits': 'Species Traits',
        'occurrence': 'Occurrence Data',
        'grids': 'Grid Setup',
        'simulation': 'Simulation Settings',
        'postprocessing': 'Post-Processing',
        'analysis': 'Analysis Options',
        'output': 'Output Settings'
    }

    print("\nCurrent settings:")
    display_line("-")

    current_section = None
    for num, name, value, section, key in params:
        if section != current_section:
            current_section = section
            print(f"\n  [{sections.get(section, section)}]")

        # Format value for display
        if value is None:
            display_val = "None"
        elif isinstance(value, bool):
            display_val = "Yes" if value else "No"
        elif isinstance(value, list):
            if value == [None]:
                display_val = "full_domain"
            else:
                display_val = str(value)
        else:
            display_val = str(value)

        if len(str(display_val)) > 35:
            display_val = str(display_val)[:32] + "..."

        print(f"    {num:>2}. {name:<28} {display_val}")

    display_line("-")

    while True:
        edit_choice = input("Enter parameter number to edit or press Enter to continue: ").strip()

        if not edit_choice:
            break

        try:
            edit_num = int(edit_choice)
            for num, name, old_value, section, key in params:
                if num == edit_num:
                    if section not in config:
                        config[section] = {}

                    print(f"\n  Current value: {old_value}")
                    new_value = input(f"  New value for {name}: ").strip()

                    if new_value:
                        if isinstance(old_value, bool):
                            new_value = new_value.lower() in ['yes', 'y', 'true', '1']
                        elif isinstance(old_value, int):
                            new_value = int(new_value)
                        elif isinstance(old_value, float):
                            new_value = float(new_value)
                        elif isinstance(old_value, list):
                            if new_value.lower() == 'none':
                                new_value = [None]
                            else:
                                try:
                                    new_value = [int(x.strip()) for x in new_value.split(',')]
                                except:
                                    new_value = [x.strip() for x in new_value.split(',')]

                        config[section][key] = new_value
                        print(f"  Updated: {name} = {new_value}")
                    break
            else:
                print("  Invalid parameter number")
        except ValueError:
            print("  Please enter a valid number")

    return config

## Reload old setup parameters
def load_config(config_path):
    """
    Load configuration from JSON file.

    Parameters
    ----------
    config_path : str
        Path to JSON configuration file

    Returns
    -------
    dict
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Prompt for password if placeholder
    if config['copernicus']['password'] == "YOUR_PASSWORD_HERE":
        config['copernicus']['password'] = getpass.getpass("Enter Copernicus password: ").strip()

    return config
# endregion


# region Internal Helper Functions for Workflow
def _calculate_overall_dispersal_period(traits, buffer_before=0, buffer_after=0):
    """
    Calculate the overall dispersal period across all species.
    (Internal function - no interactive prompts)
    """
    earliest_start = None
    latest_end = None

    for t in traits:
        spawn_start = datetime.strptime(t['spawning_start'], '%d/%m/%Y')
        spawn_end = datetime.strptime(t['spawning_end'], '%d/%m/%Y')
        max_pld = t['max_PLD']

        dispersal_end = spawn_end + timedelta(days=max_pld)

        if earliest_start is None or spawn_start < earliest_start:
            earliest_start = spawn_start
        if latest_end is None or dispersal_end > latest_end:
            latest_end = dispersal_end

    base_start = earliest_start.strftime('%Y-%m-%d')
    base_end = latest_end.strftime('%Y-%m-%d')

    overall_start = (earliest_start - timedelta(days=buffer_before)).strftime('%Y-%m-%d')
    overall_end = (latest_end + timedelta(days=buffer_after)).strftime('%Y-%m-%d')

    return overall_start, overall_end, base_start, base_end


def _load_traits_from_config(config):
    """
    Load species traits based on configuration.
    (Internal function - no interactive prompts)
    """
    traits = []

    if config['traits']['source'] == 'builtin':
        traits_df = load_builtin_traits()

        if traits_df is None:
            raise ValueError("Could not load built-in trait database")

        if config['traits']['species_indices']:
            traits_df = traits_df.iloc[config['traits']['species_indices']].reset_index(drop=True)

        for _, row in traits_df.iterrows():
            traits.append({
                "species_name": row.get('species_name'),
                "aphia_id": row.get('aphia_id') if pd.notna(row.get('aphia_id')) else None,
                "spawning_start": str(row['spawning_start']),
                "spawning_end": str(row['spawning_end']),
                "min_PLD": int(row['min_PLD']),
                "max_PLD": int(row['max_PLD'])
            })

        # Apply year
        start_year = config['traits']['start_year']
        for t in traits:
            start_parts = str(t['spawning_start']).split('/')
            end_parts = str(t['spawning_end']).split('/')

            start_day = start_parts[0]
            start_month = int(start_parts[1])
            end_day = end_parts[0]
            end_month = int(end_parts[1])

            start_year_use = start_year
            end_year_use = start_year + 1 if end_month < start_month else start_year

            t['spawning_start'] = f"{start_day}/{start_month:02d}/{start_year_use}"
            t['spawning_end'] = f"{end_day}/{end_month:02d}/{end_year_use}"

        # Save if requested
        if config['traits']['save_csv']:
            os.makedirs(config['traits']['output_dir'], exist_ok=True)
            filename = config['traits']['filename']
            if not filename.endswith('.csv'):
                filename += '.csv'
            species_traits_path = os.path.join(config['traits']['output_dir'], filename)
            pd.DataFrame(traits).to_csv(species_traits_path, index=False)
        else:
            species_traits_path = "builtin_database_selection"

    else:
        species_traits_path = config['traits']['custom_path']
        traits_df = pd.read_csv(species_traits_path)

        for _, row in traits_df.iterrows():
            traits.append({
                "species_name": row.get('species_name', row.get('species', None)),
                "aphia_id": row.get('aphia_id') if pd.notna(row.get('aphia_id', None)) else None,
                "spawning_start": str(row['spawning_start']),
                "spawning_end": str(row['spawning_end']),
                "min_PLD": int(row['min_PLD']),
                "max_PLD": int(row['max_PLD'])
            })

    return traits, species_traits_path
# endregion


# region Main Workflow Function
def run_workflow(config):
    """
    Execute the complete ComCon workflow automatically.

    This function runs the entire workflow using only "pure" functions
    (functions without input() calls inside). The logic from interactive
    wrapper functions is replicated here using config values.

    Parameters
    ----------
    config : str or dict
        Either path to JSON config file or configuration dictionary

    Returns
    -------
    dict
        Results dictionary containing analysis outputs
    """
    # Auto-respond to all input prompts from interactive functions
    def _auto_input(prompt=""):
        """Auto-respond to prompts in automatic mode."""
        prompt_lower = str(prompt).lower()

        # For save/export/edit/map questions, return 'no'
        if any(x in prompt_lower for x in ['save', 'export', 'edit', 'proceed',
                                           'do you want', 'create map',
                                           'create comparison', 'display']):
            print(f"  [Auto] {str(prompt).strip()} → no")
            return 'no'

        # For yes/no confirmations, return 'yes'
        if any(x in prompt_lower for x in ['continue', 'confirm', '(yes/no)']):
            print(f"  [Auto] {str(prompt).strip()} → yes")
            return 'yes'

        # For numeric/selection inputs, return empty (triggers default)
        if any(x in prompt_lower for x in ['default:', 'enter', 'select', '(1-']):
            print(f"  [Auto] {str(prompt).strip()} → [default]")
            return ''

        # Default: return empty string
        print(f"  [Auto] {str(prompt).strip()} → [skipped]")
        return ''

    # Use context manager: Wrap all remaining code in this with block
    with patch('builtins.input', _auto_input):

        # Import only pure functions from interactive.py
        # These functions take parameters and return results WITHOUT prompting
        from comcon.interactive import (
            # Pure functions (NO input() calls inside):
            download_hydrodynamic_data,  # Takes params, downloads data
            create_masks,  # Creates masks from hydro data
            read_geojson,  # Reads GeoJSON file
            get_spp_occ,  # Downloads from GBIF (single species)
            clean_spp_occ,  # Filters occurrence data
            map_spp_to_grid,  # Maps species to grid
            create_fieldset,  # Creates OceanParcels fieldset
            reflective_boundary,  # Creates boundary vectors
            process_sim_results,  # Loads simulation output
            assign_release_id,  # Assigns cohort IDs
            spawning_filter,  # Filters by spawning window
            pld_filter,  # Filters by PLD
            habitat_filter,  # Filters by habitat
            post_process,  # Full post-processing pipeline (per-chunk)
            load_traj_from_chunks,  # Loads full traj when needed
            load_traj_starts_from_chunks,  # Loads only obs=0 rows (for mortality)
            _compute_network_stats,  # Density/modularity/community count for viewers
            calculate_connectivity_metrics,  # Type 1 analysis
            network_analysis,  # Type 2 analysis
            add_custom_composite,  # build a custom composite from chosen metrics
            export_network_results,  # Type 2 CSV export
            compare_models,  # Type 3: baseline vs modified-trait comparison
            display_sensitivity_comparison,  # Type 3: comparison table
            # Visualization
            create_network_map,
            create_connectivity_matrix_plot,
            create_community_map,  # Louvain community map
            create_matrix_with_map,  # Connectivity matrix + reference map panel
            build_heatmap_figure,  # Site x metric heatmap
            save_figure,  # Standard figure saver
            plot_masks,
            plot_spp_occ,
        )

        # Load config if path provided
        if isinstance(config, str):
            subsection_header("Loading Configuration from File")
            config = load_config(config)

        # Create output directory
        os.makedirs(config['output']['directory'], exist_ok=True)

        results = {}

        # Step 1: Hydrodynamic Data Setup
        subsection_header("Step 1: Hydrodynamic and Trait Data")

        # Load traits (using internal non-interactive function)
        print("\nLoading species traits...")
        traits, species_traits_path = _load_traits_from_config(config)
        print(f"Loaded {len(traits)} species")

        # Calculate dispersal period
        overall_start, overall_end, base_start, base_end = _calculate_overall_dispersal_period(
            traits,
            config['traits']['buffer_before'],
            config['traits']['buffer_after']
        )

        runtime = (datetime.strptime(overall_end, '%Y-%m-%d') -
                   datetime.strptime(overall_start, '%Y-%m-%d')).days

        display_line("=")
        print(f"Dispersal period: {overall_start} to {overall_end}")
        print(f"Runtime: {runtime} days")
        display_line("=")

        # Download hydrodynamic data
        print("\nDownloading hydrodynamic data...")
        hydro_safety_buffer = 31
        time_range = [
            datetime.strptime(overall_start, '%Y-%m-%d'),
            datetime.strptime(overall_end, '%Y-%m-%d') + timedelta(days=hydro_safety_buffer)
        ]

        # Create data directory if needed
        if config['copernicus']['output_path']:
            hydro_dir = os.path.dirname(config['copernicus']['output_path'])
            if hydro_dir and not os.path.exists(hydro_dir):
                os.makedirs(hydro_dir, exist_ok=True)

        if config['copernicus'].get('source') == 'local':
            hydro_data = load_hydrodynamic_data(config['copernicus']['local_path'])
        else:
            hydro_data = download_hydrodynamic_data(
                username=config['copernicus']['username'],
                password=config['copernicus']['password'],
                dataset_id=config['copernicus']['dataset_id'],
                longitude_range=[config['copernicus']['min_lon'], config['copernicus']['max_lon']],
                latitude_range=[config['copernicus']['min_lat'], config['copernicus']['max_lat']],
                time_range=time_range,
                export=config['copernicus']['export'],
                output_path=config['copernicus']['output_path']
            )
        print("\n  ✓ Step 1 completed: Hydrodynamic and trait data loaded.\n")

        # Step 2: Species Occurrence Data
        subsection_header("Step 2: Species Occurrence Data")

        # Create masks
        print("\nCreating land and coastal masks...")
        landmask, coastalmask, lonarray, latarray = create_masks(hydro_data)

        # Read study area
        print("Loading study area polygon...")
        geometry, shapely_polygon = read_geojson(config['occurrence']['geojson_path'])

        # Get occurrence data
        if config['occurrence']['data_source'] == 'gbif':
            print("\nDownloading occurrence data from GBIF...")
            all_data = []

            for t in traits:
                species_name = t.get('species_name')
                aphia_id = t.get('aphia_id')

                display_name = species_name or f"AphiaID {aphia_id}"
                print(f"    {display_name}...", end=" ")

                try:
                    records = get_spp_occ(species_name, aphia_id, geometry)
                    all_data.extend(records)
                    print(f"{len(records)} records")
                except Exception as e:
                    print(f"failed: {e}")

            distributions = pd.DataFrame(all_data)
        else:
            print("\n  Loading user-provided occurrence data...")
            distributions = pd.read_csv(config['occurrence']['user_data_path'])

            display_line("=")
            print(f"Total occurrence records: {len(distributions)}")
            display_line("=")

            # Check if we have any data
            if len(distributions) == 0:
                print("\nWarning: No occurrence records found from GBIF.")
                print("This could be because:")
                print("  - Species are not recorded in this region in GBIF")
                print("  - Species names don't match GBIF taxonomy")
                print("\nOptions:")
                print("  1. Use 'all coastal sites' for release/settlement")
                print("  2. Provide your own occurrence data CSV")
                raise ValueError(
                    "No occurrence data available. Please provide your own data or adjust species selection.")

        # Clean occurrence data
        print("\nCleaning occurrence data...")
        clean_spp = clean_spp_occ(
            distributions, landmask, coastalmask, lonarray, latarray,
            buffer_cells=1,
            habitat_type=config['occurrence']['habitat_filtering']
        )

        # Flag occurrence species with no matching traits
        # Step 5 post-processes only species that have traits, so occurrence-only species are released but never processed
        # Auto mode cannot prompt, so it prints a caution and drops them by default (set config['occurrence']['drop_unmatched_species'] = False to keep them)
        if len(clean_spp) > 0:
            trait_names = {standardise_species_name(t.get('species_name')) for t in traits}
            occ_names = {standardise_species_name(n) for n in clean_spp['species'].unique().tolist()}
            unmatched = sorted(occ_names - trait_names)
            if unmatched:
                print(f"\n  Caution: {len(unmatched)} occurrence species have no matching traits:")
                for name in unmatched:
                    print(f"    - {display_species_name(name)}")
                if config['occurrence'].get('drop_unmatched_species', True):
                    clean_spp = clean_spp[clean_spp['species'].apply(
                        lambda s: standardise_species_name(s) in trait_names)].reset_index(drop=True)
                    print(
                        f"  Dropped {len(unmatched)} species. Remaining: {clean_spp['species'].nunique()} species across {len(clean_spp):,} records.")
                else:
                    print(
                        "  Keeping all species. The unmatched species will be released but not post-processed.")
        # Clean occurrence data
        if config['occurrence']['data_source'] != 'sites':
            print("\nCleaning occurrence data...")
            clean_spp = clean_spp_occ(
                distributions, landmask, coastalmask, lonarray, latarray,
                buffer_cells=1,
                habitat_type=config['occurrence']['habitat_filtering']
            )

        if config['occurrence']['save_csv'] and config['occurrence']['csv_path']:
            occ_dir = os.path.dirname(config['occurrence']['csv_path'])
            if occ_dir and not os.path.exists(occ_dir):
                os.makedirs(occ_dir, exist_ok=True)
            clean_spp.to_csv(config['occurrence']['csv_path'], index=False)
            print(f"  Saved to: {config['occurrence']['csv_path']}")

        # Display maps if requested
        if config['output'].get('display_figures', False):
            try:
                # Plot land/coastal/ocean classification
                plot_masks(landmask, coastalmask, lonarray, latarray,
                           config['occurrence']['geojson_path'])

                # Plot species distribution data
                # Skip when no filtering was applied
                if config['occurrence']['habitat_filtering'] != 'none':
                    plot_spp_occ(distributions, clean_spp,
                                 config['occurrence']['geojson_path'],
                                 config['occurrence']['habitat_filtering'],
                                 lonarray, latarray)
            except Exception as e:
                print(f"  Note: Could not display maps: {e}")

        print("\n  ✓ Step 2 completed: Species occurrence data ready.\n")

        # Step 3: Setup the Simulation
        subsection_header("Step 3: Simulation Setup")

        # Create coastal grid
        print("\nCreating release and settlement grids...")
        # Build the grid from:
        # coastal = coastal cells only; ocean = all ocean cells (coastal + offshore).
        if config['occurrence'].get('habitat_filtering') in ('ocean', 'none'):
            coastal_cells = (landmask == 0)
        else:
            coastal_cells = (landmask == 0) & (coastalmask == 1)
        coastal_coords = np.column_stack([
            lonarray[coastal_cells],
            latarray[coastal_cells]
        ])

        # Filter to study area
        study_area_mask = np.array([
            shapely_polygon.contains(Point(lon, lat))
            for lon, lat in coastal_coords
        ])
        study_coastal = coastal_coords[study_area_mask]

        # Create grid dataframe
        coastal_grid_df = pd.DataFrame({
            'longitude': study_coastal[:, 0],
            'latitude': study_coastal[:, 1],
            'grid_id': range(len(study_coastal))
        })

        # Create grid_ids array
        grid_ids = np.full(landmask.shape, -1, dtype=np.int32)
        for idx, row in coastal_grid_df.iterrows():
            lat_idx = np.argmin(np.abs(latarray[:, 0] - row['latitude']))
            lon_idx = np.argmin(np.abs(lonarray[0, :] - row['longitude']))
            grid_ids[lat_idx, lon_idx] = row['grid_id']

            # Setup release grids
            if config['grids']['release_strategy'] == 'all_sites':
                meshgrid = study_coastal
                release_grids = {standardise_species_name(t['species_name']): coastal_grid_df.copy() for t in traits}
            elif config['grids']['release_strategy'] == 'pooled_sites':
                meshgrid, _per_species = map_spp_to_grid(coastal_grid_df, clean_spp)
                pooled_df = pd.concat(_per_species.values(), ignore_index=True)[
                    ['grid_id', 'longitude', 'latitude']].drop_duplicates(subset=['grid_id']).reset_index(drop=True)
                release_grids = {standardise_species_name(t['species_name']): pooled_df.copy() for t in traits}
            else:
                meshgrid, release_grids = map_spp_to_grid(coastal_grid_df, clean_spp)

            # Setup settlement grids
            if config['grids']['settlement_strategy'] == 'all_sites':
                settlement_grids = {standardise_species_name(t['species_name']): coastal_grid_df.copy() for t in traits}
            elif config['grids']['settlement_strategy'] == 'pooled_sites':
                _, _per_species = map_spp_to_grid(coastal_grid_df, clean_spp)
                pooled_df = pd.concat(_per_species.values(), ignore_index=True)[
                    ['grid_id', 'longitude', 'latitude']].drop_duplicates(subset=['grid_id']).reset_index(drop=True)
                settlement_grids = {standardise_species_name(t['species_name']): pooled_df.copy() for t in traits}
            elif config['grids']['settlement_strategy'] == 'species_habitat':
                _, settlement_grids = map_spp_to_grid(coastal_grid_df, clean_spp)
            else:
                custom_data = pd.read_csv(config['grids']['custom_settlement_path'])
                if config['grids'].get('custom_settlement_mode') == 'pooled':
                    _, _per_species = map_spp_to_grid(coastal_grid_df, custom_data)
                    pooled_df = pd.concat(_per_species.values(), ignore_index=True)[
                        ['grid_id', 'longitude', 'latitude']].drop_duplicates(subset=['grid_id']).reset_index(drop=True)
                    settlement_grids = {standardise_species_name(t['species_name']): pooled_df.copy() for t in traits}
                else:
                    _, settlement_grids = map_spp_to_grid(coastal_grid_df, custom_data)

            # Build grid_id -> site_name lookup so names appear in outputs (any data may carry site_name)
            if 'site_name' in clean_spp.columns:
                from collections import defaultdict
                from scipy.spatial import cKDTree
                name_acc = defaultdict(list)
                grid_tree = cKDTree(coastal_grid_df[['latitude', 'longitude']].values)
                for _, row in clean_spp.drop_duplicates(subset=['site_name']).iterrows():
                    _, idx = grid_tree.query([row['latitude'], row['longitude']])
                    name_acc[int(coastal_grid_df.iloc[idx]['grid_id'])].append(str(row['site_name']))
                site_lookup = {gid: ' / '.join(names) for gid, names in name_acc.items()}
                config['grids']['site_name_lookup'] = site_lookup
                import comcon.interactive as _ci
                if getattr(_ci, 'spp_params', None) is None:
                    _ci.spp_params = {}
                _ci.spp_params['site_name_lookup'] = site_lookup
                print(f"  Mapped {len(site_lookup)} grid cells to site names.")
                for gid, names in name_acc.items():
                    if len(names) > 1:
                        joined = "' and '".join(names)
                        print(
                            f"  Note: sites '{joined}' fall in the same grid cell. Your outputs will report their results together.")

        display_line("=")
        print(f"Release sites: {len(meshgrid)}")
        print(f"Species with grids: {len(release_grids)}")
        display_line("=")

        # Create fieldset
        print("\nCreating fieldset...")
        fieldset = create_fieldset(hydro_data, landmask, grid_ids)

        # Parse release interval
        display_line("=")
        print("Particle Configuration")
        display_line("=")

        interval_mapping = {
            'daily': timedelta(days=1),
            'weekly': timedelta(weeks=1),
            'biweekly': timedelta(weeks=2),
            'monthly': timedelta(days=30)
        }

        interval_input = config['simulation']['release_interval']
        if interval_input in interval_mapping:
            interval = interval_mapping[interval_input]
        else:
            interval = timedelta(days=int(interval_input))

        runtime_td = timedelta(days=runtime)
        num_intervals = math.ceil(runtime_td / interval)
        adjusted_runtime = num_intervals * interval

        print(f"Release interval: {interval.days} days")
        print(f"Number of cohorts: {num_intervals}")
        print(f"Adjusted runtime: {adjusted_runtime.days} days")

        # Setup particle arrays
        lats = meshgrid[:, 1].flatten()
        lons = meshgrid[:, 0].flatten()
        npart = config['simulation']['particles_per_location']

        lat_list = list(itertools.chain.from_iterable(itertools.repeat(x, npart) for x in lats))
        lon_list = list(itertools.chain.from_iterable(itertools.repeat(x, npart) for x in lons))

        total_particles = len(lat_list) * num_intervals
        print(f"Particles per location: {npart}")
        print(f"Total particles: {total_particles:}")

        # Add diffusivity
        kh = config['simulation']['horizontal_diffusivity']
        fieldset.add_constant_field("Kh_zonal", kh, mesh="spherical")
        fieldset.add_constant_field("Kh_meridional", kh, mesh="spherical")
        print(f"Horizontal diffusivity: {kh} m²/s")

        # Create particle class
        extra_vars = [
            Variable("distance", initial=0.0, dtype=np.float32),
            Variable("prev_lon", dtype=np.float32, to_write=False, initial=attrgetter("lon")),
            Variable("prev_lat", dtype=np.float32, to_write=False, initial=attrgetter("lat")),
            Variable('on_land', dtype=np.int32, initial=0),
            Variable('grid_id', dtype=np.int32, initial=0)
        ]

        Particle = JITParticle.add_variables(extra_vars)

        # Create ParticleSet
        print("\nCreating ParticleSet...")
        pset = ParticleSet(
            fieldset=fieldset,
            pclass=Particle,
            lon=lon_list,
            lat=lat_list,
            repeatdt=interval
        )

        print("\n  ✓ Step 3 completed: Simulation is setup and ready to execute.\n")

        # Step 4: Execute the Simulation
        subsection_header("Step 4: Execute the Simulation")

        print("\nCreating particle kernels...")
        kernels = create_custom_kernels(pset)

        output_path = config['simulation']['output_path']
        output_dt = timedelta(hours=config['simulation']['output_dt_hours'])
        dt = timedelta(minutes=config['simulation']['dt_minutes'])
        sim_runtime = timedelta(days=adjusted_runtime.days)

        # Create output directory if needed
        sim_dir = os.path.dirname(output_path)
        if sim_dir and not os.path.exists(sim_dir):
            os.makedirs(sim_dir, exist_ok=True)

        display_line("=")
        print("Simulation Settings")
        display_line("=")
        print(f"Output path: {output_path}")
        print(f"Output frequency: {config['simulation']['output_dt_hours']} hours")
        print(f"Timestep: {config['simulation']['dt_minutes']} minutes")
        print(f"Runtime: {adjusted_runtime.days} days")

        # Memory config
        available_gb = check_system_memory()
        mem_config = get_memory_config(available_gb)
        chunk_particles = min(mem_config['chunk_size'], len(pset))

        print("\nRunning simulation (this may take a while)...")
        output_file = pset.ParticleFile(
            name=output_path,
            outputdt=output_dt,
            chunks=(chunk_particles, 1)
        )

        pset.execute(
            kernels,
            runtime=sim_runtime,
            dt=dt,
            output_file=output_file
        )

        print("Simulation complete!")

        # Load results (chunked: data stays on disk for memory efficiency)
        print("\nLoading simulation results...")
        chunks_dir = process_sim_results(output_path, mem_config['chunk_size'], adjusted_runtime.days)
        chunks_dir = assign_release_id(chunks_dir, interval.days)

        # Load only the release rows (obs=0) from the chunks for mortality calculation.
        # Memory-safe and same approach interactive mode uses
        try:
            _traj_raw = load_traj_starts_from_chunks(chunks_dir)
        except Exception:
            _traj_raw = None  # mortality will be skipped if this fails, as in interactive

        print("\n  ✓ Step 4 completed: Particle tracking simulation finished.\n")

        # Step 5: Post-Processing (uses per-chunk processing from disk)
        subsection_header("Step 5: Post-Processing")

        buffer = config['postprocessing']['settlement_buffer']
        final_processed, buffer = post_process(chunks_dir, traits, release_grids, settlement_grids, grid_ids)

        results['final_processed'] = final_processed
        results['traits'] = traits
        results['release_grids'] = release_grids
        results['settlement_grids'] = settlement_grids
        results['coastal_grid_df'] = coastal_grid_df
        results['grid_ids'] = grid_ids
        results['chunks_dir'] = chunks_dir

        # Apply species filtering for custom selection (used by all analysis types)
        scope = config['analysis'].get('scope', 'both')
        selected_indices = config['analysis'].get('selected_species_indices', None)

        if scope == 'custom' and selected_indices is not None:
            species_list = list(final_processed.keys())
            selected_species = [species_list[i] for i in selected_indices if i < len(species_list)]
            filtered_processed = {sp: final_processed[sp] for sp in selected_species}
        else:
            filtered_processed = final_processed

        print("\n  ✓ Step 5 completed: Post-processing finished.\n")

        # Step 6: Connectivity Analyses
        subsection_header("Step 6: Connectivity Analysis")

        # Type 1: Connectivity Metrics
        if config['analysis'].get('run_type1', True):
            subsection_header("Type 1: Connectivity Metrics")

            type1_results = {}
            run_comm = scope in ('community', 'both')
            run_persp = scope in ('per_species', 'custom', 'both')

            # Community level (all species combined)
            if run_comm:
                type1_results['community'] = calculate_connectivity_metrics(
                    filtered_processed, release_grids, settlement_grids, traits, _traj_raw,
                    print_summary=False, grid_ids=grid_ids
                )

            # Per-species level
            if run_persp:
                type1_results['per_species'] = {}
                for sp, df in filtered_processed.items():
                    if df is not None and not df.empty:
                        sp_data = {sp: df}
                        sp_release = {sp: release_grids.get(sp)} if sp in release_grids else {}
                        sp_settle = ({sp: settlement_grids.get(sp)}
                                     if settlement_grids and sp in settlement_grids else None)
                        sp_traits = [t for t in traits
                                     if standardise_species_name(t.get('species_name', '')) == sp]
                        if sp_traits:
                            type1_results['per_species'][sp] = calculate_connectivity_metrics(
                                sp_data, sp_release, sp_settle, sp_traits, _traj_raw,
                                print_summary=False, grid_ids=grid_ids
                            )

            results['type1'] = type1_results
            print("  Connectivity metrics calculated")

            # Export Type 1 metrics to CSV
            try:
                t1_root = config['output'].get('results_dir', config['project'].get('results', '.'))
                os.makedirs(t1_root, exist_ok=True)
                for _key, _result_data in type1_results.items():
                    if _key == 'per_species':
                        for _sp, _sp_data in _result_data.items():
                            if 'community' in _sp_data:
                                pd.DataFrame([_sp_data['community']]).to_csv(
                                    os.path.join(t1_root, f'connectivity_metrics_{_sp}.csv'), index=False)
                    elif 'community' in _result_data:
                        pd.DataFrame([_result_data['community']]).to_csv(
                            os.path.join(t1_root, f'connectivity_metrics_{_key}.csv'), index=False)
                print(f"  Type 1 metric CSVs saved to: {t1_root}")
            except Exception as _t1_err:
                print(f"  Note: Type 1 CSV export skipped ({_t1_err}).")

        # Type 2: Network Analysis
        if config['analysis'].get('run_type2', True):
            subsection_header("Type 2: Network Analysis")

            scales = config['analysis'].get('type2_scales', [30])
            all_network_results = {}
            run_comm = scope in ('community', 'both')
            run_persp = scope in ('per_species', 'custom', 'both')
            excl = config['analysis'].get('exclude_self_recruitment', True)

            for scale in scales:
                scale_name = 'full_domain' if scale is None else f'{scale}_km'
                print(f"\nProcessing scale: {scale_name}")
                scale_results = {}

                # Community-level network analysis
                if run_comm:
                    community_network = network_analysis(
                        filtered_processed, release_grids, settlement_grids,
                        coastal_grid_df, grid_ids, coastal_grid_df, scale,
                        exclude_self_recruitment=excl, verbose=False
                    )
                    if community_network:
                        _compute_network_stats(community_network)
                        scale_results['community'] = community_network

                # Per-species network analysis
                if run_persp:
                    scale_results['per_species'] = {}
                    for sp, df in filtered_processed.items():
                        if df is not None and not df.empty:
                            sp_data = {sp: df}
                            sp_settle = ({sp: settlement_grids.get(sp)}
                                         if settlement_grids and sp in settlement_grids else None)
                            # Pass FULL release_grids to keep a consistent study extent across species
                            sp_network = network_analysis(
                                sp_data, release_grids, sp_settle,
                                coastal_grid_df, grid_ids, coastal_grid_df, scale,
                                exclude_self_recruitment=excl, verbose=False
                            )
                            if sp_network:
                                _compute_network_stats(sp_network)
                                scale_results['per_species'][sp] = sp_network

                if scale_results:
                    all_network_results[scale_name] = scale_results

            results['type2'] = all_network_results

            # Build the user-selected custom composite (if chosen) so it appears in the CSVs, dropdown viewer and figures
            if config['analysis'].get('type2_composite') == 'custom' and config['analysis'].get('custom_metrics'):
                _custom_ok = True
                for scale_data in all_network_results.values():
                    targets = []
                    if 'community' in scale_data:
                        targets.append(scale_data['community'])
                    if 'per_species' in scale_data:
                        targets.extend(scale_data['per_species'].values())
                    for nr in targets:
                        try:
                            add_custom_composite(nr, config['analysis']['custom_metrics'])
                        except Exception as _cc_err:
                            print(
                                f"  Warning: custom composite could not be built ({_cc_err}); using the overall composite instead.")
                            _custom_ok = False
                            break
                    if not _custom_ok:
                        break
                if not _custom_ok:
                    config['analysis']['type2_composite'] = 'overall'

            # Export Type 2 network CSVs to the results folder (data kept separate from figures)
            results_root = config['output'].get('results_dir', config['project'].get('results', '.'))
            for s_name, scale_data in all_network_results.items():
                pairs = []
                if 'community' in scale_data:
                    pairs.append(('community', scale_data['community']))
                if 'per_species' in scale_data:
                    for sp, sp_nr in scale_data['per_species'].items():
                        pairs.append((sp, sp_nr))
                for level_label, nr in pairs:
                    csv_subdir = os.path.join(results_root, f'{level_label}_{s_name}')
                    os.makedirs(csv_subdir, exist_ok=True)
                    try:
                        with contextlib.redirect_stdout(io.StringIO()):
                            export_network_results(nr, output_dir=csv_subdir)
                    except Exception:
                        pass
            print(f"  Result CSVs saved to: {results_root}")

            # Figure export
            if (config['analysis'].get('type2_visualizations', 'skip') != 'skip'
                    and config['output'].get('save_figures', True)):
                fig_root = config['output'].get('figures_dir', config['output']['directory'])
                fmt = config['output'].get('format', 'tif')
                _dpi = config['output'].get('dpi', 300)
                subsection_header("Saving figures")
                _composite_col_map = {'functional': 'functional_composite', 'centrality': 'centrality_composite',
                                      'overall': 'overall_composite', 'custom': 'custom_composite'}
                _chosen_col = _composite_col_map.get(config['analysis'].get('type2_composite', 'overall'),
                                                     'overall_composite')

                for s_name, scale_data in all_network_results.items():
                    pairs = []
                    if 'community' in scale_data:
                        pairs.append(('community', scale_data['community']))
                    if 'per_species' in scale_data:
                        for sp, sp_nr in scale_data['per_species'].items():
                            pairs.append((sp, sp_nr))

                    for level_label, nr in pairs:
                        fig_subdir = os.path.join(fig_root, f'{level_label}_{s_name}')
                        os.makedirs(fig_subdir, exist_ok=True)
                        print(f"  Saving: {level_label} – {s_name}")
                        with contextlib.redirect_stdout(io.StringIO()):
                            _sm = nr['site_metrics']
                            _cm = nr['connectivity_matrix']
                            _t10 = nr.get('top10_overall', _sm.head(10))
                            _ng = nr.get('network_graph')
                            _title = f'{level_label} – {s_name}'

                            save_list = [
                                ('hotspot_overall',
                                 lambda sm=_sm, t10=_t10, ng=_ng, t=_title, cc=_chosen_col: create_network_map(
                                     sm, (sm.nsmallest(10, cc) if cc in sm.columns else t10), title=t,
                                     composite_col=(cc if cc in sm.columns else 'overall_composite'), cmap='viridis_r',
                                     show_figure=False, network_graph=ng, top_n_edges=200,
                                 )),
                                ('network_only', lambda sm=_sm, ng=_ng, t=_title: create_network_map(
                                    sm, sm.head(0), title=f"{t} (network only)",
                                    composite_col='overall_composite', cmap='viridis_r',
                                    show_figure=False, network_graph=ng,
                                    top_n_edges=None, network_only=True,
                                )),
                                ('communities', lambda nr_=nr, t=_title: create_community_map(
                                    nr_, title=t, show_figure=False)[0]),
                                ('connectivity_matrix', lambda cm=_cm, sm=_sm, t=_title: create_matrix_with_map(
                                    cm, sm, title=t, show_figure=False)),
                                ('heatmap', lambda nr_=nr, t=_title: build_heatmap_figure(
                                    nr_, title=t)),
                            ]

                            for metric_col, metric_label in [
                                ('source', 'source_strength'),
                                ('sink', 'sink_strength'),
                                ('self_recruitment', 'self_recruitment'),
                                ('betweenness', 'betweenness_centrality'),
                                ('closeness', 'closeness_centrality'),
                                ('degree', 'degree_centrality'),
                            ]:
                                if metric_col in _sm.columns:
                                    save_list.append((
                                        f'hotspot_{metric_label}',
                                        lambda sm=_sm, ng=_ng, t=_title, mc=metric_col: create_network_map(
                                            sm, sm.nlargest(10, mc),
                                            title=f"{t} ({mc})",
                                            composite_col=mc, cmap='viridis_r',
                                            show_figure=False, network_graph=ng, top_n_edges=200,
                                        )
                                    ))

                            for fn, make_fig in save_list:
                                try:
                                    fig = make_fig()
                                    if fig is not None:
                                        save_figure(fig, os.path.join(fig_subdir, fn), format_ext=fmt, dpi=_dpi)
                                        plt.close(fig)
                                except Exception:
                                    pass
                print(f"  Figures saved to: {fig_root}")

        # Type 3: Trait Sensitivity
        if config['analysis'].get('run_type3', False):
            subsection_header("Type 3: Trait Sensitivity Analysis")

            # Trait sensitivity
            try:
                mod_type = config['analysis'].get('trait_mod_type', 'pld')
                min_pld_change = config['analysis'].get('min_pld_change', 0)
                max_pld_change = config['analysis'].get('max_pld_change', 0)
                start_shift = config['analysis'].get('spawning_start_shift', 0)
                end_shift = config['analysis'].get('spawning_end_shift', 0)
                t3_scale = config['analysis'].get('type3_scale', None)
                exclude_lr = config['analysis'].get('exclude_self_recruitment', True)

                # Build the modified trait set from the pre-collected config
                modified_traits = []
                if mod_type == 'pld':
                    for t in traits:
                        new_t = t.copy() if isinstance(t, dict) else t.to_dict()
                        new_t['min_PLD'] = max(1, t['min_PLD'] + min_pld_change)
                        new_t['max_PLD'] = max(1, t['max_PLD'] + max_pld_change)
                        modified_traits.append(new_t)
                    scenario_name = (f"PLD min{'+' if min_pld_change >= 0 else ''}{min_pld_change}, "
                                     f"max{'+' if max_pld_change >= 0 else ''}{max_pld_change} days")
                else:
                    for t in traits:
                        new_t = t.copy() if isinstance(t, dict) else t.to_dict()
                        s = pd.to_datetime(t['spawning_start'], dayfirst=True)
                        e = pd.to_datetime(t['spawning_end'], dayfirst=True)
                        new_t['spawning_start'] = (s + pd.Timedelta(days=start_shift)).strftime('%d/%m/%Y')
                        new_t['spawning_end'] = (e + pd.Timedelta(days=end_shift)).strftime('%d/%m/%Y')
                        modified_traits.append(new_t)
                    scenario_name = (f"Spawn start{'+' if start_shift >= 0 else ''}{start_shift}, "
                                     f"end{'+' if end_shift >= 0 else ''}{end_shift} days")

                print(f"\n  Scenario: {scenario_name}")

                # Baseline network: reuse the Type 2 result if available, else compute it
                if 'type2' in results and results['type2']:
                    original_network = list(results['type2'].values())[0]
                else:
                    print("  Running baseline network analysis...")
                    original_network = network_analysis(
                        filtered_processed, release_grids, settlement_grids,
                        coastal_grid_df, grid_ids, coastal_grid_df, t3_scale,
                        exclude_self_recruitment=exclude_lr
                    )

                # Re-run post-processing + network analysis with the modified traits.
                # post_process returns (final_processed, buffer) — unpack the tuple.
                print("  Re-running post-processing with modified traits...")
                modified_processed, _ = post_process(
                    chunks_dir, modified_traits, release_grids, settlement_grids, grid_ids
                )

                print("  Running network analysis on the modified scenario...")
                modified_network = network_analysis(
                    modified_processed, release_grids, settlement_grids,
                    coastal_grid_df, grid_ids, coastal_grid_df, t3_scale,
                    exclude_self_recruitment=exclude_lr
                )

                if modified_network:
                    comparison = compare_models(original_network, modified_network)

                    original_type1 = calculate_connectivity_metrics(
                        filtered_processed, release_grids, settlement_grids, traits, None,
                        print_summary=False, grid_ids=grid_ids
                    )
                    modified_type1 = calculate_connectivity_metrics(
                        modified_processed, release_grids, settlement_grids, modified_traits, None,
                        print_summary=False, grid_ids=grid_ids
                    )

                    display_sensitivity_comparison(
                        original_type1, modified_type1, comparison,
                        title=f"Original vs {scenario_name}"
                    )

                    results['type3'] = {
                        'scenario': scenario_name,
                        'comparison': comparison,
                        'original_network': original_network,
                        'modified_network': modified_network,
                        'original_metrics': original_type1,
                        'modified_metrics': modified_type1,
                    }
                    print("\n  Type 3 sensitivity analysis complete.")

            except Exception as _t3_err:
                print(f"\n  Error in trait sensitivity: {_t3_err}")
                traceback.print_exc()

            # ── Workflow complete (runs regardless of which analyses were selected) ──
        section_header("Automatic Workflow Complete")

        # Display the interactive viewers so the user can inspect results
        # when they return to the notebook. Same viewers as interactive mode.
        try:
            from comcon.interactive import (
                trait_metrics_viewer, network_metrics_viewer,
                trait_matrix_viewer, trait_community_viewer,
                trait_heatmap_viewer, trait_metric_viewer,
            )

            if 'type1' in results:
                subsection_header("Type 1: Connectivity Metrics — Interactive Viewer")
                trait_metrics_viewer(results['type1'])

            if 'type2' in results:
                subsection_header("Type 2: Network Analysis — Interactive Viewers")
                network_metrics_viewer(results['type2'])
                trait_matrix_viewer(results['type2'])
                trait_metric_viewer(results['type2'])
                trait_heatmap_viewer(results['type2'])
                trait_community_viewer(results['type2'])
        except Exception as _viewer_err:
            print(f"  Note: interactive viewers could not be displayed ({_viewer_err}).")
            print("  Results dictionary is still returned and can be passed to viewers manually.")

        print(f"\nOutput directory: {config['output']['directory']}")

        analyses_run = []
        if 'type1' in results:
            analyses_run.append("Type 1: Connectivity Metrics")
        if 'type2' in results:
            analyses_run.append("Type 2: Network Analysis")
        if 'type3' in results:
            analyses_run.append("Type 3: Trait Sensitivity")

        print("\nCompleted analyses:")
        for a in analyses_run:
            print(f"    - {a}")

        print("\nResults stored in returned dictionary.")
        print("Access with: results['type1'], results['type2'], etc.")

        # Save metadata file (comcon_outputs.xlsx), matching interactive and passive.
        try:
            import comcon.interactive as _ci
            _loc = locals()
            metadata_path = os.path.join(config['project'].get('config', '.'), 'comcon_outputs.xlsx')
            # Build the fields trait_metadata actually reads, from automatic's config
            _spp_meta = {
                'release_strategy': config.get('grids', {}).get('release_strategy',
                                                                config.get('occurrence', {}).get('data_source',
                                                                                                 'unknown')),
                'settlement_strategy': config.get('grids', {}).get('settlement_strategy', 'unknown'),
                'habitat_filtering': config.get('occurrence', {}).get('habitat_filtering', 'unknown'),
                'data_source': config.get('occurrence', {}).get('data_source', 'unknown'),
                'has_site_names': bool((getattr(_ci, 'spp_params', {}) or {}).get('site_name_lookup')),
            }
            _hydro_meta = {'dataset_id': config.get('copernicus', {}).get('dataset_id', 'unknown')}
            metadata_dict = _ci.trait_metadata(
                _spp_meta,
                _loc.get('traits', []),
                _hydro_meta,
                _loc.get('runtime', None),
                _loc.get('adjusted_runtime', None),
                _loc.get('interval', None),
                _loc.get('release_grids', {}),
                _loc.get('settlement_grids', {}),
            )
            _ci.write_project_metadata(config['project'], metadata_dict, 'trait', metadata_path)
            print(f"  Metadata saved: {metadata_path}")
        except Exception as _meta_err:
            print(f"  Note: could not save metadata file ({_meta_err}).")

        print("\n  ✓ Step 6 completed: Connectivity analyses finished.\n")

        return results


# endregion


# region Module Exports
__all__ = [
    'start_automatic_mode',
    'run_workflow',
    'collect_all_parameters',
    'load_config',
]
# endregion