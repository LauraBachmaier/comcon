# region Automatic Mode Functions
"""
ComCon Automatic Mode
=====================

This module provides automatic workflow execution for the ComCon package.
All parameters are collected upfront, then the entire workflow runs
with progress updates only (no interactive prompts during execution).

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
    centered_text, completion_message
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
def get_yes_no(prompt, default='no'):
    """Get yes/no input with error handling."""
    while True:
        if default:
            user_input = input(f"{prompt} [default: {default}]: ").strip().lower() or default
        else:
            user_input = input(f"{prompt}: ").strip().lower()

        if user_input in ['yes', 'y', 'no', 'n']:
            return user_input in ['yes', 'y']
        else:
            print("  Invalid input. Please enter 'yes' or 'no'")
# endregion

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
    After setup, the workflow runs automatically with progress updates only.
        """)

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

    subsection_header("Choose an Option")
    print("\n  1. New analysis (collect all parameters now)")
    print("  2. Re-run with saved settings (load from config file)")

    while True:
        choice = input("\nEnter choice (1/2) [default: 1]: ").strip() or '1'

        if choice == '1':
            # New analysis - collect parameters then run
            subsection_header("Starting Parameter Collection")
            print("You will be prompted for all workflow parameters.")
            print("Once complete, the workflow will run automatically.\n")

            config = collect_all_parameters()

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
def collect_all_parameters():
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

    print("\nComCon organises all outputs into a project folder.")
    print("Each project gets its own directory with subfolders for data, figures, results, etc.\n")

    print("  1. Create a new ComCon project folder (recommended)")
    print("  2. Use your own directory\n")

    setup_choice = input("Select option (1/2) [default: 1]: ").strip() or '1'

    import re
    subfolder_names = ['config', 'data', 'occurrence', 'grids', 'simulation', 'figures', 'results']

    if setup_choice == '2':
        custom_path = input("\nEnter the path to your directory: ").strip()
        os.makedirs(custom_path, exist_ok=True)

        create_subs = input(
            "Create ComCon subfolders inside this directory? (yes/no) [default: yes]: "
        ).strip().lower() or 'yes'

        if create_subs in ['yes', 'y']:
            root = custom_path
            project_dirs = {'root': root}
            for name in subfolder_names:
                project_dirs[name] = os.path.join(root, name)
            for path in project_dirs.values():
                os.makedirs(path, exist_ok=True)
            print(f"\n  Project directory created: {root}")
            print(f"  Subfolders: {', '.join(subfolder_names)}")
        else:
            project_dirs = {key: custom_path for key in ['root'] + subfolder_names}
            print(f"\n  All outputs will be saved to: {custom_path}")
    else:
        default_name = "ComCon_Project"
        project_name = input(
            f"Enter a name for this project/simulation [default: {default_name}]: "
        ).strip() or default_name
        safe_name = re.sub(r'[^\w\-]', '_', project_name)

        root = os.path.join(".", "projects", safe_name)
        project_dirs = {'root': root}
        for name in subfolder_names:
            project_dirs[name] = os.path.join(root, name)
        for path in project_dirs.values():
            os.makedirs(path, exist_ok=True)

        print(f"\n  Project directory created: {root}")
        print(f"  Subfolders: {', '.join(subfolder_names)}")

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
        local_path = input("  Path to your hydrodynamic data file (.nc or .zarr): ").strip()
        config['copernicus']['local_path'] = local_path
        config['copernicus']['export'] = False
        # Skip to traits section
    else:
        config['copernicus'] = {}
        print("\nCopernicus Marine Service credentials:")
        config['copernicus']['username'] = input("  Username: ").strip()
        config['copernicus']['password'] = getpass.getpass("  Password: ").strip()
        default_dataset = 'cmems_mod_ibi_phy_anfc_0.027deg-2D_PT1H-m'

        print("\n  Common datasets:")
        print("    IBI (Atlantic-Iberian):     cmems_mod_ibi_phy_anfc_0.027deg-2D_PT1H-m")
        print("    NWS (NW European Shelf):    cmems_mod_nws_phy-cur_anfc_1.5km-2D_PT1H-i")
        print("    Global (0.083° resolution): cmems_mod_glo_phy_anfc_0.083deg_PT1H-m")
        print("\n  To browse all available datasets, enter 'browse' to open the Copernicus website.\n")

        dataset_input = input(f"  Dataset ID [default: {default_dataset}]: ").strip()

        if dataset_input.lower() == 'browse':
            import webbrowser
            url = "https://data.marine.copernicus.eu/products"
            webbrowser.open(url)
            print("  Opening Copernicus Marine Data Store in your browser...")
            print("  Find your dataset, copy its ID and paste it below.\n")
            dataset_input = input(f"  Dataset ID [default: {default_dataset}]: ").strip()

        config['copernicus']['dataset_id'] = dataset_input if dataset_input else default_dataset

        print("\nHydrodynamic domain extent (must cover your study area + buffer):")
        while True:
            try:
                config['copernicus']['min_lon'] = float(input("  Minimum longitude (e.g., -6.0): "))
                config['copernicus']['max_lon'] = float(input("  Maximum longitude (e.g., 2.0): "))
                if config['copernicus']['min_lon'] >= config['copernicus']['max_lon']:
                    print("  Error: min_lon must be less than max_lon")
                    continue
                break
            except ValueError:
                print("  Error: Please enter valid numbers")

        while True:
            try:
                config['copernicus']['min_lat'] = float(input("  Minimum latitude (e.g., 49.0): "))
                config['copernicus']['max_lat'] = float(input("  Maximum latitude (e.g., 52.0): "))
                if config['copernicus']['min_lat'] >= config['copernicus']['max_lat']:
                    print("  Error: min_lat must be less than max_lat")
                    continue
                break
            except ValueError:
                print("  Error: Please enter valid numbers")

        print("\n  Note: ComCon streams hydrodynamic data directly from Copernicus (using lazy loading).")
        print("  You do not need to download the data to run your analysis.")
        print("  Downloading is only useful if you want a local copy for offline use or to avoid re-downloading in future sessions.")
        print("  Warning: Downloaded files are often large (multiple GB) and take a long time to download (several hours).\n")

        export = input("  Save hydrodynamic data to file? (yes/no) [default: no]: ").strip().lower()
        config['copernicus']['export'] = export not in ['no', 'n']

        if config['copernicus']['export']:
            print("\n  Export format:")
            print("    1. NetCDF (.nc) Standard scientific data format")
            print("    2. Zarr (.zarr) Smaller, faster, recommended for large datasets")
            export_fmt = input("  Choose format (1/2) [default: 1]: ").strip() or '1'
            config['copernicus']['export_format'] = 'zarr' if export_fmt == '2' else 'netcdf'
        else:
            config['copernicus']['export_format'] = 'netcdf'

        if config['copernicus']['export']:
            default_hydro_path = os.path.join(config['project']['data'], 'hydro_data.nc')
            hydro_path = input(f"  Output path [default: {default_hydro_path}]: ").strip()
            config['copernicus']['output_path'] = hydro_path if hydro_path else default_hydro_path
            if not config['copernicus']['output_path'].endswith('.nc'):
                config['copernicus']['output_path'] += '.nc'
        else:
            config['copernicus']['output_path'] = None

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
            config['traits']['custom_path'] = input("  Path to your trait CSV file: ").strip()
            if not config['traits']['custom_path'].endswith('.csv'):
                config['traits']['custom_path'] += '.csv'
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
                from comcon.interactive import load_builtin_traits
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
        while True:
            try:
                year_input = input("\nStart year for dispersal season (e.g., 2023): ").strip()
                config['traits']['start_year'] = int(year_input)
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
        config['occurrence']['data_source'] = 'gbif' if data_source == '1' else 'user'

        if config['occurrence']['data_source'] == 'user':
            config['occurrence']['user_data_path'] = input("  Path to occurrence CSV file: ").strip()
            if not config['occurrence']['user_data_path'].endswith('.csv'):
                config['occurrence']['user_data_path'] += '.csv'
        else:
            config['occurrence']['user_data_path'] = None

        print("\nStudy area polygon (GeoJSON format):")
        print("See the tip at the start of this workflow if you need to create one.\n")

        while True:
            geojson_path = input("\nPath to GeoJSON file: ").strip()

            # Check if extension already present (case-insensitive)
            if geojson_path and not geojson_path.lower().endswith('.geojson'):
                geojson_path += '.geojson'

            # Check file exists
            if os.path.exists(geojson_path):
                config['occurrence']['geojson_path'] = geojson_path
                print(f"File found: {geojson_path}")
                break
            else:
                print(f"File not found: {geojson_path}")
                print("Please check the path and try again.")

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

        config['grids'] = {}

        # Get the habitat type from earlier selection
        habitat_type = config['occurrence']['habitat_filtering']
        habitat_label = {'coastal': 'coastal', 'ocean': 'ocean', 'none': 'all'}.get(habitat_type, 'coastal')

        print(f"\nBased on your habitat filtering selection: {habitat_label}")

        print("\nRelease strategy (where particles start):")
        print(f"  1. All {habitat_label} sites in study area")
        print(f"  2. Species-specific sites (where each species occurs)")

        release_choice = input("\nSelect option (1/2) [default: 2]: ").strip() or '2'
        config['grids']['release_strategy'] = 'all_sites' if release_choice == '1' else 'species_sites'

        print("\nSettlement strategy (where particles can settle):")
        print(f"  1. All {habitat_label} sites in study area")
        print(f"  2. Species-specific habitat (same as occurrence sites)")
        print(f"  3. Custom settlement locations (separate CSV)")

        settle_choice = input("\nSelect option (1/2/3) [default: 2]: ").strip() or '2'
        settle_map = {'1': 'all_sites', '2': 'species_habitat', '3': 'custom_habitat'}
        config['grids']['settlement_strategy'] = settle_map.get(settle_choice, 'species_habitat')

        if config['grids']['settlement_strategy'] == 'custom_habitat':
            config['grids']['custom_settlement_path'] = input("  Path to custom settlement CSV: ").strip()
            if not config['grids']['custom_settlement_path'].endswith('.csv'):
                config['grids']['custom_settlement_path'] += '.csv'
        else:
            config['grids']['custom_settlement_path'] = None

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
                from comcon.interactive import load_builtin_traits
                all_traits = load_builtin_traits()
                selected_traits = all_traits.iloc[config['traits']['species_indices']]
                for i, (_, row) in enumerate(selected_traits.iterrows()):
                    print(f"  {i + 1}. {row['species_name']}")
                max_species = len(selected_traits)
            else:
                # Show all species from config
                print("  (Species will be numbered when traits are loaded)")
                print("  Enter species numbers after seeing the full list, or")
                print("  enter 'all' to analyse all species individually")
                max_species = 999  # Will be validated later

            print("\nEnter species numbers to analyse (comma-separated, e.g., 1,3,5)")
            print("Or enter 'all' to run per-species analysis on all")
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
        print("  Type 3: Trait sensitivity (compare trait vs null models)")

        run_type1 = input("\nRun Type 1: Connectivity Metrics? (yes/no) [default: yes]: ").strip().lower()
        config['analysis']['run_type1'] = run_type1 not in ['no', 'n']

        run_type2 = input("Run Type 2: Network Analysis? (yes/no) [default: yes]: ").strip().lower()
        config['analysis']['run_type2'] = run_type2 not in ['no', 'n']

        if config['analysis']['run_type2']:
            print("\nNetwork analysis scale selection:")
            print("  1. Full domain (no distance filtering, all connections included)")
            print("  2. Single scale analysis")
            print("  3. Multi-scale analysis (local, regional, seascape)")

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
            print("  1. Composite measure of functional roles (source strength, sink strength, recruitment)")
            print("  2. Composite measure of centrality measures (betweenness, closeness, degree)")
            print("  3. Overall composite (all metrics combined)")
            print("  4. Custom selection of one or several metrics")
            print("\n A composite measure detects overall connectivity hotspots across several key metrics.")

            composite_choice = input("\nSelect option (1/2/3/4) [default: 3]: ").strip() or '3'
            if composite_choice == '4':
                print("\nAvailable metrics: source, sink, recruitment, betweenness, closeness, degree")
                custom_metrics = input("Enter metrics to combine (comma-separated): ").strip()
                config['analysis']['type2_composite'] = 'custom'
                config['analysis']['custom_metrics'] = [m.strip() for m in custom_metrics.split(',')]
            else:
                composite_map = {'1': 'functional', '2': 'centrality', '3': 'overall'}
                config['analysis']['type2_composite'] = composite_map.get(composite_choice, 'overall')

            print("\nVisualization options:")
            print("  1. Network map only")
            print("  2. Network map + connectivity matrix")
            print("  3. Skip visualizations")

            vis_choice = input("\nSelect option (1/2/3) [default: 1]: ").strip() or '1'
            vis_map = {'1': 'network_only', '2': 'network_and_matrix', '3': 'skip'}
            config['analysis']['type2_visualizations'] = vis_map.get(vis_choice, 'network_only')
        else:
            config['analysis']['type2_scales'] = [None]
            config['analysis']['scale_type'] = 'full_domain'
            config['analysis']['type2_composite'] = 'overall'
            config['analysis']['type2_visualizations'] = 'skip'

        run_type3 = input("\nRun Type 3: Trait Sensitivity Analysis? (yes/no) [default: no]: ").strip().lower()
        config['analysis']['run_type3'] = run_type3 in ['yes', 'y']

        if config['analysis']['run_type3']:
            print("\nType 3 comparison:")
            print("  1. Trait model vs null model (tests if traits matter)")
            print("  2. Trait sensitivity (test alternative trait values)")

            type3_choice = input("\nSelect option (1/2) [1]: ").strip() or '1'
            config['analysis']['type3_option'] = 'null_comparison' if type3_choice == '1' else 'trait_sensitivity'

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

            if config['analysis']['type3_option'] == 'trait_sensitivity':
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
            else:
                # Null comparison - no trait modification needed
                config['analysis']['trait_mod_type'] = None
                config['analysis']['min_pld_change'] = 0
                config['analysis']['max_pld_change'] = 0
                config['analysis']['spawning_start_shift'] = 0
                config['analysis']['spawning_end_shift'] = 0

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
        params.append((num, "Release Strategy", cfg.get('grids', {}).get('release_strategy', 'species_sites'), 'grids', 'release_strategy'))
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
    # Import the pure function from interactive
    from comcon.interactive import load_builtin_traits

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
            calculate_connectivity_metrics,  # Type 1 analysis
            network_analysis,  # Type 2 analysis
            create_null_model,  # Creates null model
            compare_models,  # Compares models
            # Visualization
            create_network_map,
            create_connectivity_matrix_plot,
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

        if config['occurrence']['save_csv'] and config['occurrence']['csv_path']:
            occ_dir = os.path.dirname(config['occurrence']['csv_path'])
            if occ_dir and not os.path.exists(occ_dir):
                os.makedirs(occ_dir, exist_ok=True)
            clean_spp.to_csv(config['occurrence']['csv_path'], index=False)
            print(f"  Saved to: {config['occurrence']['csv_path']}")

        # Display maps if requested
        if config['output'].get('display_figures', False):
            from comcon.interactive import plot_spp_occ, plot_masks
            try:
                # Plot land/coastal/ocean classification
                plot_masks(landmask, coastalmask, lonarray, latarray,
                           config['occurrence']['geojson_path'])

                # Plot species distribution data
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
        else:
            meshgrid, release_grids = map_spp_to_grid(coastal_grid_df, clean_spp)

        # Setup settlement grids
        if config['grids']['settlement_strategy'] == 'all_sites':
            settlement_grids = {standardise_species_name(t['species_name']): coastal_grid_df.copy() for t in traits}
        elif config['grids']['settlement_strategy'] == 'species_habitat':
            _, settlement_grids = map_spp_to_grid(coastal_grid_df, clean_spp)
        else:
            custom_data = pd.read_csv(config['grids']['custom_settlement_path'])
            _, settlement_grids = map_spp_to_grid(coastal_grid_df, custom_data)

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

        # Import kernel creation (this is pure - just defines kernels)
        from comcon.interactive import create_custom_kernels

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

        # Load results
        print("\nLoading simulation results...")
        traj_df = process_sim_results(output_path, mem_config['chunk_size'], adjusted_runtime.days)
        traj = assign_release_id(traj_df, interval.days)

        display_line("-")
        print(f"Trajectory observations: {len(traj):,}")
        print(f"Unique trajectories: {traj['trajectory'].nunique():,}")
        display_line("-")

        print("\n  ✓ Step 4 completed: Particle tracking simulation finished.\n")

        # Step 5: Post-Processing
        subsection_header("Step 5: Post-Processing")

        print("\nApplying spawning seasonality filter...")
        spawning_dfs = spawning_filter(traj, traits)

        print("Applying PLD filter...")
        buffer = config['postprocessing']['settlement_buffer']
        pld_dfs = pld_filter(spawning_dfs, traits, settlement_grids, grid_ids, buffer)

        print("Applying habitat filter...")
        final_processed = habitat_filter(pld_dfs, release_grids)

        # Summary
        total_successful = sum(
            df['trajectory'].nunique() for df in final_processed.values() if not df.empty
        )

        display_line("-")
        print("Post-processing Summary:")
        for species, df in final_processed.items():
            if not df.empty:
                n_traj = df['trajectory'].nunique()
                print(f"    {display_species_name(species)}: {n_traj:,} trajectories")
        print(f"\nTotal successful: {total_successful:,}")
        display_line("-")

        results['final_processed'] = final_processed
        results['traits'] = traits
        results['release_grids'] = release_grids
        results['settlement_grids'] = settlement_grids
        results['coastal_grid_df'] = coastal_grid_df
        results['grid_ids'] = grid_ids
        results['traj'] = traj

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

            if scope == 'community':
                # Run on all species combined
                type1_results = calculate_connectivity_metrics(
                    filtered_processed, release_grids, settlement_grids, traits, traj, grid_ids=grid_ids
                )
            elif scope in ['per_species', 'custom']:
                # Run on each species separately
                type1_results = {'per_species': {}}
                for sp, df in filtered_processed.items():
                    if not df.empty:
                        sp_data = {sp: df}
                        sp_release = {sp: release_grids.get(sp)} if sp in release_grids else {}
                        sp_settle = {
                            sp: settlement_grids.get(sp)} if settlement_grids and sp in settlement_grids else None
                        sp_traits = [t for t in traits if standardise_species_name(t.get('species_name', '')) == sp]
                        if sp_traits:
                            type1_results['per_species'][sp] = calculate_connectivity_metrics(
                                sp_data, sp_release, sp_settle, sp_traits, traj, grid_ids=grid_ids
                            )
            else:  # both
                # Community level
                type1_results = calculate_connectivity_metrics(
                    filtered_processed, release_grids, settlement_grids, traits, traj, grid_ids=grid_ids
                )
                # Plus per-species
                type1_results['per_species'] = {}
                for sp, df in filtered_processed.items():
                    if not df.empty:
                        sp_data = {sp: df}
                        sp_release = {sp: release_grids.get(sp)} if sp in release_grids else {}
                        sp_settle = {
                            sp: settlement_grids.get(sp)} if settlement_grids and sp in settlement_grids else None
                        sp_traits = [t for t in traits if standardise_species_name(t.get('species_name', '')) == sp]
                        if sp_traits:
                            type1_results['per_species'][sp] = calculate_connectivity_metrics(
                                sp_data, sp_release, sp_settle, sp_traits, traj, grid_ids=grid_ids
                            )
            results['type1'] = type1_results
            print("  Connectivity metrics calculated")

        # Type 2: Network Analysis
        if config['analysis'].get('run_type2', True):
            subsection_header("Type 2: Network Analysis")

            scales = config['analysis'].get('type2_scales', [30])
            all_network_results = {}

            for scale in scales:
                print(f"\nProcessing scale: {scale} km")
                network_result = network_analysis(
                    filtered_processed, release_grids, settlement_grids,  # <-- Use filtered_processed
                    coastal_grid_df, grid_ids, coastal_grid_df, scale
                )

                all_network_results[f"{scale}km"] = network_result

            results['type2'] = all_network_results

            # Visualization
            if config['analysis'].get('type2_visualizations', 'skip') != 'skip':
                if config['output']['save_figures']:
                    primary_result = list(all_network_results.values())[0]

                    composite_map = {
                        'functional': ('functional_composite', 'top10_functional'),
                        'centrality': ('centrality_composite', 'top10_centrality'),
                        'overall': ('overall_composite', 'top10_overall')
                    }
                    ranking_col, top10_key = composite_map.get(
                        config['analysis'].get('type2_composite', 'overall'),
                        ('overall_composite', 'top10_overall')
                    )

                    output_fig_path = os.path.join(
                        config['output']['directory'],
                        f"network_map.{config['output']['format']}"
                    )
                    create_network_map(
                        primary_result['site_metrics'],
                        primary_result[top10_key],
                        title="Connectivity Hotspots",
                        composite_col=ranking_col,
                        output_path=output_fig_path,
                        output_format=config['output']['format'],
                        dpi=config['output']['dpi']
                    )
                    print(f"Saved: {output_fig_path}")

        # Type 3: Trait Sensitivity
        if config['analysis'].get('run_type3', False):
            subsection_header("Type 3: Trait Sensitivity Analysis")

            if config['analysis'].get('type3_option') == 'null_comparison':
                print("\n  Creating null model...")

                study_bounds = {
                    'lat_min': config['copernicus']['min_lat'],
                    'lat_max': config['copernicus']['max_lat'],
                    'lon_min': config['copernicus']['min_lon'],
                    'lon_max': config['copernicus']['max_lon']
                }

                null_data = create_null_model(
                    traj, coastal_grid_df, study_bounds,
                    coastal_grid_df, coastal_grid_df,
                    {'release_strategy': config['grids']['release_strategy']},
                    traits
                )

                if null_data:
                    null_network = network_analysis(
                        null_data['processed'],
                        null_data['release_grids'],
                        null_data['settlement_grids'],
                        coastal_grid_df, grid_ids, coastal_grid_df,
                        config['analysis'].get('type3_scale', 30)
                    )

                    trait_network = results.get('type2', {}).get(
                        f"{config['analysis'].get('type3_scale', 30)}km"
                    )

                    if trait_network:
                        comparison = compare_models(trait_network, null_network)
                        results['type3'] = {
                            'comparison': comparison,
                            'trait_network': trait_network,
                            'null_network': null_network
                        }
                        print("Model comparison complete")

        # Workflow Complete
        section_header("Automatic Workflow Complete")

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