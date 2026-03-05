# region Interactive Functions
"""
ComCon Interactive Mode
=======================

This module provides the interactive workflow for the ComCon package.
Users are guided through each step with prompts, explanations, and
the ability to review/edit parameters before proceeding.

Usage:
    import comcon
    comcon.choose_mode()  # Select Interactive Mode

    # Then run steps sequentially:
    from comcon.interactive import workflow_step
    workflow_step(1)  # Hydrodynamic data setup
    workflow_step(2)  # Species occurrence data
    workflow_step(3)  # Release/settlement grids
    workflow_step(4)  # Particle simulation
    workflow_step(5)  # Post-processing
    workflow_step(6)  # Connectivity analysis

How It Works:
-------------
1. Each workflow_step() guides users through one phase of the analysis
2. Parameters are collected via prompts with validation
3. Users can review and edit settings before execution
4. Results are displayed and can be saved at each step
5. Variables persist between steps for the complete workflow

Workflow Steps:
---------------
Step 1: Load species traits, set dispersal period, download hydrodynamic data
Step 2: Define study area, download/load species occurrences, filter by habitat
Step 3: Create release/settlement grids, build OceanParcels fieldset
Step 4: Configure particles, execute Lagrangian simulation
Step 5: Apply spawning, PLD, and habitat filters to trajectories
Step 6: Run connectivity metrics, network analysis, trait sensitivity tests
"""
# region Import Modules
import os
import json
import gc
import math
import csv
import time
import warnings
import traceback
import getpass
import copy
import itertools
from datetime import timedelta, datetime
from operator import attrgetter
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib import colormaps
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import builtins
import seaborn as sns
import importlib.resources as pkg_resources
import scipy.interpolate as interpolate
from scipy.ndimage import binary_dilation
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from scipy import stats
from netCDF4 import Dataset
from glob import glob
import geopandas as gpd
from shapely.geometry import Point, Polygon
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import folium
from folium import plugins
import cmocean
import trajan as ta
import copernicusmarine as cm
from pygbif import occurrences
import pyworms
from tqdm import tqdm
import ipywidgets as widgets
from IPython.display import HTML, Image, display, clear_output
import psutil
import networkx as nx
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from sklearn.utils import check_random_state
import pickle
from itertools import combinations
from parcels import (
    AdvectionRK4, FieldSet, JITParticle, ParticleSet, Variable,
    download_example_dataset, Field, ParcelsRandom, VectorField,
    DiffusionUniformKh, Geographic, GeographicPolar, StatusCode

)

warnings.filterwarnings('ignore')
import logging
logging.getLogger("urllib3").setLevel(logging.ERROR)
# endregion

# region Standard formatting
# region Formatting Configuration
width = 92  # Fallback for non-HTML/terminal contexts

def display_line(char="═"):
    """
    Display a full-width line using the specified character.
    Works in both Jupyter (HTML) and terminal environments.
    """
    try:
        from IPython.display import display, HTML
        display(HTML(
            f'<pre style="font-family: monospace; margin: 0; overflow: hidden; white-space: pre; width: 100%;">{char * 300}</pre>'))
    except:
        print(char * width)

def centered_text(text):
    """
    Display centered text that adapts to container width.
    Use for titles and headers.
    """
    try:
        from IPython.display import display, HTML
        display(HTML(f'<pre style="font-family: monospace; margin: 0; text-align: center;">{text}</pre>'))
    except:
        print(text.center(width))

def section_header(title):
    """
    Level 1: Major section headers.
    Use for: Workflow step titles, Mode titles, Start/End messages.

    Output:
    ════════════════════════════════════════════════════════════════════════════════════════════════
      Step 1: Hydrodynamic and Trait Data
    ════════════════════════════════════════════════════════════════════════════════════════════════
    """
    print("\n")
    display_line("═")
    try:
        from IPython.display import display, HTML
        display(HTML(
            f'<pre style="font-family: monospace; margin: 0; text-align: center; font-weight: bold;">{title}</pre>'))
    except:
        print(f"  {title}")
    display_line("═")
    print()


def subsection_header(title):
    """
    LEVEL 2: Subsection headers within a step.
    Use for: "Data Source", "Configuration", "Validation", etc.

    Output:
    ────────────────────────────────────────────────────────────────────────────────────────────────
      Data Source
    ────────────────────────────────────────────────────────────────────────────────────────────────
    """
    print("\n")
    display_line("─")
    print(f"{title}")
    display_line("─")

def info_box(text_lines):
    """
    LEVEL 3: Information/explanation boxes.
    Use for: Instructions, descriptions, option lists.
    Pass a list of strings.

    Output:
      This package uses hydrodynamic data from the Copernicus Marine Service.
      Use the link below to set up a free account.
    ────────────────────────────────────────────────────────────────────────────────────────────────
    """
    print()
    for line in text_lines:
        print(f"  {line}")
    print()
    display_line("─")

def data_box(title, content_lines):
    """
    LEVEL 4: Data display boxes for results, summaries, tables.
    Use for: Species summaries, coordinate ranges, validation results.

    Output:
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Species Summary
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Echiichthys vipera: 121 records
        Fucus serratus: 456 records
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    """
    print()
    print("- " * 46)
    print(f"  {title}")
    print("- " * 46)
    for line in content_lines:
        print(f"    {line}")
    print("- " * 46)

def code_box(title, code_lines):
    """
    For displaying code examples (manual mode).

    Output:
    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │  Example Title                                                                               │
    ├──────────────────────────────────────────────────────────────────────────────────────────────┤
    │  code_line_1()                                                                               │
    │  code_line_2()                                                                               │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘
    """
    print()
    print("┌" + "─" * 90 + "┐")
    print(f"│  {title:<88}│")
    print("├" + "─" * 90 + "┤")
    for line in code_lines:
        # Truncate long lines
        if len(line) > 86:
            line = line[:83] + "..."
        print(f"│  {line:<88}│")
    print("└" + "─" * 90 + "┘")

def status_message(message, status="info"):
    """
    Status messages for progress updates.

    status options: "info", "success", "warning", "error"

    Output:
      → Processing data...
      ✓ Data loaded successfully
      ⚠ Warning: Some records outside study area
      ✗ Error: File not found
    """
    symbols = {
        "info": "→",
        "success": "✓",
        "warning": "⚠",
        "error": "✗"
    }
    symbol = symbols.get(status, "→")
    print(f"  {symbol} {message}")

def completion_message(step_number=None, next_step=True):
    """
    Standard completion message for end of steps or workflow.

    Parameters:
        step_number: int or None - if None, shows "Workflow completed"
        next_step: bool - whether to show "To continue..." message
    """
    print("\n")
    display_line("─")
    if step_number:
        print(f"  ✓ Step {step_number} completed.")
    else:
        print("  ✓ Workflow completed!")
    display_line("─")
    if step_number and next_step and step_number < 6:
        print(f"\n  To continue, run: workflow_step({step_number + 1})\n")

def workflow_progress_header(step_number, workflow_status):
    """
    Display the workflow progress header with dynamic centering.

    Parameters:
        step_number: Current step (1-6)
        workflow_status: dict mapping step numbers to status strings
    """
    symbols = {
        'complete': '✓',
        'in_progress': '→',
        'not_started': ' '
    }

    progress = ' '.join([f"[{symbols[workflow_status[i]]}]" for i in range(1, 7)])

    print("\n")
    display_line("═")
    centered_text("Interactive Mode")
    centered_text(f"Workflow Progress: {progress}")
    centered_text(f"Current Step: {step_number}/6")
    display_line("═")
    print()

def step_box(step_number, title):
    """Display a step header box that fills the width dynamically."""
    try:
        display(HTML(f'''
            <div style="font-family: monospace; border: 2px solid white; padding: 8px 12px; margin: 10px 0; width: 100%; box-sizing: border-box;">
                Step {step_number}: {title}
            </div>
        '''))
    except:
        print("┏" + "━" * (width - 2) + "┓")
        print(f"┃  Step {step_number}: {title}")
        print("┗" + "━" * (width - 2) + "┛\n")

## For figures
map_config = {
    'title': {
        'fontsize': 18,
        'fontweight': 'bold',
        'pad': 5
    },
    'axis_label': {
        'fontsize': 15
    },
    'tick_label': {
        'fontsize': 15
    },
    'legend': {
        'fontsize': 15,
        'framealpha': 0.9,
        'loc': 'upper right'
    },
    'colorbar': {
        'label_fontsize': 15,
        'tick_fontsize': 15,
        'shrink': 0.6,
        'pad': 0.08
    },
    'annotation': {
        'fontsize': 11,
        'fontweight': 'bold'
    },
    'gridlines': {
        'alpha': 0.3,
        'label_fontsize': 10
    }
}
# endregion

# region Helper Functions
## Helps et up the folder structure
def get_default_dir(subfolder_key, fallback="./outputs"):
    """
    Get the appropriate default directory based on project setup.

    Parameters
    ----------
    subfolder_key : str
        Key from project_dirs dict: 'figures', 'results', 'data',
        'config', 'occurrence', 'grids', 'simulation'
    fallback : str
        Fallback path if no project directory is set up

    Returns
    -------
    str
        The default directory path
    """
    global project_dirs
    try:
        if project_dirs is not None and subfolder_key in project_dirs:
            return project_dirs[subfolder_key]
    except NameError:
        pass
    return fallback

## Check that a positive integer has been entered
def get_positive_integer(prompt, default=1, min_val=0, max_val=None):
    """
    Prompt user for a positive integer with validation and bounds checking.

    Repeatedly prompts the user until a valid integer within the specified range
    is entered. Provides helpful error messages for invalid inputs and supports
    default values.

    Parameters
    ----------
    prompt : str
        The text prompt to display to the user
    default : int, optional
        Default value if user presses Enter without input (default: 1)
    min_val : int, optional
        Minimum acceptable value (inclusive) (default: 0)
    max_val : int or None, optional
        Maximum acceptable value (inclusive). If None, no upper limit is enforced
        (default: None)

    Returns
    -------
    int
        Valid integer within the specified range

    Notes
    -----
    - Input validation is performed in a loop until valid input is received
    - User can press Enter to accept the default value
    - Non-integer inputs and out-of-range values trigger error messages
    """
    while True:
        try:
            # Don't show "(default X)" if prompt already mentions "current value"
            if "current value" in prompt.lower():
                user_input = input(f"{prompt}: ").strip()
            else:
                user_input = input(f"{prompt} [default: {default}]: ").strip()

            value = int(user_input) if user_input else default

            if value < min_val:
                print(f"Error: Value must be ≥ {min_val}. Try again.")
                continue
            if max_val is not None and value > max_val:
                print(f"Error: Value must be ≤ {max_val}. Try again.")
                continue

            return value
        except ValueError:
            print("Error: Enter a valid integer.")

# Allows users to validate their file path inputs incase of typos
def get_valid_path(prompt, must_exist=True, default=None, extensions=None):
    """Get a validated file path from user input."""
    while True:
        if default:
            user_input = input(f"{prompt} [default: {default}]: ").strip() or default
        else:
            user_input = input(f"{prompt}: ").strip()

        if not user_input:
            print("  Please enter a path.")
            continue

        # Expand user home directory
        user_input = os.path.expanduser(user_input)

        if must_exist and not os.path.exists(user_input):
            print(f"  Path not found: {user_input}")
            print(f"  Please check the path and try again.")
            continue

        if extensions:
            ext = os.path.splitext(user_input)[1].lower()
            if ext and ext not in extensions:
                print(f"  Expected file type: {', '.join(extensions)}, got: {ext}")
                retry = input("  Continue anyway? (yes/no): ").strip().lower()
                if retry not in ['yes', 'y']:
                    continue

        return user_input

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

def save_session(filepath=None):
    """
    Save the current workflow session so you can resume later.

    Call this after completing any workflow step to create a checkpoint.
    To resume later, use load_session() and then workflow_step(N).

    Parameters
    ----------
    filepath : str, optional
        Path to save the session file. If not provided, saves to the project
        directory as 'comcon_session.pkl', or to the current directory if no
        project directory has been set up.
    """
    if filepath is None:
        if 'project_dirs' in globals() and globals()['project_dirs']:
            filepath = os.path.join(globals()['project_dirs'].get('config', '.'), 'comcon_session.pkl')
        else:
            filepath = 'comcon_session.pkl'

    session_data = {}

    variables_to_save = [
        'project_dirs', 'runtime', 'species_traits_path', 'traits', 'hydro_params',
        'buffer_before', 'buffer_after',
        'clean_spp', 'spp_params', 'landmask', 'coastalmask', 'lonarray', 'latarray',
        'release_grids', 'settlement_grids', 'full_grid_df', 'grid_ids',
        'coastal_grid', 'coastal_grid_df', 'coastal_grid_filtered',
        'ocean_grid', 'ocean_grid_df', 'grid_df_for_mapping', 'meshgrid',
        'adjusted_runtime', 'interval', 'output_path', 'traj',
        'final_processed', 'buffer', 'geojson_file_path'
    ]

    for var_name in variables_to_save:
        if var_name in globals() and globals()[var_name] is not None:
            try:
                session_data[var_name] = globals()[var_name]
            except Exception:
                pass

    from datetime import datetime as _dt
    session_data['_saved_at'] = _dt.now().isoformat()

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(session_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    display_line("─")
    print(f"  Session saved to: {filepath}")
    print(f"  Variables saved: {len(session_data) - 1}")
    print(f"  File size: {os.path.getsize(filepath) / (1024*1024):.1f} MB")
    display_line("─")
    print(f"\n  To resume later, run in a new notebook or after restarting the kernel:")
    print(f"    from comcon.interactive import *")
    print(f"    load_session('{filepath}')")
    print(f"    workflow_step(N)  # where N is the step you want to resume from")

# Allows the users to save their session including all data loaded into the environment to easily stop/start the workflow
def load_session(filepath='comcon_session.pkl'):
    """
    Load a previously saved workflow session.

    Restores all variables so you can resume from any workflow step without re-running previous steps.

    Parameters
    ----------
    filepath : str
        Path to the saved session file (default: 'comcon_session.pkl')
    """
    if not os.path.exists(filepath):
        print(f"  Error: Session file not found: {filepath}")
        return

    with open(filepath, 'rb') as f:
        session_data = pickle.load(f)

    saved_at = session_data.pop('_saved_at', 'Unknown')

    restored = []
    for var_name, value in session_data.items():
        globals()[var_name] = value
        restored.append(var_name)

    display_line("─")
    print(f"  Session restored from: {filepath}")
    print(f"  Originally saved at: {saved_at}")
    print(f"  Variables restored: {len(restored)}")
    display_line("─")

    can_resume = []
    if 'traits' in restored:
        can_resume.append("Step 2 (Species Occurrence Data)")
    if 'clean_spp' in restored:
        can_resume.append("Step 3 (Simulation Setup)")
    if 'grid_ids' in restored:
        can_resume.append("Step 4 (Simulation)")
    if 'traj' in restored:
        can_resume.append("Step 5 (Post-Processing)")
    if 'final_processed' in restored:
        can_resume.append("Step 6 (Analysis)")

    if can_resume:
        print("\n  You can resume from:")
        for step in can_resume:
            print(f"    - {step}")
        print(f"\n  Run: workflow_step(N) where N is the step number.")

## Species name standardisation
def standardise_species_name(name):
    """Standardise species names to use underscores consistently."""
    if name is None:
        return None
    return str(name).replace(' ', '_')

## Cleaned species name for display
def display_species_name(name):
    """Format species name for display (replace underscores with spaces)."""
    if name is None:
        return None
    return str(name).replace('_', ' ')

## Exporting visualisations in user's chosen format
def get_output_format():
    """
    Prompt user to select output format for visualisations.

    Returns
    -------
    tuple
        (format_extension, dpi) - e.g., ('png', 300), ('pdf', 300)
    """
    print("\nSelect output format:")
    print("  1. tif")
    print("  2. PDF")
    print("  3. PNG")
    print("  4. JPEG")
    print("  5. SVG")

    format_choice = input("\nSelect format (1-5) [default: 1]: ").strip() or '1'

    format_map = {
        '1': ('png', 300),
        '2': ('pdf', 300),
        '3': ('tif', 300),
        '4': ('jpeg', 300),
        '5': ('svg', 300)
    }

    return format_map.get(format_choice, ('png', 300))

## Saving figures helper function
def save_figure(fig, base_path, format_ext='png', dpi=300):
    """
    Save figure in specified format with consistent settings.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    base_path : str
        Output path without extension (e.g., './outputs/network_map')
    format_ext : str
        File extension/format ('png', 'pdf', 'tif', 'jpeg', 'svg')
    dpi : int
        Resolution in dots per inch (default 300)

    Returns
    -------
    str
        Full path to saved file
    """
    # Remove any existing extension from base_path
    base_path = os.path.splitext(base_path)[0]

    # Construct full path
    full_path = f"{base_path}.{format_ext}"

    # Save with appropriate settings
    if format_ext in ['pdf', 'svg']:
        # Vector formats don't need dpi but include for consistency
        fig.savefig(full_path, format=format_ext, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    else:
        fig.savefig(full_path, dpi=dpi, format=format_ext, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

    print(f"Figure saved: {full_path} ({format_ext.upper()}, {dpi} dpi)")
    return full_path

## Asking user what outputs to save
def get_output_choice():
    """
    Prompt user for output preference: display only, save, or both.

    Returns
    -------
    tuple
        (display_fig, save_fig, output_dir, output_format, dpi)
        - display_fig: bool, whether to display figure
        - save_fig: bool, whether to save figure
        - output_dir: str or None, directory for saving
        - output_format: str, file format
        - dpi: int, resolution
    """
    print("\nOutput options:")
    print("  1. Display only (don't save)")
    print("  2. Save only (don't display)")
    print("  3. Both display and save")

    choice = input("Select option (1-3) [default: 3]: ").strip() or '3'

    display_fig = choice in ['1', '3']
    save_fig = choice in ['2', '3']

    output_dir = None
    output_format = 'png'
    dpi = 300

    if save_fig:
        default_dir = get_default_dir('figures')
        output_dir = input(f"Output directory [default: {default_dir}]: ").strip() or default_dir
        os.makedirs(output_dir, exist_ok=True)
        output_format, dpi = get_output_format()

    return display_fig, save_fig, output_dir, output_format, dpi

## Dynamic sizing for figures
def calculate_figure_dimensions(lon_range, lat_range, base_size=10,
                                min_dim=6, max_dim=16, n_panels=1,
                                layout='horizontal'):
    """
    Calculate optimal figure dimensions based on spatial extent.

    Handles different study area shapes (tall/narrow vs wide/short)
    and multi-panel layouts intelligently.

    Parameters
    ----------
    lon_range : float
        Longitude range of study area (degrees).
    lat_range : float
        Latitude range of study area (degrees).
    base_size : float
        Base figure dimension in inches (default 10).
    min_dim : float
        Minimum dimension in inches (default 6).
    max_dim : float
        Maximum dimension in inches (default 16).
    n_panels : int
        Number of map panels (default 1).
    layout : str
        'horizontal' for side-by-side, 'vertical' for stacked,
        'auto' to choose based on aspect ratio.

    Returns
    -------
    tuple
        (fig_width, fig_height, actual_layout)
    """
    # Calculate aspect ratio (width/height)
    aspect = lon_range / lat_range if lat_range > 0 else 1

    # Auto-select layout based on aspect ratio
    if layout == 'auto':
        if aspect > 1.5:  # Wide study area
            actual_layout = 'vertical'  # Stack maps vertically
        elif aspect < 0.67:  # Tall study area
            actual_layout = 'horizontal'  # Place maps side by side
        else:
            actual_layout = 'vertical'  # Default to vertical
    else:
        actual_layout = layout

    # Calculate base dimensions
    if aspect >= 1:
        # Wider than tall
        fig_width = base_size
        fig_height = base_size / aspect
    else:
        # Taller than wide
        fig_height = base_size
        fig_width = base_size * aspect

    # Apply constraints
    fig_width = max(min_dim, min(max_dim, fig_width))
    fig_height = max(min_dim, min(max_dim, fig_height))

    # Adjust for multi-panel
    if n_panels > 1:
        if actual_layout == 'horizontal':
            fig_width = fig_width * n_panels + (n_panels - 1) * 0.5
        else:
            fig_height = fig_height * n_panels + (n_panels - 1) * 0.5

    # Final constraints
    fig_width = min(max_dim * 1.5, fig_width)
    fig_height = min(max_dim * 1.5, fig_height)

    return fig_width, fig_height, actual_layout
# endregion

# region Processing the Species Trait Database
## Accesses the trait data
def get_package_data_path(filename):
    """Get path to data files included with the package."""
    package_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(package_dir, 'data', filename)

## Load in builtin trait data
def load_builtin_traits():
    """Load the builtin trait database."""
    try:
        trait_path = get_package_data_path('trait_database.csv')
        if os.path.exists(trait_path):
            return pd.read_csv(trait_path)
        else:
            print(f"Builtin trait database not found at {trait_path}")
            return None
    except Exception as e:
        print(f"Error loading trait database: {e}")
        return None

## Allow users to select the species they want to add
def select_species_from_database(traits_df):
    """
    Interactive species selection from trait database.

    Parameters
    ----------
    traits_df : DataFrame
        Full trait database

    Returns
    -------
    DataFrame
        Selected species subset
    """
    # Build species list for display (show PLD and spawning without year)
    species_lines = []
    for i, row in traits_df.iterrows():
        spp_name = row['species_name']
        min_pld = row.get('min_PLD', 'N/A')
        max_pld = row.get('max_PLD', 'N/A')
        spawn_start_raw = str(row.get('spawning_start', 'N/A'))
        spawn_end_raw = str(row.get('spawning_end', 'N/A'))
        spawn_start = '/'.join(spawn_start_raw.split('/')[:2]) if '/' in spawn_start_raw else spawn_start_raw
        spawn_end = '/'.join(spawn_end_raw.split('/')[:2]) if '/' in spawn_end_raw else spawn_end_raw
        species_lines.append(
            f"{i + 1:2}. {spp_name:<30} PLD: {min_pld}-{max_pld} days, Spawning: {spawn_start} to {spawn_end}"
        )
    data_box("Species in Database", species_lines)

    print("\nOptions:")
    print("  1. Select specific species by number")
    print("  2. Use all species in database")

    choice = input("\nSelect option (1/2) [default: 1]: ").strip() or '1'

    if choice == '2':
        return traits_df
    else:
        indices_str = input("Enter species numbers (comma-separated, e.g., 1,3,5): ").strip()
        try:
            indices = [int(x.strip()) - 1 for x in indices_str.split(',')]
            selected = traits_df.iloc[indices].reset_index(drop=True)
            print(f"\n  Selected {len(selected)} species:")
            for _, row in selected.iterrows():
                print(f"    - {row['species_name']}")
            confirm = input("\n  Confirm selection? (yes/no) [default: yes]: ").strip().lower()
            if confirm in ['no', 'n']:
                return select_species_from_database(traits_df)  # Re-run selection
            return selected
        except:
            print("Invalid input. Using all species.")
            return traits_df

## Allow users to set the dispersal year
def apply_dispersal_year(traits_list, start_year):
    """
    Apply dispersal season year to trait dates.

    Handles both DD/MM (text) and DD/MM/YYYY formats.
    For DD/MM: adds the year
    For DD/MM/YYYY: replaces the existing year

    Parameters
    ----------
    traits_list : list of dict
        Trait data with spawning dates
    start_year : int
        Year when dispersal season starts

    Returns
    -------
    list of dict
        Traits with updated dates
    """
    updated_traits = []

    for t in traits_list:
        new_t = t.copy()

        start_parts = str(t['spawning_start']).split('/')
        end_parts = str(t['spawning_end']).split('/')

        # Extract day and month (works for both DD/MM and DD/MM/YYYY)
        start_day = start_parts[0]
        start_month = int(start_parts[1])
        end_day = end_parts[0]
        end_month = int(end_parts[1])

        # Determine year for each date
        start_year_use = start_year
        if end_month < start_month:
            # Spawning crosses year boundary (e.g., Oct-Mar)
            end_year_use = start_year + 1
        else:
            end_year_use = start_year

        # Create full dates
        new_t['spawning_start'] = f"{start_day}/{start_month:02d}/{start_year_use}"
        new_t['spawning_end'] = f"{end_day}/{end_month:02d}/{end_year_use}"

        updated_traits.append(new_t)

    return updated_traits

## Calculate the overall runtime
def calculate_overall_dispersal_period(traits, buffer_before=0, buffer_after=0):
    """
    Calculates the overall dispersal period across all species.

    Finds the earliest spawning start date and the latest dispersal end date
    (spawning end + max PLD) across all species to determine the required
    simulation runtime. Optional buffer days allow trait sensitivity analysis.

    Parameters:
    -----------
    traits : list of dict
        List of dictionaries containing species trait data. Each dictionary
        must have keys: 'spawning_start', 'spawning_end', 'max_PLD'
    buffer_before : int, optional
        Number of days to add BEFORE earliest spawning start (default: 0)
        Allows testing earlier spawning scenarios in trait sensitivity analysis
    buffer_after : int, optional
        Number of days to add AFTER latest dispersal end (default: 0)
        Allows testing extended PLD scenarios in trait sensitivity analysis

    Returns:
    --------
    buffered_start : str
        Simulation start date with buffer in 'YYYY-MM-DD' format
    buffered_end : str
        Simulation end date with buffer in 'YYYY-MM-DD' format
    base_start : str
        Biological start date without buffer (for reference/analysis)
    base_end : str
        Biological end date without buffer (for reference/analysis)

    Notes:
    ------
    - Spawning dates in traits must be in 'DD/MM/YYYY' format
    - Returns dates in 'YYYY-MM-DD' format for consistency with xarray
    - Accounts for maximum PLD to ensure propagules can complete development
    - Buffer allows future trait sensitivity analysis without re-running simulation
    """
    overall_start = None
    overall_end = None

    # Loop through all species
    for species in traits:
        # Parse spawning dates
        spawning_start_date = datetime.strptime(species['spawning_start'], "%d/%m/%Y")
        spawning_end_date = datetime.strptime(species['spawning_end'], "%d/%m/%Y")
        max_pld = int(species['max_PLD'])

        # Calculate the dispersal end date by adding max PLD to the spawning end date
        dispersal_end_date = spawning_end_date + timedelta(days=max_pld)

        # Update overall start and end dates
        if overall_start is None or spawning_start_date < overall_start:
            overall_start = spawning_start_date
        if overall_end is None or dispersal_end_date > overall_end:
            overall_end = dispersal_end_date

    # Save base (biological) period
    base_start = overall_start
    base_end = overall_end
    base_days = (base_end - base_start).days

    # Apply buffer to create simulation period
    buffered_start = overall_start - timedelta(days=buffer_before)
    buffered_end = overall_end + timedelta(days=buffer_after)
    total_days = (buffered_end - buffered_start).days

    if buffer_before > 0 or buffer_after > 0:
        buffer_days = buffer_before + buffer_after
        print(f"\nTemporal buffer: +{buffer_before} days before, +{buffer_after} days after")
        print(f"Simulation period from: {buffered_start.strftime('%Y-%m-%d')} to {buffered_end.strftime('%Y-%m-%d')}")
        print(f"Base runtime: {base_days} days")
        print(f"With the added buffer of {buffer_days} days, the runtime is extended to {total_days} days")

    else:
        print(f"Simulation period: {total_days}")

    return (buffered_start.strftime('%Y-%m-%d'),
            buffered_end.strftime('%Y-%m-%d'),
            base_start.strftime('%Y-%m-%d'),
            base_end.strftime('%Y-%m-%d'))

## Load species traits from the CSV file or built-in database
def load_spp_traits():
    """
    Load and process species trait data from built-in database or user CSV.

    Prompts the user to either:
    1. Select species from the built-in trait database
    2. Provide their own CSV file with species trait data

    Required CSV columns (if providing own file):
    - species_name: Name of the species (optional if aphia_id provided)
    - aphia_id: WoRMS AphiaID for the species (optional if species_name provided)
    - spawning_start: Start of spawning period (DD/MM or DD/MM/YYYY)
    - spawning_end: End of spawning period (DD/MM or DD/MM/YYYY)
    - min_PLD: Minimum planktonic larval duration in days
    - max_PLD: Maximum planktonic larval duration in days

    Returns:
    --------
    dispersal_period : dict
        Dictionary containing simulation timing information
    species_traits_path : str
        Path to the traits CSV file (created or provided)
    traits : list
        List of dictionaries, one per species, containing trait data
    """
    subsection_header("Species Traits Data")

    print("\nHow would you like to provide species traits data?")
    print("  1. Select species from builtin trait database")
    print("  2. Provide your own CSV file")

    source_choice = input("\nSelect option (1-2) [default: 1]: ").strip() or '1'

    traits = []
    species_traits_path = None

    # Option 1: Use built-in trait database
    if source_choice == '1':
        subsection_header("Builtin Trait Database")

        # Load the database
        traits_df = load_builtin_traits()

        if traits_df is None or traits_df.empty:
            print("\nError: Could not load builtin trait database.")
            print("Please provide your own CSV file instead.")
            source_choice = '2'  # Fall through to option 2
        else:
            # Let user select species
            selected_df = select_species_from_database(traits_df)

            print(f"\nSelected {len(selected_df)} species")

            # Convert DataFrame to list of dicts
            for _, row in selected_df.iterrows():
                traits.append({
                    "species_name": row.get('species_name'),
                    "aphia_id": row.get('aphia_id') if pd.notna(row.get('aphia_id')) else None,
                    "spawning_start": str(row['spawning_start']),
                    "spawning_end": str(row['spawning_end']),
                    "min_PLD": int(row['min_PLD']),
                    "max_PLD": int(row['max_PLD'])
                })

            # Prompt for year (database uses DD/MM format)
            subsection_header("Dispersal Year")
            info_box([
                "This package simulates dispersal for one reproductive generation per species.",
                "The trait database uses DD/MM format (no year).",
                "Enter the year when your dispersal season starts.",
                "Example: If earliest spawning is December 2023, enter 2023"
            ])

            start_year = int(input("\nStart year for dispersal season: ").strip())

            # Apply year to all traits
            for t in traits:
                start_parts = str(t['spawning_start']).split('/')
                end_parts = str(t['spawning_end']).split('/')

                start_day = start_parts[0]
                start_month = int(start_parts[1])
                end_day = end_parts[0]
                end_month = int(end_parts[1])

                # Handle year boundary crossing (e.g., Oct-Mar spans two years)
                start_year_use = start_year
                end_year_use = start_year + 1 if end_month < start_month else start_year

                t['spawning_start'] = f"{start_day}/{start_month:02d}/{start_year_use}"
                t['spawning_end'] = f"{end_day}/{end_month:02d}/{end_year_use}"

            print(f"Dates set for {start_year}/{start_year + 1} dispersal season")

            # Always save selected traits to CSV for reuse
            default_dir = get_default_dir('data')
            output_dir = input(f"Output directory [default: {default_dir}]: ").strip() or default_dir
            os.makedirs(output_dir, exist_ok=True)
            filename = input("Filename [default: selected_traits]: ").strip() or "selected_traits"
            if not filename.endswith('.csv'):
                filename += '.csv'

            species_traits_path = os.path.join(output_dir, filename)

            # Create DataFrame and save
            save_df = pd.DataFrame(traits)
            save_df.to_csv(species_traits_path, index=False)
            print(f"Traits saved to: {species_traits_path}")

    # Option 2: User provides own CSV file
    if source_choice == '2':
        subsection_header("Custom Traits CSV")
        info_box([
            "Your CSV must include either 'species_name' or 'aphia_id' for each species:",
            "  - Leave the unused column blank (empty cell)",
            "  - If both are provided, AphiaID will be used",
            "  - AphiaID is recommended for marine species (handles name changes/synonyms)",
            "  - Find AphiaIDs at: https://www.marinespecies.org/",
            "",
            "Required columns: species_name OR aphia_id, spawning_start, spawning_end, min_PLD, max_PLD",
            "Date format: DD/MM/YYYY or DD/MM (you'll be prompted for year)"
        ])

        # Get file path
        info_box([
            "Your CSV file must contain the following columns:",
            "",
            "  Required columns:",
            "    species_name    — Species name (e.g., Patella vulgata)",
            "    min_PLD         — Minimum pelagic larval duration in days (integer)",
            "    max_PLD         — Maximum pelagic larval duration in days (integer)",
            "    spawning_start  — Start of spawning window (format: DD/MM, e.g., 01/10)",
            "    spawning_end    — End of spawning window (format: DD/MM, e.g., 31/01)",
            "",
            "  Optional columns:",
            "    aphia_id        — WoRMS AphiaID for the species",
            "    lat, lon        — Coordinates (if providing occurrence data in the same file)",
            "",
            "  Additional columns are permitted and will be ignored by ComCon."
        ])
        species_traits_path = get_valid_path("Enter the path to your species traits CSV", must_exist=True, extensions=['.csv'])
        if not species_traits_path.endswith(".csv"):
            species_traits_path += ".csv"

        if not os.path.exists(species_traits_path):
            print(f"Error: File not found: {species_traits_path}")
            return None, None, None

        # Read CSV file to validate columns
        try:
            traits_df = pd.read_csv(species_traits_path)
        except Exception as e:
            print(f"Error reading file: {e}")
            return None, None, None

        # Validate required columns
        required_cols = ['min_PLD', 'max_PLD', 'spawning_start', 'spawning_end']
        missing_cols = [col for col in required_cols if col not in traits_df.columns]

        if missing_cols:
            print(f"\nError: Missing required columns:")
            for col in missing_cols:
                print(f"  - {col}")
            print(f"\nYour CSV has: {', '.join(traits_df.columns.tolist())}")
            return None, None, None

        # Check identifier columns
        has_species_name = 'species_name' in traits_df.columns
        has_aphia_id = 'aphia_id' in traits_df.columns

        if not has_species_name and not has_aphia_id:
            print(f"\nError: Your CSV must have 'species_name' or 'aphia_id' column")
            return None, None, None

        # Parse CSV into traits list
        with open(species_traits_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                species_name = row.get('species_name', '').strip()
                aphia_id_raw = row.get('aphia_id', '').strip()

                # Handle NA values
                if species_name.upper() in ['NA', 'N/A', 'NONE', ''] or species_name == '':
                    species_name = None

                if aphia_id_raw.upper() in ['NA', 'N/A', 'NONE', ''] or aphia_id_raw == '':
                    aphia_id = None
                else:
                    try:
                        aphia_id = int(float(aphia_id_raw))
                    except ValueError:
                        raise ValueError(f"Invalid aphia_id: '{aphia_id_raw}'")

                traits.append({
                    "species_name": species_name,
                    "aphia_id": aphia_id,
                    "spawning_start": row['spawning_start'],
                    "spawning_end": row['spawning_end'],
                    "min_PLD": int(row['min_PLD']),
                    "max_PLD": int(row['max_PLD'])
                })

        # Check if dates need year applied
        sample_date = str(traits[0]['spawning_start'])
        date_parts = sample_date.split('/')

        if len(date_parts) == 2 or (len(date_parts) == 3 and date_parts[2] in ['2000', '00']):
            subsection_header("Dispersal Year")
            info_box([
                "Your trait dates need a year for the simulation.",
                "Enter the year when your dispersal season STARTS."
            ])

            start_year = int(input("\nStart year for dispersal season: ").strip())

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

            print(f"Dates set for {start_year}/{start_year + 1} dispersal season")

    # ═══════════════════════════════════════════════════════════════════
    # Common processing for both options
    # ═══════════════════════════════════════════════════════════════════

    # Display loaded traits
    subsection_header("Species Traits Data")
    traits_display = pd.DataFrame(traits)
    print(traits_display.to_string(index=False))
    status_message(f"Total species: {len(traits)}", "info")
    display_line("─")

    # Offer to edit individual species traits
    edit_traits = input("\nWould you like to edit any species' traits? (yes/no) [default: no]: ").strip().lower()
    if edit_traits in ['yes', 'y']:
        while True:
            try:
                print("\nSelected species:")
                for i, t in enumerate(traits):
                    sp_name = display_species_name(t.get('species_name', 'Unknown'))
                    print(f"  {i + 1}. {sp_name:<30} PLD: {t['min_PLD']}-{t['max_PLD']} days, "
                          f"Spawning: {t['spawning_start']} to {t['spawning_end']}")

                sp_num = input("\n  Enter species number to edit (or 'done' to finish): ").strip()
                if sp_num.lower() == 'done':
                    break
                sp_idx = int(sp_num) - 1
                if 0 <= sp_idx < len(traits):
                    t = traits[sp_idx]
                    sp_name = display_species_name(t.get('species_name', 'Unknown'))
                    print(f"\n  Editing: {sp_name}")
                    print(f"  Current values: PLD {t['min_PLD']}-{t['max_PLD']} days, "
                          f"Spawning {t['spawning_start']} to {t['spawning_end']}")

                    new_min = input(f"    New min_PLD [default: {t['min_PLD']}]: ").strip()
                    if new_min:
                        t['min_PLD'] = int(new_min)

                    new_max = input(f"    New max_PLD [default: {t['max_PLD']}]: ").strip()
                    if new_max:
                        t['max_PLD'] = int(new_max)

                    new_start = input(
                        f"    New spawning_start (DD/MM/YYYY) [default: {t['spawning_start']}]: ").strip()
                    if new_start:
                        t['spawning_start'] = new_start

                    new_end = input(f"    New spawning_end (DD/MM/YYYY) [default: {t['spawning_end']}]: ").strip()
                    if new_end:
                        t['spawning_end'] = new_end

                    print(f"  Updated: PLD {t['min_PLD']}-{t['max_PLD']} days, "
                          f"Spawning {t['spawning_start']} to {t['spawning_end']}")
                else:
                    print(f"  Invalid number. Enter 1 to {len(traits)}.")
            except ValueError:
                print("  Please enter a valid number or 'done'.")

        # Re-display updated traits
        subsection_header("Updated Species Traits Data")
        traits_display = pd.DataFrame(traits)
        print(traits_display.to_string(index=False))
        display_line("─")

    # Calculate base dispersal period
    base_start, base_end, _, _ = calculate_overall_dispersal_period(traits, 0, 0)

    base_runtime_days = (datetime.strptime(base_end, '%Y-%m-%d') -
                         datetime.strptime(base_start, '%Y-%m-%d')).days

    # Ask about temporal buffer
    subsection_header("Temporal Buffer for Trait Sensitivity Analysis")
    info_box([
        "Trait Sensitivity Analysis allows you to test alternative trait scenarios:",
        "  - What if species spawn earlier or later?",
        "  - What if PLDs are longer or shorter?",
        "",
        "To enable this, add extra days to your simulation:",
        "  - Buffer before: Allows testing earlier spawning",
        "  - Buffer after: Allows testing extended PLDs",
        "",
        f"Example: A 14-day buffer would increase runtime from {base_runtime_days} to {base_runtime_days + 28} days"
    ])

    add_buffer = input("\nAdd temporal buffer? (yes/no) [default: no]: ").strip().lower()

    if add_buffer in ['yes', 'y']:
        buffer_before = get_positive_integer("Days to add before earliest spawning", default=14, min_val=0)
        buffer_after = get_positive_integer("Days to add after latest dispersal end", default=14, min_val=0)
    else:
        buffer_before = 0
        buffer_after = 0
        print("No buffer added. Trait sensitivity analysis will not be possible.")

    # Calculate final dispersal period with buffer
    overall_start, overall_end, base_start_final, base_end_final = calculate_overall_dispersal_period(
        traits, buffer_before, buffer_after
    )

    return {
        "overall_dispersal_start": overall_start,
        "overall_dispersal_end": overall_end,
        "base_start": base_start_final,
        "base_end": base_end_final,
        "buffer_before": buffer_before,
        "buffer_after": buffer_after
    }, species_traits_path, traits

## Load species traits from path
def load_spp_traits_from_path(species_traits_path):
    """
    Load species traits from a specific file path without prompting.

    This is a non-interactive version of load_spp_traits() used when
    recalculating runtime from an existing traits file.

    Parameters
    ----------
    species_traits_path : str
        Path to the species traits CSV file

    Returns
    -------
    dispersal_period : dict or None
        Dictionary containing simulation timing information
    species_traits_path : str
        Path to the traits CSV file
    traits : list
        List of dictionaries, one per species, containing trait data
    """
    # Handle built-in database selection
    if species_traits_path == 'builtin_database_selection':
        traits_df = load_builtin_traits()
        if traits_df is None:
            print("Error: Could not load builtin trait database.")
            return None, species_traits_path, None
    elif not os.path.exists(species_traits_path):
        print(f"Error: File not found: {species_traits_path}")
        return None, species_traits_path, None
    else:
        traits_df = None  # Will be loaded below

    traits = []

    try:
        # Read CSV file (only if not already loaded from built-in)
        if traits_df is None:
            traits_df = pd.read_csv(species_traits_path)

        # Validate required columns
        required_cols = ['min_PLD', 'max_PLD', 'spawning_start', 'spawning_end']
        missing_cols = [col for col in required_cols if col not in traits_df.columns]

        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return None, species_traits_path, None

        # Parse CSV into traits list
        with open(species_traits_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                species_name = row.get('species_name', '').strip()
                aphia_id_raw = row.get('aphia_id', '').strip()

                # Handle NA values
                if species_name.upper() in ['NA', 'N/A', 'NONE', ''] or species_name == '':
                    species_name = None

                if aphia_id_raw.upper() in ['NA', 'N/A', 'NONE', ''] or aphia_id_raw == '':
                    aphia_id = None
                else:
                    try:
                        aphia_id = int(float(aphia_id_raw))
                    except ValueError:
                        aphia_id = None

                traits.append({
                    "species_name": species_name,
                    "aphia_id": aphia_id,
                    "spawning_start": row['spawning_start'],
                    "spawning_end": row['spawning_end'],
                    "min_PLD": int(row['min_PLD']),
                    "max_PLD": int(row['max_PLD'])
                })

        # Calculate dispersal period (no buffer for recalculation)
        overall_start, overall_end, base_start, base_end = calculate_overall_dispersal_period(
            traits, 0, 0
        )

        dispersal_period = {
            "overall_dispersal_start": overall_start,
            "overall_dispersal_end": overall_end,
            "base_start": base_start,
            "base_end": base_end,
            "buffer_before": 0,
            "buffer_after": 0
        }

        return dispersal_period, species_traits_path, traits

    except Exception as e:
        print(f"Error reading traits file: {e}")
        return None, species_traits_path, None
# endregion


# region Downloading the Hydrodynamic Data
## region Setting the Hydrodynamic Parameters
### Collecting Hydrodynamic Parameters
def collect_hydro_params():
    """
    Collects hydrodynamic data parameters from the user.

    Interactively prompts the user for all parameters needed to download
    hydrodynamic data from Copernicus Marine Service, including credentials,
    dataset ID, spatial extent and species trait data.

    Returns:
    --------
    params : dict
        Dictionary containing all hydrodynamic parameters:
        - dataset_id: Copernicus dataset identifier
        - longitude_range: [min_lon, max_lon]
        - latitude_range: [min_lat, max_lat]
        - time_range: [start_date, end_date] as datetime objects
        - username: Copernicus username
        - password: Copernicus password
        - output_path: Path to save downloaded data
        - export: Boolean, whether to export to NetCDF
        - traits: Species trait data
        - species_traits_path: Path to traits CSV
        - runtime: Total runtime in days

    Notes:
    ------
    - Requires active Copernicus Marine Service account
    - Time range is automatically calculated from species trait data
    - Runtime is calculated as the total dispersal period across all species
    """
    subsection_header("Setting up the Hydrodynamic Extent")
    info_box([
        "Important: The hydrodynamic domain needs to be larger than your study area."
    ])

    username = input("Enter your username: ").strip()
    password = getpass.getpass("Enter your password: ").strip()
    info_box([
        "Enter the Copernicus Marine dataset ID for your hydrodynamic data.",
        "",
        "Common datasets:",
        "  Atlantic-Iberian (~3km resolution; 2022 to current): cmems_mod_ibi_phy_anfc_0.027deg-2D_PT1H-m",
        "  NW European Shelf (~1.5km resolution; 2024 to current): cmems_mod_nws_phy-cur_anfc_1.5km-2D_PT1H-i",
        "  Global (~9km resolution; 2022 to current): cmems_mod_glo_phy_anfc_0.083deg_PT1H-m",
        "",
        "Enter 'browse' to open the Copernicus Marine Data Store in your browser."
    ])

    while True:
        dataset_id = input("Enter the dataset ID: ").strip()

        if dataset_id.lower() == 'browse':
            import webbrowser
            webbrowser.open("https://data.marine.copernicus.eu/products")
            print("  Opening Copernicus Marine Data Store in your browser...")
            print("  Find your dataset, copy its ID and paste it below.\n")
            continue

        if not dataset_id:
            print("Error: Dataset ID cannot be empty.")
            continue

        print(f"\n  Selected dataset: {dataset_id}")
        confirm = input("  Is this correct? (yes/no) [default: yes]: ").strip().lower() or 'yes'
        if confirm in ['yes', 'y']:
            break
        print("  Ok, please re-enter the dataset ID.\n")

    # Longitude range with validation
    subsection_header("Spatial Extent")
    info_box([
        "Define your hydrodynamic domain. This should be larger than your study area to allow particles to disperse beyond the boundaries.",
    ])

    while True:
        try:
            min_lon = float(input("Enter the minimum longitude: "))
            max_lon = float(input("Enter the maximum longitude: "))
            if min_lon >= max_lon:
                print("Error: Minimum longitude must be less than maximum longitude.")
                continue
            longitude_range = [min_lon, max_lon]
            break
        except ValueError:
            print("Error: Please enter valid numbers for longitude.")

            # Latitude range with validation
    print()
    while True:
        try:
            min_lat = float(input("Enter the minimum latitude: "))
            max_lat = float(input("Enter the maximum latitude: "))
            if min_lat >= max_lat:
                print("Error: Minimum latitude must be less than maximum latitude.")
                continue
            if not (-90 <= min_lat <= 90) or not (-90 <= max_lat <= 90):
                print("Error: Latitude must be between -90 and 90.")
                continue
            latitude_range = [min_lat, max_lat]
            break
        except ValueError:
            print("Error: Please enter valid numbers for latitude.")

    print(f"\n  Spatial extent: {min_lon}°W to {max_lon}°E, {min_lat}°N to {max_lat}°N")

    # Load species traits from the CSV file and calculate the overall dispersal period
    dispersal_time, species_traits_path, traits = load_spp_traits()
    if not dispersal_time:
        print("Error loading species traits. Aborting operation.")
        return None

    # Use the overall dispersal period from the species traits as the default time range
    # Add buffer for:
    #   - Release interval adjustment (up to 30 days for monthly releases)
    #   - Interpolation safety margin (1 day)
    hydro_safety_buffer = 31

    time_range = [
        datetime.strptime(dispersal_time["overall_dispersal_start"], '%Y-%m-%d'),
        datetime.strptime(dispersal_time["overall_dispersal_end"], '%Y-%m-%d') + timedelta(days=hydro_safety_buffer)
    ]

    # Runtime is the biological period (user's temporal buffer is already included in dispersal_time)
    # We don't subtract hydro_safety_buffer from runtime - it's just extra hydro data coverage
    runtime = (datetime.strptime(dispersal_time["overall_dispersal_end"], '%Y-%m-%d') -
               datetime.strptime(dispersal_time["overall_dispersal_start"], '%Y-%m-%d')).days

    print(f"Runtime (in days): {runtime} days")

    # Prompt the user for export option
    subsection_header("Exporting Hydrodynamic Data")
    info_box([
        "ComCon streams hydrodynamic data directly from Copernicus (using lazy loading). You do not need to download the data to run your analysis.",
        "",
        "Downloading is only useful if you want a local copy for:",
        "  - Offline use (no internet needed for future sessions)",
        "  - Other projects outside of ComCon",
        "",
        "Warning: Downloaded files can be very large (multiple GB) and take a long time to export."
    ])

    while True:
        export_input = input(
            "Do you want to export the data to a file? (yes/no) [default: no]: ").strip().lower() or 'no'
        if export_input in ["yes", "y", "no", "n"]:
            export = export_input in ["yes", "y"]
            break
        else:
            print("Error: Please enter 'yes' or 'no'.")

    # Only ask for output path if user wants to export
    if export:
        print("\n  Export format:")
        print("    1. NetCDF (.nc) Standard scientific data format")
        print("    2. Zarr (.zarr) Smaller, faster, recommended for large datasets")
        export_fmt_choice = input("  Choose format (1/2) [default: 1]: ").strip() or '1'
        export_format = 'zarr' if export_fmt_choice == '2' else 'netcdf'

        file_ext = '.zarr' if export_format == 'zarr' else '.nc'
        global project_dirs
        if project_dirs is not None:
            default_hydro_path = os.path.join(project_dirs['data'], f"hydro_data{file_ext}")
        else:
            default_hydro_path = f"./data/hydro_data{file_ext}"

        output_path = input(
            f"Enter the output path and file name [default: {default_hydro_path}]: "
        ).strip() or default_hydro_path

        # Ensure correct extension
        if export_format == 'zarr' and not output_path.endswith('.zarr'):
            output_path = output_path.rsplit('.', 1)[0] + '.zarr'
        elif export_format == 'netcdf' and not output_path.endswith('.nc'):
            output_path += '.nc'

        # Create directory if needed
        hydro_dir = os.path.dirname(output_path)
        if hydro_dir and not os.path.exists(hydro_dir):
            os.makedirs(hydro_dir, exist_ok=True)
    else:
        export_format = 'netcdf'
        output_path = None

    # Determine if buffer was added (any non-zero buffer value)
    add_buffer = dispersal_time["buffer_before"] > 0 or dispersal_time["buffer_after"] > 0

    return {
        "dataset_id": dataset_id,
        "longitude_range": longitude_range,
        "latitude_range": latitude_range,
        "time_range": time_range,
        "username": username,
        "password": password,
        "export": export,
        "export_format": export_format,
        "output_path": output_path,
        "traits": traits,
        "species_traits_path": species_traits_path,
        "runtime": runtime,
        "add_buffer": add_buffer,
        "buffer_before": dispersal_time["buffer_before"],
        "buffer_after": dispersal_time["buffer_after"]
    }

## Edit Hydrodynamic Parameters
def edit_hydro_params(hydro_params):
    """
    Interactively edits hydrodynamic data parameters.

    Allows users to modify any parameters in the hydrodynamic parameter dictionary
    including spatial bounds, temporal range and file paths. Handles different
    data types appropriately (lists, dates, strings, booleans).

    Parameters
    ----------
    hydro_params : dict
        Dictionary containing hydrodynamic parameters with the following keys:
        - 'dataset_id' : str
            Copernicus Marine dataset identifier
        - 'longitude_range' : list of float
            [min_lon, max_lon] in decimal degrees
        - 'latitude_range' : list of float
            [min_lat, max_lat] in decimal degrees
        - 'time_range' : list of datetime
            [start_date, end_date]
        - 'username' : str
            Copernicus Marine username
        - 'password' : str
            Copernicus Marine password
        - 'output_path' : str
            Path for saving downloaded data (.nc file)
        - 'export' : bool
            Whether to export data to NetCDF
        - 'traits' : list of dict
            Species trait data
        - 'species_traits_path' : str
            Path to species traits CSV file
        - 'runtime' : int
            Simulation runtime in days

    Returns
    -------
    dict
        Updated parameters dictionary with user modifications

    Notes
    -----
    - Date inputs must be in YYYY-MM-DD format
    - List parameters are edited element by element
    - Invalid inputs trigger error messages and repeat the prompt
    - Runtime is automatically recalculated if time_range is modified
    - Changing species_traits_path triggers automatic trait reload and runtime recalculation
    """
    # Define a list of keys that are lists and their expected types
    list_keys = {
        'longitude_range': float,
        'latitude_range': float,
        'time_range': str  # Time range will be treated as datetime objects
    }

    # Iterate over each parameter in the dictionary
    # Define the order of parameters to ask about
    # 'export' must come before 'output_path' so we can skip output_path if export is no
    param_order = [
        'dataset_id', 'longitude_range', 'latitude_range',
        'username', 'password', 'species_traits_path', 'add_buffer', 'buffer_before', 'buffer_after',
        'export', 'output_path'
    ]

    # Track whether export is enabled (either originally or after user changes it)
    export_enabled = hydro_params.get('export', False)

    # Track if buffer values were just set (to avoid asking again)
    buffer_values_just_set = False

    # Iterate over parameters in the defined order
    for key in param_order:
        # Skip if parameter doesn't exist in hydro_params
        if key not in hydro_params:
            continue

        value = hydro_params[key]

        # Skip the traits dictionary. Too complex to edit manually
        if key == 'traits':
            continue

        # Skip output_path if export is not enabled
        if key == 'output_path' and not export_enabled:
            continue

        # Handle species_traits_path specially. Reload traits if changed
        if key == 'species_traits_path':
            while True:
                edit = input(f"Do you want to edit {key} (current value: {value})? (yes/no): ").strip().lower()
                if edit in ["yes", "y", "no", "n"]:
                    break
                else:
                    print("Error: Please enter 'yes' or 'no'.")

            if edit in ["yes", "y"]:
                # Give user the same choice as initial setup
                print("\nTrait data source:")
                print("  1. Use built-in trait database")
                print("  2. Provide your own CSV file")
                trait_choice = input("\nSelect option (1/2) [default: 1]: ").strip() or '1'

                if trait_choice == '1':
                    # Use built-in database — re-run species selection
                    traits_df = load_builtin_traits()
                    if traits_df is not None:
                        selected_traits, species_indices = select_species_from_database(traits_df)
                        if selected_traits is not None:
                            hydro_params['species_traits_path'] = 'builtin_database_selection'
                            # Store traits for later use
                            traits_list = []
                            for _, row in selected_traits.iterrows():
                                traits_list.append({
                                    "species_name": row.get('species_name'),
                                    "aphia_id": row.get('aphia_id') if pd.notna(
                                        row.get('aphia_id', None)) else None,
                                    "spawning_start": str(row['spawning_start']),
                                    "spawning_end": str(row['spawning_end']),
                                    "min_PLD": int(row['min_PLD']),
                                    "max_PLD": int(row['max_PLD'])
                                })
                            hydro_params['traits'] = traits_list
                            print(f"\n  Updated: {len(traits_list)} species selected from built-in database.")
                    continue  # Skip to next parameter

                # Option 2: Custom CSV file
                traits_loaded = False

                while not traits_loaded:
                    new_path = input(f"Enter the new file path for species traits CSV: ").strip()

                    # Add .csv extension if missing
                    if not new_path.endswith('.csv'):
                        new_path += '.csv'

                    # Validate the file exists
                    if not os.path.exists(new_path):
                        print(f"Error: File not found: {new_path}")
                        while True:
                            retry = input("Do you want to try a different file? (yes/no): ").strip().lower()
                            if retry in ['yes', 'y', 'no', 'n']:
                                break
                            else:
                                print("Error: Please enter 'yes' or 'no'.")

                        if retry in ['no', 'n']:
                            print("Keeping current traits.\n")
                            break
                        else:
                            continue  # Ask for file path again

                    # Reload traits from new file
                    print("\nReloading traits from new file...")
                    try:
                        # First, read and validate the CSV structure
                        traits_df = pd.read_csv(new_path)

                        # Validate required columns (excluding identifiers)
                        required_cols = ['min_PLD', 'max_PLD', 'spawning_start', 'spawning_end']
                        missing_cols = [col for col in required_cols if col not in traits_df.columns]

                        if missing_cols:
                            print(f"\nError: Missing required columns in traits CSV:")
                            for col in missing_cols:
                                print(f"  - {col}")
                            print(f"\nYour CSV has: {', '.join(traits_df.columns.tolist())}")
                            print(f"Required: {', '.join(required_cols)}")
                            print(f"Required (at least one): species_name or aphia_id")

                            while True:
                                retry = input("\nDo you want to try a different file? (yes/no): ").strip().lower()
                                if retry in ['yes', 'y', 'no', 'n']:
                                    break
                                else:
                                    print("Error: Please enter 'yes' or 'no'.")

                            if retry in ['no', 'n']:
                                print("Keeping current traits.\n")
                                break
                            else:
                                continue  # Ask for file path again

                        # Check that at least one identifier column exists
                        has_species_name = 'species_name' in traits_df.columns
                        has_aphia_id = 'aphia_id' in traits_df.columns

                        if not has_species_name and not has_aphia_id:
                            print(f"\nError: Your CSV must have at least one identifier column:")
                            print(f"  - species_name or")
                            print(f"  - aphia_id")
                            print(f"\nYour CSV has: {', '.join(traits_df.columns.tolist())}")

                            while True:
                                retry = input("\nDo you want to try a different file? (yes/no): ").strip().lower()
                                if retry in ['yes', 'y', 'no', 'n']:
                                    break
                                else:
                                    print("Error: Please enter 'yes' or 'no'.")

                            if retry in ['no', 'n']:
                                print("Keeping current traits.\n")
                                break
                            else:
                                continue  # Ask for file path again

                        # Now read the actual trait data with validation
                        new_traits = []
                        row_num = 0
                        validation_errors = []

                        with open(new_path, mode='r', encoding='utf-8-sig') as file:
                            reader = csv.DictReader(file)
                            for row in reader:
                                row_num += 1

                                # Extract and clean identifiers
                                species_name = row.get('species_name', '').strip()
                                if species_name.upper() in ['NA', 'N/A', 'NONE', '']:
                                    species_name = None

                                aphia_id_raw = row.get('aphia_id', '').strip()
                                if aphia_id_raw.upper() in ['NA', 'N/A', 'NONE', '']:
                                    aphia_id = None
                                else:
                                    try:
                                        aphia_id = int(aphia_id_raw)
                                    except ValueError:
                                        validation_errors.append(
                                            f"Row {row_num}: Invalid aphia_id '{aphia_id_raw}'. Must be a number or leave blank."
                                        )
                                        aphia_id = None

                                # Check that at least one identifier is present for this species
                                if species_name is None and aphia_id is None:
                                    validation_errors.append(
                                        f"Row {row_num}: Missing both species_name and aphia_id. At least one is required."
                                    )
                                    continue  # Skip this row

                                # Validate PLD values
                                try:
                                    min_pld = int(row['min_PLD'])
                                    max_pld = int(row['max_PLD'])

                                    if min_pld < 0 or max_pld < 0:
                                        validation_errors.append(
                                            f"Row {row_num}: PLD values must be non-negative."
                                        )
                                        continue

                                    if min_pld > max_pld:
                                        validation_errors.append(
                                            f"Row {row_num}: min_PLD ({min_pld}) cannot be greater than max_PLD ({max_pld})."
                                        )
                                        continue

                                except ValueError as e:
                                    validation_errors.append(
                                        f"Row {row_num}: Invalid PLD value. Must be an integer."
                                    )
                                    continue

                                # Validate date format
                                try:
                                    datetime.strptime(row['spawning_start'], '%d/%m/%Y')
                                    datetime.strptime(row['spawning_end'], '%d/%m/%Y')
                                except ValueError:
                                    validation_errors.append(
                                        f"Row {row_num}: Invalid date format. Must be DD/MM/YYYY."
                                    )
                                    continue

                                # If all validations pass, add to traits
                                new_traits.append({
                                    "species_name": species_name,
                                    "aphia_id": aphia_id,
                                    "spawning_start": row['spawning_start'],
                                    "spawning_end": row['spawning_end'],
                                    "min_PLD": min_pld,
                                    "max_PLD": max_pld
                                })

                        # Check if there were validation errors
                        if validation_errors:
                            print(f"\nError found in {new_path}:")
                            for error in validation_errors:
                                print(f"  • {error}")
                            print(
                                f"\nLoaded {len(new_traits)} valid species (skipped {len(validation_errors)} rows with errors).")

                            if len(new_traits) == 0:
                                print("No valid species found.")

                                while True:
                                    retry = input(
                                        "\nDo you want to try a different file? (yes/no): ").strip().lower()
                                    if retry in ['yes', 'y', 'no', 'n']:
                                        break
                                    else:
                                        print("Error: Please enter 'yes' or 'no'.")

                                if retry in ['no', 'n']:
                                    print("Keeping current traits.\n")
                                    break
                                else:
                                    continue  # Ask for file path again

                            # Ask user if they want to proceed with partial data
                            while True:
                                proceed = input("\nProceed with valid species only? (yes/no): ").strip().lower()
                                if proceed in ['yes', 'y', 'no', 'n']:
                                    break
                                else:
                                    print("Error: Please enter 'yes' or 'no'.")

                            if proceed in ['no', 'n']:
                                while True:
                                    retry = input("Do you want to try a different file? (yes/no): ").strip().lower()
                                    if retry in ['yes', 'y', 'no', 'n']:
                                        break
                                    else:
                                        print("Error: Please enter 'yes' or 'no'.")

                                if retry in ['no', 'n']:
                                    print("Keeping current traits.\n")
                                    break
                                else:
                                    continue

                        # Get buffer values if they exist
                        buffer_before = hydro_params.get('buffer_before', 0)
                        buffer_after = hydro_params.get('buffer_after', 0)

                        # Recalculate dispersal period and runtime based on new traits
                        overall_start, overall_end, base_start, base_end = calculate_overall_dispersal_period(
                            new_traits, buffer_before, buffer_after
                        )

                        new_runtime = (datetime.strptime(overall_end, '%Y-%m-%d') -
                                       datetime.strptime(overall_start, '%Y-%m-%d')).days

                        # Update parameters
                        hydro_params['species_traits_path'] = new_path
                        hydro_params['traits'] = new_traits
                        hydro_params['runtime'] = new_runtime
                        hydro_params['time_range'] = [
                            datetime.strptime(overall_start, '%Y-%m-%d'),
                            datetime.strptime(overall_end, '%Y-%m-%d')
                        ]

                        print(f"Traits reloaded: {len(new_traits)} species")

                        traits_loaded = True  # Exit retry loop

                    except Exception as e:
                        print(f"Error loading new traits file: {e}")

                        while True:
                            retry = input("Do you want to try a different file? (yes/no): ").strip().lower()
                            if retry in ['yes', 'y', 'no', 'n']:
                                break
                            else:
                                print("Error: Please enter 'yes' or 'no'.")

                        if retry in ['no', 'n']:
                            print("Keeping current traits.\n")
                            break

            continue

        # Handle add_buffer decision
        if key == 'add_buffer':
            # Format current value as yes/no for display
            current_display = "yes" if value else "no"

            while True:
                new_input = input(
                    f"Do you want to add a temporal buffer? (yes/no) (current value: {current_display}): ").strip().lower()
                if new_input in ['yes', 'y', 'no', 'n']:
                    new_value = new_input in ['yes', 'y']
                    break
                else:
                    print("Error: Please enter 'yes' or 'no'.")

            # Update add_buffer
            hydro_params['add_buffer'] = new_value

            # If buffer is enabled, allow user to update buffer values
            if new_value:  # Buffer is/will be enabled
                # Only prompt for values if we're enabling for the first time OR user kept it enabled
                if not value:  # Enabling buffer for first time
                    print("\nEnabling temporal buffer for trait sensitivity analysis.")
                else:  # Buffer was already enabled, user chose to keep it
                    print("\nBuffer is already enabled. Do you want to edit the buffer values?")
                    while True:
                        update = input("Edit buffer values? (yes/no): ").strip().lower()
                        if update in ['yes', 'y', 'no', 'n']:
                            break
                        else:
                            print("Error: Please enter 'yes' or 'no'.")

                    if update in ['no', 'n']:
                        print("Keeping current buffer values.\n")
                        continue

                    # Prompt for buffer values
                    current_before = hydro_params.get('buffer_before', 14)
                    current_after = hydro_params.get('buffer_after', 14)

                    buffer_before = get_positive_integer(
                        f"Days to add before earliest spawning (current value: {current_before})",
                        min_val=0
                    )
                    buffer_after = get_positive_integer(
                        f"Days to add after latest dispersal end (current value: {current_after})",
                        min_val=0
                    )

                hydro_params['buffer_before'] = buffer_before
                hydro_params['buffer_after'] = buffer_after

                # Recalculate with new buffers
                print("\nRecalculating dispersal period with buffer...")
                traits = hydro_params.get('traits', [])

                if traits:
                    overall_start, overall_end, base_start, base_end = calculate_overall_dispersal_period(
                        traits, buffer_before, buffer_after
                    )

                    hydro_params['time_range'] = [
                        datetime.strptime(overall_start, '%Y-%m-%d'),
                        datetime.strptime(overall_end, '%Y-%m-%d')
                    ]
                    hydro_params['runtime'] = (hydro_params['time_range'][1] - hydro_params['time_range'][0]).days

                # Always mark that buffer values were just set
                buffer_values_just_set = True

            # If changing from yes to no, remove buffers
            elif not new_value and value:  # Disabling buffer
                print("\nDisabling temporal buffer. Recalculating to base dispersal period...")
                hydro_params['buffer_before'] = 0
                hydro_params['buffer_after'] = 0

                traits = hydro_params.get('traits', [])

                if traits:
                    overall_start, overall_end, base_start, base_end = calculate_overall_dispersal_period(
                        traits, 0, 0
                    )

                    hydro_params['time_range'] = [
                        datetime.strptime(overall_start, '%Y-%m-%d'),
                        datetime.strptime(overall_end, '%Y-%m-%d')
                    ]
                    hydro_params['runtime'] = (hydro_params['time_range'][1] - hydro_params['time_range'][0]).days

            continue

        # Handle buffer parameters specially - they affect time_range and runtime
        # Only ask if add_buffer is True
        if key in ['buffer_before', 'buffer_after']:
            # Skip if buffer is disabled
            if not hydro_params.get('add_buffer', False):
                continue

            # Skip if buffer values were just set when enabling add_buffer
            if buffer_values_just_set:
                continue

            display_name = "buffer before earliest spawning" if key == 'buffer_before' else "buffer after latest dispersal"

            while True:
                edit = input(
                    f"Do you want to edit {display_name} (current value: {value} days)? (yes/no): ").strip().lower()
                if edit in ["yes", "y", "no", "n"]:
                    break
                else:
                    print("Error: Please enter 'yes' or 'no'.")

            if edit in ["yes", "y"]:
                while True:
                    try:
                        new_value = int(
                            input(f"Enter new value for {display_name} (days, current: {value}): ").strip())
                        if new_value < 0:
                            print("Error: Buffer must be non-negative (0 or more days).")
                            continue
                        break
                    except ValueError:
                        print("Error: Please enter a valid integer.")

                # Update the buffer value
                hydro_params[key] = new_value

                # Recalculate dispersal period with new buffer values
                print("\nRecalculating dispersal period with new buffer...")
                buffer_before = hydro_params.get('buffer_before', 0)
                buffer_after = hydro_params.get('buffer_after', 0)
                traits = hydro_params.get('traits', [])

                if traits:
                    overall_start, overall_end, base_start, base_end = calculate_overall_dispersal_period(
                        traits, buffer_before, buffer_after
                    )

                    # Update time_range and runtime
                    hydro_params['time_range'] = [
                        datetime.strptime(overall_start, '%Y-%m-%d'),
                        datetime.strptime(overall_end, '%Y-%m-%d')
                    ]
                    hydro_params['runtime'] = (hydro_params['time_range'][1] - hydro_params['time_range'][0]).days

                    print(f"Time range updated: {overall_start} to {overall_end}")
                    print(f"Runtime recalculated: {hydro_params['runtime']} days\n")
                else:
                    print("Warning: No traits data available for recalculation.\n")

            continue

        # Handle all other parameters normally
        # Format display value appropriately
        if key == 'password':
            display_value = "** Sensitive information **"
        elif key in ['export', 'add_buffer']:
            display_value = "yes" if value else "no"
        else:
            display_value = value

        while True:
            edit = input(f"Do you want to edit {key} (current value: {display_value})? (yes/no): ").strip().lower()
            if edit in ["yes", "y", "no", "n"]:
                break
            else:
                print("Error: Please enter 'yes' or 'no'.")

        if edit in ["yes", "y"]:
            if key in list_keys:
                # Handle list values separately (longitude_range, latitude_range, time_range)
                if len(value) == 2:
                    if key == 'time_range':
                        # Handle date input for time_range with validation loop
                        while True:
                            try:
                                start_str = input(
                                    f"Enter the new start date for {key} (YYYY-MM-DD) (current value: {value[0].strftime('%Y-%m-%d')}): "
                                ).strip()
                                new_start = datetime.strptime(start_str, '%Y-%m-%d')
                                break
                            except ValueError:
                                print("Invalid date format. Enter dates in YYYY-MM-DD format.")

                        while True:
                            try:
                                end_str = input(
                                    f"Enter the new end date for {key} (YYYY-MM-DD) (current value: {value[1].strftime('%Y-%m-%d')}): "
                                ).strip()
                                new_end = datetime.strptime(end_str, '%Y-%m-%d')

                                # Check that end date is after start date
                                if new_end <= new_start:
                                    print("Error: End date must be after start date.")
                                    continue
                                break
                            except ValueError:
                                print("Invalid date format. Enter dates in YYYY-MM-DD format.")

                        new_value = [new_start, new_end]

                        # Recalculate runtime if time_range was changed
                        hydro_params['runtime'] = (new_end - new_start).days

                    else:
                        # Handle numeric values for latitude_range, longitude_range with validation
                        while True:
                            try:
                                new_first = list_keys[key](input(
                                    f"Enter the new value for {key}. The minimum or first element (current value: {value[0]}): "
                                ).strip())
                                new_second = list_keys[key](input(
                                    f"Enter the new value for {key}. The maximum or second element (current value: {value[1]}): "
                                ).strip())

                                # Validation: first should be less than second
                                if new_first >= new_second:
                                    print(f"Error: First value must be less than second value for {key}.")
                                    continue

                                # Additional validation for latitude
                                if key == 'latitude_range':
                                    if not (-90 <= new_first <= 90) or not (-90 <= new_second <= 90):
                                        print("Error: Latitude must be between -90 and 90.")
                                        continue

                                new_value = [new_first, new_second]
                                break
                            except ValueError:
                                print(f"Error: Please enter valid {list_keys[key].__name__} values.")
                else:
                    print(f"Expected a list with 2 elements for {key}.")
                    continue
            else:
                # For non-list values, update the parameter with validation
                if key == 'export':
                    # Format current value as yes/no for display
                    current_display = "yes" if value else "no"

                    while True:
                        new_input = input(
                            f"Do you want to export the data to a NetCDF file? (yes/no) (current value: {current_display}): ").strip().lower()
                        if new_input in ['yes', 'y', 'no', 'n']:
                            new_value = new_input in ['yes', 'y']
                            # Update export_enabled tracker
                            export_enabled = new_value
                            break
                        else:
                            print("Error: Please enter 'yes' or 'no'.")

                elif key == 'runtime':
                    # Allow manual override of runtime with a warning
                    print(f"\nWarning: Runtime is normally calculated from species traits ({value} days).")
                    print("Manual changes may cause mismatches between runtime and trait data.")
                    override = input("Are you sure you want to manually set runtime? (yes/no): ").strip().lower()

                    if override in ['yes', 'y']:
                        while True:
                            try:
                                new_input = input(
                                    f"Enter the new value for runtime in days (current value: {value}): ").strip()
                                new_value = int(new_input)
                                if new_value <= 0:
                                    print("Error: Runtime must be positive.")
                                    continue
                                break
                            except ValueError:
                                print("Error: Please enter a valid integer.")
                    else:
                        print("Keeping calculated runtime.")
                        continue

                elif isinstance(value, (int, float)) and key != 'runtime':
                    while True:
                        try:
                            new_input = input(f"Enter the new value for {key} (current value: {value}): ").strip()
                            new_value = type(value)(new_input)
                            break
                        except ValueError:
                            print(f"Error: Please enter a valid {type(value).__name__}.")
                else:
                    # String values (username, password, dataset_id, output_path, etc.)
                    # For password, don't show current value
                    if key == 'password':
                        new_value = input(f"Enter the new value for {key}: ").strip()
                    elif key == 'data_source':
                        print(f"\nCurrent value: {value}")
                        print("Options:")
                        print("  'gbif' = Download from GBIF database")
                        print("  'user' = Upload your own occurrence data")
                        new_value = input("Enter 'gbif' or 'user': ").strip()
                    elif key == 'habitat_filtering':
                        print(f"\nCurrent value: {value}")
                        print("Options:")
                        print("  'coastal' = Coastal (littoral/intertidal species)")
                        print("  'ocean' = Ocean (coastal + offshore)")
                        print("  'none' = No filtering (use data as-is)")
                        new_value = input("Enter 'coastal', 'ocean', or 'none': ").strip()
                    else:
                        new_value = input(f"Enter the new value for {key} (current value: {value}): ").strip()

                    # Ensure .nc extension for output_path
                    if key == 'output_path' and new_value and not new_value.endswith('.nc'):
                        new_value += '.nc'

            # Update the parameter in the dictionary
            hydro_params[key] = new_value

    # Convert the time_range back to strings before saving
    hydro_params['time_range'] = [
        hydro_params['time_range'][0].strftime('%Y-%m-%d'),
        hydro_params['time_range'][1].strftime('%Y-%m-%d')
    ]

    # Save the updated parameters to the JSON file
    with open("hydrodynamic_parameters.json", "w") as file:
        json.dump(hydro_params, file, indent=4)

    print("\nParameters updated and saved.")

    return hydro_params
# endregion

## region Load Hydrodynamic Parameters (wrapper function)
def params():
    """
    Complete interactive workflow for hydrodynamic parameter collection and editing.

    Manages the full process of obtaining hydrodynamic data parameters either by
    collecting new parameters from the user or loading and editing previously saved
    parameters. Saves parameter configurations to JSON for reuse in future sessions.

    The function checks for existing parameter files and offers options to:
    1. Reuse existing parameters as-is
    2. Edit specific existing parameters
    3. Collect entirely new parameters

    Returns
    -------
    hydro_params : dict
        Complete dictionary of hydrodynamic parameters including:
        - dataset_id : str
            Copernicus Marine dataset identifier
        - longitude_range : list of float
            Spatial bounds [min_lon, max_lon]
        - latitude_range : list of float
            Spatial bounds [min_lat, max_lat]
        - time_range : list of datetime
            Temporal bounds [start_date, end_date]
        - username : str
            User credentials
        - password : str
            User credentials
        - output_path : str
            NetCDF output file path
        - export : bool
            Export flag

    species_traits_path : str
        Path to the species traits CSV file for downstream use

    runtime : int
        Calculated simulation runtime in days based on temporal range

    traits : list of dict
        Species biological traits data

    Notes
    -----
    - Parameters are automatically saved to 'hydrodynamic_parameters.json' in the working directory
    - The function integrates species trait data to determine appropriate temporal coverage
    - Runtime is calculated from time_range to ensure sufficient hydrodynamic data
    - All previous parameter files are overwritten when new parameters are saved
    """
    # Use project directory (should already be set up in workflow_step)
    global project_dirs
    if 'project_dirs' not in globals() or project_dirs is None:
        # Fallback: create project directory if called outside workflow_step
        subsection_header("Project Setup")
        info_box([
            "ComCon organises all outputs into a project folder.",
            "Each project gets its own directory with subfolders for configuration, data, figures, simulation output and results.",
            "",
            "This keeps your work organised and prevents files being overwritten between different analyses.",
        ])
        project_dirs = setup_project_directory()

    params_file = os.path.join(project_dirs['config'], "hydrodynamic_parameters.json")

    # Check if parameters file exists and offer to reuse
    if os.path.exists(params_file):
        while True:
            reuse_input = input("Found existing parameters. Do you want to reuse them? (yes/no): ").strip().lower()
            if reuse_input in ["yes", "y", "no", "n"]:
                break
            else:
                print("Error: Please enter 'yes' or 'no'.")

        if reuse_input in ["yes", "y"]:
            with open(params_file, "r") as file:
                hydro_params = json.load(file)
                # Convert time_range from strings back to datetime objects
                hydro_params['time_range'] = [
                    datetime.strptime(hydro_params['time_range'][0], '%Y-%m-%d'),
                    datetime.strptime(hydro_params['time_range'][1], '%Y-%m-%d')
                ]

                # Display parameters nicely formatted
                # Build parameter lines for display
                param_lines = []
                for key, value in hydro_params.items():
                    if key == 'traits':
                        continue  # Show separately below
                    elif key == 'password':
                        param_lines.append(f"{key}: ** Sensitive information **")
                    elif key == 'time_range':
                        # Format time_range as dates
                        formatted_range = f"[{value[0].strftime('%Y-%m-%d')}, {value[1].strftime('%Y-%m-%d')}]"
                        param_lines.append(f"{key}: {formatted_range}")
                    elif key in ['add_buffer', 'export']:
                        # Format boolean as yes/no instead of True/False
                        display_value = "yes" if value else "no"
                        param_lines.append(f"{key}: {display_value}")
                    else:
                        param_lines.append(f"{key}: {value}")
                data_box("Existing Hydrodynamic Parameters", param_lines)

                # Display traits as a DataFrame
                if 'species_traits_path' in hydro_params and hydro_params['species_traits_path']:
                    subsection_header("Species Traits Data")
                    species_traits_path = hydro_params['species_traits_path']
                    if species_traits_path == 'builtin_database_selection':
                        traits_df = load_builtin_traits()
                        if traits_df is not None:
                            print("(Using builtin trait database)")
                            print(traits_df)
                    else:
                        traits_df = pd.read_csv(species_traits_path)
                        print(traits_df)
                    display_line("─")

            # Offer to edit specific parameters
            while True:
                edit_input = input("\nDo you want to edit any parameters? (yes/no): ").strip().lower()
                if edit_input in ["yes", "y", "no", "n"]:
                    break
                else:
                    print("Error: Please enter 'yes' or 'no'.")

            if edit_input in ["yes", "y"]:
                hydro_params = edit_hydro_params(hydro_params)

                # Convert time_range back from strings to datetime after editing
                hydro_params['time_range'] = [
                    datetime.strptime(hydro_params['time_range'][0], '%Y-%m-%d'),
                    datetime.strptime(hydro_params['time_range'][1], '%Y-%m-%d')
                ]

                # Extract species traits path
                species_traits_path = hydro_params.get("species_traits_path", None)

                # Check if user wants to recalculate runtime from (potentially updated) traits
                print("\nNote: Runtime is calculated from your species traits file.")
                print(f"Current traits file: {species_traits_path}")
                print(f"Current runtime: {hydro_params.get('runtime')} days")
                recalc = input("\nRecalculate runtime from traits file? (yes/no) [default: no]: ").strip().lower()

                if recalc in ['yes', 'y']:
                    print("Recalculating runtime from traits file...")
                    dispersal_time, _, traits = load_spp_traits_from_path(species_traits_path)
                    if dispersal_time:
                        new_start = datetime.strptime(dispersal_time["overall_dispersal_start"], '%Y-%m-%d')
                        new_end = datetime.strptime(dispersal_time["overall_dispersal_end"], '%Y-%m-%d')
                        runtime = (new_end - new_start).days
                        hydro_params['runtime'] = runtime
                        hydro_params['traits'] = traits
                        hydro_params['time_range'] = [new_start, new_end]
                        print(f"New runtime: {runtime} days")
                    else:
                        print("Error reading traits. Keeping current runtime.")
                        traits = hydro_params.get("traits", None)
                        runtime = hydro_params.get("runtime", None)

                else:
                    traits = hydro_params.get("traits", None)
                    runtime = hydro_params.get("runtime", None)

            else:
                # User said NO to editing - load values from saved params
                species_traits_path = hydro_params.get("species_traits_path", None)
                traits = hydro_params.get("traits", None)
                runtime = hydro_params.get("runtime", None)

            # Remove the traits, species traits path, runtime, and buffer values before returning hydro_params
            hydro_params.pop("traits", None)
            hydro_params.pop("species_traits_path", None)
            hydro_params.pop("runtime", None)
            hydro_params.pop("buffer_before", None)
            hydro_params.pop("buffer_after", None)
            hydro_params.pop("add_buffer", None)

            return hydro_params, species_traits_path, runtime, traits

    # If not reusing or no existing file, collect new parameters
    hydro_params = collect_hydro_params()

    if hydro_params is None:
        print("Error: Failed to collect parameters.")
        return None, None, None, None

    # Offer to review and edit before saving
    while True:
        review_input = input(
            "\nDo you want to review and edit the parameters before saving? (yes/no): ").strip().lower()
        if review_input in ["yes", "y", "no", "n"]:
            break
        else:
            print("Error: Please enter 'yes' or 'no'.")

    if review_input in ["yes", "y"]:
        hydro_params = edit_hydro_params(hydro_params)

        # Convert time_range back from strings to datetime after editing
        hydro_params['time_range'] = [
            datetime.strptime(hydro_params['time_range'][0], '%Y-%m-%d'),
            datetime.strptime(hydro_params['time_range'][1], '%Y-%m-%d')
        ]

    # Extract runtime before conversion
    runtime = hydro_params['runtime']

    # Convert the time_range to strings for JSON storage
    hydro_params['time_range'] = [
        hydro_params['time_range'][0].strftime('%Y-%m-%d'),
        hydro_params['time_range'][1].strftime('%Y-%m-%d')
    ]

    # Save the parameters to a JSON file
    with open(params_file, "w") as file:
        json.dump(hydro_params, file, indent=4)

    print("\nParameters have been saved to 'hydrodynamic_parameters.json'.")

    # Convert time_range back to datetime objects for return
    hydro_params['time_range'] = [
        datetime.strptime(hydro_params['time_range'][0], '%Y-%m-%d'),
        datetime.strptime(hydro_params['time_range'][1], '%Y-%m-%d')
    ]

    # Extract species traits and species traits path
    traits = hydro_params.get("traits", None)
    species_traits_path = hydro_params.get("species_traits_path", None)

    # Remove the species traits, path, runtime, and buffer values from hydro_params before returning
    hydro_params.pop("traits", None)
    hydro_params.pop("species_traits_path", None)
    hydro_params.pop("runtime", None)
    hydro_params.pop("buffer_before", None)
    hydro_params.pop("buffer_after", None)
    hydro_params.pop("add_buffer", None)

    return hydro_params, species_traits_path, runtime, traits
# endregion

## region Downloading the Hydrodynamic Data
# Downloading and streaming the hydrodynamic data
def download_hydrodynamic_data(**kwargs):
    """
    Downloads hydrodynamic data from Copernicus Marine Service.

    Connects to Copernicus Marine Service API and downloads ocean velocity
    (U, V) data for the specified spatial and temporal extent. Data is
    compressed and saved as NetCDF if export=True.

    Parameters:
    -----------
    username : str
        Copernicus Marine Service username
    password : str
        Copernicus Marine Service password
    dataset_id : str
        Copernicus dataset identifier (e.g., 'cmems_mod_ibi_phy_anfc_0.027deg-2D_PT1H-m')
    longitude_range : list of float
        [min_longitude, max_longitude] in decimal degrees
    latitude_range : list of float
        [min_latitude, max_latitude] in decimal degrees
    time_range : list of str
        [start_date, end_date] in 'YYYY-MM-DD' format
    output_path : str
        Path to save the downloaded NetCDF file
    export : bool, default=True
        Whether to save data to NetCDF file

    Returns:
    --------
    hydro_data : xarray.Dataset
        Dataset containing U (zonal velocity) and V (meridional velocity)

    Raises:
    -------
    ValueError
        If output_path not provided when export=True
    ConnectionError
        If unable to connect to Copernicus Marine Service

    Notes:
    ------
    - Requires active internet connection to Copernicus Marine Service
    - By default (export=False), data is streamed directly from Copernicus
      using lazy loading — no large file download is needed
    - Set export=True only if you want a local copy for future offline use
    - When export=True, data is compressed with level 9 compression (NetCDF)
      or default compression (Zarr)
    - Downloaded files can be very large (multiple GB) depending on the
      spatial and temporal extent
    - To reload a previously saved file without internet, use
      load_hydrodynamic_data() instead
    """

    # Extract parameters from kwargs
    username = kwargs.get("username")
    password = kwargs.get("password")
    dataset_id = kwargs.get("dataset_id")
    longitude_range = kwargs.get("longitude_range")
    latitude_range = kwargs.get("latitude_range")
    time_range = kwargs.get("time_range")
    output_path = kwargs.get("output_path")
    export = kwargs.get("export", True)

    # Automatically extend end date by 31 days for release interval safety
    # This ensures hydro data always covers adjusted runtime regardless of
    # release interval chosen in step 3 (daily/weekly/monthly)
    # This is applied at download time so it works for both new and reused parameters
    hydro_safety_buffer = 31

    if isinstance(time_range[1], str):
        original_end = datetime.strptime(time_range[1], '%Y-%m-%d')
    else:
        original_end = time_range[1]

    extended_end = original_end + timedelta(days=hydro_safety_buffer)

    # Format dates for Copernicus API
    if isinstance(time_range[0], str):
        start_date = time_range[0]
    else:
        start_date = time_range[0].strftime('%Y-%m-%d')

    end_date = extended_end.strftime('%Y-%m-%d')

    # Set parameters
    data_request = {
        "dataset_id": dataset_id,
        "longitude": longitude_range,
        "latitude": latitude_range,
        "time": [start_date, end_date]
    }

    # Load xarray dataset
    data = cm.open_dataset(
        dataset_id=data_request["dataset_id"],
        minimum_longitude=data_request["longitude"][0],
        maximum_longitude=data_request["longitude"][1],
        minimum_latitude=data_request["latitude"][0],
        maximum_latitude=data_request["latitude"][1],
        start_datetime=data_request["time"][0],
        end_datetime=data_request["time"][1],
        username=username,
        password=password
    )

    # Save to file
    export_format = kwargs.get("export_format", "netcdf")

    if export:
        # Check if output_path is provided
        if not output_path:
            raise ValueError("output_path must be provided if export is True.")

        # Create directory if needed
        export_dir = os.path.dirname(output_path)
        if export_dir and not os.path.exists(export_dir):
            os.makedirs(export_dir, exist_ok=True)

        if export_format == 'zarr':
            # Ensure .zarr extension
            if not output_path.endswith('.zarr'):
                output_path = output_path.rsplit('.', 1)[0] + '.zarr'
            data.to_zarr(output_path, mode='w')
            hydro_data = xr.open_zarr(output_path)
        else:
            # Ensure .nc extension
            if not output_path.endswith('.nc'):
                output_path += '.nc'
            encoding = {var: {'zlib': True, 'complevel': 9} for var in data.data_vars}
            data.to_netcdf(output_path, encoding=encoding)
            hydro_data = xr.open_dataset(output_path)

        # Standardise coordinates
        hydro_data = standardise_dims(hydro_data)
    else:
        # Directly use the data without saving to a file and standardise coordinates
        hydro_data = standardise_dims(data)

    return hydro_data

# Standardising the coordinates
def standardise_dims(hydro_data):
    """
    Standardise Copernicus dataset dimension names and order.

    Different Copernicus products use different names:
    - time/t
    - lat/latitude
    - lon/longitude

    This function renames to standard: time, lat, lon
    """
    # Rename dimensions to standard names
    rename_dict = {}

    # Time dimension
    if 't' in hydro_data.dims and 'time' not in hydro_data.dims:
        rename_dict['t'] = 'time'

    # Latitude dimension
    if 'latitude' in hydro_data.dims and 'lat' not in hydro_data.dims:
        rename_dict['latitude'] = 'lat'

    # Longitude dimension
    if 'longitude' in hydro_data.dims and 'lon' not in hydro_data.dims:
        rename_dict['longitude'] = 'lon'

    if rename_dict:
        hydro_data = hydro_data.rename(rename_dict)

    return hydro_data

# endregion

# Loading hydrodynamic data thats been downloaded
def load_hydrodynamic_data(file_path):
    """
    Load hydrodynamic data from a local NetCDF or Zarr file.

    Use this to reload previously downloaded data without needing
    an internet connection. The file must contain 'uo' (eastward velocity)
    and 'vo' (northward velocity) variables with time, latitude, and
    longitude dimensions.

    Parameters
    ----------
    file_path : str
        Path to a NetCDF (.nc) or Zarr (.zarr) file containing
        hydrodynamic velocity data.

    Returns
    -------
    hydro_data : xarray.Dataset
        Dataset with standardised dimension names (time, lat, lon)

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist
    ValueError
        If the file does not contain required variables ('uo' and 'vo')

    Notes
    -----
    - Supported formats: NetCDF (.nc), Zarr (.zarr)
    - The file must contain at minimum: 'uo', 'vo' variables
    - Dimension names are automatically standardised (latitude→lat, longitude→lon, t→time)
    - No internet connection is required
    - This function is equivalent to downloading and reloading from Copernicus,
      but skips the download step

    Examples
    --------
    # In manual mode:
    hydro_data = load_hydrodynamic_data('./projects/MyProject/data/hydro_data.nc')

    # In interactive mode, this is offered as an option in workflow_step(1)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"\nLoading hydrodynamic data from: {file_path}")

    if file_path.endswith('.zarr'):
        hydro_data = xr.open_zarr(file_path)
    else:
        hydro_data = xr.open_dataset(file_path)

    # Standardise dimension names
    hydro_data = standardise_dims(hydro_data)

    # Validate required variables
    required_vars = ['uo', 'vo']
    missing = [v for v in required_vars if v not in hydro_data.data_vars]
    if missing:
        raise ValueError(
            f"File is missing required variables: {missing}. "
            f"The file must contain 'uo' (eastward velocity) and 'vo' (northward velocity)."
        )

    print(f"  Loaded successfully: {list(hydro_data.data_vars)}")
    print(f"  Time range: {str(hydro_data.time.values[0])[:10]} to {str(hydro_data.time.values[-1])[:10]}")
    print(f"  Spatial extent: lon [{float(hydro_data.lon.min()):.2f}, {float(hydro_data.lon.max()):.2f}], "
          f"lat [{float(hydro_data.lat.min()):.2f}, {float(hydro_data.lat.max()):.2f}]")

    return hydro_data

# endregion

# region Project Directory Setup
def setup_project_directory(project_name=None, custom_path=None):
    """
    Create or use a project directory for organising all ComCon outputs.

    Offers two options:
    1. Create a new ComCon project folder with standard subfolders
    2. Use an existing directory (with optional subfolder creation)

    Parameters
    ----------
    project_name : str, optional
        Name for the project folder. If None, prompts the user.
    custom_path : str, optional
        Path to a custom directory. If provided, skips the interactive prompt.

    Returns
    -------
    dict
        Dictionary of paths: 'root', 'config', 'data', 'occurrence',
        'grids', 'simulation', 'figures', 'results'
    """
    import re

    # Define the standard subfolder structure
    subfolder_names = ['config', 'data', 'occurrence', 'grids', 'simulation', 'figures', 'results']

    subfolder_descriptions = [
        "config/       — Saved parameter files (.json)",
        "data/         — Exported hydrodynamic data",
        "occurrence/   — Cleaned species occurrence data",
        "grids/        — Release and settlement grids",
        "simulation/   — Simulation output (.zarr)",
        "figures/      — All saved figures",
        "results/      — Connectivity metrics and network results",
    ]

    if custom_path is None:
        # Interactive: ask the user
        print("\n  1. Create a new ComCon project folder (recommended)")
        print("  2. Use your own directory\n")

        choice = input("Select option (1/2) [default: 1]: ").strip() or '1'
    else:
        choice = '2'

    if choice == '2':
        # Custom directory
        if custom_path is None:
            custom_path = input("\nEnter the path to your directory: ").strip()

        if not custom_path:
            print("No path provided. Using default project setup instead.")
            choice = '1'
        else:
            os.makedirs(custom_path, exist_ok=True)

            create_subs = input(
                "\nCreate ComCon subfolders inside this directory? (yes/no) [default: yes]: "
            ).strip().lower() or 'yes'

            if create_subs in ['yes', 'y']:
                root = custom_path
                subdirs = {'root': root}
                for name in subfolder_names:
                    subdirs[name] = os.path.join(root, name)

                for path in subdirs.values():
                    os.makedirs(path, exist_ok=True)

                # Display folder structure
                folder_display = os.path.basename(root)
                display_lines = [
                    f"Project directory created: {root}",
                    "",
                    "Folder structure:",
                    f"  {folder_display}/",
                ]
                for i, desc in enumerate(subfolder_descriptions):
                    prefix = "  └── " if i == len(subfolder_descriptions) - 1 else "  ├── "
                    display_lines.append(f"{prefix}{desc}")

                info_box(display_lines)
                return subdirs
            else:
                # All outputs go to the same directory (flat structure)
                subdirs = {key: custom_path for key in ['root'] + subfolder_names}
                print(f"\n  All outputs will be saved to: {custom_path}")
                display_line("─")
                return subdirs

    # Standard project setup (choice == '1' or fallback)
    if project_name is None:
        default_name = "ComCon_Project"
        project_name = input(
            f"\nEnter a name for this project/simulation [default: {default_name}]: "
        ).strip() or default_name

    safe_name = re.sub(r'[^\w\-]', '_', project_name)
    root = os.path.join(".", "projects", safe_name)

    subdirs = {'root': root}
    for name in subfolder_names:
        subdirs[name] = os.path.join(root, name)

    for path in subdirs.values():
        os.makedirs(path, exist_ok=True)

    info_box([
        f"Project directory created: {root}",
        "",
        "Folder structure:",
        f"  {safe_name}/",
        f"  ├── config/       — Saved parameter files (.json)",
        f"  ├── data/         — Exported hydrodynamic data",
        f"  ├── occurrence/   — Cleaned species occurrence data",
        f"  ├── grids/        — Release and settlement grids",
        f"  ├── simulation/   — Simulation output (.zarr)",
        f"  ├── figures/      — All saved figures",
        f"  └── results/      — Connectivity metrics and network results",
    ])

    return subdirs
# endregion

# region Downloading Species Occurrence Data
## region Set Up Functions
### Edit Species Parameters
def edit_spp_params(spp_params):
    """
    Interactively edits species occurrence data collection parameters.

    Allows modification of parameters controlling species distribution data download
    from GBIF, including study area definition, data cleaning options and output
    file settings.

    Parameters
    ----------
    spp_params : dict
        Dictionary containing species occurrence parameters

    Returns
    -------
    dict
        Updated species occurrence parameters with user modifications
    """
    # Iterate over each parameter in the dictionary
    for key, value in spp_params.items():
        # Skip csv_file_path if save_csv is False or 'no'
        if key == 'csv_file_path':
            if not spp_params.get('save_csv') or str(spp_params.get('save_csv')).lower() == 'no':
                continue

        # Skip user_data_file_path if data_source is 'gbif'
        if key == 'user_data_file_path':
            if spp_params.get('data_source') == 'gbif':
                continue
            save_csv_value = spp_params.get('save_csv', False)
            # Check if save_csv is False, 'no', or 'n'
            if save_csv_value in [False, 'no', 'n', 'No', 'NO']:
                continue  # Skip asking about csv_file_path

        # Display boolean values as yes/no instead of True/False
        display_value = value
        if isinstance(value, bool):
            display_value = 'yes' if value else 'no'

        edit = input(f"Do you want to edit {key} (current value: {display_value})? (yes/no): ").strip().lower()
        if edit in ["yes", "y"]:
            # Special handling for boolean parameters
            if isinstance(value, bool):
                new_value_input = input(
                    f"Enter the new value for {key} (yes/no, current: {'yes' if value else 'no'}): ").strip().lower()
                new_value = new_value_input in ['yes', 'y']

                # If changing save_csv to True, now ask for csv_file_path
                if key == 'save_csv' and new_value and not value:
                    csv_path = input("Enter the output path and file name for CSV: ").strip()
                    if not csv_path.endswith(".csv"):
                        csv_path += ".csv"
                    spp_params['csv_file_path'] = csv_path
                # If changing save_csv to False, clear csv_file_path
                elif key == 'save_csv' and not new_value:
                    spp_params['csv_file_path'] = None
            # Collect new input for other parameter types
            elif key == 'geojson_file_path':
                # Special handling for geojson - offer redraw option
                print("\nOptions for updating study area:")
                print("  1. Enter path to existing GeoJSON file")
                print("  2. Draw new study area on map")
                while True:
                    geojson_choice = input("Choose option (1/2): ").strip()
                    if geojson_choice == '1':
                        new_value = input("Enter the file path to GeoJSON: ").strip()
                        if not new_value.endswith(".geojson"):
                            new_value += ".geojson"
                        if not os.path.exists(new_value):
                            print(f"Warning: File not found: {new_value}")
                            retry = input("Try again? (yes/no): ").strip().lower()
                            if retry in ['yes', 'y']:
                                continue
                        break
                    elif geojson_choice == '2':
                        print("\nLaunching map interface...")
                        new_value = create_map()
                        break
                    else:
                        print("Invalid choice. Enter 1 or 2.")
            elif isinstance(value, list) and len(value) == 2:
                # For list-type values like longitude/latitude ranges
                new_value = [
                    input(f"Enter the new first value for {key} (current value: {value[0]}): "),
                    input(f"Enter the new second value for {key} (current value: {value[1]}): ")
                ]
            elif key == 'data_source':
                print(f"\nCurrent value: {value}")
                print("Options:")
                print("'gbif' = Download from GBIF database")
                print("'user' = Upload your own occurrence data")
                while True:
                    new_value = input("Enter 'gbif' or 'user': ").strip().lower()
                    if new_value in ['gbif', 'user']:
                        break
                    print("Invalid choice. Enter 'gbif' or 'user'.")
            elif key == 'habitat_filtering':
                print(f"\nCurrent value: {value}")
                print("Options:")
                print("  'coastal' = Coastal (littoral/intertidal species)")
                print("  'ocean' = Ocean (coastal + offshore)")
                print("  'none' = None (use data as-is)")
                while True:
                    new_value = input("Enter 'coastal', 'ocean', or 'none': ").strip().lower()
                    if new_value in ['coastal', 'ocean', 'none']:
                        break
                    print("Invalid choice. Enter 'coastal', 'ocean', or 'none'.")
            else:
                new_value = input(f"Enter the new value for {key} (current value: {value}): ").strip()

            # Update the parameter in the dictionary
            spp_params[key] = new_value

    return spp_params

### Collect Species Occurrence Parameters
def collect_spp_occ_params(species_traits_path):
    """
    Collects parameters for species occurrence data download and processing.

    Interactively gathers all necessary parameters for downloading and filtering
    species occurrence records from GBIF, including options for data sources,
    habitat filtering and output formats.

    Parameters
    ----------
    species_traits_path : str
        Path to CSV file containing species trait data. This file must include
        a 'species_name' column with scientific names for GBIF queries.

    Returns
    -------
    dict
        Dictionary containing species occurrence parameters:
        - 'species_traits_path' : str
            Path to input traits CSV
        - 'geojson_file_path' : str or None
            Path to GeoJSON polygon file (created later in workflow)
        - 'save_csv' : bool
            Whether to save cleaned occurrence data
        - 'csv_file_path' : str or None
            Output CSV file path if save_csv is True
    """
    subsection_header("Select your study area")
    info_box([
        "This package requires a polygon of your study area in GeoJSON file format.",
        "",
        "You can either upload your own file or use the package functionality to select your study area on a map."
    ])
    geojson_reuse = input("\nDo you want to use an existing GeoJSON file? (yes/no): ").strip().lower()

    if geojson_reuse in ["yes", "y"]:
        # Prompt for GeoJSON file path, adding the .geojson extension if not provided
        geojson_file_path = get_valid_path("Enter the GeoJSON file path", must_exist=True, extensions=['.geojson', '.json'])
        if not geojson_file_path.endswith(".geojson"):
            geojson_file_path += ".geojson"
    else:
        geojson_file_path = None

    # Check if the user wants to save the results as a CSV file
    save_csv = input("\nDo you want to save the species occurrence data as a CSV file? (yes/no): ").strip().lower() == 'yes'
    if save_csv:
        # Prompt for CSV file save path, adding the .csv extension if not provided
        csv_file_path = input("Enter the output path and file name: ").strip()
        if not csv_file_path.endswith(".csv"):
            csv_file_path += ".csv"
    else:
        csv_file_path = None

    return {
        'species_traits_path': species_traits_path,
        'geojson_file_path': geojson_file_path,
        'save_csv': save_csv,
        'csv_file_path': csv_file_path
    }

### Create Map
def create_map():
    """
    Creates an interactive map for defining the study area polygon.

    Launches an interactive Folium map in Jupyter lab allowing users to draw
    a polygon defining their study area. The polygon is saved as a GeoJSON file
    for use in filtering species occurrence and hydrodynamic data.

    Returns
    -------
    str
        File path to the saved GeoJSON file containing the drawn polygon

    Notes
    -----
    - This function must be run in a Jupyter lab environment
    - Users can draw only one polygon per session
    - The draw control allows polygon drawing and editing
    - The polygon coordinates are extracted from the map and saved as GeoJSON
    """
    # Step 1: Provide instructions to the user
    print("\nFollow these steps to define your study area:")
    print("  1. Use the drawing tool on the map to draw a polygon around your area of interest.")
    print("  2. After drawing, click the 'Export' button on the map to save the polygon as a GeoJSON file.")
    print("  3. Once you have saved the file, return here and enter the full path to your saved GeoJSON file below.")

    # Step 2: Create the interactive map with drawing capabilities
    m = folium.Map()
    draw = plugins.Draw(
        export=True,
        draw_options={'polygon': True, 'polyline': False, 'rectangle': False, 'circle': False, 'marker': False,
                      'circlemarker': False}
    )
    draw.add_to(m)
    display(m)

    # Step 3: Prompt user for the file path and ensure the file exists
    print("  Note: File extensions are added automatically if not included.")
    file_path = get_valid_path("Enter the input path and file name", must_exist=True, extensions=['.geojson', '.json'])
    if not file_path.endswith(".geojson"):
        file_path += ".geojson"

    # Step 4: Return the valid file path
    return file_path

### Read GeoJSON File
def read_geojson(geojson_file_path):
    """
        Reads and validates GeoJSON file containing the study area polygon.

        Loads a GeoJSON file, extracts the polygon geometry, and validates that it
        contains the expected structure for use in spatial filtering operations.

        Parameters
        ----------
        geojson_file_path : str
            Path to GeoJSON file containing a single polygon feature

        Returns
        -------
        dict
            Dictionary containing polygon geometry in GeoJSON format with structure:
            {
                'type': 'Polygon',
                'coordinates': [[[lon1, lat1], [lon2, lat2], ...]]
            }

        Raises
        ------
        FileNotFoundError
            If the specified GeoJSON file does not exist
        ValueError
            If the GeoJSON file is empty or improperly formatted
        KeyError
            If expected GeoJSON structure is not found

        Notes
        -----
        - The function expects standard GeoJSON format with 'features' array
        - Coordinates should be in [longitude, latitude] order
        - The polygon should be closed (first and last coordinates identical)
        """
    with open(geojson_file_path, 'r') as geojson_file:
        geojson_data = json.load(geojson_file)

    # Extract polygon coordinates
    polygon_coords = geojson_data['features'][0]['geometry']['coordinates'][0]

    # Ensure polygon is closed
    if polygon_coords[0] != polygon_coords[-1]:
        polygon_coords.append(polygon_coords[0])

    # Create Shapely Polygon object
    shapely_polygon = Polygon(polygon_coords)

    # Ensure counter-clockwise winding order (required by GBIF API)
    from shapely.geometry import polygon as shapely_polygon_mod
    shapely_polygon = shapely_polygon_mod.orient(shapely_polygon, sign=1.0)
    # Update polygon_coords to match the oriented polygon
    polygon_coords = list(shapely_polygon.exterior.coords)

    # Create WKT format string
    geometry = 'POLYGON((' + ', '.join([f'{round(lon, 4)} {round(lat, 4)}' for lon, lat in polygon_coords]) + '))'

    return geometry, shapely_polygon

###
def get_aphia(aphia_id):
    """
    Helper function to get current accepted species name from AphiaID.

    Parameters
    ----------
    aphia_id : int
        WoRMS AphiaID for the species

    Returns
    -------
    str
        Current accepted scientific name

    Raises
    ------
    ValueError
        If AphiaID is invalid or not found in WoRMS
    """
    try:
        record = pyworms.aphiaRecordByAphiaID(aphia_id)
        if record is None:
            raise ValueError(f"AphiaID {aphia_id} not found in WoRMS")

        # Use accepted name if this is a synonym, otherwise use the name from record
        if record.get('status') == 'accepted':
            return record['scientificname']
        elif 'valid_name' in record:
            return record['valid_name']
        else:
            return record['scientificname']

    except Exception as e:
        raise ValueError(f"Error retrieving species name for AphiaID {aphia_id}: {str(e)}")

### Get Species Occurrences
def get_spp_occ(species_name, aphia_id, geometry, limit=300):
    """
    Downloads species occurrence records from GBIF within defined study area.

    Queries the Global Biodiversity Information Facility (GBIF) database for
    occurrence records of a specified species within a defined geographic area.
    Implements pagination to retrieve large datasets and basic data cleaning.
    Species can be specified using either scientific name or WoRMS AphiaID.

    Parameters
    ----------
    species_name : str, optional
        Scientific name of the species (e.g., 'Nucella lapillus').
        Provide either species_name OR aphia_id (not both)
    aphia_id : int, optional
        WoRMS AphiaID for the species. Recommended for marine species as it
        handles taxonomic updates and synonyms automatically.
        Provide either species_name OR aphia_id (not both)
    geometry : dict
        GeoJSON polygon geometry defining the study area bounds. Must contain
        'type' and 'coordinates' keys in standard GeoJSON format
    limit : int, optional
        Maximum number of records to retrieve per GBIF API call (default: 300).
        Function automatically paginates to retrieve all available records

    Returns
    -------
    list of dict
        List of occurrence records, where each record is a dictionary containing:
        - 'species' : str
            Scientific name of the species
        - 'latitude' : float
            Decimal latitude of occurrence
        - 'longitude' : float
            Decimal longitude of occurrence

    Raises
    ------
    ValueError
        If neither or both species_name and aphia_id are provided
        If aphia_id is invalid or not found in WoRMS

    Notes
    -----
    - Using AphiaID (recommended for marine species):
        - Automatically retrieves current accepted name from WoRMS
        - Handles taxonomic updates and synonyms
        - More robust for long-term studies
    - Using species_name:
        - Direct query to GBIF
        - May miss records if name has changed or has synonyms
    - Free GBIF and WoRMS access requires no authentication
    - Large queries may take several minutes
    - Records are automatically filtered to study area during download
    - Returns only georeferenced records (records with valid coordinates)
    - Pagination continues until all records are retrieved

    API Details
    -----------
    - Uses pygbif Python package to interface with GBIF API
    - Default limit of 300 records per call is GBIF's maximum
    - Function handles GBIF's offset-based pagination automatically

    Data Quality Considerations
    ---------------------------
    - GBIF data includes records from various sources with varying quality
    - Some records may have coordinate uncertainty
    - Duplicate records possible across different datasets
    - Further cleaning recommended (see clean_spp_occ function)
    """
    # Validation: at least one identifier must be provided
    if species_name is None and aphia_id is None:
        raise ValueError("Must provide either species_name or aphia_id")

    # If both provided, prioritize AphiaID (as documented in user instructions)
    if aphia_id is not None:
        species_name = get_aphia(aphia_id)
        print(f"AphiaID {aphia_id} resolved to: {species_name}")
    # If only species_name provided, use it directly (species_name already set)

    all_records = []
    offset = 0
    max_retries = 2

    for attempt in range(max_retries + 1):
        try:
            while True:
                results = occurrences.search(scientificName=species_name, geometry=geometry, limit=limit, offset=offset)
                for record in results.get('results', []):
                    if 'decimalLatitude' in record and 'decimalLongitude' in record:
                        all_records.append({
                            'species': species_name.replace(' ', '_'),
                            'latitude': record['decimalLatitude'],
                            'longitude': record['decimalLongitude']
                        })
                offset += limit
                if len(results.get('results', [])) < limit:
                    break
                time.sleep(0.5)
            break  # Success — exit retry loop

        except Exception as e:
            if attempt < max_retries:
                print(f"    GBIF request failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                print(f"    Retrying in 3 seconds...")
                time.sleep(3)
                offset = 0
                all_records = []
            else:
                print(f"    GBIF request failed after {max_retries + 1} attempts: {e}")
                print(f"    Possible causes:")
                print(f"      - GBIF server may be temporarily unavailable")
                print(f"      - Study area polygon may have an issue")
                print(f"      - Network/firewall may be blocking the request")
                print(f"    Try: redraw your study area polygon or try again later.")

    return all_records

### Download all species records from within the study area
def get_all_spp_occ(geometry, limit=300):
    """
    Downloads occurrence records for all species within the study area.

    Useful for exploratory analysis to discover what species exist in your
    study area before selecting focal species for connectivity modelling.

    Parameters
    ----------
    geometry : dict
        GeoJSON polygon geometry defining the study area bounds. Must contain
        'type' and 'coordinates' keys in standard GeoJSON format
    limit : int, optional
        Maximum number of records to retrieve per GBIF API call (default: 300).
        Function automatically paginates to retrieve all available records

    Returns
    -------
    list of dict
        List of occurrence records, where each record is a dictionary containing:
        - 'species' : str
            Scientific name of the species
        - 'latitude' : float
            Decimal latitude of occurrence
        - 'longitude' : float
            Decimal longitude of occurrence

    Notes
    -----
    - This function is not recommended for connectivity modelling due to data volume and missing traits
    - Use this for exploratory analysis only to identify candidate species
    - After identifying species of interest, use get_spp_occ() for targeted downloads
    - Downloads are paginated in chunks of 300 records (GBIF API limit)
    """
    all_records = []
    offset = 0

    print("\nQuerying GBIF for all species in study area...")
    print("Note: This may take some time for large areas\n")

    while True:
        try:
            results = occurrences.search(geometry=geometry, limit=limit, offset=offset)
        except Exception as e:
            print(f"\nError communicating with GBIF: {e}")
            print("This is likely a temporary GBIF server issue.")
            print("Please try again in a few minutes.")
            return all_records  # Return what we have

        # Check if we got results
        if not results or not results.get('results'):
            break

        # Process results
        for record in results['results']:
            if 'decimalLatitude' in record and 'decimalLongitude' in record:
                species_name = record.get('scientificName', 'Unknown')
                all_records.append({
                    'species': species_name,
                    'latitude': record['decimalLatitude'],
                    'longitude': record['decimalLongitude']
                })

        # Progress update
        if len(all_records) > 0 and len(all_records) % 1000 == 0:
            print(f"\nRetrieved {len(all_records):} records...")

        # Check if we got fewer results than requested (last page)
        if len(results['results']) < limit:
            break

        offset += limit
        time.sleep(0.5)  # Be nice to GBIF

    if len(all_records) > 0:
        print(f"\nRetrieved {len(all_records):} total occurrence records")
    else:
        print(f"\nRetrieved 0 records. GBIF may be having issues or no data in this area")

    return all_records

### Create map of hydrodynamic data extent and masks
def plot_masks(landmask, coastalmask, lonarray, latarray, geojson_file_path=None):
    """
    Visualise land/coastal/ocean classification from hydrodynamic data.

    Shows immediate validation of downloaded hydrodynamic data before proceeding
    with species data collection. Users can verify the domain extent, grid
    resolution, and coastal zone definition.

    Parameters
    ----------
    landmask : np.ndarray
        2D binary array (1=land, 0=ocean)
    coastalmask : np.ndarray
        2D binary array (1=coastal, 0=non-coastal)
    lonarray : np.ndarray
        2D meshgrid of longitudes
    latarray : np.ndarray
        2D meshgrid of latitudes
    geojson_file_path : str, optional
        Path to study area GeoJSON file. If provided, study area outline will be displayed.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure

    Notes
    -----
    - Shows full hydrodynamic domain extent
    - Displays study area outline if provided
    - Provides quick validation before time-consuming species download
    """

    # Set high-quality rendering
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12

    # Colour scheme
    COLORS = {
        'land': 'lightgray',
        'coastal': '#DC143C',
        'ocean': '#0077BE',
        'border': '#FF4500'
    }

    # Calculate extent
    lon_min, lon_max = lonarray.min(), lonarray.max()
    lat_min, lat_max = latarray.min(), latarray.max()

    # Calculate dynamic figure size (same approach as species map)
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    aspect_ratio = lon_range / lat_range

    base_width = min(max(lon_range * 1.5, 12), 20)
    map_height = base_width / aspect_ratio

    # Create figure with GridSpec from the start
    fig = plt.figure(figsize=(base_width, map_height + 3))
    gs = GridSpec(2, 1, height_ratios=[map_height, 2.5], hspace=0.15, figure=fig)

    # Map subplot
    ax = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())

    # Set map extent to match data
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Basemap features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='black', zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.4, zorder=2)

    # Create mask arrays
    coastal_points = (landmask == 0) & (coastalmask == 1)
    ocean_points = (landmask == 0) & (coastalmask == 0)
    land_points = (landmask == 1)

    # Count cells
    n_coastal = np.sum(coastal_points)
    n_ocean = np.sum(ocean_points)
    n_land = np.sum(land_points)
    n_total = n_coastal + n_ocean + n_land

    # Plot points
    ax.scatter(lonarray[ocean_points], latarray[ocean_points],
               c=COLORS['ocean'], s=20, alpha=0.6, marker='o',
               edgecolors='black', linewidths=0.2,
               transform=ccrs.PlateCarree(), zorder=3)

    ax.scatter(lonarray[coastal_points], latarray[coastal_points],
               c=COLORS['coastal'], s=20, alpha=0.6, marker='o',
               edgecolors='black', linewidths=0.2,
               transform=ccrs.PlateCarree(), zorder=4)

    ax.scatter(lonarray[land_points], latarray[land_points],
               c=COLORS['land'], s=20, alpha=0.6, marker='o',
               edgecolors='black', linewidths=0.2,
               transform=ccrs.PlateCarree(), zorder=2)

    # Add study area outline
    if geojson_file_path:
        try:
            _, shapely_polygon = read_geojson(geojson_file_path)
            if shapely_polygon:
                x, y = shapely_polygon.exterior.xy
                ax.plot(x, y, color=COLORS['border'], linewidth=3.5,
                        transform=ccrs.PlateCarree(), zorder=6,
                        linestyle='-', alpha=0.9)
        except Exception as e:
            print(f"Note: Could not display study area outline: {e}")

    # Calculate grid info
    grid_spacing_lon = np.abs(np.diff(lonarray[0, :]).mean())
    grid_spacing_lat = np.abs(np.diff(latarray[:, 0]).mean())
    grid_spacing_km = grid_spacing_lon * 111

    # Title
    ax.set_title('Hydrodynamic Extent',
                 fontsize=20, fontweight='bold', pad=12, family='Arial')

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.4,
                      linestyle='--', color='gray')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 14, 'family': 'Arial'}
    gl.ylabel_style = {'size': 14, 'family': 'Arial'}

    # ========================================
    # Footer: 4 columns (3 info boxes + legend)
    # ========================================
    footer_gs = gs[1].subgridspec(1, 4, wspace=0.15)

    # Box 1: Model Domain info
    box1_ax = fig.add_subplot(footer_gs[0])
    box1_ax.axis('off')
    box1_text = 'Model Domain\n'
    box1_text += f'Resolution: {grid_spacing_km:.1f} km\n'
    box1_text += f'({grid_spacing_lon:.3f}° × {grid_spacing_lat:.3f}°)'
    box1_ax.text(0.5, 0.5, box1_text,
                 fontsize=12, family='Arial', va='center', ha='center',
                 transform=box1_ax.transAxes,
                 bbox=dict(boxstyle='round', facecolor='white',
                           alpha=0.95, edgecolor='#CCCCCC', linewidth=1.5))

    # Box 2: Extent info
    box2_ax = fig.add_subplot(footer_gs[1])
    box2_ax.axis('off')
    box2_text = 'Extent\n'
    box2_text += f'Lon: {lon_min:.2f}° to {lon_max:.2f}°\n'
    box2_text += f'Lat: {lat_min:.2f}° to {lat_max:.2f}°'
    box2_ax.text(0.5, 0.5, box2_text,
                 fontsize=12, family='Arial', va='center', ha='center',
                 transform=box2_ax.transAxes,
                 bbox=dict(boxstyle='round', facecolor='white',
                           alpha=0.95, edgecolor='#CCCCCC', linewidth=1.5))

    # Box 3: Grid Classification
    box3_ax = fig.add_subplot(footer_gs[2])
    box3_ax.axis('off')
    box3_text = 'Grid Classification\n'
    box3_text += f'Coastal: {n_coastal} ({100 * n_coastal / n_total:.1f}%)\n'
    box3_text += f'Ocean: {n_ocean} ({100 * n_ocean / n_total:.1f}%)\n'
    box3_text += f'Land: {n_land} ({100 * n_land / n_total:.1f}%)'
    box3_ax.text(0.5, 0.5, box3_text,
                 fontsize=12, family='Arial', va='center', ha='center',
                 transform=box3_ax.transAxes,
                 bbox=dict(boxstyle='round', facecolor='white',
                           alpha=0.95, edgecolor='#CCCCCC', linewidth=1.5))

    # Box 4: Legend
    legend_ax = fig.add_subplot(footer_gs[3])
    legend_ax.axis('off')

    legend_elements = [
        Patch(facecolor=COLORS['ocean'], alpha=0.6, edgecolor='black',
              label=f'Ocean ({n_ocean} cells)'),
        Patch(facecolor=COLORS['coastal'], alpha=0.6, edgecolor='black',
              label=f'Coastal ({n_coastal} cells)'),
        Patch(facecolor=COLORS['land'], alpha=0.6, edgecolor='black',
              label=f'Land ({n_land} cells)'),
    ]

    legend_ax.legend(handles=legend_elements, loc='center',
                     fontsize=11, frameon=True, edgecolor='#CCCCCC',
                     framealpha=0.95)

    plt.tight_layout()
    plt.show()

    # Ask if user wants to save
    save_map = input("\nSave domain map? (yes/no): ").strip().lower()
    if save_map in ['yes', 'y']:
        default_dir = get_default_dir('figures')
        save_dir = input(f"Enter file path [default: {default_dir}]: ").strip() or default_dir
        filename = input("Enter the filename [default: dispersal_domain]: ").strip() or 'dispersal_domain'
        file_format = input("Select the format (tif/png/pdf/svg) [default: tif]: ").strip().lower() or 'tif'

        if file_format not in ['tif', 'png', 'pdf', 'svg']:
            print(f"Invalid format '{file_format}', using tif")
            file_format = 'tif'

        if not filename.endswith(f'.{file_format}'):
            filename = f'{filename}.{file_format}'

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved to: {save_path}")

    return fig

### Create Coastalmask
def make_coastalmask(landmask):
    """
    Generate coastal mask from land mask.

    Identifies coastal ocean cells as those adjacent to land cells using binary
    dilation. This is a utility function called by create_masks but can also
    be used independently to regenerate coastal masks from existing land masks.

    Parameters
    ----------
    landmask : numpy.ndarray
        2D binary array where 1 indicates land and 0 indicates ocean

    Returns
    -------
    numpy.ndarray
        2D binary array where 1 indicates coastal ocean cells (ocean cells
        adjacent to land) and 0 indicates non-coastal cells (either inland or
        open ocean)

    Notes
    -----
    - Uses scipy.ndimage.binary_dilation with default structure
    - Considers 4-connectivity (N, S, E, W) for adjacency
    - Coastal mask excludes land cells themselves

    Algorithm
    ---------
    1. Dilate land mask by one pixel in all directions
    2. Subtract original land mask from dilated version
    3. Result contains only cells that were ocean but border land
    """
    coastal_mask = binary_dilation(landmask, structure=np.ones((3, 3))) & ~landmask

    return coastal_mask.astype(int)

### create Landmask
def create_masks(hydro_data, geojson_file_path=None):
    """
    Generate land and coastal masks from hydrodynamic data.

    Creates binary masks identifying land, ocean, and coastal regions from the
    downloaded hydrodynamic dataset. Coastal cells are defined as ocean cells
    adjacent to land. These masks are essential for particle release locations
    and coastal connectivity analysis.

    Parameters
    ----------
    hydro_data : xarray.Dataset
        Hydrodynamic data loaded from NetCDF file containing:
        - 'uo' : DataArray
            U-velocity component (eastward)
        - 'vo' : DataArray
            V-velocity component (northward)
        - 'latitude' : DataArray
            Latitude coordinates
        - 'longitude' : DataArray
            Longitude coordinates
        NaN values indicate land cells

    Returns
    -------
    landmask : numpy.ndarray
        2D binary array where 1=land, 0=ocean. Shape matches spatial dimensions
        of hydrodynamic data
    coastalmask : numpy.ndarray
        2D binary array where 1=coastal ocean cell, 0=non-coastal. Coastal cells
        are ocean cells adjacent to at least one land cell
    lonarray : numpy.ndarray
        2D array of longitude values matching hydrodynamic grid
    latarray : numpy.ndarray
        2D array of latitude values matching hydrodynamic grid

    Notes
    -----
    - Land cells are identified by NaN values in velocity fields
    - Coastal cells are ocean cells with at least one adjacent land cell
    - Adjacency is defined using 4-connectivity (N, S, E, W neighbours)
    - Edge cells of the domain are not considered coastal
    - Arrays are 2D with dimensions (latitude, longitude)

    Algorithm Details
    -----------------
    1. Extract velocity data from first time step
    2. Identify land cells where both U and V are NaN
    3. Create landmask (1=land, 0=ocean)
    4. Apply binary dilation to find cells adjacent to land
    5. Coastal mask = dilated land mask XOR original land mask
    6. Generate meshgrid arrays for coordinates
    """

    # Check the number of dimensions in the 'uo' variable
    var = hydro_data.variables["uo"]

    # Determine the number of dimensions
    num_dims = len(var.shape)

    if num_dims == 3:
        # 2D case (e.g., (lat, lon, time))
        landmask = var[0, :, :]

    elif num_dims == 4:
        # 3D case (e.g., (time, lat, lon, depth))
        # Here you can choose which slice to use. For example, using the first time step:
        landmask = var[0, 0, :, :]

    else:
        raise ValueError("Unsupported data dimensionality: {}".format(num_dims))

    # Mask invalid data in the 'uo' variable
    landmask = np.ma.masked_invalid(landmask)

    # Convert the mask to an integer array: 1 for valid land, 0 for invalid (ocean)
    landmask = landmask.mask.astype("int")

    # Generate the coastal mask
    coastalmask = make_coastalmask(landmask)

    # Handle both naming conventions (lat/latitude, lon/longitude)
    if 'lat' in hydro_data.coords:
        latitudes = hydro_data.coords['lat'].values
    else:
        latitudes = hydro_data.coords['latitude'].values

    if 'lon' in hydro_data.coords:
        longitudes = hydro_data.coords['lon'].values
    else:
        longitudes = hydro_data.coords['longitude'].values

    # Create 2D meshgrid from latitudes and longitudes
    lonarray, latarray = np.meshgrid(longitudes, latitudes)

    # Visualise masks for immediate validation
    subsection_header("Visualising the Hydrodynamic Data")
    plot_masks(landmask, coastalmask, lonarray, latarray, geojson_file_path)
    display_line("─")

    return landmask, coastalmask, lonarray, latarray

### Clean Species Data
def clean_spp_occ(distributions, landmask, coastalmask, lonarray, latarray, buffer_cells=1, habitat_type='coastal'):
    """
    Clean species occurrence data by filtering based on habitat type.

    Filters species occurrence data to retain only points in desired habitat zones.
    Provides detailed feedback about filtering parameters and results.

    Parameters
    ----------
    distributions : pd.DataFrame
        Species occurrence data with 'latitude' and 'longitude' columns
    landmask : np.ndarray
        2D array indicating land (1) and ocean (0)
    coastalmask : np.ndarray
        2D array indicating coastal regions (1) and non-coastal (0)
    lonarray : np.ndarray
        2D array of longitude grid points
    latarray : np.ndarray
        2D array of latitude grid points
    buffer_cells : int, optional
        Number of grid cells to extend the coastal buffer (default: 1)
    habitat_type : str, optional
        Type of habitat filtering to apply (default: 'coastal')
        Options:
        - 'coastal': Keep only species in coastal zones (for coastal/littoral species)
        - 'ocean': Keep all ocean species, remove only land (for offshore or mixed settings)
        - 'none': No filtering, return all data (for pre-cleaned data)

    Returns
    -------
    pd.DataFrame
        Cleaned species occurrences after habitat filtering

    Notes
    -----
    - Coastal filtering uses a buffer to avoid over-aggressive filtering
    - Ocean filtering is more permissive, removing only land points
    - 'none' option is useful when data is already cleaned or pre-processed
    """

    # Extract latitude and longitude from the distributions dataset
    species_lats = distributions['latitude'].values
    species_lons = distributions['longitude'].values

    # Convert species coordinates to grid indices
    # Handle both ascending and descending coordinate arrays
    lat_vals = latarray[:, 0]
    lon_vals = lonarray[0, :]

    # For latitude: check if descending (typical for geographic data)
    if lat_vals[0] > lat_vals[-1]:  # Descending
        lat_indices = np.digitize(-species_lats, -lat_vals) - 1
    else:  # Ascending
        lat_indices = np.digitize(species_lats, lat_vals) - 1

    # For longitude: check if descending (rare but possible)
    if lon_vals[0] > lon_vals[-1]:  # Descending
        lon_indices = np.digitize(-species_lons, -lon_vals) - 1
    else:  # Ascending
        lon_indices = np.digitize(species_lons, lon_vals) - 1

    # Ensure indices are within bounds
    lat_indices = np.clip(lat_indices, 0, landmask.shape[0] - 1)
    lon_indices = np.clip(lon_indices, 0, landmask.shape[1] - 1)

    # Calculate grid spacing for user feedback
    grid_spacing = np.abs(np.diff(lonarray[0, :]).mean()) * 111  # Convert to km

    # Create filtering mask based on habitat type
    if habitat_type == 'ocean':
        # Keep all ocean points and coastal land (intertidal)
        ocean_cells = (landmask == 0)

        # Land cells adjacent to ocean
        coastal_land = (landmask == 1) & (binary_dilation(landmask == 0, structure=np.ones((3, 3))))

        # Combined: all ocean and coastal land
        valid_cells = ocean_cells | coastal_land

        keep_mask = np.array([
            valid_cells[lat_idx, lon_idx] == 1
            for lat_idx, lon_idx in zip(lat_indices, lon_indices)
        ])

        # Print filtering information
        subsection_header("Ocean Filtering")
        info_box([
            f"Model resolution: {grid_spacing:.1f} km",
            f"Coastal land buffer: {grid_spacing:.1f} km landward (for intertidal records)",
            "Retains: All ocean habitat (coastal and offshore)",
            "Removes: Only inland occurrences"
        ])

    elif habitat_type == 'coastal':
        # Coastal zone = ocean cells adjacent to land
        coastal_cells = (landmask == 0) & (coastalmask == 1)

        # Apply buffer expansion in both directions (ocean and land)
        buffered_coastal = np.pad(coastal_cells, pad_width=buffer_cells,
                                  mode='constant', constant_values=0)
        for _ in range(buffer_cells):
            buffered_coastal = np.maximum(buffered_coastal,
                                          np.roll(buffered_coastal, shift=1, axis=0))
            buffered_coastal = np.maximum(buffered_coastal,
                                          np.roll(buffered_coastal, shift=-1, axis=0))
            buffered_coastal = np.maximum(buffered_coastal,
                                          np.roll(buffered_coastal, shift=1, axis=1))
            buffered_coastal = np.maximum(buffered_coastal,
                                          np.roll(buffered_coastal, shift=-1, axis=1))
        buffered_coastal = buffered_coastal[buffer_cells:-buffer_cells,
                           buffer_cells:-buffer_cells]

        # Filter: Keep points in buffered coastal zone (includes land and ocean)
        keep_mask = np.array([
            buffered_coastal[lat_idx, lon_idx] == 1
            for lat_idx, lon_idx in zip(lat_indices, lon_indices)
        ])

        # Calculate coastal zone extent
        coastal_width = grid_spacing * (1 + buffer_cells * 2)

        # Print filtering information
        subsection_header("Coastal Filtering")
        info_box([
            f"Model resolution: {grid_spacing:.1f} km",
            f"Coastal zone: ~{coastal_width:.1f} km total. Applies a {buffer_cells} cell buffer landward and seaward for intertidal zone.",
            f"Keeps species within ~{buffer_cells * 3}× grid spacing."
        ])

    elif habitat_type == 'none':
        # No filtering, keep all records
        keep_mask = np.ones(len(distributions), dtype=bool)

        # Print information
        subsection_header("No Filtering Applied")
        info_box([
            f"Model resolution: {grid_spacing:.1f} km",
            "Using data exactly as provided",
            "Ensure your data is pre-cleaned and validated"
        ])

    else:
        raise ValueError(f"Invalid habitat_type: '{habitat_type}'. "
                         "Must be 'coastal', 'ocean', or 'none'")

    # Apply the mask
    clean_spp = distributions[keep_mask].copy()

    # Print filtering results
    n_original = len(distributions)
    n_after = len(clean_spp)
    n_removed = n_original - n_after
    pct_removed = 100 * n_removed / n_original if n_original > 0 else 0

    data_box("Filtering Results", [
        f"Original occurrences: {n_original}",
        f"After filtering: {n_after}",
        f"Removed: {n_removed} ({pct_removed:.1f}%)"
    ])

    if n_after == 0:
        print("\nWarning: No occurrences remain after filtering!")
        print("Consider:")
        print("  - Expanding your study area")
        print("  - Checking if data matches study region")
        print("  - Checking if species is genuinely rare in this region")

    return clean_spp

### Validate GBIF Data Download
def validate_spp_occ(clean_spp, traits_df, min_records=10):
    """
    Validate species occurrence data for adequate sample sizes.

    After downloading and cleaning species data, this function checks whether
    sufficient occurrence records were found for each species in your traits file.
    This helps identify species with low sample sizes or missing data early in the workflow.

    Parameters
    ----------
    clean_spp : pd.DataFrame
        Cleaned species occurrence data with 'species' column
    traits_df : pd.DataFrame
        Species traits DataFrame with 'species_name' column
    min_records : int, optional
        Minimum number of records needed per species for adequate sample size (default: 10)

    Returns
    -------
    dict
        Validation report containing:
        - 'adequate_sample_species': List of species with sufficient records
        - 'low_sample_species': List of species below minimum threshold
        - 'missing_species': List of species with no records found
        - 'total_records': Dictionary mapping species to record counts
        - 'overall_success_rate': Percentage of species with adequate sample sizes

    Notes
    -----
    - All species downloads were successful if they appear in clean_spp
    - "Low sample" vs "adequate sample" refers to statistical reliability
    - Species with 0 records may have taxonomic name changes or be absent from region
    - Check GBIF directly to verify species names if many are missing
    - Low record counts may still work but reduce connectivity estimate reliability

    Recommendations
    ---------------
    - <80% adequate: Review species list before proceeding
    - Missing species: Check WoRMS for name updates, verify region
    - Low sample: Consider expanding study area or accepting uncertainty
    """

    subsection_header("Species Occurrence Data: Sample Size Validation")

    validation_report = {
        'adequate_sample_species': [],
        'low_sample_species': [],
        'missing_species': [],
        'total_records': {},
        'overall_success_rate': 0
    }

    # Get target species from traits (normalize to underscore format)
    target_species = traits_df['species_name'].tolist()

    # Get unique species in clean_spp for checking
    clean_spp_species = clean_spp['species'].unique().tolist() if len(clean_spp) > 0 else []

    # Check occurrences for each target species
    for species in target_species:
        # Normalize species name (handle both space and underscore formats)
        species_normalized = species.replace(' ', '_')

        # Try both formats when matching
        species_records = clean_spp[
            (clean_spp['species'] == species) |
            (clean_spp['species'] == species_normalized)
            ]
        record_count = len(species_records)

        validation_report['total_records'][species] = record_count

        if record_count == 0:
            validation_report['missing_species'].append(species)
            print(f"{species}: No records found")
        elif record_count < min_records:
            validation_report['low_sample_species'].append(species)
            print(f"{species}: {record_count} records (low sample size)")
        else:
            validation_report['adequate_sample_species'].append(species)
            print(f"{species}: {record_count} records")

    # Calculate success rate (adequate sample size)
    adequate_count = len(validation_report['adequate_sample_species'])
    total_species = len(target_species)
    validation_report['overall_success_rate'] = (adequate_count / total_species * 100
                                                 if total_species > 0 else 0)

    # Summary
    data_box("Summary", [
        f"Adequate sample size: {adequate_count}/{total_species} ({validation_report['overall_success_rate']:.1f}%)",
        f"Low sample size: {len(validation_report['low_sample_species'])}",
        f"No records found: {len(validation_report['missing_species'])}"
    ])

    # Detailed recommendations
    if validation_report['missing_species']:
        print(f"\nSpecies with no records found:")
        for species in validation_report['missing_species']:
            print(f"   - {species}")
        print("\nPossible causes:")
        print("  - Taxonomic name changed (check WoRMS for current accepted name)")
        print("  - Spelling variation (check GBIF for alternative names or use Aphia ID)")
        print("  - Species range doesn't overlap with study area")

    if validation_report['low_sample_species']:
        print(f"\n Species with low sample size:")
        for species in validation_report['low_sample_species']:
            count = validation_report['total_records'][species]
            print(f"   - {species}: {count} records (minimum recommended: {min_records})")
        print("\n Consider:")
        print("  - Expanding your study area")
        print("  - Checking if data matches study region")
        print("  - Checking if species is genuinely rare in this region")

    # Overall assessment
    if validation_report['overall_success_rate'] < 80:
        print(f"\nThe number of species with adequate sample sizes is below 80%")
        print("Recommendation: Review species list before proceeding")
    else:
        print(f"\nValidation Passed")
        print(f"{adequate_count} species have adequate sample sizes for analysis")

    return validation_report

## Create colours for mapping
def get_spp_colours(n_species):
    """
    Generate distinct colours for any number of species.

    Parameters:
    -----------
    n_species : int
        Number of species to generate colors for

    Returns:
    --------
    colors : list
        List of RGBA tuples, one per species
    """
    if n_species <= 10:
        # Use tab10 for up to 10 species (highly distinct)
        cmap = colormaps.get_cmap('tab10')
        return [cmap(i) for i in range(n_species)]

    elif n_species <= 20:
        # Use tab20 for 11-20 species (still distinct)
        cmap = colormaps.get_cmap('tab20')
        return [cmap(i) for i in range(n_species)]

    else:
        # For >20 species, combine multiple colormaps
        colors = []

        # Get colors from tab20 (first 20)
        tab20 = colormaps.get_cmap('tab20')
        colors.extend([tab20(i) for i in range(20)])

        # Add colors from Set3 (pastel, 12 colors)
        set3 = colormaps.get_cmap('Set3')
        colors.extend([set3(i) for i in range(12)])

        # Add colors from Paired (12 colors)
        paired = colormaps.get_cmap('Paired')
        colors.extend([paired(i) for i in range(12)])

        # If still need more (>44 species), use hsv for remainder
        if n_species > 44:
            hsv = colormaps.get_cmap('hsv')
            remaining = n_species - 44
            colors.extend([hsv(i / remaining) for i in range(remaining)])

        return colors[:n_species]

### Create Species Distribution Map
def plot_spp_occ(distributions, clean_spp, geojson_file_path, habitat_type, lonarray, latarray):
    """
    Create before/after species distribution comparison map with vertical stacking.

    Shows habitat filtering effects with vertically stacked before/after panels for better
    visibility. Automatically handles legend display and colour coding for any number of
    species (uses separate legend panel for 5+ species).

    Parameters
    ----------
    distributions : pd.DataFrame
        Original species data before filtering
    clean_spp : pd.DataFrame
        Cleaned species data after filtering
    geojson_file_path : str
        Path to study area GeoJSON file
    habitat_type : str
        Type of habitat filter applied ('coastal', 'ocean', 'none')
    lonarray : np.ndarray
        2D array of longitude values
    latarray : np.ndarray
        2D array of latitude values

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    def format_species_name(name):
        formatted = name.replace('_', ' ')
        parts = formatted.split(' ', 1)
        if len(parts) == 2:
            return f"$\\mathit{{{parts[0]}}}$ $\\mathit{{{parts[1]}}}$"
        return f"$\\mathit{{{formatted}}}$"

    # Styling
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = map_config['tick_label']['fontsize']
    TITLE_SIZE = map_config['title']['fontsize']
    LEGEND_SIZE = map_config['legend']['fontsize']

    COLORS = {
        'land': 'lightgray',
        'ocean': 'lightblue',
        'border': '#FF4500',
    }

    _, shapely_polygon = read_geojson(geojson_file_path)

    # Use study area bounds with increased buffer
    if shapely_polygon:
        bounds = shapely_polygon.bounds  # (minx, miny, maxx, maxy)
        lon_range_study = bounds[2] - bounds[0]
        lat_range_study = bounds[3] - bounds[1]
        # Increased buffer to 25% to prevent overlap with info box
        buffer_lon = lon_range_study * 0.25
        buffer_lat = lat_range_study * 0.25

        lon_min = bounds[0] - buffer_lon
        lon_max = bounds[2] + buffer_lon
        lat_min = bounds[1] - buffer_lat
        lat_max = bounds[3] + buffer_lat
    else:
        lon_min, lon_max = lonarray.min(), lonarray.max()
        lat_min, lat_max = latarray.min(), latarray.max()

    # Count species before and after
    species_list = distributions['species'].unique()
    n_species_before = len(species_list)
    n_species_after = len(clean_spp['species'].unique()) if len(clean_spp) > 0 else 0
    species_colors = get_spp_colours(n_species_before)

    # Dynamic sizing based on study area extent
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    aspect_ratio = lon_range / lat_range

    # Calculate figure dimensions. Ensure minimum size for readability
    panel_width = min(max(lon_range * 1.5, 10), 16)
    panel_height = max(panel_width / aspect_ratio, 6)

    # Calculate legend height based on species count
    # Determine number of columns for legend
    if n_species_before <= 3:
        legend_ncol = 1
    elif n_species_before <= 6:
        legend_ncol = 2
    elif n_species_before <= 12:
        legend_ncol = 3
    else:
        legend_ncol = 4

    # Calculate rows needed - cap for very large species counts
    legend_rows = (n_species_before + legend_ncol) // legend_ncol  # +1 for Study Area
    if n_species_before > 50:
        # For many species, limit legend height and rely on smaller font + more columns
        legend_height_ratio = min(0.35, max(0.15, 0.02 * legend_rows))
    else:
        legend_height_ratio = max(0.12, 0.04 * legend_rows)

    # Choose layout based on aspect ratio: side-by-side for tall areas, vertical for wide areas
    if aspect_ratio < 0.8:  # Tall/slim area - use side-by-side panels
        fig_width = (panel_width * 2) + 2
        fig_height = panel_height + 3 + (legend_rows * 0.3)
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = GridSpec(2, 2, height_ratios=[1, legend_height_ratio], width_ratios=[1, 1],
                      hspace=0.3, wspace=0.15, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
        legend_ax = fig.add_subplot(gs[1, :])
    else:  # Wide area - use vertical stacking
        fig_width = panel_width + 1
        fig_height = (panel_height * 2) + 3 + (legend_rows * 0.3)
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = GridSpec(3, 1, height_ratios=[1, 1, legend_height_ratio], hspace=0.25, figure=fig)
        ax1 = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
        ax2 = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())
        legend_ax = fig.add_subplot(gs[2])

        # Before panel
        ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax1.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none', zorder=1)
        ax1.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none', zorder=0)
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=2)
        ax1.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.4, zorder=2)

        if len(distributions) > 0:
            for i, species in enumerate(species_list):
                species_data = distributions[distributions['species'] == species]
                if len(species_data) > 0:
                    ax1.scatter(species_data['longitude'], species_data['latitude'],
                                s=50, c=[species_colors[i]], alpha=0.8,
                                edgecolors='black', linewidths=0.6,
                                transform=ccrs.PlateCarree(), zorder=3)

            ax1.text(0.02, 0.02, f'Before: {n_species_before} species, {len(distributions)} records',
                     transform=ax1.transAxes, fontsize=10, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
                               edgecolor='black', linewidth=0.8))

        if shapely_polygon:
            x, y = shapely_polygon.exterior.xy
            ax1.plot(x, y, color=COLORS['border'], linewidth=2.5,
                     transform=ccrs.PlateCarree(), zorder=5, alpha=0.9)

        gl1 = ax1.gridlines(draw_labels=True, linewidth=0.5, alpha=0.3,
                            linestyle='--', color='gray')
        gl1.top_labels = False
        gl1.right_labels = False
        gl1.xlabel_style = {'size': 9}
        gl1.ylabel_style = {'size': 9}
        ax1.set_title('Before Filtering', fontsize=12, fontweight='bold', pad=8)

        # After panel
        ax2.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax2.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none', zorder=1)
        ax2.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none', zorder=0)
        ax2.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=2)
        ax2.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.4, zorder=2)

        if len(clean_spp) > 0:
            species_in_clean = clean_spp['species'].unique()
            for i, species in enumerate(species_list):
                if species in species_in_clean:
                    species_data = clean_spp[clean_spp['species'] == species]
                    if len(species_data) > 0:
                        ax2.scatter(species_data['longitude'], species_data['latitude'],
                                    s=50, c=[species_colors[i]], alpha=0.85,
                                    edgecolors='black', linewidths=0.6,
                                    transform=ccrs.PlateCarree(), zorder=4)

            ax2.text(0.02, 0.02, f'After: {n_species_after} species, {len(clean_spp)} records',
                     transform=ax2.transAxes, fontsize=10, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
                               edgecolor='black', linewidth=0.8))

        if shapely_polygon:
            x, y = shapely_polygon.exterior.xy
            ax2.plot(x, y, color=COLORS['border'], linewidth=2.5,
                     transform=ccrs.PlateCarree(), zorder=5, alpha=0.9)

        gl2 = ax2.gridlines(draw_labels=True, linewidth=0.5, alpha=0.3,
                            linestyle='--', color='gray')
        gl2.top_labels = False
        gl2.right_labels = False
        gl2.xlabel_style = {'size': 9}
        gl2.ylabel_style = {'size': 9}
        ax2.set_title('After Filtering', fontsize=12, fontweight='bold', pad=8)

    # Bottom legend: Multi-column, scales with species count
    legend_elements = []
    for i, species in enumerate(species_list):
        if species in clean_spp['species'].unique():
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=species_colors[i],
                       markersize=10, markeredgecolor='black',
                       markeredgewidth=0.5,
                       label=format_species_name(species))
            )
    legend_elements.append(
        Line2D([0], [0], color=COLORS['border'], linewidth=2.5, label='Study Area')
    )

    # Hide the legend axes ticks/spines (removes the strange box)
    legend_ax.axis('off')

    # For many species, use smaller font and more columns
    if n_species_before > 50:
        legend_fontsize = 6
        legend_ncol = min(8, (n_species_before // 20) + 1)
    elif n_species_before > 20:
        legend_fontsize = 8
        legend_ncol = min(6, (n_species_before // 10) + 1)
    else:
        legend_fontsize = LEGEND_SIZE

    legend_ax.legend(handles=legend_elements, loc='center',
                     ncol=legend_ncol,
                     fontsize=legend_fontsize, frameon=True,
                     edgecolor='#CCCCCC', framealpha=1.0,
                     columnspacing=0.8, handletextpad=0.3)

    # Get filter label from habitat_type
    if habitat_type == 'coastal':
        filter_label = 'Coastal'
    elif habitat_type == 'ocean':
        filter_label = 'Ocean'
    else:
        filter_label = 'Unfiltered'

    fig.suptitle(f'Species Occurrence Data: {filter_label} Filtering Results',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    plt.show()

    save_map = input("\nSave species map? (yes/no): ").strip().lower()
    if save_map in ['yes', 'y']:
        default_dir = get_default_dir('figures')
        save_dir = input(f"Enter file path [default: {default_dir}]: ").strip() or default_dir
        filename = input("Enter filename [default: species_filtering]: ").strip() or 'species_filtering'
        file_format = input("Format (tif/png/pdf/svg) [default: tif]: ").strip().lower() or 'tif'

        if file_format not in ['tif', 'png', 'pdf', 'svg']:
            file_format = 'tif'
        if not filename.endswith(f'.{file_format}'):
            filename = f'{filename}.{file_format}'

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved to: {save_path}")

    return fig

### Add validation for uploaded data against selected study area
def validate_uploaded_data(distributions, shapely_polygon):
    """
    Validate uploaded species data falls within study area polygon.

    When users upload their own species occurrence data, this function ensures
    the data falls within the defined study area. Points outside the study area
    cannot be used for release site definition and will lead to incorrect mapping.

    The study area polygon defines where particles will be released and where
    settlement will be evaluated, not the full hydrodynamic extent (which is the
    larger hydrodynamic extent).

    Parameters:
    -----------
    distributions : pd.DataFrame
        Species occurrence data with 'latitude' and 'longitude' columns
    shapely_polygon : shapely.geometry.Polygon
        Study area boundary polygon

    Returns:
    --------
    pd.DataFrame
        Validated (and possibly filtered) occurrence data

    Raises:
    -------
    ValueError
        If user chooses to abort when data is outside study area
    """
    # Check each point
    within_bounds = []
    outside_bounds = []

    for idx, row in distributions.iterrows():
        point = Point(row['longitude'], row['latitude'])
        if shapely_polygon.contains(point):
            within_bounds.append(idx)
        else:
            outside_bounds.append(idx)

    # Report results
    print("Checking the geographic coverage...\n")

    print(f"Total records: {len(distributions):}")
    print(f"Within study area: {len(within_bounds):}")
    print(f"Outside study area: {len(outside_bounds):}")

    if len(outside_bounds) > 0:
        pct_outside = 100 * len(outside_bounds) / len(distributions)
        print(f"Percentage outside: {pct_outside:.1f}%")

        # Interactive: Ask user what to do
        print("\nWarning: Some data points are outside your selected study area.")
        print("\nPoints outside the study area will not be included for connectivity modelling and analysis.")
        print("\nWhat would you like to do?")
        print("  1. Automatically filter out points outside study area")
        print("  2. Keep all data (may cause incorrect results)")
        print("  3. Abort, redraw your study area polygon or fix the data manually")

        while True:
            choice = input("\nEnter your choice (1/2/3): ").strip()

            if choice == '1':
                print("\nFiltering data to study area...")
                filtered_data = distributions.iloc[within_bounds].copy()
                print(f"Filtered to {len(filtered_data):} records")
                return filtered_data

            elif choice == '2':
                print("\nWarning: Keeping all data. Results may be incorrect for points outside study area.")
                return distributions

            elif choice == '3':
                print("\nAborted. Please fix your data and try again.")
                print("Options to fix:")
                print("  - Expand your study area (GeoJSON polygon)")
                print("  - Filter your CSV to only include points within study area")
                raise ValueError("User aborted due to data outside study area")

            else:
                print("Invalid choice. Enter 1, 2, or 3.")

    else:
        print(f"All records are within the study area")
        return distributions

# endregion

## region Complete Species Occurrence Data Download (wrapper function)
def spp_occ(hydro_data, species_traits_path):
    """
        Complete workflow for species occurrence data acquisition and processing.
        A wrapper function.

        Manages the entire process of obtaining, cleaning and validating species
        distribution data. Handles both GBIF downloads and user-uploaded data,
        applies spatial filtering and generates visualisation maps. Includes
        parameter persistence for reproducibility.

        Parameters
        ----------
        hydro_data : xarray.Dataset
            Hydrodynamic dataset containing:
            - Velocity fields (uo, vo)
            - Spatial coordinates (latitude, longitude)
            - Used to generate masks for species data cleaning
        species_traits_path : str
            Path to CSV file containing species trait data with required column:
            - 'species_name' : str
                Scientific names for GBIF queries or matching uploaded data

        Returns
        -------
        clean_spp : pandas.DataFrame
            Cleaned and filtered species occurrence data containing:
            - 'species' : str
                Scientific name
            - 'latitude' : float
                Decimal latitude
            - 'longitude' : float
                Decimal longitude
            - Additional metadata from GBIF or uploaded file
        spp_params : dict
            Complete parameters dictionary for reproducibility containing all
            user-specified settings for data acquisition and processing
        landmask : numpy.ndarray
            2D binary array (1=land, 0=ocean) from hydrodynamic data
        coastalmask : numpy.ndarray
            2D binary array (1=coastal ocean, 0=non-coastal)
        lonarray : numpy.ndarray
            2D meshgrid of longitude coordinates
        latarray : numpy.ndarray
            2D meshgrid of latitude coordinates

        Notes
        -----
        - Parameters are saved to 'spp_occ_params.json' for future sessions
        - Study area polygon can be reused or redrawn
        - Habitat filtering can remove inland and open ocean records
        - Both pre-comparison and post-comparison maps can be generated
        - Data validation checks ensure uploaded data matches study area

        Workflow Steps
        --------------
        1. Check for existing parameters (spp_occ_params.json)
        2. Collect or reuse parameters
        3. Create or load study area polygon (GeoJSON)
        4. Download from GBIF or load uploaded file
        5. Generate land and coastal masks
        6. Apply habitat filtering
        7. Validate data within study area
        8. Generate comparison maps (optional)
        9. Save cleaned data to CSV (optional)
        10. Create final distribution map
        11. Return cleaned data and masks

        Data Cleaning Steps
        -------------------
        - Remove records without coordinates
        - Remove records on land (using hydrodynamic landmask)
        - Apply coastal buffer filtering (if specified)
        - Clip to study area polygon
        - Remove duplicate records

        Parameter Persistence
        ---------------------
        Parameters are saved to allow:
        - Exact replication of data acquisition
        - Quick reruns with same settings
        - Modification of specific parameters without re-entering all values
        """
    global traits
    params_file = "spp_occ_params.json"

    # Step 1: Check if a previous set of parameters exists and reuse or edit them
    if os.path.exists(params_file):
        reuse_input = input("Found existing parameters. Do you want to reuse them? (yes/no): ").strip().lower()
        if reuse_input in ["yes", "y"]:
            with open(params_file, 'r') as file:
                spp_params = json.load(file)
                print("Reusing existing parameters:")
                for key, value in spp_params.items():
                    print(f"{key}: {value}")

            # Offer to edit specific parameters
            edit_input = input("Do you want to edit any parameters? (yes/no): ").strip().lower()
            if edit_input in ["yes", "y"]:
                spp_params = edit_spp_params(spp_params)
        else:
            # Collect new parameters if not reusing
            spp_params = collect_spp_occ_params(species_traits_path)
    else:
        # If no parameters file exists, collect new parameters
        spp_params = collect_spp_occ_params(species_traits_path)

    # Save updated parameters after any potential edits
    with open(params_file, 'w') as file:
        json.dump(spp_params, file, indent=4)

    # Step 2: Use the geojson file path if it exists
    geojson_file_path = spp_params.get('geojson_file_path')

    # Check: Only create a new map if the geojson file path is missing, None, or invalid
    if not geojson_file_path or not os.path.exists(geojson_file_path):
        # Before creating map:
        subsection_header("Defining your study area")

        info_box([
            "You will be prompted to draw a polygon defining your study area.",
            "",
            "Important: Study area vs Hydrodynamic domain",
            "",
            "Hydrodynamic domain (already set):",
            "  - The dispersal area",
            "  - Large area where ocean currents are simulated",
            "  - Particles can disperse throughout this entire region",
            "",
            "Study area (current step):",
            "  - The focus area",
            "  - Smaller, focused region of interest",
            "  - Where particles will be released from",
            "  - Where particle can settle to",
            "  - Should fit comfortably within the hydrodynamic domain"
        ])

        geojson_file_path = create_map()  # Trigger map creation if needed
        spp_params['geojson_file_path'] = geojson_file_path

        # Save the new geojson path back to the parameters file
        with open(params_file, 'w') as file:
            json.dump(spp_params, file, indent=4)
    else:
        print(f"Using pre-existing GeoJSON file: {geojson_file_path}")

    # Step 3: Read the GeoJSON file
    geometry, shapely_polygon = read_geojson(geojson_file_path) # might have to remove the shapely polygon?

    # Step 4: Ask user to choose data source (GBIF vs user data)
    data_source = spp_params.get('data_source')

    # Only ask for data source if not already set
    if not data_source:
        subsection_header("Choose your species occurrence data source")
        info_box([
            "Option 1: GBIF (Global Biodiversity Information Facility)",
            "   - Automatically download species data from GBIF",
            "   - Uses species names or Aphia ID from your traits CSV",
            "   - Data will be filtered to your study area",
            "",
            "Option 2: Your own data",
            "   - Upload your own pre-collected species distribution data or habitat suitability data",
            "   - Must be in CSV format with columns: species, latitude, longitude",
            "   - Data will be filtered to your study area",
            "   - See 'Part 5: Occurrence Data CSV Format' in the User Guide for format details"
        ])

        while True:
            data_source_choice = input("\nChoose data source (1 for GBIF, 2 for your own data): ").strip()

            if data_source_choice == '1':
                data_source = 'gbif'
                print("\nSelected: GBIF data")
                break
            elif data_source_choice == '2':
                data_source = 'user'
                print("\nSelected: Your own data")
                break
            else:
                print("\nInvalid choice. Enter 1 or 2.")

        # Save data source choice
        spp_params['data_source'] = data_source
        with open(params_file, 'w') as file:
            json.dump(spp_params, file, indent=4)
    else:
        # Data source already set, just use it
        print(f"\nUsing data source: {data_source}")

    # Step 5: Get species occurrence data based on choice
    if data_source == 'gbif':
        print("\nYou have two options for downloading GBIF species occurrence data:")
        print("\n1. Download occurrences for specific species from your trait database")
        print("  - Recommended for connectivity analysis")
        print("\n2. Download all species in your study area")
        print("  - For exploratory analysis only as trait data will be missing")

        mode_choice = input("\nChoose mode (1 or 2) [default: 1]: ").strip() or "1"

        # Load the species traits CSV (needed for Mode 1)
        # Use traits_temp to avoid overwriting global traits list
        if species_traits_path == 'builtin_database_selection':
            traits_temp = load_builtin_traits()
        else:
            traits_temp = pd.read_csv(species_traits_path)
        species_list = traits_temp['species_name'].tolist()

        # Load the species traits CSV with BOTH identifiers
        if species_traits_path == 'builtin_database_selection':
            traits_df = load_builtin_traits()
        else:
            traits_df = pd.read_csv(species_traits_path)

        # Add aphia_id column if it doesn't exist (for backward compatibility)
        if 'aphia_id' not in traits_df.columns:
            traits_df['aphia_id'] = None

        if mode_choice == "1":
            # Mode 1: Download only species from trait database
            print(f"\nDownloading species occurrence data from GBIF for species in trait dataset...")

            # Download species occurrences for each species
            all_data = []
            print(f"Downloading data for {len(traits_df)} species...")

            # Resolve all AphiaIDs to species names before downloading
            for i, row in traits_df.iterrows():
                aphia_id = int(row['aphia_id']) if pd.notna(row['aphia_id']) else None
                species_name = row['species_name'] if pd.notna(row['species_name']) else None

                # If only AphiaID is provided (catch NaN, None, empty string, or whitespace)
                if aphia_id and (species_name is None or pd.isna(species_name) or
                                 str(species_name).strip() == ''):
                    try:
                        resolved_name = get_aphia(aphia_id)
                        resolved_name_formatted = resolved_name.replace(' ', '_')
                        traits_df.at[i, 'species_name'] = resolved_name_formatted
                        print(f"  AphiaID {aphia_id} → {resolved_name_formatted}")

                        # Update the global traits list as well
                        for trait_dict in traits:
                            if trait_dict.get('aphia_id') == aphia_id and trait_dict.get('species_name') is None:
                                trait_dict['species_name'] = resolved_name_formatted

                    except Exception as e:
                        print(f"  Warning: Could not resolve AphiaID {aphia_id}: {e}")

            print(f"\nDownloading data for {len(traits_df)} species...")

            for i, row in traits_df.iterrows():
                # Extract identifiers, handling NaN/empty values
                species_name = row['species_name'] if pd.notna(row['species_name']) else None
                aphia_id = int(row['aphia_id']) if pd.notna(row['aphia_id']) else None

                # Display which identifier we're using
                if aphia_id:
                    print(f"  [{i + 1}/{len(traits_df)}] AphiaID {aphia_id}...")
                elif species_name:
                    print(f"  [{i + 1}/{len(traits_df)}] {species_name}...")
                else:
                    print(f"  [{i + 1}/{len(traits_df)}] Warning: No identifier for row {i + 1}, skipping...")
                    continue

                # Call get_spp_occ with ALL 3 parameters
                try:
                    all_data.extend(get_spp_occ(species_name, aphia_id, geometry))
                except Exception as e:
                    print(f"    Error: {str(e)}")
                    print("    Continuing with next species...")
                    continue

                time.sleep(1.0)

            distributions = pd.DataFrame(all_data)

            print(f"\nDownloaded {len(distributions):} occurrence records")

            # Check if any data was retrieved
            if len(distributions) == 0:
                print("\nWarning: No species occurrence data was retrieved from GBIF.")
                print("\nPossible reasons:")
                print("  1. No GBIF records exist for these species in your study area")
                print("  2. Temporary GBIF server connectivity issue")
                print("  3. Species names/AphiaIDs may not match GBIF database")
                print("\nSuggestions:")
                print("  - Try a different/larger study area")
                print("  - Check species names match GBIF taxonomy")
                print("  - Use your own occurrence data (Option 2 at data source selection)")
                raise ValueError("Cannot proceed without species occurrence data.")

        else:

            # Mode 2: Download all species in study area (exploratory)
            print(f"\nDownloading species occurrence data from GBIF for all species in your study area.")
            print("Warning: This may download many species and take significant time.")

            all_data = get_all_spp_occ(geometry)  # Call once, not in a loop

            distributions = pd.DataFrame(all_data)

            if len(distributions) == 0:
                print("Warning: No species data retrieved")
                print("\nThis could be due to:")
                print("  1. Temporary GBIF server issue. Try again in 5 minutes")
                print("  2. No species records in your study area")
                print("\nOptions:")
                print("  - Wait a few minutes and try again")
                print("  - Use Mode 1 (species-specific download) instead")

                raise ValueError("Cannot proceed without species data. Please try again or use Mode 1.")

            print(f"\nDownloaded {len(distributions):} occurrence records")

            unique_species = distributions['species'].nunique()

            print(f"Found {unique_species} unique species in study area.")

            # Show species not in trait database
            trait_species = set(species_list)
            found_species = set(distributions['species'].unique())
            extra_species = found_species - trait_species

            if extra_species:
                print(f"\nNote: {len(extra_species)} species found that are not in your trait database.")

                print(
                    "These species will be included in occurrence data but cannot be used in connectivity modelling without trait information.")

    else:  # user data
        print("\nUpload your species occurrence data in the required format: CSV with columns 'species', 'latitude', 'longitude")

        # Check if file path already saved in parameters
        if 'user_data_file_path' in spp_params and spp_params['user_data_file_path']:
            file_path = spp_params['user_data_file_path']
            print(f"\nUsing saved file path: {file_path}")
        else:
            file_path = get_valid_path("Enter the input path and file name", must_exist=True, extensions=['.csv'])
            # Ensure .csv extension
            if not file_path.endswith(".csv"):
                file_path += ".csv"
            # Save for next time
            spp_params['user_data_file_path'] = file_path

        while True:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"Error: File not found '{file_path}'.")
                file_path = get_valid_path("Enter the input path and file name", must_exist=True, extensions=['.csv'])
                if not file_path.endswith(".csv"):
                    file_path += ".csv"
                spp_params['user_data_file_path'] = file_path
                continue

            # Try to load the file
            try:
                distributions = pd.read_csv(file_path)

                # Verify required columns
                required_columns = {'species', 'latitude', 'longitude'}
                missing_columns = required_columns - set(distributions.columns)

                if missing_columns:
                    print(f"Error: Missing required columns: {', '.join(missing_columns)}")
                    print(f"Your columns: {', '.join(distributions.columns)}")
                    retry = input("Try another file? (yes/no): ").strip().lower()
                    if retry not in ['yes', 'y']:
                        raise ValueError("Cannot proceed without valid species data.")
                    file_path = input("Enter new file path: ").strip()
                    if not file_path.endswith(".csv"):
                        file_path += ".csv"
                    spp_params['user_data_file_path'] = file_path
                    continue

                # Success
                print(
                    f"\nLoaded {len(distributions)}: occurrence records from {len(distributions['species'].unique())} species")

                # Validating the extent of the uploaded data
                print("\nValidating uploaded data...")

                # Check if data has required columns
                print("\nChecking the data format...")
                if 'latitude' not in distributions.columns or 'longitude' not in distributions.columns:
                    raise ValueError("CSV must contain 'latitude' and 'longitude' columns.")
                print("Data format is suitable. Required columns found (latitude, longitude).")

                # Validate points are within study polygon
                distributions = validate_uploaded_data(distributions, shapely_polygon)

                print(f"\nValidation complete.\n")

                # Print species summary
                species_counts = distributions['species'].value_counts()
                species_lines = [f"{species}: {count} records" for species, count in species_counts.items()]
                data_box("Species Occurrence Data Summary", species_lines)

                # Print coordinate ranges
                data_box("Coordinate Ranges", [
                    f"Latitude: {distributions['latitude'].min():.4f}° to {distributions['latitude'].max():.4f}°",
                    f"Longitude: {distributions['longitude'].min():.4f}° to {distributions['longitude'].max():.4f}°"
                ])

                # Save parameters including the file path
                with open(params_file, 'w') as file:
                    json.dump(spp_params, file, indent=4)

                break

            except Exception as e:
                print(f"Error reading file: {e}")
                retry = input("Try another file? (yes/no): ").strip().lower()
                if retry not in ['yes', 'y']:
                    raise ValueError("Cannot proceed without valid species data.")
                file_path = input("Enter new file path: ").strip()
                if not file_path.endswith(".csv"):
                    file_path += ".csv"
                spp_params['user_data_file_path'] = file_path

    # Step 6: Create landmask and clean up species data
    landmask, coastalmask, lonarray, latarray = create_masks(hydro_data, geojson_file_path=geojson_file_path)

    # Load traits_df if not already loaded (for user-uploaded data)
    if 'traits_df' not in locals():
        if species_traits_path == 'builtin_database_selection':
            traits_df = load_builtin_traits()
        else:
            traits_df = pd.read_csv(species_traits_path)
        if 'aphia_id' not in traits_df.columns:
            traits_df['aphia_id'] = None

        # For user-uploaded data: resolve AphiaIDs to match species names in uploaded data
        if data_source == 'user':
            print("\nMatching uploaded species names with trait database...")
            for i, row in traits_df.iterrows():
                aphia_id = int(row['aphia_id']) if pd.notna(row['aphia_id']) else None
                species_name = row['species_name'] if pd.notna(row['species_name']) else None

                # If only AphiaID provided, resolve it
                if aphia_id and (species_name is None or pd.isna(species_name) or
                                 str(species_name).strip() == ''):
                    try:
                        # For user-uploaded data: resolve AphiaIDs to match species names in uploaded data
                        for i, row in traits_df.iterrows():
                            aphia_id = int(row['aphia_id']) if pd.notna(row['aphia_id']) else None
                            species_name = row['species_name'] if pd.notna(row['species_name']) else None

                            if aphia_id and (species_name is None or pd.isna(species_name) or
                                             str(species_name).strip() == ''):
                                try:
                                    resolved_name = get_aphia(aphia_id)
                                    resolved_name_formatted = resolved_name.replace(' ', '_')
                                    traits_df.at[i, 'species_name'] = resolved_name_formatted
                                    print(f"  AphiaID {aphia_id} → {resolved_name_formatted}")

                                    # Update the global traits list as well
                                    for trait_dict in traits:
                                        if trait_dict.get('aphia_id') == aphia_id and trait_dict.get(
                                                'species_name') is None:
                                            trait_dict['species_name'] = resolved_name_formatted

                                except Exception as e:
                                    print(f"  Warning: Could not resolve AphiaID {aphia_id}: {e}")
                    except Exception as e:
                        print(f"  Warning: Could not resolve AphiaID {aphia_id}: {e}")

    # Load traits_df if not already loaded (for user-uploaded data)
    # When users upload their own data, traits_df was never created
    # This ensures it exists for validation regardless of data source
    if 'traits_df' not in locals():
        if species_traits_path == 'builtin_database_selection':
            traits_df = load_builtin_traits()
        else:
            traits_df = pd.read_csv(species_traits_path)
        # Add aphia_id column if it doesn't exist
        if 'aphia_id' not in traits_df.columns:
            traits_df['aphia_id'] = None

    # Step 7: Interactive data cleaning
    # Check if habitat filtering is already set in saved parameters
    habitat_type = spp_params.get('habitat_filtering', None)

    if habitat_type:
        # Use saved habitat filtering preference
        print(f"\nUsing saved habitat filtering preference: {habitat_type}")
    else:
        # Ask user to choose habitat filtering
        subsection_header("Species Occurrence Data Cleaning Options")

        info_box([
            "The species occurrence data can be filtered based on your habitat preferences:",
            "",
            "Option 1: Coastal",
            "  - This filters data to coastal habitats only and removes occurrences on land and offshore",
            "  - This option is appropriate for littoral and infralittoral settings",
            "",
            "Option 2: Ocean",
            "  - This filters data to all ocean occurrences (coastal and offshore)",
            "    and removes occurrences on land",
            "  - This option is appropriate for offshore and mixed (coastal & offshore) settings",
            "",
            "Option 3: None",
            "  - This uses your data exactly as provided",
            "  - This option is appropriate for pre-cleaned data"
        ])

        while True:
            habitat_choice = input("\nChoose habitat filtering (1/2/3): ").strip()

            if habitat_choice == '1':
                habitat_type = 'coastal'
                print("\nSelected: Coastal filtering")
                break
            elif habitat_choice == '2':
                habitat_type = 'ocean'
                print("\nSelected: Ocean filtering")
                break
            elif habitat_choice == '3':
                habitat_type = 'none'
                print("\nSelected: No filtering")
                break
            else:
                print("\nInvalid choice. Enter 1, 2 or 3.")

    # Apply the selected filtering
    clean_spp = clean_spp_occ(distributions, landmask, coastalmask, lonarray, latarray,
                              habitat_type=habitat_type)

    # Validate species occurrences
    validation_report = validate_spp_occ(clean_spp, traits_df, min_records=10)

    # Save filtering choice to params for reference
    spp_params['habitat_filtering'] = habitat_type
    with open(params_file, 'w') as file:
        json.dump(spp_params, file, indent=4)

    # Step 8: Offer to save filtered data (for ALL users)
    save_choice = input("\nSave filtered species data to CSV? (yes/no): ").strip().lower()
    if save_choice in ['yes', 'y']:
        csv_path = input("\nEnter output path and filename [default: filtered_species.csv]: ").strip()
        if not csv_path:
            csv_path = "filtered_species.csv"
        if not csv_path.endswith(".csv"):
            csv_path += ".csv"

        clean_spp.to_csv(csv_path, index=False)
        print(f"\nFiltered data saved to: {csv_path}")
        print(f"Records: {len(clean_spp):}")
        print(f"Species: {len(clean_spp['species'].unique())}")

    # Step 9: Create a map of the filtered data
    plot_spp_occ(distributions, clean_spp, geojson_file_path, habitat_type, lonarray, latarray)

    return clean_spp, spp_params, landmask, coastalmask, lonarray, latarray
# endregion

# endregion

# region Creating the Grids
## region Setup Functions
def create_grid(coastalmask, landmask, lonarray, latarray):
    """
        Generates numbered grid of coastal cells for connectivity analysis.

        Creates a pandas DataFrame containing all coastal grid cells with unique
        identifiers. This grid forms the basis for particle release locations and
        connectivity matrices in subsequent analyses.

        Parameters
        ----------
        coastalmask : numpy.ndarray
            2D binary array where 1 indicates coastal ocean cells and 0 indicates
            non-coastal cells. Shape: (n_latitudes, n_longitudes)
        landmask : numpy.ndarray
            2D binary array where 1 indicates land and 0 indicates ocean.
            Shape: (n_latitudes, n_longitudes)
        lonarray : numpy.ndarray
            2D array of longitude coordinates matching coastalmask dimensions
        latarray : numpy.ndarray
            2D array of latitude coordinates matching coastalmask dimensions

        Returns
        -------
        pandas.DataFrame
            DataFrame with one row per coastal cell containing:
            - 'grid_id' : int
                Unique sequential identifier (1, 2, 3, ...)
            - 'lon' : float
                Longitude of grid cell center
            - 'lat' : float
                Latitude of grid cell center
            - 'row_idx' : int
                Row index in original grid array
            - 'col_idx' : int
                Column index in original grid array

        Notes
        -----
        - Grid IDs are assigned sequentially starting from 1
        - Only cells where coastalmask==1 are included
        - Row and column indices are preserved for mapping back to original arrays
        - Coordinates represent cell centers, not corners
        - Grid IDs are persistent across function calls with same coastalmask

        Grid Cell Ordering
        ------------------
        Cells are numbered in row-major order (reading across rows from top to bottom,
        left to right within each row), including only coastal cells.
        """

    # Step 1: Flatten the grid to create full_grid_points
    full_grid = np.column_stack((lonarray.ravel(), latarray.ravel()))

    # Generate grid IDs
    grid_ids = np.arange(full_grid.shape[0]).reshape(lonarray.shape)

    # Create a DataFrame with full grid data
    landmask_flat = landmask.ravel()
    coastalmask_flat = coastalmask.ravel()

    full_grid_df = pd.DataFrame({
        'longitude': full_grid[:, 0],
        'latitude': full_grid[:, 1],
        'grid_id': grid_ids.ravel(),
        'is_land': landmask_flat,  # NEW: for mapping
        'is_coastal': coastalmask_flat  # NEW: for mapping
    })

    # Step 2: Filter coastal points
    coastal_cells = coastalmask.ravel()
    coastal_grid = full_grid[coastal_cells == 1]

    # Step 3: Create coastal_grid_df
    coastal_grid_df = full_grid_df[coastal_cells == 1].reset_index(drop=True)

    # Step 4: Filter all ocean points (not just coastal)
    # Ocean = anywhere that is not land (landmask == 0)
    ocean_cells = (landmask.ravel() == 0)
    ocean_grid = full_grid[ocean_cells]

    # Create ocean_grid_df
    ocean_grid_df = full_grid_df[ocean_cells].reset_index(drop=True)

    return full_grid, full_grid_df, coastal_grid, coastal_grid_df, ocean_grid, ocean_grid_df, grid_ids

## Map Species to Grid
def map_spp_to_grid(full_grid_df, clean_spp):
    """
    Assigns species occurrence records to nearest coastal grid cells.

    For each species occurrence point, finds the nearest coastal grid cell and
    assigns the species to that cell. This discretizes continuous species
    distribution data onto the modelling grid and allows for removal of duplicate
    speasies records.

    Parameters
    ----------
    full_grid_df : pandas.DataFrame
        Grid of all coastal cells with columns:
        - 'grid_id' : int
            Unique cell identifier
        - 'lon' : float
            Cell longitude
        - 'lat' : float
            Cell latitude
        Must contain at least one row
    clean_spp : pandas.DataFrame
        Cleaned species occurrence data with columns:
        - 'species' : str
            Scientific name
        - 'longitude' : float
            Occurrence longitude
        - 'latitude' : float
            Occurrence latitude
        May contain additional columns

    Returns
    -------
    pandas.DataFrame
        Modified full_grid_df with additional column:
        - 'species' : object (list)
            List of species present in each grid cell. Empty list if no species
            present. Multiple occurrences of same species are deduplicated.

    Notes
    -----
    - Uses scipy cKDTree for efficient nearest-neighbor search
    - Distance calculation uses Euclidean distance in degrees
    - All species occurrences, regardless of distance, are assigned to nearest cell
    - Species names are deduplicated within each cell
    - Cells without species assigned get empty list []
    - Original grid_df is modified in place and returned

    Algorithm Details
    -----------------
    1. Build KD-tree from grid cell coordinates
    2. Query tree for each species occurrence
    3. Find nearest grid cell for each occurrence
    4. Group occurrences by grid cell
    5. Create list of unique species per cell
    6. Assign species lists to grid dataframe
    """
    # Extract grid coordinates and IDs
    grid_coords = full_grid_df[['longitude', 'latitude']].values
    grid_ids = full_grid_df['grid_id'].values

    # Create KDTree for the grid points
    grid_tree = cKDTree(grid_coords)

    # Initialize a dictionary to store each species-specific DataFrame
    species_grids = {}

    # List to collect all species grid points (for the all_species_grid output)
    all_species_points = []
    all_species_grid_ids = []

    # Process each species separately
    for species in clean_spp['species'].unique():
        # Subset the species_data for the current species
        species_subset = clean_spp[clean_spp['species'] == species]
        species_coords = np.array(list(zip(species_subset['longitude'], species_subset['latitude'])))

        # Find the nearest grid points for this species occurrence
        _, indices = grid_tree.query(species_coords)

        # Retrieve the corresponding grid points and their IDs
        nearest_grid_points = grid_coords[indices]
        nearest_grid_ids = grid_ids[indices]

        # Create a DataFrame for this species, ensuring unique (longitude, latitude) pairs
        species_df = pd.DataFrame({
            'species': species,
            'longitude': nearest_grid_points[:, 0],
            'latitude': nearest_grid_points[:, 1],
            'grid_id': nearest_grid_ids
        }).drop_duplicates(subset=['longitude', 'latitude'])  # Ensure unique lat/lon pairs

        # Store the species-specific DataFrame in the dictionary
        species_grids[species] = species_df

        # Append species grid points and grid IDs to the lists for all_species_grid
        all_species_points.append(species_df[['longitude', 'latitude']].values)
        all_species_grid_ids.append(species_df['grid_id'].values)

    # Concatenate all species grid points into a single numpy array for all_species_grid
    all_species_points = np.vstack(all_species_points)
    all_species_grid_ids = np.hstack(all_species_grid_ids)

    # Combine the grid points and IDs, and drop duplicates based on grid IDs (ensures unique grid_id)
    unique_all_species = pd.DataFrame({
        'longitude': all_species_points[:, 0],
        'latitude': all_species_points[:, 1],
        'grid_id': all_species_grid_ids
    }).drop_duplicates(subset=['grid_id'])

    # Extract the unique grid points for all_species_grid
    all_species_grid = unique_all_species[['longitude', 'latitude']].values

    return all_species_grid, species_grids

## Filter Coastal Grid
def filter_by_polygon(grid, shapely_polygon):
    """
    Filter grid cells to include only those within a geographic polygon.

    Applies spatial filtering to retain only grid cells that fall within the
    specified polygon boundary. Useful for defining specific study areas or
    excluding regions from connectivity analyses.

    Parameters
    ----------
    grid : pandas.DataFrame
        Grid dataframe containing columns:
        - 'lon' : float
            Cell longitude
        - 'lat' : float
            Cell latitude
        - Additional columns preserved in output
    shapely_polygon : shapely.geometry.Polygon
        Polygon defining the study area boundary. Vertices should be in
        longitude-latitude coordinates (degrees)

    Returns
    -------
    pandas.DataFrame
        Subset of input grid containing only cells within the polygon.
        All columns from input grid are preserved. Index is reset to
        sequential integers starting from 0.

    Notes
    -----
    - Uses shapely's contains() method for point-in-polygon test
    - Edge cases (points exactly on boundary) are handled by shapely
    - Original grid dataframe is not modified
    - Grid IDs are preserved but may no longer be sequential
    - Returns empty DataFrame if no points fall within polygon
    """
    # Create a Shapely Point for each grid cell and filter by the polygon
    filtered_grid = np.array([
        [lon, lat] for lon, lat in grid
        if shapely_polygon.contains(Point(lon, lat))
    ])
    return filtered_grid

## Filter

## Save Grids
def save_grids(release_grids=None, settlement_grids=None, full_grid=None):
    """
    Save grid dataframes to CSV files for reproducibility and inspection.

    Exports release grids, settlement grids and full coastal grid to separate
    CSV files. This allows inspection of grid definitions, reuse in subsequent
    analyses and reproducibility of simulation setups.

    Parameters
    ----------
    release_grids : dict of pandas.DataFrame, optional
        Dictionary mapping species names to release grid DataFrames.
        Each DataFrame should contain columns: grid_id, lon, lat, species.
        If None, no release grids are saved
    settlement_grids : dict of pandas.DataFrame, optional
        Dictionary mapping species names to settlement grid DataFrames.
        Each DataFrame should contain columns: grid_id, lon, lat, species.
        If None, no settlement grids are saved
    full_grid : pandas.DataFrame, optional
        Complete coastal grid DataFrame with all cells.
        Should contain columns: grid_id, lon, lat, and optionally species.
        If None, no full grid is saved

    Returns
    -------
    None
        Files are saved to disk with user-specified paths

    Notes
    -----
    - User is prompted for each file path interactively
    - Existing files with same name will be overwritten without warning
    - CSV files include header row with column names
    - Index column is not saved
    """

    # Optionally save the full meshgrid
    if full_grid is not None:
        save_csv = input("\nDo you want to save the meshgrid data to a CSV file? (yes/no): ").strip().lower()
        if save_csv in ['yes', 'y']:
            file_path = input("Enter the output path and file name: ").strip()
            # Ensure that the file has a .nc extension
            if not file_path.endswith(".csv"):
                file_path += ".csv"
            if isinstance(full_grid, pd.DataFrame):
                full_grid.to_csv(file_path, index=False)
            else:
                # If it's an array, convert it to a DataFrame for saving
                full_grid_df = pd.DataFrame(full_grid,
                                            columns=['longitude', 'latitude', 'grid_id'][:full_grid.shape[1]])
                full_grid_df.to_csv(file_path, index=False)
            print(f"Meshgrid data saved as {file_path}.")

    # Save release grids if available
    if release_grids and len(release_grids) > 0:  # Check if the list is not empty
        save_csv = input("\nDo you want to save the release grid points to CSV files? (yes/no): ").strip().lower()
        if save_csv in ['yes', 'y']:
            file_path = input("Enter the output path and file name: ").strip()
            if not os.path.exists(file_path):
                os.makedirs(file_path)  # Create the directory if it doesn't exist
            for species_name, df in release_grids.items():
                if isinstance(df, pd.DataFrame):
                    file_name = os.path.join(file_path, f"{species_name}_release_grid.csv")
                    df.to_csv(file_name, index=False)
                    print(f"Release grid for species '{species_name}' saved as {file_name}.")
                else:
                    print(f"Invalid data format for species '{species_name}', skipping.")

    # Save settlement grids if available
    if settlement_grids and len(settlement_grids) > 0:  # Check if the list is not empty
        save_csv = input("\nDo you want to save the settlement grid points to CSV files? (yes/no): ").strip().lower()
        if save_csv in ['yes', 'y']:
            file_path = input("Enter output path and file name: ").strip()
            if not os.path.exists(file_path):
                os.makedirs(file_path)  # Create the directory if it doesn't exist
            for grid_id, df in settlement_grids.items():
                if isinstance(df, pd.DataFrame):
                    file_name = os.path.join(file_path, f"settlement_grid_{grid_id}.csv")
                    df.to_csv(file_name, index=False)
                    print(f"Settlement grid for grid ID '{grid_id}' saved as {file_name}.")
                else:
                    print(f"Invalid data format for grid ID '{grid_id}', skipping.")
# endregion

## region Setup Particle Release and Settlement Strategy (wrapper function)
def setup_release_settlement(coastalmask, landmask, clean_spp, spp_params, lonarray, latarray):
    """
    Configure species-specific release and settlement grids for connectivity modelling.

    This is a central function that orchestrates grid creation, species assignment,
    and polygon filtering to produce ready-to-use release and settlement locations
    for each species in the connectivity simulation.

    Parameters
    ----------
    coastalmask : numpy.ndarray
        2D binary array where 1=coastal ocean cell, used to define grid
    landmask : numpy.ndarray
        2D binary array where 1=land, used for visualisation
    clean_spp : pandas.DataFrame
        Cleaned species occurrence data with columns:
        - 'species' : str
        - 'longitude' : float
        - 'latitude' : float
    spp_params : dict
        Species parameters including:
        - 'geojson_file_path' : str
            Path to study area polygon
        - 'habitat_filtering' : str
            Type of filtering applied
        - Other metadata
    lonarray : numpy.ndarray
        2D meshgrid of longitude coordinates
    latarray : numpy.ndarray
        2D meshgrid of latitude coordinates

    Returns
    -------
    release_grids : dict
        Dictionary mapping species names (str) to release grid DataFrames.
        Each DataFrame contains: grid_id, lon, lat, species, row_idx, col_idx
    settlement_grids : dict
        Dictionary mapping species names (str) to settlement grid DataFrames.
        Structure identical to release_grids
    grid_ids : dict
        Dictionary mapping species names (str) to arrays of grid IDs for that
        species. Used for indexing in connectivity matrices
    full_grid_df : pandas.DataFrame
        Complete coastal grid with all cells and species assignments
    coastal_grid_df : pandas.DataFrame
        Filtered grid within study area polygon
    study_polygon : shapely.geometry.Polygon
        Study area boundary polygon

    Workflow
    --------
    1. Create full coastal grid from coastalmask
    2. Map species occurrences to nearest grid cells
    3. Load and read study area polygon
    4. Filter grid to polygon extent
    5. For each species:
       a. Extract cells where species is present
       b. Create release grid (species-specific cells)
       c. Create settlement grid (same as release by default)
       d. Extract grid IDs for indexing
    6. Generate comprehensive visualisation maps
    7. Optionally save all grids to CSV files

    Notes
    -----
    - Release and settlement grids can be identical or different
    - User can modify settlement to include additional cells (a buffer around the site)
    """
    # Extract parameters
    file_path = spp_params['geojson_file_path']
    habitat_filtering = spp_params.get('habitat_filtering', 'coastal')

    # Step 1: Create grid and mask data from hydrodynamic data
    full_grid, full_grid_df, coastal_grid, coastal_grid_df, ocean_grid, ocean_grid_df, grid_ids = create_grid(
        coastalmask, landmask, lonarray, latarray)

    # Step 2: Filter grids using the polygon from GeoJSON
    _, shapely_polygon = read_geojson(file_path)

    # Filter coordinate arrays (for meshgrid used in particle release)
    coastal_grid_filtered = filter_by_polygon(coastal_grid, shapely_polygon)
    ocean_grid_filtered = filter_by_polygon(ocean_grid, shapely_polygon)

    # Get grid_ids of filtered cells by filtering the DataFrame directly using the polygon
    if len(coastal_grid_filtered) > 0:
        # Filter DataFrame directly using polygon containment check
        coastal_grid_df_filtered = coastal_grid_df[
            coastal_grid_df.apply(
                lambda r: shapely_polygon.contains(Point(r['longitude'], r['latitude'])),
                axis=1
            )
        ].copy()
    else:
        coastal_grid_df_filtered = pd.DataFrame(columns=coastal_grid_df.columns)

    if len(ocean_grid_filtered) > 0:
        # Filter DataFrame directly using polygon containment check
        ocean_grid_df_filtered = ocean_grid_df[
            ocean_grid_df.apply(
                lambda r: shapely_polygon.contains(Point(r['longitude'], r['latitude'])),
                axis=1
            )
        ].copy()
    else:
        ocean_grid_df_filtered = pd.DataFrame(columns=ocean_grid_df.columns)

    # Step 3: Determine which grid to use for "all sites" based on habitat
    if habitat_filtering == 'coastal':
        all_sites_grid = coastal_grid_filtered
        all_sites_description = "coastal sites"
        grid_df_for_mapping = coastal_grid_df_filtered
        study_area_df_filtered = coastal_grid_df_filtered
    else:  # 'ocean' or 'none'
        all_sites_grid = ocean_grid_filtered
        all_sites_description = "ocean sites (coastal + offshore)"
        grid_df_for_mapping = ocean_grid_df_filtered
        study_area_df_filtered = ocean_grid_df_filtered

    # Step 4: Ask user to choose particle release strategy
    subsection_header("Propagule Release Strategy")

    # Build release options info
    release_lines = [
        "Choose where particles should be released from.",
        "",
        "You can use the occurrence data from step 2, upload a different dataset or release from everywhere in your study area.",
        "",
        "Option 1: All sites in the study area",
        f"  - Release particles from all {all_sites_description} in your study area",
        f"  - Does not use species occurrence data",
        f"  - This is suitable for assessing maximum dispersal scenarios or for data-poor scenarios",
        f"  - Total release sites: {len(all_sites_grid)}",
        "",
        "Option 2: Species-specific habitat (based on GBIF data)",
        "  - Release particles only from species-specific suitable habitat",
        "  - This is suitable for trait-based analysis"
    ]

    if 'data_source' in spp_params:
        data_source_text = "GBIF data" if spp_params['data_source'] == 'gbif' else "your uploaded data"
        release_lines.append(f"  - Uses: Species distributions from {data_source_text}")

    if len(clean_spp) > 0:
        unique_species = clean_spp['species'].nunique()
        # Calculate actual unique grid cells that would be created
        temp_meshgrid, temp_grids = map_spp_to_grid(grid_df_for_mapping, clean_spp)
        total_unique_cells = len(temp_meshgrid)  # Truly unique across all species
        per_species_total = sum(
            len(g['grid_id'].unique()) if 'grid_id' in g.columns else 0 for g in temp_grids.values())
        release_lines.append(f"  - Total unique release sites: {total_unique_cells}")
        release_lines.append(
            f"  - Sum across {unique_species} species: {per_species_total} (includes shared sites)")

    release_lines.extend([
        "",
        "Option 3: Custom species occurrence data",
        "  - Upload your own species occurrence or habitat suitability data (CSV format)",
        "  - This can be a different CSV file from the data loaded in step 2",
        "  - Required columns: species, latitude, longitude",
        "  - This is suitable for trait-based analysis"
    ])

    info_box(release_lines)

    while True:
        release_choice = input("\nChoose release strategy (1/2/3): ").strip()

        if release_choice == '1':
            # Release from all sites in study area
            meshgrid = all_sites_grid
            species_names = clean_spp['species'].unique() if len(clean_spp) > 0 else ['all_species']
            release_grids = {}

            # Use the pre-filtered DataFrame (has correct grid_ids)
            study_area_df = study_area_df_filtered.copy()

            for sp in species_names:
                release_grids[sp] = study_area_df.copy()

            print(f"  Release from all {len(study_area_df)} study area sites")
            break

        elif release_choice == '2':
            # Map species data to appropriate grid (coastal or ocean based on habitat)
            meshgrid, release_grids = map_spp_to_grid(grid_df_for_mapping, clean_spp)

            print(f"Created release grids for {len(release_grids)} species")
            print(f"Total unique release sites: {len(meshgrid):}")
            break

        elif release_choice == '3':
            print("\nSelected: Custom species occurrence data")
            print("\nUpload your own species occurrence or habitat suitability data.")
            print("Required format: CSV with columns 'species', 'latitude', 'longitude'.")

            while True:
                custom_file_path = input("Enter the input path and file name: ").strip()

                if not custom_file_path.endswith(".csv"):
                    custom_file_path += ".csv"

                if not os.path.exists(custom_file_path):
                    print(f"Error: File not found '{custom_file_path}'. Try again.")
                    continue

                try:
                    custom_release_data = pd.read_csv(custom_file_path)

                    # Verify required columns
                    required_columns = {'species', 'latitude', 'longitude'}
                    missing_columns = required_columns - set(custom_release_data.columns)

                    if missing_columns:
                        print(f"Error: Missing required columns: {', '.join(missing_columns)}")
                        print(f"Your columns: {', '.join(custom_release_data.columns)}")
                        retry = input("Try another file? (yes/no): ").strip().lower()
                        if retry not in ['yes', 'y']:
                            raise ValueError("Cannot proceed without valid release data.")
                        continue

                    # Success! Map to appropriate grid
                    meshgrid, release_grids = map_spp_to_grid(grid_df_for_mapping, custom_release_data)
                    print(f"Created custom release grids for {len(release_grids)} species")
                    print(f"Total unique release sites: {len(meshgrid):}")
                    break

                except Exception as e:
                    print(f"Error reading file: {e}")
                    retry = input("Try another file? (yes/no): ").strip().lower()
                    if retry not in ['yes', 'y']:
                        raise ValueError("Cannot proceed without valid release data.")

            break

        else:
            print("Invalid choice. Enter 1, 2 or 3.")

    # Save release choice
    release_strategy_map = {'1': 'all_sites', '2': 'species_habitat', '3': 'custom_habitat'}
    spp_params['release_strategy'] = release_strategy_map.get(release_choice)
    with open('spp_occ_params.json', 'w') as file:
        json.dump(spp_params, file, indent=4)

    # Step 5: Ask user to choose particle settlement strategy
    subsection_header("Propagule Settlement Strategy")

    # Build settlement options info
    settlement_lines = [
        "Choose where particles can settle.",
        "",
        "You can use the same data as for release, upload a different dataset",
        "or allow settlement everywhere in your study area.",
        "",
        "Option 1: All sites in the study area",
        f"  - Particles can settle anywhere in your study area",
        f"  - Does not use species occurrence data",
        f"  - This is suitable for assessing maximum dispersal scenarios or for data-poor scenarios",
        f"  - Total settlement sites: {len(all_sites_grid)}",
        "",
        "Option 2: Species-specific habitat (same as release data)",
        "  - Particles can only settle in suitable habitat for each species",
        "  - This is suitable for trait-based analysis"
    ]

    if 'data_source' in spp_params:
        data_source_text = "GBIF data" if spp_params['data_source'] == 'gbif' else "your uploaded data"
        settlement_lines.append(f"  - Uses: {data_source_text} from step 2")

    if len(clean_spp) > 0:
        unique_species = clean_spp['species'].nunique()
        # Use the same calculation as release to be consistent
        temp_meshgrid, temp_grids = map_spp_to_grid(grid_df_for_mapping, clean_spp)
        total_unique_cells = len(temp_meshgrid)
        per_species_total = sum(
            len(g['grid_id'].unique()) if 'grid_id' in g.columns else 0 for g in temp_grids.values())
        settlement_lines.append(f"  - Total unique settlement sites: {total_unique_cells}")
        settlement_lines.append(
            f"  - Sum across {unique_species} species: {per_species_total} (includes shared sites)")

    settlement_lines.extend([
        "",
        "Option 3: Custom species occurrence data",
        "  - Upload your own species occurrence or habitat suitability data (CSV format)",
        "  - This can be a different CSV file from the release site file",
        "  - Required columns: species, latitude, longitude",
        "  - This is suitable for trait-based analysis"
    ])

    info_box(settlement_lines)

    while True:
        settlement_choice = input("\nChoose settlement strategy (1/2/3): ").strip()

        if settlement_choice == '1':
            # Settle at all sites in study area
            species_names = clean_spp['species'].unique() if len(clean_spp) > 0 else ['all_species']
            settlement_grids = {}

            # Use the pre-filtered DataFrame (has correct grid_ids)
            study_area_df = study_area_df_filtered.copy()

            for sp in species_names:
                settlement_grids[sp] = study_area_df.copy()

            print(f"Settlement at all {len(study_area_df)} study area sites")
            break

        elif settlement_choice == '2':
            # Use the same clean_spp data for settlement, mapped to appropriate grid
            _, settlement_grids = map_spp_to_grid(grid_df_for_mapping, clean_spp)

            print(f"Created settlement grids for {len(settlement_grids)} species")
            break

        elif settlement_choice == '3':
            print("\nSelected: Custom species occurrence data")
            print("\nUpload your own species occurrence or habitat suitability data.")
            print("Required format: CSV with columns 'species', 'latitude', 'longitude'.")

            while True:
                custom_file_path = input("Enter the input path and file name: ").strip()

                if not custom_file_path.endswith(".csv"):
                    custom_file_path += ".csv"

                if not os.path.exists(custom_file_path):
                    print(f"Error: File not found '{custom_file_path}'. Try again.")
                    continue

                try:
                    custom_settlement_data = pd.read_csv(custom_file_path)

                    # Verify required columns
                    required_columns = {'species', 'latitude', 'longitude'}
                    missing_columns = required_columns - set(custom_settlement_data.columns)

                    if missing_columns:
                        print(f"Error: Missing required columns: {', '.join(missing_columns)}")
                        print(f"Your columns: {', '.join(custom_settlement_data.columns)}")
                        retry = input("Try another file? (yes/no): ").strip().lower()
                        if retry not in ['yes', 'y']:
                            raise ValueError("Cannot proceed without valid settlement data.")
                        continue

                    # Success! Map to appropriate grid
                    _, settlement_grids = map_spp_to_grid(grid_df_for_mapping, custom_settlement_data)
                    print(f"Created custom settlement grids for {len(settlement_grids)} species")
                    break

                except Exception as e:
                    print(f"Error reading file: {e}")
                    retry = input("Try another file? (yes/no): ").strip().lower()
                    if retry not in ['yes', 'y']:
                        raise ValueError("Cannot proceed without valid settlement data.")

            break

        else:
            print("Invalid choice. Enter 1, 2 or 3.")

    # Save settlement choice
    settlement_strategy_map = {'1': 'all_sites', '2': 'species_habitat', '3': 'custom_habitat'}
    spp_params['settlement_strategy'] = settlement_strategy_map.get(settlement_choice)
    with open('spp_occ_params.json', 'w') as file:
        json.dump(spp_params, file, indent=4)

    # Step 6: Optionally save release and settlement grids as CSV
    save_grids(release_grids, settlement_grids, full_grid)

    return meshgrid, release_grids, settlement_grids, full_grid_df, grid_ids, coastal_grid, coastal_grid_df, coastal_grid_filtered, grid_df_for_mapping

# endregion

# region Creating the Fieldset and related Components
## region Setup Functions

### Reflective Boundary
def reflective_boundary(fieldset, landmask):
    """
    Creates normalized vector fields that push beached particles back into the ocean.
    This prevents particles from getting stuck during the simulation and is a purely computational tool.

    For each land cell, calculates the direction to the nearest ocean cell(s) and
    creates a unit vector pointing from land to water. These vectors are used by
    the reflective kernel to prevent particles from becoming stuck on coastlines.

    Parameters:
    -----------
    fieldset : parcels.FieldSet
        FieldSet containing hydrodynamic data. Used to extract grid coordinates.

    landmask : numpy.ndarray (2D)
        Binary mask indicating land (1) vs ocean (0) cells.

    Returns:
    --------
    vectorfield_x : numpy.ndarray (2D)
        Normalized zonal (east-west) relfective vectors. Non-zero only at land cells.

    vectorfield_y : numpy.ndarray (2D)
        Normalized meridional (north-south) relfective vectors. Non-zero only at land cells.

    Lons : numpy.ndarray (1D)
        Longitude coordinates from fieldset.

    Lats : numpy.ndarray (1D)
        Latitude coordinates from fieldset.

    Notes:
    ------
    - Vectors are normalized to unit length (magnitude = 1).
    - For land cells equidistant to multiple ocean cells, uses the mean position.
    - Computational cost: O(n_land × n_ocean). This can be slow for large grids.
    """
    # Create necessary components
    Lons = fieldset.U.lon
    Lats = fieldset.U.lat
    fieldmesh_x, fieldmesh_y = np.meshgrid(Lons, Lats)

    # Find indices of land and ocean
    oceancells = np.where(landmask == 0)
    landcells = np.where(landmask == 1)

    # Create empty arrays
    vectorfield_x = np.zeros(fieldmesh_x.shape)
    vectorfield_y = np.zeros(fieldmesh_y.shape)

    # Repeat the loop for all the land cells
    for i1 in range(len(landcells[1])):
        # Find the lon and lat for all the land cells
        lon_coast = fieldmesh_x[landcells[0][i1], landcells[1][i1]]
        lat_coast = fieldmesh_y[landcells[0][i1], landcells[1][i1]]

        # Calculate the distance from each land cell to the ocean cells
        dist_lon = lon_coast - fieldmesh_x[oceancells[0], oceancells[1]]
        dist_lat = lat_coast - fieldmesh_y[oceancells[0], oceancells[1]]

        # Combine the values to get an array of the distances to ocean cells
        dist_to_ocean = np.sqrt(np.power(dist_lon, 2) + np.power(dist_lat, 2))

        # Calculate the minimum distance from each land cell to the ocean cells
        min_dist = np.min(dist_to_ocean)

        # Find the indices of the minimum distances within the overall distance array
        i_min_dist = np.where(dist_to_ocean == min_dist)

        # If there is only one ocean cell that is closest...
        if len(i_min_dist[0]) == 1:
            # Find the lon and lat of the ocean cell
            lon_ocean = fieldmesh_x[oceancells[0][i_min_dist[0][0]], oceancells[1][i_min_dist[0][0]]]
            lat_ocean = fieldmesh_y[oceancells[0][i_min_dist[0][0]], oceancells[1][i_min_dist[0][0]]]

            # Calculate the norm (avoid division by zero)
            norm = np.sqrt((lon_ocean - lon_coast)**2 + (lat_ocean - lat_coast)**2)
            if norm != 0:
                # Create a vector field
                vectorfield_x[landcells[0][i1], landcells[1][i1]] = (lon_ocean - lon_coast) / norm
                vectorfield_y[landcells[0][i1], landcells[1][i1]] = (lat_ocean - lat_coast) / norm

        # If there are multiple ocean cells that are closest...
        elif len(i_min_dist[0]) > 1:
            # Take the mean of the lons and lats of all the cells
            lon_ocean = np.mean(fieldmesh_x[oceancells[0][i_min_dist], oceancells[1][i_min_dist]])
            lat_ocean = np.mean(fieldmesh_y[oceancells[0][i_min_dist], oceancells[1][i_min_dist]])

            # Calculate the norm (avoid division by zero)
            norm = np.sqrt((lon_ocean - lon_coast)**2 + (lat_ocean - lat_coast)**2)
            if norm != 0:
                # Create a vector field
                vectorfield_x[landcells[0][i1], landcells[1][i1]] = (lon_ocean - lon_coast) / norm
                vectorfield_y[landcells[0][i1], landcells[1][i1]] = (lat_ocean - lat_coast) / norm

    return vectorfield_x, vectorfield_y, Lons, Lats

# endregion

## region Create Fieldset
def create_fieldset(hydro_data, landmask, grid_ids):
    """
    Creates a FieldSet from hydrodynamic data with unbeaching vectors.

    Parameters:
    -----------
    hydro_data : xarray.Dataset
        Hydrodynamic data with U, V velocities
    landmask : numpy.ndarray
        Land mask array
    grid_ids : numpy.ndarray
        Grid ID array for tracking

    Returns:
    --------
    fieldset : parcels.FieldSet
        Complete fieldset ready for simulation
    """

    # Step 1: Open dataset
    hydro_data_copy = hydro_data.copy()

    # Step 2: Proceed with fieldset creation
    var = hydro_data_copy.variables["uo"]
    num_dims = len(var.shape)

    if num_dims == 3:
        dimensions = {'lon': 'lon', 'lat': 'lat', 'time': 'time'}
    elif num_dims == 4:
        dimensions = {'lon': 'lon', 'lat': 'lat', 'time': 'time', 'depth': 'depth'}
    else:
        raise ValueError("Unsupported data dimensionality: {}".format(num_dims))

    variables = {"U": "uo", "V": "vo"}

    # Force garbage collection before creating the fieldset
    gc.collect()

    # Step 4: Create the FieldSet (replace FieldSet creation logic based on your needs)
    fieldset = FieldSet.from_xarray_dataset(
        ds=hydro_data_copy,
        variables=variables,
        dimensions=dimensions,
        mesh='spherical'
    )

    # Step 5: Calculate all necessary components
    landvector_U, landvector_V, Lons, Lats = reflective_boundary(fieldset, landmask)

    # Step 6: Add Landmask
    fieldset.add_field(Field("landmask", landmask, lon=fieldset.U.grid.lon, lat=fieldset.U.grid.lat, mesh="spherical",
                             interp_method='nearest'))

    # Step 7: Add Grid IDs
    fieldset.add_field(Field("grid_ids", grid_ids, lon=fieldset.U.grid.lon, lat=fieldset.U.grid.lat, mesh="spherical",
                             interp_method='nearest'))

    # Step 8: Add reflective vector fields
    # Convert to Field objects
    U_land = Field('U_land', landvector_U, lon=Lons, lat=Lats, fieldtype='U', mesh='spherical',
                   interp_method='linear_invdist_land_tracer')
    V_land = Field('V_land', landvector_V, lon=Lons, lat=Lats, fieldtype='V', mesh='spherical',
                   interp_method='linear_invdist_land_tracer')

    # Add to the fieldset
    fieldset.add_field(U_land)
    fieldset.add_field(V_land)

    vectorfield_reflective = VectorField('UV_reflect', U_land, V_land)  # Combine and convert to vector field

    fieldset.add_vector_field(vectorfield_reflective)  # Add to the fieldset

    return fieldset
# endregion

# endregion

# region Create the ParticleSet
## region Setup Function
### Creating Custom Kernels
def create_custom_kernels(pset):
    """
    Create combined kernel for particle advection and tracking.

    Parameters
    ----------
    pset : ParticleSet
        The OceanParcels ParticleSet to create kernels for

    Returns
    -------
    Kernel
        Combined kernel for simulation execution
    """
    from parcels import StatusCode

    def DeleteErrorParticle(particle, fieldset, time):
        if particle.state == StatusCode.ErrorOutOfBounds:
            particle.delete()

    def TotalDistance(particle, fieldset, time):
        """Calculate the distance travelled by the particle in latitude and longitude."""
        lat_dist = (particle.lat - particle.prev_lat) * 1.11e2
        lon_dist = (
                (particle.lon - particle.prev_lon)
                * 1.11e2
                * math.cos(particle.lat * math.pi / 180)
        )
        # Calculate the total Euclidean distance travelled by the particle
        particle.distance += math.sqrt(math.pow(lon_dist, 2) + math.pow(lat_dist, 2))

        # Set the stored values for the next iteration
        particle.prev_lon = particle.lon
        particle.prev_lat = particle.lat

    def Sample_land(particle, fieldset, time):
        """Sample if the particle is on land."""
        particle.on_land = fieldset.landmask[time, particle.depth, particle.lat, particle.lon]

    def Grid_id(particle, fieldset, time):
        """Sample if the particle is on land."""
        particle.grid_id = fieldset.grid_ids[time, particle.depth, particle.lat, particle.lon]

    def Reflect(particle, fieldset, time):
        """Update particle position if it is on land."""
        if particle.on_land == 1:  # If the particle is on land
            (u_land, v_land) = fieldset.UV_reflect[
                time, particle.depth, particle.lat, particle.lon]  # land U and V velocities
            particle.lon += u_land * particle.dt # Update lon with land U velocity
            particle.lat += v_land * particle.dt # Update lat with land V velocity
            particle.on_land = 0  # The particle is no longer on land

    # Build kernels using sequential addition to avoid potential issues
    try:
        kernel = pset.Kernel(AdvectionRK4)
        kernel = kernel + pset.Kernel(TotalDistance)
        kernel = kernel + pset.Kernel(DiffusionUniformKh)
        kernel = kernel + pset.Kernel(Sample_land)
        kernel = kernel + pset.Kernel(Reflect)
        kernel = kernel + pset.Kernel(Grid_id)
        kernel = kernel + pset.Kernel(DeleteErrorParticle)
        return kernel
    except Exception as e:
        print(f"Warning: Kernel creation issue: {e}")
        print("Attempting alternative kernel combination method...")
        # Alternative: Create as list
        kernels = [
            pset.Kernel(AdvectionRK4),
            pset.Kernel(TotalDistance),
            pset.Kernel(DiffusionUniformKh),
            pset.Kernel(Sample_land),
            pset.Kernel(Reflect),
            pset.Kernel(Grid_id),
            pset.Kernel(DeleteErrorParticle)
        ]
        combined = kernels[0]
        for k in kernels[1:]:
            combined = combined + k
        return combined
# endregion

## region Create the ParticleSet
def create_pset(fieldset, meshgrid, runtime):
    """
    Creates a ParticleSet with staggered release intervals specified by the user.

    Interactively configures a ParticleSet by asking the user for release
    frequency, number of particles, and horizontal diffusivity. Adjusts
    runtime to ensure an integer number of release cohorts.

    Parameters:
    -----------
    fieldset : parcels.FieldSet
        The fieldset for particle tracking
    meshgrid : numpy.ndarray
        2D array with [lon, lat] release locations
    runtime : int
        Total runtime in days

    Returns:
    --------
    pset : parcels.ParticleSet
        Configured particle set
    adjusted_runtime : int
        Actual runtime in days
    interval : int
        Release interval in days
    repeatdt : timedelta
        Release interval

    Notes:
    ------
    - Runtime is adjusted upward to ensure even release intervals
    - User is prompted for release frequency (daily, weekly, monthly, custom)
    - User is prompted for particle numbers per location per cohort
    - User is prompted for horizontal diffusivity (Kh)
    - Horizontal diffusivity (Kh) adds random walk to particle motion
    - Default Kh = 1 m²/s represents sub-grid turbulent mixing
    """
    # Step 1:  Extract locations
    lats = meshgrid[:, 1].flatten() # Assuming the latitude is in the second column
    lons = meshgrid[:, 0].flatten() # Assuming the longitude is in the first column
    num_locations = len(lats) # Number of release locations
    runtime_td = timedelta(days=runtime)   # Assuming runtime is the number of days, transforms runtime to timedelta object

    # Step 2: Select the number of release intervals
    subsection_header("Release Strategy")
    print("Particles can be released at regular intervals over the duration of a species' spawning window.")

    interval_mapping = {
        'daily': timedelta(days=1),
        'weekly': timedelta(weeks=1),
        'biweekly': timedelta(weeks=2),
        'monthly': timedelta(days=30)
    }

    while True:
        release_interval = input(
            "\nEnter release interval ('daily', 'weekly', 'biweekly', 'monthly', or custom days) [default: weekly]: "
        ).strip().lower()

        if not release_interval:
            release_interval = 'weekly'
            print("Using default: weekly")
            break
        elif release_interval in interval_mapping:
            break
        elif release_interval.isdigit():
            break
        else:
            print(f"Invalid input: '{release_interval}'")
            print("Please enter 'daily', 'weekly', 'biweekly', 'monthly', or a number of days (e.g., '14').")

    interval = interval_mapping.get(
        release_interval,
        timedelta(days=int(release_interval)) if release_interval.isdigit() else timedelta(weeks=1)
    )

    # Calculate how many intervals fit into the runtime
    # Adjusted runtime: To ensure that the number of intervals definitely fit into the overall runtime
    # E.g. Runtime= 120 days, release interval = 7 days, 120/7 = 17.14 release cohorts BUT not a round number so it gets adjusted to 18 days so all cohorts fit
    num_intervals = math.ceil(runtime_td / interval)
    adjusted_runtime = num_intervals * interval

    data_box("Release Settings", [
        f"Release interval: {interval.days} days",
        f"Number of release cohorts: {num_intervals}",
        f"Adjusted runtime: {adjusted_runtime.days} days to fit all release cohorts"
    ])

    # Step 3: Asks the user to set the number of particles
    subsection_header("Number of Particles")
    print(f"Specify the number of particles to release per location per release cohort.")
    print(f"\nExample: If you enter 100:")
    print(f"Per location, per cohort: 100 particles will be released")
    print(f"Per location, total particles released: 100 × {num_intervals} = {100 * num_intervals:} particles")
    print(f"All locations, total particles released: {100 * num_intervals * num_locations:} particles")

    default_npart = 100
    npart = get_positive_integer(f"\nEnter the number of particles per location per cohort", default=default_npart, min_val=1)

    # Step 4: Calculate totals and confirm
    particles_per_location = npart * num_intervals
    total_particles = particles_per_location * num_locations

    # Step 5: Summary of setup parameters
    data_box("The Particle Setup", [
        f"Release cohorts: {num_intervals}",
        f"Interval: Every {interval.days} days",
        f"Release locations: {num_locations}",
        f"Particles per location per cohort: {npart}",
        f"Total particles per location: {particles_per_location}",
        f"Total particles: {total_particles}",
        f"Adjusted runtime: {adjusted_runtime.days} days"
    ])

    confirm = input("\nProceed with these settings? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Cancelled. Please run workflow_step(3) again to reconfigure.")
        return None, None, None, None

    # Step 6: Setup particle arrays
    # Set repeatdt
    repeatdt = interval

    # Adjust per-location particles using itertools
    lat = list(itertools.chain.from_iterable(itertools.repeat(x, npart) for x in lats))
    lon = list(itertools.chain.from_iterable(itertools.repeat(x, npart) for x in lons))

    # Step 7: Random Walk via Horizontal Diffusivity
    default_kh = 1
    while True:
        try:
            print("\nThis package allows you to add a horizontal diffusivity value.")
            print("\nHorizontal diffusivity represents small-scale random motion caused by turbulence and applies evenly to all ")
            print("particles in the simulation.")
            kh_input = input(f"\nEnter horizontal diffusivity (kh) in m²/s [default: {default_kh}]: ")
            kh = float(kh_input) if kh_input else default_kh
            if kh < 0:
                print("Error: Diffusivity must be positive.")
                continue
            if kh > 1000:
                print("Warning: Very high diffusivity (>1000). Are you sure? (yes/no)")
                confirm = input().strip().lower()
                if confirm not in ['yes', 'y']:
                    continue
            break
        except ValueError:
            print("Error: Enter a valid number.")

    fieldset.add_constant_field("Kh_zonal", kh, mesh="spherical")
    fieldset.add_constant_field("Kh_meridional", kh, mesh="spherical")

    # Step 8: Create particle class
    extra_vars = [
        Variable("distance", initial=0.0, dtype=np.float32),
        Variable("prev_lon", dtype=np.float32, to_write=False, initial=attrgetter("lon")),
        Variable("prev_lat", dtype=np.float32, to_write=False, initial=attrgetter("lat")),
        Variable('on_land', dtype=np.int32, initial=0),
        Variable('grid_id', dtype=np.int32, initial=0)
    ]

    Particle = JITParticle.add_variables(extra_vars)

    # Step 9: Create ParticleSet
    pset = ParticleSet(
        fieldset=fieldset,
        pclass=Particle,
        lon=lon,
        lat=lat,
        repeatdt=repeatdt
    )

    return pset, adjusted_runtime.days, interval.days, repeatdt
# endregion
# endregion

# region Execute Simulation
## region Memory Management
### Creating a memory monitoring class
class MemoryMonitor:
    """
    Context manager for monitoring memory usage during operations.

    Usage:
        with MemoryMonitor("Simulation execution") as monitor:
            # Your memory-intensive operation here
            result = some_function()

    This will print memory usage before and after the operation and warn
    if memory usage exceeds 85%.
    """

    def __init__(self, operation_name):
        self.operation_name = operation_name

    def __enter__(self):
        gc.collect()
        self.start_memory = psutil.virtual_memory().percent
        print(f"{self.operation_name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()
        end_memory = psutil.virtual_memory().percent

        # Only warn if critically high - no other output
        if end_memory > 85:
            print(f"\nWarning: High memory usage ({end_memory:.1f}%)")

### Calculate output file size
def get_zarr_size(zarr_path):
    '''
    Calculate total size of a Zarr directory (all files within).
    '''
    total_size = 0
    if os.path.exists(zarr_path) and os.path.isdir(zarr_path):
        for dirpath, dirnames, filenames in os.walk(zarr_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
    return total_size / (1024 ** 3)  # Convert to GB

### System memory configuration
def check_system_memory():
    """
    Check available system memory and display status.

    Queries system RAM using psutil and displays total, available and
    percentage used. This information is used to configure memory-safe
    processing parameters.

    Returns:
    --------
    available_gb : float
        Available memory in gigabytes
    """
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024 ** 3)
    total_gb = memory.total / (1024 ** 3)

    print(f"System Memory:")
    print(f" - Total: {total_gb:.1f} GB")
    print(f" - Available: {available_gb:.1f} GB")
    print(f" - Used: {memory.percent:.1f}%")

    return available_gb

### Get safe chunk size based on user's computing setup
def get_safe_chunk_size(available_gb, adjusted_runtime, output_dt_hours):
    """
    Calculates safe chunk size for post-processing large trajectory files.

    Estimates memory requirements based on simulation length and available
    RAM, then calculates an appropriate chunk size to process trajectories
    without exceeding memory limits.

    Parameters:
    -----------
    available_gb : float
        Available system memory in gigabytes
    adjusted_runtime : int
        Simulation runtime in days
    output_dt_hours: int
        User-specified output frequency (e.g., 1, 6, 24 hours)

    Returns:
    --------
    chunk_size : int
        Number of trajectories to process per chunk (500-20000)

    Notes:
    ------
    - Uses 60% of available memory for safety
    - Accounts for longer simulations requiring more memory per particle
    - Conservative estimate: ~1GB per 1000 particles for 30-day simulation
    - Longer runtimes scale linearly with memory requirements
    """

    # Base estimate: memory scales with simulation length
    # For 30-day simulation with daily output: ~30 timesteps
    # Assume daily output (conservative for most use cases)
    estimated_timesteps = int((adjusted_runtime * 24) / output_dt_hours)

    # Memory per trajectory in MB
    # 10 vars × 8 bytes × timesteps × 1.6 overhead / (1024^2)
    mb_per_trajectory = (10 * 8 * estimated_timesteps * 1.6) / (1024 * 1024)

    # Convert to GB per 1000 trajectories
    gb_per_1000_trajectories = (mb_per_trajectory * 1000) / 1024

    # Use 60% of available memory for safety
    safe_memory_gb = available_gb * 0.6

    # Calculate safe number of trajectories
    safe_trajectories = int((safe_memory_gb / gb_per_1000_trajectories) * 1000)

    # Apply reasonable bounds
    # Minimum: 500 (always process at least this many)
    # Maximum: 10000 (beyond this, diminishing returns)
    chunk_size = max(500, min(safe_trajectories, 10000))

    print(f"\nMemory Calculation:")
    print(f"  - Simulation length: {adjusted_runtime} days")
    print(f"  - Output timesteps: {estimated_timesteps}")
    print(f"  - Memory per 1000 trajectories: {gb_per_1000_trajectories:.2f} GB")

    return chunk_size

### Guidelines for setting the correct chunk size based on the user's RAM
def get_memory_config(available_gb, adjusted_runtime, output_dt_hours):
    """
    Gets memory-optimized configuration based on available RAM and simulation parameters.

    Parameters:
    -----------
    available_gb : float or None
        Available memory in GB. If None, automatically detected.
    adjusted_runtime : int or None
        Simulation runtime in days
    output_dt_hours : int
        Output frequency in hours (default: 24)

    Returns:
    --------
    config : dict
        Configuration with dynamically calculated chunk_size
    """
    if available_gb is None:
        available_gb = psutil.virtual_memory().available / (1024 ** 3)

    # Get total RAM for system classification
    total_gb = psutil.virtual_memory().total / (1024 ** 3)

    # Determine system type based on total RAM (not available)
    if total_gb < 12:
        system_type = 'limited'
        use_chunked = True
    elif total_gb < 24:
        system_type = 'moderate'
        use_chunked = True
    else:
        system_type = 'high_performance'
        use_chunked = True  # Always use chunking for safety

    # Calculate dynamic chunk size if runtime provided
    if adjusted_runtime is not None:
        chunk_size = get_safe_chunk_size(available_gb, adjusted_runtime, output_dt_hours)
    else:
        # Fallback to conservative defaults
        if available_gb < 9:
            chunk_size = 1000
        elif available_gb < 17:
            chunk_size = 5000
        else:
            chunk_size = 10000

    return {
        'chunk_size': chunk_size,
        'system_type': system_type,
        'use_chunked_processing': use_chunked
    }

### Load simulation output back in Python using chunking
def process_sim_results(zarr_path, chunk_size=None, runtime_days=None):
    """
    Load and process particle trajectory data from Zarr files.

    Reads simulation output from Zarr format, handles memory-efficient chunked
    processing for large datasets and converts to pandas DataFrame. Implements
    automatic chunking based on available system memory and dataset size.

    Parameters
    ----------
    zarr_path : str
        Path to the Zarr directory containing simulation output. Created by
        ParticleSet.execute() with output_file parameter
    chunk_size : int, optional
        Number of particles to process in each chunk. If None, automatically
        calculated based on available system memory (default: None)
    runtime_days : int, optional
        Total duration of simulation in days. Used for memory estimation.
        If None, extracted from data or estimated (default: None)

    Returns
    -------
    pandas.DataFrame
        Processed trajectory data with columns:
        - 'trajectory' : int
            Unique particle ID
        - 'time' : datetime64
            Time of observation
        - 'lat' : float
            Particle latitude (degrees)
        - 'lon' : float
            Particle longitude (degrees)
        - 'z' : float
            Particle depth (meters, negative below surface)
        - 'cell_id' : int
            Release grid cell ID
        - 'species' : str
            Species name
        - 'release_id' : int
            Release event ID (for multiple releases per cell)
        - Additional fields from particle variables

    Notes
    -----
    - Zarr format allows efficient partial loading of large datasets
    - Automatic chunking prevents memory overflow on systems with limited RAM
    - Progress is displayed during chunked loading
    - Invalid/NaN particle positions are automatically removed
    - Time is converted from numeric to datetime format

    Chunking Algorithm
    ------------------
    1. Check available system memory
    2. Estimate memory per particle (based on runtime)
    3. Calculate safe chunk size
    4. Load trajectories in chunks
    5. Concatenate chunks into single DataFrame
    6. Clean and validate data

    Data Cleaning
    -------------
    - Remove particles with all-NaN coordinates (beached permanently)
    - Remove timesteps with missing data
    - Convert time to datetime objects
    - Ensure consistent data types across columns

    Performance Tips
    ---------------
    - Chunked processing adds ~10% overhead but prevents crashes
    - SSD storage significantly faster than HDD for Zarr access
    - Parallel processing possible by loading chunks simultaneously

    Common Issues
    -------------
    - Memory error: Reduce chunk_size manually
    - Slow loading: Check disk I/O speed, consider SSD
    - Missing data: Verify simulation completed successfully
    """
    print("Loading simulation results...")

    # Open dataset with chunking
    with MemoryMonitor("Opening zarr dataset"):
        ds = xr.open_zarr(zarr_path)

        # Get basic info
        n_trajectories = ds.dims.get('trajectory', 0)
        n_timesteps = ds.dims.get('obs', 0)

        data_box("Dataset properties", [
            f"Trajectories: {n_trajectories}",
            f"Time steps: {n_timesteps}",
            f"The dataset will be imported in chunks of {chunk_size} trajectories"
        ])

    # Process in chunks
    all_chunks = []
    n_chunks = (n_trajectories + chunk_size - 1) // chunk_size

    for chunk_idx in range(0, n_trajectories, chunk_size):
        chunk_end = min(chunk_idx + chunk_size, n_trajectories)
        current_chunk = chunk_end - chunk_idx
        chunk_num = (chunk_idx // chunk_size) + 1

        print(f"Importing chunk {chunk_num}/{n_chunks} ({current_chunk:} trajectories)...")

        with MemoryMonitor(f"Chunk {chunk_num}"):
            # Load just this chunk
            chunk_ds = ds.isel(trajectory=slice(chunk_idx, chunk_end))
            chunk_df = chunk_ds.to_dataframe().reset_index()

            # Drop rows where essential columns are NaN
            chunk_df = chunk_df.dropna(subset=['trajectory', 'time', 'lat', 'lon'])

            # Store this chunk
            all_chunks.append(chunk_df)

            # Clean up immediately
            del chunk_ds, chunk_df
            gc.collect()

    # Combine all chunks
    with MemoryMonitor("Combining chunks"):
        traj_df = pd.concat(all_chunks, ignore_index=True)

        # Clean up chunk list
        del all_chunks
        gc.collect()

    print("\nAll chunks imported successfully!")

    return traj_df

# endregion

## region Setup Functions
### Release ID/Cohort Assignment
def assign_release_id(traj_df, interval):
    """
    Assigns a release_id to each trajectory based on the first observation time (obs=0).
    Groups particles by release cohorts using the release interval.

    Parameters:
    -----------
    traj_df : DataFrame
        Trajectory data with columns 'trajectory', 'obs', 'time'
    interval : int
        Interval between releases in days

    Returns:
    --------
    traj_df : DataFrame
        Updated DataFrame with 'release_id' column

    Notes:
    ------
    - Uses memory-efficient merge operation
    - release_id starts at 1 (cohort 1, 2, 3, ...)
    """

    with MemoryMonitor("Release ID assignment"):
        # Step 1: Identify the first observation time for each trajectory (obs=0)
        first_obs = traj_df[traj_df['obs'] == 0][['trajectory', 'time']].copy()

        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(first_obs['time']):
            first_obs['time'] = pd.to_datetime(first_obs['time'])

        # Step 2: Find the earliest release time (simulation start)
        earliest_release = first_obs['time'].min()

        # Step 3: Calculate release cohort based on days since simulation start
        first_obs['days_since_start'] = (first_obs['time'] - earliest_release).dt.total_seconds() / (24 * 3600)

        # Step 4: Group by release interval (e.g., every 7 days for weekly releases)
        first_obs['release_id'] = (first_obs['days_since_start'] / interval).round().astype(int) + 1

        # Step 5: Merge the release_id back to the original DataFrame
        traj = traj_df.merge(first_obs[['trajectory', 'release_id']], on='trajectory', how='left')

        # Summary statistics
        n_releases = first_obs['release_id'].nunique()
        print(f"Assigned {n_releases} release cohorts")

    return traj

# endregion

## region Run Simulation
def execute_simulation(pset, kernels, adjusted_runtime, interval):
    """
    Executes the OceanParcels simulation with specified kernels, handles progress
    monitoring, implements error handling and manages output file creation. This
    is the main computational engine of the connectivity modelling.

    Parameters
    ----------
    pset : parcels.ParticleSet
        Configured particle set containing:
        - Initial particle positions
        - Release timing information
        - Species assignments
        - Custom particle variables
    kernels : parcels.Kernel or list of parcels.Kernel
        Compiled kernel(s) defining particle behavior:
        - Advection (AdvectionRK4)
        - Diffusion
        - Boundary conditions
        - Settlement behavior
        - Mortality
        Can be single kernel or combined kernel list
    adjusted_runtime : float
        Total simulation duration in seconds. Typically calculated as:
        runtime_days * 24 * 60 * 60
    interval : float
        Output timestep in seconds. Particle positions saved every interval.
        Typical value: 3600 (1 hour) to 86400 (1 day)

    Returns
    -------
    str
        Path to the Zarr output file containing trajectory data

    Notes
    -----
    - Simulation runs with real-time progress bar
    - Handles particle beaching via recovery kernel
    - Catches and logs particle errors without stopping simulation
    - Output automatically compressed in Zarr format
    - Memory-efficient streaming output (particles written incrementally)

    Simulation Execution
    --------------------
    pset.execute() parameters:
    - runtime: Total duration (seconds)
    - dt: Integration timestep (seconds, typically 300-width0)
    - interval: Output timestep (seconds)
    - output_file: Zarr file path
    - recovery: Error recovery kernel
    - verbose_progress: Progress updates

    Output File Structure
    --------------------
    Zarr directory contains:
    - trajectory: Particle ID array
    - time: Time coordinate array
    - lat, lon, z: Position arrays
    - Custom variables: species, cell_id, release_id, etc.
    - Metadata: Simulation parameters, timing info

    Performance Considerations
    --------------------------
    - Typical runtime: 1-10 minutes per simulated day (depends on particle count)
    - Scales linearly with number of particles
    - Output interval affects file size but not computation time

    Common Issues
    -------------
    - Simulation crashes: Check memory, reduce particle count or domain size
    - Slow performance: Reduce output interval, increase dt
    - Many particle errors: Check hydrodynamic data quality, boundary conditions
    - Output file corruption: Ensure sufficient disk space
    """
    try:
        subsection_header("Setting up the Simulation")

        # Step 1: Setting the output path
        global project_dirs
        if project_dirs is not None:
            default_sim_path = os.path.join(project_dirs['simulation'], "simulation_output.zarr")
        else:
            default_sim_path = os.path.join(get_default_dir('simulation'), "simulation_output.zarr")

        output_path = input(
            f"\nEnter the output path and file name for the simulation output\n"
            f"  [default: {default_sim_path}]: "
        ).strip() or default_sim_path
        if not output_path.endswith(".zarr"):
            output_path += ".zarr"

        # Step 2: Get output settings from user
        subsection_header("Output Frequency")
        info_box([
            "In a particle simulation, particles are advected at regular timesteps (e.g. every hour).",
            "You can save output less frequently than the simulation timestep to reduce file size.",
            "",
            "For example: Timestep = 1 hour, Output frequency = 6 hours.",
            "  - Higher frequency = more temporal detail but larger files",
            "  - Lower frequency = less detail but smaller files",
            "  - Higher timestep = more detail but slower simulation runtime",
            "  - Lower timestep = less detail but faster simulation runtime"
        ])

        output_dt_hours = int(input("Enter output frequency in hours [default: 24]: ") or "24")
        dt_minutes = int(input("Enter time step (dt) in minutes [default: 60]: ") or "60")

        # Step 3: Check system memory
        subsection_header("System Memory Check")
        info_box([
            "This package checks your available RAM to optimise processing.",
            "Large simulations generate significant data that must fit in memory.",
            "The package will automatically adjust chunk sizes to prevent crashes."
        ])

        available_gb = check_system_memory()
        total_gb = psutil.virtual_memory().total / (1024 ** 3)
        config = get_memory_config(available_gb, adjusted_runtime, output_dt_hours)

        print(f"\nSystem: {total_gb:.0f} GB total RAM, {available_gb:.1f} GB currently available")
        print(f"Processing will use chunks of {config['chunk_size']:} trajectories to prevent memory issues.")

        # Warn if potentially memory-intensive on limited systems
        if config['system_type'] == 'limited' and output_dt_hours < 12:
            print("\nNote: High-frequency output may require more memory during processing.")
            print("Consider output frequency ≥ 12 hours if you encounter memory issues.")

        output_dt = timedelta(hours=output_dt_hours)
        dt = timedelta(minutes=dt_minutes)
        runtime = timedelta(days=adjusted_runtime)

        data_box("Simulation settings", [
            f"Runtime: {adjusted_runtime} days",
            f"Output frequency: {output_dt_hours} hours",
            f"Time step: {dt_minutes} minutes"
        ])

        # Step 4: Run the simulation with memory monitoring
        with MemoryMonitor("Running OceanParcels simulation..."):
            # Memory-optimized chunk size for zarr output
            chunk_particles = min(config['chunk_size'], len(pset))

            output_file = pset.ParticleFile(
                name=output_path,
                outputdt=output_dt,
                chunks=(chunk_particles, 1)
            )

            pset.execute(
                kernels,
                runtime=runtime,
                dt=dt,
                output_file=output_file
            )

        # Step 5: Load results with chunked approach
        print("\nProcessing simulation results...")

        # Check if we need chunked processing
        file_size_gb = get_zarr_size(output_path)

        # Always use chunked processing for memory safety
        print(f"\nOutput file size: {file_size_gb:.2f} GB")
        print(f"Using memory-safe chunked loading ({config['chunk_size']:} trajectories per chunk)")
        traj_df = process_sim_results(output_path, config['chunk_size'], adjusted_runtime)

        # Step 6: Add release IDs for staggered releases
        traj = assign_release_id(traj_df, interval)

        print("\nSimulation complete!")
        print(f"Final dataset: {len(traj):} rows, {traj['trajectory'].nunique():} trajectories")

        return output_path, traj

    except MemoryError:
        print("\nError: Out of Memory")
        print("Your system ran out of memory during the simulation.")
        print("\nSuggestions:")
        print("  1. Reduce the number of particles")
        print("  2. Decrease output frequency (save less often)")
        print("  3. Reduce simulation runtime")
        print("  4. Close other applications")
        return None, None

    except Exception as e:
        error_msg = str(e)
        print("\nError: The Simulation Failed")
        print(f"Error message: {error_msg}")

        if "time domain" in error_msg.lower() or "time_extrapolation" in error_msg.lower():
            print("\nError: Time domain out of bounds")
            print("The simulation ran longer than your hydrodynamic data coverage.")
            print("\nTo fix:")
            print("  1. Re-run workflow_step(1)")
            print("  2. Extend the end date of time_range by a few days")
            print("  3. Re-download hydrodynamic data")
        else:
            print("\nIf this persists, check the hydrodynamic data covers the expected time_range.")
        return None, None
# endregion
# endregion

# region Post-Processing
## Setup Functions
### Assign species unique trajectory id numbers and filter based on spawning window
def spawning_filter(traj, traits):
    """
    Assigns species to trajectories based on spawning windows.

    Creates species-specific trajectory IDs to handle overlapping spawning
    periods. A single trajectory can belong to multiple species if their
    spawning windows overlap with the particle's release time.

    Parameters:
    -----------
    traj : pandas.DataFrame
        Trajectory data with columns: trajectory, time, obs
    traits : list of dict
        Species trait data including spawning windows

    Returns:
    --------
    spawning_dfs : dict
        Dictionary mapping species names to DataFrames with species assignments.

    Notes:
    ------
    - Original trajectory IDs are preserved in 'original_trajectory' column
    - New trajectory IDs are species-specific (trajectory + species_offset)
    - Only particles released during spawning windows are retained
    - Handles overlapping spawning periods
    """
    gc.collect()

    print("\nAssigning species to trajectories...")

    # Step 1: Calculate number of species
    species_list = [t['species_name'] for t in traits]
    n_species = len(species_list)

    print(f"\nCreating unique trajectory IDs for {n_species} species...")

    with MemoryMonitor("Species assignment"):
        # Convert time if needed
        if not pd.api.types.is_datetime64_any_dtype(traj['time']):
            traj['time'] = pd.to_datetime(traj['time'])

        # Store original trajectory ID
        traj['original_trajectory'] = traj['trajectory']

        # Get offset for unique IDs
        max_traj_id = traj['trajectory'].max()
        offset_multiplier = 10 ** (len(str(max_traj_id)) + 1)

        species_dataframes = []

    # Process each species separately
    for i, species_traits in enumerate(traits):
        species_name = standardise_species_name(species_traits['species_name'])

        # Get spawning window for this species
        spawn_start = pd.to_datetime(species_traits['spawning_start'], dayfirst=True)
        spawn_end = pd.to_datetime(species_traits['spawning_end'], dayfirst=True)

        # Find trajectories that started in this species' spawning window
        start_mask = traj['obs'] == 0
        time_mask = (traj.loc[start_mask, 'time'] >= spawn_start) & \
                    (traj.loc[start_mask, 'time'] <= spawn_end)

        species_trajectories = traj.loc[start_mask][time_mask]['original_trajectory'].unique()

        if len(species_trajectories) > 0:
            # Create species subset
            species_data = traj[traj['original_trajectory'].isin(species_trajectories)].copy()
            species_data['trajectory'] = species_data['original_trajectory'] + (i * offset_multiplier)
            species_data['species'] = species_name

            species_dataframes.append(species_data)
            print(f"   {species_name}: {len(species_trajectories)} trajectories")

    # Combine all species dataframes
    # This will have duplicated trajectories with different IDs if spawning windows overlap
    # Convert species dataframes to dictionary
    if species_dataframes:
        spawning_dfs = {}
        for species_data in species_dataframes:
            species_name = species_data['species'].iloc[0]  # Get species name from data
            spawning_dfs[species_name] = species_data
        print(f"Species assignment complete: {len(spawning_dfs)} species with trajectories.")
    else:
        print("No trajectories found in any spawning windows!")
        spawning_dfs = {}

    return spawning_dfs

### Planktonic larval duration filter and settlement habitat suitability filter
def pld_filter(spawning_dfs, traits, settlement_grids, grid_ids, buffer):
    """
    Filters trajectories based on planktonic larval duration (PLD) and settlement location.

    For each species, removes trajectories that:
    1. Don't reach minimum PLD before settling
    2. Exceed maximum PLD without settling
    3. Don't settle within species-specific settlement habitat

    Parameters:
    -----------
    spawning_dfs : pandas.DataFrame
        Species-assigned trajectories from spawning_filter()
    traits : list of dict
        Species trait data including min_PLD and max_PLD
    settlement_grids : dict or None
        Dictionary mapping species names to settlement grid DataFrames.
        If None (all sites strategy), all coastal sites are valid.
    grid_ids : numpy.ndarray
        2D array of grid IDs for the study area
    buffer : int, default=1
        Settlement buffer in grid cells (creates 3×3 patch around each site)

    Returns:
    --------
    pld_filtered : dict
        Dictionary mapping species names to DataFrames of trajectories
        that meet PLD and settlement criteria

    Notes:
    ------
    - Minimum PLD ensures propagules are competent to settle
    - Maximum PLD represents mortality
    - Buffer allows settlement in adjacent cells (accounts for GPS error)
    - Returns empty DataFrame if no valid settlements found
    """
    pld_dfs = {}

    # Helper function to get adjacent grids with buffer
    def get_adjacent_grids(grid_id, grid_matrix, buffer):
        position = np.where(grid_matrix == grid_id)
        if len(position[0]) == 0:
            return []
        row, col = position[0][0], position[1][0]

        adjacent = []
        for i in range(-buffer, buffer + 1):
            for j in range(-buffer, buffer + 1):
                new_row, new_col = row + i, col + j
                if 0 <= new_row < grid_matrix.shape[0] and 0 <= new_col < grid_matrix.shape[1]:
                    adjacent.append(grid_matrix[new_row, new_col])
        return adjacent

    # Convert traits to DataFrame if it's a list (for .loc access)
    if isinstance(traits, list):
        traits_df = pd.DataFrame(traits)
    elif isinstance(traits, pd.DataFrame):
        traits_df = traits
    else:
        # traits might be a string (file path) - load it
        print(f"Warning: traits is {type(traits)}, attempting to load...")
        traits_df = pd.read_csv(traits)

    # Get all valid grid IDs for "all sites" strategy
    all_valid_grids = set(grid_ids[grid_ids > 0].flatten())

    for species_name, df in spawning_dfs.items():
        if df.empty:
            pld_dfs[species_name] = pd.DataFrame()
            continue

        # Standardise species name for consistent matching
        species_name_std = standardise_species_name(species_name)

        print(f"\nProcessing PLD filter for {display_species_name(species_name_std)}...")

        # Get PLD values - standardise traits names for matching
        species_match = traits_df['species_name'].apply(standardise_species_name) == species_name_std

        if not species_match.any():
            print(f"  Warning: Species '{species_name_std}' not found in traits. Skipping.")
            pld_dfs[species_name] = pd.DataFrame()
            continue

        min_pld = int(traits_df.loc[species_match, 'min_PLD'].values[0])
        max_pld = int(traits_df.loc[species_match, 'max_PLD'].values[0])
        print(f"  PLD window: {min_pld} - {max_pld} days")

        # Get settlement grids for this species
        if settlement_grids is None:
            print(f"  Using all {len(all_valid_grids)} sites for settlement (all sites strategy)")
            buffered_settlement = all_valid_grids
        else:
            # Find matching key using standardised names
            settlement_key = None
            for key in settlement_grids.keys():
                if standardise_species_name(key) == species_name_std:
                    settlement_key = key
                    break

            if settlement_key is not None:
                species_settlement = set(settlement_grids[settlement_key]['grid_id'])
                buffered_settlement = set()
                for gid in species_settlement:
                    buffered_settlement.update(get_adjacent_grids(gid, grid_ids, buffer))
                print(f"  Settlement sites: {len(species_settlement)} (with buffer: {len(buffered_settlement)})")
            else:
                print(f"  Warning: No settlement data for {display_species_name(species_name_std)}, using all sites.")
                buffered_settlement = all_valid_grids
        valid_trajectories = []
        # Start from at least day 1 to prevent immediate settlement at release site
        start_day = max(1, min_pld)
        settlement_summary = {day: 0 for day in range(start_day, max_pld + 1)}

        # Process each trajectory
        for traj_id, traj_subset in df.groupby('trajectory'):
            traj_subset = traj_subset.sort_values('time')
            spawning_time = pd.to_datetime(traj_subset.iloc[0]['time'])

            # Get release grid_id for this trajectory (first observation)
            release_obs = traj_subset[traj_subset['obs'] == 0]
            release_grid_id = int(release_obs['grid_id'].iloc[
                                      0]) if not release_obs.empty and 'grid_id' in release_obs.columns else None

            # Check each day from start_day to max_PLD
            # Start from at least day 1 to prevent immediate settlement at release site
            settlement_found = False
            for day in range(start_day, max_pld + 1):
                check_time = spawning_time + pd.Timedelta(days=day)

                # Find the observation closest to this time
                time_diff = abs(traj_subset['time'] - check_time)
                if len(time_diff) > 0:
                    closest_idx = time_diff.idxmin()
                    closest_obs = traj_subset.loc[closest_idx]

                    # Check if in suitable habitat
                    current_grid_id = closest_obs['grid_id']

                    # On day 1, don't allow settlement at exact release site
                    # (prevents particles from "settling" immediately where they started)
                    if day == 1 and release_grid_id is not None and current_grid_id == release_grid_id:
                        continue  # Skip to next day

                    if current_grid_id in buffered_settlement:
                        # Crop trajectory at settlement time
                        settled_traj = traj_subset[traj_subset['time'] <= closest_obs['time']].copy()
                        settled_traj['settlement_day'] = day
                        valid_trajectories.append(settled_traj)
                        settlement_summary[day] += 1
                        settlement_found = True
                        break

            if not settlement_found:
                # Particle died - no suitable habitat found within PLD window
                pass

        if valid_trajectories:
            pld_dfs[species_name] = pd.concat(valid_trajectories)
            print(f"  {len(valid_trajectories)} successful trajectories")
            # Only show non-zero settlement days with clearer formatting
            non_zero_summary = {k: v for k, v in settlement_summary.items() if v > 0}
            if non_zero_summary:
                summary_str = ", ".join([f"day {day}: {count}" for day, count in non_zero_summary.items()])
                print(f"  Successful trajectories per day: {summary_str}")
        else:
            pld_dfs[species_name] = pd.DataFrame()
            print(f"  No successful trajectories")

    return pld_dfs

### Release habitat suitability filter
def habitat_filter(pld_dfs, release_grids):
    """
    Filter trajectories to ensure release from species-specific habitat.

    Removes trajectories that were released from locations outside
    species-specific release habitat. Only applies when using species-
    specific release strategy (not 'all sites').

    Parameters:
    -----------
    pld_dfs : dict
        PLD-filtered trajectories from pld_filter(), mapping species names
        to DataFrames
    release_grids : dict or None
        Dictionary mapping species names to release grid DataFrames.
        If None (all sites strategy), no filtering is applied.

    Returns:
    --------
    final_processed : dict
        Dictionary mapping species names to filtered trajectory DataFrames

    Notes:
    ------
    - Only filters if release_grids is not None
    - Checks first observation (obs==0) against release grid
    - Returns all trajectories unchanged if release_grids is None
    """
    final_processed = {}

    for species_name, df in pld_dfs.items():
        if df.empty:
            final_processed[species_name] = pd.DataFrame()
            continue

        # Handle case where release_grids is None (all sites strategy)
        if release_grids is None:
            # No release constraint - keep all trajectories
            final_processed[species_name] = df
            n_traj = df['trajectory'].nunique()
            print(f"  {species_name}: {n_traj} trajectories (all sites release strategy)")
            continue

        # Check if species is in release_grids
        if species_name in release_grids:
            release_sites = set(release_grids[species_name]['grid_id'])
        else:
            # Try with different name format
            alt_name = species_name.replace('_', ' ') if '_' in species_name else species_name.replace(' ', '_')
            if alt_name in release_grids:
                release_sites = set(release_grids[alt_name]['grid_id'])
            else:
                # No release constraint for this species - keep all
                final_processed[species_name] = df
                n_traj = df['trajectory'].nunique()
                print(f"  {species_name}: {n_traj} trajectories (no release constraint)")
                continue

        # Filter to trajectories starting from valid release sites
        valid_trajectories = []
        for traj_id, traj_subset in df.groupby('trajectory'):
            start_obs = traj_subset[traj_subset['obs'] == 0]
            if start_obs.empty:
                continue  # Skip trajectories without obs=0
            start_grid = start_obs['grid_id'].iloc[0]

            if start_grid in release_sites:
                valid_trajectories.append(traj_subset)

        if valid_trajectories:
            final_processed[species_name] = pd.concat(valid_trajectories)
            print(f"  {species_name}: {len(valid_trajectories)} trajectories from release sites")
        else:
            final_processed[species_name] = pd.DataFrame()
            print(f"  {species_name}: No trajectories from valid release sites")

    return final_processed
# endregion

## region Complete post-processing function (wrapper function)
def post_process(traj, traits, release_grids, settlement_grids, grid_ids):
    """
    Apply species-specific temporal and spatial filters to trajectory data to create species-specific connectivity data.

    This is the main post-processing pipeline that transforms raw particle
    trajectories into species-specific connectivity data by applying:
    1. Spawning season filters (temporal)
    2. Planktonic larval duration filters (temporal)
    3. Habitat suitability filters (spatial)

    Returns connectivity matrices and filtered trajectories for further analysis.

    Parameters
    ----------
    traj: pandas.DataFrame
        Raw particle trajectory data from process_sim_results() with columns:
        - 'trajectory' : int
            Particle ID
        - 'time' : datetime64
            Observation timestamp
        - 'lat', 'lon' : float
            Particle position
        - 'cell_id' : int
            Release cell ID
        - 'species' : str
            Species name
        - 'release_id' : int
            Release event ID
    traits: list of dict
        Species biological traits with each dict containing:
        - 'species_name' : str
        - 'spawning_start', 'spawning_end' : str (DD/MM/YYYY format)
        - 'min_PLD', 'max_PLD', 'average_PLD' : int (days)
        - 'fecundity' : int
    release_grids : dict
        Dictionary mapping species names to release grid DataFrames
    settlement_grids : dict
        Dictionary mapping species names to settlement grid DataFrames
    grid_ids : dict
        Dictionary mapping species names to arrays of grid IDs

    Returns
    -------
    filtered_trajectories : dict
        Dictionary mapping species names to filtered trajectory DataFrames.
        Each DataFrame contains only valid connectivity events for that species
        with columns:
        - 'source_cell' : int
            Release (source) grid cell ID
        - 'dest_cell' : int
            Settlement (destination) grid cell ID
        - 'release_time' : datetime64
            Time of particle release
        - 'settlement_time' : datetime64
            Time of successful settlement
        - 'pld' : float
            Realized planktonic larval duration (days)
        - 'distance' : float
            Great circle distance between source and destination (km)

    Notes
    -----
    - Filters are applied sequentially, each reducing the dataset
    - Buffer allows settlement near (but not exactly at) settlement site

    Buffer Parameter
    ----------------
    - buffer=0: Settlement only in exact settlement grid cells
    - buffer=1: Settlement in grid cells or immediate neighbors (recommended)
    - buffer=2: Settlement within 2-cell radius
    - Larger buffers account for sub-grid-scale habitat heterogeneity
    """
    gc.collect()

    subsection_header("Starting Post-processing")

    # Check for required columns
    required_columns = ['trajectory', 'time', 'obs', 'grid_id', 'lat', 'lon']
    missing = [col for col in required_columns if col not in traj.columns]
    if missing:
        print(f"\nError: Missing required columns: {missing}")
        return {}

    # Step 1: Spawning seasonality filter and assigning species unique trajectory numbers
    print("\n2. Applying spawning seasonality filter...")
    with MemoryMonitor("Species assignment"):
        # Ensure traits is a list of dicts for spawning_filter
        if isinstance(traits, pd.DataFrame):
            traits_list = traits.to_dict('records')
        elif isinstance(traits, list):
            traits_list = traits
        else:
            print(f"Error: traits must be list or DataFrame, got {type(traits)}")
            return {}

        # Apply spawning filter (expects list of dicts)
        spawning_dfs = spawning_filter(traj, traits_list)

        if len(spawning_dfs) == 0 or all(df.empty for df in spawning_dfs.values()):
            print("Error: No trajectories assigned to any species.")
            print("Check that:")
            print("  1. Particle release times overlap with spawning windows")
            print("  2. Species traits file has correct date format (DD/MM/YYYY)")
            print("  3. Simulation runtime covers spawning periods")
            return {}

        # Step 3: Day-wise PLD filter with settlement checking
        print("\n3. Applying day-wise PLD filter...")
        buffer = get_positive_integer("Enter buffer size around settlement sites [default: 1 adjacent grid cell]: ", default=1, min_val=0)
        pld_dfs = pld_filter(spawning_dfs, traits_list, settlement_grids, grid_ids, buffer)

        if len(pld_dfs) == 0:
            print("Error: No trajectories assigned to any species.")
            print("Check that:")
            print("  1. Particle dispersal times overlap with species PLD windows")
            print("  2. Species traits file has correct date format (DD/MM/YYYY)")
            print("  3. Simulation runtime covers spawning periods")
            return {}

        # Step 4: Release location filter
        print("\n4. Applying release location filter...")
        final_processed = habitat_filter(pld_dfs, release_grids)

        # Print summary
        summary_lines = []
        total_successful = 0
        for species, df in final_processed.items():
            if not df.empty:
                n_traj = df['trajectory'].nunique()
                total_successful += n_traj
                n_releases = df['release_id'].nunique()
                summary_lines.append(f"{species}: {n_traj} successful trajectories from {n_releases} release cohorts")
        summary_lines.append(f"")
        summary_lines.append(f"Total successful trajectories: {total_successful}")
        data_box("Post-processing Summary", summary_lines)

    return final_processed, buffer
# endregion

# region Connectivity Analyses
## region Analysis Type 1: Connectivity Metrics
def calculate_connectivity_metrics(final_processed, release_grids, settlement_grids,
                                   traits, traj_raw=None, print_summary=True,
                                   grid_ids=None, settlement_buffer=None):
    """
    Calculate standard connectivity metrics for community and species.

    Computes settlement success, mortality rates, local recruitment, dispersal
    distances, and realized planktonic larval duration from post-processed
    trajectory data.

    Parameters
    ----------
    final_processed : dict
        Post-processed trajectory data by species. Keys are species names
        (str), values are DataFrames containing columns:
        - trajectory : int - Unique trajectory identifier
        - time : datetime - Observation timestamp
        - obs : int - Observation number within trajectory
        - grid_id : int - Grid cell identifier
        - lat : float - Latitude
        - lon : float - Longitude
    release_grids : dict
        Release site information by species. Keys are species names,
        values are DataFrames with columns: grid_id, latitude, longitude
    settlement_grids : dict
        Settlement site information by species. Same structure as release_grids.
    traits : list of dict or DataFrame
        Species trait data. Required columns:
        - species_name : str
        - spawning_start : str (DD/MM/YYYY format)
        - spawning_end : str (DD/MM/YYYY format)
        - min_PLD : int
        - max_PLD : int
    traj_raw : DataFrame, optional
        Raw trajectory data before filtering. Required for accurate mortality
        calculation. If None, mortality cannot be calculated accurately.
    grid_ids : numpy.ndarray, optional
        2D grid ID array from setup_release_settlement(). Required for
        buffered local recruitment calculation. If None, falls back to
        module-level global (set automatically in Interactive Mode).
    settlement_buffer : int, optional
        Number of grid cells for settlement buffer. If None, falls back
        to the value set during post-processing (default: 1).

    Returns
    -------
    dict
        Contains two sub-dictionaries:
        - 'community' : dict - Aggregated metrics across all species
        - 'per_species' : dict - Metrics for each species individually

    Notes
    -----
    Mortality Calculation:
        If traj_raw is provided, mortality = 1 - (n_settled / n_released)
        where n_released is counted from raw trajectories released during
        the species' spawning window. If traj_raw is None, mortality cannot
        be calculated accurately.

    Local Recruitment:
        Particles settling within the settlement buffer distance of their
        release site. Uses the same buffer as settlement filtering.

    Dispersal Distance:
        Great circle distance using Haversine formula with Earth radius
        R = 6371 km.
    """
    # Convert traits to DataFrame if needed
    if isinstance(traits, list):
        traits_df = pd.DataFrame(traits)
    else:
        traits_df = traits.copy()

    # Create normalized species name lookup for matching
    traits_df['species_name_std'] = traits_df['species_name'].apply(standardise_species_name)

    results = {'per_species': {}, 'community': {}}

    # Community-level counters
    total_released = 0
    total_settled = 0
    total_retained = 0
    all_distances = []
    all_plds = []

    # Track if any species had calculable mortality
    any_mortality_calculable = False

    for species_name, df in final_processed.items():
        # Format species name for display
        display_name = display_species_name(species_name)

        if df.empty:
            print(f"No successful trajectories")
            continue

        # Get release sites for this species
        if species_name in release_grids:
            release_sites = set(release_grids[species_name]['grid_id'])
        else:
            release_sites = set()

        # Get settlement sites
        if species_name in settlement_grids:
            settle_sites = set(settlement_grids[species_name]['grid_id'])
        else:
            settle_sites = set()

        # Count settled trajectories
        n_trajectories = df['trajectory'].nunique()

        # Get buffer value: use parameter if provided, otherwise fall back to global
        if settlement_buffer is None:
            try:
                settlement_buffer = buffer
            except NameError:
                settlement_buffer = 1  # Default to 1-cell buffer

        # Build lookup dictionary: for each release site, get its buffered neighbors
        # This allows us to check if a particle returned to its own release site (+ buffer)
        release_site_buffers = {}
        if settlement_buffer > 0 and grid_ids is not None and isinstance(grid_ids, np.ndarray):
            for rid in release_sites:
                buffered_neighbors = set()
                positions = np.where(grid_ids == rid)
                if len(positions[0]) > 0:
                    for pi in range(len(positions[0])):
                        row, col = positions[0][pi], positions[1][pi]
                        for dr in range(-settlement_buffer, settlement_buffer + 1):
                            for dc in range(-settlement_buffer, settlement_buffer + 1):
                                nr, nc = row + dr, col + dc
                                if 0 <= nr < grid_ids.shape[0] and 0 <= nc < grid_ids.shape[1]:
                                    neighbor_id = int(grid_ids[nr, nc])
                                    if neighbor_id >= 0:  # Valid grid cell
                                        buffered_neighbors.add(neighbor_id)
                release_site_buffers[rid] = buffered_neighbors
        else:
            # No buffer - exact match only
            for rid in release_sites:
                release_site_buffers[rid] = {rid}

        # Calculate per-trajectory metrics
        trajectory_plds = []
        trajectory_distances = []
        recruitment_count = 0

        for traj_id in df['trajectory'].unique():
            traj_subset = df[df['trajectory'] == traj_id].sort_values('obs')

            if len(traj_subset) < 2:
                continue

            first_obs = traj_subset.iloc[0]
            last_obs = traj_subset.iloc[-1]

            # Calculate realized PLD (days in plankton)
            if 'time' in traj_subset.columns:
                time_diff = pd.to_datetime(last_obs['time']) - pd.to_datetime(first_obs['time'])
                pld_days = time_diff.total_seconds() / (24 * 3600)
                trajectory_plds.append(pld_days)
                all_plds.append(pld_days)

            # Calculate net displacement (km) using Haversine formula
            # This is the straight-line distance from release to settlement point,
            # not the total path length travelled. Particles that self-recruit
            # may show near-zero displacement despite having travelled some distance.
            if 'lat' in traj_subset.columns and 'lon' in traj_subset.columns:
                lat1, lon1 = first_obs['lat'], first_obs['lon']
                lat2, lon2 = last_obs['lat'], last_obs['lon']

                R = 6371  # Earth radius in km
                dlat = np.radians(lat2 - lat1)
                dlon = np.radians(lon2 - lon1)
                a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(
                    dlon / 2) ** 2
                c = 2 * np.arcsin(np.sqrt(a))
                distance = R * c
                trajectory_distances.append(distance)
                all_distances.append(distance)

            # Local recruitment: particle settles within buffer of ITS OWN release site
            if 'grid_id' in first_obs.index and 'grid_id' in last_obs.index:
                start_grid_id = int(first_obs['grid_id'])
                end_grid_id = int(last_obs['grid_id'])

                # Check if this particle's ending position is within buffer of where IT started
                if start_grid_id in release_site_buffers:
                    if end_grid_id in release_site_buffers[start_grid_id]:
                        recruitment_count += 1

        # Calculate n_released from raw trajectory data
        n_released = n_trajectories  # Default: assume all settled came from released
        mortality_calculable = False

        if traj_raw is None:
            print(f"Note: Raw data not provided. Mortality estimate unavailable")
        else:
            std_name = standardise_species_name(species_name)
            trait_match = traits_df[traits_df['species_name_std'] == std_name]

            if not trait_match.empty:
                species_trait = trait_match.iloc[0]
                try:
                    spawn_start = pd.to_datetime(species_trait['spawning_start'], dayfirst=True)
                    spawn_end = pd.to_datetime(species_trait['spawning_end'], dayfirst=True)

                    # Count trajectories released during spawning window
                    start_mask = traj_raw['obs'] == 0
                    traj_starts = traj_raw.loc[start_mask].copy()
                    traj_starts['time_dt'] = pd.to_datetime(traj_starts['time'])
                    time_mask = (traj_starts['time_dt'] >= spawn_start) & (traj_starts['time_dt'] <= spawn_end)
                    n_released = traj_starts.loc[time_mask, 'trajectory'].nunique()

                    if n_released >= n_trajectories:
                        mortality_calculable = True
                        any_mortality_calculable = True
                    else:
                        print(f"Note: Released ({n_released}) < settled ({n_trajectories}). Check spawning dates")
                        n_released = n_trajectories
                except Exception as e:
                    print(f"Warning: Could not parse spawning dates - {e}")
            else:
                print(f"Warning: No trait data found for {display_name}")

        # Ensure valid denominator
        if n_released == 0:
            n_released = n_trajectories

        # Calculate rates
        mortality_rate = 1 - (n_trajectories / n_released) if n_released > 0 else 0
        settlement_rate = n_trajectories / n_released if n_released > 0 else 0
        recruitment_rate = recruitment_count / n_trajectories if n_trajectories > 0 else 0

        # Distance statistics
        mean_distance = np.mean(trajectory_distances) if trajectory_distances else 0
        std_distance = np.std(trajectory_distances) if trajectory_distances else 0
        min_distance = min(trajectory_distances) if trajectory_distances else 0
        max_distance = max(trajectory_distances) if trajectory_distances else 0

        # PLD statistics
        mean_pld = np.mean(trajectory_plds) if trajectory_plds else 0
        min_pld = min(trajectory_plds) if trajectory_plds else 0
        max_pld = max(trajectory_plds) if trajectory_plds else 0
        std_pld = np.std(trajectory_plds) if trajectory_plds else 0

        # Store species results
        results['per_species'][display_name] = {
            'n_released': n_released,
            'n_settled': n_trajectories,
            'n_recruited': recruitment_count,
            'mortality_rate': mortality_rate if mortality_calculable else None,
            'settlement_rate': settlement_rate,
            'recruitment_rate': recruitment_rate,
            'mean_distance_km': mean_distance,
            'std_distance_km': std_distance,
            'min_distance_km': min_distance,
            'max_distance_km': max_distance,
            'mean_pld_days': mean_pld,
            'min_pld_days': min_pld,
            'max_pld_days': max_pld,
            'std_pld_days': std_pld,
            'n_release_sites': len(release_sites),
            'n_settlement_sites': len(settle_sites),
            'mortality_calculable': mortality_calculable
        }

        # Update community totals
        total_released += n_released
        total_settled += n_trajectories
        total_retained += recruitment_count

        # Print species summary
        if print_summary:
            subsection_header(f"Species: {display_name}")
            print(f"Released: {n_released} trajectories")
            print(f"Settled: {n_trajectories} trajectories")
            print(f"Settlement rate: {settlement_rate * 100:.1f}%")
            if mortality_calculable:
                print(f"Mortality rate: {mortality_rate * 100:.1f}%")
            print(
                f"Local recruitment rate: {recruitment_rate * 100:.1f}%, within {settlement_buffer}-cell buffer of release site ({recruitment_count} trajectories)")
            print(f"Min net displacement: {min_distance:.1f} km")
            print(f"Mean net displacement: {mean_distance:.1f} km")
            print(f"Max net displacement: {max_distance:.1f} km")
            print(f"Min PLD: {min_pld:.1f} days")
            print(f"Mean PLD: {mean_pld:.1f} days")
            print(f"Max PLD: {max_pld:.1f} days")

    # Community-level summary
    community_mortality = 1 - (total_settled / total_released) if total_released > 0 else 0
    community_recruitment = total_retained / total_settled if total_settled > 0 else 0

    results['community'] = {
        'n_species': len([s for s in final_processed.values() if not s.empty]),
        'total_released': total_released,
        'total_settled': total_settled,
        'total_recruited': total_retained,
        'community_mortality': community_mortality if any_mortality_calculable else None,
        'community_settlement': 1 - community_mortality if any_mortality_calculable else total_settled / max(1,
                                                                                                             total_released),
        'community_recruitment': community_recruitment,
        'mean_distance_km': np.mean(all_distances) if all_distances else 0,
        'std_distance_km': np.std(all_distances) if all_distances else 0,
        'min_distance_km': min(all_distances) if all_distances else 0,
        'max_distance_km': max(all_distances) if all_distances else 0,
        'mean_pld_days': np.mean(all_plds) if all_plds else 0,
        'min_pld_days': min(all_plds) if all_plds else 0,
        'max_pld_days': max(all_plds) if all_plds else 0,
        'std_pld_days': np.std(all_plds) if all_plds else 0,
        'mortality_calculable': any_mortality_calculable
    }
    return results

## Exporting the analysis 1 results
def export_connectivity_metrics(results, output_dir=None):
    """
    Export connectivity metrics to CSV files.

    Parameters
    ----------
    results : dict
        Output from calculate_connectivity_metrics()
    output_dir : str
        Directory to save output files
    """
    if output_dir is None:
        output_dir = get_default_dir('results')

    os.makedirs(output_dir, exist_ok=True)

    # Export per-species metrics
    species_df = pd.DataFrame(results['per_species']).T
    species_df.index.name = 'species'
    species_file = os.path.join(output_dir, 'connectivity_metrics_per_species.csv')
    species_df.to_csv(species_file)
    print(f"\nPer-species metrics saved to: {species_file}")

    # Export community metrics
    community_df = pd.DataFrame([results['community']])
    community_file = os.path.join(output_dir, 'connectivity_metrics_community.csv')
    community_df.to_csv(community_file, index=False)
    print(f"Community metrics saved to: {community_file}")

    return species_file, community_file

## Displaying the analysis 1 results
def display_metrics_table(results, show='both', max_species=3):
    """
    Display connectivity metrics as formatted tables.

    Parameters
    ----------
    results : dict
        Output from calculate_connectivity_metrics().
    show : str
        'community', 'species', or 'both'.
    max_species : int
        Maximum number of species to display individually (default 3).
        If more species exist, shows summary statistics only.
    """
    if show in ['community', 'both']:
        subsection_header("Community-Level Metrics")
        comm = results['community']
        print(f"{'Metric':<35} {'Value':>15}")
        print("-" * 50)
        print(f"{'Number of species':<35} {comm['n_species']:>15}")
        print(f"{'Total released':<35} {comm['total_released']:>15}")
        print(f"{'Total settled':<35} {comm['total_settled']:>15}")
        print(f"{'Settlement rate':<35} {comm['community_settlement'] * 100:>14.1f}%")
        print(f"{'Mortality rate':<35} {comm['community_mortality'] * 100:>14.1f}%")
        print(f"{'Local recruitment rate':<35} {comm['community_recruitment'] * 100:>14.1f}%")
        print(f"{'Min net displacement (km)':<35} {comm['min_distance_km']:>15.1f}")
        print(f"{'Mean net displacement (km)':<35} {comm['mean_distance_km']:>15.1f}")
        print(f"{'Max net displacement (km)':<35} {comm['max_distance_km']:>15.1f}")
        print(f"{'Min PLD (days)':<35} {comm['min_pld_days']:>15.1f}")
        print(f"{'Mean PLD (days)':<35} {comm['mean_pld_days']:>15.1f}")
        print(f"{'Max PLD (days)':<35} {comm['max_pld_days']:>15.1f}")

    if show in ['species', 'both']:
        n_species = len(results['per_species'])

        if n_species == 0:
            subsection_header("Species-Level Metrics")
            print("\nNo species data available.")
            return

        subsection_header("Species-Level Metrics")

        if n_species <= max_species:
            # Show full table for small number of species
            headers = ['Species', 'Released', 'Settled', 'Settle%', 'Mort%', 'Retain%', 'Displ(km)']
            print(f"{headers[0]:<25} {headers[1]:>10} {headers[2]:>10} {headers[3]:>8} "
                  f"{headers[4]:>7} {headers[5]:>8} {headers[6]:>10}")
            print("-" * 85)

            for species, metrics in results['per_species'].items():
                short_name = species[:24] if len(species) > 24 else species
                print(f"{short_name:<25} {metrics['n_released']:>10} {metrics['n_settled']:>10} "
                      f"{metrics['settlement_rate'] * 100:>7.1f}% {metrics['mortality_rate'] * 100:>6.1f}% "
                      f"{metrics['recruitment_rate'] * 100:>7.1f}% {metrics['mean_distance_km']:>10.1f}")
        else:
            # Show summary for large number of species
            print(f"\n{n_species} species analysed (showing summary to avoid clutter)")
            print("-" * 50)

            # Calculate summary statistics
            all_settlement = [m['settlement_rate'] for m in results['per_species'].values()]
            all_mortality = [m['mortality_rate'] for m in results['per_species'].values()]
            all_recruitment = [m['recruitment_rate'] for m in results['per_species'].values()]
            all_distance = [m['mean_distance_km'] for m in results['per_species'].values()]

            print(f"{'Metric':<25} {'Mean':>10} {'Min':>10} {'Max':>10}")
            print("-" * 55)
            print(f"{'Settlement rate (%)':<25} {np.mean(all_settlement) * 100:>10.1f} "
                  f"{np.min(all_settlement) * 100:>10.1f} {np.max(all_settlement) * 100:>10.1f}")
            print(f"{'Mortality rate (%)':<25} {np.mean(all_mortality) * 100:>10.1f} "
                  f"{np.min(all_mortality) * 100:>10.1f} {np.max(all_mortality) * 100:>10.1f}")
            print(f"{'Local recruitment rate (%)':<25} {np.mean(all_recruitment) * 100:>10.1f} "
                  f"{np.min(all_recruitment) * 100:>10.1f} {np.max(all_recruitment) * 100:>10.1f}")
            print(f"{'Mean net displacement (km)':<25} {np.mean(all_distance):>10.1f} "
                  f"{np.min(all_distance):>10.1f} {np.max(all_distance):>10.1f}")

            print("\nNote: Full per-species data available via export to CSV")
# endregion

# region Analysis Type 2: Network Analysis
## Network analysis function
def network_analysis(final_processed, release_grids, settlement_grids,
                     coastal_grid_df, grid_ids, full_grid_df, null_scale):
    """
    Perform network analysis to identify connectivity hotspots.

    Calculates functional roles (source, sink, local recruitment) and centrality
    measures (betweenness, closeness, degree) to identify sites critical for
    maintaining population connectivity.

    Parameters
    ----------
    final_processed : dict
        Post-processed trajectory data by species
    release_grids : dict
        Release locations by species
    settlement_grids : dict
        Settlement locations by species
    coastal_grid_df : DataFrame
        Coastal grid information with coordinates
    grid_ids : ndarray
       2D array of grid cell identifiers
    full_grid_df : DataFrame
        Full study area grid information
    null_scale : float
        Distance threshold in km for filtering connections.

    Returns
    -------
    dict
        Contains site_metrics, connectivity_matrix, top10 rankings, network_graph

    References
    ----------
    Methodology adapted from Melia et al. (2016) Scientific Reports.
    """
    gc.collect()

    # Study extent = all sites in the grid (coastal_grid_df is already filtered to study area)
    study_coastal = coastal_grid_df.copy()
    study_coastal_ids = set(study_coastal['grid_id'].astype(int))
    study_extent = len(study_coastal)

    # Study bounds from the actual grid (no buffer: exact bounds)
    study_bounds = {
        'lat_min': coastal_grid_df['latitude'].min(),
        'lat_max': coastal_grid_df['latitude'].max(),
        'lon_min': coastal_grid_df['longitude'].min(),
        'lon_max': coastal_grid_df['longitude'].max()
    }

    # Count potential connectivity sites (where species COULD connect based on occurrence data)
    # This includes buffered settlement sites to match what post-processing allows
    potential_sites = 0
    n_species = 0
    is_null_model = False

    # Helper function to get adjacent grids with buffer (matching pld_filter)
    def get_adjacent_grids_for_potential(grid_id, grid_matrix, buffer=1):
        """Get grid_ids within buffer distance of a given grid_id."""
        position = np.where(grid_matrix == grid_id)
        if len(position[0]) == 0:
            return set()
        row, col = position[0][0], position[1][0]
        adjacent = set()
        for i in range(-buffer, buffer + 1):
            for j in range(-buffer, buffer + 1):
                new_row, new_col = row + i, col + j
                if 0 <= new_row < grid_matrix.shape[0] and 0 <= new_col < grid_matrix.shape[1]:
                    adjacent.add(int(grid_matrix[new_row, new_col]))
        return adjacent

    if settlement_grids and release_grids:
        # Get union of all species' release AND settlement site grid_ids
        # Include buffered settlement sites to match post-processing behavior
        all_species_sites = set()
        for species_name in settlement_grids:
            # Add settlement sites WITH BUFFER
            species_settle_df = settlement_grids.get(species_name)
            if isinstance(species_settle_df, pd.DataFrame) and 'grid_id' in species_settle_df.columns:
                for gid in species_settle_df['grid_id'].astype(int).tolist():
                    # Add the site itself and its buffered neighbors
                    all_species_sites.add(gid)
                    all_species_sites.update(get_adjacent_grids_for_potential(gid, grid_ids, buffer=1))
            # Add release sites (no buffer needed for release)
            species_release_df = release_grids.get(species_name)
            if isinstance(species_release_df, pd.DataFrame) and 'grid_id' in species_release_df.columns:
                all_species_sites.update(species_release_df['grid_id'].astype(int).tolist())

        potential_sites = len(all_species_sites.intersection(study_coastal_ids))
        n_species = len([s for s in settlement_grids if s != 'null_model'])
        is_null_model = 'null_model' in settlement_grids and len(settlement_grids) == 1
    else:
        potential_sites = study_extent

    # Get realised connectivity sites (where particles actually started/ended)
    valid_sites = set()
    for species_name, df in final_processed.items():
        if df.empty:
            continue
        for traj_id in df['trajectory'].unique():
            traj_subset = df[df['trajectory'] == traj_id].sort_values('obs')
            if len(traj_subset) >= 2:
                start_obs = traj_subset[traj_subset['obs'] == 0]
                if not start_obs.empty and 'grid_id' in start_obs.columns:
                    start_grid_id = int(start_obs['grid_id'].iloc[0])
                    end_grid_id = int(traj_subset.iloc[-1]['grid_id'])
                    valid_sites.add(start_grid_id)
                    valid_sites.add(end_grid_id)

    # Filter to study coastal sites
    valid_sites = valid_sites.intersection(study_coastal_ids)
    valid_sites = sorted(list(valid_sites))

    if len(valid_sites) < 3:
        print("Error: Not enough connected sites for network analysis")
        return None

    # Create site names and connectivity matrix
    site_names = [f'site_{int(sid)}' for sid in valid_sites]
    connectivity_matrix = pd.DataFrame(0, index=site_names, columns=site_names, dtype=float)

    # Track per-species connections
    per_species_matrices = {}

    # Build connectivity counts
    for species_name, df in final_processed.items():
        if df.empty:
            continue

        display_name = display_species_name(species_name)
        species_matrix = pd.DataFrame(0, index=site_names, columns=site_names, dtype=float)

        for traj_id in df['trajectory'].unique():
            traj_subset = df[df['trajectory'] == traj_id].sort_values('obs')
            if len(traj_subset) < 2:
                continue

            start_obs = traj_subset[traj_subset['obs'] == 0]
            if start_obs.empty or 'grid_id' not in start_obs.columns:
                continue

            source_id = int(start_obs['grid_id'].iloc[0])
            dest_id = int(traj_subset.iloc[-1]['grid_id'])

            if source_id in valid_sites and dest_id in valid_sites:
                source_name = f'site_{source_id}'
                dest_name = f'site_{dest_id}'
                connectivity_matrix.loc[source_name, dest_name] += 1
                species_matrix.loc[source_name, dest_name] += 1

        per_species_matrices[display_name] = species_matrix

    unique_connections = int(np.sum(connectivity_matrix.values > 0))
    total_connections = int(connectivity_matrix.values.sum())

    # Print formatted summary table
    print(f"\n{'Metric':<35} {'Value':>15}")
    print("-" * 50)
    print(f"{'Number of species':<35} {n_species:>15}")
    print(f"{'Study extent (sites)':<35} {study_extent:>15}")
    if is_null_model:
        print(f"{'Potential connectivity sites':<35} {len(study_coastal):>15}")
    else:
        print(f"{'Potential connectivity sites':<35} {potential_sites:>15}")
    print(f"{'Realised connectivity sites':<35} {len(valid_sites):>15}")
    print(f"{'Unique connections':<35} {unique_connections:>15}")
    print(f"{'Total particles exchanged':<35} {total_connections:>15}")

    # Get site coordinates
    site_coords = study_coastal[study_coastal['grid_id'].astype(int).isin(valid_sites)][
        ['grid_id', 'longitude', 'latitude']].drop_duplicates()
    site_coords['grid_id'] = site_coords['grid_id'].astype(int)

    # Create distance matrix
    coords = []
    for site_id in valid_sites:
        site_data = site_coords[site_coords['grid_id'] == site_id]
        if not site_data.empty:
            coords.append([site_data['longitude'].iloc[0], site_data['latitude'].iloc[0]])
    coords = np.array(coords)

    # Convert to km (approximate: 1 degree ≈ 111 km)
    distance_matrix = cdist(coords, coords) * 111

    # Apply distance filtering (or skip if null_scale is None for full domain)
    if null_scale is None:
        connectivity_filtered = connectivity_matrix.copy()
    else:
        print(f"\nApplying distance filter ({null_scale} km)...")
        connectivity_filtered = connectivity_matrix.copy()
        for i in range(len(connectivity_matrix)):
            for j in range(len(connectivity_matrix)):
                if distance_matrix[i, j] > null_scale:
                    connectivity_filtered.iloc[i, j] = 0

        connections_remaining = np.sum(connectivity_filtered.values > 0)
        weight_remaining = int(connectivity_filtered.values.sum())
        print(f"Connections after {null_scale}km filter: {connections_remaining} unique ({weight_remaining:} total particles exchanged)")

    # Warning if too few connections (only for filtered analysis)
    if null_scale is not None and connections_remaining < 20:
        print(f"\nWarning: Very few connections ({connections_remaining}) at {null_scale} km scale.")
        print(f"Network metrics may not be meaningful at this scale.")
        print(f"Consider using a larger distance threshold or multi-scale analysis.")

    # Create network graph
    G = nx.from_pandas_adjacency(connectivity_filtered, create_using=nx.DiGraph())

    # =========================================================================
    # Calculate metrics
    # =========================================================================

    # Source strength: total outgoing connections (export potential)
    source_strength = dict(G.out_degree(weight='weight'))

    # Sink strength: total incoming connections (import potential)
    sink_strength = dict(G.in_degree(weight='weight'))

    # Local recruitment: self-loops (particles returning to same site)
    local_recruitment= {}
    for node in G.nodes():
        if G.has_edge(node, node):
            local_recruitment[node] = G[node][node]['weight']
        else:
            local_recruitment[node] = 0

    # Betweenness: how often site lies on shortest paths between other sites
    betweenness_cent = nx.betweenness_centrality(G, weight='weight', normalized=True)

    # Closeness: how "close" site is to all other sites
    closeness_cent = nx.closeness_centrality(G, distance='weight')

    # Degree: total number of connections (normalized)
    degree_cent = nx.degree_centrality(G)

    # Create metrics DataFrame
    site_metrics = pd.DataFrame({
        'site': list(G.nodes()),
        # Functional roles
        'source': [source_strength.get(node, 0) for node in G.nodes()],
        'sink': [sink_strength.get(node, 0) for node in G.nodes()],
        'local_recruitment': [local_recruitment.get(node, 0) for node in G.nodes()],
        # Centrality measures
        'betweenness': [betweenness_cent.get(node, 0) for node in G.nodes()],
        'closeness': [closeness_cent.get(node, 0) for node in G.nodes()],
        'degree': [degree_cent.get(node, 0) for node in G.nodes()],
    })

    # Add coordinates
    site_metrics['grid_id'] = site_metrics['site'].apply(
        lambda x: int(x.split('_')[1]) if '_' in str(x) else int(x)
    )
    site_metrics = pd.merge(site_metrics, site_coords, on='grid_id', how='left')

    # Create rankings
    functional_cols = ['source', 'sink', 'local_recruitment']
    centrality_cols = ['betweenness', 'closeness', 'degree']
    all_metric_cols = functional_cols + centrality_cols

    # Normalize and rank each metric (higher value = better = lower rank number)
    for metric in all_metric_cols:
        max_val = site_metrics[metric].max()
        if max_val > 0:
            site_metrics[f'{metric}_norm'] = site_metrics[metric] / max_val
        else:
            site_metrics[f'{metric}_norm'] = 0
        # Rank: 1 = highest value (best), ascending=False means highest gets rank 1
        site_metrics[f'{metric}_rank'] = site_metrics[metric].rank(ascending=False, method='min')

    # Composite rankings (average of ranks - lower = better)
    # Method: Average of RANKS (not values) gives equal weight to each metric
    # regardless of scale. Lower average rank = higher priority site.
    # Example: Site ranks [1,5,7,3] across 4 metrics → composite = 4.0
    # The _norm columns are for visualisation only, not used in composite.
    site_metrics['functional_composite'] = site_metrics[
        [f'{m}_rank' for m in functional_cols]].mean(axis=1)
    site_metrics['centrality_composite'] = site_metrics[
        [f'{m}_rank' for m in centrality_cols]].mean(axis=1)
    site_metrics['overall_composite'] = site_metrics[
        [f'{m}_rank' for m in all_metric_cols]].mean(axis=1)

    # Sort by overall composite (lower = higher priority)
    site_metrics = site_metrics.sort_values('overall_composite')

    # Count how many times each site appears in top 10 across all metrics
    site_metrics['top10_count'] = 0
    for metric in all_metric_cols:
        site_metrics['top10_count'] += (site_metrics[f'{metric}_rank'] <= 10).astype(int)

    # Extract top 10 for each composite
    top10_functional = site_metrics.nsmallest(10, 'functional_composite').copy()
    top10_centrality = site_metrics.nsmallest(10, 'centrality_composite').copy()
    top10_overall = site_metrics.nsmallest(10, 'overall_composite').copy()

    return {
        'site_metrics': site_metrics,
        'top10_overall': top10_overall,
        'top10_centrality': top10_centrality,
        'top10_functional': top10_functional,
        'connectivity_matrix': connectivity_filtered,
        'connectivity_raw': connectivity_matrix,
        'network_graph': G,
        'per_species_matrices': per_species_matrices,
        'study_bounds': study_bounds,
        'distance_matrix': distance_matrix,
        'distance_threshold': null_scale
    }

## Create network map
def create_network_map(site_metrics, top_10, title="Network Analysis",
                       composite_col='overall_composite', output_path=None,
                       zoom_extent=None, cmap='viridis_r',
                       output_format='png', dpi=300, show_figure=None):
    """
    Create network map with colour gradient and numbered top 10 sites.

    Parameters
    ----------
    site_metrics : DataFrame
        Site metrics from network_analysis()
    top_10 : DataFrame
        Top 10 sites to highlight
    title : str
        Map title
    composite_col : str
        Column to use for coloring
    output_path : str, optional
        Path to save figure (without extension if format specified)
    zoom_extent : list, optional
        [lon_min, lon_max, lat_min, lat_max] for zoomed view
    cmap : str
        Colormap name (viridis_r: yellow = better)
    output_format : str
        Output format ('png', 'pdf', 'tif', 'jpeg', 'svg')
    dpi : int
        Resolution for raster formats (default 300)
    """
    # Calculate dynamic figure size
    lon_range = site_metrics['longitude'].max() - site_metrics['longitude'].min()
    lat_range = site_metrics['latitude'].max() - site_metrics['latitude'].min()
    fig_width, fig_height, _ = calculate_figure_dimensions(lon_range, lat_range, base_size=10, n_panels=1)

    # Create figure with extra height for colorbar and legend
    fig = plt.figure(figsize=(fig_width, fig_height + 1.5))

    # Main map axes - leave space at bottom
    ax = fig.add_axes([0.08, 0.25, 0.84, 0.68], projection=ccrs.PlateCarree())

    # Set extent
    if zoom_extent:
        ax.set_extent(zoom_extent, crs=ccrs.PlateCarree())
    else:
        buffer = 0.15
        ax.set_extent([
            site_metrics['longitude'].min() - buffer,
            site_metrics['longitude'].max() + buffer,
            site_metrics['latitude'].min() - buffer,
            site_metrics['latitude'].max() + buffer
        ], crs=ccrs.PlateCarree())

    # Map features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')

    # All sites colored by composite score
    scatter = ax.scatter(
        site_metrics['longitude'], site_metrics['latitude'],
        c=site_metrics[composite_col], cmap=cmap,
        s=60, alpha=0.6, transform=ccrs.PlateCarree(),
        edgecolors='none',
        vmin=site_metrics[composite_col].min(),
        vmax=site_metrics[composite_col].max()
    )

    # Top 10 hotspots with numbers
    for idx, (_, site) in enumerate(top_10.iterrows()):
        ax.scatter(site['longitude'], site['latitude'], s=300, c='gold',
                   edgecolors='black', linewidth=2, transform=ccrs.PlateCarree(),
                   zorder=5, alpha=0.95)
        ax.text(site['longitude'], site['latitude'], str(idx + 1),
                ha='center', va='center', fontsize=10, fontweight='bold',
                transform=ccrs.PlateCarree(), zorder=6)

    # Title and gridlines
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False

    # Colorbar - fixed position below map
    cbar_ax = fig.add_axes([0.15, 0.13, 0.7, 0.025])
    cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Connectivity Rank (lower = higher priority)', fontsize=11)

    # Legend - fixed position below colorbar
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Top 10 Priority Sites',
               markerfacecolor='gold', markeredgecolor='black', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Other Sites (colour = score)',
               markerfacecolor='#440154', markersize=8, alpha=0.7)
    ]

    fig.legend(handles=legend_elements,
               loc='lower center',
               ncol=2,
               bbox_to_anchor=(0.5, 0.02),
               framealpha=0.95,
               fontsize=10)

    # Save if output path provided
    if output_path:
        fig.savefig(output_path, format=output_format, dpi=dpi, bbox_inches='tight')
        print(f"  Figure saved to: {output_path}")

    plt.show()

    return fig

## Create a connectivity matrix
def create_connectivity_matrix_plot(connectivity_matrix, title="Connectivity Matrix",
                                    output_path=None, log_scale=True,
                                    output_format='png', dpi=300,
                                    filter_empty=True):
    """
    Create a connectivity matrix heatmap visualisation (like Melia et al. 2016).

    Shows source sites (rows) vs destination sites (columns) with color
    intensity representing connection strength.

    Parameters
    ----------
    connectivity_matrix : DataFrame
        Site-by-site connectivity matrix
    title : str
        Plot title
    output_path : str, optional
        Path to save figure
    log_scale : bool
        Use log10 scale for intensity (recommended for sparse matrices)
    output_format : str
        Output format ('png', 'pdf', 'tif', 'jpeg', 'svg')
    dpi : int
        Resolution for raster formats (default 300)
    filter_empty : bool
        If True, remove sites with no connections (default True)
    """
    # Filter to only sites with connections (removes empty rows/columns)
    if filter_empty:
        # Keep sites that have at least one connection (as source OR sink)
        row_sums = connectivity_matrix.sum(axis=1)
        col_sums = connectivity_matrix.sum(axis=0)
        connected_sites = (row_sums > 0) | (col_sums > 0)
        connectivity_matrix = connectivity_matrix.loc[connected_sites, connected_sites]

    fig, ax = plt.subplots(figsize=(12, 12))

    # Prepare data
    matrix_data = connectivity_matrix.values.astype(float)

    if log_scale:
        # Log transform (add 1 to avoid log(0))
        matrix_data = np.log10(matrix_data + 1)
        cbar_label = 'Connectivity Strength (log10)'
    else:
        cbar_label = 'Connectivity Strength'

    # Mask zeros for better visualisation
    masked_data = np.ma.masked_where(matrix_data == 0, matrix_data)

    # Create heatmap with coolwarm colormap
    cmap = plt.cm.get_cmap('coolwarm')
    cmap.set_bad('lightgray')  # Colour for masked (zero) values

    im = ax.imshow(masked_data, cmap=cmap, aspect='auto', interpolation='nearest')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label, fontsize=14)

    # Labels and ticks
    n_sites = len(connectivity_matrix)

    if n_sites <= 30:
        # Small matrix: label every site
        ax.set_xticks(range(n_sites))
        ax.set_yticks(range(n_sites))
        ax.set_xticklabels(connectivity_matrix.columns, rotation=90, fontsize=9)
        ax.set_yticklabels(connectivity_matrix.index, fontsize=9)
    else:
        # Large matrix: label every 10th site, minor ticks for all sites
        label_step = 10

        # Major ticks with labels (every 10th site)
        major_ticks = list(range(0, n_sites, label_step))
        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)
        ax.set_xticklabels([connectivity_matrix.columns[i] for i in major_ticks],
                           rotation=90, fontsize=10)
        ax.set_yticklabels([connectivity_matrix.index[i] for i in major_ticks],
                           fontsize=10)

        # Minor ticks (every site, no labels) - smaller tick marks
        ax.set_xticks(range(n_sites), minor=True)
        ax.set_yticks(range(n_sites), minor=True)

        # Style the ticks
        ax.tick_params(axis='both', which='major', length=6, width=1)
        ax.tick_params(axis='both', which='minor', length=3, width=0.5)

    ax.set_xlabel('Sink Site', fontsize=14, labelpad=15)
    ax.set_ylabel('Source Site', fontsize=14, labelpad=15)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)

    # Invert y-axis so site order matches (first site at top-left origin)
    ax.invert_yaxis()

    # Add diagonal line to show local recruitment (self-recruitment cells)
    # After invert_yaxis: y=0 is at bottom, y=n-1 is at top
    # Line from (0,0) bottom-left to (n-1,n-1) top-right = self-recruitment diagonal
    ax.plot([0, n_sites - 1], [0, n_sites - 1], 'k--', alpha=0.3, linewidth=1)

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        fig.savefig(output_path, format=output_format, dpi=dpi, bbox_inches='tight')
        print(f"  Figure saved to: {output_path}")

    plt.show()

    return fig

## Display the network results summary
def display_network_summary(network_results, show_individual=False):
    """
    Display formatted network analysis summary.

    Parameters
    ----------
    network_results : dict
        Output from network_analysis().
    show_individual : bool
        If True, also display individual metrics (default False).
    """
    metrics = network_results['site_metrics']

    subsection_header("Network Analysis Summary")

    print(f"\nSites analysed: {len(metrics)}")
    print(f"Distance threshold: {network_results.get('distance_threshold', 5)} km")
    print(f"Total connections: {int(network_results['connectivity_matrix'].values.sum())}")

    # Show top 10 for each composite
    composites = [
        ("Overall Composite", network_results['top10_overall'], 'overall_composite'),
        ("Centrality Composite", network_results['top10_centrality'], 'centrality_composite'),
        ("Functional Roles", network_results['top10_functional'], 'functional_composite')
    ]

    for name, top10, col in composites:
        print()
        display_line("─")
        print(f"Top 10 Sites for: {name}")
        display_line("─")
        print(f"{'Rank':<6} {'Grid ID':<10} {'Location':<20} {'Score':>10}")
        print("-" * 50)

        # Print rows
        for i, (_, row) in enumerate(top10.iterrows(), 1):
            lat_str = f"({row['latitude']:.2f}, {row['longitude']:.2f})"
            print(f"{i:<6} {int(row['grid_id']):<10} {lat_str:<20} {row[col]:>10.1f}")

    # Optionally show individual metrics
    if show_individual:
        display_individual_metrics(network_results)

### Display individual species results
def display_individual_metrics(network_results, metrics_to_show='all'):
    """
    Display individual network metrics (not just composites).

    Parameters
    ----------
    network_results : dict
        Output from network_analysis().
    metrics_to_show : str or list
        'all', 'functional', 'centrality', or list of specific metrics.
        Available metrics: source, sink, local recruitment, betweenness, closeness, degree.
    """
    site_metrics = network_results['site_metrics']

    # Define metric groups
    functional_metrics = ['source', 'sink', 'local_recruitment']
    centrality_metrics = ['betweenness', 'closeness', 'degree']
    all_metrics = functional_metrics + centrality_metrics

    # Determine which metrics to display
    if metrics_to_show == 'all':
        metrics = all_metrics
    elif metrics_to_show == 'functional':
        metrics = functional_metrics
    elif metrics_to_show == 'centrality':
        metrics = centrality_metrics
    elif isinstance(metrics_to_show, list):
        metrics = [m for m in metrics_to_show if m in all_metrics]
    else:
        metrics = all_metrics

    subsection_header("Individual Network Metrics: Top 10 Sites")

    for metric in metrics:
        if metric not in site_metrics.columns:
            continue

        print(f"\n{metric.upper()}")

        # Sort by this metric (descending - higher is better)
        top10 = site_metrics.nlargest(10, metric)

        print(f"\n{'Rank':<6} {'Grid ID':<10} {'Lat':>8} {'Lon':>9} {'Value':>12}")

        for idx, (_, row) in enumerate(top10.iterrows()):
            print(f"{idx + 1:<6} {int(row['grid_id']):<10} {row['latitude']:>8.2f} "
                  f"{row['longitude']:>9.2f} {row[metric]:>12.4f}")

    return metrics

## Export the network results
def export_network_results(network_results, output_dir=None):
    """
    Export all network analysis results to CSV files.

    Saves site-level network metrics, the full connectivity matrix, and
    the top 10 ranked sites for each composite score (overall, centrality,
    functional) as separate CSV files.

    Parameters
    ----------
    network_results : dict
        Output from network_analysis(). Expected keys:
        - 'site_metrics': DataFrame with all site-level metrics
        - 'connectivity_matrix': DataFrame with pairwise connectivity values
        - 'top10_overall': DataFrame with top 10 sites by overall composite
        - 'top10_centrality': DataFrame with top 10 sites by centrality composite
        - 'top10_functional': DataFrame with top 10 sites by functional composite
    output_dir : str, optional
        Directory to save output files. If None, uses the project results
        directory or falls back to './outputs'.

    Returns
    -------
    str
        Path to the output directory where files were saved
    """
    if output_dir is None:
        output_dir = get_default_dir('results')

    os.makedirs(output_dir, exist_ok=True)

    # Export site metrics (all metrics for all sites)
    metrics_file = os.path.join(output_dir, 'network_site_metrics.csv')
    network_results['site_metrics'].to_csv(metrics_file, index=False)
    print(f"Site metrics saved: {metrics_file}")

    # Export connectivity matrix
    conn_file = os.path.join(output_dir, 'connectivity_matrix.csv')
    network_results['connectivity_matrix'].to_csv(conn_file)
    print(f"\nConnectivity matrix saved: {conn_file}")

    # Export top 10 for each composite
    for name, df in [('overall', network_results['top10_overall']),
                     ('centrality', network_results['top10_centrality']),
                     ('functional', network_results['top10_functional'])]:
        top10_file = os.path.join(output_dir, f'top10_{name}.csv')
        df.to_csv(top10_file, index=False)
        print(f"\nTop 10 {name} saved: {top10_file}")

    return output_dir
# endregion

# region Analysis Type 3: Trait sensitivity Analysis
## Mantel test to compare simulation results
def mantel_test(matrix1, matrix2, n_permutations=999, random_state=None):
    """
    Perform Mantel test to compare two connectivity matrices.

    The Mantel test assesses correlation between two distance/similarity
    matrices while accounting for non-independence of matrix elements.
    Uses Spearman correlation (appropriate for count data).

    Parameters
    ----------
    matrix1, matrix2 : ndarray or DataFrame
        Square matrices to compare
    n_permutations : int
        Number of permutations for significance testing
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    tuple
        (r, p_value) - Mantel correlation coefficient and p-value
    """
    rng = check_random_state(random_state)

    # Convert to numpy arrays if needed
    if hasattr(matrix1, 'values'):
        matrix1 = matrix1.values
    if hasattr(matrix2, 'values'):
        matrix2 = matrix2.values

    # Flatten upper triangles (exclude diagonal)
    n = matrix1.shape[0]
    triu_indices = np.triu_indices(n, k=1)

    x = matrix1[triu_indices]
    y = matrix2[triu_indices]

    # Flatten full matrices for variance check
    flat1 = matrix1.flatten()
    flat2 = matrix2.flatten()

    # Check for zero variance
    if np.std(flat1) == 0 or np.std(flat2) == 0:
        print("\nWarning: One or both connectivity matrices have no variance.")
        print("This typically occurs when:")
        print("  - Very few connections exist in one or both models")
        print("  - All connection strengths are identical")
        print("  - The study area is too small or simulation too short")
        print("The Mantel test cannot compute a meaningful correlation in this case.")
        print("Consider: running with more particles, longer simulation, or larger area.\n")
        return np.nan, 1.0  # Return tuple, not dict

    # Observed correlation (Spearman for non-normal count data)
    r_obs, _ = spearmanr(x, y)

    # Handle nan result
    if np.isnan(r_obs):
        print("Warning: Correlation is undefined (possibly identical matrices)")
        return np.nan, 1.0

    # Permutation test
    count = 0
    for _ in range(n_permutations):
        # Permute rows and columns of matrix2 together
        perm = rng.permutation(n)
        matrix2_perm = matrix2[perm][:, perm]
        y_perm = matrix2_perm[triu_indices]

        r_perm, _ = spearmanr(x, y_perm)
        if abs(r_perm) >= abs(r_obs):
            count += 1

    p_value = (count + 1) / (n_permutations + 1)

    return r_obs, p_value

## Create null model (no trait filtering)
def create_null_model(traj_raw, base_grid_df, study_bounds, full_grid_df,
                      spp_params, traits):
    """
    Create null model connectivity (no trait filtering).

    The null model uses the SAME settlement detection logic as the trait model,
    but without species-specific constraints. This ensures fair comparison.

    Parameters
    ----------
    traj_raw : DataFrame
        Raw trajectory data (before trait filtering)
    coastal_grid_df : DataFrame
        Coastal grid information
    study_bounds : dict
        Study area bounds {'lat_min', 'lat_max', 'lon_min', 'lon_max'}
    full_grid_df : DataFrame
        Full grid information
    ocean_grid_df : DataFrame
        Ocean grid information
    spp_params : dict
        Species parameters including 'habitat_filtering' setting
    traits : list
        Trait data (used to get PLD range for null model)

    Returns
    -------
    dict
        Contains:
        - 'processed': dict with 'null_model' key containing filtered trajectories
        - 'release_grids': dict with null model release grid
        - 'settlement_grids': dict with null model settlement grid
        - 'n_settled': int, number of settled trajectories
        - 'n_total': int, total trajectories processed
        - 'settlement_rate': float, proportion that settled
        - 'pld_window': tuple (min_pld, max_pld) used for null model
    """
    gc.collect()

    subsection_header("Null Model Analysis")
    status_message("Creating null model (no trait filtering)...", "info")

    # Use same grid type as trait model for fair comparison
    habitat_filtering = spp_params.get('habitat_filtering', 'coastal')

    # Use same grid type as trait model
    # base_grid_df is already set correctly based on user's habitat filtering choice
    habitat_filtering = spp_params.get('habitat_filtering', 'coastal')
    habitat_label = 'coastal' if habitat_filtering == 'coastal' else 'ocean'
    print(f"Using {habitat_label} cells for settlement (matching trait model settings)")
    base_grid = base_grid_df

    # Filter to study area
    study_mask = (
            (base_grid['latitude'] >= study_bounds['lat_min']) &
            (base_grid['latitude'] <= study_bounds['lat_max']) &
            (base_grid['longitude'] >= study_bounds['lon_min']) &
            (base_grid['longitude'] <= study_bounds['lon_max'])
    )
    valid_settlement_sites = base_grid[study_mask].copy()
    valid_grid_ids = set(valid_settlement_sites['grid_id'].astype(int))

    print(f"Valid settlement sites: {len(valid_grid_ids)}")

    # Null model PLD window selection
    # Offer ecologically meaningful presets independent of species traits
    subsection_header("Null Model PLD Window")
    info_box([
        "The null model tests connectivity without species-specific traits.",
        "Select a PLD window representing a 'generic disperser' given your community:",
        "",
        "  1. Short disperser (1-14 days)",
        "  2. Medium disperser (7-28 days)",
        "  3. Long disperser (14-60 days)",
        "  4. Custom (enter your own values)"
    ])

    while True:
        null_choice = input("\nSelect option (1/2/3/4) [default: 2]: ").strip() or '2'

        if null_choice == '1':
            null_min_pld = 1
            null_max_pld = 14
            print(f"\nSelected: Short disperser (day {null_min_pld} to day {null_max_pld})")
            break
        elif null_choice == '2':
            null_min_pld = 7
            null_max_pld = 28
            print(f"\nSelected: Medium disperser (day {null_min_pld} to day {null_max_pld})")
            break
        elif null_choice == '3':
            null_min_pld = 14
            null_max_pld = 60
            print(f"\nSelected: Long disperser (day {null_min_pld} to day {null_max_pld})")
            break
        elif null_choice == '4':
            try:
                null_min_pld = int(input("Enter minimum PLD (days): ").strip())
                null_max_pld = int(input("Enter maximum PLD (days): ").strip())
                if null_min_pld < 1:
                    null_min_pld = 1
                    print("  (Minimum set to 1 day)")
                if null_max_pld <= null_min_pld:
                    null_max_pld = null_min_pld + 7
                    print(f"  (Maximum adjusted to {null_max_pld} days)")
                print(f"\nSelected: Custom (day {null_min_pld} to day {null_max_pld})")
                break
            except ValueError:
                print("Invalid input. Please enter numbers.")
        else:
            print("Invalid choice. Enter 1, 2, 3, or 4.")

        print(f"Null model PLD window: day {null_min_pld} to day {null_max_pld}")

    # Process trajectories - same logic as trait model but without species constraints
    successful_trajectories = []
    settlement_counts = {}
    local_recruitment_count = 0  # Track particles returning to release site

    total_traj = traj_raw['trajectory'].nunique()
    print(f"\nProcessing {total_traj:} trajectories...")

    for traj_id in tqdm(traj_raw['trajectory'].unique(), desc="Filtering trajectories"):
        traj_subset = traj_raw[traj_raw['trajectory'] == traj_id].sort_values('time')

        if len(traj_subset) < 2:
            continue

        # Get release time and location
        release_obs = traj_subset[traj_subset['obs'] == 0]
        if release_obs.empty:
            continue

        release_time = pd.to_datetime(release_obs['time'].iloc[0])
        release_grid_id = int(release_obs['grid_id'].iloc[0]) if 'grid_id' in release_obs.columns else None

        # Check each day from start_day to max_PLD (same as trait model)
        # Start from at least day 1 to prevent immediate settlement at release site
        settlement_found = False
        for day in range(null_min_pld, null_max_pld + 1):
            check_time = release_time + pd.Timedelta(days=day)

            # Find observation closest to this time
            traj_subset['time_dt'] = pd.to_datetime(traj_subset['time'])
            time_diff = abs(traj_subset['time_dt'] - check_time)

            if len(time_diff) > 0:
                closest_idx = time_diff.idxmin()
                closest_obs = traj_subset.loc[closest_idx]

                # Check if in valid settlement habitat
                if 'grid_id' in closest_obs:
                    grid_id = int(closest_obs['grid_id'])

                    # On day 1, don't allow settlement at exact release site
                    # (prevents particles from "settling" immediately where they started)
                    if day == 1 and release_grid_id is not None and grid_id == release_grid_id:
                        continue  # Skip to next day

                    if grid_id in valid_grid_ids:
                        # Settlement successful. Crop trajectory
                        settled_traj = traj_subset[traj_subset['time_dt'] <= closest_obs['time_dt']].copy()
                        settled_traj['settlement_day'] = day
                        successful_trajectories.append(settled_traj)

                        settlement_counts[day] = settlement_counts.get(day, 0) + 1

                        # Check for local recruitment (settled within buffer of release site)
                        # For null model, use same logic as trait model for fair comparison
                        if release_grid_id is not None:
                            # Build buffer around this particle's release site
                            release_buffer = {release_grid_id}  # Start with exact match
                            try:
                                # Use same buffer as trait model (set by user in post-processing)
                                try:
                                    null_buffer = buffer  # User's selected buffer from workflow_step(5)
                                except NameError:
                                    try:
                                        null_buffer = buffer  # Use user's selected buffer
                                    except NameError:
                                        null_buffer = 1  # Default if not set
                                positions = np.where(grid_ids == release_grid_id)
                                if len(positions[0]) > 0:
                                    for pi in range(len(positions[0])):
                                        row, col = positions[0][pi], positions[1][pi]
                                        for dr in range(-null_buffer, null_buffer + 1):
                                            for dc in range(-null_buffer, null_buffer + 1):
                                                nr, nc = row + dr, col + dc
                                                if 0 <= nr < grid_ids.shape[0] and 0 <= nc < grid_ids.shape[1]:
                                                    neighbor_id = int(grid_ids[nr, nc])
                                                    if neighbor_id >= 0:
                                                        release_buffer.add(neighbor_id)
                            except:
                                pass  # Use exact match if buffer calculation fails

                            if grid_id in release_buffer:
                                local_recruitment_count += 1

                        settlement_found = True
                        break

    if not successful_trajectories:
        print("\nError: No trajectories settled in valid habitat!")
        print("Check that study bounds overlap with trajectory endpoints.")
        return None

    null_traj = pd.concat(successful_trajectories)
    n_settled = len(successful_trajectories)

    local_recruitment_rate = local_recruitment_count / n_settled if n_settled > 0 else 0

    # Calculate dispersal distances and PLD for null model
    distances = []
    pld_values = []

    for traj_df in successful_trajectories:
        if len(traj_df) >= 2:
            # Get start and end coordinates
            start = traj_df.iloc[0]
            end = traj_df.iloc[-1]

            # Calculate distance using Haversine formula
            lat1, lon1 = np.radians(start['lat']), np.radians(start['lon'])
            lat2, lon2 = np.radians(end['lat']), np.radians(end['lon'])

            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            c = 2 * np.arcsin(np.sqrt(a))
            distance_km = 6371 * c  # Earth radius in km

            distances.append(distance_km)

            # Get settlement day (PLD)
            if 'settlement_day' in traj_df.columns:
                pld_values.append(traj_df['settlement_day'].iloc[-1])

    # Calculate statistics
    if distances:
        mean_distance = np.mean(distances)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
    else:
        mean_distance = min_distance = max_distance = 0

    if pld_values:
        mean_pld = np.mean(pld_values)
        min_pld_actual = np.min(pld_values)
        max_pld_actual = np.max(pld_values)
    else:
        mean_pld = min_pld_actual = max_pld_actual = 0

    print(f"\nNull model results:")
    print(f"{'Total trajectories':<30} {total_traj:>14}")
    print(f"{'Settled trajectories':<30} {n_settled:>14} ({n_settled / total_traj * 100:.1f}%)")
    print(f"{'Local recruitment':<30} {local_recruitment_count:>14} ({local_recruitment_rate * 100:.1f}%)")
    print(f"{'Mean dispersal distance':<30} {mean_distance:>14.1f} km")
    print(f"{'Mean PLD':<30} {mean_pld:>14.1f} days")
    print(f"Settlement by day: {dict(sorted(settlement_counts.items()))}")

    # Create single "species" for null model
    null_processed = {'null_model': null_traj}

    null_grids = valid_settlement_sites[['grid_id', 'longitude', 'latitude']].copy()
    null_grids['species'] = 'null_model'

    return {
        'processed': null_processed,
        'release_grids': {'null_model': null_grids},
        'settlement_grids': {'null_model': null_grids},
        'n_settled': n_settled,
        'n_total': total_traj,
        'settlement_rate': n_settled / total_traj,
        'local_recruitment_count': local_recruitment_count,
        'local_recruitment_rate': local_recruitment_rate,
        'pld_window': (null_min_pld, null_max_pld),
        'mean_distance_km': mean_distance,
        'min_distance_km': min_distance,
        'max_distance_km': max_distance,
        'mean_pld_days': mean_pld,
        'min_pld_actual': min_pld_actual,
        'max_pld_actual': max_pld_actual
    }

## Compare network results between trait scenarios and null models
def compare_models(trait_results, null_results):
    """
    Compare trait-filtered and null model network results.

    Parameters
    ----------
    trait_results : dict
        Network analysis results from trait-filtered data
    null_results : dict
        Network analysis results from null model

    Returns
    -------
    dict
        Contains:
        - 'hotspot_overlap': dict with overlap analysis results
        - 'mantel_test': dict with r, p_value, n_sites
        - 'spatial_agreement': str ('high', 'moderate', or 'low')
    """
    comparison = {}

    # 1. Hotspot overlap analysis
    trait_top10_ids = set(trait_results['top10_overall']['grid_id'])
    null_top10_ids = set(null_results['top10_overall']['grid_id'])

    overlap = trait_top10_ids.intersection(null_top10_ids)
    trait_unique = trait_top10_ids - null_top10_ids
    null_unique = null_top10_ids - trait_top10_ids

    comparison['hotspot_overlap'] = {
        'shared_sites': len(overlap),
        'overlap_percentage': len(overlap) * 10,
        'trait_unique': len(trait_unique),
        'null_unique': len(null_unique),
        'shared_ids': list(overlap),
        'trait_unique_ids': list(trait_unique),
        'null_unique_ids': list(null_unique)
    }

    # 2. Spatial Agreement
    if len(overlap) >= 7:
        comparison['spatial_agreement'] = 'high'
    elif len(overlap) >= 4:
        comparison['spatial_agreement'] = 'moderate'
    else:
        comparison['spatial_agreement'] = 'low'

    # 3. Matrix comparison using Mantel test
    trait_matrix = trait_results['connectivity_matrix']
    null_matrix = null_results['connectivity_matrix']

    # Find common sites
    common_sites = set(trait_matrix.index).intersection(set(null_matrix.index))

    if len(common_sites) > 10:
        common_sites = sorted(list(common_sites))
        trait_aligned = trait_matrix.loc[common_sites, common_sites].values
        null_aligned = null_matrix.loc[common_sites, common_sites].values

        # Running mantel test
        try:
            result = mantel_test(trait_aligned, null_aligned, n_permutations=999)
            if isinstance(result, tuple):
                r, p = result
            elif isinstance(result, dict):
                r = result.get('r', np.nan)
                p = result.get('p_value', 1.0)
            else:
                r, p = np.nan, 1.0
        except Exception as e:
            r, p = np.nan, 1.0

        comparison['mantel_test'] = {
            'r': r,
            'p_value': p,
            'n_sites': len(common_sites)
        }
    else:
        comparison['mantel_test'] = {'r': None, 'p_value': None, 'n_sites': len(common_sites)}

    return comparison

## Display the null model results
def display_model_comparison(trait_metrics, null_data, comparison=None,
                             title="Trait Model vs Null Model Comparison"):
    """
    Display trait model vs null model metrics side by side in a single table.

    Parameters
    ----------
    trait_metrics : dict
        Output from calculate_connectivity_metrics() for trait model.
    null_data : dict
        Output from create_null_model().
    comparison : dict, optional
        Output from compare_models() with Mantel test and hotspot overlap.
    title : str
        Title for the comparison section.
    """
    subsection_header(title)

    # Extract trait model metrics
    # Handle both direct structure and nested structure from type1 results
    trait_comm = trait_metrics.get('community', {})
    if 'community' in trait_comm:
        trait_comm = trait_comm['community']

    # Extract all trait model values
    trait_released = trait_comm.get('total_released', 0)
    trait_settled = trait_comm.get('total_settled', 0)
    trait_settlement = trait_comm.get('community_settlement', 0)
    trait_mortality = trait_comm.get('community_mortality', 0)
    trait_recruitment = trait_comm.get('community_recruitment', 0)
    trait_mean_dist = trait_comm.get('mean_distance_km', 0)
    trait_min_dist = trait_comm.get('min_distance_km', 0)
    trait_max_dist = trait_comm.get('max_distance_km', 0)
    trait_mean_pld = trait_comm.get('mean_pld_days', 0)
    trait_min_pld = trait_comm.get('min_pld_days', 0)
    trait_max_pld = trait_comm.get('max_pld_days', 0)

    # Extract all null model values
    null_total = null_data.get('n_total', 1)
    null_settled = null_data.get('n_settled', 0)
    null_settlement = null_data.get('settlement_rate', 0)
    null_mortality = 1 - null_settlement
    null_recruitment = null_data.get('local_recruitment_rate', 0)
    null_mean_dist = null_data.get('mean_distance_km', None)
    null_min_dist = null_data.get('min_distance_km', None)
    null_max_dist = null_data.get('max_distance_km', None)
    null_mean_pld = null_data.get('mean_pld_days', None)
    null_pld_window = null_data.get('pld_window', (0, 0))
    null_min_pld_actual = null_data.get('min_pld_actual', None)
    null_max_pld_actual = null_data.get('max_pld_actual', None)

    # Print single combined table
    print(f"\n{'Metric':<35} {'Trait Model':>15} {'Null Model':>15} {'Difference':>12}")
    print("-" * 80)

    # Particles
    released_diff = trait_released - null_total
    settled_diff = trait_settled - null_settled
    print(f"{'Particles released':<35} {trait_released:>15} {null_total:>15} {released_diff:>+12}")
    print(f"{'Particles settled':<35} {trait_settled:>15} {null_settled:>15} {settled_diff:>+12}")

    # Rates
    settle_diff = (trait_settlement - null_settlement) * 100
    mort_diff = (trait_mortality - null_mortality) * 100
    print(f"{'Settlement rate':<35} {trait_settlement * 100:>14.1f}% {null_settlement * 100:>14.1f}% {settle_diff:>+11.1f}%")
    print(f"{'Mortality rate':<35} {trait_mortality * 100:>14.1f}% {null_mortality * 100:>14.1f}% {mort_diff:>+11.1f}%")

    # Local recruitment
    recruitment_diff = (trait_recruitment - null_recruitment) * 100 if null_recruitment is not None else None
    if recruitment_diff is not None:
        print(f"{'Local recruitment rate':<35} {trait_recruitment * 100:>14.1f}% {null_recruitment * 100:>14.1f}% {recruitment_diff:>+11.1f}%")
    else:
        print(f"{'Local recruitment rate':<35} {trait_recruitment * 100:>14.1f}% {'N/A':>15} {'':>12}")

    # Dispersal distances
    if null_min_dist is not None:
        min_dist_diff = trait_min_dist - null_min_dist
        print(f"{'Min dispersal distance (km)':<35} {trait_min_dist:>15.1f} {null_min_dist:>15.1f} {min_dist_diff:>+12.1f}")
    else:
        print(f"{'Min dispersal distance (km)':<35} {trait_min_dist:>15.1f} {'N/A':>15} {'':>12}")

    if null_mean_dist is not None:
        mean_dist_diff = trait_mean_dist - null_mean_dist
        print(f"{'Mean dispersal distance (km)':<35} {trait_mean_dist:>15.1f} {null_mean_dist:>15.1f} {mean_dist_diff:>+12.1f}")
    else:
        print(f"{'Mean dispersal distance (km)':<35} {trait_mean_dist:>15.1f} {'N/A':>15} {'':>12}")

    if null_max_dist is not None:
        max_dist_diff = trait_max_dist - null_max_dist
        print(f"{'Max dispersal distance (km)':<35} {trait_max_dist:>15.1f} {null_max_dist:>15.1f} {max_dist_diff:>+12.1f}")
    else:
        print(f"{'Max dispersal distance (km)':<35} {trait_max_dist:>15.1f} {'N/A':>15} {'':>12}")

    # PLD
    if null_min_pld_actual is not None:
        print(f"{'Min PLD (days)':<35} {trait_min_pld:>15.0f} {null_min_pld_actual:>15.0f} {trait_min_pld - null_min_pld_actual:>+12.0f}")
    else:
        print(f"{'Min PLD (days)':<35} {trait_min_pld:>15.0f} {null_pld_window[0]:>15.0f} {'':>12}")

    if null_mean_pld is not None:
        print(f"{'Mean PLD (days)':<35} {trait_mean_pld:>15.1f} {null_mean_pld:>15.1f} {trait_mean_pld - null_mean_pld:>+12.1f}")
    else:
        print(f"{'Mean PLD (days)':<35} {trait_mean_pld:>15.1f} {'N/A':>15} {'':>12}")

    if null_max_pld_actual is not None:
        print(f"{'Max PLD (days)':<35} {trait_max_pld:>15.0f} {null_max_pld_actual:>15.0f} {trait_max_pld - null_max_pld_actual:>+12.0f}")
    else:
        print(f"{'Max PLD (days)':<35} {trait_max_pld:>15.0f} {null_pld_window[1]:>15.0f} {'':>12}")

    print("-" * 80)

    # Interpretation
    print()
    display_line("─")
    print("Change in Emerging Connectivity")
    display_line("─")

    if abs(settle_diff) > 20:
        effect = "Strong"
        print(f"\n{effect} emerging connectivity effect detected (>20% difference)")
        print("  - Traits substantially change how many particles settle")
    elif abs(settle_diff) > 10:
        effect = "Moderate"
        print(f"\n{effect} emerging connectivity effect detected (10-20% difference)")
        print("  - Traits notably influence settlement success")
    else:
        effect = "Weak"
        print(f"\n{effect} emerging connectivity effect detected (<10% difference)")
        print("  - Similar settlement with or without trait filtering")

    # If comparison data is provided, show Mantel test and hotspot overlap
    if comparison:
        if 'hotspot_overlap' in comparison:
            overlap = comparison['hotspot_overlap']
            print(f"\nHotspot Overlap Analysis:")
            print(f"  Shared hotspots: {overlap['shared_sites']}/10 ({overlap['overlap_percentage']}%)")
            print(f"  Unique to trait model: {overlap['trait_unique']}")
            print(f"  Unique to null model: {overlap['null_unique']}")

        if 'mantel_test' in comparison and comparison['mantel_test'].get('r') is not None:
            mantel = comparison['mantel_test']
            print(f"\nConnectivity Matrix Comparison (Mantel test):")
            print(f"  Correlation: r = {mantel['r']:.3f}")
            print(f"  P-value: {mantel['p_value']:.4f}")
            print(f"  Sites compared: {mantel['n_sites']}")

            if mantel['p_value'] < 0.05:
                if mantel['r'] > 0.7:
                    print("  Interpretation: Strong spatial correlation - connectivity patterns are similar")
                elif mantel['r'] > 0.4:
                    print("  Interpretation: Moderate spatial correlation - some pattern preservation")
                else:
                    print("  Interpretation: Weak spatial correlation - traits substantially alter patterns")
            else:
                print("  Interpretation: Not statistically significant")

        if 'spatial_agreement' in comparison:
            agreement_text = {
                'high': 'High agreement (7+ shared hotspots) - Traits have minimal impact on hotspot identification',
                'moderate': 'Moderate agreement (4-6 shared) - Traits alter some hotspot locations',
                'low': 'Low agreement (<4 shared) - Traits substantially change which sites are hotspots'
            }
            print(f"\nSpatial Agreement: {agreement_text.get(comparison['spatial_agreement'], comparison['spatial_agreement'])}")

    return {
        'settlement_diff': settle_diff,
        'mortality_diff': mort_diff,
        'connectivity_effect': effect
    }


def display_sensitivity_comparison(original_metrics, modified_metrics, comparison=None,
                                   title="Original vs Modified Trait Comparison"):
    """
    Display original vs modified trait model metrics side by side in a single table.

    Used for trait sensitivity analysis (Type 3 Option 2).

    Parameters
    ----------
    original_metrics : dict
        Output from calculate_connectivity_metrics() for original traits.
    modified_metrics : dict
        Output from calculate_connectivity_metrics() for modified traits.
    comparison : dict, optional
        Output from compare_models() with Mantel test and hotspot overlap.
    title : str
        Title for the comparison section.
    """
    subsection_header(title)

    # Extract original model metrics
    orig_comm = original_metrics.get('community', {})
    if 'community' in orig_comm:
        orig_comm = orig_comm['community']

    # Extract modified model metrics
    mod_comm = modified_metrics.get('community', {})
    if 'community' in mod_comm:
        mod_comm = mod_comm['community']

    # Extract all original model values
    orig_released = orig_comm.get('total_released', 0)
    orig_settled = orig_comm.get('total_settled', 0)
    orig_settlement = orig_comm.get('community_settlement', 0)
    orig_mortality = orig_comm.get('community_mortality', 0)
    orig_recruitment = orig_comm.get('community_recruitment', 0)
    orig_mean_dist = orig_comm.get('mean_distance_km', 0)
    orig_min_dist = orig_comm.get('min_distance_km', 0)
    orig_max_dist = orig_comm.get('max_distance_km', 0)
    orig_mean_pld = orig_comm.get('mean_pld_days', 0)
    orig_min_pld = orig_comm.get('min_pld_days', 0)
    orig_max_pld = orig_comm.get('max_pld_days', 0)

    # Extract all modified model values
    mod_released = mod_comm.get('total_released', 0)
    mod_settled = mod_comm.get('total_settled', 0)
    mod_settlement = mod_comm.get('community_settlement', 0)
    mod_mortality = mod_comm.get('community_mortality', 0)
    mod_recruitment = mod_comm.get('community_recruitment', 0)
    mod_mean_dist = mod_comm.get('mean_distance_km', 0)
    mod_min_dist = mod_comm.get('min_distance_km', 0)
    mod_max_dist = mod_comm.get('max_distance_km', 0)
    mod_mean_pld = mod_comm.get('mean_pld_days', 0)
    mod_min_pld = mod_comm.get('min_pld_days', 0)
    mod_max_pld = mod_comm.get('max_pld_days', 0)

    # Print single combined table
    print(f"\n{'Metric':<35} {'Original':>15} {'Modified':>15} {'Difference':>12}")
    print("-" * 80)

    # Particles
    released_diff = mod_released - orig_released
    settled_diff = mod_settled - orig_settled
    print(f"{'Particles released':<35} {orig_released:>15} {mod_released:>15} {released_diff:>+12}")
    print(f"{'Particles settled':<35} {orig_settled:>15} {mod_settled:>15} {settled_diff:>+12}")

    # Rates
    settle_diff = (mod_settlement - orig_settlement) * 100
    mort_diff = (mod_mortality - orig_mortality) * 100
    print(
        f"{'Settlement rate':<35} {orig_settlement * 100:>14.1f}% {mod_settlement * 100:>14.1f}% {settle_diff:>+11.1f}%")
    print(f"{'Mortality rate':<35} {orig_mortality * 100:>14.1f}% {mod_mortality * 100:>14.1f}% {mort_diff:>+11.1f}%")

    # Local recruitment
    recruitment_diff = (mod_recruitment - orig_recruitment) * 100
    print(
        f"{'Local recruitment rate':<35} {orig_recruitment * 100:>14.1f}% {mod_recruitment * 100:>14.1f}% {recruitment_diff:>+11.1f}%")

    # Dispersal distances
    print(
        f"{'Min dispersal distance (km)':<35} {orig_min_dist:>15.1f} {mod_min_dist:>15.1f} {mod_min_dist - orig_min_dist:>+12.1f}")
    print(
        f"{'Mean dispersal distance (km)':<35} {orig_mean_dist:>15.1f} {mod_mean_dist:>15.1f} {mod_mean_dist - orig_mean_dist:>+12.1f}")
    print(
        f"{'Max dispersal distance (km)':<35} {orig_max_dist:>15.1f} {mod_max_dist:>15.1f} {mod_max_dist - orig_max_dist:>+12.1f}")

    # PLD
    print(f"{'Min PLD (days)':<35} {orig_min_pld:>15.0f} {mod_min_pld:>15.0f} {mod_min_pld - orig_min_pld:>+12.0f}")
    print(
        f"{'Mean PLD (days)':<35} {orig_mean_pld:>15.1f} {mod_mean_pld:>15.1f} {mod_mean_pld - orig_mean_pld:>+12.1f}")
    print(f"{'Max PLD (days)':<35} {orig_max_pld:>15.0f} {mod_max_pld:>15.0f} {mod_max_pld - orig_max_pld:>+12.0f}")

    print("-" * 80)

    # Interpretation
    print()
    display_line("─")
    print("Change in Emerging Connectivity")
    display_line("─")

    if abs(settle_diff) > 20:
        effect = "Strong"
        print(f"\n{effect} sensitivity effect detected (>20% difference)")
        print("  - Modified traits substantially change settlement success")
    elif abs(settle_diff) > 10:
        effect = "Moderate"
        print(f"\n{effect} sensitivity effect detected (10-20% difference)")
        print("  - Modified traits notably influence settlement success")
    else:
        effect = "Weak"
        print(f"\n{effect} sensitivity effect detected (<10% difference)")
        print("  - Similar settlement with original and modified traits")

    # If comparison data is provided, show Mantel test and hotspot overlap
    if comparison:
        if 'hotspot_overlap' in comparison:
            overlap = comparison['hotspot_overlap']
            print(f"\nHotspot Overlap Analysis:")
            print(f"  Shared hotspots: {overlap['shared_sites']}/10 ({overlap['overlap_percentage']}%)")
            print(f"  Unique to original: {overlap['trait_unique']}")
            print(f"  Unique to modified: {overlap['null_unique']}")

        if 'mantel_test' in comparison and comparison['mantel_test'].get('r') is not None:
            mantel = comparison['mantel_test']
            print(f"\nConnectivity Matrix Comparison (Mantel test):")
            print(f"  Correlation: r = {mantel['r']:.3f}")
            print(f"  P-value: {mantel['p_value']:.4f}")
            print(f"  Sites compared: {mantel['n_sites']}")

            if mantel['p_value'] < 0.05:
                if mantel['r'] > 0.7:
                    print("  Interpretation: Strong spatial correlation - connectivity patterns are similar")
                elif mantel['r'] > 0.4:
                    print("  Interpretation: Moderate spatial correlation - some pattern preservation")
                else:
                    print("  Interpretation: Weak spatial correlation - trait changes substantially alter patterns")
            else:
                print("  Interpretation: Not statistically significant")

        if 'spatial_agreement' in comparison:
            agreement_text = {
                'high': 'High agreement (7+ shared hotspots) - Trait changes have minimal impact on hotspot identification',
                'moderate': 'Moderate agreement (4-6 shared) - Trait changes alter some hotspot locations',
                'low': 'Low agreement (<4 shared) - Trait changes substantially change which sites are hotspots'
            }
            print(
                f"\nSpatial Agreement: {agreement_text.get(comparison['spatial_agreement'], comparison['spatial_agreement'])}")

    return {
        'settlement_diff': settle_diff,
        'mortality_diff': mort_diff,
        'connectivity_effect': effect
    }

## Create comparison maps
def create_comparison_map(trait_results, null_results, output_path=None,
                          output_format='png', dpi=300,
                          title1='Original Model', title2='Modified Model'):
    """
    Create vertically stacked comparison maps of trait vs null model.

    Parameters
    ----------
    trait_results : dict
        Network results from trait-filtered model
    null_results : dict
        Network results from null model
    output_path : str, optional
        Path to save figure (without extension)
    output_format : str
        Output format ('png', 'pdf', 'tif', 'jpeg', 'svg')
    dpi : int
        Resolution for raster formats
    """
    # Get common extent
    all_lons = list(trait_results['site_metrics']['longitude']) + \
               list(null_results['site_metrics']['longitude'])
    all_lats = list(trait_results['site_metrics']['latitude']) + \
               list(null_results['site_metrics']['latitude'])

    lon_range = max(all_lons) - min(all_lons)
    lat_range = max(all_lats) - min(all_lats)

    # Dynamic figure sizing with auto layout
    fig_width, fig_height, layout = calculate_figure_dimensions(
        lon_range, lat_range, base_size=10, n_panels=2, layout='auto'
    )

    # Create subplots based on optimal layout
    if layout == 'horizontal':
        fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height),
                                 subplot_kw={'projection': ccrs.PlateCarree()})
    else:
        fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height),
                                 subplot_kw={'projection': ccrs.PlateCarree()})

    extent = [min(all_lons) - 0.15, max(all_lons) + 0.15,
              min(all_lats) - 0.15, max(all_lats) + 0.15]

    datasets = [
        (axes[0], trait_results, title1),
        (axes[1], null_results, title2)
    ]

    for ax, results, title in datasets:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)

        metrics = results['site_metrics']
        top10 = results['top10_overall']

        # All sites
        scatter = ax.scatter(
            metrics['longitude'], metrics['latitude'],
            c=metrics['overall_composite'], cmap='viridis_r',
            s=50, alpha=0.6, transform=ccrs.PlateCarree(),
            vmin=metrics['overall_composite'].min(),
            vmax=metrics['overall_composite'].max()
        )

        # Top 10 numbered
        for i, (_, site) in enumerate(top10.iterrows()):
            ax.scatter(site['longitude'], site['latitude'], s=250, c='gold',
                       edgecolors='black', linewidth=2, transform=ccrs.PlateCarree(),
                       zorder=5)
            ax.text(site['longitude'], site['latitude'], str(i + 1),
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    transform=ccrs.PlateCarree(), zorder=6)

        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        gl = ax.gridlines(draw_labels=True, alpha=0.3)
        gl.top_labels = False
        gl.right_labels = False

    # Add overall title
    fig.suptitle('Trait Sensitivity Analysis: Model Comparison',
                 fontsize=16, fontweight='bold', y=0.98)

    # Colorbar below the map
    cbar_ax = fig.add_axes([0.25, 0.12, 0.5, 0.015])
    cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Connectivity Rank (lower = higher priority)', fontsize=11)

    # Legend below colorbar
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Top 10 Priority Sites',
               markerfacecolor='gold', markeredgecolor='black', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Other Sites (colour = score)',
               markerfacecolor='#440154', markersize=8, alpha=0.7)
    ]

    fig.legend(handles=legend_elements,
               loc='lower center',
               ncol=2,
               bbox_to_anchor=(0.5, 0.02),
               framealpha=0.95,
               fontsize=10)

    # Tight layout with room for colorbar and legend at bottom
    plt.tight_layout(rect=[0, 0.18, 1, 0.98])
    plt.subplots_adjust(top=0.93, hspace=0.15)

    # Save if output path provided
    if output_path:
        fig.savefig(output_path, format=output_format, dpi=dpi, bbox_inches='tight')
        print(f"  Figure saved to: {output_path}")

    plt.show()

    return fig
# endregion

# region Post-analysis wrapper function
def post_analysis(final_processed):
    """
    Main analysis wrapper with interactive selection of analysis types.

    Guides users through four analysis pathways:
    - Type 1: Connectivity metrics (mortality, local recruitment, dispersal, PLD)
    - Type 2: Network analysis (functional roles, centrality, hotspots)
    - Type 3: Trait sensitivity analysis (trait vs null, trait sensitivity)

    Parameters
    ----------
    final_processed : dict
        Post-processed trajectory data from workflow_step(5)

    Returns
    -------
    dict
        Results from all selected analyses, accessible via:
        - results['type1'] for connectivity metrics
        - results['type2'] for network analysis
        - results['type3'] for trait sensitivity analysis
    """
    gc.collect()

    global traits, geojson_file_path, coastal_grid_df, ocean_grid_df
    global buffer_before, buffer_after

    results = {}
    output_dir = None

    # Check if temporal buffer was used (enables trait sensitivity in Type 3)
    try:
        buffer_used = (buffer_before > 0 or buffer_after > 0)
    except NameError:
        buffer_used = False
        buffer_before = 0
        buffer_after = 0

    # Step 1: Analysis Scope Selection (applies to all analyses)
    section_header("Analysis Scope Selection")

    print("\nThis package allows you to run connectivity analyses at the species level and/or at the community level.")

    # Get list of species with data
    species_with_data = [sp for sp, df in final_processed.items() if not df.empty]
    n_species = len(species_with_data)

    print(f"\nYou have {n_species} species with trajectory data:")
    for i, sp in enumerate(species_with_data, 1):
        n_traj = final_processed[sp]['trajectory'].nunique()
        print(f"  {i}. {display_species_name(sp)} ({n_traj:} trajectories)")

    print("\nHow would you like to run the connectivity analyses?")
    print("  1. Community level only (all species combined)")
    print("  2. Per-species only (each species analysed separately)")
    print("  3. Both community and per-species")
    print("  4. Custom species selection")

    while True:
        scope_choice = input("\nSelect option (1-4) [default: 1]: ").strip() or '1'
        if scope_choice in ['1', '2', '3', '4']:
            break
        print("  Invalid input. Please enter 1, 2, 3 or 4.")

    # Set up analysis scope
    analysis_scope = {
        'run_community': scope_choice in ['1', '3'],
        'run_per_species': scope_choice in ['2', '3', '4'],
        'selected_species': species_with_data  # Default to all
    }

    if scope_choice == '4':
        # User selects specific species
        print("\nEnter species numbers separated by commas (e.g., 1,3,5)")
        print("Or press Enter to select all species")
        indices_str = input("Species numbers: ").strip()

        if indices_str:
            try:
                selected_indices = [int(x.strip()) - 1 for x in indices_str.split(',')]
                analysis_scope['selected_species'] = [
                    species_with_data[i] for i in selected_indices
                    if 0 <= i < len(species_with_data)
                ]
                print(f"\nSelected {len(analysis_scope['selected_species'])} species:")
                for sp in analysis_scope['selected_species']:
                    print(f"  - {display_species_name(sp)}")

                # Single species: run as community only (per-species would be identical)
                if len(analysis_scope['selected_species']) == 1:
                    analysis_scope['run_community'] = True
                    analysis_scope['run_per_species'] = False

            except ValueError:
                print("Invalid input. Using all species.")
                analysis_scope['selected_species'] = species_with_data

    # Create filtered data for selected species (always create this)
    selected_processed = {
        sp: final_processed[sp] for sp in analysis_scope['selected_species']
    }

    # Store scope in results
    results['analysis_scope'] = analysis_scope

    # Print scope summary
    section_header("Scope Summary")
    scope_summary = []
    if analysis_scope['run_community']:
        scope_summary.append("Community level")
    if analysis_scope['run_per_species']:
        scope_summary.append(f"Per-species ({len(analysis_scope['selected_species'])} species)")
    print(f"Analysis scope: {' + '.join(scope_summary)}")

    # Step 2: Select Which Analyses to Run
    section_header("Analysis Selection")

    run_type1 = get_yes_no("\nRun Type 1: Connectivity Metrics? (yes/no)", default='no')
    run_type2 = get_yes_no("Run Type 2: Network Analysis? (yes/no)", default='no')
    run_type3 = get_yes_no("Run Type 3: Trait Sensitivity? (yes/no)", default='no')

    if not any([run_type1, run_type2, run_type3]):
        print("\nNo analyses selected. Returning empty results.")
        return results

    # Type 1: Connectivity Metrics
    if run_type1:
        section_header("Type 1: Connectivity Metrics")

        try:
            type1_results = {}

            # Community-level analysis
            if analysis_scope['run_community'] and not analysis_scope['run_per_species']:
                # Community only: show community table, suppress per-species summaries
                community_results = calculate_connectivity_metrics(
                    selected_processed, release_grids, settlement_grids, traits, traj,
                    print_summary=False
                )
                type1_results['community'] = community_results
                display_metrics_table(community_results, show='community')

            elif analysis_scope['run_community'] and analysis_scope['run_per_species']:
                # Both: show per-species summaries AND community table
                community_results = calculate_connectivity_metrics(
                    selected_processed, release_grids, settlement_grids, traits, traj,
                    print_summary=True
                )
                type1_results['community'] = community_results
                display_metrics_table(community_results, show='community')

            # Per-species analysis (only when per-species is selected without community)
            if analysis_scope['run_per_species'] and not analysis_scope['run_community']:
                type1_results['per_species'] = {}
                for i, sp in enumerate(analysis_scope['selected_species'], 1):
                    sp_data = {sp: selected_processed[sp]}
                    sp_release = {sp: release_grids.get(sp)} if sp in release_grids else {}
                    sp_settle = {
                        sp: settlement_grids.get(sp)} if settlement_grids and sp in settlement_grids else None
                    sp_traits = [t for t in traits if standardise_species_name(t.get('species_name', '')) == sp]
                    if sp_traits:
                        sp_results = calculate_connectivity_metrics(
                            sp_data, sp_release, sp_settle, sp_traits, traj,
                            print_summary=True
                        )
                        type1_results['per_species'][sp] = sp_results

            results['type1'] = type1_results

            # Ask about saving
            if get_yes_no("\nSave Type 1 results to CSV?", default='no'):
                default_dir = get_default_dir('results')
                output_dir = input(f"Output directory [default: {default_dir}]: ").strip() or default_dir
                os.makedirs(output_dir, exist_ok=True)

                for key, result_data in type1_results.items():
                    if key == 'per_species':
                        for sp, sp_data in result_data.items():
                            if 'community' in sp_data:
                                sp_df = pd.DataFrame([sp_data['community']])
                                sp_df.to_csv(os.path.join(output_dir, f'connectivity_metrics_{sp}.csv'),
                                             index=False)
                    elif 'community' in result_data:
                        comm_df = pd.DataFrame([result_data['community']])
                        comm_df.to_csv(os.path.join(output_dir, f'connectivity_metrics_{key}.csv'), index=False)
                print(f"Results saved to: {output_dir}")

            print("\n Type 1 analysis complete")

        except Exception as e:
            print(f"\nError in Type 1 analysis: {e}")
            traceback.print_exc()
            print("Continuing to next analysis...")

    # Type 2: Network Analysis
    if run_type2:
        section_header("Type 2: Network Analysis")

        # Step 2a: Spatial Scale Selection
        subsection_header("Spatial Scale Selection")

        print("\nDistance thresholds filter which site-to-site connections are included.")
        print("Only connections where larvae travel less than the threshold are kept.")
        print("  - Smaller thresholds reveal local stepping-stone connectivity")
        print("  - Larger thresholds include long-distance dispersal events")

        print("\nOptions:")
        print("  1. Full domain (include all connections regardless of distance)")
        print("  2. Single scale (set one distance threshold)")
        print("  3. Multi-scale (compare local, regional and seascape patterns)")

        while True:
            scale_choice = input("\nSelect option (1/2/3) [default: 1]: ").strip() or '1'
            if scale_choice in ['1', '2', '3']:
                break
            print("  Invalid input. Please enter 1, 2, or 3.")

        if scale_choice == '1':
            scales = [None]
            scale_names = ['full_domain']
            print("\nUsing full domain. All connections included")
        elif scale_choice == '2':
            scale_km = get_positive_integer(
                "Enter distance threshold in km",
                default=30, min_val=1, max_val=10000
            )
            scales = [scale_km]
            scale_names = ['single']
            print(f"\nUsing {scale_km} km distance threshold")
        else:
            local_km = get_positive_integer("Local scale threshold (km)", default=10, min_val=1, max_val=50)
            regional_km = get_positive_integer("Regional scale threshold (km)", default=30, min_val=1, max_val=100)
            seascape_km = get_positive_integer("Seascape scale threshold (km)", default=100, min_val=1, max_val=1000)
            scales = [local_km, regional_km, seascape_km]
            scale_names = ['local', 'regional', 'seascape']
            print(f"\nUsing multi-scale: {local_km}km (local), {regional_km}km (regional), {seascape_km}km (seascape)")

        # Step 2b: Metric Selection
        subsection_header("Metric Selection")

        print("\nWhich metrics do you want to use for hotspot ranking?")
        print("  1. Functional roles only (source, sink, local recruitment)")
        print("  2. Centrality measures only (betweenness, closeness, degree)")
        print("  3. Overall composite (all 6 metrics)")
        print("  4. Individual metric (choose one)")
        print("  5. Custom composite (choose which metrics to combine)")

        while True:
            metric_choice = input("\nSelect option (1-5) [default: 3]: ").strip() or '3'
            if metric_choice in ['1', '2', '3', '4', '5']:
                break
            print("  Invalid input. Please enter 1, 2, 3, 4 or 5.")

        # Determine which composite/metric to use
        if metric_choice == '1':
            ranking_col = 'functional_composite'
            top10_key = 'top10_functional'
            composite_name = "Functional Roles"
            selected_metrics = ['source', 'sink', 'local_recruitment']
        elif metric_choice == '2':
            ranking_col = 'centrality_composite'
            top10_key = 'top10_centrality'
            composite_name = "Centrality Measures"
            selected_metrics = ['betweenness', 'closeness', 'degree']
        elif metric_choice == '4':
            # Individual metric selection
            print("\nAvailable metrics:")
            print("  1. Source: Outgoing connection strength")
            print("  2. Sink: Incoming connection strength")
            print("  3. Local Recruitment: Self-recruitment")
            print("  4. Betweenness: Bridge importance")
            print("  5. Closeness: Network centrality")
            print("  6. Degree: Total connections")

            metric_map = {
                '1': 'source', '2': 'sink', '3': 'local_recruitment',
                '4': 'betweenness', '5': 'closeness', '6': 'degree'
            }
            while True:
                ind_choice = input("\nSelect metric (1-6): ").strip()
                if ind_choice in metric_map:
                    break
                print("  Invalid input. Please enter 1-6.")

            selected_metric = metric_map[ind_choice]
            ranking_col = f'{selected_metric}_rank'
            top10_key = None  # Will create custom
            composite_name = selected_metric.replace('_', ' ').title()
            selected_metrics = [selected_metric]
        elif metric_choice == '5':
            # Custom composite
            print("\nAvailable metrics: source, sink, local_recruitment, betweenness, closeness, degree")
            custom_input = input("Enter metrics to combine (comma-separated): ").strip().lower()
            selected_metrics = [m.strip().replace(' ', '_') for m in custom_input.split(',')]

            # Validate
            valid_metrics = ['source', 'sink', 'local_recruitment', 'betweenness', 'closeness', 'degree']
            selected_metrics = [m for m in selected_metrics if m in valid_metrics]

            if not selected_metrics:
                print("No valid metrics selected. Using overall composite.")
                ranking_col = 'overall_composite'
                top10_key = 'top10_overall'
                composite_name = "Overall (All Metrics)"
                selected_metrics = valid_metrics
            else:
                ranking_col = 'custom_composite'
                top10_key = None
                composite_name = f"Custom ({', '.join(selected_metrics)})"
        else:  # Default: overall
            ranking_col = 'overall_composite'
            top10_key = 'top10_overall'
            composite_name = "Overall (All Metrics)"
            selected_metrics = ['source', 'sink', 'local_recruitment', 'betweenness', 'closeness', 'degree']

        print(f"\nUsing: {composite_name}")

        # Step 2c: Visualization Selection
        subsection_header("Visualisation Options")

        print("\n  1. Network map only")
        print("  2. Connectivity matrix heatmap only")
        print("  3. Both network map and heatmap")
        print("  4. Skip visualisations")

        while True:
            vis_choice = input("\nSelect option (1-4) [default: 3]: ").strip() or '3'
            if vis_choice in ['1', '2', '3', '4']:
                break
            print("  Invalid input. Please enter 1, 2, 3 or 4.")

        save_figs = False
        if vis_choice in ['1', '2', '3']:
            save_figs = get_yes_no("Save figures to file?", default='no')
            if save_figs:
                default_dir = get_default_dir('figures')
                output_dir = input(f"Output directory [default: {default_dir}]: ").strip() or default_dir
                os.makedirs(output_dir, exist_ok=True)

        # Step 2d: Run Analysis
        subsection_header("Running Network Analysis")

        try:
            all_network_results = {}

            for scale_km, scale_name in zip(scales, scale_names):
                if scale_km is None:
                    print(f"\nAnalysing: Full Domain (no distance filtering)")
                else:
                    print(f"\nAnalysing: {scale_km} km scale")

                scale_results = {}

                # Community-level network analysis
                if analysis_scope['run_community']:
                    subsection_header(f"Community Level ({len(analysis_scope['selected_species'])} species combined)")
                    community_network = network_analysis(
                        selected_processed, release_grids, settlement_grids,
                        grid_df_for_mapping, grid_ids, full_grid_df, scale_km
                    )
                    if community_network:
                        scale_results['community'] = community_network

                # Per-species network analysis
                if analysis_scope['run_per_species']:
                    scale_results['per_species'] = {}
                    for i, sp in enumerate(analysis_scope['selected_species'], 1):
                        subsection_header(
                            f"Species: {display_species_name(sp)} ({i}/{len(analysis_scope['selected_species'])})")
                        sp_data = {sp: final_processed[sp]}
                        sp_settle = {
                            sp: settlement_grids[sp]} if settlement_grids and sp in settlement_grids else None

                        # Pass FULL release_grids to maintain consistent study extent across all species
                        sp_network = network_analysis(
                            sp_data, release_grids, sp_settle,
                            grid_df_for_mapping, grid_ids, full_grid_df, scale_km
                        )
                        if sp_network:
                            scale_results['per_species'][sp] = sp_network

                if scale_results:
                    all_network_results[scale_name] = scale_results

            results['type2'] = all_network_results

            # Step 2e: Display Results with Selected Metric
            if all_network_results:
                # Get primary results for display
                first_scale = scale_names[0]
                if 'community' in all_network_results[first_scale]:
                    primary_results = all_network_results[first_scale]['community']
                elif 'per_species' in all_network_results[first_scale]:
                    first_sp = list(all_network_results[first_scale]['per_species'].keys())[0]
                    primary_results = all_network_results[first_scale]['per_species'][first_sp]
                else:
                    primary_results = None

                if primary_results:
                    # Handle custom composite or individual metric
                    if ranking_col == 'custom_composite':
                        valid_cols = [f'{m}_rank' for m in selected_metrics
                                      if f'{m}_rank' in primary_results['site_metrics'].columns]
                        if valid_cols:
                            primary_results['site_metrics']['custom_composite'] = \
                                primary_results['site_metrics'][valid_cols].mean(axis=1)
                            primary_results['top10_custom'] = \
                                primary_results['site_metrics'].nsmallest(10, 'custom_composite').copy()
                            top10_to_use = primary_results['top10_custom']
                        else:
                            top10_to_use = primary_results['top10_overall']
                    elif top10_key and top10_key in primary_results:
                        top10_to_use = primary_results[top10_key]
                    elif ranking_col.endswith('_rank'):
                        # Individual metric - create top 10
                        if ranking_col in primary_results['site_metrics'].columns:
                            top10_to_use = primary_results['site_metrics'].nsmallest(10, ranking_col).copy()
                        else:
                            top10_to_use = primary_results['top10_overall']

                    # Display top 10 with correct header (OUTSIDE the elif - runs for ALL cases)
                    subsection_header(f"Top 10 Hotspots: {composite_name}")
                    print(f"{'Rank':<6} {'Grid ID':<10} {'Location':<22} {'Score':>8}")
                    print("-" * 48)

                    score_col = ranking_col if ranking_col in top10_to_use.columns else 'overall_composite'
                    for idx, (_, site) in enumerate(top10_to_use.iterrows()):
                        location = f"({site['latitude']:.2f}, {site['longitude']:.2f})"
                        print(f"{idx + 1:<6} {int(site['grid_id']):<10} {location:<22} {site[score_col]:>8.1f}")

                    # Create visualizations (OUTSIDE the elif - runs for ALL cases)
                    if vis_choice in ['1', '2', '3']:
                        # Determine what we're showing
                        if 'community' in all_network_results[first_scale]:
                            fig_level = "Community Level"
                        else:
                            first_sp = list(all_network_results[first_scale]['per_species'].keys())[0]
                            fig_level = f"Species: {display_species_name(first_sp)}"

                        subsection_header(f"Displaying Figures: {fig_level}")

                    if vis_choice in ['1', '3']:
                        scale_label = scales[0] if scales[0] else "Full Domain"
                        title = f"Connectivity Hotspots ({fig_level}): {composite_name}"
                        if scales[0]:
                            title += f" (≤{scale_label} km)"

                        if save_figs and output_dir:
                            output_path = os.path.join(output_dir,
                                                       f'network_map_{composite_name.replace(" ", "_")}.png')
                        else:
                            output_path = None

                        create_network_map(
                            primary_results['site_metrics'],
                            top10_to_use,
                            title=title,
                            composite_col=score_col,
                            output_path=output_path,
                            show_figure=True
                        )

                    if vis_choice in ['2', '3']:
                        scale_label = scales[0] if scales[0] else "Full Domain"
                        title = f"Connectivity Matrix ({fig_level})"
                        if scales[0]:
                            title += f" (≤{scale_label} km)"

                        create_connectivity_matrix_plot(
                            primary_results['connectivity_matrix'],
                            title=title
                        )

                    # Offer to display individual species figures if per-species was run
                    if analysis_scope['run_per_species'] and 'per_species' in all_network_results[first_scale]:
                        n_species_with_results = len(all_network_results[first_scale]['per_species'])
                        if n_species_with_results > 0:
                            print(f"\nNote: {n_species_with_results} individual species figures are available.")
                            if save_figs and output_dir:
                                print(f"These will be saved to: {output_dir}")
                            show_individual = get_yes_no("Display individual species figures?", default='no')

                            if show_individual or save_figs:
                                for sp, sp_results in all_network_results[first_scale]['per_species'].items():
                                    sp_display = display_species_name(sp)

                                    # Create top10 for this species
                                    if ranking_col == 'custom_composite':
                                        valid_cols = [f'{m}_rank' for m in selected_metrics
                                                      if f'{m}_rank' in sp_results['site_metrics'].columns]
                                        if valid_cols:
                                            sp_results['site_metrics']['custom_composite'] = \
                                                sp_results['site_metrics'][valid_cols].mean(axis=1)
                                            sp_top10 = sp_results['site_metrics'].nsmallest(10,
                                                                                            'custom_composite').copy()
                                        else:
                                            sp_top10 = sp_results['top10_overall']
                                    elif top10_key and top10_key in sp_results:
                                        sp_top10 = sp_results[top10_key]
                                    elif ranking_col.endswith('_rank') and ranking_col in sp_results[
                                        'site_metrics'].columns:
                                        sp_top10 = sp_results['site_metrics'].nsmallest(10, ranking_col).copy()
                                    else:
                                        sp_top10 = sp_results['top10_overall']

                                    if show_individual and vis_choice in ['1', '3']:
                                        sp_title = f"Connectivity Hotspots ({sp_display}): {composite_name}"
                                        if scales[0]:
                                            sp_title += f" (≤{scale_label} km)"
                                        sp_output = os.path.join(output_dir,
                                                                 f'network_map_{sp}_{composite_name.replace(" ", "_")}.png') if save_figs and output_dir else None
                                        create_network_map(sp_results['site_metrics'], sp_top10, title=sp_title,
                                                           composite_col=score_col, output_path=sp_output,
                                                           show_figure=True)

                                    if show_individual and vis_choice in ['2', '3']:
                                        sp_title = f"Connectivity Matrix ({sp_display})"
                                        if scales[0]:
                                            sp_title += f" (≤{scale_label} km)"
                                        create_connectivity_matrix_plot(sp_results['connectivity_matrix'],
                                                                        title=sp_title)

                                    # Save only (no display)
                                    if save_figs and not show_individual and output_dir:
                                        if vis_choice in ['1', '3']:
                                            sp_output = os.path.join(output_dir,
                                                                     f'network_map_{sp}_{composite_name.replace(" ", "_")}.png')
                                            fig = create_network_map(sp_results['site_metrics'], sp_top10,
                                                                     title=f"Connectivity Hotspots ({sp_display})",
                                                                     composite_col=score_col, output_path=sp_output,
                                                                     show_figure=False)
                                        if vis_choice in ['2', '3']:
                                            sp_output = os.path.join(output_dir,
                                                                     f'connectivity_matrix_{sp}_{composite_name.replace(" ", "_")}.png')
                                            fig = create_connectivity_matrix_plot(sp_results['connectivity_matrix'],
                                                                                  title=f"Connectivity Matrix ({sp_display})",
                                                                                  output_path=sp_output)
                                            plt.close(fig)
                                print(f"  Individual species figures {'saved' if save_figs else 'displayed'}.")

                print("\n Type 2 analysis complete")

        except Exception as e:
            print(f"\nError in Type 2 analysis: {e}")
            traceback.print_exc()
            print("Continuing to next analysis...")

    # Type 3: Trait Sensitivity Analysis
    if run_type3:
        section_header("Type 3: Trait Sensitivity Analysis")

        print("\nOptions:")
        print("  1. Trait model vs null model comparison")
        if buffer_used:
            print("  2. Trait sensitivity (alternative trait scenarios)")
        else:
            print("  2. [Unavailable] Trait sensitivity requires a temporal buffer in the simulation")
            print("To use this feature, re-run workflow_step(1) with:")
            print("buffer_before > 0  or  buffer_after > 0")

        while True:
            type3_choice = input(f"\nSelect option (1{'/2' if buffer_used else ''}): ").strip()
            if type3_choice == '1' or (type3_choice == '2' and buffer_used):
                break
            print(f"  Invalid input. Please enter 1{' or 2' if buffer_used else ''}.")

        if type3_choice == '1':
            # Option 1: Null model comparison
            subsection_header("Trait Model vs Null Model Comparison")

            print("The null model assesses maximum dispersal ability by removing trait-based filtering.")
            print("\nIt assesses connectivity if species are allowed to settle anywhere, anytime.")
            print("\nIt uses the overall simulation length and allows species to settle anytime between the start and end.")
            print("\nThe null model results are then compared to trait-filtered results.")
            print("\nThis assesses how traits influence connectivity.")

            # Scale selection
            print("\nNetwork analysis scale selection:")
            print("  1. Full domain (no distance filtering, all connections included)")
            print("  2. Specify distance threshold")
            if 'type2' in results and scales:
                print("  3. Use same scale as Type 2 analysis")

            while True:
                scale_opt = input(f"\nSelect option (1/2{'/3' if 'type2' in results else ''}) [default: 1]: ").strip() or '1'
                if scale_opt in ['1', '2', '3']:
                    break
                print("  Invalid input.")

            if scale_opt == '1':
                null_scale = None
                print("\nUsing full domain. All connections included")
            elif scale_opt == '3' and 'type2' in results and scales:
                null_scale = scales[0]
                print(f"\nUsing {null_scale} km threshold (same as Type 2)")
            else:
                null_scale = get_positive_integer("Distance threshold in km", default=30, min_val=1, max_val=10000)
                print(f"\nUsing {null_scale} km distance threshold")

            try:
                # Get or create trait model results
                if 'type2' in results:
                    first_key = list(results['type2'].keys())[0]
                    if 'community' in results['type2'][first_key]:
                        trait_network = results['type2'][first_key]['community']
                    elif 'per_species' in results['type2'][first_key]:
                        first_sp = list(results['type2'][first_key]['per_species'].keys())[0]
                        trait_network = results['type2'][first_key]['per_species'][first_sp]
                    else:
                        trait_network = None
                else:
                    print("\nRunning network analysis for trait model...")
                    trait_network = network_analysis(
                        selected_processed, release_grids, settlement_grids,
                        grid_df_for_mapping, grid_ids, full_grid_df, null_scale
                    )

                if trait_network:
                    # Get study bounds
                    if 'study_bounds' in trait_network:
                        study_bounds = trait_network['study_bounds']
                    else:
                        study_bounds = {
                            'lat_min': coastal_grid_df['latitude'].min(),
                            'lat_max': coastal_grid_df['latitude'].max(),
                            'lon_min': coastal_grid_df['longitude'].min(),
                            'lon_max': coastal_grid_df['longitude'].max()
                        }

                    # Create null model
                    print("\nCreating null model...")
                    null_data = create_null_model(
                        traj, grid_df_for_mapping, study_bounds,
                        full_grid_df, spp_params, traits
                    )

                    if null_data:
                        print("\nRunning network analysis on null model...")
                        null_network = network_analysis(
                            null_data['processed'],
                            null_data['release_grids'],
                            null_data['settlement_grids'],
                            grid_df_for_mapping, grid_ids, full_grid_df, null_scale
                        )

                        if null_network:
                            # Compare models
                            comparison = compare_models(trait_network, null_network)

                            # Display comparison
                            if 'type1' in results:
                                display_model_comparison(results['type1'], null_data, comparison)

                            results['type3'] = {
                                'comparison': comparison,
                                'trait_network': trait_network,
                                'null_network': null_network,
                                'null_data': null_data,
                                'scale_km': null_scale
                            }

                            # Offer comparison map
                            if get_yes_no("\nCreate comparison map?", default='no'):
                                create_comparison_map(
                                    trait_network, null_network,
                                    title1='Trait Model',
                                    title2='Null Model'
                                )

                print("\n Type 3 analysis complete")

            except Exception as e:
                print(f"\nError in null model analysis: {e}")
                traceback.print_exc()

        elif type3_choice == '2' and buffer_used:
            # Option 2: Trait sensitivity
            subsection_header("Trait Sensitivity Analysis")

            print("\nThis tests how connectivity changes with different trait values.")
            print(f"Your temporal buffer: {buffer_before} days before, {buffer_after} days after")

            # Get scale
            if 'type2' in results and scales:
                sensitivity_scale = scales[0]
                print(f"\nUsing {sensitivity_scale} km threshold from Type 2")
            else:
                sensitivity_scale = get_positive_integer("Distance threshold in km", default=30, min_val=1, max_val=500)

            # Get original network
            if 'type2' in results:
                first_key = list(results['type2'].keys())[0]
                if 'community' in results['type2'][first_key]:
                    original_network = results['type2'][first_key]['community']
                else:
                    first_sp = list(results['type2'][first_key]['per_species'].keys())[0]
                    original_network = results['type2'][first_key]['per_species'][first_sp]
            else:
                print("\nRunning network analysis first...")
                original_network = network_analysis(
                    selected_processed, release_grids, settlement_grids,
                    grid_df_for_mapping, grid_ids, full_grid_df, sensitivity_scale
                )

            print("\nSelect trait modification scenario:")
            print("  1. Modify PLD (pelagic larval duration)")
            print("  2. Shift spawning window")

            while True:
                mod_choice = input("\nSelect scenario (1/2): ").strip()
                if mod_choice in ['1', '2']:
                    break
                print("  Invalid input. Please enter 1 or 2.")

            modified_traits = None
            scenario_name = ""

            if mod_choice == '1':
                # PLD modification
                print("\nPLD Modification (in days):")
                print("  - You can modify min_PLD and max_PLD separately.")
                print("  - Enter positive to increase, negative to decrease.")
                print("  - Example: +5 adds 5 days, -3 removes 3 days")

                print("\nOptions:")
                print("  1. Same change to both min and max PLD")
                print("  2. Different changes to min and max PLD")
                pld_option = input("\nSelect (1/2) [default: 1]: ").strip() or '1'

                if pld_option == '2':
                    min_change = int(input("Days to add/subtract from min_PLD: ").strip())
                    max_change = int(input("Days to add/subtract from max_PLD: ").strip())
                    scenario_name = f"PLD min{'+' if min_change >= 0 else ''}{min_change}, max{'+' if max_change >= 0 else ''}{max_change}"
                else:
                    days = int(input("Days to add/subtract from both: ").strip())
                    min_change = days
                    max_change = days
                    scenario_name = f"PLD {'+' if days > 0 else ''}{days} days"

                modified_traits = []
                for t in traits:
                    new_t = t.copy() if isinstance(t, dict) else t.to_dict()
                    new_t['min_PLD'] = max(1, t['min_PLD'] + min_change)
                    new_t['max_PLD'] = max(1, t['max_PLD'] + max_change)
                    modified_traits.append(new_t)

            else:
                # Spawning window modification
                print("\nSpawning Window Modification (in days):")
                print("  - You can modify the start and end dates separately")
                print("  - Positive = later, Negative = earlier")
                print("  - Example: Start -3, End +5 = spawning starts 3 days earlier, ends 5 days later")

                print("\nOptions:")
                print("  1. Shift entire window (same change to start and end)")
                print("  2. Modify start and end separately")
                spawn_option = input("\nSelect (1/2) [default: 1]: ").strip() or '1'

                if spawn_option == '2':
                    start_change = int(input("Days to shift spawning start (negative = earlier): ").strip())
                    end_change = int(input("Days to shift spawning end (positive = later): ").strip())
                    scenario_name = f"Spawn start{'+' if start_change >= 0 else ''}{start_change}, end{'+' if end_change >= 0 else ''}{end_change}"
                else:
                    shift_days = int(input("Days to shift entire window (positive = later): ").strip())
                    start_change = shift_days
                    end_change = shift_days
                    scenario_name = f"Spawn {'+' if shift_days > 0 else ''}{shift_days} days"

                modified_traits = []
                for t in traits:
                    new_t = t.copy() if isinstance(t, dict) else t.to_dict()
                    spawn_start = pd.to_datetime(t['spawning_start'], dayfirst=True)
                    spawn_end = pd.to_datetime(t['spawning_end'], dayfirst=True)
                    new_t['spawning_start'] = (spawn_start + pd.Timedelta(days=start_change)).strftime('%d/%m/%Y')
                    new_t['spawning_end'] = (spawn_end + pd.Timedelta(days=end_change)).strftime('%d/%m/%Y')
                    modified_traits.append(new_t)

            if modified_traits:
                print(f"\nScenario: {scenario_name}")
                print("Re-running post-processing with modified traits...")

                try:
                    modified_processed = post_process(
                        traj, modified_traits, release_grids, settlement_grids, grid_ids
                    )

                    print("\nRunning network analysis on modified scenario...")
                    modified_network = network_analysis(
                        modified_processed, release_grids, settlement_grids,
                        grid_df_for_mapping, grid_ids, full_grid_df, sensitivity_scale
                    )

                    if modified_network:
                        comparison = compare_models(original_network, modified_network)

                        # Calculate connectivity metrics for both models for the comparison table
                        print("\nCalculating comparison metrics...")
                        original_type1 = calculate_connectivity_metrics(
                            selected_processed, release_grids, settlement_grids, traits, traj
                        )
                        modified_type1 = calculate_connectivity_metrics(
                            modified_processed, release_grids, settlement_grids, modified_traits, traj
                        )

                        # Display comparison table
                        display_sensitivity_comparison(
                            original_type1, modified_type1, comparison,
                            title=f"Original vs {scenario_name}"
                        )

                        results['type3_sensitivity'] = {
                            'scenario': scenario_name,
                            'comparison': comparison,
                            'original_network': original_network,
                            'modified_network': modified_network,
                            'original_metrics': original_type1,
                            'modified_metrics': modified_type1
                        }

                        if get_yes_no("\nCreate comparison map?", default='no'):
                            create_comparison_map(
                                original_network, modified_network,
                                title1='Original Traits',
                                title2=scenario_name
                            )

                    print("\n Type 3 sensitivity analysis complete")

                except Exception as e:
                    print(f"\nError in trait sensitivity: {e}")
                    traceback.print_exc()

    # Final Summary
    section_header("Analyses Complete")

    print("\nCompleted analyses:")
    if 'type1' in results:
        print("  Type 1: Connectivity Metrics")
    if 'type2' in results:
        print("  Type 2: Network Analysis")
    if 'type3' in results or 'type3_sensitivity' in results:
        print("  Type 3: Trait Sensitivity Analysis")

    return results
# endregion
# endregion

# region Complete Interactive Workflow
def workflow_step(step_number):
    """
    Main function to guide the user through the interactive workflow.
    Calls necessary functions in sequence based on step number.

    Steps:
    1. Hydrodynamic data setup
    2. Species occurrence data
    3. Simulation setup (grids, fieldset, particles)
    4. Execute simulation
    5. Post-processing (trait filters)
    6. Connectivity analyses

    Global Variables
    ----------------
    Extensive use of global variables for data persistence between steps.
    See declaration list at start of function.
    """

    # Declare Global variables
    global project_dirs, runtime, species_traits_path, traits, hydro_params, hydro_data, buffer_before, buffer_after
    global clean_spp, spp_params, landmask, coastalmask, lonarray, latarray
    global meshgrid, release_grids, settlement_grids, full_grid_df, grid_ids
    global coastal_grid, coastal_grid_df, coastal_grid_filtered
    global ocean_grid, ocean_grid_df, grid_df_for_mapping
    global fieldset, adjusted_runtime, interval, pset
    global kernels
    global output_path, traj
    global final_processed, buffer
    global analysis_results

    # Track workflow progress
    workflow_status = {
        1: 'not_started',
        2: 'not_started',
        3: 'not_started',
        4: 'not_started',
        5: 'not_started',
        6: 'not_started'
    }

    # Update status based on globals
    if 'hydro_params' in globals():
        workflow_status[1] = 'complete'
    if 'spp_params' in globals():
        workflow_status[2] = 'complete'
    if 'release_grids' in globals() or 'settlement_grids' in globals():
        workflow_status[3] = 'complete'
    if 'pset' in globals():
        workflow_status[4] = 'complete'
    if 'traj' in globals() or 'filtered_trajectories' in globals():
        workflow_status[5] = 'complete'

    # Mark current step
    workflow_status[step_number] = 'in_progress'

    # Display progress
    symbols = {
        'complete': '✓',
        'in_progress': '→',
        'not_started': ' '
    }

    progress = ' '.join([f"[{symbols[workflow_status[i]]}]" for i in range(1, 7)])

    # Print title with dynamic centering
    workflow_progress_header(step_number, workflow_status)

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 1: Hydrodynamic Data
    # ═══════════════════════════════════════════════════════════════════════════

    if step_number == 1:
        step_box(1, "Hydrodynamic and Trait Data")

        subsection_header("Data Source")
        info_box([
            "This package uses hydrodynamic data from the Copernicus Marine Service.",
            "",
            "Use the link below to set up a free account to browse and select your hydrodynamic data.",
            "",
            "After finding and clicking on the data you want to use, click on 'Data access' and copy the dataset ID.",
            "",
            "For example: cmems_mod_ibi_phy_anfc_0.027deg-2D_PT1H-m",
            "",
            "To access the data, you will need:",
            "  - A stable internet connection",
            "  - Username and password",
            "  - Dataset ID number (e.g. cmems_mod_ibi_phy_anfc_0.027deg-2D_PT1H-m)",
            "  - Longitude and latitude ranges of your study area"
        ])

        subsection_header("Species Trait Data")
        info_box([
            "This package requires species trait data with information on:",
            "  - Planktonic larval duration (PLD)",
            "  - Timing of spawning for each species",
            "",
            "This data will be used to run the species-specific connectivity analyses, as well as select the time range of the hydrodynamic data and the simulation runtime.",
            "",
            "You can upload this data via this workflow or select species and their traits directly from an integrated dataset of common species found in intertidal rocky shore habitats of the temperate Atlantic. This dataset is based on a literature review of peer-reviewed data.",
            "",
            "For more information, see 'Part 3: Step 1' and 'Part 5: Species Traits CSV Format' in the User Guide."
        ])

        # Project Folder Setup (before data selection)
        # Set up project directory if not already created in this session
        if 'project_dirs' not in globals() or project_dirs is None:
            subsection_header("Project Setup")
            info_box([
                "ComCon organises all outputs into a project folder.",
                "Each project gets its own directory with subfolders for configuration, data, figures, simulation output and results.",
                "",
                "This keeps your work organised and prevents files being overwritten between different analyses.",
            ])
            project_dirs = setup_project_directory()
        else:
            subsection_header("Project Directory")
            print(f"\n  Using existing project directory: {project_dirs['root']}\n")
            display_line("─")

        subsection_header("Copernicus Marine Service Website")
        print("https://data.marine.copernicus.eu/products\n")

        subsection_header("Setting up the Hydrodynamic Extent")
        info_box([
            "The hydrodynamic extent needs to be larger than your study area.",
            "",
            "The hydrodynamic extent is the area in which particles will be dispersing during the simulation.",
            "",
            "The study extent is the area in which you are interested in assessing connectivity (the release and settlement sites of your species, not the area in which they disperse).",
            "",
            "You will set up the hydrodynamic extent in workflow step 1 and the study extent in workflow step 2."
        ])

        display_line("═")

        # Offer choice: download new or load existing
        section_header("Select your Hydrodynamic Data")

        info_box([
            "Choose how to load your hydrodynamic data:",
            "",
            "  1. Download from Copernicus Marine Service (requires internet)",
            "     - ComCon streams the data directly and there is no large download needed.",
            "     - You can optionally save a local copy for future offline use.",
            "",
            "  2. Load from a local file (no internet required)",
            "     - Use a previously saved NetCDF (.nc) or Zarr (.zarr) file.",
            "     - The file must contain 'uo' and 'vo' velocity variables with time, latitude and longitude dimensions.",
            "     - See 'Part 5: Hydrodynamic Data File Format' in the User Guide."
        ])

        while True:
            hydro_source = input("\nSelect data source (1/2) [default: 1]: ").strip() or '1'
            if hydro_source in ['1', '2']:
                break
            print("Please enter 1 or 2.")

        if hydro_source == '2':
            # Load from local file
            subsection_header("Load Local Hydrodynamic Data")

            while True:
                local_path = input("\nEnter the path to your hydrodynamic data file (.nc or .zarr): ").strip()
                try:
                    hydro_data = load_hydrodynamic_data(local_path)
                    break
                except (FileNotFoundError, ValueError) as e:
                    print(f"\n  Error: {e}")
                    retry = input("  Try again? (yes/no): ").strip().lower()
                    if retry not in ['yes', 'y']:
                        print("Aborting Step 1.")
                        return

            # Still need traits for the rest of the workflow
            subsection_header("Species Trait Data")
            info_box([
                "Even when loading local hydrodynamic data, you still need species trait data to define the simulation parameters (PLD, spawning windows).",
            ])
            dispersal_time, species_traits_path, traits = load_spp_traits()

            if not dispersal_time:
                print("Error loading species traits. Aborting Step 1.")
                return

            runtime = (datetime.strptime(dispersal_time["overall_dispersal_end"], '%Y-%m-%d') -
                       datetime.strptime(dispersal_time["overall_dispersal_start"], '%Y-%m-%d')).days

            buffer_before = dispersal_time.get("buffer_before", 0)
            buffer_after = dispersal_time.get("buffer_after", 0)

            # Set hydro_params to None since we loaded locally
            hydro_params = {
                'time_range': [
                    datetime.strptime(dispersal_time["overall_dispersal_start"], '%Y-%m-%d'),
                    datetime.strptime(dispersal_time["overall_dispersal_end"], '%Y-%m-%d')
                ],
                'export': False,
                'output_path': local_path
            }

        else:
            # Original path: download from Copernicus
            hydro_params, species_traits_path, runtime, traits = params()

            # Download Hydrodynamic Data
            hydro_data = download_hydrodynamic_data(**hydro_params)

        # Validate hydro data covers simulation runtime
        hydro_time = hydro_data['time'].values
        hydro_start = pd.Timestamp(hydro_time[0])
        hydro_end = pd.Timestamp(hydro_time[-1])

        # Get simulation dates from time_range
        sim_start = pd.Timestamp(hydro_params['time_range'][0])
        sim_end = pd.Timestamp(hydro_params['time_range'][1])

        # Calculate actual biological runtime end (before hydro safety buffer)
        bio_end = sim_start + pd.Timedelta(days=runtime)

        subsection_header("Data Coverage Check")
        print(f"Hydrodynamic data:  {hydro_start.date()} to {hydro_end.date()}")
        print(f"Dispersal period:  {sim_start.date()} to {bio_end.date()} ({runtime} days)")

        # Check coverage
        start_ok = hydro_start <= sim_start
        start_buffer = (sim_start - hydro_start).days
        end_buffer = (hydro_end - bio_end).days

        if not start_ok:
            print(f"\nWarning: The hydrodynamic data starts {abs(start_buffer)} days after simulation start")
        elif end_buffer < 31:
            print(f"\nWarning: End buffer ({end_buffer} days) is less than recommended 31 days")
            print(f"Release interval adjustment may extend runtime beyond hydro data coverage")
            print(f"Consider re-downloading with extended end date")
        else:
            print(
                f"Hydrodynamic data coverage is suitable")

        completion_message(1)
        print("\nOutputs: Species traits, hydrodynamic parameters, dispersal period")
        save_now = input("\nSave session checkpoint? (yes/no) [default: yes]: ").strip().lower()
        if save_now not in ['no', 'n']:
            save_session()
        print("\nTo continue, run in the next cell:")
        print("workflow_step(2)\n")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 2: Species Occurrence Data
    # ═══════════════════════════════════════════════════════════════════════════
    elif step_number == 2:
        step_box(2, "Species Occurrence Data")

        print("\nThis package has the option to use species-specific occurrence data to determine appropriate release and settlement areas for each species.\n")

        print("In this step, you will select your study area and choose your data source.\n")
        print("In step 3, you then have the option to:")
        print("  - Release particles from all locations in your study area or known occurrences")
        print("  - Settle particles from all locations in your study area or known occurrences\n")

        subsection_header("Data source options")
        info_box([
            "Option 1: Download occurrence data from GBIF (Global Biodiversity Information Facility)",
            "  - Automatically determines suitable sites based on species distributions",
            "  - Uses the species identifiers (AphiaID or scientific name) from your trait dataset",
            "  - Same data will be used for both release and settlement",
            "  - Requires a stable internet connection",
            "",
            "Option 2: Provide your own species occurrence data",
            "  - This can be species occurrence data or habitat suitability data",
            "  - Must be in CSV format with columns: species, latitude, longitude",
            "  - Species names must match those in your trait dataset",
            "  - You can provide different datasets for release and settlement in step 3",
            "  - See 'Part 5: Occurrence Data CSV Format' in the User Guide for format details"
        ])

        print("Note: If you plan to release and settle everywhere in your study area, you still need to complete this step to set up the simulation.")

        subsection_header("GBIF Website")
        print("https://www.gbif.org/\n")

        print("For more information, see 'Part 3: Step 2 (Species Occurrence Data)' in the User Guide.\n")

        # Call spp_occ to gather species occurrence data
        section_header("Select your species occurrence data")

        clean_spp, spp_params, landmask, coastalmask, lonarray, latarray = spp_occ(
            hydro_data, species_traits_path
        )

        completion_message(2)
        print("\nOutputs: Cleaned species occurrences, land/coastal masks, study area polygon")
        save_now = input("\nSave session checkpoint? (yes/no) [default: yes]: ").strip().lower()
        if save_now not in ['no', 'n']:
            save_session()
        print("\nTo continue, run in the next cell:")
        print("workflow_step(3)\n")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 3: Simulation Setup
    # ═══════════════════════════════════════════════════════════════════════════

    elif step_number == 3:
        step_box(3, "Simulation Setup")

        subsection_header("Release and Settlement Strategy")
        print("This package has several options for releasing and settling propagules.\n")

        print("You have the option to release propagules from all sites in your study area or from known species-specific distributions.\n")

        print("For more information, see 'Part 3: Step 3 (Release and Settlement Setup)' in the User Guide.\n")

        # Set up release and settlement grids
        meshgrid, release_grids, settlement_grids, full_grid_df, grid_ids, coastal_grid, coastal_grid_df, coastal_grid_filtered, grid_df_for_mapping = setup_release_settlement(
            coastalmask, landmask, clean_spp, spp_params, lonarray, latarray
        )

        # Create Fieldset from hydrodynamic data
        fieldset = create_fieldset(hydro_data, landmask, grid_ids)

        print("Release and settlement grid setup complete.\n")

        subsection_header("ParticleSet Configuration")
        print("The next step is to create a ParticleSet for your community.\n")

        print("A ParticleSet is simply a collection of 'rules' or 'behaviours' applied to")
        print("all particles within your simulation.\n")

        print("This package allows you to choose how many particles to release per site,")
        print("the release frequency and horizontal diffusivity.\n")

        print("For more information, see 'Part 3: Step 3 (Particle Parameters)' in the User Guide.\n")

        # Create ParticleSet
        pset, adjusted_runtime, interval, repeatdt = create_pset(fieldset, meshgrid, runtime)

        # Check if user cancelled
        if pset is None:
            subsection_header("Step 3 cancelled by user.")
            return  # Exit gracefully without error

        # Create custom kernels
        kernels = create_custom_kernels(pset)

        completion_message(3)
        print("\nOutputs: Release/settlement grids, grid IDs, fieldset, particle set")
        save_now = input("\nSave session checkpoint? (yes/no) [default: yes]: ").strip().lower()
        if save_now not in ['no', 'n']:
            save_session()
        print("\nTo continue, run in the next cell:")
        print("workflow_step(4)\n")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 4: Execute Simulation
    # ═══════════════════════════════════════════════════════════════════════════

    elif step_number == 4:
        step_box(4, "Execute the Simulation")

        subsection_header("Running the Particle Simulation")
        print("The simulation will now run using the parameters you have configured.\n")

        print("For more information, see 'Part 3: Step 4 (Running the Simulation)' in the User Guide.\n")

        # Execute simulation
        output_path, traj = execute_simulation(pset, kernels, adjusted_runtime, interval)

        completion_message(4)
        print("\nOutputs: Simulation .zarr file (saved to disk), trajectory DataFrame (in memory)")
        print("Note: The .zarr file is permanently saved on disk. Save your session to preserve the processed trajectory DataFrame and avoid re-loading from .zarr in future.")
        save_now = input("\nSave session checkpoint? (yes/no) [default: yes]: ").strip().lower()
        if save_now not in ['no', 'n']:
            save_session()
        print("\nTo continue, run in the next cell:")
        print("workflow_step(5)\n")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 5: Post-processing
    # ═══════════════════════════════════════════════════════════════════════════

    elif step_number == 5:
        step_box(5, "Post-Processing")

        subsection_header("Species-Specific Filtering")
        print("The simulation output will now be filtered using species-specific traits.\n")

        print("This includes:")
        print("  - Spawning window filtering")
        print("  - Planktonic larval duration (PLD) filtering")
        print("  - Release and settlement habitat suitability filtering\n")

        print("For more information, see 'Part 3: Step 5 (Post-Processing)' in the User Guide.\n")

        # Run post-processing
        final_processed, buffer = post_process(traj, traits, release_grids, settlement_grids, grid_ids)

        completion_message(5)
        print("\nOutputs: Filtered trajectory data by species (in memory)")
        print("Note: Save your session here to re-run analyses (Step 6) without re-processing.")
        save_now = input("\nSave session checkpoint? (yes/no) [default: yes]: ").strip().lower()
        if save_now not in ['no', 'n']:
            save_session()
        print("\nTo continue, run in the next cell:")
        print("workflow_step(6)\n")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 6: Connectivity Analyses
    # ═══════════════════════════════════════════════════════════════════════════

    elif step_number == 6:
        step_box(6, "Connectivity Analyses")

        print("\nThis package includes three types of connectivity analyses:\n")

        subsection_header("Type 1: Connectivity Metrics")
        print("  - Settlement & mortality rates")
        print("  - Local recruitment")
        print("  - Dispersal distance")
        print("  - Realized PLD")

        subsection_header("Type 2: Network Analysis")
        print("  - Functional roles (source, sink, local recruitment)")
        print("  - Centrality measures (betweenness, closeness, degree")
        print("  - Composite hotspot rankings, network maps and connectivity matrices")

        subsection_header("Type 3: Trait Sensitivity Analysis")
        print("  - Compare trait-filtered vs null model connectivity")
        print("  - Compare different trait scenarios (changes in PLD or spawning window)\n")

        print("For more information, see 'Part 3: Step 6 (Connectivity Analysis)' in the User Guide.\n")

        # Run the main analysis function
        analysis_results = post_analysis(final_processed)

        section_header("Interactive workflow completed!")
# endregion
# endregion
