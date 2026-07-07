# region Interactive Functions
"""
ComCon Interactive Mode
=======================

This module provides the interactive workflow for the ComCon package.
Users are guided through each step with prompts, explanations and the ability to review/edit parameters before proceeding.

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
Step 2: Define study extent, download/load species occurrences, filter by habitat
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
import re
import warnings
import traceback
import time
import getpass
import copy
import inspect
import itertools
import textwrap
import webbrowser
from collections import defaultdict
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
from matplotlib.patches import Patch, FancyArrowPatch
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
from shapely.geometry import polygon as shapely_polygon_mod
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import folium
from folium import plugins
import cmocean
import fnmatch
import trajan as ta
import copernicusmarine as cm
from pygbif import occurrences
import pyworms
from tqdm import tqdm
import ipywidgets as widgets
from IPython.display import HTML, Image, display, clear_output
import psutil
import threading
import contextlib
import io
import networkx as nx
try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
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

## Full-width horizontal line (works in Jupyter HTML and terminal)
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

## Centred text that adapts to container width
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

## Level 1 header which is used for workflow step titles, mode titles, start/end messages
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

## Level 2 header which is used for sub-sections within a step
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

## Level 3 info/explanation box which is used for instructions, descriptions, option lists
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
        if not line.strip():
            print()
            continue
        print(line)
    print()
    display_line("─")

## Level 4 data box which is used for results, summaries, validation tables
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

## Code example box which is used in manual mode docs
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

## Status message line with info/success/warning/error symbol
def status_message(message, status="info"):
    """
    Status messages for progress updates.

    status options: "info", "success", "warning", "error"

    Output:
      → Processing data...
      ✓ Data loaded successfully
      ⚠ Warning: Some records outside study extent
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

## Standard step/workflow completion message
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

## Workflow progress header showing tick/arrow status for all 6 steps
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

## Step header box for the start of each workflow step
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

## Automatic DMS to/from Decimal Lat/Lon conversion
def parse_coordinate(value):
    """
    Parse a coordinate value, automatically detecting and converting DMS format.

    Accepts:
    - Decimal degrees: 51.5074, -3.456
    - DMS with symbols: 51°30'26.6"N, 3°27'21.6"W
    - DMS with letters: 51d30m26.6sN, 51 30 26.6 N
    - Degrees and decimal minutes: 51°30.443'N

    Returns float in decimal degrees.
    """
    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).strip()

    # Try simple float first
    try:
        return float(value)
    except ValueError:
        pass

    # DMS pattern: degrees, minutes, seconds, direction
    # Matches: 51°30'26.6"N, 51d30m26.6sN, 51 30 26.6 N, etc.
    dms_pattern = r"""
        ^\s*(-?)                          # Optional negative sign
        (\d+)\s*[°dD\s]\s*               # Degrees
        (\d+)\s*['\u2032mM\s]\s*         # Minutes  
        (\d+\.?\d*)\s*["\u2033sS]?\s*    # Seconds (optional symbol)
        ([NSEWnsew]?)\s*$                 # Optional direction
    """

    match = re.match(dms_pattern, value, re.VERBOSE)
    if match:
        sign, deg, mins, secs, direction = match.groups()
        decimal = float(deg) + float(mins) / 60 + float(secs) / 3600
        if sign == '-' or direction.upper() in ('S', 'W'):
            decimal = -decimal
        return decimal

    # Degrees and decimal minutes: 51°30.443'N
    dm_pattern = r"""
        ^\s*(-?)                          # Optional negative sign
        (\d+)\s*[°dD\s]\s*               # Degrees
        (\d+\.?\d*)\s*['\u2032mM]?\s*    # Decimal minutes
        ([NSEWnsew]?)\s*$                 # Optional direction
    """

    match = re.match(dm_pattern, value, re.VERBOSE)
    if match:
        sign, deg, mins, direction = match.groups()
        decimal = float(deg) + float(mins) / 60
        if sign == '-' or direction.upper() in ('S', 'W'):
            decimal = -decimal
        return decimal

    raise ValueError(f"Cannot parse coordinate: '{value}'")

## Automatic DMS to/from decimal lat/lon conversion across a DataFrame
def auto_convert_coordinates(df, lat_col='latitude', lon_col='longitude'):
    """
    Automatically detect and convert DMS coordinates to decimal degrees.

    If coordinates are already decimal, returns unchanged.
    If DMS detected, converts and adds original columns as backup.

    Parameters
    ----------
    df : DataFrame
        Data with coordinate columns
    lat_col, lon_col : str
        Column names for latitude and longitude

    Returns
    -------
    DataFrame with decimal degree coordinates
    """
    converted = False

    for col in [lat_col, lon_col]:
        if col not in df.columns:
            continue

        # Check if already numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        # Try converting , if any value is non-numeric string, attempt DMS parsing
        try:
            original_col = f'{col}_original'
            df[original_col] = df[col].copy()
            df[col] = df[col].apply(parse_coordinate)
            converted = True
            print(f"  Converted {col} from DMS to decimal degrees.")
        except Exception as e:
            print(f"  Warning: Could not auto-convert {col}: {e}")
            if f'{col}_original' in df.columns:
                df[col] = df[f'{col}_original']
                df.drop(columns=[f'{col}_original'], inplace=True)

    if converted:
        status_message("Coordinates automatically converted from DMS to decimal degrees.", "success")

    return df

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

## Allows users to validate their file path inputs incase of typos
def get_valid_path(prompt, must_exist=True, default=None, extensions=None):
    """Get a validated file path from user input. Auto-appends extension if needed.
    Press Enter with no input to open a file browser (if available)."""

    # Build file type filter for tkinter
    ## Build the filetypes list for the tkinter file dialog
    def _get_filetypes(extensions):
        """Return a tkinter filetypes list for the given list of file extensions."""
        if not extensions:
            return [("All files", "*.*")]
        ext_str = " ".join(f"*{e}" for e in extensions)
        return [(f"Supported files ({ext_str})", ext_str), ("All files", "*.*")]

    while True:
        browse_hint = " (or press Enter to browse)" if must_exist and not default else ""
        if default:
            user_input = input(f"{prompt} [default: {default}]{browse_hint}: ").strip() or default
        else:
            user_input = input(f"{prompt}{browse_hint}: ").strip()
        if not user_input and must_exist and not default:
            # Try file browser
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True)
                filepath = filedialog.askopenfilename(
                    title=prompt,
                    filetypes=_get_filetypes(extensions)
                )
                root.destroy()
                if filepath:
                    print(f"  Selected: {filepath}")
                    user_input = filepath
                else:
                    print("  No file selected. Please enter a path or try again.")
                    continue
            except (ImportError, Exception):
                print("  File browser not available. Please enter the file path manually.")
                continue
        user_input = os.path.expanduser(user_input)
        if must_exist:
            if os.path.exists(user_input):
                if extensions:
                    ext = os.path.splitext(user_input)[1].lower()
                    if ext and ext not in extensions:
                        print(f"  Expected file type: {', '.join(extensions)}, got: {ext}")
                        retry = get_yes_no("  Continue anyway?", default='no')
                        if not retry:
                            continue
                return user_input
            if not os.path.splitext(user_input)[1]:
                if extensions:
                    for ext in extensions:
                        test_path = user_input + ext
                        if os.path.exists(test_path):
                            print(f"  Found: {test_path}")
                            return test_path
                for ext in ['.csv', '.xlsx', '.nc', '.geojson', '.json']:
                    test_path = user_input + ext
                    if os.path.exists(test_path):
                        print(f"  Found: {test_path}")
                        return test_path
            print(f"  Path not found: {user_input}")
            if extensions and not os.path.splitext(user_input)[1]:
                print(f"  Also tried: {', '.join(user_input + e for e in extensions)}")
            print(f"  Please check the path and try again.")
            continue
        if extensions:
            ext = os.path.splitext(user_input)[1].lower()
            if ext and ext not in extensions:
                print(f"  Expected file type: {', '.join(extensions)}, got: {ext}")
                retry = get_yes_no("  Continue anyway?", default='no')
                if not retry:
                    continue
        return user_input

## Get a directory path from the user, opening a folder browser if they press Enter
def get_valid_dir(prompt, default=None):
    """Get a directory path from user input, opening a folder browser on Enter.

    Parameters
    ----------
    prompt : str
        Prompt text shown to the user.
    default : str, optional
        Default directory returned if the user presses Enter with a typed path omitted.

    Returns
    -------
    str
        Validated or user-supplied directory path (created if it doesn't exist).
    """
    browse_hint = " (or press Enter to browse)"
    if default:
        typed = input(f"{prompt} [default: {default}] (press Enter for default, type b to browse): ").strip()
        if typed == '':
            # Enter accepts the default. No browser is opened.
            return default
        if typed.lower() in ('b', 'browse'):
            # Only open the file browser when the user explicitly asks for it.
            try:
                import tkinter as tk
                from tkinter import filedialog
                _root = tk.Tk()
                _root.withdraw()
                _root.attributes('-topmost', True)
                chosen = filedialog.askdirectory(title=prompt)
                _root.destroy()
                if chosen:
                    print(f"  Selected: {chosen}")
                    return chosen
            except Exception:
                pass
            return default
        return os.path.expanduser(typed)
    else:
        typed = input(f"{prompt}{browse_hint}: ").strip()
        if typed:
            return os.path.expanduser(typed)
        try:
            import tkinter as tk
            from tkinter import filedialog
            _root = tk.Tk()
            _root.withdraw()
            _root.attributes('-topmost', True)
            chosen = filedialog.askdirectory(title=prompt)
            _root.destroy()
            if chosen:
                print(f"  Selected: {chosen}")
                return chosen
        except Exception:
            pass
        return typed  # empty string , caller should handle


## Get a save file path from the user, opening a save dialog if they press Enter
def get_save_path(prompt, default=None, extensions=None):
    """Get a save file path from user input, opening a save-file dialog on Enter.

    Parameters
    ----------
    prompt : str
        Prompt text shown to the user.
    default : str, optional
        Default path returned when the user presses Enter without typing.
    extensions : list of str, optional
        Allowed file extensions, e.g. ['.csv', '.nc'].  The first entry becomes
        the default extension in the save dialog.

    Returns
    -------
    str
        The path the user chose or typed.
    """
    browse_hint = " (or press Enter to browse)"
    if default:
        typed = input(f"{prompt} [default: {default}] (press Enter for default, type b to browse): ").strip()
        if typed == '':
            # Enter accepts the default. No browser is opened.
            return default
        if typed.lower() in ('b', 'browse'):
            # Only open the save dialog when the user explicitly asks for it.
            try:
                import tkinter as tk
                from tkinter import filedialog
                _root = tk.Tk()
                _root.withdraw()
                _root.attributes('-topmost', True)
                ft = [("All files", "*.*")]
                if extensions:
                    ext_str = " ".join(f"*{e}" for e in extensions)
                    ft = [(f"Supported files ({ext_str})", ext_str)] + ft
                defext = extensions[0].lstrip('.') if extensions else ''
                chosen = filedialog.asksaveasfilename(
                    title=prompt,
                    defaultextension=defext,
                    filetypes=ft,
                )
                _root.destroy()
                if chosen:
                    print(f"  Selected: {chosen}")
                    return chosen
            except Exception:
                pass
            return default
        return os.path.expanduser(typed)
    else:
        typed = input(f"{prompt}{browse_hint}: ").strip()
        if typed:
            return os.path.expanduser(typed)
        try:
            import tkinter as tk
            from tkinter import filedialog
            _root = tk.Tk()
            _root.withdraw()
            _root.attributes('-topmost', True)
            ft = [("All files", "*.*")]
            if extensions:
                ext_str = " ".join(f"*{e}" for e in extensions)
                ft = [(f"Supported files ({ext_str})", ext_str)] + ft
            defext = extensions[0].lstrip('.') if extensions else ''
            chosen = filedialog.asksaveasfilename(
                title=prompt,
                defaultextension=defext,
                filetypes=ft,
            )
            _root.destroy()
            if chosen:
                print(f"  Selected: {chosen}")
                return chosen
        except Exception:
            pass
        return typed  # empty string , caller should handle


## Auto scales the grid resolution used for release and settlement (not the hydro grid resolution used for the simulation)
def calculate_resolution(lat_extent, lon_extent):
    """
    Calculate recommended grid resolution (in km) based on domain size.

    Domain resolution for release and settlement grid is auto-scaled to keep total particle count within reasonable range (~400,000 particles per season).
    This will keep simulation runtime on a standard laptop within ~30-60 minutes.
    This follows the approach used by James et al. (2025) for basin-scale analyses.

    Parameters
    ----------
    lat_extent : float
        Latitudinal extent of the domain in degrees (lat_max - lat_min).
    lon_extent : float
        Longitudinal extent of the domain in degrees (lon_max - lon_min).

    Returns
    -------
    int
        Recommended grid resolution in kilometres.
        Possible values: 9, 15, 25 or 30.

    Notes
    -----
    The resolution lookup table is based on domain area in square degrees:
        <25 sq deg   -> 9 km
        25-100       -> 15 km
        100-400      -> 25 km
        >400         -> 30 km

    Boundary cases use strict less-than, e.g. a 25 sq deg domain gets 15 km.
    These values are an analysis-grid recommendation chosen to keep the particle
    count manageable, not the dataset's native resolution. The caller only bins
    when this value is coarser than the native grid, so on a fine-resolution
    dataset (e.g. the ~1.5 km NW European Shelf) a small domain that returns 9 km
    is still binned down from native; select Native or Custom to keep finer cells.
    """
    area_sqdeg = float(lat_extent) * float(lon_extent)

    if area_sqdeg < 25:
        return 9
    elif area_sqdeg < 100:
        return 15
    elif area_sqdeg < 400:
        return 25
    else:
        return 30

## Grid resolution adjustment
def bin_mask(native_landmask, native_lonarray, native_latarray, target_km):
    """
    Bin the native landmask to a coarser resolution using majority vote.

    For each coarse cell, counts native land cells. Classified as land if >=50% are land.
    Used by grid auto-scaling to reduce site count for large domains.

    Parameters
    ----------
    native_landmask : numpy.ndarray
        2D binary array at native resolution (1=land, 0=ocean).
    native_lonarray, native_latarray : numpy.ndarray
        2D meshgrids of coordinates at native resolution.
    target_km : int
        Target resolution in kilometres.

    Returns
    -------
    binned_landmask, binned_lonarray, binned_latarray : numpy.ndarray
        Binned arrays at target resolution.
    """
    native_lat_step = abs(float(native_latarray[1, 0] - native_latarray[0, 0]))
    native_lon_step = abs(float(native_lonarray[0, 1] - native_lonarray[0, 0]))

    lat_mid = float(np.mean(native_latarray))
    target_lat_step = target_km / 111.0
    target_lon_step = target_km / (111.0 * max(np.cos(np.deg2rad(lat_mid)), 0.1))

    # If target is within 10% of native, return native unchanged
    native_km_lat = native_lat_step * 111.0
    if abs(target_km - native_km_lat) / native_km_lat < 0.10:
        return native_landmask.copy(), native_lonarray.copy(), native_latarray.copy()

    lat_min = float(np.min(native_latarray))
    lat_max = float(np.max(native_latarray))
    lon_min = float(np.min(native_lonarray))
    lon_max = float(np.max(native_lonarray))

    coarse_lat_edges = np.arange(lat_min, lat_max + target_lat_step, target_lat_step)
    coarse_lon_edges = np.arange(lon_min, lon_max + target_lon_step, target_lon_step)

    if len(coarse_lat_edges) < 2 or len(coarse_lon_edges) < 2:
        raise ValueError(
            f"Target resolution ({target_km} km) is too coarse for the domain extent. "
            "Try a smaller target resolution or a larger domain."
        )

    n_coarse_lat = len(coarse_lat_edges) - 1
    n_coarse_lon = len(coarse_lon_edges) - 1

    binned_landmask = np.ones((n_coarse_lat, n_coarse_lon), dtype=np.int32)
    binned_lat_centroids = np.zeros(n_coarse_lat)
    binned_lon_centroids = np.zeros(n_coarse_lon)

    # Assign each native cell to a coarse cell by its centroid (np.digitize), then majority-vote.
    # Identical to passive.bin_mask, so every pathway shares one binning algorithm and no
    # assumption is made about grid regularity.
    native_lon_flat = native_lonarray.ravel()
    native_lat_flat = native_latarray.ravel()
    native_land_flat = native_landmask.ravel()

    lat_bin_idx = np.digitize(native_lat_flat, coarse_lat_edges) - 1
    lon_bin_idx = np.digitize(native_lon_flat, coarse_lon_edges) - 1
    lat_bin_idx = np.clip(lat_bin_idx, 0, n_coarse_lat - 1)
    lon_bin_idx = np.clip(lon_bin_idx, 0, n_coarse_lon - 1)

    for i in range(n_coarse_lat):
        binned_lat_centroids[i] = (coarse_lat_edges[i] + coarse_lat_edges[i + 1]) / 2.0
        for j in range(n_coarse_lon):
            mask = (lat_bin_idx == i) & (lon_bin_idx == j)
            native_cells_in_bin = native_land_flat[mask]
            if len(native_cells_in_bin) == 0:
                # Empty edge cell (no native coverage): mark ocean, NOT land, so it doesn't
                # become phantom coast via dilation.
                binned_landmask[i, j] = 0
            else:
                frac_land = np.mean(native_cells_in_bin)
                binned_landmask[i, j] = 1 if frac_land >= 0.5 else 0

    for j in range(n_coarse_lon):
        binned_lon_centroids[j] = (coarse_lon_edges[j] + coarse_lon_edges[j + 1]) / 2.0

    binned_lonarray, binned_latarray = np.meshgrid(binned_lon_centroids, binned_lat_centroids)
    return binned_landmask, binned_lonarray, binned_latarray


## Map each native-resolution cell to its corresponding binned coastal grid_id
def create_native_grid_ids_from_binned(native_lonarray, native_latarray, native_landmask,
                                        binned_coastalmask, binned_lonarray, binned_latarray,
                                        binned_grid_df):
    """
    Create a native-resolution grid_ids array that maps each native cell to its corresponding binned coastal grid cell.

    This allows the OceanParcels fieldset (at native resolution) to assign binned grid_ids to particles during simulation.

    Parameters
    ----------
    native_lonarray, native_latarray : numpy.ndarray
        2D meshgrids at native hydro resolution.
    native_landmask : numpy.ndarray
        2D binary mask at native resolution.
    binned_coastalmask : numpy.ndarray
        2D binary coastal mask at binned resolution.
    binned_lonarray, binned_latarray : numpy.ndarray
        2D meshgrids at binned resolution.
    binned_grid_df : pandas.DataFrame
        DataFrame of binned coastal cells with grid_id, longitude, latitude.

    Returns
    -------
    grid_ids_native : numpy.ndarray
        2D array at native resolution where each cell holds the grid_id of the nearest binned coastal cell, or -1 if the cell is land.
    """
    # Get binned cell centroids
    binned_lats_1d = binned_latarray[:, 0]
    binned_lons_1d = binned_lonarray[0, :]

    # For each native cell, find which binned cell it falls in
    native_lats_1d = native_latarray[:, 0]
    native_lons_1d = native_lonarray[0, :]

    # Map native rows/cols to binned rows/cols using nearest neighbour
    row_idx = np.searchsorted(binned_lats_1d, native_lats_1d, side='right') - 1
    col_idx = np.searchsorted(binned_lons_1d, native_lons_1d, side='right') - 1

    row_idx = np.clip(row_idx, 0, len(binned_lats_1d) - 1)
    col_idx = np.clip(col_idx, 0, len(binned_lons_1d) - 1)

    # Build lookup: binned (row, col) -> grid_id
    n_blat, n_blon = binned_coastalmask.shape
    lookup = np.full((n_blat, n_blon), -1, dtype=np.int64)
    if 'row_idx' in binned_grid_df.columns and 'col_idx' in binned_grid_df.columns:
        for _, r in binned_grid_df.iterrows():
            lookup[int(r['row_idx']), int(r['col_idx'])] = int(r['grid_id'])
    else:
        # Fallback: use position matching
        for _, r in binned_grid_df.iterrows():
            bi = np.argmin(np.abs(binned_lats_1d - r['latitude']))
            bj = np.argmin(np.abs(binned_lons_1d - r['longitude']))
            lookup[bi, bj] = int(r['grid_id'])

    # Create native-resolution grid_ids
    grid_ids_native = np.full(native_landmask.shape, -1, dtype=np.int64)
    for ni in range(len(native_lats_1d)):
        bi = row_idx[ni]
        for nj in range(len(native_lons_1d)):
            bj = col_idx[nj]
            grid_ids_native[ni, nj] = lookup[bi, bj]

    return grid_ids_native


## Prompt the user to select a grid resolution for release/settlement sites
def choose_grid_resolution(lonarray, latarray):
    """
    Present grid resolution options to the user for the trait-defined pathway.

    Calculates the recommended resolution based on domain extent and offers
    three choices: recommended, custom, or native (with memory warning).

    Parameters
    ----------
    lonarray, latarray : numpy.ndarray
        2D meshgrids from the hydrodynamic data.

    Returns
    -------
    resolution_km : int
        Selected resolution in km. 0 means 'use native' (no binning).
    needs_binning : bool
        True if binning is required.
    """
    lat_extent = float(latarray.max() - latarray.min())
    lon_extent = float(lonarray.max() - lonarray.min())
    recommended_km = calculate_resolution(lat_extent, lon_extent)

    # Calculate native resolution
    native_lat_step = abs(float(latarray[1, 0] - latarray[0, 0]))
    native_km = round(native_lat_step * 111.0, 1)

    subsection_header("Grid Resolution for Release and Settlement")

    info_box([
        "The grid resolution only controls the resolution of the analysis (release and settlement sites).",
        "Particle advection always runs on the full native-resolution hydrodynamic grid.",
        "",
        "A coarser grid means fewer sites, fewer particles and faster simulations, but less spatial detail in the connectivity results.",
        "",
        "For a site-based analysis on specific study sites, the native (finest) resolution is typically the best choice as there will be less release sites compared with a full-domain run and your sites will remain more delineated. The coarser recommended resolution is aimed at whole-domain runs.",
        "",
        f"Native hydrodynamic resolution: ~{native_km} km",
        "",
        "Whether your release and settlement sites are drawn from the coastal cells or include open-ocean cells is set later in the workflow, not here.",
        "",
        f"Recommended release/settlement resolution: {recommended_km} km",
    ])

    if recommended_km <= native_km * 1.1:
        # Small domain: native IS the recommended resolution , show 2 options
        print(f"\n  Options:")
        print(f"    1. Native resolution (~{native_km} km)  [recommended for your domain size]")
        print(f"    2. Choose your own resolution")

        choice = input(f"\n  Select option (1/2) [default: 1]: ").strip() or '1'

        if choice == '2':
            while True:
                try:
                    raw = input(f"  Enter resolution in km (≥ {int(native_km)}) [default: {int(native_km)}]: ").strip()
                    custom = int(raw) if raw else int(native_km)
                except ValueError:
                    print(f"  Please enter a whole number ≥ {int(native_km)}.")
                    continue
                if custom >= native_km:
                    print(f"\n  Using custom resolution: {custom} km")
                    return custom, custom > native_km * 1.1
                print(f"  Please enter a value ≥ {int(native_km)}.")
        else:
            print(f"\n  Using native resolution: ~{native_km} km")
            return int(native_km), False

    else:
        # Large domain: native and recommended differ , show all 3
        print(f"\n  Options:")
        print(f"    1. Native resolution (~{native_km} km)")
        print(f"    2. Recommended resolution ({recommended_km} km)  [default]")
        print(f"    3. Choose your own resolution")

        choice = input(f"\n  Select option (1/2/3) [default: 2]: ").strip() or '2'

        if choice == '1':
            print(f"\n  Using native resolution: ~{native_km} km")
            return int(native_km), False

        elif choice == '2':
            print(f"\n  Using recommended resolution: {recommended_km} km")
            return recommended_km, True

        else:
            while True:
                try:
                    raw = input(
                        f"  Enter resolution in km (between {int(native_km)} and 50) [default: {recommended_km}]: ").strip()
                    custom = int(raw) if raw else recommended_km
                except ValueError:
                    print(f"  Please enter a whole number.")
                    continue
                if native_km <= custom <= 50:
                    print(f"\n  Using custom resolution: {custom} km")
                    return custom, custom > native_km * 1.1
                print(f"  Please enter a value between {int(native_km)} and 50.")

## Retrieves the copericus credentials
def get_copernicus_credentials():
    """
    Retrieve Copernicus Marine Service credentials using a saved credentials file if available
    or prompting the user interactively on first use.

    Returns
    -------
    username : str or None
        Copernicus username. None if saved credentials are used (in which case copernicusmarine will pick them up from the credentials file automatically)
    password : str or None
        Copernicus password. None if saved credentials are used.

    Notes
    -----
    Checks for a saved credentials file at
    ``~/.copernicusmarine/.copernicusmarine-credentials``.
    If found, returns (None, None) so that copernicusmarine falls back to the saved credentials.
    If not found, calls ``cm.login()`` which prompts the user interactively and saves the credentials for future use.
    Falls back to manual input() prompts if ``cm.login()`` fails for any reason.
    """
    credentials_file = os.path.expanduser("~/.copernicusmarine/.copernicusmarine-credentials")
    if os.path.exists(credentials_file):
        print("  Using saved Copernicus Marine credentials.")
        print("  (To change credentials, run: copernicusmarine login)")
        return None, None
    else:
        print("\n  First-time setup: Your credentials will be saved for future sessions.")
        print("  You won't need to enter them again.\n")
        try:
            cm.login()
            print("\n  Credentials saved successfully.")
            return None, None
        except Exception as e:
            print(f"\n  Automatic login failed: {e}")
            print("  Falling back to manual credentials.\n")
            username = input("Enter your username: ").strip()
            password = getpass.getpass("Enter your password: ").strip()
            return username, password

## Allows users to easily stop mid-workflow once completing a step and jump back in
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

    # Large DataFrames saved as parquet separately (too big for pickle)
    large_df_vars = ['traj', 'final_processed']

    # Everything else saved via pickle
    variables_to_save = [
        'project_dirs', 'runtime', 'species_traits_path', 'traits', 'hydro_params',
        '_selected_resolution_km', '_needs_binning',
        'buffer_before', 'buffer_after',
        'clean_spp', 'spp_params', 'landmask', 'coastalmask', 'lonarray', 'latarray',
        'release_grids', 'settlement_grids', 'full_grid_df', 'grid_ids',
        'coastal_grid', 'coastal_grid_df', 'coastal_grid_filtered',
        'ocean_grid', 'ocean_grid_df', 'grid_df_for_mapping', 'meshgrid',
        'adjusted_runtime', 'interval', 'traj', 'output_path', 'chunks_dir',
        'buffer', 'geojson_file_path'
    ]

    for var_name in variables_to_save:
        if var_name in globals() and globals()[var_name] is not None:
            try:
                session_data[var_name] = globals()[var_name]
            except Exception as e:
                print(f"  Warning: Could not save '{var_name}': {e}")

    # Save hydro_data: strip the unpicklable Copernicus/S3 backend by converting to a plain in-memory dataset.
    # Cache the clean copy so subsequent saves are instant.
    # Once the simulation has produced trajectory chunks, the velocity fields are no longer needed.
    if 'chunks_dir' in globals() and globals().get('chunks_dir') and os.path.isdir(
            str(globals()['chunks_dir'])):
        print(
            "\n  Simulation already complete, so the hydrodynamic velocities are not needed to resume and won't be saved (this keeps the checkpoint fast).")
    elif 'hydro_data' in globals() and globals()['hydro_data'] is not None:
        try:
            _hd = globals()['hydro_data']
            if _hd.attrs.get('_comcon_cleaned'):
                # Already cleaned on a previous save , reuse directly
                session_data['hydro_data'] = _hd
            else:
                print("  Preparing hydrodynamic data for session save. This may take a moment...")
                _needed_vars = [v for v in ['uo', 'vo'] if v in _hd.data_vars]
                _hd_clean = xr.Dataset(
                    {var: (list(_hd[var].dims), _hd[var].values) for var in _needed_vars},
                    coords={coord: _hd[coord].values for coord in _hd.coords}
                )
                _hd_clean.attrs['_comcon_cleaned'] = True
                globals()['hydro_data'] = _hd_clean
                session_data['hydro_data'] = _hd_clean
        except Exception as e:
            print(f"  Note: Could not save hydro_data ({e}). Step 1 will need to be re-run after restart.")

    # Save large DataFrames as parquet files alongside the pickle
    parquet_dir = os.path.dirname(os.path.abspath(filepath))
    saved_parquets = {}

    for var_name in large_df_vars:
        if var_name in globals() and globals()[var_name] is not None:
            var_val = globals()[var_name]
            try:
                if isinstance(var_val, pd.DataFrame):
                    pq_path = os.path.join(parquet_dir, f'comcon_session_{var_name}.parquet')
                    var_val.to_parquet(pq_path, index=False, engine='pyarrow')
                    saved_parquets[var_name] = pq_path
                    print(f"  Saved {var_name} to parquet ({os.path.getsize(pq_path) / (1024 ** 2):.1f} MB)")
                elif isinstance(var_val, dict):
                    # final_processed is a dict of DataFrames
                    dict_paths = {}
                    for key, df in var_val.items():
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            pq_path = os.path.join(parquet_dir, f'comcon_session_{var_name}_{key}.parquet')
                            df.to_parquet(pq_path, index=False, engine='pyarrow')
                            dict_paths[key] = pq_path
                    if dict_paths:
                        saved_parquets[var_name] = dict_paths
                        print(f"  Saved {var_name} ({len(dict_paths)} species) to parquet files")
            except Exception as e:
                print(f"  Warning: Could not save '{var_name}' to parquet: {e}")

    # Store parquet paths in session data so load_session knows where to find them
    session_data['_parquet_files'] = saved_parquets

    session_data['_saved_at'] = datetime.now().isoformat()

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(session_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    n_saved = len(session_data) - 2 + len(saved_parquets)
    display_line("─")
    print(f"  Session saved to: {os.path.abspath(filepath)}")
    print(f"  Variables saved: {n_saved}")
    print(f"  Pickle size: {os.path.getsize(filepath) / (1024 * 1024):.1f} MB")
    if saved_parquets:
        print(f"  Large variables saved as parquet: {', '.join(saved_parquets.keys())}")
    display_line("─")
    # Determine which step was last completed based on saved variables
    if 'final_processed' in session_data or '_parquet_files' in session_data and 'final_processed' in session_data.get(
            '_parquet_files', {}):
        last_step = 5
    elif 'output_path' in session_data and session_data.get('output_path'):
        last_step = 4
    elif 'release_grids' in session_data:
        last_step = 3
    elif 'clean_spp' in session_data:
        last_step = 2
    elif 'hydro_params' in session_data:
        last_step = 1
    else:
        last_step = 0

    next_step = min(last_step + 1, 6)

    print(f"\nSession saved after Step {last_step}.")
    print(f"\nTo resume later, after restarting the kernel or in a new notebook, run in a new cell:")
    print(f"from comcon.interactive import *")
    print(f"load_session('{filepath}')")
    print(f"workflow_step({next_step})")

## Load a previously saved workflow session and restore all variables to the module namespace
def load_session(filepath=None):
    """
    Load a previously saved workflow session.

    Restores all variables so you can resume from any workflow step without re-running previous steps.

    Parameters
    ----------
    filepath : str, optional
        Path to the saved session file. If omitted, a file picker opens so you can select the .pkl file.
    """
    if filepath is None:
        try:
            import tkinter as tk
            from tkinter import filedialog
            _root = tk.Tk()
            _root.withdraw()
            _root.attributes('-topmost', True)
            filepath = filedialog.askopenfilename(
                title="Select a ComCon session file",
                filetypes=[("ComCon session", "*.pkl"), ("All files", "*.*")]
            )
            _root.destroy()
        except Exception:
            filepath = input("Enter the path to a saved session (.pkl) file: ").strip()

        if not filepath:
            print("  No session file selected.")
            return

    if not os.path.exists(filepath):
        print(f"  Error: Session file not found: {filepath}")
        return

    with open(filepath, 'rb') as f:
        session_data = pickle.load(f)

    saved_at = session_data.pop('_saved_at', 'Unknown')
    parquet_files = session_data.pop('_parquet_files', {})

    restored = []

    # Restore pickle variables to module globals
    for var_name, value in session_data.items():
        globals()[var_name] = value
        restored.append(var_name)

    # Restore parquet files (traj, final_processed)
    for var_name, pq_info in parquet_files.items():
        try:
            if isinstance(pq_info, str):
                # Single DataFrame (e.g. traj)
                if os.path.exists(pq_info):
                    globals()[var_name] = pd.read_parquet(pq_info, engine='pyarrow')
                    restored.append(var_name)
                    print(f"  Restored {var_name} from parquet ({len(globals()[var_name]):,} rows)")
                else:
                    print(f"  Warning: Parquet file not found for '{var_name}': {pq_info}")
            elif isinstance(pq_info, dict):
                # Dict of DataFrames (e.g. final_processed)
                restored_dict = {}
                for key, pq_path in pq_info.items():
                    if os.path.exists(pq_path):
                        restored_dict[key] = pd.read_parquet(pq_path, engine='pyarrow')
                if restored_dict:
                    globals()[var_name] = restored_dict
                    restored.append(var_name)
                    print(f"  Restored {var_name} ({len(restored_dict)} species) from parquet")
        except Exception as e:
            print(f"  Warning: Could not restore '{var_name}' from parquet: {e}")

        # Validate chunks_dir if present (verify chunk files still exist on disk)
        if 'chunks_dir' in globals() and globals()['chunks_dir']:
            cd = globals()['chunks_dir']
            if os.path.isdir(cd):
                n_chunks = len(glob(os.path.join(cd, 'chunk_*.parquet')))
                if n_chunks > 0:
                    print(f"  Trajectory data: {n_chunks} chunks on disk at {cd}")
                    restored.append('chunks_dir')
                else:
                    print(f"  Warning: chunks_dir exists but contains no chunk files: {cd}")
                    globals()['chunks_dir'] = None
            else:
                print(f"  Warning: chunks_dir not found on disk: {cd}")
                print(f"  You may need to re-run Step 4, or re-process from the .zarr file.")
                globals()['chunks_dir'] = None

        # Also inject into caller's namespace (so variables are accessible in notebook cells)
        caller_globals = inspect.currentframe().f_back.f_globals
        for var_name in restored:
            caller_globals[var_name] = globals()[var_name]

    display_line("─")
    print(f"Session restored from: {filepath}")
    print(f"Originally saved at: {saved_at}")
    print(f"Variables restored: {len(restored)}")
    display_line("─")

    can_resume = []
    if 'traits' in restored:
        can_resume.append("Step 2 (Species Occurrence Data)")
    if 'clean_spp' in restored:
        can_resume.append("Step 3 (Simulation Setup)")
    if 'grid_ids' in restored:
        can_resume.append("Step 4 (Simulation)")
    if 'chunks_dir' in restored or 'traj' in restored:
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
    """Format species name for display: underscores to spaces and capitalise the first
    letter (genus), leaving the rest untouched so a binomial reads 'Carcinus maenas'."""
    if name is None:
        return None
    s = str(name).replace('_', ' ')
    return s[:1].upper() + s[1:] if s else s

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
def save_figure(fig, base_path, format_ext='tif', dpi=300):
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
        output_dir = get_valid_dir("Output directory", default=default_dir)
        os.makedirs(output_dir, exist_ok=True)
        output_format, dpi = get_output_format()

    return display_fig, save_fig, output_dir, output_format, dpi

## Dynamic sizing for figures
def calculate_figure_dimensions(lon_range, lat_range, base_size=10,
                                min_dim=6, max_dim=16, n_panels=1,
                                layout='horizontal'):
    """
    Calculate optimal figure dimensions based on spatial extent.

    Handles different study extent shapes (tall/narrow vs wide/short)
    and multi-panel layouts intelligently.

    Parameters
    ----------
    lon_range : float
        Longitude range of study extent (degrees).
    lat_range : float
        Latitude range of study extent (degrees).
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
        if aspect > 1.5:  # Wide study extent
            actual_layout = 'vertical'  # Stack maps vertically
        elif aspect < 0.67:  # Tall study extent
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
            print(f"Built-in trait database not found at {trait_path}")
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

    while True:
        choice = input("\nSelect option (1/2) [default: 1]: ").strip() or '1'
        if choice in ['1', '2']:
            break
        print("  Invalid input. Please enter 1 or 2.")

    if choice == '2':
        return traits_df
    else:
        all_indices = ','.join(str(i) for i in range(1, len(traits_df) + 1))
        indices_str = input(f"Enter species numbers (comma-separated, e.g., 1,3,5) [default: all]: ").strip() or all_indices
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

        # Warn when the dispersal season crosses a calendar-year boundary.
        if base_start.year != base_end.year:
            print(f"\nNote: Your spawning windows span two calendar years ({base_start.year} to {base_end.year}).")
            print(
                f"The simulation runs from {base_start:%Y-%m-%d} to {base_end:%Y-%m-%d}, so make sure your hydrodynamic data covers that whole period.")

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
    subsection_header("Species Trait Data")

    print("\nHow would you like to provide species traits data?")
    print("  1. Select species from built-in trait database")
    print("  2. Provide your own CSV file")

    source_choice = input("\nSelect option (1-2) [default: 1]: ").strip() or '1'

    traits = []
    species_traits_path = None

    # Option 1: Use built-in trait database
    if source_choice == '1':
        subsection_header("Built-in Trait Database")

        # Load the database
        traits_df = load_builtin_traits()

        if traits_df is None or traits_df.empty:
            print("\nError: Could not load built-in trait database.")
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

            start_year = int(
                input(f"\nStart year for dispersal season [default: {datetime.now().year - 1}]: ").strip() or str(datetime.now().year - 1))

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
            output_dir = get_valid_dir("Output directory", default=default_dir)
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
            "Your CSV needs one row per species, with these columns.",
            "",
            "  Required:",
            "    species_name OR aphia_id: at least one species identifier (see below)",
            "    spawning_start:            start of spawning window (DD/MM or DD/MM/YYYY, e.g. 01/10)",
            "    spawning_end:              end of spawning window (DD/MM or DD/MM/YYYY, e.g. 31/01)",
            "    min_PLD:                   minimum pelagic larval duration in whole days",
            "    max_PLD:                   maximum pelagic larval duration in whole days",
            "",
            "  Species identifier:",
            "    You must provide a species_name, aphia_id or both. If both are given, aphia_id is used.",
            "    AphiaID is recommended for marine species because it tracks name changes and synonyms.",
            "    If you provide only an aphia_id, ComCon looks up the species name online, so a stable internet connection is required.",
            "    Find AphiaIDs at https://www.marinespecies.org/",
            "",
            "  Notes:",
            "    Column order does not matter and any extra columns are ignored.",
            "    If a date has no year you will be asked which dispersal year to use. If DD/MM/YYYY is provided, this year is used.",
            "    PLD values should be given as integers.",
            "    Species occurrence/distribution data is uploaded separately in Step 2.",
        ])
        species_traits_path = get_valid_path("Enter the path to your species traits CSV", must_exist=True,
                                             extensions=['.csv'])

        # Read CSV file to validate columns
        # Uses sep=None with python engine to auto-detect delimiter (handles semicolons from international Excel versions) and encoding='utf-8-sig' to handle BOM characters
        try:
            traits_df = pd.read_csv(species_traits_path, sep=None, engine='python',
                                    encoding='utf-8-sig')
            # Strip whitespace from column names (handles stray spaces)
            traits_df.columns = traits_df.columns.str.strip()
            # Auto-convert DMS coordinates if present
            if 'latitude' in traits_df.columns and 'longitude' in traits_df.columns:
                traits_df = auto_convert_coordinates(traits_df)
            if 'lat' in traits_df.columns and 'lon' in traits_df.columns:
                traits_df = auto_convert_coordinates(traits_df, lat_col='lat', lon_col='lon')
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

        # Parse DataFrame into traits list (using pandas directly, not csv.DictReader,
        # to avoid BOM and delimiter issues)
        for _, row in traits_df.iterrows():
            species_name = str(row.get('species_name', '')).strip() if has_species_name else ''
            aphia_id_raw = str(row.get('aphia_id', '')).strip() if has_aphia_id else ''

            # Handle NA values
            if species_name.upper() in ['NA', 'N/A', 'NONE', 'NAN', ''] or species_name == '':
                species_name = None

            if aphia_id_raw.upper() in ['NA', 'N/A', 'NONE', 'NAN', ''] or aphia_id_raw == '':
                aphia_id = None
            else:
                try:
                    aphia_id = int(float(aphia_id_raw))
                except ValueError:
                    raise ValueError(f"Invalid aphia_id: '{aphia_id_raw}'")

            traits.append({
                "species_name": species_name,
                "aphia_id": aphia_id,
                "spawning_start": str(row['spawning_start']).strip(),
                "spawning_end": str(row['spawning_end']).strip(),
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

            start_year = int(
                input(f"\nStart year for dispersal season [default: {datetime.now().year - 1}]: ").strip() or str(datetime.now().year - 1))

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
    subsection_header("Species Trait Data")
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
        subsection_header("Updated Species Trait Data")
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
    if species_traits_path == 'built-in_database_selection':
        traits_df = load_builtin_traits()
        if traits_df is None:
            print("Error: Could not load built-in trait database.")
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
            traits_df = pd.read_csv(species_traits_path, sep=None, engine='python',
                                    encoding='utf-8-sig')
            traits_df.columns = traits_df.columns.str.strip()
            # Auto-convert DMS coordinates if present
            if 'latitude' in traits_df.columns and 'longitude' in traits_df.columns:
                traits_df = auto_convert_coordinates(traits_df)
            if 'lat' in traits_df.columns and 'lon' in traits_df.columns:
                traits_df = auto_convert_coordinates(traits_df, lat_col='lat', lon_col='lon')

        # Validate required columns
        required_cols = ['min_PLD', 'max_PLD', 'spawning_start', 'spawning_end']
        missing_cols = [col for col in required_cols if col not in traits_df.columns]

        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return None, species_traits_path, None

        has_species_name = 'species_name' in traits_df.columns
        has_aphia_id = 'aphia_id' in traits_df.columns

        # Parse DataFrame into traits list
        for _, row in traits_df.iterrows():
            species_name = str(row.get('species_name', '')).strip() if has_species_name else ''
            aphia_id_raw = str(row.get('aphia_id', '')).strip() if has_aphia_id else ''

            # Handle NA values
            if species_name.upper() in ['NA', 'N/A', 'NONE', 'NAN', ''] or species_name == '':
                species_name = None

            if aphia_id_raw.upper() in ['NA', 'N/A', 'NONE', 'NAN', ''] or aphia_id_raw == '':
                aphia_id = None
            else:
                try:
                    aphia_id = int(float(aphia_id_raw))
                except ValueError:
                    aphia_id = None

            traits.append({
                "species_name": species_name,
                "aphia_id": aphia_id,
                "spawning_start": str(row['spawning_start']).strip(),
                "spawning_end": str(row['spawning_end']).strip(),
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
        "Important: The hydrodynamic domain needs to be larger than your study extent."
    ])

    # Check if credentials are already saved
    credentials_file = os.path.expanduser("~/.copernicusmarine/.copernicusmarine-credentials")
    if os.path.exists(credentials_file):
        print("  Using saved Copernicus Marine credentials.")
        print("  (To change credentials, run: copernicusmarine login)")
        username = None
        password = None
    else:
        print("\n  First-time setup: Your credentials will be saved for future sessions.")
        print("  You won't need to enter them again.\n")
        try:
            cm.login()
            print("\n  Credentials saved successfully.")
            username = None
            password = None
        except Exception as e:
            print(f"\n  Automatic login failed: {e}")
            print("  Falling back to manual credentials.\n")
            username = input("Enter your username: ").strip()
            password = getpass.getpass("Enter your password: ").strip()
    subsection_header("Copernicus Marine Dataset Selection")
    info_box([
        "Select a hydrodynamic dataset. The most commonly used datasets are listed below.",
        "Start dates are updated as new data is published, so data coverage is checked automatically before any download.",
        "You can also enter a custom dataset ID if your region is not covered.",
        "",
        "  1. Global Ocean (~9 km resolution)",
        "     cmems_mod_glo_phy_anfc_0.083deg_PT1H-m",
        "",
        "  2. NW European Shelf (~1.5 km resolution)",
        "     cmems_mod_nws_phy-cur_anfc_1.5km-2D_PT1H-i",
        "",
        "  3. Atlantic-Iberian-Mediterranean (~3 km resolution)",
        "     cmems_mod_ibi_phy_anfc_0.027deg-2D_PT1H-m",
        "",
        "  4. Mediterranean Sea (~4 km resolution)",
        "     cmems_mod_med_phy-cur_anfc_4.2km-2D_PT1H-m",
        "",
        "  5. Enter a custom dataset ID",
        "",
        "  6. Browse the Copernicus Marine Data Store",
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
            dataset_id = dataset_options[choice]
            print(f"\n  Selected: {dataset_id}")
            break
        elif choice == '5':
            dataset_id = input("Enter the dataset ID: ").strip()
            if not dataset_id:
                print("Error: Dataset ID cannot be empty.")
                continue
            print(f"\n  Selected: {dataset_id}")
            break
        elif choice == '6':
            webbrowser.open("https://data.marine.copernicus.eu/products")
            print("  Opening Copernicus Marine Data Store in your browser...")
            print("  Find your dataset, then select option 5 to enter its ID.\n")
            continue
        else:
            print("  Invalid choice. Enter 1-6.")
            continue

        confirm = input("  Is this correct? (yes/no) [default: yes]: ").strip().lower() or 'yes'
        if confirm not in ['yes', 'y']:
            print("  Ok, please re-select.\n")
            continue

    # Longitude range with validation
    subsection_header("Spatial Extent")
    info_box([
        "Define your hydrodynamic domain. This should be larger than your study extent to allow particles to disperse beyond the boundaries.",
    ])
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
    print()
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

    print(f"\n  Spatial extent: {min_lat}°N to {max_lat}°N, {min_lon}°W to {max_lon}°E")

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
            default_hydro_path = os.path.join(project_dirs['hydro_data'], f"hydro_data{file_ext}")
        else:
            default_hydro_path = f"./data/hydro_data{file_ext}"

        output_path = get_save_path("Output path and file name", default=default_hydro_path, extensions=[file_ext]) or default_hydro_path

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
            edit = get_yes_no(f"Do you want to edit {key} (current value: {value})?", default='no')

            if edit:
                # Give user the same choice as initial setup
                print("\nTrait data source:")
                print("  1. Use built-in trait database")
                print("  2. Provide your own CSV file")
                trait_choice = input("\nSelect option (1/2) [default: 1]: ").strip() or '1'

                if trait_choice == '1':
                    # Use built-in database , re-run species selection
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
                    new_path = get_valid_path("Species traits CSV", must_exist=True, extensions=['.csv'])

                    # Add .csv extension if missing
                    if not new_path.endswith('.csv'):
                        new_path += '.csv'

                    # Validate the file exists
                    if not os.path.exists(new_path):
                        print(f"Error: File not found: {new_path}")
                        retry = get_yes_no("Do you want to try a different file?", default='yes')

                        if not retry:
                            print("Keeping current traits.\n")
                            break
                        else:
                            continue  # Ask for file path again

                    # Reload traits from new file
                    print("\nReloading traits from new file...")
                    try:
                        # First, read and validate the CSV structure
                        traits_df = pd.read_csv(new_path, sep=None, engine='python',
                                                encoding='utf-8-sig')
                        traits_df.columns = traits_df.columns.str.strip()

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

                            retry = get_yes_no("Do you want to try a different file?", default='yes')

                            if not retry:
                                print("Keeping current traits.\n")
                                break
                            else:
                                continue  # Ask for file path again

                        # Check that at least one identifier column exists
                        has_species_name = 'species_name' in traits_df.columns
                        has_aphia_id = 'aphia_id' in traits_df.columns

                        if not has_species_name and not has_aphia_id:
                            print(f"\nError: Your CSV must have at least one identifier column:")
                            print(f"  - species_name")
                            print(f"  - aphia_id")
                            print(f"\nYour CSV has: {', '.join(traits_df.columns.tolist())}")

                            retry = get_yes_no("Do you want to try a different file?", default='yes')

                            if not retry:
                                print("Keeping current traits.\n")
                                break
                            else:
                                continue  # Ask for file path again

                        # Now read the actual trait data with validation
                        # (using traits_df already loaded above with pandas)
                        new_traits = []
                        row_num = 0
                        validation_errors = []

                        for _, row in traits_df.iterrows():
                            row_num += 1

                            # Extract and clean identifiers
                            species_name = str(row.get('species_name', '')).strip() if has_species_name else ''
                            if species_name.upper() in ['NA', 'N/A', 'NONE', 'NAN', '']:
                                species_name = None

                            aphia_id_raw = str(row.get('aphia_id', '')).strip() if has_aphia_id else ''
                            if aphia_id_raw.upper() in ['NA', 'N/A', 'NONE', 'NAN', '']:
                                aphia_id = None
                            else:
                                try:
                                    aphia_id = int(float(aphia_id_raw))
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
                                datetime.strptime(str(row['spawning_start']).strip(), '%d/%m/%Y')
                                datetime.strptime(str(row['spawning_end']).strip(), '%d/%m/%Y')
                            except ValueError:
                                validation_errors.append(
                                    f"Row {row_num}: Invalid date format. Must be DD/MM/YYYY."
                                )
                                continue

                            # If all validations pass, add to traits
                            new_traits.append({
                                "species_name": species_name,
                                "aphia_id": aphia_id,
                                "spawning_start": str(row['spawning_start']).strip(),
                                "spawning_end": str(row['spawning_end']).strip(),
                                "min_PLD": min_pld,
                                "max_PLD": max_pld
                            })

                        # Check if there were validation errors
                        if validation_errors:
                            print(f"\nError found in {new_path}:")
                            for error in validation_errors:
                                print(f"  • {error}")
                            print(f"\nLoaded {len(new_traits)} valid species (skipped {len(validation_errors)} rows with errors).")

                            if len(new_traits) == 0:
                                print("No valid species found.")
                                retry = get_yes_no("Do you want to try a different file?", default='yes')
                                if not retry:
                                    print("Keeping current traits.\n")
                                    break
                                else:
                                    continue  # Ask for file path again

                            # Ask user if they want to proceed with partial data
                            proceed = get_yes_no("Proceed with valid species only?", default='yes')

                            if not proceed:
                                retry = get_yes_no("Do you want to try a different file?", default='yes')

                                if not retry:
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

                        retry = get_yes_no("Do you want to try a different file?", default='yes')

                        if not retry:
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
                # Only prompt for values if we're enabling for the first time or user kept it enabled
                if not value:  # Enabling buffer for first time
                    print("\nEnabling temporal buffer for trait sensitivity analysis.")
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
                else:  # Buffer was already enabled, user chose to keep it
                    print("\nBuffer is already enabled. Do you want to edit the buffer values?")
                    update = get_yes_no("Edit buffer values?", default='no')

                    if not update:
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

        edit = get_yes_no(f"Do you want to edit {key} (current value: {display_value})?", default='no')

        if edit:
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
                    # Runtime cannot be manually set , it is derived from species traits + buffer
                    print(f"\nRuntime ({value} days) is automatically calculated from species traits.")
                    print("To change runtime, edit the species traits file or temporal buffer instead.")
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
        project_dirs = setup_project_directory(mode_tag='int')

    params_file = os.path.join(project_dirs['config'], "hydrodynamic_parameters.json")

    # Check if parameters file exists and offer to reuse
    if os.path.exists(params_file):
        while True:
            reuse_input = 'yes' if get_yes_no("Found existing parameters. Do you want to reuse them?", default='yes') else 'no'
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
                    subsection_header("Species Trait Data")
                    species_traits_path = hydro_params['species_traits_path']
                    if species_traits_path == 'builtin_database_selection':
                        traits_df = load_builtin_traits()
                        if traits_df is not None:
                            print("(Using built-in trait database)")
                            print(traits_df)
                    else:
                        traits_df = pd.read_csv(species_traits_path)
                        print(traits_df)
                    display_line("─")

            # Offer to edit specific parameters
            while True:
                edit_input = 'yes' if get_yes_no("\nDo you want to edit any parameters?", default='no') else 'no'
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

                # Auto-recalculate runtime from traits file
                print("\nRecalculating runtime from traits file...")
                dispersal_time, _, traits = load_spp_traits_from_path(species_traits_path)
                if dispersal_time:
                    new_start = datetime.strptime(dispersal_time["overall_dispersal_start"], '%Y-%m-%d')
                    new_end = datetime.strptime(dispersal_time["overall_dispersal_end"], '%Y-%m-%d')
                    runtime = (new_end - new_start).days
                    hydro_params['runtime'] = runtime
                    hydro_params['traits'] = traits
                    hydro_params['time_range'] = [new_start, new_end]
                    print(f"Runtime: {runtime} days (from {new_start.strftime('%Y-%m-%d')} to {new_end.strftime('%Y-%m-%d')})")
                else:
                    print("Warning: Could not read traits file. Keeping saved runtime.")
                    traits = hydro_params.get("traits", None)
                    runtime = hydro_params.get("runtime", None)

            else:
                # User said no to editing: Load values from saved params
                species_traits_path = hydro_params.get("species_traits_path", None)
                traits = hydro_params.get("traits", None)
                runtime = hydro_params.get("runtime", None)

            # Remove the traits, species traits path, and runtime before returning hydro_params.
            # Keep buffer_before/buffer_after/add_buffer: they are read later to detect whether the Type 3 trait sensitivity analysis is available
            hydro_params.pop("traits", None)
            hydro_params.pop("species_traits_path", None)
            hydro_params.pop("runtime", None)

            return hydro_params, species_traits_path, runtime, traits

    # If not reusing or no existing file, collect new parameters
    hydro_params = collect_hydro_params()

    if hydro_params is None:
        print("Error: Failed to collect parameters.")
        return None, None, None, None

    # Offer to review and edit before saving
    while True:
        review_input = input(
            "\nDo you want to review and edit the parameters before saving? (yes/no) [default: no]: ").strip().lower() or 'no'
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
    # Keep buffer_before/buffer_after in hydro_params so post_analysis can detect that a temporal buffer was used
    # Without these, Type 3 trait sensitivity is always skipped even when a buffer was added
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
      using lazy loading , no large file download is needed
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

    # Build request kwargs. Only include credentials if manual fallback was used
    open_kwargs = dict(
        dataset_id=data_request["dataset_id"],
        minimum_longitude=data_request["longitude"][0],
        maximum_longitude=data_request["longitude"][1],
        minimum_latitude=data_request["latitude"][0],
        maximum_latitude=data_request["latitude"][1],
        start_datetime=data_request["time"][0],
        end_datetime=data_request["time"][1],
    )
    if username and password:
        open_kwargs["username"] = username
        open_kwargs["password"] = password

    data = cm.open_dataset(**open_kwargs)

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

## Checks the start and end dates of the copernicus data before download
def check_dataset_coverage(dataset_id, time_range, username=None, password=None,
                           longitude_range=None, latitude_range=None):
    """
    Check, before downloading, whether a Copernicus dataset covers the requested dispersal period.

    Opens the dataset lazily (metadata only) and reads its published time coordinate to report the true available start and end.
    This catches the common case where an analysis-forecast product begins a few days or weeks after the date you asked for.

    Returns
    -------
    ok : bool
        True if the dataset fully covers the requested period.
    avail_start, avail_end : pandas.Timestamp
        The dataset's actual published start and end.
    """
    req_start = pd.Timestamp(time_range[0])
    req_end = pd.Timestamp(time_range[1])

    open_kwargs = dict(dataset_id=dataset_id)
    if longitude_range is not None:
        open_kwargs["minimum_longitude"] = longitude_range[0]
        open_kwargs["maximum_longitude"] = longitude_range[1]
    if latitude_range is not None:
        open_kwargs["minimum_latitude"] = latitude_range[0]
        open_kwargs["maximum_latitude"] = latitude_range[1]
    if username and password:
        open_kwargs["username"] = username
        open_kwargs["password"] = password

    probe = cm.open_dataset(**open_kwargs)
    times = pd.to_datetime(probe["time"].values)
    avail_start = pd.Timestamp(times.min())
    avail_end = pd.Timestamp(times.max())

    ok = (avail_start <= req_start) and (avail_end >= req_end)
    return ok, avail_start, avail_end

## Standardise Copernicus dataset dimension names to time/lat/lon
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

## Load hydrodynamic data from a local NetCDF or Zarr file (no internet required)
def load_hydrodynamic_data(file_path):
    """
    Load hydrodynamic data from a local NetCDF or Zarr file.

    Use this to reload previously downloaded data without needing an internet connection. The file must contain 'uo' (eastward velocity) and 'vo' (northward velocity) variables with time, latitude, and longitude dimensions.

    Parameters
    ----------
    file_path : str
        Path to a NetCDF (.nc) or Zarr (.zarr) file containing hydrodynamic velocity data.

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
## Create or reuse a project folder with standard subfolders for organising outputs
def setup_project_directory(project_name=None, custom_path=None, subfolders=None, mode_tag=None):
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
    subfolders : list of str, optional
        Override the default set of subfolders.
        Path A passes a reduced list ('config', 'simulation', 'figures', 'results')
        Path B leaves this as None to get the full default structure.

    Returns
    -------
    dict
        Dictionary of paths: 'root' plus one entry per subfolder created.
    """
    # Default (Path B) subfolder set. Path A passes a reduced list via subfolders=.
    if subfolders is None:
        subfolder_names = ['config', 'hydro_data', 'occurrence_data', 'grids', 'simulation', 'figures', 'animation', 'results']
    else:
        subfolder_names = list(subfolders)

    _all_descriptions = {
        'config': "config/:           Saved parameter files (.json)",
        'hydro_data': "hydro_data/:       Exported hydrodynamic data",
        'occurrence_data': "occurrence_data/:  Cleaned species occurrence data",
        'grids': "grids/:            Release and settlement grids",
        'simulation': "simulation/:       Simulation output (.zarr)",
        'figures': "figures/:          All saved figures",
        'results': "results/:          Connectivity metrics and network results",
    }
    subfolder_descriptions = [_all_descriptions[n] for n in subfolder_names if n in _all_descriptions]
    if custom_path is None:
        # Interactive: ask the user
        print("\n  1. Create a new ComCon project folder (recommended)")
        print("  2. Use your own directory")
        print("  3. Continue an existing ComCon project\n")

        choice = input("Select option (1/2/3) [default: 1]: ").strip() or '1'
    else:
        # A path was supplied programmatically (e.g. Manual Mode): treat it as a custom directory
        choice = '2'

    if choice == '3':
        # Browse for existing ComCon project folder
        try:
            import tkinter as tk
            from tkinter import filedialog
            _root = tk.Tk()
            _root.withdraw()
            _root.attributes('-topmost', True)
            selected_dir = filedialog.askdirectory(
                title="Select an existing ComCon project folder",
                mustexist=True
            )
            _root.destroy()
        except Exception:
            selected_dir = input("Enter the path to an existing ComCon project folder: ").strip()

        if selected_dir and os.path.isdir(selected_dir):
            subdirs = {'root': selected_dir}
            for name in subfolder_names:
                sub_path = os.path.join(selected_dir, name)
                if os.path.isdir(sub_path):
                    subdirs[name] = sub_path
                else:
                    subdirs[name] = selected_dir  # fallback to root

            print(f"\n  Continuing project: {selected_dir}")
            existing_files = [f for f in os.listdir(selected_dir) if not f.startswith('.')]
            if existing_files:
                print(f"  Found {len(existing_files)} existing items")
                print("  Note: Continuing a project reuses this folder. New results will overwrite previous outputs if the same name is assigned.")
            _session_pkl = os.path.join(subdirs.get('config', selected_dir), 'comcon_session.pkl')
            if os.path.exists(_session_pkl):
                print("\nA saved session checkpoint was found in this project.")
                print(f"  To resume where you left off instead of re-entering Step 1, run: load_session(r'{_session_pkl}')")
            display_line("─")
            return subdirs
        else:
            print("  No folder selected. Using default project setup instead.")
            choice = '1'

    if choice == '2':
        # Custom directory , open a folder browser (same pattern as choice 3)
        if custom_path is None:
            try:
                import tkinter as tk
                from tkinter import filedialog
                _root = tk.Tk()
                _root.withdraw()
                _root.attributes('-topmost', True)
                custom_path = filedialog.askdirectory(title="Select your project directory")
                _root.destroy()
            except Exception:
                custom_path = input("Enter the path to your project directory: ").strip()

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
        _tag = f"{mode_tag}_" if mode_tag else ""
        default_name = f"ComCon_{_tag}{datetime.now().strftime('%Y%m%d_%H%M')}"
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

    # Build the folder-structure box dynamically so Path A doesn't show entries
    # for folders that don't exist (e.g. occurrence, data).
    # Printed ONCE after all directories are created (was inside the loop , printed 5x).
    display_lines = [
        f"Project directory created: {root}",
        "",
        "Folder structure:",
        f"  {safe_name}/",
    ]
    for i, desc in enumerate(subfolder_descriptions):
        prefix = "  └── " if i == len(subfolder_descriptions) - 1 else "  ├── "
        display_lines.append(f"{prefix}{desc}")
    info_box(display_lines)

    return subdirs
# endregion

# region Downloading Species Occurrence Data
## region Set Up Functions
### Edit Species Parameters
def edit_spp_params(spp_params):
    """
    Interactively edits species occurrence data collection parameters.

    Allows modification of parameters controlling species distribution data download
    from GBIF, including study extent definition, data cleaning options and output
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
        # Skip user_data_file_path if data_source is 'gbif'
        if key == 'user_data_file_path':
            if spp_params.get('data_source') == 'gbif':
                continue

        # Display boolean values as yes/no instead of True/False
        display_value = value
        if isinstance(value, bool):
            display_value = 'yes' if value else 'no'

        edit = get_yes_no(f"Do you want to edit {key} (current value: {display_value})?", default='no')
        if edit:
            # Special handling for boolean parameters
            if isinstance(value, bool):
                new_value_input = input(
                    f"Enter the new value for {key} (yes/no, current: {'yes' if value else 'no'}): ").strip().lower()
                new_value = new_value_input in ['yes', 'y']
            # Collect new input for other parameter types
            elif key == 'geojson_file_path':
                # Special handling for geojson - offer redraw option
                print("\nOptions for updating study extent:")
                print("  1. Enter path to existing GeoJSON file")
                print("  2. Draw new study extent on map")
                while True:
                    geojson_choice = input("Choose option (1/2): ").strip()
                    if geojson_choice == '1':
                        new_value = get_valid_path("GeoJSON file", must_exist=True, extensions=['.geojson', '.json'])
                        if not new_value.endswith(".geojson"):
                            new_value += ".geojson"
                        if not os.path.exists(new_value):
                            print(f"Warning: File not found: {new_value}")
                            retry = get_yes_no("Try again?", default='yes')
                            if retry:
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
    """
    subsection_header("Select your study extent")
    info_box([
        "Your study extent defines the region where particles will be released and where connectivity will be analysed. ",
        "This should be smaller than your hydrodynamic domain.",
        "",
        "You can either upload an existing GeoJSON polygon file or draw one on an interactive map.",
    ])
    geojson_reuse = 'yes' if get_yes_no("Do you want to use an existing GeoJSON file?", default='no') else 'no'

    if geojson_reuse in ["yes", "y"]:
        # Prompt for GeoJSON file path, adding the .geojson extension if not provided
        geojson_file_path = get_valid_path("Enter the GeoJSON file path", must_exist=True, extensions=['.geojson', '.json'])
        if not geojson_file_path.endswith(".geojson"):
            geojson_file_path += ".geojson"
    else:
        geojson_file_path = None

    return {
        'species_traits_path': species_traits_path,
        'geojson_file_path': geojson_file_path,
    }

### Create Map
def create_map():
    """
    Creates an interactive map for defining the study extent polygon.

    Launches an interactive map in Jupyter allowing users to draw
    a polygon defining their study extent. The polygon is captured directly
    and saved as a GeoJSON file automatically.

    Returns
    -------
    str
        File path to the saved GeoJSON file containing the drawn polygon

    Notes
    -----
    - Uses ipyleaflet if available (captures polygon directly , no manual export needed)
    - Falls back to folium if ipyleaflet is not installed (requires manual export)
    - Must be run in a Jupyter environment
    """
    # Determine save location. Use globals().get so this never raises NameError when called from
    # automatic mode (where interactive's module-level project_dirs may never have been assigned).
    _pd = globals().get('project_dirs')
    save_dir = _pd.get('config', '.') if _pd else '.'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "study_area.geojson")

    # Try ipyleaflet first (better UX , captures polygon directly)
    try:
        from ipyleaflet import Map, DrawControl, basemaps

        print("\nDraw a polygon around your study extent on the map below.")
        print("  1. Click the polygon tool (pentagon icon) on the left side of the map")
        print("  2. Click on the map to place vertices of your polygon")
        print("  3. Click the first vertex again to close the polygon")
        print("  4. Click the 'Confirm' button below the map when done")

        # Create map
        m = Map(basemap=basemaps.OpenStreetMap.Mapnik, center=(45, 0), zoom=4,
                layout=widgets.Layout(width='100%', height='500px'))

        # Add draw control , polygon only
        draw_control = DrawControl(
            polygon={'shapeOptions': {'color': '#FF4500', 'weight': 3, 'fillOpacity': 0.1}},
            polyline={}, rectangle={}, circle={}, circlemarker={}, marker={}
        )
        m.add(draw_control)

        # Status output
        status_output = widgets.Output()
        drawn_polygons = []

        ## Handle draw/delete events from the Leaflet drawing control
        def handle_draw(target, action, geo_json):
            """Update the drawn polygon list and print status when the user draws or deletes a polygon."""
            with status_output:
                clear_output()
                if action == 'created':
                    drawn_polygons.clear()
                    drawn_polygons.append(geo_json)
                    print(f"  Polygon drawn with {len(geo_json['geometry']['coordinates'][0])} vertices.")
                    print("  Click 'Confirm' to save, or redraw to replace.")
                elif action == 'deleted':
                    drawn_polygons.clear()
                    print("  Polygon deleted. Draw a new one.")

        draw_control.on_draw(handle_draw)

        # Confirm button
        confirm_button = widgets.Button(
            description="Confirm study extent",
            button_style='success',
            icon='check',
            layout=widgets.Layout(width='200px', height='40px')
        )

        confirmed = [False]
        confirm_output = widgets.Output()

        ## Save the drawn polygon to a GeoJSON file when the Confirm button is clicked
        def on_confirm(b):
            """Export the most recently drawn polygon to a GeoJSON file."""
            with confirm_output:
                clear_output()
                if not drawn_polygons:
                    print("  Error: No polygon drawn. Please draw a polygon first.")
                    return
                # Save as GeoJSON
                geojson_data = {
                    "type": "FeatureCollection",
                    "features": [{
                        "type": "Feature",
                        "geometry": drawn_polygons[-1]['geometry'],
                        "properties": {}
                    }]
                }
                with open(save_path, 'w') as f:
                    json.dump(geojson_data, f, indent=2)
                print(f"  study extent saved to: {save_path}")
                confirmed[0] = True

        confirm_button.on_click(on_confirm)

        # Display everything
        display(m)
        display(widgets.HBox([confirm_button, status_output]))
        display(confirm_output)

        # Wait for user to confirm
        print("\n  After drawing and confirming, press Enter to continue...")
        input()

        if confirmed[0] and os.path.exists(save_path):
            return save_path
        elif os.path.exists(save_path):
            # User might have confirmed before pressing Enter
            return save_path
        else:
            print("  No polygon was confirmed. Falling back to manual file entry.")
            file_path = get_valid_path("Enter the path to your GeoJSON file",
                                       must_exist=True, extensions=['.geojson', '.json'])
            if not file_path.endswith(".geojson"):
                file_path += ".geojson"
            return file_path

    except ImportError:
        # Fall back to folium
        print("\n  Note: Install 'ipyleaflet' for a better experience (pip install ipyleaflet)")
        print("  Using folium. You will need to manually export and enter the file path.\n")
        print("Follow these steps to define your study extent:")
        print("  1. Use the drawing tool on the map to draw a polygon around your area of interest")
        print("  2. After drawing, click the 'Export' button on the map to save the polygon as a GeoJSON file")
        print("  3. Once you have saved the file, return here and enter the full path to your saved GeoJSON file below")

        m = folium.Map()
        draw = plugins.Draw(
            export=True,
            draw_options={'polygon': True, 'polyline': False, 'rectangle': False, 'circle': False,
                          'marker': False, 'circlemarker': False}
        )
        draw.add_to(m)
        display(m)

        print("  Note: File extensions are added automatically if not included.")
        file_path = get_valid_path("Enter the path to your saved GeoJSON file",
                                   must_exist=True, extensions=['.geojson', '.json'])
        if not file_path.endswith(".geojson"):
            file_path += ".geojson"
        return file_path

### Read GeoJSON File
def read_geojson(geojson_file_path):
    """
        Reads and validates GeoJSON file containing the study extent polygon.

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
    shapely_polygon = shapely_polygon_mod.orient(shapely_polygon, sign=1.0)
    # Update polygon_coords to match the oriented polygon
    polygon_coords = list(shapely_polygon.exterior.coords)

    # Create WKT format string
    geometry = 'POLYGON((' + ', '.join([f'{round(lon, 4)} {round(lat, 4)}' for lon, lat in polygon_coords]) + '))'

    return geometry, shapely_polygon

### Get the aphia id
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
    Downloads species occurrence records from GBIF within defined study extent.

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
        GeoJSON polygon geometry defining the study extent bounds. Must contain
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
    - Records are automatically filtered to study extent during download
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
            break  # Success , exit retry loop

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
                print(f"      - study extent polygon may have an issue")
                print(f"      - Network/firewall may be blocking the request")
                print(f"    Try: Redraw your study extent polygon or try again later.")

    return all_records

### Download all species records from within the study extent
def get_all_spp_occ(geometry, limit=300):
    """
    Downloads occurrence records for all species within the study extent.

    Useful for exploratory analysis to discover what species exist in your
    study extent before selecting focal species for connectivity modelling.

    Parameters
    ----------
    geometry : dict
        GeoJSON polygon geometry defining the study extent bounds. Must contain
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

    print("\nQuerying GBIF for all species in study extent...")
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
            print(f"\nRetrieved {len(all_records):,} records...")

        # Check if we got fewer results than requested (last page)
        if len(results['results']) < limit:
            break

        offset += limit
        time.sleep(0.5)  # Be nice to GBIF

    if len(all_records) > 0:
        print(f"\nRetrieved {len(all_records):,} total occurrence records")
    else:
        print(f"\nRetrieved 0 records. GBIF may be having issues or no data in this area")

    return all_records

### Create map of hydrodynamic data extent and masks
def plot_masks(landmask, coastalmask, lonarray, latarray, geojson_file_path=None, show_coastal=True, show_save_prompt=True, title='Hydrodynamic Extent'):
    """
    Visualise land/coastal/ocean classification from hydrodynamic data.

    Shows immediate validation of downloaded hydrodynamic data before proceeding
    with species data collection. Users can verify the domain extent, grid
    resolution, and coastal zone definition.

    Parameters
    ----------
    landmask : np.ndarray
        2D binary array (1=land, 0=ocean)
    coastalmask : np.ndarray or None
        2D binary array (1=coastal, 0=non-coastal). Ignored when show_coastal=False.
    lonarray : np.ndarray
        2D meshgrid of longitudes
    latarray : np.ndarray
        2D meshgrid of latitudes
    geojson_file_path : str, optional
        Path to study extent GeoJSON file. If provided, study extent outline will be displayed.
    show_coastal : bool, default True
        If False, the coastal layer and its legend/info entries are omitted.
        Used by Path A which has no coastal classification.
    show_save_prompt : bool, default True
        If False, the interactive "save figure?" input prompt is suppressed.
        Used by Path A which runs end-to-end without user prompts.
    title : str, default 'Hydrodynamic Extent'
        Plot title. Override when reusing this function (e.g. native vs binned).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure

    Notes
    -----
    - Shows full hydrodynamic domain extent
    - Displays study extent outline if provided
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

    # Standardised landscape figsize
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 1, height_ratios=[5, 1.5], hspace=0.12, figure=fig)

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
    if show_coastal and coastalmask is not None:
        coastal_points = (landmask == 0) & (coastalmask == 1)
        ocean_points = (landmask == 0) & (coastalmask == 0)
    else:
        coastal_points = np.zeros_like(landmask, dtype=bool)
        ocean_points = (landmask == 0)
    land_points = (landmask == 1)

    # Count cells
    n_coastal = int(np.sum(coastal_points))
    n_ocean = int(np.sum(ocean_points))
    n_land = int(np.sum(land_points))
    n_total = n_coastal + n_ocean + n_land

    # Plot points
    ax.scatter(lonarray[ocean_points], latarray[ocean_points],
               c=COLORS['ocean'], s=20, alpha=0.6, marker='o',
               edgecolors='black', linewidths=0.2,
               transform=ccrs.PlateCarree(), zorder=3)

    if show_coastal and n_coastal > 0:
        ax.scatter(lonarray[coastal_points], latarray[coastal_points],
                   c=COLORS['coastal'], s=20, alpha=0.6, marker='o',
                   edgecolors='black', linewidths=0.2,
                   transform=ccrs.PlateCarree(), zorder=4)

    ax.scatter(lonarray[land_points], latarray[land_points],
               c=COLORS['land'], s=20, alpha=0.6, marker='o',
               edgecolors='black', linewidths=0.2,
               transform=ccrs.PlateCarree(), zorder=2)

    # Add study extent outline
    if geojson_file_path:
        try:
            _, shapely_polygon = read_geojson(geojson_file_path)
            if shapely_polygon:
                x, y = shapely_polygon.exterior.xy
                ax.plot(x, y, color=COLORS['border'], linewidth=3.5,
                        transform=ccrs.PlateCarree(), zorder=6,
                        linestyle='-', alpha=0.9)
        except Exception as e:
            print(f"Note: Could not display study extent outline: {e}")

    # Calculate grid info. Convert to km using the LATITUDE spacing, where 1 deg ~= 111 km
    # everywhere. Using the longitude spacing needs a cos(latitude) factor; without it the value
    # is inflated (e.g. 0.210 deg lon at 50N is ~15 km, not 0.210*111 = 23.3 km). The binning
    # targets square cells in km, so the latitude-based value is the true cell size.
    grid_spacing_lon = np.abs(np.diff(lonarray[0, :]).mean())
    grid_spacing_lat = np.abs(np.diff(latarray[:, 0]).mean())
    grid_spacing_km = grid_spacing_lat * 111

    # Title
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10, family='Arial')

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.4,
                      linestyle='--', color='gray')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 14, 'family': 'Arial'}
    gl.ylabel_style = {'size': 14, 'family': 'Arial'}

    # Footer: 3 evenly spaced boxes
    footer_gs = gs[1].subgridspec(1, 5, wspace=0.05,
                                  width_ratios=[0.15, 1, 1, 1.2, 0.15])

    # Box 1: Model Domain
    box1_ax = fig.add_subplot(footer_gs[1])
    box1_ax.axis('off')
    box1_text = 'Model Domain\n\n'
    box1_text += f'Resolution: {grid_spacing_km:.1f} km\n\n'
    box1_text += f'({grid_spacing_lat:.3f}° lat × {grid_spacing_lon:.3f}° lon)'
    box1_ax.text(0.5, 0.5, box1_text,
                 fontsize=12, family='Arial', va='center', ha='center',
                 transform=box1_ax.transAxes, linespacing=0.8,
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                           alpha=0.95, edgecolor='#CCCCCC', linewidth=1.5))

    # Box 2: Extent (column 2)
    box2_ax = fig.add_subplot(footer_gs[2])
    box2_ax.axis('off')
    box2_text = 'Extent\n\n'
    box2_text += f'Lon: {lon_min:.2f}° to {lon_max:.2f}°\n\n'
    box2_text += f'Lat: {lat_min:.2f}° to {lat_max:.2f}°'
    box2_ax.text(0.5, 0.5, box2_text,
                 fontsize=12, family='Arial', va='center', ha='center',
                 transform=box2_ax.transAxes, linespacing=0.8,
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                           alpha=0.95, edgecolor='#CCCCCC', linewidth=1.5))

    # Box 3: Grid Classification + Legend combined
    box3_ax = fig.add_subplot(footer_gs[3])
    box3_ax.axis('off')
    legend_elements = [
        Patch(facecolor=COLORS['ocean'], alpha=0.6, edgecolor='black',
              label=f'Ocean: {n_ocean} cells ({100 * n_ocean / n_total:.1f}%)'),
    ]
    if show_coastal:
        legend_elements.append(
            Patch(facecolor=COLORS['coastal'], alpha=0.6, edgecolor='black',
                  label=f'Coastal: {n_coastal} cells ({100 * n_coastal / n_total:.1f}%)')
        )
    legend_elements.append(
        Patch(facecolor=COLORS['land'], alpha=0.6, edgecolor='black',
              label=f'Land: {n_land} cells ({100 * n_land / n_total:.1f}%)')
    )
    if geojson_file_path:
        legend_elements.append(
            Line2D([0], [0], color=COLORS['border'], linewidth=2.5, label='Study Extent')
        )
    box3_ax.legend(handles=legend_elements, loc='center',
                   fontsize=12, frameon=True, edgecolor='#CCCCCC',
                   framealpha=0.95, title='Grid Classification', title_fontsize=12)

    plt.tight_layout()
    plt.show()

    if not show_save_prompt:
        return fig

    # Ask if user wants to save
    save_map = input("\nSave domain map? (yes/no) [default: no]: ").strip().lower() or 'no'

    if save_map in ['yes', 'y']:
        default_dir = get_default_dir('grids')
        save_dir = get_valid_dir("Output directory", default=default_dir)
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
    - Uses scipy.ndimage.binary_dilation with a 3x3 structure
    - Considers 8-connectivity (N, S, E, W and the four diagonals) for adjacency
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
def create_masks(hydro_data, geojson_file_path=None, show_map=True):
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
    - Adjacency is defined using 8-connectivity (N, S, E, W and diagonal neighbours)
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

    # Visualise masks for immediate validation (Path B default, Path A suppresses and shows its own native+binned view)
    if show_map:
        if geojson_file_path:
            subsection_header("Visualising the Study Extent")
            plot_masks(landmask, coastalmask, lonarray, latarray, geojson_file_path,
                       title='Hydrodynamic Extent with Study Extent Outline')
        else:
            subsection_header("Visualising the Hydrodynamic Extent")
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

    # Print filtering results (skip for option 3 , no filtering was applied)
    n_original = len(distributions)
    n_after = len(clean_spp)
    n_removed = n_original - n_after
    pct_removed = 100 * n_removed / n_original if n_original > 0 else 0

    if habitat_type != 'none':
        data_box("Filtering Results", [
            f"Original occurrences: {n_original}",
            f"After filtering: {n_after}",
            f"Removed: {n_removed} ({pct_removed:.1f}%)"
        ])

    if n_after == 0:
        print("\nWarning: No occurrences remain after filtering!")
        print("Consider:")
        print("  - Expanding your study extent")
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
    - Low sample: Consider expanding study extent or accepting uncertainty
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
        print("  - Species range doesn't overlap with study extent")

    if validation_report['low_sample_species']:
        print(f"\n Species with low sample size:")
        for species in validation_report['low_sample_species']:
            count = validation_report['total_records'][species]
            print(f"   - {species}: {count} records (minimum recommended: {min_records})")
        print("\n Consider:")
        print("  - Expanding your study extent")
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
        Number of species to generate colours for

    Returns:
    --------
    colours : list
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

        # Get colours from tab20 (first 20)
        tab20 = colormaps.get_cmap('tab20')
        colors.extend([tab20(i) for i in range(20)])

        # Add colours from Set3 (pastel, 12 colours)
        set3 = colormaps.get_cmap('Set3')
        colors.extend([set3(i) for i in range(12)])

        # Add colours from Paired (12 colours)
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
    Create before/after species distribution maps as two separate figures.

    Uses the same layout as plot_masks (single map + legend row) to ensure consistent styling across all ComCon figures.
    """
    ## Format a species name for use as a matplotlib label (italicised LaTeX string)
    def format_species_name(name, italic=True):
        """Return the species name with underscores replaced by spaces and the first letter
        (genus) capitalised, optionally formatted as an italic LaTeX string."""
        formatted = str(name).replace('_', ' ')
        formatted = formatted[:1].upper() + formatted[1:] if formatted else formatted
        if not italic:
            return formatted
        parts = formatted.split(' ', 1)
        if len(parts) == 2:
            return f"$\\mathit{{{parts[0]}}}$ $\\mathit{{{parts[1]}}}$"
        return f"$\\mathit{{{formatted}}}$"

    # Styling , match plot_masks
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12

    COLORS = {
        'land': 'lightgray',
        'ocean': 'lightblue',
        'border': '#FF4500',
    }

    _, shapely_polygon = read_geojson(geojson_file_path)

    # Use study extent bounds with buffer
    if shapely_polygon:
        bounds = shapely_polygon.bounds
        lon_range_study = bounds[2] - bounds[0]
        lat_range_study = bounds[3] - bounds[1]
        buffer_lon = lon_range_study * 0.15
        buffer_lat = lat_range_study * 0.20
        lon_min = bounds[0] - buffer_lon
        lon_max = bounds[2] + buffer_lon
        lat_min = bounds[1] - buffer_lat
        lat_max = bounds[3] + buffer_lat
    else:
        lon_min, lon_max = lonarray.min(), lonarray.max()
        lat_min, lat_max = latarray.min(), latarray.max()

    # Group by site when the data is site-based (a 'site_name' column is present), otherwise by species.
    group_col = 'site_name' if 'site_name' in distributions.columns else 'species'
    group_label = 'sites' if group_col == 'site_name' else 'species'
    italicise = (group_col == 'species')

    species_list = distributions[group_col].unique()
    n_species_before = len(species_list)
    n_species_after = clean_spp[group_col].nunique() if len(clean_spp) > 0 else 0
    species_colors = get_spp_colours(n_species_before)

    # Filter label
    filter_labels = {'coastal': 'Coastal', 'ocean': 'Ocean', 'none': 'Unfiltered'}
    filter_label = filter_labels.get(habitat_type, 'Unfiltered')

    # Legend elements (shared)
    legend_elements = []
    after_groups = clean_spp[group_col].unique() if len(clean_spp) > 0 else []
    for i, species in enumerate(species_list):
        if species in after_groups:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=species_colors[i],
                       markersize=7, markeredgecolor='black',
                       markeredgewidth=0.5,
                       label=format_species_name(species, italic=italicise))
            )
    legend_elements.append(
        Line2D([0], [0], color=COLORS['border'], linewidth=2.5, label='study extent')
    )

    # Legend layout
    if n_species_before <= 3:
        legend_ncol = n_species_before + 1  # +1 for study extent
    elif n_species_before <= 8:
        legend_ncol = 3
    else:
        legend_ncol = 4

    ## Build a single species occurrence map figure with a legend row (used for before and after panels)
    def _make_panel(data, title_text, info_text):
        """Create a single figure with map + legend row, matching plot_masks style."""
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 1, height_ratios=[5, 0.4], hspace=0.05, figure=fig)

        ax = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none', zorder=1)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=2)
        ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.4, zorder=2)

        # Plot species points
        if len(data) > 0:
            for i, species in enumerate(species_list):
                species_data = data[data[group_col] == species]
                if len(species_data) > 0:
                    ax.scatter(species_data['longitude'], species_data['latitude'],
                               s=40, c=[species_colors[i]], alpha=0.8,
                               edgecolors='black', linewidths=0.5,
                               transform=ccrs.PlateCarree(), zorder=3)

        # study extent polygon
        if shapely_polygon:
            x, y = shapely_polygon.exterior.xy
            ax.plot(x, y, color=COLORS['border'], linewidth=2.5,
                    transform=ccrs.PlateCarree(), zorder=5, alpha=0.9)

        # Info text box
        ax.text(0.02, 0.05, info_text,
                transform=ax.transAxes, fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
                          edgecolor='black', linewidth=0.8))

        # Gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.3,
                           linestyle='--', color='gray')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 12}
        gl.ylabel_style = {'size': 12}

        ax.set_title(title_text, fontsize=14, fontweight='bold', pad=8)

        # Legend row , reflect the groups actually present in THIS panel's data
        # (so the "before" map legends its own groups, not just the post-filter ones).
        _present = set(data[group_col].unique()) if len(data) > 0 else set()
        _panel_legend = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=species_colors[i],
                   markersize=7, markeredgecolor='black', markeredgewidth=0.5,
                   label=format_species_name(sp, italic=italicise))
            for i, sp in enumerate(species_list) if sp in _present
        ]
        _panel_legend.append(
            Line2D([0], [0], color=COLORS['border'], linewidth=2.5, label='study extent'))
        legend_ax = fig.add_subplot(gs[1])
        legend_ax.axis('off')
        legend_ax.legend(handles=_panel_legend, loc='upper center',
                         ncol=legend_ncol,
                         fontsize=12, frameon=True,
                         edgecolor='#CCCCCC', framealpha=1.0,
                         columnspacing=0.8, handletextpad=0.3)

        plt.subplots_adjust(top=0.94, bottom=0.04, hspace=0.05)
        return fig

    # For 'none' filtering, distributions == clean_spp, so a before/after comparison would be
    # identical: show a single occurrence map instead. For coastal/ocean, show before + after.
    # (Site names are used automatically in place of species when a 'site_name' column is present.)
    single = (habitat_type == 'none')

    if single:
        fig1 = _make_panel(
            distributions,
            'Species Occurrence Data',
            f'{n_species_before} {group_label}, {len(distributions):,} records'
        )
    else:
        fig1 = _make_panel(
            distributions,
            f'Species Occurrence Data: {filter_label} Filtering (Before)',
            f'Before: {n_species_before} {group_label}, {len(distributions):,} records'
        )
    plt.show()

    fig2 = None
    if not single:
        fig2 = _make_panel(
            clean_spp,
            f'Species Occurrence Data: {filter_label} Filtering (After)',
            f'After: {n_species_after} {group_label}, {len(clean_spp):,} records'
        )
        plt.show()

    prompt = "Save the occurrence map?" if single else "Save species maps?"
    save_map = input(f"\n{prompt} (yes/no) [default: no]: ").strip().lower() or 'no'
    if save_map in ['yes', 'y']:
        default_dir = get_default_dir('figures')
        save_dir = get_valid_dir("Output directory", default=default_dir)
        file_format = input("Format (tif/png/pdf/svg) [default: tif]: ").strip().lower() or 'tif'
        if file_format not in ['tif', 'png', 'pdf', 'svg']:
            file_format = 'tif'
        os.makedirs(save_dir, exist_ok=True)

        _to_save = [(fig1, 'occurrence')] if single else [(fig1, 'before'), (fig2, 'after')]
        for fig_obj, suffix in _to_save:
            filename = (f'species_{suffix}.{file_format}' if single
                        else f'species_filtering_{suffix}.{file_format}')
            save_path = os.path.join(save_dir, filename)
            fig_obj.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"  Saved to: {save_path}")

    plt.close(fig1)
    if fig2 is not None:
        plt.close(fig2)
    return fig1, fig2

### Add validation for uploaded data against selected study extent
def validate_uploaded_data(distributions, shapely_polygon):
    """
    Validate uploaded species data falls within study extent polygon.

    When users upload their own species occurrence data, this function ensures
    the data falls within the defined study extent. Points outside the study extent
    cannot be used for release site definition and will lead to incorrect mapping.

    The study extent polygon defines where particles will be released and where
    settlement will be evaluated, not the full hydrodynamic extent (which is the
    larger hydrodynamic extent).

    Parameters:
    -----------
    distributions : pd.DataFrame
        Species occurrence data with 'latitude' and 'longitude' columns
    shapely_polygon : shapely.geometry.Polygon
        study extent boundary polygon

    Returns:
    --------
    pd.DataFrame
        Validated (and possibly filtered) occurrence data

    Raises:
    -------
    ValueError
        If user chooses to abort when data is outside study extent
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
    subsection_header("Geographic Coverage Check")
    print(f"Total records: {len(distributions):,}")
    print(f"Within study extent: {len(within_bounds):,}")
    print(f"Outside study extent: {len(outside_bounds):,}")

    if len(outside_bounds) > 0:
        pct_outside = 100 * len(outside_bounds) / len(distributions)
        print(f"Percentage outside: {pct_outside:.1f}%")

        # Interactive: Ask user what to do
        print("\nWarning: Some data points are outside your selected study extent.")
        print("\nPoints outside the study extent will not be included for connectivity modelling and analysis.")
        print("\nWhat would you like to do?")
        print("  1. Automatically filter out points outside study extent")
        print("  2. Keep all data (may cause incorrect results)")
        print("  3. Abort, redraw your study extent polygon or fix the data manually")

        while True:
            choice = input("\nEnter your choice (1/2/3) [default: 1]: ").strip() or '1'

            if choice == '1':
                print("\nFiltering data to study extent...")
                filtered_data = distributions.iloc[within_bounds].copy()
                print(f"Filtered to {len(filtered_data):,} records")
                return filtered_data

            elif choice == '2':
                print("\nWarning: Keeping all data. Results may be incorrect for points outside study extent.")
                return distributions

            elif choice == '3':
                print("\nAborted. Please fix your data and try again.")
                print("Options to fix:")
                print("  - Expand your study extent (GeoJSON polygon)")
                print("  - Filter your CSV to only include points within study extent")
                raise ValueError("User aborted due to data outside study extent")

            else:
                print("Invalid choice. Enter 1, 2, or 3.")

    else:
        print(f"All records are within the study extent")
        return distributions

# endregion

## region Complete Species Occurrence Data Download (wrapper function)
def spp_occ(hydro_data, species_traits_path):
    """
        Complete workflow for species occurrence data setup and processing.
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
        - study extent polygon can be reused or redrawn
        - Habitat filtering can remove inland and open ocean records
        - Both pre-comparison and post-comparison maps can be generated
        - Data validation checks ensure uploaded data matches study extent

        Workflow Steps
        --------------
        1. Check for existing parameters (spp_occ_params.json)
        2. Collect or reuse parameters
        3. Create or load study extent polygon (GeoJSON)
        4. Download from GBIF or load uploaded file
        5. Generate land and coastal masks
        6. Apply habitat filtering
        7. Validate data within study extent
        8. Generate comparison maps (optional)
        9. Save cleaned data to CSV (optional)
        10. Create final distribution map
        11. Return cleaned data and masks

        Data Cleaning Steps
        -------------------
        - Remove records without coordinates
        - Remove records on land (using hydrodynamic landmask)
        - Apply coastal buffer filtering (if specified)
        - Clip to study extent polygon
        - Remove duplicate records

        Parameter Persistence
        ---------------------
        Parameters are saved to allow:
        - Exact replication of data acquisition
        - Quick reruns with same settings
        - Modification of specific parameters without re-entering all values
        """
    params_file = os.path.join(project_dirs['config'], "spp_occ_params.json")

    # Step 1: Check if a previous set of parameters exists and reuse or edit them
    spp_params = None
    if os.path.exists(params_file):
        with open(params_file, 'r') as file:
            spp_params = json.load(file)
        print("\nFound existing parameters:")
        for key, value in spp_params.items():
            print(f"  {key}: {value}")

        reuse = get_yes_no("\nDo you want to reuse these parameters?", default='yes')
        if not reuse:
            os.remove(params_file)
            print("Starting fresh.")
            spp_params = None
        else:
            edit = get_yes_no("\nDo you want to edit any parameters?", default='no')
            if edit:
                spp_params = edit_spp_params(spp_params)
            else:
                print("Reusing existing parameters.")

    # If no params yet (no file existed, or user chose fresh), collect new ones
    if spp_params is None:
        spp_params = collect_spp_occ_params(species_traits_path)

    # Save updated parameters after any potential edits
    with open(params_file, 'w') as file:
        json.dump(spp_params, file, indent=4)

    # Step 2: Use the geojson file path if it exists
    geojson_file_path = spp_params.get('geojson_file_path')

    # Check: Only create a new map if the geojson file path is missing, None, or invalid
    if not geojson_file_path or not os.path.exists(geojson_file_path):
        # Before creating map:
        subsection_header("Defining your study extent")

        info_box([
            "You will be prompted to draw a polygon defining your study extent.",
            "",
            "Why do you need a study extent?",
            "  Your hydrodynamic domain covers a large area where ocean currents are simulated.",
            "  The study extent defines the smaller region where your analysis will focus.",
            "",
            "Hydrodynamic domain (set in Step 1):",
            "  - The large area where ocean currents are simulated",
            "  - Particles can disperse throughout this entire region",
            "",
            "Study extent (this step):",
            "  - Your focused region of interest within the domain",
            "  - Determines where particles can be released from and can settle to",
            "  - Network analysis results will focus on sites within this extent",
            "  - Should fit within the hydrodynamic domain with some margin",
            "  - Example: If your hydro domain covers the Mediterranean, your study extent might focus on the Adriatic Sea",
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
            "   - Data will be filtered to your study extent",
            "",
            "Option 2: Your own data",
            "   - Upload your own pre-collected species distribution data or habitat suitability data",
            "   - Must be in CSV format with columns: species, latitude, longitude",
            "   - Data will be filtered to your study extent",
            "   - See 'Part 7: Occurrence Data CSV Format' in the User Guide for format details"
        ])

        while True:
            data_source_choice = input(
                "\nChoose data source (1 for GBIF, 2 for your own data) [default: 1]: ").strip() or '1'

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
        print("\n2. Download all species in your study extent")
        print("  - For exploratory analysis only as trait data will be missing")

        if 'gbif_mode' in spp_params and spp_params['gbif_mode']:
            mode_choice = spp_params['gbif_mode']
            print(f"\n  Using previously selected GBIF mode: {mode_choice}")
        else:
            while True:
                mode_choice = input("\nChoose mode (1 or 2) [default: 1]: ").strip() or "1"
                if mode_choice in ["1", "2"]:
                    break
                print("  Invalid input. Please enter 1 or 2.")
            spp_params['gbif_mode'] = mode_choice

        # Load the species traits CSV (needed for Mode 1)
        # Use traits_temp to avoid overwriting global traits list
        if species_traits_path == 'builtin_database_selection':
            traits_temp = load_builtin_traits()
        else:
            traits_temp = pd.read_csv(species_traits_path, sep=None, engine='python',
                                      encoding='utf-8-sig')
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

            print(f"\nDownloaded {len(distributions):,} occurrence records")

            # Check if any data was retrieved
            if len(distributions) == 0:
                print("\nWarning: No species occurrence data was retrieved from GBIF.")
                print("\nPossible reasons:")
                print("  1. No GBIF records exist for these species in your study extent")
                print("  2. Temporary GBIF server connectivity issue")
                print("  3. Species names/AphiaIDs may not match GBIF database")
                print("\nSuggestions:")
                print("  - Try a different/larger study extent")
                print("  - Check species names match GBIF taxonomy")
                print("  - Use your own occurrence data (Option 2 at data source selection)")
                raise ValueError("Cannot proceed without species occurrence data.")

        else:

            # Mode 2: Download all species in study extent (exploratory)
            print(f"\nDownloading species occurrence data from GBIF for all species in your study extent.")
            print("Warning: This may download many species and take significant time.")

            all_data = get_all_spp_occ(geometry)  # Call once, not in a loop

            distributions = pd.DataFrame(all_data)

            if len(distributions) == 0:
                print("Warning: No species data retrieved")
                print("\nThis could be due to:")
                print("  1. Temporary GBIF server issue. Try again in 5 minutes")
                print("  2. No species records in your study extent")
                print("\nOptions:")
                print("  - Wait a few minutes and try again")
                print("  - Use Mode 1 (species-specific download) instead")

                raise ValueError("Cannot proceed without species data. Please try again or use Mode 1.")

            print(f"\nDownloaded {len(distributions):,} occurrence records")

            unique_species = distributions['species'].nunique()

            print(f"Found {unique_species} unique species in study extent.")

            # Show species not in trait database
            trait_species = set(species_list)
            found_species = set(distributions['species'].unique())
            extra_species = found_species - trait_species

            if extra_species:
                print(f"\nNote: {len(extra_species)} species found that are not in your trait database.")

                print("These species will be included in occurrence data but cannot be used in connectivity modelling without trait information.")

    else:  # user data
        # Check if file path already saved in parameters
        if 'user_data_file_path' in spp_params and spp_params['user_data_file_path']:
            file_path = spp_params['user_data_file_path']
            print(f"\nUsing saved file path: {file_path}")
        else:
            print(
                "\nUpload your species occurrence data in the required format: CSV with columns 'species', 'latitude', 'longitude'.")
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
                    retry = get_yes_no("Try another file?", default='yes')
                    if not retry:
                        raise ValueError("Cannot proceed without valid species data.")
                    file_path = get_valid_path("Occurrence data CSV", must_exist=True, extensions=['.csv'])
                    if not file_path.endswith(".csv"):
                        file_path += ".csv"
                    spp_params['user_data_file_path'] = file_path
                    continue

                # Success
                print(f"\nLoaded {len(distributions)} occurrence records from {len(distributions['species'].unique())} species")

                # Auto-convert DMS coordinates if detected
                distributions = auto_convert_coordinates(distributions)
                # Also try lat/lon column name variants
                if 'lat' in distributions.columns and 'latitude' not in distributions.columns:
                    distributions = distributions.rename(columns={'lat': 'latitude'})
                if 'lon' in distributions.columns and 'longitude' not in distributions.columns:
                    distributions = distributions.rename(columns={'lon': 'longitude'})

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
                _n_locations = distributions[['latitude', 'longitude']].drop_duplicates().shape[0]
                species_lines.append("")
                species_lines.append(f"Total records: {len(distributions):,}")
                species_lines.append(f"Unique locations (sites/cells): {_n_locations:,}")
                if 'site_name' in distributions.columns:
                    species_lines.append(f"Named sites: {distributions['site_name'].nunique():,}")
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
                retry = get_yes_no("Try another file?", default='yes')
                if not retry:
                    raise ValueError("Cannot proceed without valid species data.")
                file_path = get_valid_path("Occurrence data CSV", must_exist=True, extensions=['.csv'])
                if not file_path.endswith(".csv"):
                    file_path += ".csv"
                spp_params['user_data_file_path'] = file_path

    # Load traits_df and match species names BEFORE showing the domain map, so this
    # appears as part of the data-validation section rather than after the plot.
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

                # If only an AphiaID is provided (no species name), resolve it to a name.
                # (Single pass: species names supplied directly need no lookup and this is instant.)
                if aphia_id and (species_name is None or pd.isna(species_name) or
                                 str(species_name).strip() == ''):
                    try:
                        resolved_name = get_aphia(aphia_id)
                        resolved_name_formatted = resolved_name.replace(' ', '_')
                        traits_df.at[i, 'species_name'] = resolved_name_formatted
                        print(f"  AphiaID {aphia_id} → {resolved_name_formatted}")
                        for trait_dict in traits:
                            if trait_dict.get('aphia_id') == aphia_id and trait_dict.get('species_name') is None:
                                trait_dict['species_name'] = resolved_name_formatted
                    except Exception as e:
                        print(f"  Warning: Could not resolve AphiaID {aphia_id}: {e}")
            display_line("─")

    # Ensure traits_df exists for all paths
    if 'traits_df' not in locals():
        if species_traits_path == 'builtin_database_selection':
            traits_df = load_builtin_traits()
        else:
            traits_df = pd.read_csv(species_traits_path)
        # Add aphia_id column if it doesn't exist
        if 'aphia_id' not in traits_df.columns:
            traits_df['aphia_id'] = None

    # Step 6: Create landmask and show domain map (after data validation and name matching)
    landmask, coastalmask, lonarray, latarray = create_masks(hydro_data, geojson_file_path=geojson_file_path)

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
            "  - This filters data to all ocean occurrences (coastal and offshore) and removes occurrences on land",
            "  - This option is appropriate for offshore and mixed (coastal & offshore) settings",
            "",
            "Option 3: None",
            "  - This uses your data exactly as provided",
            "  - This option is appropriate for pre-cleaned data"
        ])

        while True:
            habitat_choice = input("\nChoose habitat filtering (1/2/3) [default: 1]: ").strip() or '1'

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

    # Flag occurrence species that have no matching entry in the traits file
    if len(clean_spp) > 0:
        trait_names = {standardise_species_name(n) for n in traits_df['species_name'].tolist()}
        occ_names = {standardise_species_name(n) for n in clean_spp['species'].unique().tolist()}
        unmatched = sorted(occ_names - trait_names)
        if unmatched:
            subsection_header("Occurrence species without matching traits")
            print(f"{len(unmatched)} species in your occurrence data have no matching entry in your traits file:")
            for name in unmatched:
                print(f"   - {display_species_name(name)}")
            print("\nStep 5 only post-processes species that have traits, so they will not be included in the final results.")
            print("Names are matched after standardising spaces to underscores, so if a species is unexpectedly missing check the spelling across both the traits and occurrence data files.")
            if get_yes_no("\nDo you want to drop these species from the occurrence data before building the release grids?",
                          default='yes'):
                clean_spp = clean_spp[
                    clean_spp['species'].apply(lambda s: standardise_species_name(s) in trait_names)].reset_index(
                    drop=True)
                print(f"Dropped {len(unmatched)} species. Remaining: {clean_spp['species'].nunique()} species across {len(clean_spp):,} records.")
            else:
                print("Keeping all species. The unmatched species will be released but not post-processed.")

    # Validate species occurrences
    if spp_params.get('data_source') == 'gbif':
        # Fixed low-sample threshold: flag species with fewer than 10 GBIF records (all species
        # are still kept). This replaces an earlier grid-relative threshold that could not read the
        # grid cell count at this point in the workflow, so it always collapsed to 1.
        min_threshold = 10
        info_box([
            "Validating GBIF occurrence data for adequate sample sizes.",
            "",
            f"Species with fewer than {min_threshold} GBIF records may not adequately represent the species' true distribution.",
            "When occurrence data is sparse, grid-mapping may assign the species to very few cells, potentially missing important habitat areas where the species actually occurs.",
            "",
            "All species will still be included regardless of sample size. It is an informational check only.",
        ])
        validation_report = validate_spp_occ(clean_spp, traits_df, min_records=min_threshold)
    else:
        # For user-uploaded data, skip sample size validation
        # (users determine their own sampling adequacy)
        n_species = clean_spp['species'].nunique()
        total_records = len(clean_spp)
        status_message(f"User data loaded: {total_records} records across {n_species} species.", "success")

        # Auto-detect site_name column if present
        if 'site_name' in clean_spp.columns:
            n_sites = clean_spp['site_name'].nunique()
            print(f"\n  Site names detected: {n_sites} unique sites.")
            print()
            print("  You are now running a site-based analysis: Your sites become the labelled units and every figure and output table is reported by site name rather than by grid cell.")
            print()
            print("  See 'Site-Based Analyses' in the User Guide for what this means and how to pick a suitable grid resolution.")
            spp_params['has_site_names'] = True
        else:
            spp_params['has_site_names'] = False
        validation_report = {
            'adequate_sample_species': list(clean_spp['species'].unique()),
            'low_sample_species': [],
            'missing_species': [],
            'overall_success_rate': 100.0
        }

    # Save filtering choice to params for reference
    spp_params['habitat_filtering'] = habitat_type
    with open(params_file, 'w') as file:
        json.dump(spp_params, file, indent=4)

    # Step 8: Offer to save the occurrence data
    # GBIF users have no local copy of the download, so they can save the raw and the cleaned data
    # Users who uploaded their own data are only offered the cleaned version, and only when filtering actually changed it (habitat_type is not 'none')
    is_gbif = spp_params.get('data_source') == 'gbif'
    cleaning_applied = habitat_type != 'none'
    spp_params['saved_files'] = {}

    ## Prompt the user for an output path and save a DataFrame to CSV
    def _save_occ_csv(df, label, default_name):
        """Ask the user for a filename and save the given DataFrame to CSV."""
        path = get_save_path(f"Output path and filename for the {label} data", default=default_name, extensions=['.csv'])
        if not path:
            path = default_name
        if not path.endswith(".csv"):
            path += ".csv"
        df.to_csv(path, index=False)
        print(f"  Saved to: {path}")
        print(f"  Records: {len(df):,} across {df['species'].nunique()} species.")
        return path

    if is_gbif:
        if get_yes_no("Save the raw (unfiltered) GBIF occurrence data to CSV?", default='no'):
            spp_params['saved_files']['raw'] = _save_occ_csv(distributions, 'raw GBIF occurrence',
                                                             'raw_species.csv')
        if cleaning_applied and get_yes_no("Save the cleaned occurrence data to CSV?", default='no'):
            spp_params['saved_files']['cleaned'] = _save_occ_csv(clean_spp, 'cleaned occurrence',
                                                                 'cleaned_species.csv')
    elif cleaning_applied:
        if get_yes_no("Save the cleaned occurrence data to CSV?", default='no'):
            spp_params['saved_files']['cleaned'] = _save_occ_csv(clean_spp, 'cleaned occurrence',
                                                                 'cleaned_species.csv')

    # Record what was saved so the params file documents the run.
    with open(params_file, 'w') as file:
        json.dump(spp_params, file, indent=4)

    # Step 9: Create an occurrence map. For coastal/ocean filtering this is a before/after
    # comparison; for 'none' a single occurrence map is shown (users may still want to see their data).
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
                Longitude of grid cell centre
            - 'lat' : float
                Latitude of grid cell centre
            - 'row_idx' : int
                Row index in original grid array
            - 'col_idx' : int
                Column index in original grid array

        Notes
        -----
        - Grid IDs are assigned sequentially starting from 1
        - Only cells where coastalmask==1 are included
        - Row and column indices are preserved for mapping back to original arrays
        - Coordinates represent cell centres, not corners
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
    specified polygon boundary. Useful for defining specific study extents or
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
        Polygon defining the study extent boundary. Vertices should be in
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
def save_grids(release_grids=None, settlement_grids=None, full_grid=None, grid_df_for_mapping=None, full_grid_df=None):
    """
    Save grid dataframes to CSV files for reproducibility and inspection.

    Two prompts:
    1. Export domain grids (full hydro domain + study extent subset)
    2. Export per-species release/settlement grids
    """
    global project_dirs

    # Determine default directory
    if 'project_dirs' in globals() and project_dirs:
        default_dir = project_dirs.get('grids', '.')
    else:
        default_dir = '.'

    # Prompt 1: Domain and study extent grids
    if grid_df_for_mapping is not None or full_grid_df is not None:
        print("\n  Two grid files are available for export:")
        if full_grid_df is not None:
            print(f"    1. Full hydrodynamic extent grid: {len(full_grid_df)} cells")
            print(f"       Every cell in your hydrodynamic domain (including land).")
            print(f"       Useful for understanding the broader model domain or custom spatial analysis.")
        if grid_df_for_mapping is not None:
            print(f"    2. Study extent grid: {len(grid_df_for_mapping)} cells")
            print(f"       The cells within your study extent polygon focused on your habitat (not including land cells).")
            print(f"       This is the grid used for all analyses.")

        if get_yes_no("\nExport these grids to CSV?", default='no'):
            output_dir = get_valid_dir("  Output directory", default=default_dir)
            os.makedirs(output_dir, exist_ok=True)

            if full_grid_df is not None:
                file_path = os.path.join(output_dir, 'grid_full_domain.csv')
                full_grid_df.to_csv(file_path, index=False)
                print(f"  Saved: {file_path}")
                print(f"    Columns: grid_id, longitude, latitude, is_land, is_coastal")
                print(f"    Rows: {len(full_grid_df)} cells (entire hydrodynamic domain)")

            if grid_df_for_mapping is not None:
                file_path = os.path.join(output_dir, 'grid_study_area.csv')
                grid_df_for_mapping.to_csv(file_path, index=False)
                print(f"  Saved: {file_path}")
                print(f"    Columns: grid_id, longitude, latitude")
                print(f"    Rows: {len(grid_df_for_mapping)} cells (filtered to your study extent polygon)")

    # Prompt 2: Per-species grids
    has_release = release_grids and len(release_grids) > 0
    has_settlement = settlement_grids and len(settlement_grids) > 0

    if has_release or has_settlement:
        n_release = len(release_grids) if has_release else 0
        n_settlement = len(settlement_grids) if has_settlement else 0
        print(f"\n  Species-specific grids available:")
        if has_release:
            print(f"    Release grids: {n_release} species")
        if has_settlement:
            print(f"    Settlement grids: {n_settlement} species")
        print(f"    These show which grid cells each species releases from/settles to.")

        if get_yes_no("Export species-specific grids to CSV?", default='no'):
            output_dir = get_valid_dir("  Output directory", default=default_dir)
            os.makedirs(output_dir, exist_ok=True)

            if has_release:
                n_saved = 0
                for species_name, df in release_grids.items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        safe_name = species_name.replace(' ', '_').replace('/', '_')
                        file_name = os.path.join(output_dir, f"{safe_name}_release_grid.csv")
                        df.to_csv(file_name, index=False)
                        n_saved += 1
                print(f"  Release grids saved: {n_saved} species to {output_dir}")

            if has_settlement:
                n_saved = 0
                for species_name, df in settlement_grids.items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        safe_name = str(species_name).replace(' ', '_').replace('/', '_')
                        file_name = os.path.join(output_dir, f"{safe_name}_settlement_grid.csv")
                        df.to_csv(file_name, index=False)
                        n_saved += 1
                print(f"  Settlement grids saved: {n_saved} species to {output_dir}")
# endregion

## region Setup Particle Release and Settlement Strategy (wrapper function)
def setup_release_settlement(coastalmask, landmask, clean_spp, spp_params, lonarray, latarray, traits=None):
    """
    Configure species-specific release and settlement grids for connectivity modelling.

    This is a central function that orchestrates grid creation, species assignment, and polygon filtering to produce ready-to-use release and settlement locations for each species in the connectivity simulation.

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
            Path to study extent polygon
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
        Filtered grid within study extent polygon
    study_polygon : shapely.geometry.Polygon
        study extent boundary polygon

    Workflow
    --------
    1. Create full coastal grid from coastalmask
    2. Map species occurrences to nearest grid cells
    3. Load and read study extent polygon
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

    # Build site_name lookup if user provided named sites
    # Maps grid_id → site_name by snapping each user site to its nearest grid cell
    if spp_params.get('has_site_names') and len(clean_spp) > 0 and 'site_name' in clean_spp.columns:
        grid_coords = grid_df_for_mapping[['latitude', 'longitude']].values
        tree = cKDTree(grid_coords)
        name_acc = defaultdict(list)
        for _, row in clean_spp.drop_duplicates(subset=['site_name']).iterrows():
            if pd.notna(row.get('latitude')) and pd.notna(row.get('longitude')):
                _, idx = tree.query([row['latitude'], row['longitude']])
                grid_id = int(grid_df_for_mapping.iloc[idx]['grid_id'])
                name_acc[grid_id].append(str(row['site_name']))
        site_lookup = {gid: ' / '.join(names) for gid, names in name_acc.items()}
        spp_params['site_name_lookup'] = site_lookup
        print(f"\nSite names: {len(site_lookup)} grid cells labelled")
        for gid, names in name_acc.items():
            if len(names) > 1:
                joined = "' and '".join(names)
                print(f"   \nNote: Sites '{joined}' fall in the same grid cell. Your outputs will report their results together.")

    # Step 4: Ask user to choose particle release strategy
    subsection_header("Propagule Release Strategy")

    # Build release options
    data_source_text = "GBIF" if spp_params.get('data_source') == 'gbif' else "your uploaded data"
    release_lines = [
        "Release and settlement sites are chosen independently in the next two steps.",
        "Use Step 2 to choose your general study area and occurrence data for release with the option to use different occurrence data for settlement."
        "",
        "Choose where particles are released from.",
        "",
        "Option 1: Everywhere in your study extent",
        f"  - Release from all {all_sites_description} in your study extent",
        "  - Does not use species occurrence data",
        "  - Best for maximum-dispersal or data-poor scenarios",
        f"  - Total release sites: {len(all_sites_grid)}",
    ]
    if len(clean_spp) > 0:
        unique_species = clean_spp['species'].nunique()
        temp_meshgrid, temp_grids = map_spp_to_grid(grid_df_for_mapping, clean_spp)
        total_unique_cells = len(temp_meshgrid)
        release_lines.extend([
            "",
            "Option 2: All sites in your occurrence data (pooled, not species-specific)",
            f"  - Release from every site in {data_source_text}, combined across all species",
            "   - Best when you want to release using habitat suitability data",
            f"  - Total release sites: {total_unique_cells}",
            "",
            "Option 3: Species-specific habitat in your study extent (uses the occurrence data selected in Step 2)",
            "  - Release each species only from its own suitable habitat",
            "  - Best for trait-based analyses",
            f"  - Species distributions from {data_source_text}",
            f"  - Total release sites: {total_unique_cells} (across {unique_species} species)",
        ])
        if spp_params.get('has_site_names'):
            release_lines.append(
                "  - Your Step 2 data includes site names, so the names carry through to all outputs")

    info_box(release_lines)

    # Species to model for the release/settlement grids. Strategies that do NOT use per-species
    # occurrence (all sites, pooled) key by the full set of TRAIT species, so every modelled species
    # is released/settled even if it has no occurrence records - matching automatic mode. The
    # species-specific strategy keys by occurrence species via map_spp_to_grid (unchanged). Falls
    # back to occurrence species when traits are not provided (e.g. a manual call without traits).
    if traits:
        modelled_species = [standardise_species_name(t['species_name']) for t in traits]
    elif len(clean_spp) > 0:
        modelled_species = list(clean_spp['species'].unique())
    else:
        modelled_species = ['all_species']

    if len(clean_spp) > 0:
        release_valid = "1/2/3"
        release_default = '3'
    else:
        release_valid = "1"
        release_default = '1'

    while True:
        release_choice = input(
            f"\nChoose release strategy ({release_valid}) [default: {release_default}]: ").strip() or release_default

        if release_choice == '1':
            # Release from all sites in the study extent (does not use occurrence data)
            meshgrid = all_sites_grid
            species_names = modelled_species
            study_area_df = study_area_df_filtered.copy()
            release_grids = {sp: study_area_df.copy() for sp in species_names}
            print(f"  Release from all {len(study_area_df)} study extent sites")
            break

        elif release_choice == '2' and len(clean_spp) > 0:
            # Pooled: every site in the occurrence data, combined across species (pooled, not species-specific)
            meshgrid, _per_species = map_spp_to_grid(grid_df_for_mapping, clean_spp)
            pooled_df = pd.concat(_per_species.values(), ignore_index=True)[
                ['grid_id', 'longitude', 'latitude']].drop_duplicates(subset=['grid_id']).reset_index(drop=True)
            species_names = modelled_species
            release_grids = {sp: pooled_df.copy() for sp in species_names}
            print(f"  Pooled release from all {len(pooled_df):,} occurrence sites, applied to {len(species_names)} species")
            break

        elif release_choice == '3' and len(clean_spp) > 0:
            # Species-specific: each species released only from its own occurrence sites
            meshgrid, release_grids = map_spp_to_grid(grid_df_for_mapping, clean_spp)
            print(f"  Created species-specific release grids for {len(release_grids)} species")
            print(f"  Total unique release sites: {len(meshgrid):,}")
            break

        else:
            print(f"  Invalid choice. Enter {release_valid}.")

    # Save release choice (the JSON config is written after the settlement step below)
    release_strategy_map = {'1': 'all_sites', '2': 'pooled_sites', '3': 'species_habitat'}
    spp_params['release_strategy'] = release_strategy_map.get(release_choice)

    # Step 5: Ask user to choose particle settlement strategy
    subsection_header("Propagule Settlement Strategy")

    # Build settlement options info. Options 2 and 3 use the Step 2 occurrence data, so they
    # only appear when occurrence data is present. Option 4 (a different dataset) is always available.
    settlement_lines = [
        "Choose where particles can settle.",
        "",
        "Option 1: Everywhere in your study extent",
        f"  - Settle at all {all_sites_description} in your study extent",
        "  - Does not use species occurrence data",
        "  - Best for maximum-dispersal or data-poor scenarios",
        f"  - Total settlement sites: {len(all_sites_grid)}",
    ]
    if len(clean_spp) > 0:
        unique_species = clean_spp['species'].nunique()
        temp_meshgrid, temp_grids = map_spp_to_grid(grid_df_for_mapping, clean_spp)
        total_unique_cells = len(temp_meshgrid)
        settlement_lines.extend([
            "",
            "Option 2: All sites in your occurrence data (pooled, not species-specific)",
            f"  - Settle at every site in {data_source_text}, combined across all species",
            "  - Best when you want to settle using habitat suitability data",
            f"  - Total settlement sites: {total_unique_cells}",
            "",
            "Option 3: Species-specific habitat in your study extent (uses the occurrence data selected in Step 2)",
            "  - Each species can only settle in its own suitable habitat",
            "  - Best for trait-based analyses",
            f"  - Species distributions from {data_source_text}",
            f"  - Total settlement sites: {total_unique_cells} (across {unique_species} species)",
        ])
    settlement_lines.extend([
        "",
        "Option 4: A different occurrence dataset",
        "  - Settle using a separate occurrence or habitat-suitability CSV (different from release)",
        "  - Required columns: species, latitude, longitude",
    ])

    if len(clean_spp) > 0:
        settlement_valid = "1/2/3/4"
        settlement_default = '3'
    else:
        settlement_valid = "1/4"
        settlement_default = '1'

    info_box(settlement_lines)

    while True:
        settlement_choice = input(
            f"\nChoose settlement strategy ({settlement_valid}) [default: {settlement_default}]: ").strip() or settlement_default

        if settlement_choice == '1':
            # Settle at all sites in the study extent (does not use occurrence data)
            species_names = modelled_species
            study_area_df = study_area_df_filtered.copy()
            settlement_grids = {sp: study_area_df.copy() for sp in species_names}
            print(f"Settlement at all {len(study_area_df)} study extent sites")
            break

        elif settlement_choice == '2' and len(clean_spp) > 0:
            # Pooled: every site in the occurrence data, combined across species
            _, _per_species = map_spp_to_grid(grid_df_for_mapping, clean_spp)
            pooled_df = pd.concat(_per_species.values(), ignore_index=True)[
                ['grid_id', 'longitude', 'latitude']].drop_duplicates(subset=['grid_id']).reset_index(drop=True)
            species_names = modelled_species
            settlement_grids = {sp: pooled_df.copy() for sp in species_names}
            print(f"  Pooled settlement at all {len(pooled_df):,} occurrence sites, applied to {len(species_names)} species")
            break

        elif settlement_choice == '3' and len(clean_spp) > 0:
            # Species-specific: each species settles only in its own occurrence sites
            _, settlement_grids = map_spp_to_grid(grid_df_for_mapping, clean_spp)
            print(f"Created species-specific settlement grids for {len(settlement_grids)} species")
            break

        elif settlement_choice == '4':
            print("\nSelected: A different occurrence dataset")
            print("\nUpload your own species occurrence or habitat suitability data.")
            print("Required format: CSV with columns 'species', 'latitude', 'longitude'.")

            while True:
                custom_file_path = get_valid_path("Enter the input path and file name", must_exist=True,
                                                  extensions=['.csv'])

                try:
                    custom_settlement_data = pd.read_csv(custom_file_path)
                    custom_settlement_data = auto_convert_coordinates(custom_settlement_data)

                    # Verify required columns
                    required_columns = {'species', 'latitude', 'longitude'}
                    missing_columns = required_columns - set(custom_settlement_data.columns)

                    if missing_columns:
                        print(f"Error: Missing required columns: {', '.join(missing_columns)}")
                        print(f"Your columns: {', '.join(custom_settlement_data.columns)}")
                        retry = get_yes_no("Try another file?", default='yes')
                        if not retry:
                            raise ValueError("Cannot proceed without valid settlement data.")
                        continue

                    # Re-validate the uploaded settlement species against the traits file, exactly as the release dataset is validated, so a settlement CSV with species absent from the traits file does not silently build grids that Step 5 can never post-process.
                    _traits_records = traits.to_dict('records') if isinstance(traits, pd.DataFrame) else traits
                    _trait_names = {standardise_species_name(t['species_name']) for t in _traits_records}
                    _settle_names = {standardise_species_name(n) for n in
                                     custom_settlement_data['species'].unique().tolist()}
                    _unmatched_settle = sorted(_settle_names - _trait_names)
                    if _unmatched_settle:
                        subsection_header("Settlement species without matching traits")
                        print(
                            f"{len(_unmatched_settle)} species in your settlement data have no matching entry in your traits file:")
                        for _name in _unmatched_settle:
                            print(f"   - {display_species_name(_name)}")
                        print("\nStep 5 only post-processes species that have traits, so they will not be included in the final results.")
                        print("Names are matched after standardising spaces to underscores, so if a species is unexpectedly missing check the spelling across both the traits and settlement occurrence data files.")
                        if get_yes_no(
                                "\nDo you want to drop these species from the settlement data before building the settlement grids?",
                                default='yes'):
                            custom_settlement_data = custom_settlement_data[
                                custom_settlement_data['species'].apply(
                                    lambda s: standardise_species_name(s) in _trait_names)].reset_index(drop=True)
                            print(
                                f"Dropped {len(_unmatched_settle)} species. Remaining: {custom_settlement_data['species'].nunique()} species across {len(custom_settlement_data):,} records.")
                        else:
                            print(
                                "Keeping all species. The unmatched species get settlement grids but are never post-processed.")

                    species_names = clean_spp['species'].unique() if len(clean_spp) > 0 else ['all_species']
                    print("\nHow should this dataset be used?")
                    print("  1. All sites in your dataset (pooled, not species-specific)")
                    print("  2. Species-specific habitat in your study extent (uses your occurrence data)")
                    sub = input("Choose (1/2) [default: 2]: ").strip() or '2'
                    if sub == '1':
                        _, _per = map_spp_to_grid(grid_df_for_mapping, custom_settlement_data)
                        pooled_df = pd.concat(_per.values(), ignore_index=True)[
                            ['grid_id', 'longitude', 'latitude']].drop_duplicates(subset=['grid_id']).reset_index(
                            drop=True)
                        settlement_grids = {sp: pooled_df.copy() for sp in species_names}
                        print(f"Pooled settlement at all {len(pooled_df):,} sites in this dataset, applied to {len(species_names)} species")
                    else:
                        _, settlement_grids = map_spp_to_grid(grid_df_for_mapping, custom_settlement_data)
                        print(f"Created species-specific settlement grids for {len(settlement_grids)} species")
                    break

                except Exception as e:
                    print(f"Error reading file: {e}")
                    retry = input("Try another file? (yes/no) [default: no]: ").strip().lower() or 'no'
                    if retry not in ['yes', 'y']:
                        raise ValueError("Cannot proceed without valid settlement data.")

            break

        else:
            print(f"Invalid choice. Enter {settlement_valid}.")

    # Save settlement choice
    settlement_strategy_map = {'1': 'all_sites', '2': 'pooled_sites', '3': 'species_habitat', '4': 'custom_habitat'}
    spp_params['settlement_strategy'] = settlement_strategy_map.get(settlement_choice)

    # Runtime objects (DataFrames/arrays) are kept in memory but cannot be written to JSON, so exclude them.
    _runtime_only_keys = ('uploaded_sites_data', 'uploaded_sites_meshgrid', 'uploaded_sites_release_grids')
    spp_params_json = {k: v for k, v in spp_params.items() if k not in _runtime_only_keys}
    with open(os.path.join(project_dirs['config'], 'spp_occ_params.json'), 'w') as file:
        json.dump(spp_params_json, file, indent=4)

    # Step 6: Optionally save release and settlement grids as CSV
    save_grids(release_grids, settlement_grids, full_grid, grid_df_for_mapping, full_grid_df)

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
### Creating Custom Kernels
# Kernel functions defined at module level to avoid Parcels AST scope issues
# (nested functions can cause "Cannot convert 'pset' to C-code" errors)
from parcels import StatusCode

## OceanParcels kernel: delete particles that move out of bounds
def _DeleteErrorParticle(particle, fieldset, time):
    """Delete particles whose state is ErrorOutOfBounds to prevent simulation crashes."""
    if particle.state == StatusCode.ErrorOutOfBounds:
        particle.delete()

## OceanParcels kernel: accumulate total distance travelled
def _TotalDistance(particle, fieldset, time):
    """Calculate the distance travelled by the particle."""
    lat_dist = (particle.lat - particle.prev_lat) * 1.11e2
    lon_dist = (
            (particle.lon - particle.prev_lon)
            * 1.11e2
            * math.cos(particle.lat * math.pi / 180)
    )
    particle.distance += math.sqrt(math.pow(lon_dist, 2) + math.pow(lat_dist, 2))
    particle.prev_lon = particle.lon
    particle.prev_lat = particle.lat

## OceanParcels kernel: flag whether a particle is currently on land
def _Sample_land(particle, fieldset, time):
    """Sample if the particle is on land."""
    particle.on_land = fieldset.landmask[time, particle.depth, particle.lat, particle.lon]

## OceanParcels kernel: record the grid cell ID at the particle's current location
def _Grid_id(particle, fieldset, time):
    """Sample the grid ID at the particle location."""
    particle.grid_id = fieldset.grid_ids[time, particle.depth, particle.lat, particle.lon]

## OceanParcels kernel: nudge land-stranded particles back into the ocean
def _Reflect(particle, fieldset, time):
    """Update particle position if it is on land."""
    if particle.on_land == 1:
        (u_land, v_land) = fieldset.UV_reflect[
            time, particle.depth, particle.lat, particle.lon]
        particle.lon += u_land * particle.dt
        particle.lat += v_land * particle.dt
        particle.on_land = 0


## Combine all OceanParcels kernels into a single execution kernel
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
    try:
        kernel = pset.Kernel(AdvectionRK4)
        kernel = kernel + pset.Kernel(_TotalDistance)
        kernel = kernel + pset.Kernel(DiffusionUniformKh)
        kernel = kernel + pset.Kernel(_Sample_land)
        kernel = kernel + pset.Kernel(_Reflect)
        kernel = kernel + pset.Kernel(_Grid_id)
        kernel = kernel + pset.Kernel(_DeleteErrorParticle)
        return kernel
    except Exception as e:
        print(f"Warning: Kernel creation issue: {e}")
        print("Attempting alternative kernel combination method...")
        kernels = [
            pset.Kernel(AdvectionRK4),
            pset.Kernel(_TotalDistance),
            pset.Kernel(DiffusionUniformKh),
            pset.Kernel(_Sample_land),
            pset.Kernel(_Reflect),
            pset.Kernel(_Grid_id),
            pset.Kernel(_DeleteErrorParticle)
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
        Configured ParticleSet
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
            print("Using default: Weekly")
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
    print(f"Per location, total particles released: 100 × {num_intervals} = {100 * num_intervals:,} particles")
    print(f"All locations, total particles released: {100 * num_intervals * num_locations:,} particles")

    default_npart = 100
    npart = get_positive_integer(f"\nEnter the number of particles per location per cohort", default=default_npart, min_val=1)

    # Step 4: Calculate totals and confirm
    particles_per_location = npart * num_intervals
    total_particles = particles_per_location * num_locations

    # Step 5: Horizontal diffusivity , collected before the summary so it appears in the box
    subsection_header("Horizontal Diffusivity")
    info_box([
        "Horizontal diffusivity represents small-scale random motion caused by turbulence.",
        "It applies evenly to all particles and adds a random-walk component to their trajectories.",
        "A value of 1 m²/s is a common default for shelf-sea simulations.",
        "Set to 0 to disable random-walk diffusion entirely."
    ])
    default_kh = 1
    while True:
        try:
            kh_input = input(f"\nEnter horizontal diffusivity (kh) in m²/s [default: {default_kh}]: ")
            kh = float(kh_input) if kh_input else default_kh
            if kh < 0:
                print("Error: Diffusivity must be positive.")
                continue
            if kh > 1000:
                print("Warning: Very high diffusivity (>1000). Are you sure? (yes/no)")
                _kh_confirm = input().strip().lower()
                if _kh_confirm not in ['yes', 'y']:
                    continue
            break
        except ValueError:
            print("Error: Enter a valid number.")

    # Step 6: Summary of all setup parameters (including kh)
    data_box("The Particle Setup", [
        f"Release cohorts: {num_intervals}",
        f"Interval: Every {interval.days} days",
        f"Release locations: {num_locations}",
        f"Particles per location per cohort: {npart}",
        f"Total particles per location (all cohorts): {particles_per_location}",
        f"Total particles released (all cohorts x locations): {total_particles}",
        f"Adjusted runtime: {adjusted_runtime.days} days",
        f"Horizontal diffusivity (kh): {kh} m²/s"
    ])

    confirm = get_yes_no("\nProceed with these settings?", default='yes')
    if not confirm:
        print("Cancelled. Please run workflow_step(3) again to reconfigure.")
        return None, None, None, None

    # Step 7: Setup particle arrays
    repeatdt = interval

    lat = list(itertools.chain.from_iterable(itertools.repeat(x, npart) for x in lats))
    lon = list(itertools.chain.from_iterable(itertools.repeat(x, npart) for x in lons))

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

    # Save pset configuration for session restore (pset itself can't be pickled)
    global project_dirs
    try:
        pset_config = {
            'release_interval_days': interval.days,
            'npart': npart,
            'kh': kh,
        }
        config_path = os.path.join(project_dirs['config'], 'pset_config.json')
        with open(config_path, 'w') as f:
            json.dump(pset_config, f, indent=4)
    except Exception:
        pass

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

    ## Store the operation name label used in printed output
    def __init__(self, operation_name):
        """Initialise the monitor with a human-readable operation name."""
        self.operation_name = operation_name

    ## Record baseline memory usage and announce the operation
    def __enter__(self):
        """Collect garbage, record starting memory usage, and print the operation name."""
        gc.collect()
        self.start_memory = psutil.virtual_memory().percent
        print(f"{self.operation_name}...")
        return self

    ## Collect garbage and warn if memory is critically high after the operation
    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()
        end_memory = psutil.virtual_memory().percent

        # Only warn if critically high - no other output
        if end_memory > 92:
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
    str
        Path to a directory of parquet chunk files (chunk_*.parquet) holding the processed trajectory data. Each chunk has columns:
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
    5. Write each processed chunk to a parquet file on disk
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

        # Create temp directory for parquet chunks
    zarr_dir = os.path.dirname(zarr_path)
    chunks_dir = os.path.join(zarr_dir, '_temp_chunks')
    os.makedirs(chunks_dir, exist_ok=True)

    # Process in chunks , write each to disk instead of keeping in memory
    n_chunks = (n_trajectories + chunk_size - 1) // chunk_size
    chunk_files = []

    for chunk_idx in range(0, n_trajectories, chunk_size):
        chunk_end = min(chunk_idx + chunk_size, n_trajectories)
        current_chunk = chunk_end - chunk_idx
        chunk_num = (chunk_idx // chunk_size) + 1

        # In-place progress update (overwrites the same line)
        print(f"\rImporting chunk {chunk_num}/{n_chunks} ({current_chunk:,} trajectories)...    ", end='', flush=True)

        # Load just this chunk
        chunk_ds = ds.isel(trajectory=slice(chunk_idx, chunk_end))
        chunk_df = chunk_ds.to_dataframe().reset_index()

        # Drop rows where essential columns are NaN
        chunk_df = chunk_df.dropna(subset=['trajectory', 'time', 'lat', 'lon'])

        # Write to parquet on disk (frees memory)
        chunk_path = os.path.join(chunks_dir, f'chunk_{chunk_num:04d}.parquet')
        chunk_df.to_parquet(chunk_path, index=False, engine='pyarrow')
        chunk_files.append(chunk_path)

        # Clean up immediately , data is safely on disk
        del chunk_ds, chunk_df
        gc.collect()

    # Chunks stay on disk for memory-efficient per-chunk processing.
    # Use load_traj_from_chunks(chunks_dir) if you need the full DataFrame later.
    print()  # newline after the in-place progress
    total_rows = sum(
        pd.read_parquet(f, engine='pyarrow', columns=['trajectory']).shape[0]
        for f in chunk_files
    )
    print(f"Simulation data processed successfully ({total_rows:,} trajectory observations).")

    return chunks_dir

## Load all parquet trajectory chunks into one combined DataFrame
def load_traj_from_chunks(chunks_dir):
    """
    Load all parquet chunks into a single DataFrame.

    Use this only when the full trajectory DataFrame is needed (e.g. for dispersal animation).
    For normal processing, use per-chunk functions instead.

    Parameters
    ----------
    chunks_dir : str
        Path to directory containing parquet chunk files.

    Returns
    -------
    pandas.DataFrame
        Complete trajectory data.
    """
    if not os.path.isdir(chunks_dir):
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")

    chunk_files = sorted(glob(os.path.join(chunks_dir, 'chunk_*.parquet')))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {chunks_dir}")

    print(f"Loading {len(chunk_files)} chunks into memory...")
    with MemoryMonitor("Loading all chunks"):
        traj_df = pd.read_parquet(chunks_dir, engine='pyarrow')
    print(f"Loaded {len(traj_df):,} rows.")
    return traj_df

## Load only obs=0 (release) rows from chunks for memory-efficient mortality calculation
def load_traj_starts_from_chunks(chunks_dir):
    """
    Load only obs=0 rows from all chunks which is the release timestamps.

    Much more memory-efficient than loading full trajectories.
    Used for mortality calculation in connectivity metrics.

    Parameters
    ----------
    chunks_dir : str
        Path to directory containing parquet chunk files.

    Returns
    -------
    pandas.DataFrame
        Only the obs=0 rows from all chunks.
    """
    chunk_files = sorted(glob(os.path.join(chunks_dir, 'chunk_*.parquet')))
    starts = []
    for cf in chunk_files:
        chunk = pd.read_parquet(cf, engine='pyarrow')
        starts.append(chunk[chunk['obs'] == 0].copy())
        del chunk
    return pd.concat(starts, ignore_index=True) if starts else pd.DataFrame()

# endregion

## region Setup Functions
### Release ID/Cohort Assignment
def assign_release_id(chunks_dir, interval):
    """
    Assigns a release_id to each trajectory based on the first observation time (obs=0).
    Works per-chunk on disk for memory efficiency.

    Parameters:
    -----------
    chunks_dir : str
        Path to directory containing parquet chunk files.
    interval : int
        Interval between releases in days.

    Returns:
    --------
    chunks_dir : str
        Same directory path , chunk files are updated in-place with 'release_id' column.

    Notes:
    ------
    - Two-pass algorithm: Pass 1 finds global earliest release, Pass 2 assigns IDs.
    - Each chunk file is read, updated, and rewritten individually.
    - release_id starts at 1 (cohort 1, 2, 3, ...).
    """
    chunk_files = sorted(glob(os.path.join(chunks_dir, 'chunk_*.parquet')))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {chunks_dir}")

    n_chunks = len(chunk_files)

    # Pass 1: Find global earliest release time across all chunks
    global_earliest = None
    for cf in chunk_files:
        chunk = pd.read_parquet(cf, engine='pyarrow', columns=['trajectory', 'obs', 'time'])
        obs0 = chunk[chunk['obs'] == 0]['time']
        if not pd.api.types.is_datetime64_any_dtype(obs0):
            obs0 = pd.to_datetime(obs0)
        chunk_min = obs0.min()
        if global_earliest is None or chunk_min < global_earliest:
            global_earliest = chunk_min
        del chunk, obs0
        gc.collect()

    global_earliest = pd.Timestamp(global_earliest)

    # Pass 2: Assign release_id per chunk and rewrite files
    total_releases = set()
    for i, cf in enumerate(chunk_files):
        print(f"\r  Chunk {i + 1}/{n_chunks}...    ", end='', flush=True)
        chunk = pd.read_parquet(cf, engine='pyarrow')

        first_obs = chunk[chunk['obs'] == 0][['trajectory', 'time']].copy()
        if not pd.api.types.is_datetime64_any_dtype(first_obs['time']):
            first_obs['time'] = pd.to_datetime(first_obs['time'])

        first_obs['days_since_start'] = (
            first_obs['time'] - global_earliest
        ).dt.total_seconds() / (24 * 3600)
        first_obs['release_id'] = (first_obs['days_since_start'] / interval).round().astype(int) + 1

        release_lookup = first_obs.set_index('trajectory')['release_id']
        chunk['release_id'] = chunk['trajectory'].map(release_lookup)

        total_releases.update(first_obs['release_id'].unique())

        chunk.to_parquet(cf, index=False, engine='pyarrow')
        del chunk, first_obs, release_lookup
        gc.collect()

    print()  # newline after in-place progress
    data_box("Release Cohort Assignment", [
        f"Release cohorts assigned: {len(total_releases)}",
        f"Chunks processed: {n_chunks}"
    ])

    return chunks_dir

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
        Configured ParticleSet containing:
        - Initial particle positions
        - Release timing information
        - Species assignments
        - Custom particle variables
    kernels : parcels.Kernel or list of parcels.Kernel
        Compiled kernel(s) defining particle behaviour:
        - Advection (AdvectionRK4)
        - Diffusion
        - Boundary conditions
        - Settlement behaviour
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
    tuple
        (output_path, chunks_dir) , the path to the Zarr output file, and the path to the directory of parquet chunk files holding the processed trajectory data.

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

        output_path = get_save_path("Output path and file name for the simulation output", default=default_sim_path, extensions=['.zarr']) or default_sim_path
        if not output_path.endswith(".zarr"):
            output_path += ".zarr"

        # Step 2: Get output settings from user
        subsection_header("Output Frequency")
        info_box([
            "In a particle simulation, particles are advected at regular timesteps (e.g. every hour).",
            "You can save output less frequently than the simulation timestep to reduce file size.",
            "",
            "For example: Timestep = 1 hour, Output frequency = 6 hours.",
            "  - Higher timestep = more detail but slower simulation runtime",
            "  - Lower timestep = less detail but faster simulation runtime"
            "  - Higher frequency = more temporal detail but larger files",
            "  - Lower frequency = less detail but smaller files",
        ])

        dt_minutes = int(input("Enter time step (dt) in minutes [default: 60]: ") or "60")
        output_dt_hours = int(input("Enter output frequency in hours [default: 24]: ") or "24")

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
        print(f"Processing will use chunks of {config['chunk_size']:,} trajectories to prevent memory issues.")

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

        # Memory estimate: will the output fit in RAM?
        # Particles are released once per cohort via repeatdt
        # The initial pset is only the first cohort, so multiply by the cohort count to match total_particles in the Particle Setup box
        _n_release_events = int(adjusted_runtime / interval) if interval else 1
        n_particles = len(pset) * _n_release_events
        n_output_steps = int((adjusted_runtime * 24) / output_dt_hours) + 1
        estimated_rows = n_particles * n_output_steps
        bytes_per_row = 80  # ~10 columns × 8 bytes each
        estimated_gb = (estimated_rows * bytes_per_row) / (1024 ** 3)

        data_box("Estimated output size", [
            f"Total particles: {n_particles:,}",
            f"Output timesteps: {n_output_steps}",
            f"Estimated rows: {estimated_rows:,}",
            f"Estimated memory: {estimated_gb:.1f} GB"
        ])

        if estimated_gb > available_gb * 0.4:
            print("\n  Warning: The estimated output may be too large for your system memory.")
            print(f"  Estimated size: {estimated_gb:.1f} GB | Available RAM: {available_gb:.1f} GB")
            print("\n  Options to reduce output size (re-run from indicated step):")
            print(f"    - Increase output frequency: Re-run Step 4 (current: {output_dt_hours}h → try 24h)")
            print(f"    - Fewer particles per cell: Re-run from Step 3")
            print(f"    - Coarser grid resolution: Re-run from Step 1")
            print(f"    - Smaller study domain: Re-run from Step 1\n")

            proceed = get_yes_no("Continue with current settings?", default='no')
            if not proceed:
                print("\nSimulation cancelled. Adjust settings and re-run Step 4.")
                print("To change grid resolution, re-run Step 1 with a larger grid cell size.")
                print("To change particle count, re-run Step 3.")
                return None, None

        # Step 4: Run the simulation with memory monitoring
        print("\n  Note: The progress bar may stay at 0% for 1-2 minutes while the simulation initialises.")
        print("  This is normal. Please wait for it to start running.\n")
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
                output_file=output_file,
                verbose_progress=True
            )

        # Step 5: Load results with chunked approach
        subsection_header("Processing Simulation Results")
        file_size_gb = get_zarr_size(output_path)
        info_box([
            f"Output file size: {file_size_gb:.2f} GB",
            f"Trajectories will be loaded in chunks of {config['chunk_size']:,} for memory efficiency"
        ])
        chunks_dir = process_sim_results(output_path, config['chunk_size'], adjusted_runtime)

        subsection_header("Assigning Release Cohorts")
        chunks_dir = assign_release_id(chunks_dir, interval)

        status_message("Simulation complete.", "success")
        data_box("Simulation Output", [
            f"Trajectory chunks: {chunks_dir}"
        ])

        return output_path, chunks_dir

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

    Memory-safe: never modifies the original traj DataFrame.

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

    n_species = len(traits)
    print(f"\nCreating unique trajectory IDs for {n_species} species...")

    # Convert time if needed (in-place, no extra memory)
    if not pd.api.types.is_datetime64_any_dtype(traj['time']):
        traj['time'] = pd.to_datetime(traj['time'])

    # Pre-compute release times from obs==0 rows only (small: one row per trajectory)
    release_rows = traj[traj['obs'] == 0][['trajectory', 'time']].copy()
    release_rows.rename(columns={'time': 'release_time'}, inplace=True)

    # Get offset for unique IDs
    max_traj_id = traj['trajectory'].max()
    offset_multiplier = 10 ** (len(str(max_traj_id)) + 1)

    spawning_dfs = {}

    for i, species_traits in enumerate(traits):
        species_name = standardise_species_name(species_traits['species_name'])

        # Get spawning window for this species
        spawn_start = pd.to_datetime(species_traits['spawning_start'], dayfirst=True)
        spawn_end = pd.to_datetime(species_traits['spawning_end'], dayfirst=True)

        # Find trajectories released during this species' spawning window
        in_window = release_rows[
            (release_rows['release_time'] >= spawn_start) &
            (release_rows['release_time'] <= spawn_end)
        ]['trajectory'].unique()

        if len(in_window) > 0:
            # Extract only matching trajectories (never modifies original traj)
            species_data = traj[traj['trajectory'].isin(in_window)].copy()
            species_data['original_trajectory'] = species_data['trajectory']
            species_data['trajectory'] = species_data['original_trajectory'] + (i * offset_multiplier)
            species_data['species'] = species_name

            spawning_dfs[species_name] = species_data
            print(f"   {species_name}: {len(in_window)} trajectories")

    del release_rows
    gc.collect()

    if spawning_dfs:
        print(f"Species assignment complete: {len(spawning_dfs)} species with trajectories.")
    else:
        print("No trajectories found in any spawning windows!")

    return spawning_dfs

### Planktonic larval duration filter and settlement habitat suitability filter
def _build_buffer_to_site(species_settlement, grid_ids, buffer):
    """
    For every buffer cell surrounding each real settlement site, map the buffer cell
    back to its nearest real settlement site using a cKDTree on grid positions.

    Parameters
    ----------
    species_settlement : set of int
        Grid IDs of the actual (non-buffered) settlement cells.
    grid_ids : numpy.ndarray
        2D array of grid IDs for the study extent.
    buffer : int
        Number of cells to expand outward from each settlement site.

    Returns
    -------
    dict
        {buffer_cell_grid_id: attributed_settlement_site_grid_id}
        Empty dict when buffer == 0 or species_settlement is empty.
    """
    if buffer == 0 or not species_settlement:
        return {}
    _site_pos, _site_ids = [], []
    for _sid in species_settlement:
        _p = np.where(grid_ids == _sid)
        if len(_p[0]) > 0:
            _site_pos.append((int(_p[0][0]), int(_p[1][0])))
            _site_ids.append(int(_sid))
    if not _site_pos:
        return {}
    _tree = cKDTree(np.array(_site_pos))
    buffer_to_site = {}
    buffered = set()
    for _sid in species_settlement:
        _p = np.where(grid_ids == _sid)
        if len(_p[0]) == 0:
            continue
        r, c = int(_p[0][0]), int(_p[1][0])
        for i in range(-buffer, buffer + 1):
            for j in range(-buffer, buffer + 1):
                nr_, nc = r + i, c + j
                if 0 <= nr_ < grid_ids.shape[0] and 0 <= nc < grid_ids.shape[1]:
                    buffered.add(int(grid_ids[nr_, nc]))
    for _cell in buffered:
        _cp = np.where(grid_ids == _cell)
        if len(_cp[0]) > 0:
            _, _idx = _tree.query((int(_cp[0][0]), int(_cp[1][0])))
            buffer_to_site[_cell] = _site_ids[_idx]
    return buffer_to_site


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
        2D array of grid IDs for the study extent
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

    ## Return all grid IDs within buffer cells of a given grid ID
    def get_adjacent_grids(grid_id, grid_matrix, buffer):
        """Return a list of grid IDs within *buffer* cells of the given grid ID in the 2D grid matrix."""
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
        print(f"Warning: Traits is {type(traits)}, attempting to load...")
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
        # buffer_to_site maps each buffer cell to its nearest real settlement site (populated below when buffer > 0)
        # empty means exact-cell, no remapping
        buffer_to_site = {}
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
                # Build the buffer-cell -> nearest-real-site mapping via the shared helper
                buffer_to_site = _build_buffer_to_site(species_settlement, grid_ids, buffer)
            else:
                print(f"Warning: '{display_species_name(species_name_std)}' not found in the settlement dataset. Using all valid sites for settlement.")
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
                        # Crop at settlement time, then credit settlement to the owning real site (collapse buffer cells into their parent site so the site count does not inflate
                        owner_site = buffer_to_site.get(int(current_grid_id), int(current_grid_id))
                        settled_traj = traj_subset[traj_subset['time'] <= closest_obs['time']].copy()
                        settled_traj.loc[settled_traj['obs'] == settled_traj['obs'].max(), 'grid_id'] = owner_site
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
            # No release constraint and keep all trajectories
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
                # release_grids is a dict (species-specific/pooled strategy) but this species has
                # no release grid. That only happens for a trait species with no occurrence records
                # under the species-specific strategy: it has no defined habitat, so it cannot be
                # released from anywhere and is dropped (not kept). The all-sites strategy passes
                # release_grids=None and is handled above, so it never reaches here.
                final_processed[species_name] = pd.DataFrame()
                print(f"  {species_name}: dropped (no occurrence records, so no release habitat)")
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
def post_process(chunks_dir_or_traj, traits, release_grids, settlement_grids, grid_ids, buffer=None):
    """
    Apply species-specific temporal and spatial filters to trajectory data to create species-specific connectivity data.

    This is the main post-processing pipeline that transforms raw particle trajectories into species-specific connectivity data by applying:
    1. Spawning season filters (temporal)
    2. Planktonic larval duration filters (temporal)
    3. Habitat suitability filters (spatial)

    Returns connectivity matrices and filtered trajectories for further analysis.

    Parameters
    ----------
    chunks_dir_or_traj : str or pandas.DataFrame
        Either a path to the directory of on-disk trajectory chunks (memory-efficient
        mode) or an in-memory DataFrame of raw particle trajectories from
        process_sim_results(). When a DataFrame, it has the columns:
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
    - buffer=0: Settlement only in exact settlement grid cells (default)
    - buffer=1: Settlement in grid cells or immediate neighbors
    - buffer=2: Settlement within 2-cell radius
    - Larger buffers account for sub-grid-scale habitat heterogeneity
    """
    gc.collect()

    subsection_header("Starting Post-processing")

    # Check for required columns (read first chunk to verify if using chunks)
    if isinstance(chunks_dir_or_traj, str) and os.path.isdir(chunks_dir_or_traj):
        sample_files = sorted(glob(os.path.join(chunks_dir_or_traj, 'chunk_*.parquet')))
        if sample_files:
            sample_cols = pd.read_parquet(sample_files[0], engine='pyarrow', columns=None).columns.tolist()
        else:
            print("\nError: No chunk files found.")
            return {}, buffer
        required_columns = ['trajectory', 'time', 'obs', 'grid_id', 'lat', 'lon']
        missing = [col for col in required_columns if col not in sample_cols]
    elif chunks_dir_or_traj is None or not hasattr(chunks_dir_or_traj, 'columns'):
        print(
            "\nError: No trajectory data available to post-process (expected a chunk directory or a trajectory DataFrame).")
        return {}, buffer
    else:
        required_columns = ['trajectory', 'time', 'obs', 'grid_id', 'lat', 'lon']
        missing = [col for col in required_columns if col not in chunks_dir_or_traj.columns]
    if missing:
        print(f"\nError: Missing required columns: {missing}")
        return {}, buffer

        # Ensure traits is a list of dicts
    if isinstance(traits, pd.DataFrame):
        traits_list = traits.to_dict('records')
    elif isinstance(traits, list):
        traits_list = traits
    else:
        print(f"Error: Traits must be list or DataFrame, got {type(traits)}")
        return {}, buffer

    # Settlement buffer: a propagule settles if it reaches a settlement cell or a cell within 'buffer' cells of it
    # Buffer-cell settlements are credited to the nearest real settlement site (in pld_filter), so the buffer can never inflate the number of settlement sites
    # Interactive passes buffer=None and is prompted here, automatic/manual pass their config value, so they never block on input
    if buffer is None:
        buffer = get_positive_integer(
            "Enter buffer size around settlement sites [default: 0, exact cell only]: ",
            default=0, min_val=0
        )

    # Determine if we received a chunks_dir (str) or a DataFrame (backward compat)
    if isinstance(chunks_dir_or_traj, str) and os.path.isdir(chunks_dir_or_traj):
        chunks_dir = chunks_dir_or_traj
        chunk_files = sorted(glob(os.path.join(chunks_dir, 'chunk_*.parquet')))
        use_chunks = True
        print(f"  Processing {len(chunk_files)} chunks from disk (memory-efficient mode).")
    else:
        # Backward compatibility: received a DataFrame directly
        use_chunks = False
        traj = chunks_dir_or_traj

    # Process each species one at a time to manage memory
    final_processed = {}
    summary_lines = []
    total_successful = 0
    n_species = len(traits_list)

    # Count total unique trajectories in the raw input , used to report
    # how many were filtered out at each step (per species).
    if use_chunks:
        _input_traj_ids = set()
        for cf in chunk_files:
            _input_traj_ids.update(
                pd.read_parquet(cf, columns=['trajectory'], engine='pyarrow')['trajectory'].unique()
            )
        n_input_traj = len(_input_traj_ids)
        del _input_traj_ids
    else:
        n_input_traj = traj['trajectory'].nunique()
    print(f"  Raw input: {n_input_traj:,} unique trajectories before filtering.")

    for i, species_trait in enumerate(traits_list):
        species_name = standardise_species_name(species_trait['species_name'])
        print(f"\n  [{i + 1}/{n_species}] Processing: {display_species_name(species_name)}")

        if use_chunks:
            # Per-chunk processing: apply all three filters to each chunk, collect only the surviving rows (typically 1-5% of data).
            # Suppress per-chunk print output from filter functions.
            species_survivors = []
            _n_spawning = 0
            _n_pld = 0
            _n_habitat = 0
            for ci, cf in enumerate(chunk_files):
                print(f"\r    Processing chunk {ci + 1}/{len(chunk_files)}...    ", end='', flush=True)
                chunk = pd.read_parquet(cf, engine='pyarrow')
                # Apply filters with stdout suppressed to avoid per-chunk noise
                with contextlib.redirect_stdout(io.StringIO()):
                    sp_spawning = spawning_filter(chunk, [species_trait])
                    if not sp_spawning or species_name not in sp_spawning or sp_spawning[species_name].empty:
                        del chunk
                        continue
                    _n_spawning += sp_spawning[species_name]['trajectory'].nunique()
                    sp_pld = pld_filter(sp_spawning, [species_trait], settlement_grids, grid_ids, buffer)
                    del sp_spawning
                    if not sp_pld or species_name not in sp_pld or sp_pld[species_name].empty:
                        del chunk
                        continue
                    _n_pld += sp_pld[species_name]['trajectory'].nunique()
                    sp_final = habitat_filter({species_name: sp_pld[species_name]}, release_grids)
                    del sp_pld
                if species_name in sp_final and not sp_final[species_name].empty:
                    _n_habitat += sp_final[species_name]['trajectory'].nunique()
                    species_survivors.append(sp_final[species_name])
                del chunk, sp_final
                gc.collect()
            print()
            _out_spawning = max(0, n_input_traj - _n_spawning)
            _out_pld = max(0, _n_spawning - _n_pld)
            _out_habitat = max(0, _n_pld - _n_habitat)
            print(f"Spawning filter: {_n_spawning:,} retained ({_out_spawning:,} filtered out: Released outside the spawning window)")
            print(f"PLD filter: {_n_pld:,} retained ({_out_pld:,} filtered out: Exceeded their PLD before reaching suitable settlement habitat)")
            print(f"Habitat filter: {_n_habitat:,} retained ({_out_habitat:,} filtered out: Not released from this species' habitat)")

            if species_survivors:
                combined = pd.concat(species_survivors, ignore_index=True)
                final_processed[species_name] = combined
                n_traj = combined['trajectory'].nunique()
                n_releases = combined['release_id'].nunique() if 'release_id' in combined.columns else 0
                total_successful += n_traj
                summary_lines.append(
                    f"{display_species_name(species_name)}: {n_traj} successful trajectories from {n_releases} release cohorts"
                )
                print(f"    Filters complete: {n_traj} successful trajectories from {n_releases} release cohorts")
                del species_survivors
            else:
                print(f"    No trajectories survived all filters.")
            gc.collect()
            continue  # skip the non-chunked path below

        # Non-chunked path (backward compatibility with DataFrame input)
        # Spawning filter for this species only
        sp_spawning = spawning_filter(traj, [species_trait])

        if not sp_spawning or species_name not in sp_spawning or sp_spawning[species_name].empty:
            print(f"    No trajectories within spawning window. Skipping.")
            continue

        _n_spawning = sp_spawning[species_name]['trajectory'].nunique()
        _out_spawning = max(0, n_input_traj - _n_spawning)
        print(f"Spawning filter: {_n_spawning:,} retained  ({_out_spawning:,} filtered out: Released outside the spawning window)")

        # PLD filter for this species only
        sp_pld = pld_filter(sp_spawning, [species_trait], settlement_grids, grid_ids, buffer)

        # Free spawning data
        del sp_spawning
        gc.collect()

        if not sp_pld or species_name not in sp_pld or sp_pld[species_name].empty:
            print(f"    No trajectories survived PLD filter. Skipping.")
            continue

        _n_pld = sp_pld[species_name]['trajectory'].nunique()
        _out_pld = max(0, _n_spawning - _n_pld)
        print(f"PLD filter: {_n_pld:,} retained ({_out_pld:,} filtered out: Exceeded their PLD before reaching suitable settlement habitat)")

        # Habitat filter for this species only
        sp_final = habitat_filter({species_name: sp_pld[species_name]}, release_grids)

        # Free PLD data
        del sp_pld
        gc.collect()

        if species_name in sp_final and not sp_final[species_name].empty:
            final_processed[species_name] = sp_final[species_name]
            n_traj = sp_final[species_name]['trajectory'].nunique()
            n_releases = sp_final[species_name]['release_id'].nunique()
            total_successful += n_traj
            _out_habitat = max(0, _n_pld - n_traj)
            summary_lines.append(
                f"{display_species_name(species_name)}: {n_traj} successful trajectories from {n_releases} release cohorts"
            )
            print(
                f"Habitat filter: {n_traj:,} retained ({_out_habitat:,} filtered out: Not released from this species' habitat)")
            print(f"From {n_releases} release cohorts.")
        else:
            print(f"No trajectories survived habitat filter.")

        del sp_final
        gc.collect()

    if total_successful == 0:
        print("\nError: No trajectories survived filtering for any species.")
        print("Check that:")
        print("  1. Particle release times overlap with spawning windows")
        print("  2. Species traits file has correct date format (DD/MM/YYYY)")
        print("  3. Simulation runtime covers spawning periods")
        return {}, buffer

    summary_lines.append(f"")
    summary_lines.append(f"Total successful trajectories: {total_successful}")
    data_box("Post-processing Summary", summary_lines)

    return final_processed, buffer
# endregion

# region Connectivity Analyses
## region Analysis Type 1: Connectivity Metrics
def calculate_connectivity_metrics(final_processed, release_grids, settlement_grids,
                                   traits, traj_raw=None, print_summary=True):
    """
    Calculate standard connectivity metrics for community and species.

    Computes settlement success, mortality rates, self-recruitment, dispersal
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
        The release rows (obs=0) of the trajectory data, e.g. from load_traj_starts_from_chunks(chunks_dir).
        Only these rows are used, to count how many particles were released, so the full trajectories are not needed.
        Required for accurate mortality calculation.
        If None, mortality cannot be calculated accurately.

    Returns
    -------
    dict
        Contains two sub-dictionaries:
        - 'community' : dict - Aggregated metrics across all species
        - 'per_species' : dict - Metrics for each species individually

    Notes
    -----
    Mortality Calculation:
        If traj_raw is provided, mortality = 1 - (n_settled / n_released).
        n_released is counted from raw trajectories released during the species' spawning window.
        If traj_raw is None, mortality cannot be calculated accurately.

    Self-recruitment:
        Particles whose settlement site is their own release site:
        an exact match on post-processed site IDs (the diagonal of the connectivity matrix).
        The settlement buffer is resolved earlier in post-processing, so this is consistent with every other metric.

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

    # Mortality needs the raw released counts (traj_raw). If those aren't supplied
    # it can't be estimated, so note it once here rather than once per species.
    if traj_raw is None and print_summary:
        print("  Note: Mortality estimate unavailable (raw release counts were not supplied for this run).")

    for species_name, df in final_processed.items():
        # Format species name for display
        display_name = display_species_name(species_name)

        if df.empty:
            print(f"No successful trajectories")
            continue

        # Get release sites for this species. Match keys by standardised name so a space/underscore spelling difference between the grids and the trait names cannot cause a silent miss.
        species_name_std = standardise_species_name(species_name)
        release_key = next((k for k in release_grids if standardise_species_name(k) == species_name_std),
                           None) if release_grids else None
        release_sites = set(release_grids[release_key]['grid_id']) if release_key is not None else set()

        # Get settlement sites, using the same standardised matching.
        settle_key = next((k for k in settlement_grids if standardise_species_name(k) == species_name_std),
                          None) if settlement_grids else None
        settle_sites = set(settlement_grids[settle_key]['grid_id']) if settle_key is not None else set()

        # Count settled trajectories
        n_trajectories = df['trajectory'].nunique()

        # Calculate per-trajectory metrics
        # Vectorised: each trajectory's first/last observation via one sort + drop_duplicates, instead of filtering the DataFrame once per trajectory (O(N^2), slow when many particles reach this step, e.g. passive).
        trajectory_plds = []
        trajectory_distances = []
        recruitment_count = 0

        df_sorted = df.sort_values(['trajectory', 'obs'])
        traj_sizes = df_sorted.groupby('trajectory')['obs'].transform('size')
        df_multi = df_sorted[traj_sizes >= 2]

        if not df_multi.empty:
            firsts = df_multi.drop_duplicates('trajectory', keep='first').set_index('trajectory')
            lasts = df_multi.drop_duplicates('trajectory', keep='last').set_index('trajectory')
            lasts = lasts.reindex(firsts.index)

            # Calculate realized PLD (days in plankton)
            if 'time' in df_multi.columns:
                pld_series = (pd.to_datetime(lasts['time']) - pd.to_datetime(
                    firsts['time'])).dt.total_seconds() / (24 * 3600)
                trajectory_plds = pld_series.tolist()
                all_plds.extend(trajectory_plds)

            # Calculate net displacement (km) using Haversine formula
            # This is the straight-line distance from release to settlement point, not the total path length travelled
            # Particles that self-recruit may show near-zero displacement despite having travelled some distance
            if 'lat' in df_multi.columns and 'lon' in df_multi.columns:
                R = 6371  # Earth radius in km
                lat1 = np.radians(firsts['lat'].to_numpy())
                lon1 = np.radians(firsts['lon'].to_numpy())
                lat2 = np.radians(lasts['lat'].to_numpy())
                lon2 = np.radians(lasts['lon'].to_numpy())
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
                distances = R * 2 * np.arcsin(np.sqrt(a))
                trajectory_distances = distances.tolist()
                all_distances.extend(trajectory_distances)

            # Self-recruitment
            if 'grid_id' in df_multi.columns:
                start_ids = firsts['grid_id'].astype(int).to_numpy()
                end_ids = lasts['grid_id'].astype(int).to_numpy()
                recruitment_count += int(np.sum(start_ids == end_ids))

        # Calculate n_released from raw trajectory data
        n_released = n_trajectories  # Default: assume all settled came from released
        mortality_calculable = False

        if traj_raw is None:
            pass  # mortality not calculable is already noted once before the loop
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
        min_distance = min(trajectory_distances) if trajectory_distances else 0
        max_distance = max(trajectory_distances) if trajectory_distances else 0

        # PLD statistics
        mean_pld = np.mean(trajectory_plds) if trajectory_plds else 0
        min_pld = min(trajectory_plds) if trajectory_plds else 0
        max_pld = max(trajectory_plds) if trajectory_plds else 0

        # Store species results
        results['per_species'][display_name] = {
            'n_species': 1,
            'n_released': n_released,
            'n_settled': n_trajectories,
            'n_recruited': recruitment_count,
            'mortality_rate': mortality_rate if mortality_calculable else None,
            'settlement_rate': settlement_rate,
            'self_recruitment_rate': recruitment_rate,
            'mean_distance_km': mean_distance,
            'min_distance_km': min_distance,
            'max_distance_km': max_distance,
            'mean_pld_days': mean_pld,
            'min_pld_days': min_pld,
            'max_pld_days': max_pld,
            'n_release_sites': len(release_sites),
            'n_settlement_sites': len(settle_sites),
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
            print(f"Self-recruitment rate: {recruitment_rate * 100:.1f}%, ({recruitment_count} trajectories)")
            print(f"Min net displacement: {min_distance:.1f} km")
            print(f"Mean net displacement: {mean_distance:.1f} km")
            print(f"Max net displacement: {max_distance:.1f} km")
            print(f"Min PLD: {min_pld:.1f} days")
            print(f"Mean PLD: {mean_pld:.1f} days")
            print(f"Max PLD: {max_pld:.1f} days")

    # Community-level summary
    community_mortality = 1 - (total_settled / total_released) if total_released > 0 else 0
    community_recruitment = total_retained / total_settled if total_settled > 0 else 0

    # Community release/settlement site counts = the union of every species' release/settlement grid cells, so species sharing the same cells are not double counted
    _comm_release_sites = {gid for _g in (release_grids or {}).values()
                           if isinstance(_g, pd.DataFrame) and 'grid_id' in _g.columns
                           for gid in _g['grid_id'].tolist()}
    _comm_settle_sites = {gid for _g in (settlement_grids or {}).values()
                          if isinstance(_g, pd.DataFrame) and 'grid_id' in _g.columns
                          for gid in _g['grid_id'].tolist()}

    results['community'] = {
        'n_species': len([s for s in final_processed.values() if not s.empty]),
        'total_released': total_released,
        'total_settled': total_settled,
        'total_recruited': total_retained,
        'mortality_rate': community_mortality if any_mortality_calculable else None,
        'settlement_rate': 1 - community_mortality if any_mortality_calculable else total_settled / max(1,
                                                                                                        total_released),
        'self_recruitment_rate': community_recruitment,
        'mean_distance_km': round(np.mean(all_distances), 2) if all_distances else 0,
        'min_distance_km': round(min(all_distances), 2) if all_distances else 0,
        'max_distance_km': round(max(all_distances), 2) if all_distances else 0,
        'mean_pld_days': np.mean(all_plds) if all_plds else 0,
        'min_pld_days': min(all_plds) if all_plds else 0,
        'max_pld_days': max(all_plds) if all_plds else 0,
        'n_release_sites': len(_comm_release_sites),
        'n_settlement_sites': len(_comm_settle_sites),
    }
    return results

## Exporting the analysis 1 results
def export_connectivity_metrics(results, output_dir=None):
    """
    Export connectivity metrics to CSV files.

    Each label (community + per-species) gets its own subfolder under output_dir,
    and the filename embeds the label: <label>_connectivity_summary.csv.

    Parameters
    ----------
    results : dict
        Output from calculate_connectivity_metrics()
    output_dir : str
        Root results directory; subfolders are created automatically.

    Returns
    -------
    list of str
        Paths of all files written.
    """
    if output_dir is None:
        output_dir = get_default_dir('results')

    written = []

    def _drop_std(_df):
        return _df.drop(columns=[c for c in _df.columns if c.endswith('_std') or c.startswith('std_')], errors='ignore')

    # Community-level summary
    if 'community' in results:
        comm_dir = os.path.join(output_dir, 'community')
        os.makedirs(comm_dir, exist_ok=True)
        comm_file = os.path.join(comm_dir, 'community_connectivity_summary.csv')
        _drop_std(pd.DataFrame([results['community']])).to_csv(comm_file, index=False)
        print(f"Community metrics saved to: {comm_file}")
        written.append(comm_file)

    # Per-species summaries
    for sp, sp_data in results.get('per_species', {}).items():
        if 'community' in sp_data:
            sp_dir = os.path.join(output_dir, sp)
            os.makedirs(sp_dir, exist_ok=True)
            sp_file = os.path.join(sp_dir, f'{sp}_connectivity_summary.csv')
            _drop_std(pd.DataFrame([sp_data['community']])).to_csv(sp_file, index=False)
            print(f"Species metrics saved to: {sp_file}")
            written.append(sp_file)

    return written


def build_settlement_attribution(settlement_grids, grid_ids, buffer):
    """
    Build the buffer-cell-to-settlement-site mapping for every species and return
    it as a dict of DataFrames ready for CSV export.

    Uses the same cKDTree nearest-real-site logic as pld_filter but as a
    standalone call so it can be invoked at export time without touching pld_filter.

    Parameters
    ----------
    settlement_grids : dict or None
        Mapping of species name to settlement grid DataFrame (must have 'grid_id' column).
    grid_ids : numpy.ndarray
        2D grid of cell IDs for the study extent.
    buffer : int
        Settlement buffer radius in cells (same value passed to pld_filter).

    Returns
    -------
    dict
        {species_name: DataFrame(buffer_cell_grid_id, attributed_settlement_site)}
        Empty dict when settlement_grids is None, buffer == 0, or no mapping exists.
    """
    if settlement_grids is None or buffer == 0:
        return {}
    attribution = {}
    for sp, sg in settlement_grids.items():
        if sg is None or (hasattr(sg, 'empty') and sg.empty) or 'grid_id' not in sg.columns:
            continue
        species_settlement = set(sg['grid_id'].astype(int))
        b2s = _build_buffer_to_site(species_settlement, grid_ids, buffer)
        if b2s:
            attribution[sp] = pd.DataFrame(
                [{'buffer_cell_grid_id': k, 'attributed_settlement_site': v}
                 for k, v in b2s.items() if k != v]
            )
    return attribution


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
        print(f"{'Settlement rate':<35} {comm['settlement_rate'] * 100:>14.1f}%")
        print(f"{'Mortality rate':<35} {comm['mortality_rate'] * 100:>14.1f}%")
        print(f"{'Self-recruitment rate':<35} {comm['self_recruitment_rate'] * 100:>14.1f}%")
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
                      f"{metrics['self_recruitment_rate'] * 100:>7.1f}% {metrics['mean_distance_km']:>10.1f}")
        else:
            # Show summary for large number of species
            print(f"\n{n_species} species analysed (showing summary to avoid clutter)")
            print("-" * 50)

            # Calculate summary statistics
            all_settlement = [m['settlement_rate'] for m in results['per_species'].values()]
            all_mortality = [m['mortality_rate'] for m in results['per_species'].values()]
            all_recruitment = [m['self_recruitment_rate'] for m in results['per_species'].values()]
            all_distance = [m['mean_distance_km'] for m in results['per_species'].values()]

            print(f"{'Metric':<25} {'Mean':>10} {'Min':>10} {'Max':>10}")
            print("-" * 55)
            print(f"{'Settlement rate (%)':<25} {np.mean(all_settlement) * 100:>10.1f} "
                  f"{np.min(all_settlement) * 100:>10.1f} {np.max(all_settlement) * 100:>10.1f}")
            print(f"{'Mortality rate (%)':<25} {np.mean(all_mortality) * 100:>10.1f} "
                  f"{np.min(all_mortality) * 100:>10.1f} {np.max(all_mortality) * 100:>10.1f}")
            print(f"{'Self-recruitment rate (%)':<25} {np.mean(all_recruitment) * 100:>10.1f} "
                  f"{np.min(all_recruitment) * 100:>10.1f} {np.max(all_recruitment) * 100:>10.1f}")
            print(f"{'Mean net displacement (km)':<25} {np.mean(all_distance):>10.1f} "
                  f"{np.min(all_distance):>10.1f} {np.max(all_distance):>10.1f}")

            print("\nNote: Full per-species data available via export to CSV")
# endregion

# region Analysis Type 2: Network Analysis
## Network analysis function
def network_analysis(final_processed, release_grids, settlement_grids,
                     coastal_grid_df, grid_ids, full_grid_df, null_scale,
                     exclude_self_recruitment=False, verbose=True):
    """
    Perform network analysis to identify connectivity hotspots.

    Calculates functional roles (source, sink, self-recruitment) and centrality
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
        Full study extent grid information
    null_scale : float
        Distance threshold in km for filtering connections.
    exclude_self_recruitment : bool
        If True, each site's self-recruitment is subtracted from its source and sink metric so particle import/export is not inflated.
        Self-recruitment is still computed and reported as its own metric and the centrality measures are unaffected.

    Returns
    -------
    dict
        Contains site_metrics, connectivity_matrix, top10 rankings, network_graph

    References
    ----------
    Methodology adapted from Melia et al. (2016) Scientific Reports.
    """
    gc.collect()

    # Study extent = the sites the user actually defined as valid release or settlement locations (from the chosen strategies).
    study_site_ids = set()
    for _grids in (release_grids or {}, settlement_grids or {}):
        for _df in _grids.values():
            if isinstance(_df, pd.DataFrame) and 'grid_id' in _df.columns:
                study_site_ids.update(_df['grid_id'].astype(int).tolist())
    if not study_site_ids:
        study_site_ids = set(coastal_grid_df['grid_id'].astype(int))
    study_extent = len(study_site_ids)

    # Study bounds from the actual grid
    _bound_coords = []
    for _grids in (release_grids or {}, settlement_grids or {}):
        for _df in _grids.values():
            if isinstance(_df, pd.DataFrame) and {'latitude', 'longitude'}.issubset(_df.columns):
                _bound_coords.append(_df[['latitude', 'longitude']])
    if _bound_coords:
        _all_coords = pd.concat(_bound_coords, ignore_index=True)
        study_bounds = {
            'lat_min': _all_coords['latitude'].min(),
            'lat_max': _all_coords['latitude'].max(),
            'lon_min': _all_coords['longitude'].min(),
            'lon_max': _all_coords['longitude'].max()
        }
    else:
        study_bounds = {
            'lat_min': coastal_grid_df['latitude'].min(),
            'lat_max': coastal_grid_df['latitude'].max(),
            'lon_min': coastal_grid_df['longitude'].min(),
            'lon_max': coastal_grid_df['longitude'].max()
        }

    # Count potential connectivity sites (where species COULD connect based on occurrence data)
    # This includes buffered settlement sites to match what post-processing allows
    potential_sites = 0
    n_species = len([s for s in final_processed.values() if isinstance(s, pd.DataFrame) and not s.empty])

    ## Return grid IDs within buffer cells of a given grid ID for potential connectivity counting
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
        # Include buffered settlement sites to match post-processing behaviour
        all_species_sites = set()
        for species_name in settlement_grids:
            # Add settlement sites WITH BUFFER
            species_settle_df = settlement_grids.get(species_name)
            if isinstance(species_settle_df, pd.DataFrame) and 'grid_id' in species_settle_df.columns:
                for gid in species_settle_df['grid_id'].astype(int).tolist():
                    # Add the site itself and its buffered neighbors
                    all_species_sites.add(gid)
                    # Buffer removed. Potential sites are exact-cell (no neighbour expansion).
                    # Original expansion preserved below (commented) in case the buffer is reinstated.
                    # all_species_sites.update(get_adjacent_grids_for_potential(gid, grid_ids, buffer=1))
            # Add release sites (no buffer needed for release)
            species_release_df = release_grids.get(species_name)
            if isinstance(species_release_df, pd.DataFrame) and 'grid_id' in species_release_df.columns:
                all_species_sites.update(species_release_df['grid_id'].astype(int).tolist())

        potential_sites = len(all_species_sites.intersection(study_site_ids))
    else:
        potential_sites = study_extent

    # Get realised connectivity sites (where particles actually started/ended)
    valid_sites = set()
    for species_name, df in final_processed.items():
        if df.empty:
            continue
        # Vectorised: source = grid cell at obs == 0, destination = grid cell at the largest obs
        # Only trajectories with >=2 observations and an obs == 0 row contribute
        if not df.empty and 'grid_id' in df.columns:
            _ds = df.sort_values(['trajectory', 'obs'])
            _sizes = _ds.groupby('trajectory')['obs'].transform('size')
            _multi = _ds[_sizes >= 2]
            _starts = \
            _multi[_multi['obs'] == 0].drop_duplicates('trajectory', keep='first').set_index('trajectory')[
                'grid_id'].astype(int)
            _lasts = _multi.drop_duplicates('trajectory', keep='last').set_index('trajectory')['grid_id']
            _lasts = _lasts.reindex(_starts.index).dropna().astype(int)
            valid_sites.update(_starts.tolist())
            valid_sites.update(_lasts.tolist())

    # Filter to the valid study sites
    valid_sites = valid_sites.intersection(study_site_ids)
    valid_sites = sorted(list(valid_sites))

    if len(valid_sites) < 3:
        print("Error: Not enough connected sites for network analysis")
        return None

    # Create site names and connectivity matrix
    # Use named sites if available (from site-based analysis Option 3)
    site_name_lookup = spp_params.get('site_name_lookup', {}) if 'spp_params' in dir() else {}
    # Also check globals in case spp_params is not in local scope
    if not site_name_lookup:
        try:
            site_name_lookup = globals().get('spp_params', {}).get('site_name_lookup', {})
        except Exception:
            site_name_lookup = {}

    # Build the labels and a label -> grid_id map together
    site_labels = []
    _label_to_gid = {}
    for sid in valid_sites:
        _lbl = site_name_lookup.get(int(sid), f'site_{int(sid)}') if site_name_lookup else f'site_{int(sid)}'
        site_labels.append(_lbl)
        _label_to_gid[_lbl] = int(sid)
    connectivity_matrix = pd.DataFrame(0, index=site_labels, columns=site_labels, dtype=float)

    # Track per-species connections
    per_species_matrices = {}

    # Build connectivity counts
    for species_name, df in final_processed.items():
        if df.empty:
            continue

        display_name = display_species_name(species_name)
        species_matrix = pd.DataFrame(0, index=site_labels, columns=site_labels, dtype=float)

        # Vectorised: build (source, destination) once per trajectory instead of filtering the DataFrame N times
        # Only pairs where both ends are valid_sites are counted
        if not df.empty and 'grid_id' in df.columns:
            _ds = df.sort_values(['trajectory', 'obs'])
            _sizes = _ds.groupby('trajectory')['obs'].transform('size')
            _multi = _ds[_sizes >= 2]
            _src = _multi[_multi['obs'] == 0].drop_duplicates('trajectory', keep='first').set_index('trajectory')[
                'grid_id']
            _dst = _multi.drop_duplicates('trajectory', keep='last').set_index('trajectory')['grid_id'].reindex(
                _src.index)
            _valid_set = set(valid_sites)
            for _s, _d in zip(_src.to_numpy(), _dst.to_numpy()):
                if pd.isna(_s) or pd.isna(_d):
                    continue
                _s = int(_s);
                _d = int(_d)
                if _s in _valid_set and _d in _valid_set:
                    _ls = site_name_lookup.get(_s, f'site_{_s}') if site_name_lookup else f'site_{_s}'
                    _ld = site_name_lookup.get(_d, f'site_{_d}') if site_name_lookup else f'site_{_d}'
                    connectivity_matrix.loc[_ls, _ld] += 1
                    species_matrix.loc[_ls, _ld] += 1

        per_species_matrices[display_name] = species_matrix

    unique_connections = int(np.sum(connectivity_matrix.values > 0))

    # Print formatted summary table (suppressed when verbose=False. The dropdown viewer shows this instead)
    if verbose:
        print(f"\n{'Metric':<35} {'Value':>15}")
        print("-" * 50)
        print(f"{'Number of species':<35} {n_species:>15}")
        print(f"{'Study extent (sites)':<35} {study_extent:>15}")
        print(f"{'Potential connectivity sites':<35} {potential_sites:>15}")
        print(f"{'Realised connectivity sites':<35} {len(valid_sites):>15}")
        print(f"{'Unique connections':<35} {unique_connections:>15}")

    # Get site coordinates
    site_coords = coastal_grid_df[coastal_grid_df['grid_id'].astype(int).isin(valid_sites)][
        ['grid_id', 'longitude', 'latitude']].drop_duplicates()
    site_coords['grid_id'] = site_coords['grid_id'].astype(int)

    # Create distance matrix
    coords = []
    for site_id in valid_sites:
        site_data = site_coords[site_coords['grid_id'] == site_id]
        if not site_data.empty:
            coords.append([site_data['longitude'].iloc[0], site_data['latitude'].iloc[0]])
    coords = np.array(coords)

    # Great-circle (haversine) distance in km between every pair of sites, so the scale thresholds
    # (local/regional/seascape) are geographically accurate and consistent with the dispersal-distance
    # metric. A flat "degrees x 111" would overestimate east-west distances by ~1/cos(lat) (~30% at
    # mid-latitudes) and mis-bin mostly-longitudinal connections.
    if len(coords) > 0:
        _lon = np.radians(coords[:, 0])
        _lat = np.radians(coords[:, 1])
        _dlon = _lon[:, None] - _lon[None, :]
        _dlat = _lat[:, None] - _lat[None, :]
        _a = np.sin(_dlat / 2) ** 2 + np.cos(_lat[:, None]) * np.cos(_lat[None, :]) * np.sin(_dlon / 2) ** 2
        distance_matrix = 6371.0 * 2 * np.arcsin(np.sqrt(_a))
    else:
        distance_matrix = np.zeros((0, 0))

    # Apply distance filtering (or skip if null_scale is None for full domain)
    if null_scale is None:
        connectivity_filtered = connectivity_matrix.copy()
    else:
        connectivity_filtered = connectivity_matrix.copy()
        for i in range(len(connectivity_matrix)):
            for j in range(len(connectivity_matrix)):
                if distance_matrix[i, j] > null_scale:
                    connectivity_filtered.iloc[i, j] = 0

        connections_remaining = int(np.sum(connectivity_filtered.values > 0))
        weight_remaining = int(connectivity_filtered.values.sum())
        if verbose:
            _few = "  (few connections, so metrics may be unreliable at this scale)" if connections_remaining < 20 else ""
            print(
                f"  {null_scale} km filter: {connections_remaining} connections, {weight_remaining:,} particles exchanged{_few}")

    # Recompute the realised-network summary on the SCALE-FILTERED matrix so the network summary
    # reflects the selected distance scale. For the full domain (null_scale is None) connectivity_filtered
    # equals the unfiltered matrix, so these are unchanged. Previously both used the unfiltered matrix, so
    # every scale reported the full-domain unique-connection and realised-site counts even though the
    # graph, matrix and Louvain communities were scale-filtered.
    _filt_vals = connectivity_filtered.values
    unique_connections = int(np.sum(_filt_vals > 0))
    realised_connectivity_sites = int(np.sum((_filt_vals.sum(axis=1) + _filt_vals.sum(axis=0)) > 0))

    # Create network graph
    G = nx.from_pandas_adjacency(connectivity_filtered, create_using=nx.DiGraph())

    # =========================================================================
    # Calculate metrics
    # =========================================================================

    # Self-recruitment: self-loops (particles returning to the same site).
    self_recruitment = {}
    for node in G.nodes():
        if G.has_edge(node, node):
            self_recruitment[node] = G[node][node]['weight']
        else:
            self_recruitment[node] = 0

    # Source strength: total outgoing connections (export potential)
    source_strength = dict(G.out_degree(weight='weight'))
    # Sink strength: total incoming connections (import potential)
    sink_strength = dict(G.in_degree(weight='weight'))

    # Optionally remove the self-recruitment contribution from source/sink only
    # A self-loop counts toward both out- and in-degree, so it is subtracted from each. Self-recruitment is still reported as its own metric and the centrality measures are unaffected.
    if exclude_self_recruitment:
        source_strength = {n: source_strength.get(n, 0) - self_recruitment.get(n, 0) for n in G.nodes()}
        sink_strength = {n: sink_strength.get(n, 0) - self_recruitment.get(n, 0) for n in G.nodes()}
        if verbose:
            status_message("Self-recruitment excluded from source/sink metrics.", "info")

    # Betweenness: how often site lies on shortest paths between other sites
    # Betweenness and closeness are shortest-path measures: NetworkX treats the edge attribute as a
    # DISTANCE (higher = longer, less-preferred path). Our edge 'weight' is connectivity STRENGTH
    # (higher = more larval exchange), so we route shortest paths on an inverse-strength distance.
    # This makes strong connections "short": sites bridging strong flows score high on betweenness,
    # and well-connected sites score high on closeness - so higher = more connected, and the
    # downstream ranking (higher value = better = lower rank number) reads correctly.
    for _u, _v, _edata in G.edges(data=True):
        _w = _edata.get('weight', 0)
        _edata['distance'] = (1.0 / _w) if _w > 0 else float('inf')

    betweenness_cent = nx.betweenness_centrality(G, weight='distance', normalized=True)

    # Closeness: how "close" (strongly connected) a site is to all others
    closeness_cent = nx.closeness_centrality(G, distance='distance')

    # Degree: total number of connections (normalized)
    degree_cent = nx.degree_centrality(G)

    # Create metrics DataFrame
    site_metrics = pd.DataFrame({
        'site': list(G.nodes()),
        # Functional roles
        'source_strength': [source_strength.get(node, 0) for node in G.nodes()],
        'sink_strength': [sink_strength.get(node, 0) for node in G.nodes()],
        'self_recruitment': [self_recruitment.get(node, 0) for node in G.nodes()],
        # Centrality measures
        'betweenness': [betweenness_cent.get(node, 0) for node in G.nodes()],
        'closeness': [closeness_cent.get(node, 0) for node in G.nodes()],
        'degree': [degree_cent.get(node, 0) for node in G.nodes()],
    })

    # Add coordinates. Recover grid_id from the label map above so this is robust to human-readable site names; the old 'site'.split('_') parse crashed on any name that was not literally 'site_<int>'.
    site_metrics['grid_id'] = site_metrics['site'].map(_label_to_gid).astype(int)
    site_metrics = pd.merge(site_metrics, site_coords, on='grid_id', how='left')

    # Attach human-readable site names as a display column. Internal 'site' labels stay
    # 'site_<grid_id>' for all computation and figures.
    if site_name_lookup:
        site_metrics['site_name'] = site_metrics['grid_id'].map(
            lambda g: site_name_lookup.get(int(g), f'site_{int(g)}'))
    else:
        site_metrics['site_name'] = site_metrics['site']

    # Create rankings
    functional_cols = ['source_strength', 'sink_strength', 'self_recruitment']
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
        site_metrics[f'{metric}_rank'] = site_metrics[metric].rank(ascending=False, method='dense')

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
        'summary_stats': {
            'Number of species': n_species,
            'Study extent (sites)': study_extent,
            'Potential connectivity sites': potential_sites,
            'Realised connectivity sites': realised_connectivity_sites,
            'Unique connections': unique_connections,
        },
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

## Add a user-defined composite metric to network analysis results
def add_custom_composite(network_results, metrics, name='custom_composite', top_n=10, verbose=True):
    """
    Add a user-defined custom composite of connectivity metrics for use in network_analysis().

    The composite is the mean of the chosen metrics' RANK columns.
    The same equal-weight, scale-independent method as the built-in functional/centrality/overall composites.
    Lower = higher priority.
    This lets a study prioritise whichever aspects of connectivity are relevant.

    Parameters
    ----------
    network_results : dict
        A single result from network_analysis() (must contain 'site_metrics').
    metrics : list of str
        Any of: 'source_strength', 'sink_strength', 'self_recruitment', 'betweenness',
        'closeness', 'degree'. e.g. ['betweenness', 'closeness'].
    name : str
        Column name for the new composite (default 'custom_composite').
    top_n : int
        Size of the top-N table returned alongside.

    Returns
    -------
    dict
        The same network_results, with the new composite column added to 'site_metrics' and a 'top{top_n}_{name}' table added.
        Display it with create_network_map(..., composite_col=name) or create_matrix_with_map.
    """
    valid = ['source_strength', 'sink_strength', 'self_recruitment', 'betweenness', 'closeness', 'degree']
    chosen = [m.strip().lower().replace(' ', '_') for m in metrics]
    chosen = [m for m in chosen if m in valid]
    if not chosen:
        raise ValueError(f"No valid metrics. Choose from: {', '.join(valid)}")
    sm = network_results['site_metrics']
    sm[name] = sm[[f'{m}_rank' for m in chosen]].mean(axis=1)
    network_results[f'top{top_n}_{name}'] = sm.nsmallest(top_n, name).copy()
    if verbose:
        print(f"  Added composite '{name}' from: {', '.join(chosen)}")
    return network_results


## Create network map
def create_network_map(site_metrics, top_10, title="Directed connectivity network with hotspots",
                       composite_col='overall_composite', output_path=None,
                       zoom_extent=None, cmap='viridis_r',
                       output_format='png', dpi=300, show_figure=None,
                       network_graph=None, top_n_edges=200, network_only=False,
                       subtitle=None):
    """
        Hotspot/metric map (default) or network-only map.

        Hotspot/metric map (network_only=False): all sites are drawn as nodes coloured
        by rank (rank 1 = best); the top 10 are highlighted with larger dark-outlined
        circles and numbered labels. No edges, arrows or self-recruitment rings are
        drawn here, so the ranking stays legible.

        Network-only map (network_only=True): neutral site dots plus the directed
        connectivity structure , arrows (width/opacity by weight) for the strongest
        site-to-site links and rings for the strongest self-recruiting sites. The
        number of arrows/rings scales with the network so it works for small and large
        datasets alike; a figure caption states how many of each are shown.

        Parameters
        ----------
        top_10 : DataFrame
            Top 10 sites for the current metric (rows copied from site_metrics).
        title : str
            Figure title.
        composite_col : str
            Column in site_metrics used for colouring (usually a `{metric}_rank`
            or a `{type}_composite` column , both are rank-based).
        cmap : str
            Matplotlib colormap name. Default `viridis_r` so low rank = yellow = best.
        network_graph : networkx.DiGraph, optional
            Connectivity graph; its edges are overlaid only on the network-only map.
        top_n_edges : int
            Retained for backward compatibility. Edge counts are now chosen
            adaptively from the network size on the network-only map.
        """
    lon_range = site_metrics['longitude'].max() - site_metrics['longitude'].min()
    lat_range = site_metrics['latitude'].max() - site_metrics['latitude'].min()
    fig_width, fig_height, _ = calculate_figure_dimensions(lon_range, lat_range, base_size=6, n_panels=1)

    if network_only:
        fig = plt.figure(figsize=(12, 8.5))
        gs = GridSpec(1, 1, figure=fig)
    else:
        fig = plt.figure(figsize=(12, 10.5))
        gs = GridSpec(3, 1, height_ratios=[6, 0.3, 1.3],
                      hspace=0.4, figure=fig)

    # Main map
    ax = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())

    # Reduced buffer , less wasted ocean space
    if zoom_extent:
        ax.set_extent(zoom_extent, crs=ccrs.PlateCarree())
    else:
        _lon0, _lon1 = site_metrics['longitude'].min(), site_metrics['longitude'].max()
        _lat0, _lat1 = site_metrics['latitude'].min(), site_metrics['latitude'].max()
        # Border of ~5 grid cells (consistent across maps), inferred from the site spacing, so dots
        # fill the frame and the numbered top-10 labels have room.
        _lons_u = np.unique(np.round(site_metrics['longitude'].values, 4))
        _lats_u = np.unique(np.round(site_metrics['latitude'].values, 4))
        _cell_lon = float(np.min(np.diff(_lons_u))) if len(_lons_u) > 1 else 0.1
        _cell_lat = float(np.min(np.diff(_lats_u))) if len(_lats_u) > 1 else 0.1
        _buf_lon = 5 * _cell_lon
        _buf_lat = 5 * _cell_lat
        ax.set_extent([
            _lon0 - _buf_lon, _lon1 + _buf_lon,
            _lat0 - _buf_lat, _lat1 + _buf_lat
        ], crs=ccrs.PlateCarree())

    # Land only, softer outline, standardised across all maps (Fix 15)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='#555555')

    # Directed edges and self-recruitment rings are drawn ONLY on the network-only
    # map. Hotspot/metric maps stay clean (edges clutter the map and obscure the
    # rank colouring), so they draw no arrows, rings or connection legend entries.
    _n_between_total = _n_between_shown = 0
    _n_loops_total = _n_loops_shown = 0
    if network_only and network_graph is not None and len(network_graph.edges()) > 0:
        pos = {row['site']: (row['longitude'], row['latitude'])
               for _, row in site_metrics.iterrows()}
        edges_with_weight = [(u, v, d.get('weight', 1))
                             for u, v, d in network_graph.edges(data=True)
                             if u in pos and v in pos]
        edges_with_weight.sort(key=lambda e: e[2], reverse=True)
        # Separate self-recruitment (self-loops) from site-to-site edges and cap each
        # stream on its own, so strong self-recruitment never crowds out the directed
        # arrows. The caps scale with the network: small datasets show their whole
        # structure, large ones show the strongest links without becoming a hairball.
        _between = [e for e in edges_with_weight if e[0] != e[1]]
        _loops = [e for e in edges_with_weight if e[0] == e[1]]
        _arrow_cap = int(min(500, max(150, len(_between) * 0.05)))
        _loop_cap = int(min(60, max(25, len(_loops) * 0.10)))
        _shown_between = _between[:_arrow_cap]
        _shown_loops = _loops[:_loop_cap]
        _n_between_total, _n_between_shown = len(_between), len(_shown_between)
        _n_loops_total, _n_loops_shown = len(_loops), len(_shown_loops)

        # Arrow-head size scales with the map extent so arrows look proportional at any zoom.
        extent = max(lon_range, lat_range, 0.5)
        _arrow_scale = max(3, min(10, extent * 0.8))

        # Directed arrows: width AND opacity proportional to weight, so the strongest
        # connections dominate visually even on a large, densely-sampled domain.
        if _shown_between:
            _max_arrow_w = max(w for _, _, w in _shown_between)
            for u, v, w in _shown_between:
                _frac = w / _max_arrow_w
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                arrow = FancyArrowPatch(
                    (x1, y1), (x2, y2),
                    arrowstyle='-|>',
                    color='#2166AC',
                    alpha=min(0.9, 0.2 + _frac * 0.7),
                    linewidth=max(0.6, min(3.0, 0.6 + _frac * 2.4)),
                    mutation_scale=_arrow_scale,
                    zorder=4,
                )
                ax.add_patch(arrow)

        # Self-recruitment rings on top, opacity proportional to retention strength.
        if _shown_loops:
            _max_loop_w = max(w for _, _, w in _shown_loops)
            for u, v, w in _shown_loops:
                x, y = pos[u]
                ax.scatter(x, y, s=320, facecolors='none', edgecolors='#1f4e79',
                           linewidths=2.2, alpha=max(0.3, min(1.0, w / _max_loop_w)),
                           transform=ccrs.PlateCarree(), zorder=5)

    # All sites as outlined circles, coloured by rank
    if network_only:
        # No metric , all sites as small neutral dots, drawn UNDER the edges so the
        # connections stay visible even on large, densely-sampled domains. (Also
        # never references composite_col, avoiding the old None-metric crash, Fix 10.)
        scatter = ax.scatter(
            site_metrics['longitude'], site_metrics['latitude'],
            c='#888888', s=10, alpha=0.55,
            transform=ccrs.PlateCarree(),
            edgecolors='none', zorder=2,
        )
        score_col = None
        score_min = score_max = None
    else:
        base_size = 70

        # Rank everywhere: per-metric maps colour by the metric's rank (1 = best), matching the heatmap, every per-site figure shares one scale
        # zero are drawn grey ("metric absent") rather than at a misleading mid-rank colour
        _raw_metrics = {'source_strength', 'sink_strength', 'self_recruitment',
                        'betweenness', 'closeness', 'degree'}
        # Accept either the raw metric name or its '<metric>_rank' column (the viewers pass the rank
        # column). Either way, colour by rank and grey the sites whose RAW value is zero, so this map
        # matches the heatmap (which greys value-zero cells). This is what fixes the self-recruitment
        # map showing coloured dots where the heatmap correctly greyed them.
        _base_metric = composite_col[:-5] if str(composite_col).endswith('_rank') else composite_col
        if _base_metric in _raw_metrics and f'{_base_metric}_rank' in site_metrics.columns:
            score_col = f'{_base_metric}_rank'
            _zero_mask = site_metrics[_base_metric] == 0
            cmap = plt.cm.viridis_r  # 1 = best = yellow, consistent with the heatmap
        else:
            score_col = composite_col
            _zero_mask = pd.Series(False, index=site_metrics.index)

        _vis = site_metrics.loc[~_zero_mask, score_col]
        score_min, score_max = (_vis.min(), _vis.max()) if len(_vis) else (0, 1)

        # Grey "metric absent" sites first, underneath the coloured sites
        if _zero_mask.any():
            ax.scatter(
                site_metrics.loc[_zero_mask, 'longitude'],
                site_metrics.loc[_zero_mask, 'latitude'],
                c='#d9d9d9', s=base_size, alpha=0.85,
                transform=ccrs.PlateCarree(),
                edgecolors='#999999', linewidths=0.3, zorder=2,
            )

        scatter = ax.scatter(
            site_metrics.loc[~_zero_mask, 'longitude'],
            site_metrics.loc[~_zero_mask, 'latitude'],
            c=site_metrics.loc[~_zero_mask, score_col], cmap=cmap, s=base_size, alpha=0.85,
            transform=ccrs.PlateCarree(),
            edgecolors='#666666', linewidths=0.3,
            vmin=score_min, vmax=score_max, zorder=3,
        )

        # Top 10: larger + dark outline
        top10_ids = set(top_10['grid_id'].astype(int))
        top10_mask = site_metrics['grid_id'].astype(int).isin(top10_ids)
        top10_data = site_metrics[top10_mask]
        top10_size = base_size * 3.4

        if len(top10_data) > 0:
            ax.scatter(
                top10_data['longitude'], top10_data['latitude'],
                c=top10_data[score_col], cmap=cmap, s=top10_size, alpha=0.95,
                transform=ccrs.PlateCarree(),
                edgecolors='#1a1a1a', linewidths=1.8,
                vmin=score_min, vmax=score_max, zorder=5,
            )

        # Numbered top-10 labels with collision avoidance: start from a radial offset
        # off the top-10 centroid, then iteratively push each label away from every
        # other label AND from every hotspot marker until none overlap, keeping each
        # one clamped inside the axes. A thin leader line ties each number to its site.
        _ext = ax.get_extent(crs=ccrs.PlateCarree())
        _pad_x = (_ext[1] - _ext[0]) * 0.03
        _pad_y = (_ext[3] - _ext[2]) * 0.03
        _span = max(_ext[1] - _ext[0], _ext[3] - _ext[2])
        _off = max(0.03, _span * 0.045)
        _min_sep = _span * 0.05  # minimum spacing between labels and markers
        _markers = [(float(s['longitude']), float(s['latitude'])) for _, s in top_10.iterrows()]
        _cx = sum(m[0] for m in _markers) / len(_markers)
        _cy = sum(m[1] for m in _markers) / len(_markers)
        _labels = []
        for (mx, my) in _markers:
            _dx, _dy = mx - _cx, my - _cy
            _d = (_dx ** 2 + _dy ** 2) ** 0.5
            _ux, _uy = (_dx / _d, _dy / _d) if _d > 1e-9 else (0.0, 1.0)
            _labels.append([mx + _ux * _off, my + _uy * _off])
        for _ in range(80):
            for i in range(len(_labels)):
                _fx = _fy = 0.0
                for _px, _py in ([(_l[0], _l[1]) for j, _l in enumerate(_labels) if j != i] + _markers):
                    _dx = _labels[i][0] - _px
                    _dy = _labels[i][1] - _py
                    _dist = (_dx ** 2 + _dy ** 2) ** 0.5 or 1e-6
                    if _dist < _min_sep:
                        _push = (_min_sep - _dist) / _min_sep * _min_sep * 0.5
                        _fx += (_dx / _dist) * _push
                        _fy += (_dy / _dist) * _push
                _labels[i][0] = min(max(_labels[i][0] + _fx, _ext[0] + _pad_x), _ext[1] - _pad_x)
                _labels[i][1] = min(max(_labels[i][1] + _fy, _ext[2] + _pad_y), _ext[3] - _pad_y)
        for idx, ((mx, my), (lx, ly)) in enumerate(zip(_markers, _labels)):
            # Leader line + number sit above the markers and rings so labels stay readable.
            ax.plot([mx, lx], [my, ly], color='#333333', linewidth=0.6, alpha=0.7,
                    transform=ccrs.PlateCarree(), zorder=9)
            ax.text(lx, ly, f"{idx + 1}",
                    fontsize=11, fontweight='bold', fontname='Arial',
                    ha='center', va='center', color='#000000',
                    bbox=dict(boxstyle='circle,pad=0.15', fc='white', ec='#333333', lw=0.6, alpha=0.9),
                    transform=ccrs.PlateCarree(), zorder=10)

    if network_only:
        # Title + caption at the top (consistent with the matrix map). The legend
        # sits at the bottom, so the caption can never overlap it.
        fig.suptitle(title, fontsize=13, fontweight='bold', y=0.97)
        if _n_between_total or _n_loops_total:
            _cap_txt = f"Arrows: top {_n_between_shown} of {_n_between_total} site-to-site links"
            if _n_loops_total:
                _cap_txt += f"      Rings: top {_n_loops_shown} of {_n_loops_total} self-recruiting sites"
            fig.text(0.5, 0.925, _cap_txt, ha='center', va='top',
                     fontsize=9, style='italic', color='#444444')
        fig.subplots_adjust(top=0.90, bottom=0.20)
    else:
        # Grey notes under the title: an optional caller-supplied subtitle (e.g. which metric the
        # Type 3 comparison maps show) and/or the zero-value/grey-sites note. When any note is
        # shown, push the title up so the note has its own space beneath it and never overlaps.
        _notes = []
        if subtitle:
            _notes.append(subtitle)
        if _zero_mask.any():
            _notes.append('Grey sites have a value of zero for this metric (not included in the rank)')
        if _notes:
            ax.set_title(title, fontsize=12, fontweight='bold', pad=(28 if len(_notes) == 1 else 40))
            ax.text(0.5, 1.01, '\n'.join(_notes),
                    transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=9, style='italic', color='#666666')
        else:
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    # Dotted lighter gridlines, standardised (Fix 15)
    gl = ax.gridlines(draw_labels=True, linestyle=':', linewidth=0.5, color='gray', alpha=0.4)
    gl.top_labels = False
    gl.right_labels = False

    # Legend , entries depend on the map type.
    legend_handles = []
    if network_only:
        # Network-only map: neutral sites, self-recruitment rings and connection strengths.
        legend_handles.append(Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor='#888888', markeredgecolor='none',
                                     markersize=8, label='Site'))
        legend_handles.append(Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor='none', markeredgecolor='#1f4e79',
                                     markersize=12, markeredgewidth=2.0, label='Self-recruitment'))
        legend_handles.append(Line2D([0], [0], color='#2166AC', linewidth=2.8, alpha=0.85, label='Strong connection'))
        legend_handles.append(Line2D([0], [0], color='#2166AC', linewidth=1.8, alpha=0.5, label='Medium connection'))
        legend_handles.append(Line2D([0], [0], color='#2166AC', linewidth=1.0, alpha=0.25, label='Weak connection'))
        # Two columns: [Site, Self-recruitment] in the left column, [Strong, Medium, Weak
        # connection] in the right column. matplotlib fills a multi-column legend COLUMN-major
        # (top-to-bottom down the first column, then the second), so the handle order is simply
        # left-column entries first, then right-column entries. A blank spacer pads the shorter
        # left column so the two columns stay top-aligned.
        _spacer = Line2D([0], [0], marker='', color='none', label=' ')
        legend_handles = [legend_handles[0], legend_handles[1], _spacer,   # left column
                          legend_handles[2], legend_handles[3], legend_handles[4]]  # right column
        ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  ncol=2, fontsize=10, framealpha=0.9, handletextpad=0.8, columnspacing=1.6)
    else:
        # Hotspot/metric map: site rank and top-10 only. No edges are drawn here, so
        # there are no connection or self-recruitment entries.
        legend_handles.append(Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor='#440154', markeredgecolor='#666666',
                                     markersize=10, markeredgewidth=0.3, label='Site (colour = rank)'))
        legend_handles.append(Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor='#fde725', markeredgecolor='#1a1a1a',
                                     markersize=13, markeredgewidth=1.5, label='Top 10 hotspot (numbered)'))
        # Legend goes in its own bottom slot (gs[2]) so it never overlaps the colourbar.
        legend_ax = fig.add_subplot(gs[2])
        legend_ax.axis('off')
        legend_ax.legend(handles=legend_handles, loc='center', ncol=2, fontsize=12,
                         framealpha=0.9, handletextpad=0.5, columnspacing=2.0)


    if not network_only:
        # Colorbar in the middle slot
        cbar_ax = fig.add_subplot(gs[1])
        cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
        if composite_col.endswith('_composite'):
            cbar_label_text = 'Average rank across metrics (1 = best, lower is better)'
        else:
            # Single-metric and rank maps both colour by RANK now (1 = best); grey = value of zero, not included in the rank
            cbar_label_text = 'Rank (1 = best, lower is better; grey = value of zero, not included in the rank)'
        cbar.set_label(cbar_label_text, fontsize=12, labelpad=6)
        cbar.ax.tick_params(labelsize=11)

    if output_path:
        fig.savefig(output_path, format=output_format, dpi=dpi, bbox_inches='tight')
        print(f"  Figure saved to: {output_path}")

    if show_figure is None or show_figure:
        plt.show()
    else:
        plt.close(fig)

    return fig


## Create a connectivity matrix
def create_connectivity_matrix_plot(connectivity_matrix, title="Connectivity Matrix",
                                    output_path=None, log_scale=True,
                                    output_format='png', dpi=300,
                                    filter_empty=True, show_figure=None):
    """
    Create a connectivity matrix heatmap visualisation (like Melia et al. 2016).

    Shows source sites (rows) vs destination sites (columns) with colour
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

    fig, ax = plt.subplots(figsize=(9, 9))

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

    # Create heatmap with RdBu_r diverging colormap (standardised across ComCon figures)
    cmap = plt.cm.get_cmap('RdBu_r').copy()
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

    # Match the live combined matrix viewer (create_matrix_with_map): invert the y-axis so the
    # first source sits at the bottom-left origin and the self-recruitment diagonal runs
    # bottom-left -> top-right. Skip the diagonal when there are fewer than two sites, where the
    # endpoints would be degenerate.
    ax.invert_yaxis()
    if n_sites >= 2:
        ax.plot([0, n_sites - 1], [0, n_sites - 1], 'k--', alpha=0.7, linewidth=1.2)
    plt.tight_layout()

    # Save if output path provided
    if output_path:
        fig.savefig(output_path, format=output_format, dpi=dpi, bbox_inches='tight')
        print(f"  Figure saved to: {output_path}")

    if show_figure is None or show_figure:
        plt.show()
    else:
        plt.close(fig)

    return fig

## Add a map for spatial reference
def create_matrix_with_map(connectivity_matrix, site_metrics,
                           title="Connectivity Matrix with Site Map",
                           output_path=None, log_scale=True,
                           output_format='png', dpi=300,
                           filter_empty=True, show_figure=None):
    """
    Create connectivity matrix heatmap with a geographic reference map side by side.
    The map shows site locations with their grid_id labels so users can
    orientate themselves on the matrix.
    """
    # Filter empty sites if requested
    if filter_empty:
        row_sums = connectivity_matrix.sum(axis=1)
        col_sums = connectivity_matrix.sum(axis=0)
        connected_sites = (row_sums > 0) | (col_sums > 0)
        connectivity_matrix = connectivity_matrix.loc[connected_sites, connected_sites]

    n_sites = len(connectivity_matrix)

    # Map matrix axis labels (site_<grid_id>) to display names: use the real
    # site_name when one was supplied, otherwise fall back to site_<grid_id>.
    # (site_name already equals site_<grid_id> when no real names were given.)
    _name_by_gid = {}
    if 'site_name' in site_metrics.columns and 'grid_id' in site_metrics.columns:
        for _, _r in site_metrics.iterrows():
            _name_by_gid[int(_r['grid_id'])] = str(_r['site_name'])

    def _disp_label(_lbl):
        _g = str(_lbl).replace('site_', '')
        try:
            return _name_by_gid.get(int(_g), str(_lbl))
        except (TypeError, ValueError):
            return str(_lbl)

    def _wrap_label(_txt):
        # Long combined site names (several sites sharing one grid cell, joined by ' / ')
        # are stacked onto separate lines so rotated axis labels stay short and do not
        # overlap the colourbar, and map labels stay compact.
        return str(_txt).replace(' / ', '/\n')

    # Cap the number of tick labels to a fixed target so the axes and the map stay readable at any site count
    _max_tick_labels = 30
    label_every = max(1, int(np.ceil(n_sites / _max_tick_labels)))

    # Standardised landscape aspect ratio across all four Type 2 viewers
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.35)

    # Left: Connectivity matrix
    ax_matrix = fig.add_subplot(gs[0])
    matrix_data = connectivity_matrix.values.astype(float)

    if log_scale:
        matrix_data = np.log10(matrix_data + 1)
        cbar_label = 'Connectivity Strength (log10)'
    else:
        cbar_label = 'Connectivity Strength'

    masked_data = np.ma.masked_where(matrix_data == 0, matrix_data)
    cmap_matrix = plt.cm.get_cmap('RdBu_r').copy()
    cmap_matrix.set_bad('lightgray')

    im = ax_matrix.imshow(masked_data, cmap=cmap_matrix, aspect='equal', interpolation='nearest')

    # Colorbar full width at the very bottom, clear of the matrix x-axis label
    cbar_ax = fig.add_axes([0.08, 0.02, 0.84, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(cbar_label, fontsize=13, labelpad=8)
    cbar.ax.tick_params(labelsize=11)

    # Label every Nth site
    label_indices = list(range(0, n_sites, label_every))
    ax_matrix.set_xticks(label_indices)
    ax_matrix.set_yticks(label_indices)
    ax_matrix.set_xticklabels([_wrap_label(_disp_label(connectivity_matrix.columns[i])) for i in label_indices], rotation=90, fontsize=9.5)
    ax_matrix.set_yticklabels([_wrap_label(_disp_label(connectivity_matrix.index[i])) for i in label_indices], fontsize=9.5)

    ax_matrix.set_xlabel('Sink Site', fontsize=13, labelpad=8)
    ax_matrix.set_ylabel('Source Site', fontsize=13, labelpad=8)
    ax_matrix.set_title('Connectivity Matrix', fontsize=12, fontweight='bold', pad=10)
    ax_matrix.invert_yaxis()
    # Only draw the self-recruitment diagonal when there are at least two sites; on an empty
    # (fully filtered) matrix the endpoints collapse and matplotlib would draw a misleading line.
    if n_sites >= 2:
        ax_matrix.plot([0, n_sites - 1], [0, n_sites - 1], 'k--', alpha=0.3, linewidth=1)

    # Right: Reference map
    ax_map = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())

    # Get coordinates for connected sites
    site_ids_in_matrix = [s.replace('site_', '') for s in connectivity_matrix.index]
    map_sites = site_metrics[site_metrics['grid_id'].astype(str).isin(site_ids_in_matrix)].copy()

    if len(map_sites) == 0:
        # Fallback: try matching differently
        map_sites = site_metrics.copy()

    _mlon = np.unique(np.round(map_sites['longitude'].values, 4))
    _mlat = np.unique(np.round(map_sites['latitude'].values, 4))
    buffer = 5 * min(float(np.min(np.diff(_mlon))) if len(_mlon) > 1 else 0.1,
                     float(np.min(np.diff(_mlat))) if len(_mlat) > 1 else 0.1)
    ax_map.set_extent([
        map_sites['longitude'].min() - buffer,
        map_sites['longitude'].max() + buffer,
        map_sites['latitude'].min() - buffer,
        map_sites['latitude'].max() + buffer
    ], crs=ccrs.PlateCarree())

    ax_map.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
    ax_map.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5, color='#555555')

    # Uniform fill colour. No latitude gradient , the colour isn't encoding anything meaningful.
    ax_map.scatter(
        map_sites['longitude'], map_sites['latitude'],
        c='#7b9ec9', s=40,
        transform=ccrs.PlateCarree(), edgecolors='#2a4a6b', linewidth=0.4,
        zorder=3
    )

    # Label the same sites that are labelled on the matrix axes
    # The matrix index may be 'site_<grid_id>', a bare grid_id or a real site_name
    # Resolve each labelled position back to a grid_id
    _name_to_gid = {}
    if 'grid_id' in site_metrics.columns:
        for _c in ('site_name', 'site'):
            if _c in site_metrics.columns:
                for _, _r in site_metrics.iterrows():
                    _name_to_gid[str(_r[_c])] = int(_r['grid_id'])

    def _index_to_gid(_lbl):
        _g = str(_lbl).replace('site_', '')
        try:
            return int(_g)
        except (TypeError, ValueError):
            return _name_to_gid.get(str(_lbl))

    labelled_gids = set()
    for i in range(0, n_sites, label_every):
        _gid = _index_to_gid(connectivity_matrix.index[i])
        if _gid is not None:
            labelled_gids.add(_gid)

    # Label the same sites that are labelled on the matrix axes, placed directly on their
    # markers (no offset / leader lines) so this reference map matches the heatmap's exactly.
    _lbl_sites = [s for _, s in map_sites.iterrows() if int(s['grid_id']) in labelled_gids]
    for site in _lbl_sites:
        mx, my = float(site['longitude']), float(site['latitude'])
        ax_map.text(
            mx, my,
            _wrap_label(_disp_label('site_' + str(int(site['grid_id'])))), fontsize=6, ha='center', va='center',
            transform=ccrs.PlateCarree(), zorder=4,
            bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.85, edgecolor='none')
        )

    ax_map.set_title('Site Reference Map', fontsize=12, fontweight='bold', pad=10)
    gl = ax_map.gridlines(draw_labels=True, linestyle=':', linewidth=0.5, color='gray', alpha=0.4)
    gl.top_labels = False
    gl.right_labels = False

    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.99)
    _note = (f"Showing {n_sites} site{'s' if n_sites != 1 else ''} with at least one connection and "
             f"sites with no imported or exported particles are hidden.")
    fig.text(0.5, 0.935, _note, ha='center', va='top', fontsize=11, style='italic', color='#444444')
    fig.subplots_adjust(top=0.91, bottom=0.24, left=0.08, right=0.97, wspace=0.35)

    if output_path:
        fig.savefig(output_path, format=output_format, dpi=dpi, bbox_inches='tight')
        print(f"  Figure saved to: {output_path}")

    if show_figure is None or show_figure:
        plt.show()
    else:
        plt.close(fig)

    return fig

## Create a map of Louvain community detection groupings for connectivity
def create_community_map(network_results, title="Connectivity Communities (Louvain)",
                         output_path=None, output_format='png', dpi=300,
                         show_figure=None, edge_scale=0.05):
    """
    Detect and visualise connectivity communities using Louvain clustering.

    Nodes are coloured by community membership. Edges show directed connectivity.
    Based on Tony Knights' code (cells 93-94).

    Parameters
    ----------
    network_results : dict
        Output from network_analysis()
    """
    if not HAS_LOUVAIN:
        print("  Community detection requires python-louvain.")
        print("  Install with: pip install python-louvain")
        return None

    G = network_results['network_graph']
    site_metrics = network_results['site_metrics']

    if len(G.nodes()) < 3:
        print("  Too few nodes for community detection.")
        return None

    # Build position dict
    pos = {}
    for _, row in site_metrics.iterrows():
        pos[row['site']] = (row['longitude'], row['latitude'])

    nodes_with_pos = [n for n in G.nodes() if n in pos]
    if len(nodes_with_pos) < 3:
        print("  Too few nodes with coordinates for community detection.")
        return None

    # Louvain community detection (requires undirected graph)
    G_undirected = G.to_undirected()
    # Filter to nodes with positions
    G_sub = G_undirected.subgraph(nodes_with_pos)
    partition = community_louvain.best_partition(G_sub)

    num_communities = len(set(partition.values()))

    # Order communities by size (largest first) and give EACH a distinct colour, so no two share a
    # colour even at 100+ communities. Up to 40 use the well-separated tab20/tab20b palettes; beyond
    # that hues are spaced evenly around the colour wheel (with alternating shade) so every community
    # is unique. No sites are greyed - every community keeps its own colour and dot size.
    import matplotlib.colors as _mcol
    _sizes = pd.Series([partition.get(n, 0) for n in nodes_with_pos]).value_counts()
    _by_size = list(_sizes.index)                        # community ids, largest first
    _size_rank = {c: i for i, c in enumerate(_by_size)}  # community id -> size rank (0 = largest)
    if num_communities <= 40:
        _base = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
        _colours = [_base[i] for i in range(num_communities)]
    else:
        _hsv = np.zeros((num_communities, 3))
        _hsv[:, 0] = np.linspace(0.0, 1.0, num_communities, endpoint=False)
        _hsv[:, 1] = np.where(np.arange(num_communities) % 2 == 0, 0.85, 0.60)
        _hsv[:, 2] = np.where((np.arange(num_communities) // 2) % 2 == 0, 0.95, 0.75)
        _colours = [tuple(c) for c in _mcol.hsv_to_rgb(_hsv)]

    def _community_colour(c):
        return _colours[_size_rank.get(int(c), 0)]

    # Standardised landscape aspect ratio across all four Type 2 viewers
    fig, ax = plt.subplots(figsize=(12, 8),
                           subplot_kw={'projection': ccrs.PlateCarree()})
    plt.rcParams['font.family'] = 'Arial'

    # Border of ~5 grid cells (consistent with the other maps), inferred from the site spacing.
    _lon0, _lon1 = site_metrics['longitude'].min(), site_metrics['longitude'].max()
    _lat0, _lat1 = site_metrics['latitude'].min(), site_metrics['latitude'].max()
    _lons_u = np.unique(np.round(site_metrics['longitude'].values, 4))
    _lats_u = np.unique(np.round(site_metrics['latitude'].values, 4))
    _buf_lon = 5 * (float(np.min(np.diff(_lons_u))) if len(_lons_u) > 1 else 0.1)
    _buf_lat = 5 * (float(np.min(np.diff(_lats_u))) if len(_lats_u) > 1 else 0.1)
    ax.set_extent([
        _lon0 - _buf_lon, _lon1 + _buf_lon,
        _lat0 - _buf_lat, _lat1 + _buf_lat
    ], crs=ccrs.PlateCarree())

    # Softer land outline, standardised with other maps (Fix 15)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='#555555')

    # Draw nodes coloured by community
    node_lons = [pos[n][0] for n in nodes_with_pos]
    node_lats = [pos[n][1] for n in nodes_with_pos]
    node_colors = [_community_colour(partition.get(n, 0)) for n in nodes_with_pos]

    ax.scatter(node_lons, node_lats, c=node_colors, s=50, alpha=0.9,
               edgecolors='#333333', linewidths=0.4, zorder=2,
               transform=ccrs.PlateCarree())

    # Legend: the biggest 20 communities (1 = largest), coloured to match the map. Any beyond 20 are
    # still coloured on the map; a grey italic note under the title says so.
    _n_legend = min(num_communities, 20)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               label=f'Community {i + 1}',
               markerfacecolor=_colours[i], markeredgecolor='#333333',
               markersize=11, markeredgewidth=0.4)
        for i in range(_n_legend)
    ]

    n_legend_items = len(legend_elements)
    n_cols = min(5, max(3, (n_legend_items + 1) // 2))
    ax.legend(handles=legend_elements,
              loc='upper center', bbox_to_anchor=(0.5, -0.06),
              fontsize=11, framealpha=0.9,
              title='Communities', title_fontsize=12,
              ncol=n_cols, columnspacing=1.2, handletextpad=0.4)

    # When the "biggest N of M communities" note is shown, push the title up so the note has
    # its own space beneath it and never overlaps the title.
    if num_communities > _n_legend:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=28)
        ax.text(0.5, 1.01,
                f'Biggest {_n_legend} of {num_communities} communities shown in the legend; all are coloured on the map',
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=9, style='italic', color='#666666')
    else:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, color='gray', alpha=0.4)
    gl.top_labels = False
    gl.right_labels = False

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, format=output_format, dpi=dpi, bbox_inches='tight')
        print(f"  Figure saved to: {output_path}")

    if show_figure is None or show_figure:
        plt.show()
    else:
        plt.close(fig)

    # Store community assignments in site_metrics for export
    for node, comm in partition.items():
        mask = site_metrics['site'] == node
        site_metrics.loc[mask, 'louvain_community'] = int(comm)

    return fig, partition

## Compute and store density, modularity, and community count in the results dict
def _compute_network_stats(network_results):
    """
    Compute density, modularity, and community count from a `network_graph`
    stored in `network_results`, and stash them back into the same dict.

    Called once per `network_analysis()` result (community or per-species).
    The values are displayed by the Louvain community viewer.
    """
    try:
        G = network_results.get('network_graph')
        if G is None:
            return
        n = G.number_of_nodes()
        n_edges = G.number_of_edges()
        density = (n_edges / (n * (n - 1))) if n > 1 else 0.0
        network_results['network_density'] = round(density, 4)
        network_results['network_edges'] = int(n_edges)
        network_results['network_nodes'] = int(n)
        try:
            partition = community_louvain.best_partition(G.to_undirected())
            modularity = community_louvain.modularity(partition, G.to_undirected())
            network_results['network_modularity'] = round(modularity, 4)
            network_results['network_communities'] = len(set(partition.values()))
        except Exception:
            pass
    except Exception:
        pass

## Build level and scale option lists for viewer dropdown widgets
def _get_viewer_options(all_network_results):
    """
    Build level and scale options from the nested results dict.

    Parameters
    ----------
    all_network_results : dict
        Nested dict: {scale_name: {'community': results, 'per_species': {sp: results}}}.

    Returns
    -------
    levels : list of str
        Available analysis levels (e.g. ['Community', 'Patella vulgata']).
    scales : list of str
        Available scale names (e.g. ['full_domain', '30_km']).
    level_lookup : dict
        Maps display name back to (key_type, key_value) for result lookup.
    """
    levels = []
    level_lookup = {}
    for scale_name, scale_data in all_network_results.items():
        if 'community' in scale_data and 'Community' not in levels:
            levels.append('Community')
            level_lookup['Community'] = ('community', None)
        if 'per_species' in scale_data:
            for sp in scale_data['per_species']:
                sp_display = display_species_name(sp)
                if sp_display not in levels:
                    levels.append(sp_display)
                    level_lookup[sp_display] = ('per_species', sp)
    scales = list(all_network_results.keys())
    return levels, scales, level_lookup

## Helps to display the results using the dropdown
def _lookup_results(all_network_results, level_lookup, level, scale):
    """
    Look up `network_results` from the nested all_network_results dict.

    Returns None if the level-scale combination does not exist.
    """
    scale_data = all_network_results.get(scale)
    if scale_data is None:
        return None
    key_type, key_value = level_lookup[level]
    if key_type == 'community':
        return scale_data.get('community')
    else:
        return scale_data.get('per_species', {}).get(key_value)

## Actually builds the dropdown
def _make_viewer_dropdowns(levels, scales):
    """
    Build the shared level and scale dropdowns used by all Path B viewers.
    """
    level_dd = widgets.Dropdown(
        options=levels,
        value=levels[0],
        description='Level:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='400px'),
    )
    scale_dd = widgets.Dropdown(
        options=scales,
        value=scales[0],
        description='Scale:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='400px'),
    )
    return level_dd, scale_dd

## Site name offset so labels don't overlap
def _offset_point_labels(ax, markers):
    """Return slightly-offset label positions for a set of marker coordinates.

    This is the exact same gentle radial offset used for the numbered hotspot
    labels: each label starts just off its marker (pointing away from the centre
    of the cluster) and is nudged only enough to reduce overlap. `markers` is a
    list of (lon, lat) tuples; returns matching [lon, lat] label positions.
    """
    if not markers:
        return []
    _ext = ax.get_extent(crs=ccrs.PlateCarree())
    _pad_x = (_ext[1] - _ext[0]) * 0.03
    _pad_y = (_ext[3] - _ext[2]) * 0.03
    _span = max(_ext[1] - _ext[0], _ext[3] - _ext[2])
    _off = max(0.03, _span * 0.045)
    _min_sep = _span * 0.05
    _cx = sum(m[0] for m in markers) / len(markers)
    _cy = sum(m[1] for m in markers) / len(markers)
    _labels = []
    for (mx, my) in markers:
        _dx, _dy = mx - _cx, my - _cy
        _d = (_dx ** 2 + _dy ** 2) ** 0.5
        _ux, _uy = (_dx / _d, _dy / _d) if _d > 1e-9 else (0.0, 1.0)
        _labels.append([mx + _ux * _off, my + _uy * _off])
    _mk = list(markers)
    for _ in range(80):
        for i in range(len(_labels)):
            _fx = _fy = 0.0
            for _px, _py in ([(_l[0], _l[1]) for j, _l in enumerate(_labels) if j != i] + _mk):
                _dx = _labels[i][0] - _px
                _dy = _labels[i][1] - _py
                _dist = (_dx ** 2 + _dy ** 2) ** 0.5 or 1e-6
                if _dist < _min_sep:
                    _push = (_min_sep - _dist) / _min_sep * _min_sep * 0.5
                    _fx += (_dx / _dist) * _push
                    _fy += (_dy / _dist) * _push
            _labels[i][0] = min(max(_labels[i][0] + _fx, _ext[0] + _pad_x), _ext[1] - _pad_x)
            _labels[i][1] = min(max(_labels[i][1] + _fy, _ext[2] + _pad_y), _ext[3] - _pad_y)
    return _labels

## Site × metric heatmap figure builder
def build_heatmap_figure(network_results, title=None):
    """
    Build the site × metric heatmap with a reference map panel stacked below.

    Rows are metrics (source, sink, self-recruitment, betweenness, closeness,
    degree), columns are sites (top 100 by overall composite rank). Cells are
    coloured by rank using viridis_r (rank 1 = best = yellow).

    Returns the matplotlib Figure, or None if the required rank columns are
    not present in `network_results['site_metrics']`.
    """
    site_metrics = network_results['site_metrics']
    metric_cols = ['source_strength', 'sink_strength', 'self_recruitment',
                   'betweenness', 'closeness', 'degree']
    metric_labels = ['Source', 'Sink', 'Self-recruit.',
                     'Betweenness', 'Closeness', 'Degree']

    rank_cols = [f'{m}_rank' for m in metric_cols]
    available = [c for c in rank_cols if c in site_metrics.columns]
    if not available:
        return None

    n_sites = len(site_metrics)
    cap = min(100, n_sites)

    if 'overall_composite' in site_metrics.columns:
        selected = site_metrics.sort_values('overall_composite').iloc[:cap].copy()
    else:
        selected = site_metrics.iloc[:cap].copy()

    selected['__gid_int'] = selected['grid_id'].astype(int)
    top_sites = selected.sort_values('__gid_int').drop(columns='__gid_int')

    rank_matrix_t = top_sites[available].values.astype(float).T

    # Rank colouring stays (consistent with the per-metric maps), but cells whose raw metric value is exactly zero are greyed out
    # Zeros otherwise tie at a middling rank and read as "average" when they really mean "metric absent" (e.g. sites that never self-recruit)
    value_cols = [c.replace('_rank', '') for c in available]
    value_matrix_t = top_sites[value_cols].values.astype(float).T
    rank_matrix_t = np.ma.masked_where(value_matrix_t == 0, rank_matrix_t)
    # Use real site names on the axis when supplied (site_name already falls back
    # to site_<grid_id> when no names were given).
    if 'site_name' in top_sites.columns:
        site_labels = top_sites['site_name'].astype(str).values
    else:
        site_labels = top_sites['site'].astype(str).values

    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 1, height_ratios=[1.2, 0.05, 1.3], hspace=0.5)

    ax_hm = fig.add_subplot(gs[0])
    _hm_cmap = plt.cm.viridis_r.copy()
    _hm_cmap.set_bad(color='#d9d9d9')  # raw-value-zero cells render grey
    im = ax_hm.imshow(rank_matrix_t, cmap=_hm_cmap, aspect='auto')

    ax_hm.set_yticks(range(len(available)))
    ax_hm.set_yticklabels(
        [metric_labels[metric_cols.index(c.replace('_rank', ''))] for c in available],
        fontsize=11,
    )
    # Thin x-axis labels to the same set used on the inset map below, so the labelled sites are identical on the axis and the reference map (read a grid_id off either, find it on the other)
    _label_target = 30  # number of labels shown on both the axis and the inset; matches the matrix
    label_step = max(1, cap // _label_target)
    _tick_pos = list(range(0, cap, label_step))
    ax_hm.set_xticks(_tick_pos)
    # Stack long combined site names (several sites in one grid cell, joined by ' / ') onto
    # separate lines so the rotated labels stay short and do not overlap the colourbar below.
    ax_hm.set_xticklabels([str(site_labels[i]).replace(' / ', '/\n') for i in _tick_pos], fontsize=7, rotation=90)

    cbar_ax = fig.add_subplot(gs[1])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Rank (1 = best; grey = value of zero, not included in the rank)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    heatmap_title = title or "Site × metric heatmap"
    fig.suptitle(heatmap_title, fontsize=14, fontweight='bold', y=0.98)
    # Always-shown descriptive caption (like the connectivity-matrix figure). The colour bar
    # already explains the rank scale and grey cells, so this only describes what is shown.
    _hm_caption = "Sites (columns) × network metrics (rows); cell colour = that site's rank for that metric."
    if n_sites > cap:
        _hm_caption += (f" Showing the top {cap} of {n_sites} sites by overall composite rank, "
                        f"in grid_id order (full rankings in the exported network_site_metrics.csv).")
    else:
        _hm_caption += " All sites shown, in grid_id order."
    fig.text(0.5, 0.945, _hm_caption, ha='center', va='top',
             fontsize=10, style='italic', color='#444444', wrap=True)

    ax_map = fig.add_subplot(gs[2], projection=ccrs.PlateCarree())

    # Plot all connected sites as dots so this map matches the reference map in create_matrix_with_map
    # Labels stay thinned below
    map_sites = site_metrics.copy()
    if len(map_sites) == 0:
        map_sites = top_sites.copy()

    _lon0, _lon1 = map_sites['longitude'].min(), map_sites['longitude'].max()
    _lat0, _lat1 = map_sites['latitude'].min(), map_sites['latitude'].max()
    _mlon = np.unique(np.round(map_sites['longitude'].values, 4))
    _mlat = np.unique(np.round(map_sites['latitude'].values, 4))
    _buf_lon = 5 * (float(np.min(np.diff(_mlon))) if len(_mlon) > 1 else 0.1)
    _buf_lat = 5 * (float(np.min(np.diff(_mlat))) if len(_mlat) > 1 else 0.1)
    ax_map.set_extent([
        _lon0 - _buf_lon, _lon1 + _buf_lon,
        _lat0 - _buf_lat, _lat1 + _buf_lat
    ], crs=ccrs.PlateCarree())

    ax_map.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
    ax_map.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5, color='#555555')

    ax_map.scatter(
        map_sites['longitude'], map_sites['latitude'],
        c='#7b9ec9', s=40,
        transform=ccrs.PlateCarree(),
        edgecolors='#2a4a6b', linewidth=0.4,
        zorder=3,
    )

    # Reuse label_step from the x-axis above so the inset labels exactly the sites labelled on the heatmap axis
    axis_grid_ids = top_sites['grid_id'].astype(int).tolist()
    label_ids = set(axis_grid_ids[::label_step])
    _name_by_gid = (dict(zip(map_sites['grid_id'].astype(int), map_sites['site_name'].astype(str)))
                    if 'site_name' in map_sites.columns else {})
    # Label the heatmap sites directly on their markers (no offset / leader lines).
    _lbl_sites = [s for _, s in map_sites.iterrows() if int(s['grid_id']) in label_ids]
    for site in _lbl_sites:
        gid_int = int(site['grid_id'])
        mx, my = float(site['longitude']), float(site['latitude'])
        ax_map.text(
            mx, my,
            str(_name_by_gid.get(gid_int, gid_int)), fontsize=6, ha='center', va='center',
            transform=ccrs.PlateCarree(), zorder=4,
            bbox=dict(boxstyle='round,pad=0.1',
                      facecolor='white', alpha=0.85, edgecolor='none'),
        )

    ax_map.set_title('Site Reference Map', fontsize=13, fontweight='bold', pad=10)
    gl = ax_map.gridlines(draw_labels=True, linestyle=':', linewidth=0.5, color='gray', alpha=0.4)
    gl.top_labels = False
    gl.right_labels = False

    return fig

# Path B viewers
## Matirx viewer
def trait_matrix_viewer(all_network_results):
    """
    Interactive connectivity matrix viewer for the trait-defined pathway.

    Two dropdowns: analysis level (Community or species) and spatial scale.
    """
    levels, scales, level_lookup = _get_viewer_options(all_network_results)
    if not levels or not scales:
        return
    level_dd, scale_dd = _make_viewer_dropdowns(levels, scales)
    output = widgets.Output()

    ## Redraw the connectivity matrix when level or scale dropdown changes
    def update(*_):
        """Re-render the connectivity matrix for the currently selected level and scale."""
        nr = _lookup_results(all_network_results, level_lookup,
                             level_dd.value, scale_dd.value)
        with output:
            clear_output(wait=True)
            if nr is None:
                print(f"  No results for {level_dd.value} at {scale_dd.value}.")
                return
            print(f"  Rendering {level_dd.value} , {scale_dd.value}: "
                  f"connectivity matrix... (~10-20 seconds)")
            try:
                fig = create_matrix_with_map(
                    nr['connectivity_matrix'],
                    nr['site_metrics'],
                    title=f"{level_dd.value} , {scale_dd.value}",
                    show_figure=False,
                )
                clear_output(wait=True)
                display(fig)
                plt.close(fig)
            except Exception as e:
                print(f"  Could not display matrix: {e}")

    level_dd.observe(update, names='value')
    scale_dd.observe(update, names='value')
    display(level_dd, scale_dd, output)
    update()  # render immediately (synchronous; a Timer thread fails to draw into the Output)

## Louvain viewer
def trait_community_viewer(all_network_results):
    """
    Interactive Louvain community map viewer for the trait-defined pathway.

    Shows sites coloured by community membership, with network-level stats
    (density, modularity, number of communities) displayed below.
    Two dropdowns: analysis level and spatial scale.
    """
    levels, scales, level_lookup = _get_viewer_options(all_network_results)
    if not levels or not scales:
        return
    level_dd, scale_dd = _make_viewer_dropdowns(levels, scales)
    output = widgets.Output()

    ## Redraw the community map when level or scale dropdown changes
    def update(*_):
        """Re-render the Louvain community map for the currently selected level and scale."""
        nr = _lookup_results(all_network_results, level_lookup,
                             level_dd.value, scale_dd.value)
        with output:
            clear_output(wait=True)
            if nr is None:
                print(f"  No results for {level_dd.value} at {scale_dd.value}.")
                return
            print(f"  Rendering {level_dd.value} , {scale_dd.value}: "
                  f"Louvain communities... (~10-20 seconds)")
            try:
                result = create_community_map(
                    nr,
                    title=f"{level_dd.value} , {scale_dd.value}",
                    show_figure=False,
                )
                fig = result[0] if isinstance(result, tuple) else result
                clear_output(wait=True)
                display(fig)
                plt.close(fig)
                # Network stats table
                stats_rows = []
                for label, key in [
                    ('Nodes', 'network_nodes'),
                    ('Edges', 'network_edges'),
                    ('Density', 'network_density'),
                    ('Communities', 'network_communities'),
                    ('Modularity', 'network_modularity'),
                ]:
                    v = nr.get(key)
                    if v is None:
                        continue
                    if key in ('network_density', 'network_modularity'):
                        stats_rows.append((label, f"{float(v):.3f}"))
                    else:
                        stats_rows.append((label, f"{int(v)}"))
                if stats_rows:
                    stats_df = pd.DataFrame(stats_rows,
                                            columns=['Network statistic', 'Value'])
                    stats_df.index = range(1, len(stats_df) + 1)
                    print()
                    print()
                    display(stats_df)
            except Exception as e:
                print(f"  Could not display community map: {e}")

    level_dd.observe(update, names='value')
    scale_dd.observe(update, names='value')
    display(level_dd, scale_dd, output)
    update()

## Heatmap viewer
def trait_heatmap_viewer(all_network_results):
    """
    Interactive site × metric heatmap viewer for the trait-defined pathway.

    Two dropdowns: analysis level and spatial scale.
    """
    levels, scales, level_lookup = _get_viewer_options(all_network_results)
    if not levels or not scales:
        return
    level_dd, scale_dd = _make_viewer_dropdowns(levels, scales)
    output = widgets.Output()

    ## Redraw the site×metric heatmap when level or scale dropdown changes
    def update(*_):
        """Re-render the site × metric heatmap for the currently selected level and scale."""
        nr = _lookup_results(all_network_results, level_lookup,
                             level_dd.value, scale_dd.value)
        with output:
            clear_output(wait=True)
            if nr is None:
                print(f"  No results for {level_dd.value} at {scale_dd.value}.")
                return
            print(f"  Rendering {level_dd.value} , {scale_dd.value}: "
                  f"site × metric heatmap...")
            try:
                fig = build_heatmap_figure(
                    nr,
                    title=f"{level_dd.value} , {scale_dd.value}",
                )
                if fig is None:
                    clear_output(wait=True)
                    print(f"  No rank columns available for "
                          f"{level_dd.value} at {scale_dd.value}.")
                    return
                clear_output(wait=True)
                # Force a render before display so the first (synchronous) draw into the freshly
                # created Output widget isn't blank. The heavier cartopy heatmap otherwise shows a
                # white box until the dropdown is toggled; lighter viewers don't need this.
                fig.canvas.draw()
                display(fig)
                plt.close(fig)
            except Exception as e:
                print(f"  Could not display heatmap: {e}")

    level_dd.observe(update, names='value')
    scale_dd.observe(update, names='value')
    print("  Sites (columns) × metrics (rows), coloured by rank.")
    print("  Top 100 sites by overall composite are shown, "
          "displayed left-to-right in grid_id order.")
    print("  The reference map below shows where the "
          "displayed sites are located.\n")
    display(level_dd, scale_dd, output)
    update()  # render immediately (synchronous; a Timer thread fails to draw into the Output)

## Metric and network map viewer
def trait_metric_viewer(all_network_results):
    """
    Interactive hotspot/metric map viewer for the trait-defined pathway.

    Three dropdowns: analysis level, spatial scale and network metric.
    Shows a network map coloured by the selected metric
    With the top 200 strongest connections as arrows and the top 10 hotspots highlighted.
    """
    levels, scales, level_lookup = _get_viewer_options(all_network_results)
    if not levels or not scales:
        return
    level_dd, scale_dd = _make_viewer_dropdowns(levels, scales)

    metric_options = {
        'Network only (no metric)': None,
        'Overall composite': 'overall_composite',
        'Functional roles': 'functional_composite',
        'Centrality measures': 'centrality_composite',
        'Source strength': 'source_strength',
        'Sink strength': 'sink_strength',
        'Self-recruitment': 'self_recruitment',
        'Betweenness centrality': 'betweenness',
        'Closeness centrality': 'closeness',
        'Degree centrality': 'degree',
    }

    # Offer the custom composite as a selectable metric only if one was built for this run.
    ## Iterate over every network result dict in all scales and levels
    def _iter_nr(anr):
        """Yield every individual network result dict from the nested all_network_results structure."""
        for scale_results in anr.values():
            if 'community' in scale_results:
                yield scale_results['community']
            if 'per_species' in scale_results:
                yield from scale_results['per_species'].values()

    if any('custom_composite' in nr['site_metrics'].columns for nr in _iter_nr(all_network_results)):
        metric_options['Custom composite'] = 'custom_composite'
    metric_dd = widgets.Dropdown(
        options=list(metric_options.keys()),
        value='Overall composite',
        description='Metric:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='400px'),
    )
    output = widgets.Output()

    ## Redraw the network map when any dropdown changes
    def update(*_):
        """Re-render the network metric map for the currently selected level, scale, and metric."""
        nr = _lookup_results(all_network_results, level_lookup,
                             level_dd.value, scale_dd.value)
        metric_name = metric_dd.value
        col = metric_options[metric_name]

        with output:
            clear_output(wait=True)
            if nr is None:
                print(f"  No results for {level_dd.value} "
                      f"at {scale_dd.value}.")
                return

            site_metrics = nr['site_metrics']
            network_only = col is None

            if network_only:
                score_col = 'overall_composite'
                top10 = site_metrics.head(0)
            elif col.endswith('_composite'):
                score_col = col
                # overall/functional/centrality use 'top10_overall' etc, custom uses 'top10_custom_composite'
                top10 = nr.get('top10_' + col.replace('_composite', ''),
                               nr.get('top10_' + col, nr.get('top10_overall')))
            else:
                rank_col = f'{col}_rank'
                score_col = rank_col if rank_col in site_metrics.columns else col
                if col in site_metrics.columns:
                    top10 = site_metrics.nlargest(10, col).copy()
                else:
                    top10 = nr.get('top10_overall')

            print(f"  Rendering {level_dd.value} , {scale_dd.value}: "
                  f"{metric_name}... (~10-20 seconds)")
            try:
                top_n = None if network_only else 200
                fig = create_network_map(
                    site_metrics, top10,
                    title=f"{level_dd.value} , {scale_dd.value}",
                    composite_col=score_col,
                    cmap='viridis_r',
                    show_figure=False,
                    network_graph=nr.get('network_graph'),
                    top_n_edges=top_n,
                    network_only=network_only,
                )
                clear_output(wait=True)
                display(fig)
                plt.close(fig)
                if not network_only and top10 is not None and len(top10) > 0:
                    # Include a real site-name column when names were supplied
                    # (site_name equals site_<grid_id> when no names were given).
                    _has_names = ('site_name' in top10.columns and not
                                  (top10['site_name'].astype(str)
                                   == ('site_' + top10['grid_id'].astype(int).astype(str))).all())
                    table_rows = []
                    for idx, (_, site) in enumerate(top10.iterrows()):
                        _row = {'#': idx + 1}
                        if _has_names:
                            _row['Site'] = str(site['site_name'])
                        _row['Grid ID'] = int(site['grid_id'])
                        _row['Latitude'] = round(site['latitude'], 2)
                        _row['Longitude'] = round(site['longitude'], 2)
                        _row['Rank'] = round(site[score_col], 1)
                        table_rows.append(_row)
                    print()
                    print("Top 10 hotspots:")
                    top10_df = pd.DataFrame(table_rows)
                    display(top10_df.style.hide(axis='index').format({
                        'Latitude': '{:.2f}',
                        'Longitude': '{:.2f}',
                        'Rank': '{:.1f}',
                    }))
            except Exception as e:
                print(f"  Could not display metric map: {e}")

    level_dd.observe(update, names='value')
    scale_dd.observe(update, names='value')
    metric_dd.observe(update, names='value')

    info_box([
        "The map uses three visual indicators:",
        "  - Node colour: Indicates each site's rank for the selected metric",
        "      - A higher rank (e.g. 1) is indicated by a brighter colour (e.g. yellow)",
        "  - Black outline and numbered labels: The top 10 connectivity hotspots for the selected metric",
        "  - Blue arrows: The top 200 strongest connections in the network",
        "       - Site-to-site connections are drawn as straight arrows",
        "       - Self-recruitment is drawn as a small curved arrow at that site",
        "       - Arrow opacity is proportional to connection strength (i.e. weight, the number of particles exchanged)",
        "",
        "Selecting 'Network only (no metric)' displays only the network structure (arrows) with all sites drawn as neutral dots.",
        "",
        "The arrows show the underlying network and depend only on the level and scale selected, not the metric.",
        "",
        "Switching between metrics changes node colours and ranks, not the arrows.",
        "",
        "Switching to a different level or scale will change the underlying network.",
    ])
    print()
    display(level_dd, scale_dd, metric_dd, output)
    update()

## Interactive dropdown viewer for Type 1 connectivity metrics
def trait_metrics_viewer(type1_results):
    """
    Interactive dropdown viewer for Type 1 connectivity metrics in the trait pathway.

    Lets users switch between community-level and per-species metrics tables.

    Parameters
    ----------
    type1_results : dict
        Output from the Type 1 analysis block containing:
        - 'community' : dict with 'community' and optionally 'per_species' keys
        - 'per_species' : dict mapping species names to result dicts
    """
    # Build list of available views
    views = []
    view_data = {}

    if 'community' in type1_results and type1_results['community']:
        comm = type1_results['community']
        if isinstance(comm, dict) and 'community' in comm:
            views.append('Community')
            view_data['Community'] = comm['community']
        # Per-species inside community results
        if isinstance(comm, dict) and 'per_species' in comm:
            for sp, sp_result in comm['per_species'].items():
                display_name = display_species_name(sp)
                views.append(display_name)
                view_data[display_name] = sp_result

    if 'per_species' in type1_results:
        for sp, sp_result in type1_results['per_species'].items():
            display_name = display_species_name(sp)
            if display_name not in views:
                views.append(display_name)
                if isinstance(sp_result, dict) and 'community' in sp_result:
                    view_data[display_name] = sp_result['community']
                else:
                    view_data[display_name] = sp_result

    if not views:
        print("  No Type 1 metrics available to display.")
        return

    dropdown = widgets.Dropdown(
        options=views,
        value=views[0],
        description='View:',
        style={'description_width': 'initial'}
    )
    output = widgets.Output()

    ## Build a formatted display table from a metrics dict for the Type 1 viewer
    def _format_metrics(metrics_dict):
        """Build a clean display table from metrics dict."""
        if not isinstance(metrics_dict, dict):
            return pd.DataFrame([['No data available', '']], columns=['Metric', 'Value'])

        ## Format a single metric value according to its explicit kind (never guess from magnitude:
        ## a small distance like 0.08 km must not be shown as a percentage). Units live in the row
        ## label (matching the Type 3 table), so no unit suffix is added to the value here.
        def _fmt(v, kind):
            """Format a metric value. kind: 'pct' (fraction 0-1 -> %), 'val' (1 dp number),
            'int' (thousands-separated integer)."""
            if v is None:
                return '-'
            try:
                if kind == 'pct':
                    return f"{float(v) * 100:.1f}%"
                if kind == 'int':
                    return f"{int(round(float(v))):,}"
                if kind == 'val':
                    return f"{float(v):.1f}"
            except (TypeError, ValueError):
                return str(v)
            return str(v)

        rows = []
        # Unified label map covering both community and species keys. Each entry is
        # (display label with units, format kind). Units are in the label to match the Type 3 table.
        label_map = {
            'n_species': ('N species', 'int'),
            'total_released': ('Total particles released', 'int'),
            'n_released': ('Total particles released', 'int'),
            'total_settled': ('Total particles settled', 'int'),
            'n_settled': ('Total particles settled', 'int'),
            'settlement_rate': ('Settlement rate', 'pct'),
            'mortality_rate': ('Mortality rate', 'pct'),
            'self_recruitment_rate': ('Self-recruitment rate', 'pct'),
            'total_recruited': ('Particles recruited (community total)', 'int'),
            'n_recruited': ('Particles recruited (per species)', 'int'),
            'mean_distance_km': ('Mean dispersal distance (km)', 'val'),
            'min_distance_km': ('Min dispersal distance (km)', 'val'),
            'max_distance_km': ('Max dispersal distance (km)', 'val'),
            'mean_pld_days': ('Mean PLD (days)', 'val'),
            'min_pld_days': ('Min PLD (days)', 'val'),
            'max_pld_days': ('Max PLD (days)', 'val'),
            'n_release_sites': ('N release sites', 'int'),
            'n_settlement_sites': ('N settlement sites', 'int'),
        }

        # Keys to skip entirely
        skip_keys = {'std_distance_km', 'std_pld_days', '_comcon_cleaned',
                     'species_name_std'}

        for key, (label, kind) in label_map.items():
            if key in metrics_dict:
                rows.append((label, _fmt(metrics_dict[key], kind)))

        return pd.DataFrame(rows, columns=['Metric', 'Value']) if rows else pd.DataFrame()

    ## Redisplay the metrics table when the dropdown selection changes
    def update(*_):
        """Re-render the metrics table for the currently selected view."""
        selected = dropdown.value
        with output:
            clear_output(wait=True)
            metrics = view_data.get(selected, {})
            df = _format_metrics(metrics)
            if df.empty:
                print(f"  No metrics available for {selected}.")
            else:
                display(df)

    dropdown.observe(update, names='value')
    print("  Use the dropdown to switch between community and species views.\n")
    display(dropdown, output)
    # Trigger initial render after display
    with output:
        clear_output(wait=True)
        metrics = view_data.get(dropdown.value, {})
        df = _format_metrics(metrics)
        if not df.empty:
            display(df)

## Shared: build the Type 2 network-level summary rows (used by the interactive AND passive viewers)
def build_network_summary(network_results):
    """Return (statistic, value) rows for the dataset-level network summary shown in its own viewer
    ABOVE the matrix: number of species, study extent, potential/realised connectivity sites and
    unique connections (from network_results['summary_stats']). The Louvain/graph parameters (nodes,
    edges, density, communities, modularity) are a SEPARATE table shown beneath the Louvain map, not
    here. Lives in interactive and is imported by passive so every mode shows the same summary."""
    rows = []
    for k, v in (network_results.get('summary_stats') or {}).items():
        rows.append((k, f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)))
    return rows


## Interactive viewer for Type 2 network results: summary stats and per-site metrics table
def network_metrics_viewer(all_network_results):
    """Interactive viewer for Type 2 network results (summary + per-site metrics).

    Two dropdowns , analysis level (Community or species) and spatial scale ,
    matching the other network viewers.
    """
    levels, scales, level_lookup = _get_viewer_options(all_network_results)
    if not levels or not scales:
        print("  No network results to display.")
        return
    level_dd, scale_dd = _make_viewer_dropdowns(levels, scales)
    output = widgets.Output()

    ## Redisplay summary stats and per-site metrics when dropdowns change
    def update(*_):
        """Re-render the network summary and per-site metrics table for the selected level and scale."""
        nr = _lookup_results(all_network_results, level_lookup,
                             level_dd.value, scale_dd.value)
        with output:
            clear_output(wait=True)
            if not isinstance(nr, dict):
                print(f"  No results for {level_dd.value} at {scale_dd.value}.")
                return
            rows = build_network_summary(nr)   # shared helper: summary_stats + nodes/edges/density/communities/modularity
            if rows:
                print("Network summary")
                _df = pd.DataFrame(rows, columns=['Network statistic', 'Value'])
                _df.index = range(1, len(_df) + 1)
                display(_df)
            else:
                print(f"  No data available for {level_dd.value} at {scale_dd.value}.")

    level_dd.observe(update, names='value')
    scale_dd.observe(update, names='value')
    print("  Use the dropdowns to switch between community/species and spatial scale.\n")
    display(level_dd, scale_dd, output)
    update()

## Particle dispersal animation
def dispersal_animation(traj, output_path=None, fps=5, dpi=150, show_in_notebook=True):
    """
    Create an animated map of particle dispersal over time.
    Shows particles moving from release sites over the simulation period with a day counter.

    Parameters
    ----------
    traj : DataFrame
        Trajectory data with columns: trajectory, obs, lat, lon, time
    output_path : str, optional
        Path to save MP4 file (requires ffmpeg). If None, displays in notebook.
    fps : int
        Frames per second for the animation (default 5)
    dpi : int
        Resolution (default 150, use 300 for publication quality but slower)
    show_in_notebook : bool
        If True, display the animation inline in Jupyter

    Returns
    -------
    animation object or None
    """
    # Extract unique timesteps
    if 'time' in traj.columns:
        traj_sorted = traj.sort_values(['trajectory', 'obs'])
    else:
        traj_sorted = traj.sort_values(['trajectory', 'obs'])

    # Subsample particles (trajectories) only for very large simulations. The per-frame render
    # cost scales with the number of particles drawn, so the hundreds of thousands produced by
    # all-sites / GBIF runs make the animation take hours (or appear to hang). Runs at or below
    # the cap are shown in full; only larger ones are thinned to a random subset, which still
    # conveys the overall dispersal pattern at a fraction of the render time. Fixed seed for
    # reproducibility. This only affects the animation, never any analysis.
    max_particles = 25000
    _all_traj_ids = traj_sorted['trajectory'].unique()
    if len(_all_traj_ids) > max_particles:
        _rng = np.random.default_rng(0)
        _keep_ids = _rng.choice(_all_traj_ids, size=max_particles, replace=False)
        traj_sorted = traj_sorted[traj_sorted['trajectory'].isin(_keep_ids)]
        print(f"  Large particle set detected: Showing a random {max_particles:,} of {len(_all_traj_ids):,} particles to keep the animation fast.")

    # Get unique observation steps
    obs_values = sorted(traj_sorted['obs'].unique())
    n_frames = len(obs_values)

    # Subsample frames for long simulations to keep animation practical
    # Shows every Nth day to stay under ~60 frames
    max_frames = 60
    if n_frames > max_frames:
        step = max(1, n_frames // max_frames)
        obs_values = obs_values[::step]
        n_frames = len(obs_values)
        print(f"  Long simulation detected: Showing every {step} day(s) ({n_frames} frames).")

    if n_frames < 2:
        print("  Not enough timesteps for animation.")
        return None

    # Get coordinate bounds
    lon_min, lon_max = traj_sorted['lon'].min(), traj_sorted['lon'].max()
    lat_min, lat_max = traj_sorted['lat'].min(), traj_sorted['lat'].max()
    buf = 0.1

    # Figure setup with cartopy
    lon_range = (lon_max - lon_min) + 2 * buf
    lat_range = (lat_max - lat_min) + 2 * buf
    aspect = lon_range / lat_range if lat_range > 0 else 1.0
    fig_width = 9.5
    fig_height = max(4.5, min(8, fig_width / aspect))
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    ax.set_extent([lon_min - buf, lon_max + buf, lat_min - buf, lat_max + buf],
                  crs=ccrs.PlateCarree())
    ax.coastlines(resolution='10m', zorder=2)
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=4)

    # Initial scatter (first timestep). Particles drawn in the same dark blue as the hydro mask ocean fill for visual consistency. zorder=3 is below the land feature so particles disappear behind coastlines.
    first_obs = traj_sorted[traj_sorted['obs'] == obs_values[0]]
    sc = ax.scatter(
        first_obs['lon'].values, first_obs['lat'].values,
        s=2, c='#0077BE', zorder=3, transform=ccrs.PlateCarree(),
        alpha=0.75
    )

    # Day counter text. zorder=10 puts it above land (which is zorder=4) so the label is always readable even over coastlines.
    time_text = ax.text(
        0.03, 0.03, "",
        transform=ax.transAxes, fontsize=11,
        color='black', verticalalignment='bottom',
        horizontalalignment='left', zorder=10,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='#888888')
    )

    ax.set_title("Particle Dispersal", fontsize=12, fontweight='bold')
    gl = ax.gridlines(draw_labels=True, alpha=0.3, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # Leave room on the left and bottom for the latitude/longitude tick labels
    # (a tight left margin clipped the latitude labels in passive and the other modes).
    fig.subplots_adjust(top=0.95, bottom=0.10, left=0.12, right=0.97)

    ## Matplotlib FuncAnimation callback: update particle positions for each frame
    def animate(frame_idx):
        """Update the scatter plot positions and day counter for the given animation frame index."""
        obs_val = obs_values[frame_idx]
        frame_data = traj_sorted[traj_sorted['obs'] == obs_val]

        if len(frame_data) > 0:
            sc.set_offsets(np.c_[frame_data['lon'].values, frame_data['lat'].values])

        # Calculate day from observation index
        if 'time' in frame_data.columns and len(frame_data) > 0:
            try:
                first_time = pd.to_datetime(traj_sorted[traj_sorted['obs'] == obs_values[0]]['time'].iloc[0])
                current_time = pd.to_datetime(frame_data['time'].iloc[0])
                day = (current_time - first_time).days
            except Exception:
                day = frame_idx
        else:
            day = frame_idx

        time_text.set_text(f"Day {day}")
        return sc, time_text

    anim = FuncAnimation(fig, animate, frames=n_frames, interval=1000 // fps, blit=False)

    # Save or display
    if output_path:
        try:
            from matplotlib.animation import FFMpegWriter
            import logging
            writer = FFMpegWriter(fps=fps, bitrate=1800)
            # Silence the INFO chatter matplotlib/ffmpeg emit during the save (the
            # alarming-looking "INFO:MovieWriter..." / "INFO:Animation.save..." lines)
            # while still letting genuine warnings and errors through.
            _anim_log = logging.getLogger('matplotlib.animation')
            _mpl_log = logging.getLogger('matplotlib')
            _prev_anim, _prev_mpl = _anim_log.level, _mpl_log.level
            _anim_log.setLevel(logging.WARNING)
            _mpl_log.setLevel(logging.WARNING)
            try:
                anim.save(output_path, writer=writer, dpi=dpi)
            finally:
                _anim_log.setLevel(_prev_anim)
                _mpl_log.setLevel(_prev_mpl)
            print(f"  Animation saved to: {output_path}")

            # Also display in notebook if requested
            if show_in_notebook:
                try:
                    from IPython.display import Video
                    display(Video(output_path, embed=True,
                                  html_attributes="controls autoplay loop style='width:100%;'"))
                except Exception:
                    print("  Video saved but cannot display inline.")
        except Exception as e:
            print(f"  Could not save MP4: {e}")
            print("  If the package ffmpeg is missing, run: conda install -c conda-forge ffmpeg")
            print("  Displaying animation in notebook instead using HTML5...")
            try:
                from IPython.display import HTML as IPHTML
                display(IPHTML(anim.to_jshtml()))
            except Exception as e2:
                print(f"  Could not display animation: {e2}")
    else:
        if show_in_notebook:
            try:
                from IPython.display import HTML as IPHTML
                display(IPHTML(anim.to_jshtml()))
            except Exception as e:
                print(f"  Could not display animation: {e}")
                plt.show()

    plt.close(fig)
    return anim

## Interactive dropdown menu map viewer for all results for easy comparison
def create_interactive_metric_viewer(network_results, composite_name="Overall"):
    """
    Create an interactive dropdown widget for browsing network metric maps.

    Lets users switch between different metrics (source, sink, betweenness, etc.) without re-running the analysis.
    Each selection updates the hotspot map.

    Parameters
    ----------
    network_results : dict
        Output from network_analysis()
    composite_name : str
        Name of the default composite metric
    """
    site_metrics = network_results['site_metrics']
    top10_overall = network_results['top10_overall']

    # Available metrics for dropdown
    metric_options = {
        'Overall composite': ('overall_composite', top10_overall),
        'Functional roles': ('functional_composite', network_results['top10_functional']),
        'Centrality measures': ('centrality_composite', network_results['top10_centrality']),
        'Source strength': ('source_strength', None),
        'Sink strength': ('sink_strength', None),
        'Self-recruitment': ('self_recruitment', None),
        'Betweenness centrality': ('betweenness', None),
        'Closeness centrality': ('closeness', None),
        'Degree centrality': ('degree', None),
    }

    # Output area for the figure
    output = widgets.Output()

    # Dropdown widget
    dropdown = widgets.Dropdown(
        options=list(metric_options.keys()),
        value='Overall composite',
        description='Metric:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='350px')
    )

    ## Redraw the network map when the metric dropdown selection changes
    def update_map(change):
        """Re-render the network map for the newly selected metric."""
        metric_name = change['new']
        col, top10 = metric_options[metric_name]

        # For individual metrics, create top10 from that column
        if top10 is None:
            if col in site_metrics.columns:
                top10 = site_metrics.nlargest(10, col).copy()
            else:
                top10 = top10_overall

        # Always display as ranks (rank 1 = best). Composites are already
        # averages of ranks. Raw metrics have a `{metric}_rank` column.
        if col.endswith('_composite'):
            score_col = col
        else:
            rank_col = f'{col}_rank'
            score_col = rank_col if rank_col in site_metrics.columns else col
        cmap = 'viridis_r'  # low rank = best = yellow

        with output:
            clear_output(wait=True)
            create_network_map(
                site_metrics, top10,
                title=f"Connectivity: {metric_name}",
                composite_col=score_col,
                cmap=cmap,
                show_figure=True,
                network_graph=network_results.get('network_graph'),
                top_n_edges=200,
            )

    dropdown.observe(update_map, names='value')

    # Display widget
    print("  Use the dropdown to explore different metrics.")
    print("  The map updates automatically when you select a new metric.\n")
    display(dropdown)
    display(output)

    # Show initial map
    update_map({'new': 'Overall composite'})

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

### Heatmap table of metrics (summary)
def create_metrics_summary_table(network_results, output_path=None, show_styled=True):
    """
    Create a formatted summary table of all network metrics per site.

    Displays a site × metric table with colour-coded values for quick
    identification of important sites. Optionally exports to CSV.

    Parameters
    ----------
    network_results : dict
        Output from network_analysis()
    output_path : str, optional
        Path to save CSV file
    show_styled : bool
        If True, display styled table in Jupyter (default True)

    Returns
    -------
    DataFrame
        The summary table
    """
    site_metrics = network_results['site_metrics']

    # Select display columns
    # 'site' carries human-readable site names from spp_params['site_name_lookup'] when present, and 'grid_id' is always there as a fallback
    # The "only include columns that exist" filter below keeps the column list valid regardless of which are populated
    display_cols = ['site', 'grid_id', 'latitude', 'longitude',
                    'source_strength', 'sink_strength', 'self_recruitment',
                    'betweenness', 'closeness', 'degree',
                    'overall_composite']

    # Only include columns that exist
    display_cols = [c for c in display_cols if c in site_metrics.columns]

    # Add community if available
    if 'louvain_community' in site_metrics.columns:
        display_cols.append('louvain_community')

    summary = site_metrics[display_cols].copy()
    summary = summary.sort_values('overall_composite')

    # Round for display
    float_cols = summary.select_dtypes(include='float').columns
    summary[float_cols] = summary[float_cols].round(4)

    # Rename for readability
    rename_map = {
        'grid_id': 'Grid ID',
        'latitude': 'Lat',
        'longitude': 'Lon',
        'source_strength': 'Source',
        'sink_strength': 'Sink',
        'self_recruitment': 'Self-recruitment',
        'betweenness': 'Betweenness',
        'closeness': 'Closeness',
        'degree': 'Degree',
        'overall_composite': 'Overall Rank',
        'louvain_community': 'Community'
    }
    summary = summary.rename(columns={k: v for k, v in rename_map.items() if k in summary.columns})

    # Display
    n_sites = len(summary)
    subsection_header(f"Network Metrics Summary ({n_sites} sites)")

    if show_styled:
        try:
            # Styled display in Jupyter
            metric_cols = ['Source', 'Sink', 'Self-recruit.', 'Betweenness', 'Closeness', 'Degree']
            metric_cols = [c for c in metric_cols if c in summary.columns]

            styled = summary.style.background_gradient(
                subset=metric_cols, cmap='YlOrRd', axis=0
            ).format(precision=4, subset=metric_cols)

            if 'Overall Rank' in summary.columns:
                styled = styled.background_gradient(
                    subset=['Overall Rank'], cmap='YlOrRd_r', axis=0
                )

            display(styled)
        except Exception:
            # Fallback: plain print
            print(summary.to_string(index=False))
    else:
        print(summary.to_string(index=False))

    # Print top 5 summary
    print(f"\n  Top 5 sites by overall rank:")
    top5 = summary.head(5)
    for idx, (_, row) in enumerate(top5.iterrows(), 1):
        grid_id = int(row['Grid ID']) if 'Grid ID' in row.index else idx
        rank = row.get('Overall Rank', 'N/A')
        print(f"    {idx}. Grid {grid_id} (rank: {rank:.1f})")

    # Export
    if output_path:
        summary.to_csv(output_path, index=False)
        print(f"\n  Summary table saved to: {output_path}")

    return summary

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
        Available metrics: source_strength, sink_strength, self-recruitment, betweenness, closeness, degree.
    """
    site_metrics = network_results['site_metrics']

    # Define metric groups
    functional_metrics = ['source_strength', 'sink_strength', 'self_recruitment']
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

## Export the Type 2 network summary scalars (one row per metric)
def export_network_summary(network_results, output_dir, label):
    """
    Write the network-level summary scalars (network_results['summary_stats']) to
    <label>_network_summary.csv as a tidy Metric | Value table.

    These scalars (number of species, study extent, potential and realised connectivity
    sites, unique connections) are otherwise only shown in the interactive viewer.
    Returns the written path, or None if there are no summary stats.
    """
    stats = dict(network_results.get('summary_stats') or {})
    # Also include the network-level graph statistics (previously display-only), so the exported
    # summary matches the viewer table. Uses precomputed keys only (no re-running of Louvain).
    for _label, _key in [('Nodes', 'network_nodes'), ('Edges', 'network_edges'),
                         ('Density', 'network_density'), ('Communities', 'network_communities'),
                         ('Modularity', 'network_modularity')]:
        _v = network_results.get(_key)
        if _v is not None:
            stats[_label] = _v
    if not stats:
        return None
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{label}_network_summary.csv")
    pd.DataFrame(list(stats.items()), columns=['Metric', 'Value']).to_csv(path, index=False)
    return path

## Export the network results
def export_network_results(network_results, output_dir=None, prefix=''):
    """
    Export all network analysis results to CSV files.

    Saves site-level network metrics and the full connectivity matrix.

    Parameters
    ----------
    network_results : dict
        Output from network_analysis(). Expected keys:
        - 'site_metrics': DataFrame with all site-level metrics
        - 'connectivity_matrix': DataFrame with pairwise connectivity values
    output_dir : str, optional
        Directory to save output files. If None, uses the project results
        directory or falls back to './outputs'.
    prefix : str, optional
        Identifier embedded in each output filename, e.g. 'community_30_km'.
        If empty, filenames are 'network_site_metrics.csv' and 'connectivity_matrix.csv'.

    Returns
    -------
    str
        Path to the output directory where files were saved
    """
    if output_dir is None:
        output_dir = get_default_dir('results')

    os.makedirs(output_dir, exist_ok=True)

    # Reorder columns: identifiers first, then metrics
    sm = network_results['site_metrics'].copy()
    id_cols = [c for c in ['grid_id', 'site', 'latitude', 'longitude', 'louvain_community'] if c in sm.columns]
    other_cols = [c for c in sm.columns if c not in id_cols]
    sm = sm[id_cols + other_cols]

    # Remove _norm columns (internal visualisation artifacts, not useful for users)
    norm_cols = [c for c in sm.columns if c.endswith('_norm')]
    sm = sm.drop(columns=norm_cols, errors='ignore')

    # Drop the redundant 'site' column (it is just 'site_' + grid_id).
    sm = sm.drop(columns=['site'], errors='ignore')
    # Keep 'site_name' only when real names were supplied
    # Otherwise it equals 'site_<grid_id>' for every row
    if 'site_name' in sm.columns and 'grid_id' in sm.columns:
        _auto_name = ('site_' + sm['grid_id'].astype(int).astype(str)) == sm['site_name'].astype(str)
        if _auto_name.all():
            sm = sm.drop(columns=['site_name'])
    # Drop standard-deviation columns (removed from the Jupyter tables too).
    sm = sm.drop(columns=[c for c in sm.columns if c.endswith('_std') or c.startswith('std_')], errors='ignore')

    # Flag the composite columns as rank-based in the exported CSV header only.
    # The in-memory dict keys are left unchanged because compare_models, the
    # viewers and the figures reference them by their original names.
    # top10_count is a count, not a rank, so it is deliberately left alone.
    _composite_rank_rename = {
        'functional_composite': 'functional_composite_rank',
        'centrality_composite': 'centrality_composite_rank',
        'overall_composite': 'overall_composite_rank',
        'custom_composite': 'custom_composite_rank',
    }
    sm = sm.rename(columns={k: v for k, v in _composite_rank_rename.items() if k in sm.columns})

    # Export site metrics (all metrics for all sites)
    _pfx = f'{prefix}_' if prefix else ''
    metrics_file = os.path.join(output_dir, f'{_pfx}network_site_metrics.csv')
    sm.to_csv(metrics_file, index=False)
    print(f"Site metrics saved: {metrics_file}")

    # Export connectivity matrix
    conn_file = os.path.join(output_dir, f'{_pfx}connectivity_matrix.csv')
    network_results['connectivity_matrix'].to_csv(conn_file)
    print(f"\nConnectivity matrix saved: {conn_file}")

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

    # Connectivity matrices are DIRECTED (source -> sink) and therefore asymmetric, so compare ALL
    # off-diagonal elements (both i->j and j->i), not just the upper triangle. Using one triangle
    # would drop half the directed connections and make the result depend on arbitrary site ordering.
    n = matrix1.shape[0]
    offdiag = ~np.eye(n, dtype=bool)

    x = matrix1[offdiag]
    y = matrix2[offdiag]

    # Flatten full matrices for variance check
    flat1 = matrix1.flatten()
    flat2 = matrix2.flatten()

    # Check for zero variance
    if np.std(flat1) == 0 or np.std(flat2) == 0:
        print("\nWarning: One or both connectivity matrices have no variance.")
        print("This typically occurs when:")
        print("  - Very few connections exist in one or both models")
        print("  - All connection strengths are identical")
        print("  - The study extent is too small or simulation too short")
        print("The Mantel test cannot compute a meaningful correlation in this case.")
        print("Consider: Running with more particles, longer simulation, or larger area.\n")
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
        y_perm = matrix2_perm[offdiag]

        r_perm, _ = spearmanr(x, y_perm)
        if abs(r_perm) >= abs(r_obs):
            count += 1

    p_value = (count + 1) / (n_permutations + 1)

    return r_obs, p_value

## Compare network results between baseline and modified trait scenarios
def compare_models(baseline_results, modified_results):
    """
    Compare network results between two trait scenarios (e.g. baseline vs PLD shift).

    Parameters
    ----------
    baseline_results : dict
        Network analysis results from the baseline trait scenario
    modified_results : dict
        Network analysis results from the modified trait scenario

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
    baseline_top10_ids = set(baseline_results['top10_overall']['grid_id'])
    modified_top10_ids = set(modified_results['top10_overall']['grid_id'])

    overlap = baseline_top10_ids.intersection(modified_top10_ids)
    baseline_unique = baseline_top10_ids - modified_top10_ids
    modified_unique = modified_top10_ids - baseline_top10_ids

    # Denominator = the number of hotspots actually identified (the "top 10" holds
    # fewer than 10 when there are fewer than 10 sites), so the percentage is out
    # of the real hotspot count, capped at 10, not always out of 10.
    n_possible = max(len(baseline_top10_ids), len(modified_top10_ids), 1)
    n_possible = min(n_possible, 10)

    comparison['hotspot_overlap'] = {
        'shared_sites': len(overlap),
        'n_possible': n_possible,
        'overlap_percentage': round(len(overlap) / n_possible * 100),
        'baseline_unique': len(baseline_unique),
        'modified_unique': len(modified_unique),
        'shared_ids': list(overlap),
        'baseline_unique_ids': list(baseline_unique),
        'modified_unique_ids': list(modified_unique)
    }

    # 2. Spatial Agreement
    if len(overlap) >= 7:
        comparison['spatial_agreement'] = 'high'
    elif len(overlap) >= 4:
        comparison['spatial_agreement'] = 'moderate'
    else:
        comparison['spatial_agreement'] = 'low'

    # 3. Matrix comparison using Mantel test
    baseline_matrix = baseline_results['connectivity_matrix']
    modified_matrix = modified_results['connectivity_matrix']

    # Find common sites
    common_sites = set(baseline_matrix.index).intersection(set(modified_matrix.index))

    if len(common_sites) > 2:
        common_sites = sorted(list(common_sites))
        baseline_aligned = baseline_matrix.loc[common_sites, common_sites].values
        modified_aligned = modified_matrix.loc[common_sites, common_sites].values

        # Running mantel test
        try:
            result = mantel_test(baseline_aligned, modified_aligned, n_permutations=999)
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

## Display side-by-side metrics table comparing original and modified trait model outputs
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
    orig_settlement = orig_comm.get('settlement_rate', 0)
    orig_mortality = orig_comm.get('mortality_rate', 0)
    orig_recruitment = orig_comm.get('self_recruitment_rate', 0)
    orig_mean_dist = orig_comm.get('mean_distance_km', 0)
    orig_min_dist = orig_comm.get('min_distance_km', 0)
    orig_max_dist = orig_comm.get('max_distance_km', 0)
    orig_mean_pld = orig_comm.get('mean_pld_days', 0)
    orig_min_pld = orig_comm.get('min_pld_days', 0)
    orig_max_pld = orig_comm.get('max_pld_days', 0)

    # Extract all modified model values
    mod_released = mod_comm.get('total_released', 0)
    mod_settled = mod_comm.get('total_settled', 0)
    mod_settlement = mod_comm.get('settlement_rate', 0)
    mod_mortality = mod_comm.get('mortality_rate', 0)
    mod_recruitment = mod_comm.get('self_recruitment_rate', 0)
    mod_mean_dist = mod_comm.get('mean_distance_km', 0)
    mod_min_dist = mod_comm.get('min_distance_km', 0)
    mod_max_dist = mod_comm.get('max_distance_km', 0)
    mod_mean_pld = mod_comm.get('mean_pld_days', 0)
    mod_min_pld = mod_comm.get('min_pld_days', 0)
    mod_max_pld = mod_comm.get('max_pld_days', 0)

    # Mortality is None whenever it could not be estimated (no raw release counts)
    # Track that so the row can say "unavailable" and coerce the remaining rate values so the formatting below can never hit None
    mortality_available = (orig_mortality is not None) and (mod_mortality is not None)
    orig_settlement = orig_settlement or 0
    mod_settlement = mod_settlement or 0
    orig_mortality = orig_mortality or 0
    mod_mortality = mod_mortality or 0
    orig_recruitment = orig_recruitment or 0
    mod_recruitment = mod_recruitment or 0

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
    print(
        f"{'Settlement rate':<35} {orig_settlement * 100:>14.1f}% {mod_settlement * 100:>14.1f}% {settle_diff:>+11.1f}%")
    if mortality_available:
        mort_diff = (mod_mortality - orig_mortality) * 100
        print(
            f"{'Mortality rate':<35} {orig_mortality * 100:>14.1f}% {mod_mortality * 100:>14.1f}% {mort_diff:>+11.1f}%")
    else:
        print(f"{'Mortality rate':<35} {'unavailable':>15} {'unavailable':>15} {'-':>12}")

    # Self-recruitment
    recruitment_diff = (mod_recruitment - orig_recruitment) * 100
    print(f"{'Self-recruitment rate':<35} {orig_recruitment * 100:>14.1f}% {mod_recruitment * 100:>14.1f}% {recruitment_diff:>+11.1f}%")

    # Dispersal distances
    print(f"{'Min dispersal distance (km)':<35} {orig_min_dist:>15.1f} {mod_min_dist:>15.1f} {mod_min_dist - orig_min_dist:>+12.1f}")
    print(f"{'Mean dispersal distance (km)':<35} {orig_mean_dist:>15.1f} {mod_mean_dist:>15.1f} {mod_mean_dist - orig_mean_dist:>+12.1f}")
    print(f"{'Max dispersal distance (km)':<35} {orig_max_dist:>15.1f} {mod_max_dist:>15.1f} {mod_max_dist - orig_max_dist:>+12.1f}")

    # PLD
    print(f"{'Min PLD (days)':<35} {orig_min_pld:>15.0f} {mod_min_pld:>15.0f} {mod_min_pld - orig_min_pld:>+12.0f}")
    print(f"{'Mean PLD (days)':<35} {orig_mean_pld:>15.1f} {mod_mean_pld:>15.1f} {mod_mean_pld - orig_mean_pld:>+12.1f}")
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
            print(f"  Shared hotspots: {overlap['shared_sites']}/{overlap.get('n_possible', 10)} ({overlap['overlap_percentage']}%)")
            print(f"  Unique to original: {overlap['baseline_unique']}")
            print(f"  Unique to modified: {overlap['modified_unique']}")

        if 'mantel_test' in comparison and comparison['mantel_test'].get('r') is not None:
            mantel = comparison['mantel_test']
            print(f"\nConnectivity Matrix Comparison (Mantel test):")
            print(f"  Correlation: r = {mantel['r']:.3f}")
            print(f"  P-value: {mantel['p_value']:.4f}")
            print(f"  Sites compared: {mantel['n_sites']}")

            if mantel['p_value'] < 0.05:
                if mantel['r'] > 0.7:
                    print("  Interpretation: Strong spatial correlation. Connectivity patterns are similar")
                elif mantel['r'] > 0.4:
                    print("  Interpretation: Moderate spatial correlation. Some pattern preservation")
                else:
                    print("  Interpretation: Weak spatial correlation. Trait changes substantially alter patterns")
            else:
                print("  Interpretation: Not statistically significant")

        if 'spatial_agreement' in comparison:
            agreement_text = {
                'high': 'High agreement (7+ shared hotspots). Trait changes have minimal impact on hotspot identification',
                'moderate': 'Moderate agreement (4-6 shared). Trait changes alter some hotspot locations',
                'low': 'Low agreement (<4 shared). Trait changes substantially change which sites are hotspots'
            }
            print(f"\nSpatial Agreement: {agreement_text.get(comparison['spatial_agreement'], comparison['spatial_agreement'])}")

    return {
        'settlement_diff': settle_diff,
        'mortality_diff': (mod_mortality - orig_mortality) * 100 if mortality_available else None,
        'connectivity_effect': effect
    }

## Create comparison maps
def create_comparison_map(baseline_results, modified_results, output_path=None,
                          output_format='png', dpi=300,
                          title1='Original Traits', title2='Modified Traits'):
    """
    Render the two trait scenarios as two separate hotspot maps
    Uses the same style as the main network hotspot map (create_network_map):
    every site is coloured by overall composite rank and the top 10 hotspots are numbered.

    Returns
    -------
    list of matplotlib.figure.Figure
        [baseline_figure, modified_figure]
    """
    figs = []
    for _results, _title, _suffix in [
        (baseline_results, title1, 'original'),
        (modified_results, title2, 'modified'),
    ]:
        _sm = _results['site_metrics']
        _t10 = _results.get('top10_overall', _sm.head(10))
        _ng = _results.get('network_graph')
        _out = f"{output_path}_{_suffix}.{output_format}" if output_path else None
        # Build (and save) each map without letting create_network_map show it; both are displayed
        # explicitly below so the original always appears above the modified, regardless of how much
        # other output the surrounding cell produces (two separate plt.show() calls can otherwise be
        # collapsed under Jupyter's "Show more outputs" fold).
        fig = create_network_map(
            _sm, _t10, title=_title,
            composite_col='overall_composite', cmap='viridis_r',
            output_path=_out, output_format=output_format, dpi=dpi,
            show_figure=False, network_graph=_ng, top_n_edges=200,
            subtitle='Sites coloured by overall composite hotspot rank (1 = best)',
        )
        figs.append(fig)

    # Display the two maps in order (original first, then modified) as separate outputs.
    try:
        from IPython.display import display as _ipy_display
        for _fig in figs:
            _ipy_display(_fig)
    except Exception:
        pass
    return figs

    # endregion

    # endregion

# region Post-analysis wrapper function
## Writes the two-sheet xlsx outputs reference + run configuration metadata file
def write_project_metadata(project_dirs, config_dict, pathway, output_path):
    """
    Write a two-sheet Excel file documenting the project's outputs and run configuration.

    Replaces the older `passive_metadata.json` and `trait_metadata.json` files.
    Designed to be opened alongside the project folder to help users interpret their results.

    Sheet 1 ("Outputs Reference"):
        Catalogue of every file or subfolder the pathway can produce, with columns
        Folder | File / Subfolder | Description | Saved. Each expected output is marked
        YES if found on disk and NO if not. Any user-created files that don't match the
        catalogue are appended at the bottom as User-added.
    Sheet 2 ("Run Configuration"):
        Flat Parameter | Value table containing the full run-configuration record that
        used to be in the JSON metadata file (pathway, domain, simulation parameters, etc.).

    Parameters
    ----------
    project_dirs : dict
        The dict returned by `setup_project_directory()` mapping folder roles
        ('config', 'data', 'simulation', 'figures', 'results', etc.) to their
        absolute paths. Walking these paths is how the function works out which
        expected outputs were actually saved.
    config_dict : dict
        Nested run-configuration dictionary produced by `passive_metadata()` or
        `trait_metadata()`. Written verbatim (flattened) into Sheet 2.
    pathway : str
        Either 'passive' or 'trait'. Selects which expected-output catalogue to use.
    output_path : str
        Full path to the .xlsx file to write (typically in `project_dirs['config']`).

    Returns
    -------
    str
        The path the file was written to.
    """
    # =====================================================================
    # Master catalogue of expected outputs per pathway.
    # Each entry: (folder_role, file_or_subfolder_pattern, description)
    # Patterns:
    #   - plain filenames match exactly
    #   - '*'  wildcards inside a name match any characters
    #   - trailing '/'  means a subfolder rather than a file
    #   - '<dataset>/'  inside a pattern means "any first-level subfolder",
    #                   used for the passive pathway's per-dataset figure/results layout
    # =====================================================================
    if pathway == 'passive':
        reference = [
            # config
            ('config', 'comcon_outputs.xlsx',
             'This file, outputs reference catalogue (Sheet 1) and run configuration metadata (Sheet 2).'),

            # simulation
            ('simulation', 'passive_summer.zarr',
             'Raw OceanParcels simulation output for the summer season (July).'),
            ('simulation', 'passive_winter.zarr',
             'Raw OceanParcels simulation output for the winter season (February, following James et al. 2025).'),
            ('simulation', 'passive_summer_processed.parquet',
             'Post-processed summer trajectory DataFrame (release grid_id, time, lat, lon).'),
            ('simulation', 'passive_winter_processed.parquet',
             'Post-processed winter trajectory DataFrame (release grid_id, time, lat, lon).'),

            # figures (workflow-level)
            ('grids', 'landmask_native.*',
             'Native-resolution land/ocean mask figure showing the full hydrodynamic grid extent.'),
            ('grids', 'landmask_binned.*',
             'Binned-resolution land/ocean mask figure used as the basis for the network analyses.'),

            # figures (per-dataset, one subfolder per dataset)
            ('figures', '<dataset>/connectivity_matrix.*',
             'Site × site connectivity matrix figure for this dataset.'),
            ('figures', '<dataset>/communities.*',
             'Louvain community detection map, sites coloured by community membership.'),
            ('figures', '<dataset>/heatmap.*',
             'Site × metric heatmap: rows = network metrics, columns = top 100 sites by overall rank.'),
            ('figures', '<dataset>/network_only.*',
             'Directed connectivity network map without hotspots (edges only, no top-site highlighting).'),
            ('figures', '<dataset>/hotspot_overall.*',
             'Top 10 hotspots by overall composite score (centrality + functional combined).'),
            ('figures', '<dataset>/hotspot_source_strength.*',
             'Top 10 hotspots ranked by source strength (sites contributing the most particles).'),
            ('figures', '<dataset>/hotspot_sink_strength.*',
             'Top 10 hotspots ranked by sink strength (sites receiving the most particles).'),
            ('figures', '<dataset>/hotspot_self_recruitment.*',
             'Top 10 hotspots ranked by self-recruitment (sites retaining their own larvae).'),
            ('figures', '<dataset>/hotspot_betweenness_centrality.*',
             'Top 10 hotspots ranked by betweenness centrality (stepping-stone sites).'),
            ('figures', '<dataset>/hotspot_closeness_centrality.*',
             'Top 10 hotspots ranked by closeness centrality (sites reachable from many others).'),
            ('figures', '<dataset>/hotspot_degree_centrality.*',
             'Top 10 hotspots ranked by degree centrality (sites with the most connections).'),

            # animation (per-season)
            ('animation', '<season>/dispersal_animation.mp4',
             'Particle dispersal animation (MP4) for the season, built from the raw combined trajectory data for the run.'),

            # results (per-dataset)
            ('results', '<dataset>/<dataset>_connectivity_summary.csv',
             'Type 1 connectivity summary metrics for this dataset (dispersal distances and self-recruitment rate).'),
            ('results', '<dataset>/<dataset>_network_site_metrics.csv',
             'Type 2 per-site network metrics for this dataset (degree, betweenness, closeness, source/sink strength, composites).'),
            ('results', '<dataset>/<dataset>_connectivity_matrix.csv',
             'Type 2 site × site connectivity matrix for this dataset (raw particle exchange values).'),
            ('results', '<dataset>/<dataset>_network_summary.csv',
             'Type 2 network summary scalars for this dataset (number of species, study extent, potential & realised connectivity sites, unique connections).'),

            # config (workflow-level)
            ('config', 'analysis_timing.txt',
             'Timing summary of total seconds per analysis step across all datasets, including the grand total and the slowest dataset per step.'),
        ]
    else:  # 'trait'
        reference = [
            # config
            ('config', 'comcon_outputs.xlsx',
             'This file, outputs reference catalogue (Sheet 1) and run configuration metadata (Sheet 2).'),
            ('config', 'spp_occ_params.json',
             'Saved species/occurrence parameters (data source, release/settlement strategy, habitat filtering, site-name lookup).'),
            ('config', 'hydrodynamic_parameters.json',
             'Saved hydrodynamic parameters (dataset ID, date range, spatial extent, resolution, traits).'),
            ('config', 'pset_config.json',
             'Saved particle-set parameters (release interval, particles per cell, horizontal diffusivity).'),
            ('config', 'comcon_config.json',
             'Saved full automatic-mode configuration (all auto-run settings).'),
            ('config', 'comcon_session*.pkl',
             'Saved interactive session state, used to reload/restore a run.'),
            ('config', 'comcon_session_final_processed_*.parquet',
             'Saved post-processed trajectory data per species, used to restore a session.'),

            # hydro_data
            ('hydro_data', '*.nc',
             'Downloaded NetCDF hydrodynamic data file from Copernicus Marine Service.'),
            ('hydro_data', '*.zarr',
             'Downloaded Zarr hydrodynamic data store from Copernicus Marine Service.'),
            ('hydro_data', 'species_traits*.csv',
             'Saved species trait table (PLD, spawning window, habitat preferences).'),

            # occurrence_data
            ('occurrence_data', '*occurrences*.csv',
             'Cleaned species occurrence records used to build the release and settlement grids.'),

            # grids
            ('grids', 'grid_study_area.csv',
             'Coastal grid within the user-defined study extent polygon (used as release cells).'),
            ('grids', 'grid_full_domain.csv',
             'Full coastal grid across the entire hydrodynamic domain (used as settlement cells).'),
            ('grids', '*_release_grid.csv',
             'Per-species release grid: study-area coastal cells filtered by species occurrence.'),
            ('grids', '*_settlement_grid.csv',
             'Per-species settlement grid: full-domain coastal cells filtered by species occurrence.'),

            # simulation
            ('simulation', '*.zarr',
             'Raw OceanParcels simulation output (one .zarr per simulation run).'),
            ('simulation', 'chunks_*/',
             'Per-chunk parquet files with assigned release IDs (memory-efficient post-processing intermediate).'),

            # figures
            ('grids', '*landmask*',
             'Land/ocean mask visualisation of the hydrodynamic data extent and coastal mask.'),
            ('figures', '*distribution*',
             'Species occurrence distribution map showing cleaned records on the hydro extent.'),
            ('figures', 'dispersal_domain.*',
             'Map of the simulation domain showing the release/settlement extent.'),
            ('figures', '<dataset>/*matrix*',
             'Connectivity matrix figures (site x site particle exchange), per species and/or community.'),
            ('figures', '<dataset>/*network*',
             'Directed connectivity network maps with arrows showing pairwise particle exchange.'),
            ('figures', '<dataset>/*communities*',
             'Louvain community detection maps, sites coloured by community membership.'),
            ('figures', '<dataset>/*heatmap*',
             'Site x metric heatmaps, rows are network metrics, columns are top sites by overall rank.'),
            ('figures', '<dataset>/*hotspot*',
             'Top 10 hotspot maps coloured by composite or single-metric rank.'),
            ('figures', 'trait_sensitivity_*_original.*',
             'Type 3 hotspot map under the original (baseline) traits.'),
            ('figures', 'trait_sensitivity_*_modified.*',
             'Type 3 hotspot map under the modified-trait scenario.'),

            # animation
            ('animation', 'dispersal_animation.mp4',
             'Particle dispersal animation (MP4), built from the raw combined trajectory data for the run.'),

            # results (per-label subfolders)
            ('results', 'community/community_connectivity_summary.csv',
             'Type 1 community-level connectivity metrics (mortality, self-recruitment, mean dispersal, mean PLD).'),
            ('results', '<species>/<species>_connectivity_summary.csv',
             'Type 1 per-species connectivity metrics (mortality, self-recruitment, mean dispersal, mean PLD).'),
            ('results', '<species>/<species>_settlement_attribution.csv',
             'Buffer-cell to real settlement site mapping for each species (columns: species, buffer_cell_grid_id, attributed_settlement_site).'),
            ('results', '<species>/<species>_<scale>_network_site_metrics.csv',
             'Type 2 per-site network metrics (degree, betweenness, closeness, source/sink strength, composites).'),
            ('results', '<species>/<species>_<scale>_connectivity_matrix.csv',
             'Type 2 site × site connectivity matrix (raw particle exchange values).'),
            ('results', '<species>/<species>_network_summary.csv',
             'Type 2 network summary scalars (number of species, study extent, potential & realised connectivity sites, unique connections).'),
            ('results', 'community/community_trait_sensitivity_*.csv',
             'Type 3 trait-sensitivity comparison table (original vs modified community metrics, Mantel test, hotspot overlap, spatial agreement).'),
        ]

    # =====================================================================
    # Sheet 1: walk the project folder, match against the reference catalogue
    # =====================================================================
    rows = []
    matched_paths = set()  # absolute paths already accounted for in the catalogue
    # The metadata file itself doesn't exist on disk yet at walk time , special-case it.
    _self_filename = os.path.basename(output_path)

    for folder_role, pattern, description in reference:
        folder_path = project_dirs.get(folder_role)

        # Special case: this file itself. It's being written right now, so mark YES.
        if pattern == _self_filename and folder_role == 'config':
            rows.append({
                'Folder': 'config/',
                'File / Subfolder': _self_filename,
                'Description': description,
                'Saved': 'YES',
            })
            if folder_path:
                matched_paths.add(os.path.abspath(os.path.join(folder_path, _self_filename)))
            continue

        # If the user's project doesn't have this folder at all
        if not folder_path or not os.path.isdir(folder_path):
            rows.append({
                'Folder': f"{folder_role}/",
                'File / Subfolder': pattern,
                'Description': description,
                'Saved': 'NO (folder not present)',
            })
            continue

        found = []  # list of (display_subfolder_or_None, basename, abs_path)

        # Pattern ending with '/' -> match a subfolder
        if pattern.endswith('/'):
            dir_pattern = pattern.rstrip('/')
            for entry in sorted(os.listdir(folder_path)):
                full_path = os.path.join(folder_path, entry)
                if os.path.isdir(full_path) and fnmatch.fnmatch(entry, dir_pattern):
                    found.append((None, entry + '/', full_path))
        # Pattern with a subfolder part (placeholder <dataset>/<species>/<season>/ or a literal
        # community/) -> match files inside one-level subfolders. A placeholder subfolder matches
        # any subfolder and is bound into the file pattern; any remaining placeholders (<scale>,
        # ...) expand to '*'. This recognises the real <species>_<scale> / <species>/<scale>
        # naming as standard output rather than letting it fall through to "user-added".
        elif '/' in pattern:
            sub_pat, file_pat = pattern.split('/', 1)
            sub_is_ph = sub_pat.startswith('<') and sub_pat.endswith('>')
            for entry in sorted(os.listdir(folder_path)):
                sub_path = os.path.join(folder_path, entry)
                if not os.path.isdir(sub_path):
                    continue
                if not sub_is_ph and not fnmatch.fnmatch(entry, sub_pat):
                    continue
                fp = file_pat.replace(sub_pat, entry) if sub_is_ph else file_pat
                fp = re.sub(r'<[^>]+>', '*', fp)
                for f in sorted(os.listdir(sub_path)):
                    abs_f = os.path.join(sub_path, f)
                    if os.path.abspath(abs_f) in matched_paths:
                        continue
                    if fnmatch.fnmatch(f, fp):
                        found.append((entry, f, abs_f))
        # Plain filename pattern -> match files at top level
        else:
            for f in sorted(os.listdir(folder_path)):
                full_path = os.path.join(folder_path, f)
                if fnmatch.fnmatch(f, pattern) and os.path.isfile(full_path):
                    found.append((None, f, full_path))
                elif fnmatch.fnmatch(f, pattern) and os.path.isdir(full_path) and pattern.endswith('.zarr'):
                    # .zarr is technically a directory , treat as a single match
                    found.append((None, f, full_path))

        if found:
            for sub, fname, abs_path in found:
                display_folder = f"{folder_role}/{sub}/" if sub else f"{folder_role}/"
                rows.append({
                    'Folder': display_folder,
                    'File / Subfolder': fname,
                    'Description': description,
                    'Saved': 'YES',
                })
                matched_paths.add(os.path.abspath(abs_path))
        else:
            rows.append({
                'Folder': f"{folder_role}/",
                'File / Subfolder': pattern,
                'Description': description,
                'Saved': 'NO',
            })

    # =====================================================================
    # Walk for any user-added files that didn't match the catalogue
    # =====================================================================
    for role, folder_path in project_dirs.items():
        if role == 'root' or not folder_path or not os.path.isdir(folder_path):
            continue
        for walk_root, dirs, files in os.walk(folder_path):
            # Don't recurse into .zarr stores or chunk intermediates
            dirs[:] = [d for d in dirs if not d.endswith('.zarr')
                       and not d.startswith('chunks_') and d != '_temp_chunks']
            for f in files:
                # Skip transient Office lock / temp files (e.g. ~$comcon_outputs.xlsx)
                if f.startswith('~$') or f.startswith('.~'):
                    continue
                full_path = os.path.abspath(os.path.join(walk_root, f))
                if full_path in matched_paths:
                    continue
                rel = os.path.relpath(full_path, folder_path).replace('\\', '/')
                sub_dir = os.path.dirname(rel)
                display_folder = f"{role}/" + (sub_dir + '/' if sub_dir else '')
                rows.append({
                    'Folder': display_folder,
                    'File / Subfolder': os.path.basename(rel),
                    'Description': 'User-added file, not part of the standard ComCon output catalogue.',
                    'Saved': 'YES',
                })

    outputs_df = pd.DataFrame(rows, columns=['Folder', 'File / Subfolder', 'Description', 'Saved'])

    # =====================================================================
    # Sheet 2: flatten the run configuration dict into Parameter | Value rows
    # =====================================================================
    ## Recursively flatten a nested config dict into (key, value) pairs for the export sheet
    def _flatten(d, prefix=''):
        """Recursively convert a nested dict to a flat list of (dotted.key, value) tuples."""
        out = []
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.extend(_flatten(v, key))
            elif isinstance(v, (list, tuple)):
                # List of dicts: expand each item with a 1-based [i] index (display label only)
                if v and all(isinstance(x, dict) for x in v):
                    for i, x in enumerate(v):
                        out.extend(_flatten(x, f"{key}[{i + 1}]"))
                else:
                    out.append((key, ', '.join(str(x) for x in v)))
            else:
                out.append((key, v))
        return out

    config_rows = _flatten(config_dict) if isinstance(config_dict, dict) else []
    config_rows.append(('Documentation',
                        'See the ComCon User Guide (Methods & Assumptions) and the methods paper for full methodology.'))
    config_df = pd.DataFrame(config_rows, columns=['Parameter', 'Value'])

    # =====================================================================
    # Write the workbook
    # =====================================================================
    # Sheet 3: metric definitions , how every reported metric is calculated, so
    # results can be interpreted and reproduced without reading the source code.
    metric_definitions = [
        ('Connectivity (matrix value)',
         'Number of particles released from a source site that settled at a sink site.',
         'count of particles (source -> sink)',
         'particles'),
        ('Connectivity strength (displayed)',
         'The value shown on connectivity-matrix figures. Log-scaled for display only. '
         'The exported connectivity_matrix.csv holds the raw counts.',
         'log10(particles + 1)',
         'log10 particles'),
        ('Settlement rate',
         'Proportion of released particles that successfully settled at any site.',
         'n_settled / n_released',
         'proportion (0-1)'),
        ('Mortality rate',
         'Proportion of released particles that did not settle (reached maximum PLD without '
         'settling). Requires release counts. Left blank if these are unavailable.',
         '1 - (n_settled / n_released)',
         'proportion (0-1)'),
        ('Self-recruitment',
         'Particles that settled at the same site they were released from.',
         'self-connection weight (source site == sink site)',
         'particles'),
        ('Source strength',
         'Total particles a site exports to all sinks (sum of its outgoing connections).',
         "weighted out-degree of the site in the connectivity graph",
         'particles'),
        ('Sink strength',
         'Total particles a site receives from all sources (sum of its incoming connections).',
         "weighted in-degree of the site in the connectivity graph",
         'particles'),
        ('Degree centrality',
         'How many other sites a site is directly connected to, relative to the network size.',
         'NetworkX degree_centrality (normalised)',
         'proportion (0-1)'),
        ('Betweenness centrality',
         'How often a site lies on the strongest-connection shortest paths between other site '
         'pairs, identifies stepping-stone sites. Higher = more central.',
         'betweenness(v) = sum over s!=v!=t of sigma_st(v)/sigma_st (fraction of shortest paths through v); shortest paths minimise edge distance = 1/exchange, so strong connections are short; NetworkX betweenness_centrality, normalised',
         'proportion (0-1)'),
        ('Closeness centrality',
         'How strongly connected a site is to all others (reachability via strong connections). '
         'Higher = more connected.',
         'closeness(v) = (n-1) / sum over u of d(v,u), where d = shortest-path distance using edge distance = 1/exchange (strong = short); NetworkX closeness_centrality',
         'proportion (0-1)'),
        ('Overall composite (hotspot rank)',
         'Combined ranking used to identify overall hotspots. Each of the six metrics is ranked '
         '(rank 1 = highest value). The composite is the mean of those six ranks, so a lower '
         'composite means a higher-priority site.',
         'mean of ranks of [source, sink, self-recruitment, betweenness, closeness, degree]',
         'mean rank (lower = better)'),
        ('Functional composite',
         'As overall composite, but using only the functional metrics.',
         'mean of ranks of [source, sink, self-recruitment]',
         'mean rank (lower = better)'),
        ('Centrality composite',
         'As overall composite, but using only the centrality metrics.',
         'mean of ranks of [betweenness, closeness, degree]',
         'mean rank (lower = better)'),
        ('Custom composite',
         'Optional user-defined composite combining any chosen subset of the six metrics, so a '
         'study can prioritise the aspects of connectivity that matter to it. Built the same way '
         'as the others (mean of the chosen metrics\' ranks). Present only when requested.',
         'mean of ranks of the user-selected metrics',
         'mean rank (lower = better)'),
        ('Network density',
         'Proportion of all possible site-to-site connections that are present.',
         'edges / (n_sites * (n_sites - 1))',
         'proportion (0-1)'),
        ('Modularity',
         'Strength of the community structure found by Louvain: higher means communities are more '
         'clearly separated. It is a property of the network (a structureless network cannot reach '
         'a high value no matter how it is partitioned), which Louvain maximises.',
         'Q = (1/2m) sum_ij [A_ij - k_i*k_j/(2m)] * delta(c_i,c_j); A = weighted adjacency, m = total edge weight, k = node strength, c = community; maximised by Louvain',
         'score (-0.5 to 1)'),
        ('Communities',
         'Number of distinct communities (clusters of strongly inter-connected sites) detected '
         'by the Louvain algorithm.',
         'count of Louvain communities',
         'count'),
        ('Study extent (sites)',
         'Number of sites the user defined as valid release or settlement locations under the chosen grid strategies.',
         'count of valid user-defined sites',
         'count'),
        ('Potential connectivity sites',
         'The number of grid cells where connectivity could occur. If using species-specific release/settlement, this is the count of cells with at least one species occurrence datapoint. If using "all sites", this equals the total number of coastal/ocean cells in your study extent.',
         'intersection of occurrence sites and study-extent sites',
         'count'),
        ('Realised connectivity sites',
         'The number of sites that actually contribute to overall network connectivity (i.e., at least one successfully settled particle arrived at or departed from the site in the simulation results). This is always less than or equal to potential connectivity sites.',
         'count of sites with at least one non-zero matrix entry',
         'count'),
        ('Unique connections',
         'Number of distinct directed site-to-site links that carried at least one particle (the network edge count, independent of particle volume).',
         'count of non-zero cells in the connectivity matrix',
         'count'),
    ]
    metrics_df = pd.DataFrame(
        metric_definitions, columns=['Metric', 'Definition', 'Formula / method', 'Units']
    )

    # =====================================================================
    # Sheet 4: column definitions for each result CSV file.
    # Every column that appears in connectivity_summary.csv,
    # network_site_metrics.csv, connectivity_matrix.csv, and
    # settlement_attribution.csv is listed with: file, column name,
    # one-sentence definition, units, formula (where one applies), and a
    # short worked example.
    # Notes:
    #   - The community connectivity_summary uses total_released / total_settled /
    #     total_recruited; the per-species file uses n_released / n_settled / n_recruited
    #     for the same quantities.  Both variants are listed here.
    #   - The passive pathway writes a subset of connectivity_summary columns:
    #     total_released, self_recruitment_rate, mean_distance_km,
    #     min_distance_km, max_distance_km only.
    #   - louvain_community and site_name are optional columns in
    #     network_site_metrics.csv (present only when community analysis was
    #     run, or when the user supplied named sites, respectively).
    #   - custom_composite is optional in network_site_metrics.csv (present
    #     only when add_custom_composite() was called).
    # =====================================================================
    column_definitions = [
        # ---- connectivity_summary.csv (community level) ----
        ('connectivity_summary.csv', 'n_species',
         'Number of species with at least one successfully settled trajectory.',
         'count', '—',
         '3  (three species had data; a fourth was present but all trajectories failed to settle)'),
        ('connectivity_summary.csv', 'total_released',
         'Total particles released across all species and release sites combined (community file only; '
         'per-species file uses n_released for the same quantity).',
         'count', '—',
         '15000  (5000 particles per species × 3 species)'),
        ('connectivity_summary.csv', 'n_released',
         'Particles released for this species (per-species file only; '
         'community file uses total_released for the equivalent quantity).',
         'count', '—',
         '5000'),
        ('connectivity_summary.csv', 'total_settled',
         'Particles that settled at any valid habitat cell within the PLD window, summed across all '
         'species (community file only; per-species file uses n_settled).',
         'count', '—',
         '8742'),
        ('connectivity_summary.csv', 'n_settled',
         'Particles that settled for this species (per-species file only; '
         'community file uses total_settled).',
         'count', '—',
         '2914'),
        ('connectivity_summary.csv', 'total_recruited',
         'Particles that settled at their own release site (the diagonal of the connectivity matrix), across all species (community file only; per-species file uses n_recruited).'
         'count', '—',
         '312'),
        ('connectivity_summary.csv', 'n_recruited',
         'Self-recruited particles for this species (per-species file only; '
         'community file uses total_recruited).',
         'count', '—',
         '104'),
        ('connectivity_summary.csv', 'settlement_rate',
         'Fraction of released particles that settled successfully at any valid habitat cell.',
         'proportion [0–1]',
         'n_settled / n_released  (or total_settled / total_released for the community file)',
         '0.583  (8742 settled out of 15000 released)'),
        ('connectivity_summary.csv', 'mortality_rate',
         'Fraction of released particles that failed to settle (exceeded maximum PLD without '
         'reaching suitable habitat). Null when release counts are unavailable.',
         'proportion [0–1]',
         '1 − settlement_rate  =  1 − (n_settled / n_released)',
         '0.417  (6258 particles out of 15000 did not settle)'),
        ('connectivity_summary.csv', 'self_recruitment_rate',
         'Fraction of settled particles whose settlement site is their own release site (the diagonal '
         'of the connectivity matrix). The settlement buffer is resolved during post-processing, so this '
         'is an exact match consistent with every other metric.',
         'proportion [0–1]',
         'n_recruited / n_settled  (or total_recruited / total_settled for the community file)',
         '0.036  (312 out of 8742 settled particles were self-recruited)'),
        ('connectivity_summary.csv', 'mean_distance_km',
         'Mean net straight-line displacement from release to final settlement location across '
         'all successfully settled trajectories.',
         'km', 'mean of great-circle distances between release and settlement cell centroids',
         '47.3  (average across all settled particles)'),
        ('connectivity_summary.csv', 'min_distance_km',
         'Minimum net displacement across all settled trajectories.',
         'km', 'min of per-trajectory great-circle displacements',
         '0.0  (particle settled in its own cell)'),
        ('connectivity_summary.csv', 'max_distance_km',
         'Maximum net displacement across all settled trajectories.',
         'km', 'max of per-trajectory great-circle displacements',
         '312.8'),
        ('connectivity_summary.csv', 'mean_pld_days',
         'Mean realised planktonic larval duration: average number of days between release and '
         'settlement across all settled trajectories.',
         'days', 'mean of settlement_day values recorded in pld_filter',
         '11.2'),
        ('connectivity_summary.csv', 'min_pld_days',
         'Minimum realised PLD across all settled trajectories.',
         'days', 'min of settlement_day values',
         '3'),
        ('connectivity_summary.csv', 'max_pld_days',
         'Maximum realised PLD across all settled trajectories.',
         'days', 'max of settlement_day values',
         '28'),
        ('connectivity_summary.csv', 'n_release_sites',
         'Number of distinct grid cells from which particles were released '
         '(union across species for the community file; exact species count for the per-species file).',
         'count', '—',
         '84'),
        ('connectivity_summary.csv', 'n_settlement_sites',
         'Number of distinct grid cells designated as valid settlement habitat '
         '(union across species for the community file; per-species count otherwise).',
         'count', '—',
         '1247'),

        # ---- network_site_metrics.csv ----
        ('network_site_metrics.csv', 'grid_id',
         'Unique integer identifier of the ocean grid cell; matches the row/column labels '
         'in connectivity_matrix.csv and the grid_id column in the grids/ CSV files.',
         'integer', '—',
         '42'),
        ('network_site_metrics.csv', 'latitude',
         'Latitude of the grid cell centroid, in decimal degrees north.',
         'degrees N', '—',
         '50.25'),
        ('network_site_metrics.csv', 'longitude',
         'Longitude of the grid cell centroid, in decimal degrees east.',
         'degrees E', '—',
         '−4.12'),
        ('network_site_metrics.csv', 'louvain_community',
         'Integer community membership label assigned by the Louvain algorithm '
         '(sites with the same integer belong to the same community). '
         'Optional — present only when community detection was run.',
         'integer category', 'Louvain best_partition on the undirected weighted graph',
         '2  (this site belongs to community 2)'),
        ('network_site_metrics.csv', 'site_name',
         'Human-readable name supplied via a site_name column in the occurrence data. '
         'Optional — absent when only grid-based (site_<grid_id>) labels were available.',
         'string', '—',
         'Plymouth Sound'),
        ('network_site_metrics.csv', 'source_strength',
         'Total number of particles that left this site and settled at any other site '
         '(weighted out-degree). Self-recruitment is excluded when exclude_self_recruitment=True.',
         'particles', 'sum of row entries in connectivity_matrix for this site, '
         'excluding the diagonal if self-recruitment is excluded',
         '847  (847 particles exported to other sites)'),
        ('network_site_metrics.csv', 'sink_strength',
         'Total number of particles that arrived at this site from any other site '
         '(weighted in-degree). Self-recruitment is excluded when exclude_self_recruitment=True.',
         'particles', 'sum of column entries in connectivity_matrix for this site, '
         'excluding the diagonal if self-recruitment is excluded',
         '1203'),
        ('network_site_metrics.csv', 'self_recruitment',
         'Number of particles released from this site that settled at this site '
         '(the diagonal entry of the connectivity matrix for this site).',
         'particles', 'connectivity_matrix[site, site]',
         '23'),
        ('network_site_metrics.csv', 'betweenness',
         'Betweenness centrality: fraction of all strongest-connection shortest paths between other '
         'site pairs that pass through this site. High values identify stepping-stone sites.',
         'proportion [0–1]',
         'NetworkX betweenness_centrality(G, weight="distance", normalized=True), distance = 1/exchange',
         '0.041  (this site lies on 4.1 % of the strongest-path routes)'),
        ('network_site_metrics.csv', 'closeness',
         'Closeness centrality: how strongly connected this site is to all others '
         '(inverse of mean inverse-strength shortest path length; higher = more connected).',
         'proportion [0–1]',
         'NetworkX closeness_centrality(G, distance="distance"), distance = 1/exchange',
         '0.219'),
        ('network_site_metrics.csv', 'degree',
         'Degree centrality: proportion of other sites this site is directly connected to '
         '(in- and out-edges combined, normalised by 2×(n−1) for directed graphs).',
         'proportion [0–1]',
         'NetworkX degree_centrality(G)',
         '0.083  (connected to 8.3 % of all possible partners)'),
        ('network_site_metrics.csv', 'source_strength_rank',
         'Integer rank of this site by source_strength (rank 1 = highest source strength = '
         'best exporter). Ties share the lowest rank number (pandas method="min").',
         'integer [1–n_sites]',
         'source_strength.rank(ascending=False, method="dense")',
         '7  (7th strongest exporter in the network)'),
        ('network_site_metrics.csv', 'sink_strength_rank',
         'Integer rank of this site by sink_strength (rank 1 = highest sink strength).',
         'integer [1–n_sites]',
         'sink_strength.rank(ascending=False, method="dense")',
         '3'),
        ('network_site_metrics.csv', 'self_recruitment_rank',
         'Integer rank of this site by self_recruitment (rank 1 = most self-recruited).',
         'integer [1–n_sites]',
         'self_recruitment.rank(ascending=False, method="dense")',
         '12'),
        ('network_site_metrics.csv', 'betweenness_rank',
         'Integer rank of this site by betweenness centrality (rank 1 = highest betweenness).',
         'integer [1–n_sites]',
         'betweenness.rank(ascending=False, method="dense")',
         '4'),
        ('network_site_metrics.csv', 'closeness_rank',
         'Integer rank of this site by closeness centrality (rank 1 = highest closeness).',
         'integer [1–n_sites]',
         'closeness.rank(ascending=False, method="dense")',
         '15'),
        ('network_site_metrics.csv', 'degree_rank',
         'Integer rank of this site by degree centrality (rank 1 = highest degree).',
         'integer [1–n_sites]',
         'degree.rank(ascending=False, method="dense")',
         '9'),
        ('network_site_metrics.csv', 'functional_composite_rank',
         'Mean of the three functional metric ranks (source_strength_rank, sink_strength_rank, '
         'self_recruitment_rank). Lower value = higher priority site. Ties in any component '
         'rank propagate into the composite.',
         'mean rank (lower = higher priority)',
         'mean(source_strength_rank, sink_strength_rank, self_recruitment_rank)',
         '7.33  (mean of ranks 7, 3, 12)'),
        ('network_site_metrics.csv', 'centrality_composite_rank',
         'Mean of the three centrality metric ranks (betweenness_rank, closeness_rank, '
         'degree_rank). Lower value = higher priority site.',
         'mean rank (lower = higher priority)',
         'mean(betweenness_rank, closeness_rank, degree_rank)',
         '9.33  (mean of ranks 4, 15, 9)'),
        ('network_site_metrics.csv', 'overall_composite_rank',
         'Mean of all six metric ranks. The primary sorting column in the exported file and '
         'the default hotspot-map metric. Lower value = higher overall priority.',
         'mean rank (lower = higher priority)',
         'mean(source_strength_rank, sink_strength_rank, self_recruitment_rank, '
         'betweenness_rank, closeness_rank, degree_rank)',
         '8.33  (mean of 7, 3, 12, 4, 15, 9)'),
        ('network_site_metrics.csv', 'top10_count',
         'Number of the six individual metrics for which this site ranks in the top 10. '
         'A quick tally of how consistently high-ranking a site is across different metric types.',
         'integer [0–6]',
         'sum of (rank <= 10) across the six metric rank columns',
         '3  (top 10 for source_strength, betweenness, and degree)'),
        ('network_site_metrics.csv', 'custom_composite_rank',
         'Optional user-defined composite built by add_custom_composite(). Mean of the ranks '
         'of whichever subset of the six metrics the user selected. Same scale as the other '
         'composites. Present only when this function was called. Exported as '
         'custom_composite_rank; the in-memory column is named custom_composite.',
         'mean rank (lower = higher priority)',
         'mean of user-chosen metric ranks',
         '9.5  (user selected betweenness and closeness; mean of ranks 4, 15)'),

        # ---- connectivity_matrix.csv ----
        ('connectivity_matrix.csv', 'Row index (unlabelled first column)',
         'Source site label in the form site_<grid_id>. Each row represents one release site.',
         'string', '—',
         'site_42'),
        ('connectivity_matrix.csv', 'Column headers (one per sink site)',
         'Sink site labels in the form site_<grid_id>, one column per site. '
         'The value at row i, column j is the number of particles released from site i '
         'that settled at site j. The diagonal (i == j) is the self-recruitment count.',
         'string (header); integer particle count (cell value)',
         'connectivity_matrix[source_site, sink_site]',
         'Row site_42, column site_77 = 14  '
         '(14 particles from site 42 settled at site 77). '
         'Row site_42, column site_42 = 23  (23 self-recruited).'),

        # ---- settlement_attribution.csv ----
        ('settlement_attribution.csv', 'species',
         'Species name matching the subfolder and connectivity_summary filename for this species.',
         'string', '—',
         'Semibalanus balanoides'),
        ('settlement_attribution.csv', 'buffer_cell_grid_id',
         'Grid ID of a cell that falls within the settlement buffer around at least one real '
         'settlement site. Particles landing here are credited to the attributed_settlement_site.',
         'integer', '—',
         '43  (a cell one step away from real site 42)'),
        ('settlement_attribution.csv', 'attributed_settlement_site',
         'Grid ID of the real (non-buffered) settlement site nearest to this buffer cell, '
         'found by cKDTree nearest-neighbour search on 2-D grid position.',
         'integer',
         'cKDTree.query on grid (row, col) positions of all real settlement sites',
         '42  (site 42 is the closest real settlement site to buffer cell 43)'),

        # ---- network_summary.csv (Type 2 network-level scalars) ----
        ('network_summary.csv', 'Metric',
         'Name of the network-level summary statistic (e.g. number of species, study extent, '
         'potential connectivity sites, realised connectivity sites, unique connections).',
         'text', '—', 'unique_connections'),
        ('network_summary.csv', 'Value',
         'Value of the corresponding network-level statistic for this dataset/scale.',
         'varies', '—', '128'),

        # ---- community_trait_sensitivity_<scenario>.csv (Type 3 trait sensitivity) ----
        ('community_trait_sensitivity_<scenario>.csv', 'Metric',
         'Connectivity metric or comparison statistic (e.g. mean_distance_km, Mantel correlation (r), '
         'Hotspot overlap (%)) compared between the original and trait-modified scenario.',
         'text', '—', 'mean_distance_km'),
        ('community_trait_sensitivity_<scenario>.csv', 'Original',
         'Metric value under the original (unmodified) traits. Blank for statistics that only exist '
         'for the comparison (e.g. Mantel r, hotspot overlap).',
         'varies', '—', '84.2'),
        ('community_trait_sensitivity_<scenario>.csv', 'Modified',
         'Metric value under the trait-modified scenario.',
         'varies', '—', '91.7'),
        ('community_trait_sensitivity_<scenario>.csv', 'Change',
         'Modified minus Original. Blank where Original is not defined.',
         'varies', 'Modified - Original', '+7.5'),
    ]
    col_defs_df = pd.DataFrame(
        column_definitions,
        columns=['File', 'Column', 'Definition', 'Units', 'Formula', 'Example']
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            outputs_df.to_excel(writer, sheet_name='Outputs Reference', index=False)
            config_df.to_excel(writer, sheet_name='Run Configuration', index=False)
            metrics_df.to_excel(writer, sheet_name='Metric Definitions', index=False)
            col_defs_df.to_excel(writer, sheet_name='Column Definitions', index=False)

            # Tidy column widths for readability
            for sheet_name, df in [('Outputs Reference', outputs_df),
                                   ('Run Configuration', config_df),
                                   ('Metric Definitions', metrics_df),
                                   ('Column Definitions', col_defs_df)]:
                ws = writer.sheets[sheet_name]
                for i, col in enumerate(df.columns):
                    max_len = max(
                        df[col].astype(str).map(len).max() if not df.empty else 0,
                        len(str(col))
                    )
                    col_letter = chr(ord('A') + i)
                    ws.column_dimensions[col_letter].width = min(max_len + 2, 80)
    except Exception as e:
        print(f"  Warning: Could not write Excel metadata file ({e}).")
        return None

    return output_path

## Build the Type 3 (trait sensitivity) original-vs-modified comparison table
def build_trait_sensitivity_table(type3):
    """Build the Type 3 original-vs-modified comparison table as a tidy DataFrame.

    Returns columns: Metric, Original, Modified, Change. Change is (modified - original)
    for paired numeric community metrics; comparison-only statistics (Mantel test, hotspot
    overlap, spatial agreement) carry their single value in the Modified column with Change
    left blank.
    """
    orig = (type3.get('original_metrics', {}) or {}).get('community', {}) or {}
    mod = (type3.get('modified_metrics', {}) or {}).get('community', {}) or {}
    cmp_ = type3.get('comparison', {}) or {}
    mt = cmp_.get('mantel_test', {}) or {}
    ov = cmp_.get('hotspot_overlap', {}) or {}

    rows = []
    paired = [
        ('Self-recruitment rate', 'self_recruitment_rate'),
        ('Settlement rate', 'settlement_rate'),
        ('Mortality rate', 'mortality_rate'),
        ('Mean realised PLD (days)', 'mean_pld_days'),
        ('Mean dispersal distance (km)', 'mean_distance_km'),
    ]
    for label, key in paired:
        o, m = orig.get(key), mod.get(key)
        change = (m - o) if isinstance(o, (int, float)) and isinstance(m, (int, float)) else None
        if o is not None or m is not None:
            rows.append({'Metric': label, 'Original': o, 'Modified': m, 'Change': change})

    if mt:
        rows.append({'Metric': 'Mantel correlation (r)', 'Original': None, 'Modified': mt.get('r'), 'Change': None})
        rows.append({'Metric': 'Mantel p-value', 'Original': None, 'Modified': mt.get('p_value'), 'Change': None})
        rows.append({'Metric': 'Mantel sites compared', 'Original': None, 'Modified': mt.get('n_sites'), 'Change': None})
    if ov:
        rows.append({'Metric': 'Shared hotspots', 'Original': None, 'Modified': ov.get('shared_sites'), 'Change': None})
        rows.append({'Metric': 'Hotspots possible', 'Original': None, 'Modified': ov.get('n_possible'), 'Change': None})
        rows.append({'Metric': 'Hotspot overlap (%)', 'Original': None, 'Modified': ov.get('overlap_percentage'), 'Change': None})
    if cmp_.get('spatial_agreement') is not None:
        rows.append({'Metric': 'Spatial agreement', 'Original': None, 'Modified': cmp_.get('spatial_agreement'), 'Change': None})

    return pd.DataFrame(rows, columns=['Metric', 'Original', 'Modified', 'Change'])

## Build the run-configuration dict for the trait-defined pathway metadata export
def trait_metadata(spp_params, traits, hydro_params, runtime, adjusted_runtime,
                   interval, release_grids, settlement_grids, type3=None, extra=None):
    """
    Build the run-configuration dictionary for the trait-defined pathway.

    Returns a nested dict that mirrors the passive pathway's `passive_metadata()` output.
    The dict is written into the 'Run Configuration' sheet of the project's `comcon_outputs.xlsx` metadata file by `write_project_metadata()`.

    Parameters
    ----------
    spp_params : dict
        Species parameters from the workflow.
    traits : list of dict
        Species trait data.
    hydro_params : dict
        Hydrodynamic data parameters.
    runtime : float
        Requested runtime in days.
    adjusted_runtime : float
        Adjusted runtime after buffer.
    interval : float
        Release interval in days.
    release_grids, settlement_grids : dict
        Release and settlement grid DataFrames by species.

    Returns
    -------
    dict
        Nested configuration dictionary.
    """
    # Build species summary
    species_summary = []
    for t in (traits if isinstance(traits, list) else traits.to_dict('records')):
        sp_name = t.get('species_name', 'unknown')
        n_release = len(release_grids.get(sp_name, release_grids.get(
            standardise_species_name(sp_name), pd.DataFrame())))
        species_summary.append({
            'species': sp_name,
            'min_PLD': t.get('min_PLD'),
            'max_PLD': t.get('max_PLD'),
            'spawning_start': t.get('spawning_start'),
            'spawning_end': t.get('spawning_end'),
            'n_release_sites': n_release,
        })

    # Coerce runtimes to a float number of days. They may arrive as a timedelta (e.g. from a datetime subtraction), which float() cannot convert directly.
    def _coerce_days(v):
        if v is None:
            return None
        if isinstance(v, timedelta):
            return float(v.days + v.seconds / 86400)
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    metadata = {
        'comcon_version': '1.0.0',
        'pathway': 'trait_defined',
        'run_timestamp': datetime.now().isoformat(),
        'domain': {
            'dataset_id': hydro_params.get('dataset_id', 'unknown') if hydro_params else 'unknown',
        },
        'species': species_summary,
        'simulation_parameters': {
            'runtime_days': _coerce_days(runtime),
            'adjusted_runtime_days': _coerce_days(adjusted_runtime),
            'release_interval_days': _coerce_days(interval),
            'release_strategy': spp_params.get('release_strategy', 'unknown'),
            'settlement_strategy': spp_params.get('settlement_strategy', 'unknown'),
            'habitat_filtering': spp_params.get('habitat_filtering', 'unknown'),
            'data_source': spp_params.get('data_source', 'unknown'),
            'has_site_names': spp_params.get('has_site_names', False),
        },
        'n_species': len(species_summary),
        'total_release_sites': sum(s['n_release_sites'] for s in species_summary),
    }

    # Type 3 trait sensitivity: only the user-chosen scenario belongs in Run Configuration.
    # The numeric comparison results live in the Type 3 results CSV
    # (<dataset>_trait_sensitivity_<scenario>.csv); the comparison maps are listed in the
    # Outputs Reference catalogue, not here.
    if type3:
        metadata['trait_sensitivity'] = {
            'scenario': type3.get('scenario', 'unknown'),
        }

    # Merge any extra user-chosen settings (diffusivity, domain bounds, timestep, particles,
    # scales, which analyses ran, exclude_self_recruitment, ...). Dict sections are merged into
    # the matching metadata section; scalars are set directly. Rule: only config, never results.
    if extra:
        for _section, _vals in extra.items():
            if isinstance(_vals, dict) and isinstance(metadata.get(_section), dict):
                metadata[_section].update({k: v for k, v in _vals.items() if v is not None})
            elif isinstance(_vals, dict):
                metadata[_section] = {k: v for k, v in _vals.items() if v is not None}
            elif _vals is not None:
                metadata[_section] = _vals

    return metadata

## Main analysis wrapper: interactive selection and dispatch for all analysis types
def post_analysis(final_processed):
    """
    Main analysis wrapper with interactive selection of analysis types.

    Guides users through four analysis pathways:
    - Type 1: Connectivity metrics (mortality, self-recruitment, dispersal, PLD)
    - Type 2: Network analysis (functional roles, centrality, hotspots)
    - Type 3: Trait sensitivity analysis (baseline vs modified trait scenario)

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

    results = {}
    output_dir = None

    # Resolve raw trajectory data for mortality calculation.
    # Uses full traj DataFrame if available, otherwise loads just obs=0 rows from chunks.
    # This is fully automatic , users don't need to know about the chunked storage.
    _traj_raw = None
    if 'traj' in globals() and isinstance(traj, pd.DataFrame) and not traj.empty:
        _traj_raw = traj
    elif 'chunks_dir' in globals() and chunks_dir and os.path.isdir(str(chunks_dir)):
        try:
            _traj_raw = load_traj_starts_from_chunks(chunks_dir)
        except Exception:
            pass  # mortality will be skipped if this fails

    # Check if temporal buffer was used (enables trait sensitivity in Type 3).
    # The buffer is stored in hydro_params during Step 1, so read it from there with sensible fallbacks.
    _hp = globals().get('hydro_params', {}) or {}
    buffer_before = _hp.get('buffer_before', globals().get('buffer_before', 0))
    buffer_after = _hp.get('buffer_after', globals().get('buffer_after', 0))
    buffer_used = (buffer_before > 0 or buffer_after > 0)

    # Step 1: Analysis Scope Selection (applies to all analyses)
    section_header("Analysis Scope Selection")

    print("\nThis package allows you to run connectivity analyses at the species level and/or at the community level.")

    # Get list of species with data
    species_with_data = [sp for sp, df in final_processed.items() if not df.empty]
    n_species = len(species_with_data)

    print(f"\nYou have {n_species} species with trajectory data:")
    for i, sp in enumerate(species_with_data, 1):
        n_traj = final_processed[sp]['trajectory'].nunique()
        print(f"  {i}. {display_species_name(sp)} ({n_traj:,} trajectories)")

    print("\nHow would you like to run the connectivity analyses?")
    print("  1. Community level only (all species combined)")
    print("  2. Per-species only (each species analysed separately)")
    print("  3. Both community and per-species")
    print("  4. Custom species selection")

    while True:
        scope_choice = input("\nSelect option (1-4) [default: 3]: ").strip() or '3'
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
        all_indices = ','.join(str(i) for i in range(1, len(species_with_data) + 1))
        indices_str = input("Species numbers [default: all]: ").strip() or all_indices

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

    run_type1 = get_yes_no("\nRun Type 1: Connectivity Metrics? (yes/no)", default='yes')
    run_type2 = get_yes_no("Run Type 2: Network Analysis? (yes/no)", default='yes')
    run_type3 = get_yes_no("Run Type 3: Trait Sensitivity? (yes/no)", default='yes')

    if not any([run_type1, run_type2, run_type3]):
        print("\nNo analyses selected. Returning empty results.")
        return results

        # Safe default for cross-type variables
    exclude_lr = False

    # Type 1: Connectivity Metrics
    if run_type1:
        section_header("Type 1: Connectivity Metrics")

        try:
            type1_results = {}

            # Run connectivity metrics (community + per-species as selected)
            print("  Computing connectivity metrics...\n")
            with contextlib.redirect_stdout(io.StringIO()):
                community_results = calculate_connectivity_metrics(
                    selected_processed, release_grids, settlement_grids, traits, _traj_raw,
                    print_summary=False
                )
            type1_results['community'] = community_results

            if analysis_scope['run_per_species']:
                type1_results['per_species'] = {}
                for sp in analysis_scope['selected_species']:
                    sp_data = {sp: selected_processed[sp]}
                    sp_release = {sp: release_grids.get(sp)} if sp in release_grids else {}
                    sp_settle = {
                        sp: settlement_grids.get(sp)} if settlement_grids and sp in settlement_grids else None
                    sp_traits = [t for t in traits if standardise_species_name(t.get('species_name', '')) == sp]
                    if sp_traits:
                        with contextlib.redirect_stdout(io.StringIO()):
                            sp_results = calculate_connectivity_metrics(
                                sp_data, sp_release, sp_settle, sp_traits, _traj_raw,
                                print_summary=False
                            )
                        type1_results['per_species'][sp] = sp_results

            results['type1'] = type1_results

            # Display interactive summary table viewer
            subsection_header("Connectivity Metrics Summary")
            trait_metrics_viewer(type1_results)

            # Ask about saving
            if get_yes_no("\nSave Type 1 results to CSV?", default='yes'):
                default_dir = get_default_dir('results')
                output_dir = get_valid_dir("Output directory", default=default_dir)

                def _clean_trait_summary(_df):
                    return _df.drop(columns=[c for c in _df.columns if c.endswith('_std') or c.startswith('std_')], errors='ignore')

                for key, result_data in type1_results.items():
                    if key == 'per_species':
                        for sp, sp_data in result_data.items():
                            if 'community' in sp_data:
                                sp_dir = os.path.join(output_dir, sp)
                                os.makedirs(sp_dir, exist_ok=True)
                                _clean_trait_summary(pd.DataFrame([sp_data['community']])).to_csv(
                                    os.path.join(sp_dir, f'{sp}_connectivity_summary.csv'), index=False)
                    elif 'community' in result_data:
                        comm_dir = os.path.join(output_dir, key)
                        os.makedirs(comm_dir, exist_ok=True)
                        _clean_trait_summary(pd.DataFrame([result_data['community']])).to_csv(
                            os.path.join(comm_dir, f'{key}_connectivity_summary.csv'), index=False)

                # Settlement attribution: buffer-cell → real settlement site, per species
                _gids = globals().get('grid_ids')
                _buf = globals().get('buffer', 0)
                if settlement_grids is not None and _gids is not None and _buf > 0:
                    _attr = build_settlement_attribution(settlement_grids, _gids, _buf)
                    for sp, attr_df in _attr.items():
                        sp_dir = os.path.join(output_dir, sp)
                        os.makedirs(sp_dir, exist_ok=True)
                        attr_df.insert(0, 'species', sp)
                        attr_df.to_csv(os.path.join(sp_dir, f'{sp}_settlement_attribution.csv'), index=False)

                print(f"Results saved to: {output_dir}")

            # Offer dispersal animation
                # NEW
                subsection_header("Dispersal Animation")
                if get_yes_no("\nCreate particle dispersal animation?", default='yes'):
                    print("\n  This creates an animated map showing particles moving over time.")
                    print("  The animation uses the raw, combined trajectory data (all particles pooled across species), not the filtered per-species connectivity data used for the matrices and metrics.")

                save_anim = get_yes_no("Save animation as MP4 file?", default='yes')

                # Auto-load trajectory data from chunks if needed
                anim_traj = None
                if 'traj' in globals() and isinstance(traj, pd.DataFrame) and not traj.empty:
                    anim_traj = traj
                if anim_traj is None and 'chunks_dir' in globals() and chunks_dir and os.path.isdir(str(chunks_dir)):
                    print("  Loading trajectory data for animation...")
                    anim_traj = load_traj_from_chunks(chunks_dir)

                if anim_traj is None or (isinstance(anim_traj, pd.DataFrame) and anim_traj.empty):
                    print("  Warning: No trajectory data available for animation. Skipping.")
                else:
                    if save_anim:
                        default_dir = get_default_dir('animation')
                        anim_path = os.path.join(default_dir, 'dispersal_animation.mp4')
                        anim_path = get_save_path("Output path", default=anim_path, extensions=['.mp4']) or anim_path
                        os.makedirs(os.path.dirname(anim_path), exist_ok=True)
                        dispersal_animation(anim_traj, output_path=anim_path)
                    else:
                        dispersal_animation(anim_traj)
                    del anim_traj
                    gc.collect()

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
        print("  3. Multi-scale (compare local, regional, seascape and full domain patterns)")

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
            scale_names = [f'{scale_km}_km']
            print(f"\nUsing {scale_km} km distance threshold")
        else:
            local_km = get_positive_integer("Local scale threshold (km)", default=10, min_val=1, max_val=50)
            regional_km = get_positive_integer("Regional scale threshold (km)", default=30, min_val=1, max_val=100)
            seascape_km = get_positive_integer("Seascape scale threshold (km)", default=100, min_val=1, max_val=1000)
            scales = [None, local_km, regional_km, seascape_km]
            scale_names = ['full_domain', 'local', 'regional', 'seascape']
            print(f"\nUsing multi-scale: full domain, {local_km}km (local), {regional_km}km (regional), {seascape_km}km (seascape)")

        # Step 2b: Run Analysis
        exclude_lr = get_yes_no(
            "Do you want to exclude self-recruitment (particles returning to their release site) from the source/sink metrics?",
            default='yes')
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
                    _nsp = len(analysis_scope['selected_species'])
                    print(f"  community ({_nsp} species) ... ", end="", flush=True)
                    community_network = network_analysis(
                        selected_processed, release_grids, settlement_grids,
                        grid_df_for_mapping, grid_ids, full_grid_df, scale_km,
                        exclude_self_recruitment=exclude_lr, verbose=False
                    )
                    if community_network:
                        _compute_network_stats(community_network)
                        scale_results['community'] = community_network
                        _nc = community_network.get('summary_stats', {}).get('Unique connections', 0)
                        _note = "  (few connections, so metrics may be unreliable at this scale)" if (
                                    scale_km is not None and _nc < 20) else ""
                        print(f"{_nc} connections{_note}")
                    else:
                        print("no result")

                # Per-species network analysis
                if analysis_scope['run_per_species']:
                    scale_results['per_species'] = {}
                    for i, sp in enumerate(analysis_scope['selected_species'], 1):
                        print(f"  {display_species_name(sp)} ... ", end="", flush=True)
                        sp_data = {sp: final_processed[sp]}
                        sp_settle = {
                            sp: settlement_grids[sp]} if settlement_grids and sp in settlement_grids else None

                        # Pass full release_grids to maintain consistent study extent across all species
                        sp_network = network_analysis(
                            sp_data, release_grids, sp_settle,
                            grid_df_for_mapping, grid_ids, full_grid_df, scale_km,
                            exclude_self_recruitment=exclude_lr, verbose=False
                        )
                        if sp_network:
                            _compute_network_stats(sp_network)
                            scale_results['per_species'][sp] = sp_network
                            _nc = sp_network.get('summary_stats', {}).get('Unique connections', 0)
                            _note = "  (few connections, so metrics may be unreliable at this scale)" if (
                                        scale_km is not None and _nc < 20) else ""
                            print(f"{_nc} connections{_note}")
                        else:
                            print("no result")

                if scale_results:
                    all_network_results[scale_name] = scale_results

            results['type2'] = all_network_results

            # Optional: build a custom composite (combine specific metrics for ranking)
            if all_network_results:
                print("\n  All six individual metrics (source, sink, self-recruitment, betweenness, closeness, degree) are always computed. A composite only adds an extra hotspot-ranking layer built from them.")
            if all_network_results and get_yes_no(
                    "\nBuild a custom composite?", default='no'):
                _metric_options = ['source_strength', 'sink_strength', 'self_recruitment', 'betweenness', 'closeness', 'degree']
                print("  Available metrics:")
                for _i, _m in enumerate(_metric_options, 1):
                    print(f"    {_i}. {_m}")
                chosen = []
                while not chosen:
                    raw = input("  Enter the numbers of the metrics to combine (comma-separated, e.g. 1,4): ").strip()
                    _picks = [p.strip() for p in raw.split(',') if p.strip()]
                    chosen = [_metric_options[int(p) - 1] for p in _picks if
                              p.isdigit() and 1 <= int(p) <= len(_metric_options)]
                    if not chosen:
                        print("  No valid selection. Please enter numbers from the list above.")
                ok = True
                for scale_data in all_network_results.values():
                    targets = []
                    if 'community' in scale_data:
                        targets.append(scale_data['community'])
                    if 'per_species' in scale_data:
                        targets.extend(scale_data['per_species'].values())
                    for nr in targets:
                        try:
                            add_custom_composite(nr, chosen, verbose=False)
                        except ValueError as e:
                            print(f"  {e}")
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    print(f"  Custom composite created from: {', '.join(chosen)}. Select custom_composite in the network metric viewer dropdown to view it, or use the 'custom_composite' column in the exported CSVs.")

            # Optional: whether to include the combined composite hotspot ranking figure at all.
            _build_composite_fig = (get_yes_no("\nInclude the composite hotspot ranking figure?", default='yes')
                                    if all_network_results else True)

            # Step 2c: Save figures (optional)
            save_figs = get_yes_no("\nSave Type 2 figures to file?", default='yes')
            output_dir = None
            if save_figs:
                default_dir = get_default_dir('grids')
                output_dir = get_valid_dir("Output directory", default=default_dir)
                os.makedirs(output_dir, exist_ok=True)

                subsection_header("Saving figures")
                _summary_done = set()  # write the network summary scalars once per dataset
                for s_name, scale_data in all_network_results.items():
                    # Collect all (level_label, network_results) pairs for this scale
                    pairs = []
                    if 'community' in scale_data:
                        pairs.append(('community', scale_data['community']))
                    if 'per_species' in scale_data:
                        for sp, sp_nr in scale_data['per_species'].items():
                            pairs.append((sp, sp_nr))

                    for level_label, nr in pairs:
                        fig_subdir = os.path.join(output_dir, f'{level_label}_{s_name}')
                        os.makedirs(fig_subdir, exist_ok=True)
                        print(f"  Saving: {level_label} , {s_name}")
                        with contextlib.redirect_stdout(io.StringIO()):
                            # Capture values in default args to avoid lambda late-binding bug
                            _sm = nr['site_metrics']
                            _cm = nr['connectivity_matrix']
                            _t10 = nr.get('top10_overall', _sm.head(10))
                            _ng = nr.get('network_graph')
                            _title = f'{level_label} , {s_name}'

                            save_list = [
                                ('hotspot_overall', lambda sm=_sm, t10=_t10, ng=_ng, t=_title: create_network_map(
                                    sm, t10, title=t,
                                    composite_col='overall_composite', cmap='viridis_r',
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

                            # Per-metric hotspot maps
                            for metric_col, metric_label in [
                                ('source_strength', 'source_strength'),
                                ('sink_strength', 'sink_strength'),
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

                            # Skip the combined composite hotspot ranking figure if not wanted.
                            if not _build_composite_fig:
                                save_list = [x for x in save_list if x[0] != 'hotspot_overall']

                            for fn, make_fig in save_list:
                                try:
                                    fig = make_fig()
                                    if fig is not None:
                                        save_figure(fig, os.path.join(fig_subdir, fn))
                                        plt.close(fig)
                                except Exception:
                                    pass

                                # Save result CSVs: results/<level>/<level>_<scale>_*.csv
                            results_root = get_default_dir('results')
                            csv_subdir = os.path.join(results_root, level_label)
                            os.makedirs(csv_subdir, exist_ok=True)
                            _csv_prefix = f'{level_label}_{s_name}'
                            try:
                                with contextlib.redirect_stdout(io.StringIO()):
                                    export_network_results(nr, output_dir=csv_subdir, prefix=_csv_prefix)
                            except Exception:
                                pass
                            if level_label not in _summary_done:
                                try:
                                    export_network_summary(nr, csv_subdir, level_label)
                                except Exception:
                                    pass
                                _summary_done.add(level_label)

                print(f"\n  Figures saved to: {output_dir}")
                print(f"  Result CSVs saved to: {get_default_dir('results')}")

            # Step 2f: Interactive Results Viewer
            if all_network_results:
                section_header("Type 2: Results")

                info_box([
                    "Type 2 treats the connectivity data as a directed weighted network.",
                    "In this network nodes = sites, edges = counts of particles moving between sites.",
                    "Five complementary outputs are generated:",
                    "",
                    "1. Network analysis summary: ",
                    "   - Number of species"
                    "   - Study extent"
                    "   - Potential connectivity sites"
                    "   - Realised connectivity sites"
                    "   - Unique connections",
                    "",
                    "2. Connectivity matrix: ",
                    "   - Source-sink matrix of particle exchanges",
                    "",
                    "3. Directed network with hotspots: ",
                    "   - All sites are coloured by the selected network metric (rank 1 = best)",
                    "   - The top 10 connectivity hotspots are indicated",
                    "   - Select 'Network only (no metric)' to see the directed structure: arrows for the strongest site-to-site links and rings for self-recruitment",
                    "",
                    "4. Site × metric heatmap: ",
                    "   - Compact overview of rank across all metrics for every site in the dataset",
                    "",
                    "5. Louvain communities: ",
                    "   - Sites are grouped by shared connectivity patterns",
                    "   - A distinct colour per community (the biggest communities are labelled in the legend)",
                    "   - A table beneath the map gives the Louvain/graph parameters (nodes, edges, communities, and):",
                    "       - Density: percentage of possible site-to-site connections that exist (0-1). Low (<0.5) = most sites unconnected; high (>0.5) = most connected",
                    "       - Modularity: how cleanly the network separates into communities (-0.5 to 1). High (>=0.3) = well-separated communities; low (<0.3) = weak partition",
                    "",
                    "Use the dropdowns below each viewer to switch between analysis levels, spatial scales and (for the hotspot map) network metrics.",
                    "",
                ])

                print()
                subsection_header("Network summary")
                try:
                    network_metrics_viewer(all_network_results)
                    time.sleep(0.3)
                except Exception as e:
                    print(f"  Note: Network viewer could not be displayed: {e}")

                print()
                subsection_header("Connectivity matrix")
                try:
                    trait_matrix_viewer(all_network_results)
                except Exception as e:
                    print(f"  Note: Matrix viewer could not be displayed: {e}")

                print()
                subsection_header("Directed connectivity network with hotspots")
                try:
                    trait_metric_viewer(all_network_results)
                except Exception as e:
                    print(f"  Note: Metric viewer could not be displayed: {e}")
                subsection_header("Site × metric heatmap")
                try:
                    trait_heatmap_viewer(all_network_results)
                    time.sleep(0.5)  # Allow widget to render before continuing
                except Exception as e:
                    print(f"  Note: Heatmap viewer could not be displayed: {e}")

                print()
                subsection_header("Louvain communities")
                try:
                    trait_community_viewer(all_network_results)
                except Exception as e:
                    print(f"  Note: Community viewer could not be displayed: {e}")

            print("\n Type 2 analysis complete")

        except Exception as e:
            print(f"\nError in Type 2 analysis: {e}")
            traceback.print_exc()
            print("Continuing to next analysis...")

    # Type 3: Trait Sensitivity Analysis
    if run_type3:
        try:
            section_header("Type 3: Trait Sensitivity Analysis")
            print("\nThis tests how connectivity changes with different trait values.")
            # Ask about self-recruitment for Type 3 (if not already set by Type 2)
            if not run_type2:
                exclude_lr = get_yes_no(
                    "Do you want to exclude self-recruitment (particles returning to their release site) from the source/sink metrics?",
                    default='yes')

            if not buffer_used:
                print("\nTrait sensitivity analysis requires a temporal buffer in the simulation.")
                print("To use this feature, re-run workflow_step(1) with:")
                print("  buffer_before > 0  or  buffer_after > 0")
                print("\nSkipping Type 3.")
            else:
                # Trait sensitivity (the only Type 3 option in V1)
                print(f"Your temporal buffer: {buffer_before} days before, {buffer_after} days after")

                if 'type2' in results and scales:
                    if len(scales) == 1:
                        sensitivity_scale = scales[0]
                    else:
                        # Several Type 2 scales , let the user pick which to reuse.
                        # None is the full domain (no distance filter); keep it sortable (placed last).
                        _ordered = sorted(scales, key=lambda s: (s is None, s if s is not None else 0))
                        print("\nType 2 was run at several distance thresholds, so choose which one the sensitivity analysis should use.")
                        for _i, _s in enumerate(_ordered, start=1):
                            _lbl = "full domain (no distance filter)" if _s is None else f"{_s} km"
                            print(f"  {_i}. {_lbl}")
                        _choice = get_positive_integer(
                            f"Select a scale (1-{len(_ordered)}) [default: {len(_ordered)}]",
                            default=len(_ordered), min_val=1, max_val=len(_ordered)
                        )
                        sensitivity_scale = _ordered[_choice - 1]
                    if sensitivity_scale is None:
                        print("\nUsing the full domain (no distance filter) from Type 2.")
                    else:
                        print(f"\nUsing the {sensitivity_scale} km threshold from Type 2.")
                else:
                    # No Type 2 scale to reuse , ask with the same style as Type 2 (not a bare km default).
                    print("\nNo Type 2 scale to reuse. Choose a distance scale for the sensitivity analysis:")
                    print("  1. Full domain (include all connections regardless of distance)")
                    print("  2. Single scale (set one distance threshold)")
                    _sc = input("\nSelect option (1/2) [default: 1]: ").strip() or '1'
                    if _sc == '2':
                        sensitivity_scale = get_positive_integer("Distance threshold in km", default=30, min_val=1, max_val=500)
                    else:
                        sensitivity_scale = None

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
                        grid_df_for_mapping, grid_ids, full_grid_df, sensitivity_scale,
                        exclude_self_recruitment=exclude_lr, verbose=False)

                print("\nSelect trait modification scenario:")
                print("  1. Modify PLD (pelagic larval duration)")
                print("  2. Shift spawning window")

                while True:
                    mod_choice = input("\nSelect scenario (1/2) [default: 1]: ").strip() or '1'
                    if mod_choice in ['1', '2']:
                        break
                    print("  Invalid input. Please enter 1 or 2.")

                modified_traits = None
                scenario_name = ""

                if mod_choice == '1':
                    # PLD modification
                    print("\nPLD Modification (in days):")
                    print("  - You can modify min_PLD and max_PLD separately")
                    print("  - Enter a positive value to increase, or a negative value to decrease")
                    print("  - Example: +5 adds 5 days, -3 removes 3 days")

                    print("\nOptions:")
                    print("  1. Same change to both min and max PLD")
                    print("  2. Different changes to min and max PLD")
                    pld_option = input("\nSelect (1/2) [default: 1]: ").strip() or '1'

                    if pld_option == '2':
                        min_change = int(input("Days to add/subtract from min_PLD (e.g. enter -4 to subtract 4 days) [default: 7]: ").strip() or '0')
                        max_change = int(input("Days to add/subtract from max_PLD (e.g. enter -4 to subtract 4 days) [default: 7]: ").strip() or '7')
                        scenario_name = f"PLD min{'+' if min_change >= 0 else ''}{min_change}, max{'+' if max_change >= 0 else ''}{max_change}"
                    else:
                        days = int(input("Days to add/subtract from both (e.g. enter -4 to subtract 4 days) [default: 7]: ").strip() or '7')
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
                        start_change = int(
                            input("Days to shift spawning start (negative = earlier) [default: -7]: ").strip() or '-7')
                        end_change = int(
                            input("Days to shift spawning end (positive = later) [default: 7]: ").strip() or '7')
                        scenario_name = f"Spawn start {'+' if start_change >= 0 else ''}{start_change}, end{'+' if end_change >= 0 else ''}{end_change}"
                    else:
                        shift_days = int(
                            input("Days to shift entire window (positive = later) [default: 7]: ").strip() or '7')
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
                    print("Starting post-processing for the trait sensitivity analysis...")

                    try:
                        if 'traj' in globals() and isinstance(traj, pd.DataFrame) and not traj.empty:
                            _traj_source = traj
                        elif 'chunks_dir' in globals() and chunks_dir and os.path.isdir(str(chunks_dir)):
                            _traj_source = chunks_dir
                        else:
                            _traj_source = traj

                        modified_processed, _ = post_process(
                            _traj_source, modified_traits, release_grids, settlement_grids,
                            grid_ids, buffer=globals().get('buffer', 0)
                        )

                        if not modified_processed:
                            print(
                                "\nNote: The modified scenario produced no successful trajectories, so the sensitivity comparison was skipped.")
                            modified_network = None
                        else:
                            print("\nRunning network analysis on modified scenario...")
                            modified_network = network_analysis(
                                modified_processed, release_grids, settlement_grids,
                                grid_df_for_mapping, grid_ids, full_grid_df, sensitivity_scale,
                                exclude_self_recruitment=exclude_lr)

                        if modified_network:
                            comparison = compare_models(original_network, modified_network)

                            # Calculate connectivity metrics for both models for the comparison table
                            print("\nCalculating comparison metrics...")
                            original_type1 = calculate_connectivity_metrics(
                                selected_processed, release_grids, settlement_grids, traits, _traj_raw,
                                print_summary=False
                            )
                            modified_type1 = calculate_connectivity_metrics(
                                modified_processed, release_grids, settlement_grids, modified_traits, _traj_raw,
                                print_summary=False
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

                            _safe_scn = (scenario_name.replace(' ', '_')
                                         .replace('/', '-').replace(',', ''))

                            if get_yes_no("\nSave Type 3 results to CSV?", default='yes'):
                                _t3_root = get_valid_dir(
                                    "  Output directory",
                                    default=globals().get('project_dirs', {}).get('results', '.'))
                                _t3_dir = os.path.join(_t3_root, 'community')
                                os.makedirs(_t3_dir, exist_ok=True)
                                _t3_path = os.path.join(_t3_dir, f"community_trait_sensitivity_{_safe_scn}.csv")
                                build_trait_sensitivity_table(results['type3_sensitivity']).to_csv(_t3_path, index=False)
                                print(f"  Type 3 results saved to: {_t3_path}")

                            if get_yes_no("\nCreate comparison map?", default='yes'):
                                _fig_dir = globals().get('project_dirs', {}).get('figures', '.')
                                _safe_scn = (scenario_name.replace(' ', '_')
                                             .replace('/', '-').replace(',', ''))
                                create_comparison_map(
                                    original_network, modified_network,
                                    title1='Original Traits',
                                    title2=scenario_name,
                                    output_path=os.path.join(_fig_dir, f"trait_sensitivity_{_safe_scn}"),
                                    output_format='tif', dpi=300,
                                )

                        print("\n Type 3 sensitivity analysis complete")

                    except Exception as e:
                        print(f"\nError in trait sensitivity: {e}")
                        traceback.print_exc()

        except Exception as e:
            print(f"\nError in Type 3 analysis: {e}")
            print("Continuing to the final summary.")
    # Final Summary
    section_header("Analyses Complete")

    print("\nCompleted analyses:")
    if 'type1' in results:
        print("  Type 1: Connectivity Metrics")
    if 'type2' in results:
        print("  Type 2: Network Analysis")
    if 'type3' in results or 'type3_sensitivity' in results:
        print("  Type 3: Trait Sensitivity Analysis")

    # Save metadata file for reproducibility (xlsx with outputs reference + run config)
    try:
        default_dir = get_default_dir('config')
        metadata_path = os.path.join(default_dir, 'comcon_outputs.xlsx')

        # Gather extra user-chosen settings for Run Configuration from the saved config JSONs
        # (written earlier in the workflow) plus which analyses were run.
        _extra_meta = {}
        try:
            def _load_cfg(_name):
                _p = os.path.join(default_dir, _name)
                if os.path.exists(_p):
                    with open(_p) as _f:
                        return json.load(_f)
                return {}
            _pset = _load_cfg('pset_config.json')
            _hyd = _load_cfg('hydrodynamic_parameters.json')
            _lonr = _hyd.get('longitude_range') or [None, None]
            _latr = _hyd.get('latitude_range') or [None, None]
            _timr = _hyd.get('time_range') or [None, None]
            _hydro_src = {'1': 'Copernicus download', '2': 'local file'}.get(
                str(_hyd.get('hydro_source')), _hyd.get('hydro_source'))
            _extra_meta = {
                'domain': {
                    'min_lat': _latr[0], 'max_lat': _latr[1],
                    'min_lon': _lonr[0], 'max_lon': _lonr[1],
                    'date_range': (f"{_timr[0]} to {_timr[1]}" if _timr[0] else None),
                    'hydro_source': _hydro_src,
                },
                'simulation_parameters': {
                    'horizontal_diffusivity_kh': _pset.get('kh'),
                    'particles_per_release_site': _pset.get('npart'),
                    'grid_resolution_km': _hyd.get('selected_resolution_km'),
                    'grid_binned': _hyd.get('needs_binning'),
                },
                'analysis': {
                    'run_type1': 'type1' in results,
                    'run_type2': 'type2' in results,
                    'run_type3': ('type3' in results) or ('type3_sensitivity' in results),
                    'type2_scales': (list(results.get('type2', {}).keys()) if 'type2' in results else None),
                },
            }
        except Exception:
            _extra_meta = {}

        metadata_dict = trait_metadata(
            spp_params, traits,
            globals().get('hydro_params', {}),
            globals().get('runtime', None),
            globals().get('adjusted_runtime', None),
            globals().get('interval', None),
            release_grids, settlement_grids,
            type3=results.get('type3') or results.get('type3_sensitivity'),
            extra=_extra_meta,
        )
        # Resolve project_dirs , set up by setup_project_directory(), may be a global
        _proj_dirs = globals().get('project_dirs', None)
        if _proj_dirs is None:
            # Fallback: build a minimal project_dirs from get_default_dir() for each known role
            _proj_dirs = {
                role: get_default_dir(role)
                for role in ('config', 'data', 'occurrence', 'grids', 'simulation', 'figures', 'results')
            }
        _meta_ok = write_project_metadata(_proj_dirs, metadata_dict, 'trait', metadata_path)
        if not _meta_ok:
            print("  Metadata file was not written. It may be open in another program (close it and re-run), or openpyxl may not be installed (pip install openpyxl).")
    except Exception as e:
        print(f"  Note: Could not save metadata file: {e}")

    # End-of-run summary (mirrors automatic/passive): analyses run, output folders, metadata path.
    try:
        _completed = []
        if 'type1' in results:
            _completed.append("Type 1: Connectivity Metrics")
        if 'type2' in results:
            _completed.append("Type 2: Network Analysis")
        if ('type3' in results) or ('type3_sensitivity' in results):
            _completed.append("Type 3: Trait Sensitivity")
        section_header("Workflow Complete")
        _lines = ["Run summary:"]
        try:
            _lines.append(f"  Species analysed:    {len(traits)}")
            _lines.append(f"  Total release sites: {sum(len(v) for v in release_grids.values())}")
        except Exception:
            pass
        _lines.append(f"  Analyses completed:  {', '.join(_completed) if _completed else 'none'}")
        _pd = globals().get('project_dirs', {}) or {}
        _lines += [
            "",
            "Output files:",
            f"  Project folder:  {_pd.get('root', get_default_dir('config'))}",
            f"  Simulation data: {_pd.get('simulation', get_default_dir('simulation'))}",
            f"  Results:         {_pd.get('results', get_default_dir('results'))}",
            f"  Figures:         {_pd.get('figures', get_default_dir('figures'))}",
            f"  Metadata:        {os.path.join(get_default_dir('config'), 'comcon_outputs.xlsx')}",
        ]
        info_box(_lines)
    except Exception:
        pass

    return results

# endregion
# endregion

# region Complete Interactive Workflow
## Run a single numbered step of the interactive trait-defined pathway workflow
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
    global coastal_grid, coastal_grid_df, coastal_grid_filtered, geojson_file_path
    global ocean_grid, ocean_grid_df, grid_df_for_mapping
    global fieldset, adjusted_runtime, interval, pset
    global kernels
    global output_path, traj, chunks_dir
    global final_processed, buffer
    global analysis_results
    global _selected_resolution_km, _needs_binning

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

        # Project setup first, so the user sets up where outputs go before choosing any data.
        if 'project_dirs' not in globals() or project_dirs is None:
            subsection_header("Project Setup")
            info_box([
                "ComCon organises all outputs into a project folder.",
                "Each project gets its own directory with subfolders for configuration, data, figures, simulation output and results.",
                "",
                "This keeps your work organised and prevents files being overwritten between different analyses.",
            ])
            project_dirs = setup_project_directory(mode_tag='int')
        else:
            subsection_header("Project Directory")
            print(f"\n  Using existing project directory: {project_dirs['root']}\n")
            display_line("─")

        subsection_header("Setting up the Hydrodynamic Extent")
        info_box([
            "The hydrodynamic extent needs to be larger than your study extent.",
            "",
            "The hydrodynamic extent is the area in which particles will be dispersing during the simulation.",
            "",
            "The study extent is the area in which you are interested in assessing connectivity (the release and settlement sites of your species, not the area in which they disperse).",
            "",
            "You will set up the hydrodynamic extent here in Step 1 and the study extent in Step 2.",
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
            "For more information, see 'Part 5: Step 1' and 'Part 7: Species Traits CSV Format' in the User Guide.",
        ])

        display_line("═")

        # Offer choice: download new or load existing
        section_header("Select your Hydrodynamic Data")

        info_box([
            "Choose how to load your hydrodynamic data:",
            "",
            "  1. Download from Copernicus Marine Service (requires internet)",
            "     - ComCon streams the data directly and there is no large download needed",
            "     - You can optionally save a local copy for future offline use",
            "",
            "  2. Load from a local file (no internet required)",
            "     - Use a previously saved NetCDF (.nc) or Zarr (.zarr) file",
            "     - The file must contain 'uo' and 'vo' velocity variables with time, latitude and longitude dimensions",
            "     - See 'Part 7: Hydrodynamic Data File Format' in the User Guide",
        ])

        # Use the saved data-source choice as the default if this project already has parameters, so the user is not re-asked from scratch when continuing.
        _saved_hydro_source = None
        _hp_path = os.path.join(project_dirs['config'], "hydrodynamic_parameters.json")
        if os.path.exists(_hp_path):
            try:
                with open(_hp_path, 'r') as _f:
                    _saved_hydro_source = json.load(_f).get('hydro_source')
            except Exception:
                _saved_hydro_source = None
        if _saved_hydro_source in ['1', '2']:
            # This project already saved a data-source choice, so reuse it silently instead of
            # re-asking. (To change it, delete or edit hydrodynamic_parameters.json in the config folder.)
            hydro_source = _saved_hydro_source
            _src_label = ('Download from Copernicus Marine Service' if hydro_source == '1'
                          else 'Load from a local file')
            print(f"\nUsing your saved data source: {_src_label}.")
        else:
            while True:
                hydro_source = input(
                    "\nSelect data source (1/2) [default: 1]: ").strip() or '1'
                if hydro_source in ['1', '2']:
                    break
                print("Please enter 1 or 2.")

        if hydro_source == '2':
            # Load from local file
            subsection_header("Load Local Hydrodynamic Data")

            while True:
                local_path = get_valid_path("Enter the path to your hydrodynamic data file (.nc or .zarr)",
                                            must_exist=True, extensions=['.nc', '.zarr'])
                try:
                    hydro_data = load_hydrodynamic_data(local_path)
                    break
                except (FileNotFoundError, ValueError) as e:
                    print(f"\n  Error: {e}")
                    retry = input("  Try again? (yes/no) [default: yes]: ").strip().lower() or 'yes'
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
            subsection_header("Copernicus Marine Service Data")
            info_box([
                "ComCon downloads hydrodynamic data from the Copernicus Marine Service.",
                "",
                "Set up a free account, browse to the data you want, click 'Data access' and copy the dataset ID.",
                "",
                "For example: cmems_mod_ibi_phy_anfc_0.027deg-2D_PT1H-m",
                "",
                "Website: https://data.marine.copernicus.eu/products",
                "",
                "You will need a stable internet connection, your username and password, the dataset ID, and the longitude and latitude ranges of your study extent.",
            ])
            hydro_params, species_traits_path, runtime, traits = params()

            # Extract buffer values to globals (needed for Type 3 analysis)
            buffer_before = hydro_params.get('buffer_before', 0)
            buffer_after = hydro_params.get('buffer_after', 0)

            # Pre-download coverage check: confirm the dataset covers the dispersal period before streaming anything.
            _cov_ok, _avail_start, _avail_end = check_dataset_coverage(
                hydro_params['dataset_id'], hydro_params['time_range'],
                username=hydro_params.get('username'), password=hydro_params.get('password'),
                longitude_range=hydro_params.get('longitude_range'),
                latitude_range=hydro_params.get('latitude_range'),
            )
            # Only surface this pre-download catalogue check when there is a problem.
            # When coverage is fine, the post-download "Data Coverage Check" below reports
            # the same thing against the actual streamed data, so printing both is redundant.
            if not _cov_ok:
                subsection_header("Dataset Coverage Check")
                print(f"Dataset available:  {_avail_start.date()} to {_avail_end.date()}")
                print(f"Dispersal period:  {pd.Timestamp(hydro_params['time_range'][0]).date()} to {pd.Timestamp(hydro_params['time_range'][1]).date()}")
                print("This dataset does not fully cover your dispersal period.")
                print("Pick a different year, choose another dataset or adjust your spawning window, then re-run Step 1.")
                _go = input("Continue with this dataset anyway? (yes/no) [default: no]: ").strip().lower()
                if _go not in ['yes', 'y']:
                    return

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
            print(f"Hydrodynamic data coverage is suitable")

        # Grid resolution selection and domain visualisation runs regardless of the coverage
        # result above. A coverage warning must not skip the resolution choice or Step 1 completion.
        try:
            _native_lm, _native_cm, _native_lon, _native_lat = create_masks(
                hydro_data, geojson_file_path=None, show_map=False
            )

            # 1. Grid resolution , reuse the value saved for this project on a rerun,
            # otherwise let the user choose (the choice is saved at the end of Step 1).
            _saved_res = _saved_bin = None
            if os.path.exists(_hp_path):
                try:
                    with open(_hp_path, 'r') as _f:
                        _hp_prev = json.load(_f)
                    _saved_res = _hp_prev.get('selected_resolution_km')
                    _saved_bin = _hp_prev.get('needs_binning')
                except Exception:
                    _saved_res = _saved_bin = None
            if _saved_res is not None and _saved_bin is not None and get_yes_no(
                    f"Reuse the saved grid resolution ({_saved_res} km)?", default='yes'):
                _selected_resolution_km, _needs_binning = _saved_res, bool(_saved_bin)
                print(f"\n  Using the saved grid resolution: {_selected_resolution_km} km")
            else:
                _selected_resolution_km, _needs_binning = choose_grid_resolution(
                    _native_lon, _native_lat
                )

            # 2. Bin if needed
            _n_native_ocean = int((_native_lm == 0).sum())
            _n_native_coastal = int((_native_cm == 1).sum())
            if _needs_binning:
                subsection_header("Binning grid to target resolution")
                _b_lm, _b_lon, _b_lat = bin_mask(
                    _native_lm, _native_lon, _native_lat, _selected_resolution_km
                )
                _b_cm = make_coastalmask(_b_lm)
                _n_binned_ocean = int((_b_lm == 0).sum())
                _n_binned_coastal = int((_b_cm == 1).sum())
                data_box("Grid summary", [
                    f"Native grid:          {_native_lm.shape[0]} x {_native_lm.shape[1]} cells",
                    f"Native ocean cells:   {_n_native_ocean:,}",
                    f"Native coastal cells: {_n_native_coastal:,}",
                    f"Binned grid:          {_b_lm.shape[0]} x {_b_lm.shape[1]} cells",
                    f"Binned ocean cells:   {_n_binned_ocean:,}",
                    f"Binned coastal cells: {_n_binned_coastal:,}",
                ])
            else:
                data_box("Grid summary", [
                    f"Native grid:          {_native_lm.shape[0]} x {_native_lm.shape[1]} cells",
                    f"Native ocean cells:   {_n_native_ocean:,}",
                    f"Native coastal cells: {_n_native_coastal:,}",
                ])
            # 3. Visualise , one map if native, two if binned (with coastal layer)
            # Capture each figure, save it to the grids folder (not figures), then close, mirroring the passive pathway so both maps render reliably and are exported.
            subsection_header("Visualising the Hydrodynamic Extent")
            _loading_out = widgets.Output()
            display(_loading_out)
            with _loading_out:
                print("  Loading map... this may take a moment.")
            _loading_out.clear_output()

            _grids_dir = project_dirs.get('grids', '.')
            _fig_native = plot_masks(_native_lm, _native_cm, _native_lon, _native_lat,
                                     geojson_file_path=None, show_coastal=True,
                                     show_save_prompt=False,
                                     title=f'Native resolution ({_native_lm.shape[0]} x {_native_lm.shape[1]} cells)')
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    save_figure(_fig_native, os.path.join(_grids_dir, 'landmask_native'))
                except Exception:
                    pass
            plt.close(_fig_native)

            if _needs_binning:
                _fig_binned = plot_masks(_b_lm, _b_cm, _b_lon, _b_lat,
                                         geojson_file_path=None, show_coastal=True,
                                         show_save_prompt=False,
                                         title=f'Binned resolution ({_b_lm.shape[0]} x {_b_lm.shape[1]} cells)')
                with contextlib.redirect_stdout(io.StringIO()):
                    try:
                        save_figure(_fig_binned, os.path.join(_grids_dir, 'landmask_binned'))
                    except Exception:
                        pass
                plt.close(_fig_binned)

        except Exception as e:
            print(f"  Note: Could not display hydrodynamic extent ({e}).")
            _selected_resolution_km, _needs_binning = None, None

        completion_message(1)
        print("\nOutputs: Species traits, hydrodynamic parameters, dispersal period")

        # Capture the data-source and grid-resolution choices in the parameter file
        try:
            _hp_path = os.path.join(project_dirs['config'], "hydrodynamic_parameters.json")
            _hp_saved = {}
            if os.path.exists(_hp_path):
                with open(_hp_path, 'r') as _f:
                    _hp_saved = json.load(_f)
            _hp_saved['hydro_source'] = hydro_source
            _hp_saved['selected_resolution_km'] = _selected_resolution_km
            _hp_saved['needs_binning'] = _needs_binning
            with open(_hp_path, 'w') as _f:
                json.dump(_hp_saved, _f, indent=4)
        except Exception as _e:
            print(f"  Note: Could not update the parameter file ({_e}).")

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

        print("\nIn this step, you will select your study extent and choose your data source.")

        print("\nThis package has the option to use species-specific occurrence data to determine appropriate release and settlement areas for each species.\n")

        print("In step 3, you then have the option to:")
        print("  - Release propagules from all locations in your study extent, specific site or known occurrences")
        print("  - Settle propagules from all locations in your study extent, specific sites or known occurrences")

        subsection_header("Data source options")
        info_box([
            "Option 1: Download occurrence data from GBIF (Global Biodiversity Information Facility)",
            "  - Automatically determines suitable sites based on species distributions",
            "  - Uses the species identifiers (AphiaID or scientific name) from your trait dataset",
            "  - Same data will be used for both release and settlement",
            "  - Requires a stable internet connection",
            "  - GBIF website: https://www.gbif.org/",
            "  - For more information, see 'Part 5: Step 2 (Species Occurrence Data)' in the User Guide",
            "",
            "Option 2: Provide your own species occurrence data",
            "  - This can be species occurrence data or habitat suitability data",
            "  - Must be in CSV format with columns: species, latitude, longitude",
            "  - Species names must match those in your trait dataset",
            "  - You can provide different datasets for release and settlement in step 3",
            "  - See 'Part 7: Occurrence Data CSV Format' in the User Guide for format details"
        ])

        # Call spp_occ to gather species occurrence data
        section_header("Select your Species Occurrence Data")

        # Re-load hydro_data if session was saved without it (backward compatibility)
        if 'hydro_data' not in globals() or hydro_data is None:
            if 'hydro_params' in globals() and hydro_params is not None:
                _hydro_path = hydro_params.get('output_path', '')
                if _hydro_path and os.path.exists(_hydro_path):
                    print(f"  Re-loading hydrodynamic data from: {_hydro_path}")
                    hydro_data = xr.open_dataset(_hydro_path)
                else:
                    print("  Error: Cannot find hydrodynamic data file. Please re-run Step 1.")
                    return
            else:
                print("  Error: No hydrodynamic parameters found. Please re-run Step 1.")
                return

        clean_spp, spp_params, landmask, coastalmask, lonarray, latarray = spp_occ(
            hydro_data, species_traits_path
        )

        completion_message(2)
        print("\nOutputs: Cleaned species occurrences, land/coastal masks, study extent polygon")
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

        # Grid resolution selection , use Step 1 choice if available, otherwise ask now
        if '_selected_resolution_km' in dir() or (
                '_selected_resolution_km' in globals() and _selected_resolution_km is not None):
            resolution_km = _selected_resolution_km
            needs_binning = _needs_binning
        else:
            resolution_km, needs_binning = choose_grid_resolution(lonarray, latarray)

        if needs_binning:
            print(f"  Binning grid to {resolution_km} km resolution...")
            binned_landmask, binned_lonarray, binned_latarray = bin_mask(
                landmask, lonarray, latarray, target_km=resolution_km
            )
            binned_coastalmask = make_coastalmask(binned_landmask)

            n_native_coastal = int((make_coastalmask(landmask) == 1).sum())
            n_binned_coastal = int((binned_coastalmask == 1).sum())
            print(f"  Native coastal cells: {n_native_coastal:,}")
            print(f"  Binned coastal cells: {n_binned_coastal:,} "
                  f"({100 * n_binned_coastal / max(n_native_coastal, 1):.0f}% of native)\n")

            # Use binned arrays for release/settlement grid creation
            grid_lonarray = binned_lonarray
            grid_latarray = binned_latarray
            grid_landmask = binned_landmask
            grid_coastalmask = binned_coastalmask
        else:
            grid_lonarray = lonarray
            grid_latarray = latarray
            grid_landmask = landmask
            grid_coastalmask = coastalmask

        subsection_header("Release and Settlement Strategy")
        print("This package has several options for releasing and settling propagules.\n")

        print("You have the option to release propagules from all sites in your study extent or from known species-specific distributions.\n")

        print("For more information, see 'Part 5: Step 3 (Simulation Setup)' in the User Guide.\n")

        # Set up release and settlement grids (using binned arrays if applicable)
        meshgrid, release_grids, settlement_grids, full_grid_df, grid_ids, coastal_grid, coastal_grid_df, coastal_grid_filtered, grid_df_for_mapping = setup_release_settlement(
            grid_coastalmask, grid_landmask, clean_spp, spp_params, grid_lonarray, grid_latarray, traits=traits
        )

        # Create Fieldset from hydrodynamic data
        # The fieldset uses NATIVE-resolution landmask for anti-beaching vectors.
        # grid_ids must be mapped to native resolution so particles can sample it.
        if needs_binning:
            # Map binned grid_ids to native resolution for the fieldset
            native_grid_ids = create_native_grid_ids_from_binned(
                lonarray, latarray, landmask,
                grid_coastalmask, grid_lonarray, grid_latarray,
                coastal_grid_df,
            )
            fieldset = create_fieldset(hydro_data, landmask, native_grid_ids)
        else:
            fieldset = create_fieldset(hydro_data, landmask, grid_ids)

        print("Release and settlement grid setup complete.\n")

        subsection_header("ParticleSet Configuration")
        print("The next step is to create a ParticleSet for your species.\n")

        print("A ParticleSet is simply a collection of 'rules' or 'behaviours' applied to all particles within your simulation.\n")

        print("This package allows you to choose how many particles to release per site, the release frequency and horizontal diffusivity.\n")

        print("For more information, see 'Part 5: Step 3 (Particle Parameters)' in the User Guide.\n")

        # Create ParticleSet
        pset, adjusted_runtime, interval, repeatdt = create_pset(fieldset, meshgrid, runtime)

        # Check if user cancelled
        if pset is None:
            subsection_header("Step 3 cancelled by user.")
            return  # Exit gracefully without error

        # Create custom kernels
        kernels = create_custom_kernels(pset)

        completion_message(3)
        print("\nOutputs: Release/settlement grids, grid IDs, fieldset, ParticleSet")
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

        # pset/kernels contain compiled C code , rebuild from saved config if missing
        if 'pset' not in globals() or pset is None or 'kernels' not in globals() or kernels is None:
            config_path = os.path.join(project_dirs['config'],
                                       'pset_config.json') if 'project_dirs' in globals() and project_dirs else None
            if config_path and os.path.exists(config_path):
                print("Rebuilding ParticleSet and kernels from saved configuration...")
                try:
                    # Re-load hydro_data if needed
                    if 'hydro_data' not in globals() or hydro_data is None:
                        _hydro_path = hydro_params.get('output_path', '')
                        if _hydro_path and os.path.exists(_hydro_path):
                            hydro_data = xr.open_dataset(_hydro_path)
                        else:
                            print("Error: Cannot find hydrodynamic data. Please re-run from Step 1.")
                            return

                    # Rebuild fieldset
                    if '_needs_binning' in globals() and _needs_binning:
                        native_grid_ids = create_native_grid_ids_from_binned(
                            lonarray, latarray, landmask,
                            make_coastalmask(landmask), lonarray, latarray,
                            coastal_grid_df,
                        )
                        fieldset = create_fieldset(hydro_data, landmask, native_grid_ids)
                    else:
                        fieldset = create_fieldset(hydro_data, landmask, grid_ids)

                    # Load saved pset config
                    with open(config_path) as f:
                        pset_cfg = json.load(f)

                    _interval = timedelta(days=pset_cfg['release_interval_days'])
                    _npart = pset_cfg['npart']
                    _kh = pset_cfg['kh']

                    # Rebuild pset
                    lats = meshgrid[:, 1].flatten()
                    lons = meshgrid[:, 0].flatten()
                    runtime_td = timedelta(days=runtime)
                    num_intervals = math.ceil(runtime_td / _interval)
                    _adjusted_runtime = num_intervals * _interval

                    lat = list(itertools.chain.from_iterable(itertools.repeat(x, _npart) for x in lats))
                    lon = list(itertools.chain.from_iterable(itertools.repeat(x, _npart) for x in lons))

                    fieldset.add_constant_field("Kh_zonal", _kh, mesh="spherical")
                    fieldset.add_constant_field("Kh_meridional", _kh, mesh="spherical")

                    extra_vars = [
                        Variable("distance", initial=0.0, dtype=np.float32),
                        Variable("prev_lon", dtype=np.float32, to_write=False, initial=attrgetter("lon")),
                        Variable("prev_lat", dtype=np.float32, to_write=False, initial=attrgetter("lat")),
                        Variable('on_land', dtype=np.int32, initial=0),
                        Variable('grid_id', dtype=np.int32, initial=0)
                    ]
                    Particle = JITParticle.add_variables(extra_vars)
                    pset = ParticleSet(fieldset=fieldset, pclass=Particle, lon=lon, lat=lat, repeatdt=_interval)
                    adjusted_runtime = _adjusted_runtime.days
                    interval = _interval.days

                    kernels = create_custom_kernels(pset)
                    print("\nRebuild complete.")
                except Exception as e:
                    print(f"Error rebuilding: {e}")
                    print("Please re-run: workflow_step(3)")
                    return
            else:
                print("The ParticleSet and kernels from Step 3 are not available.")
                print("These contain compiled simulation code that cannot be saved between sessions.")
                print("Please re-run: workflow_step(3)")
                return

        subsection_header("Running the Particle Simulation")
        print("The simulation will now run using the parameters you have configured.\n")

        print("For more information, see 'Part 5: Step 4 (Execute the Simulation)' in the User Guide.")

        # Execute simulation
        output_path, chunks_dir = execute_simulation(pset, kernels, adjusted_runtime, interval)

        traj = None  # Trajectory data stays on disk as parquet chunks for memory efficiency

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

        print("For more information, see 'Part 5: Step 5 (Post-Processing)' in the User Guide.\n")

        # Check that trajectory data exists (chunks on disk or traj DataFrame in memory)
        has_chunks = 'chunks_dir' in globals() and chunks_dir and os.path.isdir(str(chunks_dir))
        has_traj = 'traj' in globals() and isinstance(traj, pd.DataFrame) and not traj.empty

        if not has_chunks and not has_traj:
            print("Error: No trajectory data found.")
            print("This usually means Step 4 did not complete successfully.")
            print("\nRecovery options:")
            if 'output_path' in globals() and output_path and os.path.exists(str(output_path)):
                print(f"  1. A .zarr file exists at: {output_path}")
                print("     You can reload it by re-running Step 4, or manually with:")
                print(f"     chunks_dir = process_sim_results('{output_path}', chunk_size=5000, runtime_days={adjusted_runtime if 'adjusted_runtime' in dir() else 'YOUR_RUNTIME'})")
                print(f"     chunks_dir = assign_release_id(chunks_dir, interval)")
                print("     workflow_step(5)")
            else:
                zarr_found = []
                if 'project_dirs' in globals() and project_dirs:
                    sim_dir = project_dirs.get('simulation', '')
                    if os.path.isdir(sim_dir):
                        zarr_found = [f for f in os.listdir(sim_dir) if f.endswith('.zarr')]
                if zarr_found:
                    print(f"  Found .zarr file(s) in your simulation folder: {zarr_found}")
                    zarr_path = os.path.join(project_dirs['simulation'], zarr_found[0])
                    print(f"  Reload with:")
                    print(f"     chunks_dir = process_sim_results('{zarr_path}', chunk_size=5000, runtime_days=YOUR_RUNTIME)")
                    print(f"     chunks_dir = assign_release_id(chunks_dir, interval)")
                    print("     workflow_step(5)")
                else:
                    print("  No .zarr file found. You will need to re-run Step 4.")
            return

        # Run post-processing (uses chunks from disk if available, else traj DataFrame)
        traj_input = chunks_dir if has_chunks else traj
        final_processed, buffer = post_process(traj_input, traits, release_grids, settlement_grids, grid_ids)

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
        print("  - Self-recruitment")
        print("  - Dispersal distance")
        print("  - Realized PLD")

        subsection_header("Type 2: Network Analysis")
        print("  - Functional roles (source, sink, self-recruitment)")
        print("  - Centrality measures (betweenness, closeness, degree")
        print("  - Composite hotspot rankings, network maps and connectivity matrices")

        subsection_header("Type 3: Trait Sensitivity Analysis")
        print("  - Compare a baseline trait scenario against a modified scenario")
        print("  - Modify PLD range or spawning window to test sensitivity\n")

        print("For more information, see 'Part 5: Step 6 (Connectivity Analyses)' in the User Guide.\n")

        # Run the main analysis function
        analysis_results = post_analysis(final_processed)

        section_header("Interactive workflow completed!")
# endregion
# endregion
