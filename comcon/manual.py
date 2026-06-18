# region Manual Mode Functions
"""
ComCon Manual Mode
==================

This module provides access to the individual workflow functions of the trait-defined pathway, for building custom connectivity modelling pipelines.
Users can run specific sections of the workflow independently, which is useful for:

- Re-running analyses on existing data
- Testing different parameters without full workflow reruns
- Integrating ComCon functions into custom scripts
- Debugging specific workflow sections

Note:
    Manual mode applies only to the trait-defined pathway.
    The passive dispersal pathway (Path A) runs end-to-end via a single entry point (`passive_dispersal_pathway()` in `comcon.passive`) and is not split into manually callable sections.

Usage:
------
    from comcon.manual import *

    # See all available functions:
    list_functions()

    # See example workflows:
    example_workflow()

    # Run specific workflow sections:
    # (see documentation for each function)

Key Functions by Workflow Section:
----------------------------------

STEP 1 - Hydrodynamic Data:
    run_step1_hydro()           - Complete Step 1 with prompts
    download_hydrodynamic_data()- Download data (requires params)

STEP 2 - Species Occurrence:
    run_step2_species_data()    - Complete Step 2 with prompts
    get_spp_occ()               - Download species from GBIF from a provided species list
    clean_spp_occ()             - Filter occurrence data

STEP 3 - Grid Setup:
    run_step3_grids()           - Complete Step 3 with prompts
    map_spp_to_grid()           - Map species to model grid
    create_fieldset()           - Create OceanParcels fieldset

STEP 4 - Simulation:
    run_step4_simulation()      - Execute simulation with prompts

STEP 5 - Post-Processing:
    run_step5_postprocessing()  - Post-process with prompts
    spawning_filter()           - Filter by spawning window
    pld_filter()                - Filter by PLD
    habitat_filter()            - Filter by habitat

STEP 6 - Analysis:
    run_step6_analysis()        - Run analyses with prompts
    calculate_connectivity_metrics() - Type 1 analysis
    network_analysis()          - Type 2 analysis
"""

# region Imports
import os
import json
import warnings
import pandas as pd
import numpy as np
import builtins
import pickle
from IPython.display import display, HTML

warnings.filterwarnings('ignore')
import logging
logging.getLogger("urllib3").setLevel(logging.ERROR)

# Standard formatting (matching interactive.py)
from comcon.interactive import (
    display_line, section_header, subsection_header,
    info_box, data_box, status_message, code_box,
    centered_text, completion_message
)
# endregion

# region Import Functions from Interactive Module
# These are imported when the module loads
# Users can access them directly

from comcon.interactive import (
    # Constants and Formatting
    map_config,

    # Helper Functions
    get_positive_integer,
    standardise_species_name,
    display_species_name,
    get_output_format,
    save_figure,
    get_output_choice,
    calculate_figure_dimensions,

    # Step 1: Hydrodynamic Data Functions
    # Wrapper functions (have input() - use for interactive work)
    params,  # Collect all hydro params interactively
    load_spp_traits,  # Load traits with prompts
    collect_hydro_params,  # Collect hydro params only
    setup_project_directory,  # Create/use project directory
    get_default_dir,  # Get project-aware default path

    # Direct functions (no input() - use in scripts)
    get_package_data_path,
    load_builtin_traits,
    select_species_from_database,
    calculate_overall_dispersal_period,
    download_hydrodynamic_data,  # Takes params, downloads data
    load_hydrodynamic_data,  # Load from local .nc or .zarr file

    # Step 2: Species Occurrence Data Functions
    # Wrapper function (has input() - use for interactive work)
    spp_occ,  # Complete species occurrence workflow
    collect_spp_occ_params,  # Collect occurrence params
    edit_spp_params,  # Edit existing params

    # Direct functions (no input() - use in scripts)
    create_map,  # Create interactive map (displays widget)
    read_geojson,  # Read GeoJSON file
    get_aphia,  # Get species name from AphiaID
    get_spp_occ,  # Download species from GBIF from a list
    get_all_spp_occ,  # Download all species in area
    make_coastalmask,  # Create coastal mask
    create_masks,  # Create land/coastal masks (displays plot)
    clean_spp_occ,  # Filter occurrence data
    validate_spp_occ,  # Validate sample sizes
    get_spp_colours,  # Get colours for species
    plot_spp_occ,  # Plot species occurrences
    validate_uploaded_data,  # Validate uploaded data

    # Step 3: Grid Setup Functions
    # Wrapper function (has input() - use for interactive work)
    setup_release_settlement,  # Complete grid setup with prompts

    # Direct functions (no input() - use in scripts)
    map_spp_to_grid,  # Map species to grid
    filter_by_polygon,  # Filter grid by polygon
    save_grids,  # Save grids (has input for paths)
    reflective_boundary,  # Create boundary vectors
    create_fieldset,  # Create OceanParcels fieldset

    # Step 4: Simulation Functions
    # Wrapper functions (have input() - use for interactive work)
    create_pset,  # Create ParticleSet with prompts
    execute_simulation,  # Run simulation with prompts

    # Direct functions (no input() - use in scripts)
    create_custom_kernels,  # Create particle kernels
    MemoryMonitor,  # Memory monitoring context manager
    check_system_memory,  # Check available RAM
    get_memory_config,  # Get memory configuration
    get_zarr_size,  # Get zarr file size
    process_sim_results,  # Load simulation output
    assign_release_id,  # Assign cohort IDs
    load_traj_from_chunks,  # Load full trajectory DataFrame from chunks
    load_traj_starts_from_chunks,  # Load only release rows (obs=0) for mortality
    save_session,  # Save workflow state for resuming later
    load_session,  # Load saved workflow state

    # Step 5: Post-Processing Functions
    # Wrapper function (has input() - for buffer size)
    post_process,  # Complete post-processing

    # Direct functions (no input() - use in scripts)
    spawning_filter,  # Filter by spawning window
    pld_filter,  # Filter by PLD
    habitat_filter,  # Filter by habitat

    # Step 6: Analysis Functions
    # Wrapper function (has input() - use for interactive analysis selection)
    post_analysis,  # Interactive analysis selection

    # Direct functions (no input() - use in scripts)
    calculate_connectivity_metrics,  # Type 1: Connectivity metrics
    display_metrics_table,  # Display Type 1 results
    network_analysis,  # Type 2: Network analysis
    add_custom_composite,  # build a user-defined composite from chosen metrics
    display_network_summary,  # Display Type 2 results
    compare_models,  # Compare baseline vs modified-trait scenarios (Type 3)
    display_sensitivity_comparison,  # Display Type 3 sensitivity results
    export_connectivity_metrics,  # Export Type 1 metrics to CSV
    export_network_results,  # Export Type 2 network results to CSV

    # Visualisation Functions
    plot_masks,  # Visualise masks
    create_network_map,  # Network connectivity map
    create_connectivity_matrix_plot,  # Matrix heatmap
    create_comparison_map,  # Side-by-side comparison

    # Interactive result viewers (dropdown widgets — use in Jupyter Lab)
    trait_metrics_viewer,  # Type 1 metrics, per species (dropdown)
    network_metrics_viewer,  # Type 2 summary stats (dropdown)
    trait_matrix_viewer,  # Type 2 connectivity matrix (dropdown)
    trait_metric_viewer,  # Type 2 per-metric hotspot map (dropdown)
    trait_heatmap_viewer,  # Type 2 site x metric heatmap (dropdown)
    trait_community_viewer,  # Type 2 Louvain community map (dropdown)
    build_heatmap_figure,  # Type 2 site x metric heatmap (static figure)
    create_community_map,  # Type 2 Louvain community map (static figure)
    create_matrix_with_map,  # Type 2 connectivity matrix + map (static figure)
)
# endregion

# region Documentation Functions
## Print a categorised list of all available manual-mode functions
def list_functions():
    """
    Print a categorized list of all available functions with brief descriptions.
    """
    section_header("Manual Mode: Available Functions")

    print("""Functions are organized by workflow step. Each function is marked as:
  [I] = Interactive (has input() prompts and needs to be used in Jupyter Lab)
  [D] = Direct (no prompts and can use in scripts)
""")

    categories = {
        "Step 1: Hydrodynamic and Trait Data": [
            ("[I] params", "Collect all hydro params interactively"),
            ("[I] load_spp_traits", "Load species traits with prompts"),
            ("[I] collect_hydro_params", "Collect hydrodynamic parameters"),
            ("[I] setup_project_directory", "Create or use a project directory"),
            ("[D] load_builtin_traits", "Load built-in trait database"),
            ("[D] download_hydrodynamic_data", "Download NEMO data from Copernicus (needs internet)"),
            ("[D] load_hydrodynamic_data", "Load hydro data from local .nc or .zarr file (no internet)"),
            ("[D] calculate_overall_dispersal_period", "Calculate dispersal dates"),
        ],
        "Step 2: Species Occurrence Data": [
            ("[I] spp_occ", "Complete species occurrence workflow"),
            ("[D] read_geojson", "Load study area polygon"),
            ("[D] get_spp_occ", "- Download species from GBIF from a provided species list"),
            ("[D] get_all_spp_occ", "Download all species in study area"),
            ("[D] create_masks", "Create land/coastal masks"),
            ("[D] clean_spp_occ", "Filter occurrences by habitat type"),
            ("[D] validate_spp_occ", "Validate sample sizes"),
        ],
        "Step 3: Release and Settlement Setup": [
            ("[I] setup_release_settlement", "Complete grid setup with prompts"),
            ("[D] map_spp_to_grid", "Map species occurrences to grid"),
            ("[D] filter_by_polygon", "Filter grid to study area"),
            ("[D] create_fieldset", "Create OceanParcels fieldset"),
            ("[D] reflective_boundary", "Create boundary reflection vectors"),
        ],
        "Step 4: Particle Simulation": [
            ("[I] create_pset", "Create ParticleSet with prompts"),
            ("[I] execute_simulation", "Run simulation with prompts"),
            ("[D] create_custom_kernels", "Build particle behaviour kernels"),
            ("[D] process_sim_results", "Load simulation output"),
            ("[D] assign_release_id", "Assign release cohort IDs"),
        ],
        "Step 5: Post-Processing": [
            ("[I] post_process", "Complete post-processing pipeline"),
            ("[D] spawning_filter", "Filter by spawning window"),
            ("[D] pld_filter", "Filter by PLD with settlement check"),
            ("[D] habitat_filter", "Filter by release habitat"),
        ],
        "Step 6: Connectivity Analyses": [
            ("[I] post_analysis", "Interactive analysis selection"),
            ("[D] calculate_connectivity_metrics", "Type 1: connectivity metrics"),
            ("[D] display_metrics_table", "Type 1: display the metrics as a table"),
            ("[D] display_network_summary", "Type 2: display the network summary"),
            ("[D] network_analysis", "Type 2: network analysis"),
            ("[D] run_multiscale_network", "Setup to run Type 2 analysis at several spatial scales"),
            ("[D] add_custom_composite", "Type 2: build a custom composite from chosen metrics"),
            ("[D] compare_models", "Type 3: compare baseline vs modified-trait scenarios"),
            ("[D] display_sensitivity_comparison", "Display Type 3 sensitivity results"),
            ("[D] export_connectivity_metrics", "Export Type 1 metrics to CSV"),
            ("[D] export_network_results", "Export Type 2 network results to CSV"),
        ],
        "Visualisation": [
            ("[D] plot_masks", "Visualise land/coastal classification"),
            ("[D] plot_spp_occ", "Plot species occurrences"),
            ("[D] create_network_map", "Network connectivity map"),
            ("[D] create_connectivity_matrix_plot", "Connectivity matrix"),
            ("[D] build_heatmap_figure", "Type 2: site x metric heatmap (static)"),
            ("[D] create_community_map", "Type 2: Louvain community map (static)"),
            ("[D] create_matrix_with_map", "Type 2: connectivity matrix + map (static)"),
        ],
        "Interactive Result Viewers (dropdown widgets, run in Jupyter)": [
            ("[I] trait_metrics_viewer", "Type 1: browse connectivity metrics per species"),
            ("[I] network_metrics_viewer", "Type 2: browse network summary stats"),
            ("[I] trait_matrix_viewer", "Type 2: browse connectivity matrices"),
            ("[I] trait_metric_viewer", "Type 2: browse per-metric hotspot maps"),
            ("[I] trait_heatmap_viewer", "Type 2: browse the site x metric heatmap"),
            ("[I] trait_community_viewer", "Type 2: browse Louvain community maps"),
        ],
        "Utilities": [
            ("[D] standardise_species_name", "Convert spaces to underscores"),
            ("[D] display_species_name", "Convert underscores to spaces"),
            ("[D] get_positive_integer", "Validated integer input"),
            ("[D] check_system_memory", "Check available RAM"),
        ],
        "Session & Shortcuts": [
            ("[D] save_session", "Save the workflow so you can resume later"),
            ("[D] load_session", "Reload a previously saved workflow"),
            ("[D] run_postprocessing_only", "Re-run post-processing on loaded trajectory data without prompts (scriptable post_processing"),
            ("[D] run_analysis_only", "Run Type 1/2 without no prompts (scriptable post_analysis)"),
        ],
    }

    for category, functions in categories.items():
        print(f"\n  {category}")
        display_line("─")
        for func_info, description in functions:
            print(f"    {func_info:<40} {description}")

    section_header("End of Function List")
    print("For detailed help: help(function_name)")
    print("For example workflows: example_workflow()\n")

## Print example workflows illustrating common manual-mode usage patterns
def example_workflow():
    """
    Print example workflows showing how to use manual mode functions.
    """

    # Main title
    section_header("Manual Mode: Example Workflows")

    print("This guide shows common usage patterns for manual mode functions.")
    print("Each example demonstrates a specific use case.\n")

    # =========================================================================
    # Example 1: Full Workflow Using Wrapper Functions
    # =========================================================================
    subsection_header("Example 1: Full Workflow Using Wrapper Functions")

    info_box([
        "Run the complete ComCon workflow using the main wrapper functions.",
        "Each function handles one major step and includes interactive prompts.",
        "This is the recommended approach for most users."
    ])

    code_box("Import manual mode and set up project directory", [
        "from comcon.manual import *",
        "",
        "# Set up a project directory (organises all outputs into subfolders)",
        "project_dirs = setup_project_directory()",
        "",
        "# Or use your own directory:",
        "# project_dirs = setup_project_directory(custom_path='/path/to/your/folder')"
    ])

    code_box("Step 1: Collect parameters and download hydrodynamic data", [
        "hydro_params, species_traits_path, runtime, traits = params()",
        "hydro_data = download_hydrodynamic_data(**hydro_params)",
        "",
        "# or load from a previously saved local file (no internet needed):",
        "# hydro_data = load_hydrodynamic_data(project_dirs['data'] + '/hydro_data.nc')"
    ])

    code_box("Step 2: Download and clean species occurrence data", [
        "clean_spp, spp_params, landmask, coastalmask, lonarray, latarray = spp_occ(",
        "    hydro_data, species_traits_path",
        ")"
    ])

    code_box("Step 3: Set up release and settlement grids, create fieldset", [
        "(meshgrid, release_grids, settlement_grids, full_grid_df, grid_ids,",
        " coastal_grid, coastal_grid_df, coastal_grid_filtered,",
        " grid_df_for_mapping) = setup_release_settlement(",
        "    coastalmask, landmask, clean_spp, spp_params, lonarray, latarray",
        ")",
        "",
        "fieldset = create_fieldset(hydro_data, landmask, grid_ids)"
    ])

    code_box("Step 4: Create particle set and run simulation", [
        "pset, adjusted_runtime, interval, repeatdt = create_pset(",
        "    fieldset, meshgrid, runtime",
        ")",
        "kernels = create_custom_kernels(pset)",
        "output_path, chunks_dir = execute_simulation(",
        "    pset, kernels, adjusted_runtime, interval",
        ")"
    ])

    code_box("Step 5: Post-process trajectories", [
        "final_processed, buffer = post_process(",
        "    chunks_dir, traits, release_grids, settlement_grids, grid_ids",
        ")"
    ])

    code_box("Step 6: Run analyses", [
        "# Option A: Use the interactive analysis menu",
        "results = post_analysis(final_processed)",
        "",
        "# Option B: Run specific analyses directly",
        "traj_starts = load_traj_starts_from_chunks(chunks_dir)",
        "metrics = calculate_connectivity_metrics(",
        "    final_processed, release_grids, settlement_grids, traits,",
        "    traj_raw=traj_starts, grid_ids=grid_ids",
        ")",
        "display_metrics_table(metrics)"
    ])

    # =========================================================================
    # Example 2: Re-run Analysis on Existing Data
    # =========================================================================
    subsection_header("Example 2: Re-run Analysis on Existing Post-Processed Data")

    info_box([
        "If you already have post-processed data and want to try different analyses, you can call the analysis functions directly without re-running the full workflow.",
    ])

    code_box("Run analyses on existing data", [
        "from comcon.manual import *",
        "",
        "# Your existing data (from a previous workflow run)",
        "# final_processed, release_grids, settlement_grids, traits, traj",
        "",
        "# Type 1: Connectivity metrics",
        "metrics = calculate_connectivity_metrics(",
        "    final_processed, release_grids, settlement_grids, traits, traj",
        ")",
        "display_metrics_table(metrics)",
        "",
        "# Type 2: Network analysis",
        "network_results = network_analysis(",
        "    final_processed, release_grids, settlement_grids,",
        "    coastal_grid_df, grid_ids, full_grid_df, null_scale=30",
        ")",
        "display_network_summary(network_results)"
    ])

    # =========================================================================
    # Example 3: Custom Post-Processing Pipeline
    # =========================================================================
    subsection_header("Example 3: Custom Post-Processing Pipeline")

    info_box([
        "Use the individual filter functions to build a custom post-processing pipeline. ",
        "This is useful for testing different filtering parameters without re-running the full workflow.",
    ])

    code_box("Apply filters step by step", [
        "from comcon.manual import *",
        "",
        "# Filter by spawning window",
        "spawning_dfs = spawning_filter(traj, traits)",
        "",
        "# Filter by PLD with custom buffer",
        "pld_dfs = pld_filter(spawning_dfs, traits, settlement_grids,",
        "                     grid_ids, buffer_cells=2)",
        "",
        "# Filter by release habitat",
        "final_processed = habitat_filter(pld_dfs, release_grids)"
    ])

    # =========================================================================
    # Example 4: Download and Process Species Data Only
    # =========================================================================
    subsection_header("Example 4: Download and Process Species Data Only")

    info_box([
        "Work with species occurrence data independently.",
        "Useful for data exploration before running full simulations."
    ])

    code_box("Download and clean species data", [
        "from comcon.manual import *",
        "",
        "# Optional: Set up project directory",
        "project_dirs = setup_project_directory()",
        "",
        "# Load study area polygon",
        "geometry, shapely_polygon = read_geojson('./study_area.geojson')",
        "",
        "# Download species from GBIF",
        "records = get_spp_occ('Patella vulgata', None, geometry)",
        "distributions = pd.DataFrame(records)",
        "",
        "# Create masks from hydrodynamic data",
        "hydro_data = xr.open_dataset(project_dirs/hydro_data.nc')",
        "landmask, coastalmask, lonarray, latarray = create_masks(hydro_data)",
        "",
        "# Clean and filter data",
        "clean_spp = clean_spp_occ(",
        "    distributions, landmask, coastalmask, lonarray, latarray,",
        "    buffer_cells=1, habitat_type='coastal'",
        ")",
        "",
        "# Visualise",
        "plot_spp_occ(distributions, clean_spp, './study_area.geojson',",
        "             'coastal', lonarray, latarray)"
    ])

    # =========================================================================
    # Example 5: Load Existing Simulation and Re-Process
    # =========================================================================
    subsection_header("Example 5: Load Existing Simulation and Re-Process")

    info_box([
        "Load an existing simulation output and process with different parameters.",
        "Useful for testing different trait scenarios without re-running simulations."
    ])

    code_box("Load and reprocess simulation", [
        "from comcon.manual import *",
        "",
        "# Load existing simulation",
        "chunks_dir = process_sim_results(project_dirs/simulation.zarr',",
        "                                 chunk_size=50000, runtime_days=120)",
        "chunks_dir = assign_release_id(chunks_dir, interval=7)",
        "",
        "# Re-run post-processing with different traits or buffers",
        "final_processed, buffer = post_process(",
        "    chunks_dir, traits, release_grids, settlement_grids, grid_ids",
        ")"
    ])

    # =========================================================================
    # Parameter Reference
    # =========================================================================
    subsection_header("Parameter Reference: Wrapper Functions")

    data_box("params()", [
        "No parameters required (interactive prompts)",
        "→ Returns: hydro_params, species_traits_path, runtime, traits"
    ])

    data_box("download_hydrodynamic_data(**hydro_params)", [
        "Takes the hydro_params dict from params()",
        "→ Returns: xarray.Dataset"
    ])

    data_box("spp_occ(hydro_data, species_traits_path)", [
        "hydro_data: xarray.Dataset from download_hydrodynamic_data()",
        "species_traits_path: str from params()",
        "→ Returns: clean_spp, spp_params, landmask, coastalmask, lonarray, latarray"
    ])

    data_box("setup_release_settlement(coastalmask, landmask, clean_spp, spp_params, lonarray, latarray)", [
        "All parameters from spp_occ() output",
        "→ Returns: meshgrid, release_grids, settlement_grids, full_grid_df,",
        "           grid_ids, coastal_grid, coastal_grid_df, coastal_grid_filtered,",
        "           grid_df_for_mapping"
    ])

    data_box("create_fieldset(hydro_data, landmask, grid_ids)", [
        "hydro_data: from download_hydrodynamic_data()",
        "landmask: from spp_occ()",
        "grid_ids: from setup_release_settlement()",
        "→ Returns: fieldset"
    ])

    data_box("create_pset(fieldset, meshgrid, runtime)", [
        "fieldset: from create_fieldset()",
        "meshgrid: from setup_release_settlement()",
        "runtime: from params()",
        "→ Returns: pset, adjusted_runtime, interval, repeatdt"
    ])

    data_box("execute_simulation(pset, kernels, adjusted_runtime, interval)", [
        "pset: from create_pset()",
        "kernels: from create_custom_kernels(pset)",
        "adjusted_runtime, interval: from create_pset()",
        "→ Returns: output_path, chunks_dir"
    ])

    data_box("post_process(chunks_dir, traits, release_grids, settlement_grids, grid_ids)", [
        "chunks_dir: from execute_simulation()",
        "traits: from params()",
        "release_grids, settlement_grids, grid_ids: from setup_release_settlement()",
        "→ Returns: final_processed, buffer"
    ])

    data_box("post_analysis(final_processed)", [
        "final_processed: from post_process()",
        "→ Returns: results (runs the interactive analysis menu)"
    ])

    section_header("End of Example Workflows")
    print("For detailed help on any function: help(function_name)")
    print("For a list of all functions: list_functions()\n")

## Print detailed parameter requirements for a named function
def get_function_requirements(function_name):
    """
    Get detailed parameter requirements for a specific function.

    Parameters
    ----------
    function_name : str
        Name of the function to get info for
    """
    requirements = {
        'download_hydrodynamic_data': """
download_hydrodynamic_data(username, password, dataset_id, longitude_range, 
                           latitude_range, time_range, export, output_path)

Parameters:
-----------
username : str
    Your Copernicus Marine Service username
password : str
    Your Copernicus Marine Service password
dataset_id : str
    Copernicus dataset identifier
    Example: 'cmems_mod_ibi_phy_anfc_0.027deg-2D_PT1H-m'
longitude_range : list of 2 floats
    [min_longitude, max_longitude] in degrees
latitude_range : list of 2 floats
    [min_latitude, max_latitude] in degrees
time_range : list of 2 datetime objects
    [start_datetime, end_datetime]
    Example: [datetime(2023, 1, 1), datetime(2023, 6, 1)]
export : bool
    If True, save to NetCDF file
output_path : str
    Path for output file (required if export=True)

Returns:
--------
xarray.Dataset
    Hydrodynamic data with variables: uo, vo, latitude, longitude, time
""",
        'get_spp_occ': """
get_spp_occ(species_name, aphia_id, geometry)

Downloads occurrence records from GBIF for a single species.

Parameters:
-----------
species_name : str or None
    Scientific name (e.g., 'Nucella lapillus')
    Set to None if using aphia_id instead
aphia_id : int or None
    WoRMS AphiaID for the species
    Set to None if using species_name instead
geometry : str
    WKT polygon string defining search area
    Get this from: geometry, _ = read_geojson('study_area.geojson')

Returns:
--------
list of dict
    Each dict contains: species, latitude, longitude
""",
        'clean_spp_occ': """
clean_spp_occ(distributions, landmask, coastalmask, lonarray, latarray,
              buffer_cells=1, habitat_type='coastal')

Filter species occurrence data by habitat type.

Parameters:
-----------
distributions : pandas.DataFrame
    Must have columns: species, latitude, longitude
landmask : numpy.ndarray
    2D binary array (1=land, 0=ocean) from create_masks()
coastalmask : numpy.ndarray
    2D binary array (1=coastal, 0=non-coastal) from create_masks()
lonarray : numpy.ndarray
    2D longitude grid from create_masks()
latarray : numpy.ndarray
    2D latitude grid from create_masks()
buffer_cells : int, default 1
    Number of grid cells for coastal buffer
habitat_type : str, default 'coastal'
    'coastal' - Keep only coastal zone
    'ocean' - Keep all ocean (remove only land)
    'none' - No filtering

Returns:
--------
pandas.DataFrame
    Filtered occurrence data
""",
        'network_analysis': """
network_analysis(final_processed, release_grids, settlement_grids,
                 coastal_grid_df, grid_ids, full_grid_df, null_scale)

Run network analysis to identify connectivity hotspots.

Parameters:
-----------
final_processed : dict
    Keys = species names, Values = filtered trajectory DataFrames
    Each DataFrame needs: trajectory, time, obs, grid_id, lat, lon
release_grids : dict
    Keys = species names, Values = DataFrames with grid_id, latitude, longitude
settlement_grids : dict
    Keys = species names, Values = DataFrames with grid_id, latitude, longitude
coastal_grid_df : pandas.DataFrame
    All coastal grid cells with: grid_id, latitude, longitude
grid_ids : numpy.ndarray
    2D array mapping lat/lon indices to grid_id values
full_grid_df : pandas.DataFrame
    Full study area grid
null_scale : float
    Distance threshold in km for filtering connections

Returns:
--------
dict containing:
    'site_metrics' - DataFrame with source/sink/centrality metrics per site
    'connectivity_matrix' - DataFrame of connection counts
    'top10_overall' - Top 10 sites by overall composite
    'top10_functional' - Top 10 by functional roles
    'top10_centrality' - Top 10 by centrality measures
    'network_graph' - NetworkX graph object
""",
    }

    if function_name in requirements:
        print(requirements[function_name])
    else:
        print(f"No detailed requirements available for '{function_name}'")
        print("Try: help(function_name) for the function's docstring")


# endregion


# region Convenience Wrapper Functions
## Run connectivity analyses on existing post-processed data without re-running the full workflow
def run_analysis_only(final_processed, release_grids, settlement_grids, traits,
                      coastal_grid_df, grid_ids, full_grid_df,
                      run_type1=True, run_type2=True, type2_scale=30):
    """
    Run connectivity analyses on existing post-processed data.

    Use this when you have already completed Steps 1-5 and want to
    re-run or modify the analyses without re-running the full workflow.

    Parameters
    ----------
    final_processed : dict
        Output from post_process() or habitat_filter()
        Keys = species names, Values = filtered trajectory DataFrames
    release_grids : dict
        Release grids by species from setup_release_settlement()
    settlement_grids : dict
        Settlement grids by species from setup_release_settlement()
    traits : list of dict
        Species trait data
    coastal_grid_df : pandas.DataFrame
        Coastal grid from setup_release_settlement()
    grid_ids : numpy.ndarray
        Grid ID array from setup_release_settlement()
    full_grid_df : pandas.DataFrame
        Full grid from setup_release_settlement()
    run_type1 : bool, default True
        Run connectivity metrics analysis
    run_type2 : bool, default True
        Run network analysis
    type2_scale : int, default 30
        Distance scale in km for network analysis

    Returns
    -------
    dict
        Results dictionary with 'type1' and/or 'type2' keys
    """
    results = {}

    subsection_header("Running Connectivity Analyses")

    if run_type1:
        print("\nType 1: Connectivity Metrics")
        display_line("-")
        results['type1'] = calculate_connectivity_metrics(
            final_processed, release_grids, settlement_grids, traits, grid_ids=grid_ids
        )
        display_metrics_table(results['type1'])

    if run_type2:
        subsection_header("Type 2: Network Analysis")
        results['type2'] = network_analysis(
            final_processed, release_grids, settlement_grids,
            coastal_grid_df, grid_ids, full_grid_df, type2_scale
        )
        display_network_summary(results['type2'])

    return results


## Re-run post-processing on loaded trajectory data with different parameters
def run_postprocessing_only(traj, traits, release_grids, settlement_grids,
                            grid_ids, buffer=1):
    """
    Run post-processing on existing trajectory data.

    Use this when you have simulation output and want to re-run
    post-processing with different parameters.

    Parameters
    ----------
    traj : pandas.DataFrame
        Trajectory data loaded with load_traj_from_chunks(chunks_dir), where chunks_dir comes from process_sim_results() then assign_release_id().
        Must have columns: trajectory, time, obs, grid_id, lat, lon, release_id
    traits : list of dict
        Species trait data with spawning_start, spawning_end, min_PLD, max_PLD
    release_grids : dict
        Release grids by species
    settlement_grids : dict
        Settlement grids by species
    grid_ids : numpy.ndarray
        2D grid ID array
    buffer : int, default 1
        Settlement buffer in grid cells

    Returns
    -------
    dict
        final_processed - filtered trajectories by species
    """
    subsection_header("Running Post-Processing")

    print("\nApplying spawning seasonality filter...")
    spawning_dfs = spawning_filter(traj, traits)

    print("\nApplying PLD filter...")
    pld_dfs = pld_filter(spawning_dfs, traits, settlement_grids, grid_ids, buffer)

    print("\nApplying habitat filter...")
    final_processed = habitat_filter(pld_dfs, release_grids)

    # Summary
    print("\n")
    display_line("-")
    print("Post-processing Summary")
    display_line("-")
    total = 0
    for species, df in final_processed.items():
        if not df.empty:
            n = df['trajectory'].nunique()
            total += n
            print(f"  {display_species_name(species)}: {n} trajectories")
    print(f"\nTotal successful: {total:}")

    return final_processed


## Load and process an existing simulation zarr file, adding release_id assignments
def load_simulation_output(zarr_path, interval_days, chunk_size=50000, runtime_days=None):
    """
    Load existing simulation output for further processing.

    Parameters
    ----------
    zarr_path : str
        Path to .zarr simulation output directory
    interval_days : int
        Release interval used in simulation (in days)
        e.g., 7 for weekly, 1 for daily
    chunk_size : int, default 50000
        Trajectories per processing chunk
    runtime_days : int, optional
        Simulation runtime for memory estimation

    Returns
    -------
    pandas.DataFrame
        Trajectory data with release_id column added
    """
    subsection_header("Loading Simulation Output")

    print(f"Path: {zarr_path}")
    print(f"Chunk size: {chunk_size:}")

    chunks_dir = process_sim_results(zarr_path, chunk_size, runtime_days)
    chunks_dir = assign_release_id(chunks_dir, interval_days)

    display_line("-")
    print(f"Trajectory data processed and stored on disk at: {chunks_dir}")
    print(f"Use load_traj_from_chunks(chunks_dir) if you need the full DataFrame.")
    display_line("-")

    return chunks_dir

## Allows users to create the nested data structure to run analyses at multiple spatial scales
def run_multiscale_network(final_processed, release_grids, settlement_grids,
                           coastal_grid_df, grid_ids, full_grid_df,
                           scales=(None,), site_name_lookup=None,
                           exclude_self_recruitment=False):
    """
    Run network_analysis at one or more distance scales and return the nested structure the dropdown viewers expect.

    Output structure: {scale_label: {'community': ..., 'per_species': {species: ...}}}

    Parameters
    ----------
    final_processed : dict
        Post-processed trajectory data from post_process() or habitat_filter().
    release_grids : dict
        Release grids by species from setup_release_settlement().
    settlement_grids : dict
        Settlement grids by species from setup_release_settlement().
    coastal_grid_df : pandas.DataFrame
        Coastal grid from setup_release_settlement().
    grid_ids : numpy.ndarray
        Grid ID array from setup_release_settlement().
    full_grid_df : pandas.DataFrame
        Full grid from setup_release_settlement().
    scales : iterable
        Distance scales in km. Use None for the full domain (no distance threshold). e.g. scales=[None, 30, 50].
    site_name_lookup : dict, optional
        {grid_id: name} for named site-based runs. If given, site names appear in the outputs automatically.
    exclude_self_recruitment : bool, default False
        If True, exclude self-recruitment from source/sink metrics.

    Returns
    -------
    dict
        Nested results ready to pass to network_metrics_viewer(), trait_matrix_viewer(), or trait_community_viewer().
    """
    if site_name_lookup is not None:
        import comcon.interactive as _ci
        if getattr(_ci, 'spp_params', None) is None:
            _ci.spp_params = {}
        _ci.spp_params['site_name_lookup'] = site_name_lookup

    all_network_results = {}
    for scale in scales:
        label = 'full_domain' if scale is None else f'{scale}km'
        all_network_results[label] = network_analysis(
            final_processed, release_grids, settlement_grids,
            coastal_grid_df, grid_ids, full_grid_df, scale,
            exclude_self_recruitment=exclude_self_recruitment)
    return all_network_results
# endregion


# region Module Exports
__all__ = [
    # Documentation functions
    'list_functions',
    'example_workflow',
    'get_function_requirements',

    # Convenience wrappers
    'run_analysis_only',
    'run_postprocessing_only',
    'run_multiscale_network',
    'load_simulation_output',

    # Constants
    'map_config',

    # Helper functions
    'get_positive_integer',
    'standardise_species_name',
    'display_species_name',
    'get_output_format',
    'save_figure',
    'get_output_choice',
    'calculate_figure_dimensions',

    # Step 1 functions
    'params',
    'load_spp_traits',
    'collect_hydro_params',
    'setup_project_directory',
    'get_default_dir',
    'get_package_data_path',
    'load_builtin_traits',
    'select_species_from_database',
    'calculate_overall_dispersal_period',
    'download_hydrodynamic_data',

    # Step 2 functions
    'spp_occ',
    'collect_spp_occ_params',
    'edit_spp_params',
    'create_map',
    'read_geojson',
    'get_aphia',
    'get_spp_occ',
    'get_all_spp_occ',
    'make_coastalmask',
    'create_masks',
    'clean_spp_occ',
    'validate_spp_occ',
    'get_spp_colours',
    'plot_spp_occ',
    'validate_uploaded_data',

    # Step 3 functions
    'setup_release_settlement',
    'map_spp_to_grid',
    'filter_by_polygon',
    'save_grids',
    'reflective_boundary',
    'create_fieldset',

    # Step 4 functions
    'create_pset',
    'execute_simulation',
    'create_custom_kernels',
    'MemoryMonitor',
    'check_system_memory',
    'get_memory_config',
    'get_zarr_size',
    'process_sim_results',
    'assign_release_id',
    'save_session',
    'load_session',

    # Step 5 functions
    'post_process',
    'spawning_filter',
    'pld_filter',
    'habitat_filter',

    # Step 6 functions
    'post_analysis',
    'calculate_connectivity_metrics',
    'display_metrics_table',
    'network_analysis',
    'add_custom_composite',
    'run_multiscale_network',
    'display_network_summary',
    'compare_models',
    'display_sensitivity_comparison',
    'export_connectivity_metrics',
    'export_network_results',

    # Visualisation
    'plot_masks',
    'create_network_map',
    'create_connectivity_matrix_plot',
    'trait_metrics_viewer',
    'network_metrics_viewer',
    'trait_matrix_viewer',
    'trait_metric_viewer',
    'trait_heatmap_viewer',
    'trait_community_viewer',
    'build_heatmap_figure',
    'create_community_map',
    'create_matrix_with_map',
    'create_comparison_map',

]
# endregion