"""
ComCon: Passive Dispersal Pathway
==================================

Implements the passive dispersal pathway for ComCon
A hydrodynamic-only assessment of marine larval connectivity with no species-specific trait data.

This pathway follows James et al. (2025):
- Particles are released from all ocean cells in the study domain.
- Two 28-day simulations are run (summer + winter).
- Particle positions are extracted at four fixed timepoints: 7, 14, 21, 28 days.
- Network analyses are computed automatically and saved to the project folder.

The main function for users is `passive_dispersal_pathway()`.
All other functions in this module are called by this main entry point.
"""

# region Imports
import os
import io
import contextlib
import gc
import json
import math
import time
import threading
import itertools
import warnings
from datetime import datetime, timedelta
from operator import attrgetter

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import numpy as np
import pandas as pd
import copernicusmarine as cm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as _mpl
import ipywidgets as widgets
import community as community_louvain

from IPython.display import display, clear_output, HTML as IPHTML
from parcels import ParticleSet, JITParticle, Variable
from tqdm.auto import tqdm

from comcon.interactive import (
    calculate_resolution,
    get_copernicus_credentials,
    setup_project_directory,
    write_project_metadata,
    download_hydrodynamic_data,
    create_masks,
    create_network_map,
    create_fieldset,
    dispersal_animation,
    create_custom_kernels,
    process_sim_results,
    save_figure,
    assign_release_id,
    check_system_memory,
    get_memory_config,
    get_zarr_size,
    MemoryMonitor,
    calculate_connectivity_metrics,
    export_connectivity_metrics,
    network_analysis,
    export_network_results,
    create_metrics_summary_table,
    create_community_map,
    build_heatmap_figure,
    create_connectivity_matrix_plot,
    create_matrix_with_map,
    create_interactive_metric_viewer,
    plot_masks,
    get_yes_no,
    get_positive_integer,
    parse_coordinate,
    section_header,
    subsection_header,
    info_box,
    data_box,
    display_line,
    status_message,
    completion_message,
)

warnings.filterwarnings('ignore')
# endregion

# region Domain and Grid Binning
def validate_domain(lat_min, lat_max, lon_min, lon_max):
    """
    Validate user-supplied latitude/longitude bounds for a passive dispersal run.

    Raises an error for invalid bounds (reversed or out of global range).
    Warns for large domains that may result in long simulation runtimes.

    Parameters
    ----------
    lat_min, lat_max : float
        Latitude bounds in decimal degrees.
    lon_min, lon_max : float
        Longitude bounds in decimal degrees.

    Returns
    -------
    tuple of (float, float)
        (lat_extent, lon_extent) in degrees.

    Raises
    ------
    ValueError
        If bounds are reversed, out of global range, or outside valid latitude.
    """
    # Type check
    try:
        lat_min = float(lat_min)
        lat_max = float(lat_max)
        lon_min = float(lon_min)
        lon_max = float(lon_max)
    except (TypeError, ValueError):
        raise ValueError("Latitude and longitude bounds must be numeric.")

    # Range checks
    if not (-90 <= lat_min <= 90) or not (-90 <= lat_max <= 90):
        raise ValueError(f"Latitudes must be in [-90, 90]. Got lat_min={lat_min}, lat_max={lat_max}.")
    if not (-180 <= lon_min <= 180) or not (-180 <= lon_max <= 180):
        raise ValueError(f"Longitudes must be in [-180, 180]. Got lon_min={lon_min}, lon_max={lon_max}.")

    # Ordering checks
    if lat_min >= lat_max:
        raise ValueError(f"lat_min ({lat_min}) must be less than lat_max ({lat_max}).")
    if lon_min >= lon_max:
        raise ValueError(f"lon_min ({lon_min}) must be less than lon_max ({lon_max}).")

    lat_extent = lat_max - lat_min
    lon_extent = lon_max - lon_min

    # Size warning
    if lat_extent > 20 or lon_extent > 20:
        print("\n  Warning: Large domain detected.")
        print(f"    Latitude extent:  {lat_extent:.2f} degrees")
        print(f"    Longitude extent: {lon_extent:.2f} degrees")
        print("    Simulations on domains this large may take several hours.")
        print()

    return lat_extent, lon_extent

## Bins the landmask into coarser resolution
def bin_mask(native_landmask, native_lonarray, native_latarray, target_km):
    """
    Makes the resolution of the native landmask based on the hydrodynamic data coarser using majority vote.

    For each new coarse cell in the target grid, it counts the native land cells whose centroids fall inside it.
    The coarse cell is classified as land if >=50% of its native cells are land, otherwise ocean.
    Coarse cells containing zero native cells are treated as land (excluded from release). This only occurs at edge-cases if the coarser resolution does not overlap with the native resolution exactly.

    Parameters
    ----------
    native_landmask : numpy.ndarray
        2D binary array at native resolution where 1=land and 0=ocean.
        Shape: (n_lat, n_lon).
    native_lonarray : numpy.ndarray
        2D meshgrid of longitudes at native resolution. Shape matches native_landmask.
    native_latarray : numpy.ndarray
        2D meshgrid of latitudes at native resolution. Shape matches native_landmask.
    target_km : int or float
        Target resolution for the coarse grid in kilometres.

    Returns
    -------
    binned_landmask : numpy.ndarray
        2D binary array at target resolution where 1=land and 0=ocean.
    binned_lonarray : numpy.ndarray
        2D meshgrid of coarse-cell-centroid longitudes.
    binned_latarray : numpy.ndarray
        2D meshgrid of coarse-cell-centroid latitudes.

    Notes
    -----
    Target resolution is converted from km to degrees using a spherical approximation:
    1 degree of latitude ~ 111 km everywhere, 1 degree of longitude ~ 111 * cos(lat_mid) km at the midpoint latitude of the domain.

    If the target resolution is equal to the native resolution (within 10%), the native arrays are returned unchanged.
    """
    # Extract native resolution in degrees
    native_lat_step = abs(float(native_latarray[1, 0] - native_latarray[0, 0]))
    native_lon_step = abs(float(native_lonarray[0, 1] - native_lonarray[0, 0]))

    # Convert target km to degrees using spherical approximation
    lat_mid = float(np.mean(native_latarray))
    target_lat_step = target_km / 111.0
    target_lon_step = target_km / (111.0 * max(np.cos(np.deg2rad(lat_mid)), 0.1))

    # If target is within 10% of native, return native unchanged (no binning needed)
    native_km_lat = native_lat_step * 111.0
    if abs(target_km - native_km_lat) / native_km_lat < 0.10:
        return native_landmask.copy(), native_lonarray.copy(), native_latarray.copy()

    # Build coarse grid edges
    lat_min = float(np.min(native_latarray))
    lat_max = float(np.max(native_latarray))
    lon_min = float(np.min(native_lonarray))
    lon_max = float(np.max(native_lonarray))

    coarse_lat_edges = np.arange(lat_min, lat_max + target_lat_step, target_lat_step)
    coarse_lon_edges = np.arange(lon_min, lon_max + target_lon_step, target_lon_step)

    # Guard against too-small domains producing zero or one cell
    if len(coarse_lat_edges) < 2 or len(coarse_lon_edges) < 2:
        raise ValueError(
            f"Target resolution ({target_km} km) is too coarse for the domain extent. "
            "Try a smaller target resolution or a larger domain."
        )

    n_coarse_lat = len(coarse_lat_edges) - 1
    n_coarse_lon = len(coarse_lon_edges) - 1

    binned_landmask = np.ones((n_coarse_lat, n_coarse_lon), dtype=np.int32)  # default land
    binned_lat_centroids = np.zeros(n_coarse_lat)
    binned_lon_centroids = np.zeros(n_coarse_lon)

    # Flatten native arrays once
    native_lon_flat = native_lonarray.ravel()
    native_lat_flat = native_latarray.ravel()
    native_land_flat = native_landmask.ravel()

    # Assign each native cell to a coarse cell via digitize
    lat_bin_idx = np.digitize(native_lat_flat, coarse_lat_edges) - 1
    lon_bin_idx = np.digitize(native_lon_flat, coarse_lon_edges) - 1

    # Clip to valid range (digitize can return n for values at the max edge)
    lat_bin_idx = np.clip(lat_bin_idx, 0, n_coarse_lat - 1)
    lon_bin_idx = np.clip(lon_bin_idx, 0, n_coarse_lon - 1)

    # Majority vote per coarse cell
    for i in range(n_coarse_lat):
        binned_lat_centroids[i] = (coarse_lat_edges[i] + coarse_lat_edges[i + 1]) / 2.0
        for j in range(n_coarse_lon):
            mask = (lat_bin_idx == i) & (lon_bin_idx == j)
            native_cells_in_bin = native_land_flat[mask]
            if len(native_cells_in_bin) == 0:
                binned_landmask[i, j] = 1  # land by default (excluded from release)
            else:
                frac_land = np.mean(native_cells_in_bin)
                binned_landmask[i, j] = 1 if frac_land >= 0.5 else 0

    for j in range(n_coarse_lon):
        binned_lon_centroids[j] = (coarse_lon_edges[j] + coarse_lon_edges[j + 1]) / 2.0

    binned_lonarray, binned_latarray = np.meshgrid(binned_lon_centroids, binned_lat_centroids)

    return binned_landmask, binned_lonarray, binned_latarray

## Creates grid for release and settlement
def create_binned_grid(binned_landmask, binned_lonarray, binned_latarray):
    """
    Build a DataFrame of binned ocean cells for the passive dispersal pathway.

    Takes the binned landmask (from bin_landmask) and constructs a DataFrame of ocean-only cells with sequential grid IDs.
    This DataFrame serves as both the release location source and the analysis grid for the passive pathway.

    Parameters
    ----------
    binned_landmask : numpy.ndarray
        2D binary array where 1=land and 0=ocean.
    binned_lonarray : numpy.ndarray
        2D meshgrid of coarse-cell-centroid longitudes.
    binned_latarray : numpy.ndarray
        2D meshgrid of coarse-cell-centroid latitudes.

    Returns
    -------
    pandas.DataFrame
        DataFrame of binned ocean cells with columns:
        - grid_id : int, sequential from 1
        - longitude : float
        - latitude : float
        - row_idx : int, row index in the binned grid
        - col_idx : int, column index in the binned grid
        - is_land : int, always 0 (ocean cells only)
        - is_coastal : int, always 1 (all ocean cells treated as coastal
                       in Path A — network_analysis expects this column)

    Raises
    ------
    ValueError
        If no ocean cells are found in the binned grid.
    """
    ocean_rows, ocean_cols = np.where(binned_landmask == 0)

    if len(ocean_rows) == 0:
        raise ValueError(
            "No ocean cells found in the binned grid. "
            "Check that your latitude/longitude bounds include open ocean."
        )

    df = pd.DataFrame({
        'grid_id': np.arange(1, len(ocean_rows) + 1, dtype=np.int64),
        'longitude': binned_lonarray[ocean_rows, ocean_cols],
        'latitude': binned_latarray[ocean_rows, ocean_cols],
        'row_idx': ocean_rows.astype(np.int32),
        'col_idx': ocean_cols.astype(np.int32),
        'is_land': 0,
        'is_coastal': 1,  # network_analysis expects this column; all ocean treated as coastal in Path A
    })

    return df

## Creates the grid points for release and settlement (meshgrid only, no dataframe)
def create_release_grid(binned_ocean_grid_df):
    """
    Build the release meshgrid array from the binned ocean grid DataFrame.

    Matches the ``meshgrid`` format used by the trait pathway's ``setup_release_settlement``.
    One row per release cell, column 0 = longitude, column 1 = latitude).

    Parameters
    ----------
    binned_ocean_grid_df : pandas.DataFrame
        Output from `create_binned_grid`. Must contain 'longitude' and 'latitude' columns.

    Notes
    -----
    Particles whose position falls inside a binned cell that is classified as
    land (i.e. the cell is majority-land under the binning majority vote) are
    reassigned to their nearest binned ocean cell by Euclidean distance on row
    and column indices. This is necessary because the particle simulation runs
    at native NEMO resolution (where the particle's position is genuinely ocean)
    but the connectivity analysis runs at binned resolution (where the bin
    containing the particle may be classified as land). The reassignment error
    is bounded by roughly half a binned cell width and preserves connectivity
    signal that would otherwise be silently lost by dropping these particles.
    This behaviour mirrors the effective grid-alignment of James et al. (2025),
    where particles are released and settled on a single fixed ocean grid.

    Returns
    -------
    numpy.ndarray
        2D array of shape (n_cells, 2): column 0 is longitude, column 1 is
        latitude. One row per binned ocean cell centroid.
    """
    meshgrid = np.column_stack([
        binned_ocean_grid_df['longitude'].values,
        binned_ocean_grid_df['latitude'].values,
    ])
    return meshgrid

## Snap the particle positions to the binned grid
def snap_to_binned_grid(particle_lons, particle_lats, binned_lonarray, binned_latarray, binned_ocean_grid_df):
    """
    Snap particle positions to the nearest binned ocean cell.

    For each particle position, finds the binned cell ID using nearest-neighbour and returns the corresponding binned grid_id.
    Particles that fall in binned land cells or outside the domain are assigned grid_id = -1 (sentinel for "no valid cell").

    Parameters
    ----------
    particle_lons : numpy.ndarray
        1D array of particle longitudes.
    particle_lats : numpy.ndarray
        1D array of particle latitudes. Must be same length as particle_lons.
    binned_lonarray : numpy.ndarray
        2D meshgrid of binned cell centroid longitudes (from bin_masks).
    binned_latarray : numpy.ndarray
        2D meshgrid of binned cell centroid latitudes (from bin_masks).
    binned_ocean_grid_df : pandas.DataFrame
        Binned ocean grid DataFrame (from create_binned_grid). Must contain
        'row_idx', 'col_idx' and 'grid_id' columns.

    Returns
    -------
    numpy.ndarray
        1D array of binned grid_ids, same length as input arrays.
        Particles in land cells or outside the domain get grid_id = -1.

    Notes
    -----
    Implementation is fully vectorised. Nearest-neighbour index lookup uses np.searchsorted on the 1D centroid arrays.
    Faster than KDTree for the typical Path A grid sizes.
    The (row, col) -> grid_id conversion uses a flat 2D numpy array instead of a Python dict.
    """
    # Extract 1D centroid arrays
    lat_centroids_1d = binned_latarray[:, 0]
    lon_centroids_1d = binned_lonarray[0, :]

    n_lat = len(lat_centroids_1d)
    n_lon = len(lon_centroids_1d)

    def nearest_id(values, centroids_1d):
        """Vectorised nearest-neighbour index lookup on a sorted 1D array."""
        idx = np.searchsorted(centroids_1d, values)
        idx = np.clip(idx, 1, len(centroids_1d) - 1)
        left = centroids_1d[idx - 1]
        right = centroids_1d[idx]
        pick_left = (values - left) <= (right - values)
        return np.where(pick_left, idx - 1, idx)

    # NEMO data is usually stored south-to-north (ascending lat), but some products are stored north-to-south.
    # Handle both cases by flipping if the centroid array is descending, since np.searchsorted requires ascending.
    if lat_centroids_1d[0] > lat_centroids_1d[-1]:
        lat_sorted = lat_centroids_1d[::-1]
        row_idx_flipped = nearest_id(particle_lats, lat_sorted)
        row_idx = (n_lat - 1) - row_idx_flipped
    else:
        row_idx = nearest_id(particle_lats, lat_centroids_1d)

    col_idx = nearest_id(particle_lons, lon_centroids_1d)

    row_idx = np.clip(row_idx, 0, n_lat - 1)
    col_idx = np.clip(col_idx, 0, n_lon - 1)

    # Vectorised (row, col) -> grid_id lookup via a flat 2D array.
    # Build it once from the ocean grid DataFrame
    # Default value -1 marks "no valid binned ocean cell" (i.e. the particle landed in a land cell or outside the domain).
    lookup_array = np.full((n_lat, n_lon), -1, dtype=np.int64)
    ocean_rows_arr = binned_ocean_grid_df['row_idx'].values.astype(np.int32)
    ocean_cols_arr = binned_ocean_grid_df['col_idx'].values.astype(np.int32)
    ocean_ids = binned_ocean_grid_df['grid_id'].values.astype(np.int64)
    lookup_array[ocean_rows_arr, ocean_cols_arr] = ocean_ids

    grid_ids = lookup_array[row_idx, col_idx]

    # Any particle that snapped to a binned land cell (value -1) is reassigned to its nearest binned ocean cell
    # Uses Euclidean distance on row/col indices
    invalid_mask = grid_ids == -1
    if invalid_mask.any():
        invalid_rows = row_idx[invalid_mask]
        invalid_cols = col_idx[invalid_mask]

        # For each invalid particle, find the closest ocean cell on the grid.
        # Use broadcasting: shape (n_invalid, n_ocean_cells) distance matrix.
        ocean_rc = np.column_stack([ocean_rows_arr, ocean_cols_arr]).astype(np.float64)
        invalid_rc = np.column_stack([invalid_rows, invalid_cols]).astype(np.float64)

        diffs = invalid_rc[:, None, :] - ocean_rc[None, :, :]
        dist_sq = (diffs ** 2).sum(axis=2)
        nearest_idx = np.argmin(dist_sq, axis=1)

        grid_ids[invalid_mask] = ocean_ids[nearest_idx]

    return grid_ids

    grid_ids = lookup_array[row_idx, col_idx]

    return grid_ids
# endregion

# region Particle Set Creation
def create_passive_pset(fieldset, release_grid, release_time, n_particles_per_cell=100, kh=1):
    """
    Create a non-interactive ParticleSet for the passive dispersal pathway.

    Mirrors the trait pathway's `create_pset()` but hardcodes all parameters so no user input is required.
    100-particles-per-cell, Kh=1 configuration that follows James et al. (2025).

    Parameters
    ----------
    fieldset : parcels.FieldSet
        The fieldset for particle tracking, built at native hydrodynamic resolution by `create_fieldset()`.
    release_grid : numpy.ndarray
        2D array of shape (n_cells, 2) where column 0 is longitude and column 1 is latitude.
        Output of `create_release_grid()`.
    release_time : datetime.datetime
        The single timepoint at which all particles are released (day 1 of the simulation window).
    n_particles_per_cell : int, optional
        Number of particles released per binned ocean cell. Default 100, matching James et al. (2025).
    kh : float, optional
        Horizontal diffusivity in m^2/s, applied uniformly across the domain.
        Default 1.0, matching the trait pathway's default.

    Returns
    -------
    parcels.ParticleSet
        Configured particle set with all particles released at
        ``release_time``. Unlike the trait pathway, there is no ``repeatdt``
        (single release event).

    Notes
    -----
    Particle variables match the trait pathway's particle class exactly (distance, prev_lon, prev_lat, on_land, grid_id).
    The custom kernels built by `create_custom_kernels()` work without modification.
    The in-kernel `grid_id` field is sampled but never used downstream in Path A as particle positions are snapped to the binned grid after the simulation in `extract_fixed_timepoints()`.
    """
    # Step 1: Extract release locations from the meshgrid array
    lons = release_grid[:, 0].flatten()
    lats = release_grid[:, 1].flatten()
    num_locations = len(lats)

    # Step 2: Expand each location to n_particles_per_cell copies
    lat = list(itertools.chain.from_iterable(
        itertools.repeat(x, n_particles_per_cell) for x in lats
    ))
    lon = list(itertools.chain.from_iterable(
        itertools.repeat(x, n_particles_per_cell) for x in lons
    ))

    total_particles = num_locations * n_particles_per_cell

    data_box("Particle setup", [
        f"Release locations: {num_locations}",
        f"Particles per location: {n_particles_per_cell}",
        f"Total particles: {total_particles}",
        f"Release event: single (day 1 of simulation window)",
        f"Horizontal diffusivity: {kh} m^2/s",
    ])

    # Step 3: Apply horizontal diffusivity to the fieldset
    fieldset.add_constant_field("Kh_zonal", kh, mesh="spherical")
    fieldset.add_constant_field("Kh_meridional", kh, mesh="spherical")

    # Step 4: Build the particle class with the same variables as the trait pathway
    extra_vars = [
        Variable("distance", initial=0.0, dtype=np.float32),
        Variable("prev_lon", dtype=np.float32, to_write=False, initial=attrgetter("lon")),
        Variable("prev_lat", dtype=np.float32, to_write=False, initial=attrgetter("lat")),
        Variable('on_land', dtype=np.int32, initial=0),
        Variable('grid_id', dtype=np.int32, initial=0),
    ]
    Particle = JITParticle.add_variables(extra_vars)

    # Step 5: Create the ParticleSet with a single release event (no repeatdt)
    pset = ParticleSet(
        fieldset=fieldset,
        pclass=Particle,
        lon=lon,
        lat=lat,
        time=release_time,
    )

    return pset
# endregion


# region Simulation Execution
def execute_passive_simulation(pset, kernels, runtime_days, output_path, dt_minutes=60, output_dt_hours=24):
    """
    Execute a non-interactive passive dispersal simulation.

    Mirrors the trait pathway's `execute_simulation()` but hardcodes timestep and output frequency, skips all user prompts and suppresses the user-facing MemoryMonitor messages.
    Used by Path A for the two 28-day seasonal simulations.

    Parameters
    ----------
    pset : parcels.ParticleSet
        Configured particle set from ``create_passive_pset()``.
    kernels : parcels.Kernel
        Combined kernel from ``create_custom_kernels()``.
    runtime_days : int
        Total runtime in days. Path A uses 28.
    output_path : str
        Full path to the zarr directory for trajectory output.
    dt_minutes : int, optional
        Simulation timestep in minutes. Default 60.
    output_dt_hours : int, optional
        Output frequency in hours. Default 24 (daily positions, matching James et al. 2025).

    Returns
    -------
    output_path : str
        Path to the zarr file where raw trajectory data was written.
    traj : pandas.DataFrame
        Processed trajectory DataFrame loaded back from the zarr file via `process_sim_results()`.

    Notes
    -----
    Unlike ``execute_simulation()``, this function:
    - Does not prompt for output path, dt or output frequency.
    - Does not prompt for confirmation on estimated memory use.
    - Does not display MemoryMonitor start/end messages.
    - Runs as a single release event with no staggered cohorts.

    On memory errors or simulation failures, returns (None, None) after printing an error message, matching the trait pathway's error handling.
    """
    try:
        subsection_header("Running the simulation")

        # Ensure the output path ends with .zarr
        if not output_path.endswith(".zarr"):
            output_path += ".zarr"

        # Build timedelta objects
        dt = timedelta(minutes=dt_minutes)
        output_dt = timedelta(hours=output_dt_hours)
        # Parcels stops writing observations at t <= runtime.
        # Adding one day here ensures obs=runtime_days (day 28 for a standard Path A run) is  inside the window
        runtime = timedelta(days=runtime_days + 1)

        # Memory config (same sizing logic as the trait pathway, used internally by the chunked loading below).
        # Suppress its user-facing print from check_system_memory() since Path A runs are small enough that the memory details aren't useful
        with contextlib.redirect_stdout(io.StringIO()):
            available_gb = check_system_memory()
        mem_config = get_memory_config(available_gb, runtime_days, output_dt_hours)

        n_particles = len(pset)

        data_box("Simulation settings", [
            f"Runtime: {runtime_days} days",
            f"Particles: {n_particles:,}",
            f"Output frequency: {output_dt_hours} hours",
            f"Time step: {dt_minutes} minutes",
            f"Output path: {output_path}",
        ])

        # Run the simulation. Suppress MemoryMonitor user messages by using
        # a plain gc.collect() before and after instead of the context manager.
        gc.collect()
        print("\nRunning OceanParcels simulation...")
        chunk_particles = min(mem_config['chunk_size'], len(pset))

        output_file = pset.ParticleFile(
            name=output_path,
            outputdt=output_dt,
            chunks=(chunk_particles, 1),
        )

        pset.execute(
            kernels,
            runtime=runtime,
            dt=dt,
            output_file=output_file,
        )
        gc.collect()

        # Load the zarr back into a pandas DataFrame
        print("\nProcessing simulation results...")
        file_size_gb = get_zarr_size(output_path)
        print(f"Output file size: {file_size_gb:.2f} GB")
        print(f"Using memory-safe chunked loading ({mem_config['chunk_size']:,} trajectories per chunk)")
        traj_df = process_sim_results(output_path, mem_config['chunk_size'], runtime_days)

        # Path A has a single release event, so assign_release_id would just produce a column of 0s
        traj_df['release_id'] = 0
        traj = traj_df

        print("\nSimulation complete.")
        print(f"Final dataset: {len(traj):,} rows, {traj['trajectory'].nunique():,} trajectories")

        # Save the processed DataFrame to parquet for reliable session recovery
        traj_parquet_path = output_path.replace('.zarr', '_processed.parquet')
        try:
            traj.to_parquet(traj_parquet_path, index=False, engine='pyarrow')
            print(f"Processed trajectories saved: {traj_parquet_path}")
            print(f"  ({os.path.getsize(traj_parquet_path) / (1024 ** 2):.1f} MB)")
        except Exception as e:
            print(f"Warning: Could not save processed trajectories to parquet: {e}")

        return output_path, traj

    except MemoryError:
        print("\nError: Out of memory during simulation.")
        print("Suggestions:")
        print("  - Reduce the domain size and re-run passive_dispersal_pathway().")
        print("  - Close other applications to free RAM.")
        return None, None

    except Exception as e:
        error_msg = str(e)
        print("\nError: The simulation failed.")
        print(f"Error message: {error_msg}")
        if "time domain" in error_msg.lower() or "time_extrapolation" in error_msg.lower():
            print("\nThe simulation ran beyond the hydrodynamic data coverage.")
            print("This should not happen in Path A since the hydro download is sized to the simulation window. Please report this as a bug.")
        return None, None
# endregion

# region Fixed-Timepoint Extraction
def extract_fixed_timepoints(traj, binned_ocean_grid_df, binned_lonarray, binned_latarray,
                              timepoints=(7, 14, 21, 28), season_label=''):
    """
    Extract particle positions at fixed timepoints and snap to the binned grid.

    For each specified day (7, 14, 21, 28 by default) this function finds every particle's position at that day of the simulation and snaps it to a binned grid cell.
    It then builds a DataFrame in the format expected by `network_analysis()`:
    Two rows per particle, `obs=0` holding the binned grid_id of the release cell and `obs=1` holding the binned grid_id of the cell containing the particle at the timepoint.

    Parameters
    ----------
    traj : pandas.DataFrame
        Raw trajectory data from `process_sim_results()`. Must contain columns: trajectory, obs, time, lat, lon.
    binned_ocean_grid_df : pandas.DataFrame
        Binned ocean grid DataFrame from `create_binned_grid()`. Used to snap particle positions to binned grid_ids.
    binned_lonarray : numpy.ndarray
        2D meshgrid of binned cell centroid longitudes (from `bin_mask`).
    binned_latarray : numpy.ndarray
        2D meshgrid of binned cell centroid latitudes (from `bin_mask`).
    timepoints : tuple of int, optional
        Days at which to extract particle positions. Default (7, 14, 21, 28) follows James et al. (2025).
    season_label : str, optional
        Label prefix for the output dict keys, e.g. 'summer' or 'winter'. Default ''. Keys are formatted as f'{season_label}_day{N}' if season_label is non-empty, otherwise f'day{N}'.

    Returns
    -------
    dict
        Dictionary mapping timepoint labels to `final_processed`-style dicts. Each value is `{'passive': DataFrame}` where the DataFrame has columns: trajectory, obs, grid_id, lat, lon, time.
        Feed each value directly into `network_analysis()` as the `final_processed` argument.

    Notes
    -----
    The parcels simulation outputs particle positions every 24 hours, so `obs=N` in the raw trajectory data corresponds to day N of the simulation.
    This function filters by `obs` rather than matching on time, because the passive pathway uses a single release event and fixed daily output, so `obs` and elapsed days are equivalent.
    The trait pathway has to match on time because it uses multiple staggered release cohorts.

    Path A uses the native-resolution landmask for the reflective boundary kernel, so particles should not end up on land during the simulation.
    However, a particle in a native ocean cell can still snap to a binned land cell if the binned cell has a majority of native-land cells inside it (from the binning majority vote).
    These particles are dropped from the returned DataFrames.
    In practice this affects a very small percentage of particles.

    The analysis functions (`network_analysis`, `calculate_connectivity_metrics`) expect a dict-by-species structure because they are shared with the trait pathway.
    Path A wraps its single community-level result in a one-entry dict under the key `'passive'` to fit this interface.
    """
    # Step 1: Get release positions (obs=0) for every trajectory and snap them to the binned grid.
    release_rows = traj[traj['obs'] == 0][['trajectory', 'time', 'lat', 'lon']].copy()
    if release_rows.empty:
        raise ValueError(
            "No release positions were found in the trajectory data. "
            "The simulation output may be incomplete - please re-run the passive dispersal pathway."
        )

    release_grid_ids = snap_to_binned_grid(
        release_rows['lon'].values,
        release_rows['lat'].values,
        binned_lonarray,
        binned_latarray,
        binned_ocean_grid_df,
    )
    release_rows = release_rows.assign(release_grid_id=release_grid_ids)

    if release_rows.empty:
        raise ValueError(
            "No release positions were recovered from the trajectory data. "
            "Please re-run the passive dispersal pathway."
        )

    n_released = len(release_rows)

    # Step 2: For each timepoint, build a final_processed-style DataFrame.
    results = {}

    for day in timepoints:
        day = int(day)

        # Find the row per trajectory at obs == day
        timepoint_rows = traj[traj['obs'] == day][['trajectory', 'time', 'lat', 'lon']].copy()

        if timepoint_rows.empty:
            print(f"  Day {day}: no particles found at this observation. Skipping.")
            continue

        # Snap settlement positions to the binned grid.
        settlement_grid_ids = snap_to_binned_grid(
            timepoint_rows['lon'].values,
            timepoint_rows['lat'].values,
            binned_lonarray,
            binned_latarray,
            binned_ocean_grid_df,
        )
        timepoint_rows = timepoint_rows.assign(settlement_grid_id=settlement_grid_ids)

        # Merge release positions with timepoint positions on trajectory id.
        merged = release_rows.merge(
            timepoint_rows,
            on='trajectory',
            suffixes=('_rel', '_set'),
        )

        if merged.empty:
            print(f"  Day {day}: no trajectories found. Skipping.")
            continue

        # Build two rows per trajectory: obs=0 (release) and obs=1 (settlement)
        release_rows_df = pd.DataFrame({
            'trajectory': merged['trajectory'].values,
            'obs': 0,
            'grid_id': merged['release_grid_id'].astype(int).values,
            'lat': merged['lat_rel'].values,
            'lon': merged['lon_rel'].values,
            'time': merged['time_rel'].values,
        })
        settlement_rows_df = pd.DataFrame({
            'trajectory': merged['trajectory'].values,
            'obs': 1,
            'grid_id': merged['settlement_grid_id'].astype(int).values,
            'lat': merged['lat_set'].values,
            'lon': merged['lon_set'].values,
            'time': merged['time_set'].values,
        })

        combined = (
            pd.concat([release_rows_df, settlement_rows_df], ignore_index=True)
              .sort_values(['trajectory', 'obs'])
              .reset_index(drop=True)
        )

        # Wrap in the single-entry dict that network_analysis expects.
        final_processed = {'passive': combined}

        label = f"{season_label}_day{day}" if season_label else f"day{day}"
        results[label] = final_processed

        print(f"  Day {day}: {len(merged):,} trajectories")

    return results
# endregion

# region Configuration Display and Metadata
## Prints full setup before running simulation
def display_passive_config_start(config):
    """
    Display the start-of-run configuration summary and ask the user to confirm.

    Prints all auto-selected parameters (domain, resolution, year, seasons, simulation settings) so the user can verify them before committing to a ~30-60 minute run.
    Returns True if the user confirms and False if they abort.

    Parameters
    ----------
    config : dict
        Configuration dictionary built by `passive_dispersal_pathway()`.
        Must contain keys: lat_min, lat_max, lon_min, lon_max, lat_extent, lon_extent, resolution_km, simulation_year, summer_range, winter_range, n_particles_per_cell, runtime_days, dt_minutes, output_dt_hours, kh, dataset_id, timepoints.

    Returns
    -------
    bool
        True if the user confirms, False if the user aborts.
    """
    section_header("Configuration Summary")

    info_box([
        "Study domain:",
        f"  Latitude:  {config['min_lat']:.2f} to {config['max_lat']:.2f}  ({config['lat_extent']:.2f} degrees)",
        f"  Longitude: {config['min_lon']:.2f} to {config['max_lon']:.2f}  ({config['lon_extent']:.2f} degrees)",
        f"  Area:      {config['lat_extent'] * config['lon_extent']:.2f} square degrees",
        "",
        f" Grid resolution (auto-selected by domain size): {config['resolution_km']} km",
        "",
        f"Simulation year: {config['simulation_year']}",
        "",
        "Seasonal windows (28 days each):",
        f"  Summer: {config['summer_range'][0]:%Y-%m-%d} to {config['summer_range'][1]:%Y-%m-%d}",
        f"  Winter: {config['winter_range'][0]:%Y-%m-%d} to {config['winter_range'][1]:%Y-%m-%d}",
        "",
        "Hydrodynamic dataset:",
        f"  {config['dataset_id']}",
        "",
        "Simulation parameters:",
        f"  Particles per cell:      {config['n_particles_per_cell']}",
        f"  Runtime:                 {config['runtime_days']} days",
        f"  Timestep:                {config['dt_minutes']} minutes",
        f"  Output frequency:        {config['output_dt_hours']} hours",
        f"  Horizontal diffusivity:  {config['kh']} m^2/s",
        "",
        "Fixed-timepoint extraction:",
        f"  Days: {', '.join(str(t) for t in config['timepoints'])}",
    ])

    return get_yes_no("\nProceed with these settings?", default='yes')

## Same as above but displays at the end of the run
def display_passive_config_end(config, run_info):
    """
    Display the end-of-run completion summary.

    Shows actual runtime, number of cells and particles, number of datasets produced, and paths to all output files.
    Called at the very end of `passive_dispersal_pathway()`.

    Parameters
    ----------
    config : dict
        Configuration dictionary, updated with project_dirs.
    run_info : dict
        Run information dictionary populated during the pathway execution.
        Must contain keys: total_runtime_seconds, n_binned_ocean_cells, n_particles_per_season, n_datasets, metadata_path.
    """
    section_header("Passive Dispersal Pathway Complete")

    duration_min = run_info['pathway_duration_seconds'] / 60.0

    info_box([
        "Run summary:",
        f"  Total duration:        {duration_min:.1f} minutes",
        f"  Number of ocean cells: {run_info['n_binned_ocean_cells']:,}",
        f"  Particles per season:  {run_info['n_particles_per_season']:,}",
        f"  Datasets produced:     {run_info['n_datasets']}",
        "",
        "Output files:",
        f"  Project folder:  {config['project_dirs']['root']}",
        f"  Simulation data: {config['project_dirs']['simulation']}",
        f"  Results:         {config['project_dirs']['results']}",
        f"  Figures:         {config['project_dirs']['figures']}",
        f"  Metadata:        {run_info['metadata_path']}",
    ])

## Builds the passive pathway run-configuration dict (for the metadata file)
def passive_metadata(config, run_info):
    """
    Build the run-configuration dictionary for the passive pathway.

    Returns a nested dict that captures every auto-selected parameter plus actual runtime and output statistics.
     The dict is written into the 'Run Configuration' sheet of the project's `comcon_outputs.xlsx` metadata file by `write_project_metadata()`.

    Parameters
    ----------
    config : dict
        Configuration dictionary from `passive_dispersal_pathway()`.
    run_info : dict
        Run information dictionary populated during pathway execution.

    Returns
    -------
    dict
        Nested configuration dictionary.
    """
    metadata = {
        'comcon_version': '1.0.0',
        'pathway': 'passive_dispersal',
        'run_timestamp': datetime.now().isoformat(),
        'domain': {
            'min_lat': float(config['min_lat']),
            'max_lat': float(config['max_lat']),
            'min_lon': float(config['min_lon']),
            'max_lon': float(config['max_lon']),
            'lat_extent_deg': float(config['lat_extent']),
            'lon_extent_deg': float(config['lon_extent']),
            'area_sqdeg': float(config['lat_extent'] * config['lon_extent']),
        },
        'grid': {
            'resolution_km': int(config['resolution_km']),
            'n_binned_ocean_cells': int(run_info.get('n_binned_ocean_cells', 0)),
        },
        'simulation_year': int(config['simulation_year']),
        'seasons': {
            'summer': {
                'start': config['summer_range'][0].strftime('%Y-%m-%d'),
                'end': config['summer_range'][1].strftime('%Y-%m-%d'),
            },
            'winter': {
                'start': config['winter_range'][0].strftime('%Y-%m-%d'),
                'end': config['winter_range'][1].strftime('%Y-%m-%d'),
            },
        },
        'simulation_parameters': {
            'dataset_id': config['dataset_id'],
            'n_particles_per_cell': int(config['n_particles_per_cell']),
            'n_particles_per_season': int(run_info.get('n_particles_per_season', 0)),
            'runtime_days': int(config['runtime_days']),
            'dt_minutes': int(config['dt_minutes']),
            'output_dt_hours': int(config['output_dt_hours']),
            'kh_m2_per_s': float(config['kh']),
            'timepoints_days': list(config['timepoints']),
        },
        'output_options': {
            'save_figures': bool(config.get('save_figures', True)),
            'save_results': bool(config.get('save_results', True)),
        },
        'run_info': {
            'pathway_duration_seconds': float(run_info.get('pathway_duration_seconds', 0)),
            'n_datasets_produced': int(run_info.get('n_datasets', 0)),
        },
    }

    return metadata
# endregion

# region Network Analysis Support
## Creates two extra parameters/datasets needed for the network analysis functions from pathway b
def prepare_network_inputs(binned_landmask, binned_lonarray, binned_latarray, binned_ocean_grid_df):
    """
    Build the full grid DataFrame and 2D grid_ids array needed by `network_analysis()`.

    The trait pathway produces these using`setup_release_settlement`, but the passive pathway builds the binned grid separately.
    This helper creates the two extra objects that `network_analysis` expects, using the binned landmask and the binned ocean grid DataFrame already produced by `create_binned_grid()`.

    Parameters
    ----------
    binned_landmask : numpy.ndarray
        2D binary array at binned resolution where 1=land and 0=ocean.
    binned_lonarray : numpy.ndarray
        2D meshgrid of binned cell centroid longitudes.
    binned_latarray : numpy.ndarray
        2D meshgrid of binned cell centroid latitudes.
    binned_ocean_grid_df : pandas.DataFrame
        Binned ocean grid DataFrame from `create_binned_grid()`.

    Returns
    -------
    full_grid_df : pandas.DataFrame
        DataFrame of every binned cell (land + ocean) with columns: longitude, latitude, grid_id, is_land, is_coastal.
        Ocean cells have their grid_id populated from `binned_ocean_grid_df`; land cells have grid_id=0.
    grid_ids_2d : numpy.ndarray
        2D array matching the shape of `binned_landmask`, where each cell holds the binned grid_id of that cell if it is ocean, or 0 if it is land.
    """
    n_rows, n_cols = binned_landmask.shape

    # Build full_grid_df from flat arrays (every cell, land and ocean)
    full_grid_df = pd.DataFrame({
        'longitude': binned_lonarray.ravel(),
        'latitude': binned_latarray.ravel(),
        'grid_id': 0,
        'is_land': binned_landmask.ravel().astype(int),
        'is_coastal': (binned_landmask.ravel() == 0).astype(int),
    })

    # Build 2D grid_ids array and patch ocean grid_ids into full_grid_df
    grid_ids_2d = np.zeros((n_rows, n_cols), dtype=np.int64)
    ocean_rows = binned_ocean_grid_df['row_idx'].values.astype(int)
    ocean_cols = binned_ocean_grid_df['col_idx'].values.astype(int)
    ocean_ids = binned_ocean_grid_df['grid_id'].values.astype(np.int64)
    grid_ids_2d[ocean_rows, ocean_cols] = ocean_ids

    flat_indices = ocean_rows * n_cols + ocean_cols
    full_grid_df.loc[flat_indices, 'grid_id'] = ocean_ids

    return full_grid_df, grid_ids_2d

## Display native and binned land/ocean masks stacked vertically (Path A specific, no coastal layer)
def display_passive_masks(native_landmask, native_lonarray, native_latarray, binned_landmask, binned_lonarray, binned_latarray):
    """
    Display native + binned land/ocean masks for the passive pathway.

    Reuses `plot_masks` from interactive.py for visual consistency with Path B.
    Suppresses the coastal layer (not used in Path A).
    Shows native resolution first, then binned.

    Parameters
    ----------
    native_landmask : np.ndarray
        Native resolution land mask (1 = land, 0 = ocean).
    binned_landmask : np.ndarray
        Binned resolution land mask (1 = land, 0 = ocean).
    """
    fig_native = plot_masks(
        native_landmask, None, native_lonarray, native_latarray,
        geojson_file_path=None,
        show_coastal=False,
        show_save_prompt=False,
        title=f'Native resolution ({native_landmask.shape[0]} x {native_landmask.shape[1]} cells)',
    )

    fig_binned = plot_masks(
        binned_landmask, None, binned_lonarray, binned_latarray,
        geojson_file_path=None,
        show_coastal=False,
        show_save_prompt=False,
        title=f'Binned resolution ({binned_landmask.shape[0]} x {binned_landmask.shape[1]} cells)',
    )

    return fig_native, fig_binned

## Allows for the display of the different seasons and timepoints for the results and figures
## Builds the dataset dropdown used by all four figure viewers
def make_dataset_dropdown(dataset_keys, default_key):
    """
    Build an ipywidgets Dropdown for selecting between datasets.

    All four figure viewers below share the same dropdown styling.
    This helper function keeps them consistent without duplicating the widget construction each time.

    Parameters
    ----------
    dataset_keys : iterable of str
        Dataset labels to show in the dropdown (e.g. 'summer_day7', 'pooled_all').
    default_key : str
        The dataset to show when the dropdown first appears.

    Returns
    -------
    ipywidgets.Dropdown
        Configured dataset dropdown widget.
    """
    return widgets.Dropdown(
        options=sorted(dataset_keys),
        value=default_key,
        description='Dataset:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='400px'),
    )


## Chooses the default dataset to display when a viewer first appears
def pick_default_key(network_results_by_dataset):
    """
    Choose the dataset that should be shown by default in each figure viewer.

    Returns 'pooled_all' if available (this is the main view across all seasons and timepoints).
    Otherwise, falls back to the first key in the dict.

    Parameters
    ----------
    network_results_by_dataset : dict
        Mapping from dataset label to network_results dict.

    Returns
    -------
    str
        Dataset label to select initially.
    """
    if 'pooled_all' in network_results_by_dataset:
        return 'pooled_all'
    return next(iter(network_results_by_dataset))

## Interactive viewer for the connectivity matrix with dataset dropdown
def matrix_viewer(network_results_by_dataset):
    """
    Interactive viewer for the connectivity matrix, with a dataset dropdown.

    Shows a heatmap of source-to-sink connection counts for the currently selected dataset.
    Users can switch between datasets (summer_day7, winter_pooled, pooled_all, etc.) using the dropdown.

    Parameters
    ----------
    network_results_by_dataset : dict
        Mapping from dataset label to a network_results dict produced by `network_analysis()`.
    """
    if not network_results_by_dataset:
        return

    default_key = pick_default_key(network_results_by_dataset)
    dataset_dropdown = make_dataset_dropdown(network_results_by_dataset.keys(), default_key)
    output = widgets.Output()

    def update(*_):
        dataset_label = dataset_dropdown.value
        network_results = network_results_by_dataset[dataset_label]
        with output:
            clear_output(wait=True)
            print(f"  Rendering {dataset_label}: connectivity matrix... (~10-20 seconds)")
            try:
                # Capture the return value directly. create_matrix_with_map calls
                # plt.close(fig) internally when show_figure=False, so plt.gcf()
                # afterwards returns a fresh empty figure, not the one we just
                # built — that's why the matrix wasn't appearing in the viewer.
                fig = create_matrix_with_map(
                    network_results['connectivity_matrix'],
                    network_results['site_metrics'],
                    title=dataset_label,
                    show_figure=False,
                )
                clear_output(wait=True)
                display(fig)
                plt.close(fig)
            except Exception as e:
                print(f"  Could not display matrix: {e}")

    dataset_dropdown.observe(update, names='value')

    display(dataset_dropdown, output)
    # Delay initial render
    threading.Timer(0.5, update).start()

## Interactive viewer for the Louvain community map with dataset dropdown
def community_viewer(network_results_by_dataset):
    """
    Interactive viewer for the Louvain community map, with a dataset dropdown.

    Shows sites coloured by Louvain community membership for the currently selected dataset.
    Above the map, prints the network-level summary stats (density, modularity, n communities) for context.
    Users can switch between datasets using the dropdown.

    Parameters
    ----------
    network_results_by_dataset : dict
        Mapping from dataset label to a network_results dict produced by `network_analysis()`.
    """
    if not network_results_by_dataset:
        return

    default_key = pick_default_key(network_results_by_dataset)
    dataset_dropdown = make_dataset_dropdown(network_results_by_dataset.keys(), default_key)
    output = widgets.Output()

    def update(*_):
        dataset_label = dataset_dropdown.value
        network_results = network_results_by_dataset[dataset_label]
        with output:
            clear_output(wait=True)
            print(f"  Rendering {dataset_label}: Louvain communities... (~10-20 seconds)")
            try:
                # create_community_map returns (fig, partition) — capture properly.
                # Same reason as the matrix viewer: plt.gcf() after plt.close() is
                # unreliable. Passing show_figure=False suppresses the internal
                # plt.show() so we don't render twice inside the widget.
                result = create_community_map(
                    network_results,
                    title=dataset_label,
                    show_figure=False,
                )
                fig = result[0] if isinstance(result, tuple) else result
                clear_output(wait=True)
                display(fig)
                plt.close(fig)
                # Network stats table below the map
                stats_rows = []
                for label, key in [
                    ('Nodes', 'network_nodes'),
                    ('Edges', 'network_edges'),
                    ('Density', 'network_density'),
                    ('Communities', 'network_communities'),
                    ('Modularity', 'network_modularity'),
                ]:
                    v = network_results.get(key)
                    if v is None:
                        continue
                    if key in ('network_density', 'network_modularity'):
                        stats_rows.append((label, f"{float(v):.3f}"))
                    else:
                        stats_rows.append((label, f"{int(v)}"))
                if stats_rows:
                    stats_df = pd.DataFrame(stats_rows, columns=['Network statistic', 'Value'])
                    stats_df.index = range(1, len(stats_df) + 1)
                    # Extra vertical space so the table sits further below the map
                    print()
                    print()
                    display(stats_df)
            except Exception as e:
                print(f"  Could not display community map: {e}")

    dataset_dropdown.observe(update, names='value')

    display(dataset_dropdown, output)
    update()

## Interactive viewer for the full site × metric heatmap (Giada's layout)
def build_heatmap_figure(network_results, title=None):
    """
    Build the site x metric heatmap with a reference map panel stacked below,
    matching the style/size of the reference map in create_matrix_with_map.

    The reference map shows only the sites that appear on the heatmap x-axis
    (top 100 by overall composite), with the top-ranked sites labelled.

    Returns the matplotlib Figure, or None if the required rank columns are
    not present in network_results['site_metrics'].
    """
    site_metrics = network_results['site_metrics']
    metric_cols = ['source', 'sink', 'self_recruitment',
                   'betweenness', 'closeness', 'degree']
    metric_labels = ['Source', 'Sink', 'Self-recruit.',
                     'Betweenness', 'Closeness', 'Degree']

    rank_cols = [f'{m}_rank' for m in metric_cols]
    available = [c for c in rank_cols if c in site_metrics.columns]
    if not available:
        return None

    # Display top 100 sites based on overall_composite
    # Sort by grid_id on the x-axis
    n_sites = len(site_metrics)
    cap = min(100, n_sites)

    if 'overall_composite' in site_metrics.columns:
        selected = site_metrics.sort_values('overall_composite').iloc[:cap].copy()
    else:
        selected = site_metrics.iloc[:cap].copy()

    # Now re-sort the selected subset by grid_id numerically for display
    selected['__gid_int'] = selected['grid_id'].astype(int)
    top_sites = selected.sort_values('__gid_int').drop(columns='__gid_int')

    # Metrics as rows, sites as columns (wide heatmap)
    rank_matrix_t = top_sites[available].values.astype(float).T
    site_labels = top_sites['site'].astype(str).values

    # Standardised figure size. Taller heatmap panel so cells are more rectangular (not squished).
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 1, height_ratios=[1.2, 1.3], hspace=0.55)

    # Top: heatmap — aspect='auto' stretches cells to fill, but the taller panel makes them less squished
    ax_hm = fig.add_subplot(gs[0])
    im = ax_hm.imshow(rank_matrix_t, cmap='viridis_r', aspect='auto')

    ax_hm.set_yticks(range(len(available)))
    ax_hm.set_yticklabels(
        [metric_labels[metric_cols.index(c.replace('_rank', ''))] for c in available],
        fontsize=11,
    )
    ax_hm.set_xticks(range(cap))
    ax_hm.set_xticklabels(site_labels, fontsize=7, rotation=90)

    # Horizontal colorbar below the heatmap — full width to match the heatmap
    cbar = plt.colorbar(im, ax=ax_hm, orientation='horizontal',
                        shrink=0.95, pad=0.25, aspect=50)
    cbar.set_label('Rank (1 = best)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    heatmap_title = title or "Site x metric heatmap"
    ax_hm.set_title(heatmap_title, fontsize=13, fontweight='bold', pad=10)
    # Caption sits below the colorbar
    if n_sites > cap:
        ax_hm.text(
            0.5, -0.95,
            f"Showing top {cap} of {n_sites} sites (selected by overall composite rank, displayed in grid_id order). "
            f"Full rankings exported in per-dataset network_site_metrics.csv.",
            transform=ax_hm.transAxes, ha='center', va='top',
            fontsize=10, style='italic', wrap=True,
        )

    # Bottom: reference map for the same sites as the heatmap
    ax_map = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())

    heatmap_site_ids = set(top_sites['grid_id'].astype(int))
    map_sites = site_metrics[site_metrics['grid_id'].astype(int).isin(heatmap_site_ids)].copy()
    if len(map_sites) == 0:
        map_sites = top_sites.copy()

    buffer = 0.2
    ax_map.set_extent([
        map_sites['longitude'].min() - buffer,
        map_sites['longitude'].max() + buffer,
        map_sites['latitude'].min() - buffer,
        map_sites['latitude'].max() + buffer
    ], crs=ccrs.PlateCarree())

    ax_map.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
    ax_map.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5, color='#555555')

    ax_map.scatter(
        map_sites['longitude'], map_sites['latitude'],
        c='#7b9ec9', s=55,
        transform=ccrs.PlateCarree(),
        edgecolors='#2a4a6b', linewidth=0.4,
        zorder=3,
    )

    # Label a subset of the sites that actually appear on the x-axis
    label_step = max(1, cap // 20)  # ~20 labels maximum
    axis_grid_ids = top_sites['grid_id'].astype(int).tolist()
    label_ids = set(axis_grid_ids[::label_step])
    for _, site in map_sites.iterrows():
        gid_int = int(site['grid_id'])
        if gid_int in label_ids:
            ax_map.text(
                site['longitude'], site['latitude'],
                str(gid_int), fontsize=6, ha='left', va='bottom',
                transform=ccrs.PlateCarree(), zorder=4,
                bbox=dict(boxstyle='round,pad=0.1',
                          facecolor='white', alpha=0.85, edgecolor='none'),
            )

    ax_map.set_title('Site Reference Map', fontsize=13, fontweight='bold', pad=10)
    gl = ax_map.gridlines(draw_labels=True, linestyle=':', linewidth=0.5, color='gray', alpha=0.4)
    gl.top_labels = False
    gl.right_labels = False

    return fig


def heatmap_viewer(network_results_by_dataset):
    """
    Display a site-by-metric heatmap for the currently selected dataset, with
    a reference map below matching the heatmap's sites.

    Rows are metrics (source, sink, self-recruitment, betweenness, closeness,
    degree), columns are sites (top 100 by overall composite rank). Cells are
    coloured by rank using viridis_r (rank 1 = best = yellow).

    Parameters
    ----------
    network_results_by_dataset : dict
        Mapping from dataset label to a network_results dict.
    """
    if not network_results_by_dataset:
        return

    default_key = pick_default_key(network_results_by_dataset)
    dataset_dropdown = make_dataset_dropdown(network_results_by_dataset.keys(), default_key)
    output = widgets.Output()

    def update(*_):
        dataset_label = dataset_dropdown.value
        network_results = network_results_by_dataset[dataset_label]
        with output:
            # CRITICAL: every line below stays inside `with output:` (was the bug —
            # previously indentation broke here so display/clear_output targeted
            # the whole cell, wiping every other viewer when the animation rendered)
            clear_output(wait=True)
            print(f"  Rendering {dataset_label}: site x metric heatmap...")
            try:
                fig = build_heatmap_figure(
                    network_results,
                    title=dataset_label,
                )
                if fig is None:
                    clear_output(wait=True)
                    print(f"  No rank columns available for {dataset_label}.")
                    return
                clear_output(wait=True)
                display(fig)
                plt.close(fig)
            except Exception as e:
                print(f"  Could not display heatmap: {e}")

    dataset_dropdown.observe(update, names='value')
    print("  Sites (columns) x metrics (rows), coloured by rank.")
    print("  Top 100 sites by overall composite are shown, displayed left-to-right in grid_id order.")
    print("  The reference map below shows where the displayed sites are located.\n")
    display(dataset_dropdown, output)
    # Delay initial render slightly
    threading.Timer(0.5, update).start()

## Interactive viewer for hotspot metric maps with dataset and metric dropdowns
def metric_viewer(network_results_by_dataset):
    """
    Interactive viewer for network metric hotspot maps, with dataset and metric dropdowns.

    Shows a hotspot map coloured by the selected network metric (source strength, betweenness, etc.) for the currently selected dataset.
    This is the only viewer with two dropdowns because metrics are a natural second dimension for hotspot exploration.

    Parameters
    ----------
    network_results_by_dataset : dict
        Mapping from dataset label to a network_results dict produced by `network_analysis()`.
    """
    if not network_results_by_dataset:
        return

    default_key = pick_default_key(network_results_by_dataset)

    metric_options = {
        'Network only (no metric)': None,
        'Overall composite': 'overall_composite',
        'Functional roles': 'functional_composite',
        'Centrality measures': 'centrality_composite',
        'Source strength': 'source',
        'Sink strength': 'sink',
        'Self-recruitment': 'self_recruitment',
        'Betweenness centrality': 'betweenness',
        'Closeness centrality': 'closeness',
        'Degree centrality': 'degree',
    }

    dataset_dropdown = make_dataset_dropdown(network_results_by_dataset.keys(), default_key)
    metric_dropdown = widgets.Dropdown(
        options=list(metric_options.keys()),
        value='Overall composite',
        description='Metric:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='400px'),
    )
    output = widgets.Output()

    def update(*_):
        dataset_label = dataset_dropdown.value
        metric_name = metric_dropdown.value
        col = metric_options[metric_name]
        network_results = network_results_by_dataset[dataset_label]
        site_metrics = network_results['site_metrics']

        network_only = col is None

        if network_only:
            # No metric colouring
            score_col = 'overall_composite'
            top10 = site_metrics.head(0)  # empty, no top 10 drawn
            cmap = 'viridis_r'
        elif col.endswith('_composite'):
            score_col = col
            top10_key = 'top10_' + col.replace('_composite', '')
            top10 = network_results.get(top10_key, network_results.get('top10_overall'))
            cmap = 'viridis_r'
        else:
            rank_col = f'{col}_rank'
            score_col = rank_col if rank_col in site_metrics.columns else col
            if col in site_metrics.columns:
                top10 = site_metrics.nlargest(10, col).copy()
            else:
                top10 = network_results.get('top10_overall')
            cmap = 'viridis_r'

        with output:
            clear_output(wait=True)
            print(f"  Rendering {dataset_label}: {metric_name}... (~10-20 seconds)")
            try:
                # Network-only mode shows ALL edges so natural weight variation
                # reads through the alpha channel. Metric mode caps at 200 edges
                # so the metric colour stays the dominant signal on the map.
                top_n = None if network_only else 200
                fig = create_network_map(
                    site_metrics, top10,
                    title=dataset_label,
                    composite_col=score_col,
                    cmap=cmap,
                    show_figure=False,
                    network_graph=network_results.get('network_graph'),
                    top_n_edges=top_n,
                    network_only=network_only,
                )
                clear_output(wait=True)
                display(fig)
                plt.close(fig)
                # Top-10 table below the figure
                if not network_only and len(top10) > 0:
                    table_rows = []
                    for idx, (_, site) in enumerate(top10.iterrows()):
                        table_rows.append({
                            '#': idx + 1,
                            'Grid ID': int(site['grid_id']),
                            'Latitude': round(site['latitude'], 2),
                            'Longitude': round(site['longitude'], 2),
                            'Rank': round(site[score_col], 1),
                        })
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

    dataset_dropdown.observe(update, names='value')
    metric_dropdown.observe(update, names='value')

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
        "The arrows show the underlying network and depend only on the dataset selected, not the metric.",
        "Switching between metrics for the same dataset changes node colours and ranks, not the arrows.",
        "Switching to a different dataset (e.g. from summer_day7 to winter_day28) will change both.",
    ])
    print()
    display(dataset_dropdown, metric_dropdown, output)
    update()

## Interactive viewer for the dispersal animation with season dropdown
def animation_viewer(trajectory_paths_by_season):
    """
    Display the particle dispersal animation with a season dropdown.

    Shows the animated particle positions over time for the selected season.
    Trajectory data is lazy-loaded from the parquet files on disk each time the user switches seasons.
    This way, so no multi-GB trajectory DataFrame is held in memory between selections.

    Parameters
    ----------
    trajectory_paths_by_season : dict
        Mapping from season label ('summer', 'winter') to the parquet file path containing the raw trajectory data for that season.
    """
    # Lift the matplotlib HTML embed limit so larger animations don't get truncated
    _mpl.rcParams['animation.embed_limit'] = 100.0
    if not trajectory_paths_by_season:
        return

    # Exclude sentinel keys (start with underscore) from the season dropdown options.
    season_keys = sorted(k for k in trajectory_paths_by_season.keys() if not k.startswith('_'))
    if not season_keys:
        return

    season_dropdown = widgets.Dropdown(
        options=season_keys,
        value='summer' if 'summer' in season_keys else season_keys[0],
        description='Season:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='400px'),
    )
    output = widgets.Output()

    def update(*_):
        season_label = season_dropdown.value
        parquet_path = trajectory_paths_by_season[season_label]
        with output:
            clear_output(wait=True)
            print(f"  Loading {season_label} trajectory data...")
            try:
                traj = pd.read_parquet(parquet_path)
                clear_output(wait=True)
                print(f"  Rendering {season_label} animation... (this may take a minute)")
                # Build the animation object WITHOUT letting dispersal_animation display it.
                # That way the "Rendering..." message can be cleared before the HTML5 player appears.
                anim = dispersal_animation(traj, output_path=None, show_in_notebook=False)
                clear_output(wait=True)
                if anim is not None:
                    display(IPHTML(anim.to_jshtml()))
                del traj
                gc.collect()
            except Exception as e:
                print(f"  Could not display animation: {e}")

    season_dropdown.observe(update, names='value')

    display(season_dropdown, output)
    # Trigger initial render so the user sees the default season's animation without clicking
    update()

## Interactive viewer for the Type 1 connectivity metric summary per dataset
def summary_table_viewer(metrics_by_dataset):
    """
    Display per-dataset Type 1 connectivity metric summaries with a dropdown.

    Parameters
    ----------
    metrics_by_dataset : dict
        Mapping from dataset label to the result dict returned by `calculate_connectivity_metrics()`.
        Entries may be `None` if the metric computation failed for that dataset.
    """
    valid_keys = [k for k, v in metrics_by_dataset.items() if v is not None]
    if not valid_keys:
        print("  No Type 1 metrics available to display.")
        return

    default_key = 'pooled_all' if 'pooled_all' in valid_keys else valid_keys[0]
    dataset_dropdown = make_dataset_dropdown(valid_keys, default_key)
    output = widgets.Output()

    def _flatten(metrics):
        """Build clean, user-facing (label, value) rows from the metrics dict.

        Path A specific: settlement rate is always 100% (no settlement logic)
        and realised PLD equals the extraction timepoint, so both are hidden.
        Only the rows that carry genuine information about Path A output are
        retained.
        """
        community = metrics.get('community', {}) if isinstance(metrics.get('community'), dict) else {}

        def _fmt_int(v):
            return f"{int(v):,}" if v is not None else '-'

        def _fmt_pct(v):
            return f"{v * 100:.1f}%" if v is not None else '-'

        def _fmt_num(v, units=''):
            return f"{v:.2f} {units}".strip() if v is not None else '-'

        rows = [
            ('Total particles released', _fmt_int(community.get('total_released'))),
            ('Self-recruitment rate', _fmt_pct(community.get('community_recruitment'))),
            ('Min dispersal distance', _fmt_num(community.get('min_distance_km'), 'km')),
            ('Mean dispersal distance', _fmt_num(community.get('mean_distance_km'), 'km')),
            ('Max dispersal distance', _fmt_num(community.get('max_distance_km'), 'km')),
        ]
        return rows

    def update(*_):
        dataset_label = dataset_dropdown.value
        metrics = metrics_by_dataset.get(dataset_label)
        with output:
            clear_output(wait=True)
            print(f"  Loading {dataset_label}...")
            if metrics is None:
                clear_output(wait=True)
                print(f"  No Type 1 metrics available for {dataset_label}.")
                return
            clear_output(wait=True)

            try:
                rows = _flatten(metrics)
                if not rows:
                    print(f"  No displayable metrics for {dataset_label}.")
                    return
                summary_df = pd.DataFrame(rows, columns=['Metric', 'Value'])
                display(summary_df)
            except Exception as e:
                print(f"  Could not display summary: {e}")

    dataset_dropdown.observe(update, names='value')
    print("  Use the dropdown to switch between datasets.\n")
    display(dataset_dropdown, output)
    update()

## Pools all the seasons/timepoints of the passive pathway together for a complete 'community' assessment
def pool_final_processed(list_of_final_processed):
    """
    Pool multiple `final_processed`-style dicts into a single dict.

    Used by the passive pathway to pool datasets across seasons or across timepoints for a combined connectivity view.
    Trajectory IDs from each input dict are offset so that no two particles end up with the same ID in the pooled DataFrame.

    Parameters
    ----------
    list_of_final_processed : list of dict
        Each element is a `{'passive': DataFrame}` dict from `extract_fixed_timepoints()`.

    Returns
    -------
    dict
        A single `{'passive': DataFrame}` dict containing all rows from all inputs with unique trajectory IDs.
    """
    dfs = []
    traj_offset = 0
    for fp in list_of_final_processed:
        if 'passive' not in fp or fp['passive'].empty:
            continue
        df = fp['passive'].copy()
        df['trajectory'] = df['trajectory'].values + traj_offset
        traj_offset = int(df['trajectory'].max()) + 1
        dfs.append(df)

    if not dfs:
        return {'passive': pd.DataFrame()}

    combined = pd.concat(dfs, ignore_index=True)
    return {'passive': combined}
# endregion

# region Main Wrapper Function
def passive_dispersal_pathway(simulation_year=None):
    """
    Runs the complete passive dispersal pathway end-to-end.

    This is the main function for users to use for Path A.
    The pathway runs a hydrodynamic connectivity overview following James et al. (2025):
    Particles are released from all binned ocean cells in the study domain, advected through two 28-day simulations (summer and winter).
    Particle positions are extracted at four fixed timepoints (7, 14, 21, 28 days).
    Network analyses are computed automatically and saved to the project folder.

    User input is minimal: latitude and longitude bounds, a project folder name and two yes/no prompts for automatic figure and result saving.
    All other parameters are set automatically based on the domain size and the most recent available Copernicus data.

    Parameters
    ----------
    simulation_year : int, optional
        Override the auto-detected simulation year.
        By default (None) the function queries Copernicus metadata to find the most recent year for which both February and July are fully published.
        A user can pass a specific year to bypass this check.

    Returns
    -------
    dict or None
        Configuration dictionary if the run completes successfully.
        None if the user aborts at the confirmation prompt or if setup fails.
    """
    section_header("Passive Dispersal Pathway")
    info_box([
        "This pathway runs a quick hydrodynamic connectivity overview following James et al. (2025).",
        "",
        "Hydrodynamic data is sourced from the Copernicus Marine Service:",
        "  https://data.marine.copernicus.eu/products",
        "",
        "You will be asked for your study domain (lat/lon bounds) and a project folder name.",
        "All other settings are configured automatically.",
    ])

    pathway_start_time = time.time()

    # Step 1: Project directory setup
    subsection_header("Project Folder")
    project_dirs = setup_project_directory(
        subfolders=['config', 'simulation', 'figures', 'results']
    )
    if project_dirs is None:
        print("\nPassive dispersal pathway cancelled: no project folder created.")
        return None

    # Step 2: Copernicus credentials
    subsection_header("Copernicus Marine Service")
    info_box([
        "Hydrodynamic data will be downloaded from the Copernicus Marine Service.",
        "",
        "If this is your first time, you will be prompted for your Copernicus credentials now.",
        "You can create a free account at:",
        "  https://data.marine.copernicus.eu/register",
        "",
        "Credentials are saved for future sessions, so you will not be prompted again.",
    ])
    cmems_username, cmems_password = get_copernicus_credentials()

    # Step 3: Prompt for lat/lon bounds and validate
    subsection_header("Spatial Extent")
    info_box([
        "Define the extent of your area of interest.",
        "",
        "Coordinates can be entered in decimal degrees or DMS format.",
        "Examples: 48.5, -3.456, 48°30'26.6\"N, 3°27'21.6\"W",
    ])
    while True:
        try:
            min_lat = parse_coordinate(input("Enter the minimum latitude: "))
            max_lat = parse_coordinate(input("Enter the maximum latitude: "))
            if min_lat >= max_lat:
                print("Error: Minimum latitude must be less than maximum latitude.")
                continue
            if not (-90 <= min_lat <= 90) or not (-90 <= max_lat <= 90):
                print("Error: Latitude must be between -90 and 90.")
                continue
            break
        except ValueError:
            print("Error: Please enter valid numbers for latitude (decimal or DMS).")
    print()
    while True:
        try:
            min_lon = parse_coordinate(input("Enter the minimum longitude: "))
            max_lon = parse_coordinate(input("Enter the maximum longitude: "))
            if min_lon >= max_lon:
                print("Error: Minimum longitude must be less than maximum longitude.")
                continue
            if not (-180 <= min_lon <= 180) or not (-180 <= max_lon <= 180):
                print("Error: Longitude must be between -180 and 180.")
                continue
            break
        except ValueError:
            print("Error: Please enter valid numbers for longitude (decimal or DMS).")
    print(f"\n  Spatial extent: {min_lat}°N to {max_lat}°N, {min_lon}°W to {max_lon}°E")

    # Validate the combined domain (catches any remaining issues and emits size warning)
    try:
        lat_extent, lon_extent = validate_domain(min_lat, max_lat, min_lon, max_lon)
    except ValueError as e:
        print(f"\n  Error: {e}")
        print("  Please re-run passive_dispersal_pathway() with valid bounds.")
        return None

    # Step 2: Auto-compute grid resolution
    resolution_km = calculate_resolution(lat_extent, lon_extent)

    # Auto-select simulation year unless the user has passed an override.
    # The Copernicus global NEMO reanalysis product lags real time by 12-18 months, so using current_year - 2 reliably produces a year with Feb and Jul fully published.
    if simulation_year is None:
        simulation_year = datetime.now().year - 2

    # Step 4: Build the configuration dictionary
    summer_start = datetime(simulation_year, 7, 1)
    summer_end = datetime(simulation_year, 7, 28)
    winter_start = datetime(simulation_year, 2, 1)
    winter_end = datetime(simulation_year, 2, 28)

    config = {
        'min_lat': min_lat,
        'max_lat': max_lat,
        'min_lon': min_lon,
        'max_lon': max_lon,
        'lat_extent': lat_extent,
        'lon_extent': lon_extent,
        'resolution_km': resolution_km,
        'simulation_year': simulation_year,
        'summer_range': (summer_start, summer_end),
        'winter_range': (winter_start, winter_end),
        'n_particles_per_cell': 100,
        'runtime_days': 28,
        'dt_minutes': 60,
        'output_dt_hours': 24,
        'timepoints': (7, 14, 21, 28),
        'dataset_id': 'cmems_mod_glo_phy_my_0.083deg_P1D-m',
        'kh': 1,
    }

    # Step 5: Display configuration and ask the user to confirm
    if not display_passive_config_start(config):
        print("\nPassive dispersal pathway cancelled.")
        print("Re-run passive_dispersal_pathway() with different parameters if needed.")
        return None

    # Step 6: Run each season end-to-end (the simulation)
    # Save options prompts
    subsection_header("Output Options")
    save_figures = get_yes_no("Save all figures automatically?", default='yes')
    save_results = get_yes_no("Save all results tables automatically?", default='yes')
    config['save_figures'] = save_figures
    config['save_results'] = save_results
    # Self-recruitment is always excluded from source/sink/centrality metrics
    # in Path A. The dedicated self_recruitment metric is still reported,
    # so no information is lost — we just avoid double-counting self-loops.
    config['exclude_self_recruitment'] = True
    config['project_dirs'] = project_dirs
    config['_cmems_username'] = cmems_username
    config['_cmems_password'] = cmems_password

    # Native masks and binned grid are computed once (identical across seasons)
    native_landmask = None
    native_coastalmask = None
    native_lonarray = None
    native_latarray = None
    binned_landmask = None
    binned_lonarray = None
    binned_latarray = None
    binned_ocean_grid_df = None
    release_grid = None
    full_grid_df_path_a = None
    grid_ids_2d = None
    n_particles_per_season = 0

    all_extractions = {}
    all_trajectory_paths = {}  # keep raw traj per season for the animation viewer

    for season_name, (season_start, season_end) in [
        ('summer', config['summer_range']),
        ('winter', config['winter_range']),
    ]:
        section_header(f"{season_name.capitalize()} simulation "
                       f"({season_start:%Y-%m-%d} to {season_end:%Y-%m-%d})")

        # Step a: Download hydrodynamic data for this season
        subsection_header(f"Downloading {season_name} hydrodynamic data")
        try:
            hydro_data = download_hydrodynamic_data(
                username=cmems_username,
                password=cmems_password,
                dataset_id=config['dataset_id'],
                longitude_range=[config['min_lon'], config['max_lon']],
                latitude_range=[config['min_lat'], config['max_lat']],
                time_range=[season_start.strftime('%Y-%m-%d'),
                            season_end.strftime('%Y-%m-%d')],
                output_path=None,
                export=False,
            )
        except Exception as e:
            print(f"\n  Error: Failed to download {season_name} hydrodynamic data.")
            print(f"  {e}")
            print("  Please check your internet connection and re-run passive_dispersal_pathway().")
            return None

        # Step b: Masks and binned grid (first season only, reused for winter)
        if native_landmask is None:
            native_landmask, native_coastalmask, native_lonarray, native_latarray = create_masks(
                hydro_data, geojson_file_path=None, show_map=False
            )

            subsection_header("Binning grid to target resolution")
            binned_landmask, binned_lonarray, binned_latarray = bin_mask(
                native_landmask, native_lonarray, native_latarray,
                target_km=config['resolution_km'],
            )
            binned_ocean_grid_df = create_binned_grid(
                binned_landmask, binned_lonarray, binned_latarray
            )

            n_native_ocean = int((native_landmask == 0).sum())
            data_box("Grid summary", [
                f"Native grid:        {native_landmask.shape[0]} x {native_landmask.shape[1]} cells",
                f"Native ocean cells: {n_native_ocean:,}",
                f"Binned grid:        {binned_landmask.shape[0]} x {binned_landmask.shape[1]} cells",
                f"Binned ocean cells: {len(binned_ocean_grid_df):,}",
            ])

            # Show mask(s): one if native resolution was used, two if binned
            _actually_binned = binned_landmask.shape != native_landmask.shape
            subsection_header("Visualising the Hydrodynamic Extent")
            print("  Note: the same grid is used for both summer and winter simulations.\n")
            try:
                if _actually_binned:
                    fig_native, fig_binned = display_passive_masks(
                        native_landmask, native_lonarray, native_latarray,
                        binned_landmask, binned_lonarray, binned_latarray,
                    )
                    with contextlib.redirect_stdout(io.StringIO()):
                        try:
                            save_figure(fig_native, os.path.join(project_dirs['figures'], 'landmask_native'))
                        except Exception:
                            pass
                        try:
                            save_figure(fig_binned, os.path.join(project_dirs['figures'], 'landmask_binned'))
                        except Exception:
                            pass
                    plt.close(fig_native)
                    plt.close(fig_binned)
                else:
                    fig_native = plot_masks(
                        native_landmask, None, native_lonarray, native_latarray,
                        geojson_file_path=None, show_coastal=False, show_save_prompt=False,
                        title=f'Native resolution ({native_landmask.shape[0]} x {native_landmask.shape[1]} cells)',
                    )
                    with contextlib.redirect_stdout(io.StringIO()):
                        try:
                            save_figure(fig_native, os.path.join(project_dirs['figures'], 'landmask_native'))
                        except Exception:
                            pass
                    plt.close(fig_native)
            except Exception as e:
                print(f"  Note: could not display hydrodynamic masks ({e}).")

            release_grid = create_release_grid(binned_ocean_grid_df)
            full_grid_df_path_a, grid_ids_2d = prepare_network_inputs(
                binned_landmask, binned_lonarray, binned_latarray, binned_ocean_grid_df
            )

            n_particles_per_season = len(binned_ocean_grid_df) * config['n_particles_per_cell']

        # Step c: Build fieldset (per-season, using this season's hydro data)
        subsection_header(f"Creating {season_name} fieldset")
        native_grid_ids = np.arange(native_lonarray.size).reshape(native_lonarray.shape)
        fieldset = create_fieldset(hydro_data, native_landmask, native_grid_ids)

        # Step d: Create particle set
        subsection_header(f"Creating {season_name} particle set")
        pset = create_passive_pset(
            fieldset,
            release_grid,
            release_time=season_start,
            n_particles_per_cell=config['n_particles_per_cell'],
            kh=config['kh'],
        )

        # Step e: Build kernels
        kernels = create_custom_kernels(pset)


        # Step f: Execute simulation
        sim_output_path = os.path.join(project_dirs['simulation'], f'passive_{season_name}.zarr')
        sim_output_path, traj = execute_passive_simulation(
            pset, kernels,
            runtime_days=config['runtime_days'],
            output_path=sim_output_path,
            dt_minutes=config['dt_minutes'],
            output_dt_hours=config['output_dt_hours'],
        )

        if sim_output_path is None or traj is None:
            print(f"\n  {season_name.capitalize()} simulation did not complete. Aborting pathway.")
            return None

        # Step j: Extract fixed timepoints
        subsection_header(f"Extracting {season_name} timepoints")
        extractions = extract_fixed_timepoints(
            traj, binned_ocean_grid_df, binned_lonarray, binned_latarray,
            timepoints=config['timepoints'],
            season_label=season_name,
        )
        all_extractions.update(extractions)

        # Stash the parquet path for lazy-loading by the animation viewer.
        # The parquet is the post-processed (chunked) trajectory written by execute_passive_simulation
        # The original .zarr stays in /simulation as the canonical raw data.
        traj_parquet_path = sim_output_path.replace('.zarr', '_processed.parquet')
        if os.path.exists(traj_parquet_path):
            all_trajectory_paths[season_name] = traj_parquet_path
        del hydro_data, fieldset, pset, traj
        gc.collect()

    # Step 7: Build (+ pool) the final datasets users will actually view
    section_header("Preparing Datasets")

    final_datasets = {}

    # Per-season per-timepoint (raw extractions, 8 datasets)
    for label, fp in all_extractions.items():
        final_datasets[label] = fp

    # Per-timepoint pooled across both seasons (4 datasets)
    for day in config['timepoints']:
        summer_fp = all_extractions.get(f'summer_day{day}')
        winter_fp = all_extractions.get(f'winter_day{day}')
        to_pool = [fp for fp in (summer_fp, winter_fp) if fp is not None]
        if to_pool:
            final_datasets[f'pooled_day{day}'] = pool_final_processed(to_pool)

    # Per-season pooled across all timepoints (2 datasets)
    for season in ('summer', 'winter'):
        season_fps = [all_extractions[f'{season}_day{d}'] for d in config['timepoints']
                      if f'{season}_day{d}' in all_extractions]
        if season_fps:
            final_datasets[f'{season}_pooled'] = pool_final_processed(season_fps)

    # Fully pooled (default view, 1 dataset)
    all_fps = list(all_extractions.values())
    if all_fps:
        final_datasets['pooled_all'] = pool_final_processed(all_fps)

    print(f"  Datasets prepared: {len(final_datasets)}")

    # Step 7: Run metrics + network analysis on each dataset
    # Placeholder values to reuse function from Path B
    passive_traits = [{
        'species_name': 'passive',
        'spawning_start': '01/01/1900',
        'spawning_end': '31/12/2099',
        'min_PLD': 0,
        'max_PLD': int(config['runtime_days']),
    }]

    subsection_header("Processing all datasets")
    print(f"  Processing {len(final_datasets)} datasets. This is the slowest step.\n")

    network_results_by_dataset = {}
    metrics_by_dataset = {}
    timing_log = {}

    # Process non-default datasets first (silently), then default last (with full output).
    # This way the user's last visible analysis output is the most informative one.
    dataset_order = sorted(
        final_datasets.keys(),
        key=lambda k: (k == 'pooled_all', k),  # pooled_all sorts last
    )

    for dataset_label in tqdm(dataset_order, desc="Analysing datasets", unit="dataset"):
        final_processed = final_datasets[dataset_label]
        dataset_timings = {}

        # Type 1: Connectivity Metrics (silent; one result dict per dataset)
        t0 = time.time()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                metrics_result = calculate_connectivity_metrics(
                    final_processed,
                    release_grids={'passive': binned_ocean_grid_df},
                    settlement_grids={'passive': binned_ocean_grid_df},
                    traits=passive_traits,
                    traj_raw=None,
                    print_summary=False,
                    grid_ids=grid_ids_2d,
                    settlement_buffer=0,
                )
            metrics_by_dataset[dataset_label] = metrics_result
        except Exception as e:
            print(f"  Warning: connectivity metrics failed for {dataset_label}: {e}")
            metrics_by_dataset[dataset_label] = None
        dataset_timings['metrics'] = time.time() - t0

        # Type 2: Network Analysis (silent)
        t0 = time.time()
        try:
            # Always suppress network_analysis's internal prints.
            # The full results are available in the viewers and saved CSVs.
            with contextlib.redirect_stdout(io.StringIO()):
                network_results = network_analysis(
                    final_processed=final_processed,
                    release_grids={'passive': binned_ocean_grid_df},
                    settlement_grids={'passive': binned_ocean_grid_df},
                    coastal_grid_df=binned_ocean_grid_df,
                    grid_ids=grid_ids_2d,
                    full_grid_df=full_grid_df_path_a,
                    null_scale=None,
                    exclude_self_recruitment=config['exclude_self_recruitment'],
                )
        except Exception as e:
            print(f"  Warning: network analysis failed for {dataset_label}: {e}")
            dataset_timings['network'] = time.time() - t0
            timing_log[dataset_label] = dataset_timings
            continue
        dataset_timings['network'] = time.time() - t0

        if network_results is None:
            print(f"  Network analysis produced no results for {dataset_label}. Skipping.")
            timing_log[dataset_label] = dataset_timings
            continue

        network_results_by_dataset[dataset_label] = network_results

        # Compute network-level summary stats (Type 2, displayed in the Louvain communities viewer)
        # Density = fraction of possible directed edges realised
        # Modularity = Louvain partition quality (0 = random, >=0.4 = strong community structure)
        try:
            G = network_results.get('network_graph')
            if G is not None:
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

        # Save figures.
        if save_figures:
            dataset_fig_dir = os.path.join(project_dirs['figures'], dataset_label)
            os.makedirs(dataset_fig_dir, exist_ok=True)

            with contextlib.redirect_stdout(io.StringIO()):
                # Build the list of figures to save.
                # Capture current values in default args to avoid late-binding issues.
                _sm = network_results['site_metrics']
                _cm = network_results['connectivity_matrix']
                _t10 = network_results.get('top10_overall', _sm.head(10))
                _ng = network_results.get('network_graph')
                _dl = dataset_label

                save_list = [
                    ('hotspot_overall', lambda sm=_sm, t10=_t10, ng=_ng, dl=_dl: create_network_map(
                        sm, t10, title=dl,
                        composite_col='overall_composite', cmap='viridis_r',
                        show_figure=False, network_graph=ng, top_n_edges=200,
                    )),
                    ('network_only', lambda sm=_sm, ng=_ng, dl=_dl: create_network_map(
                        sm, sm.head(0), title=f"{dl} (network only)",
                        composite_col='overall_composite', cmap='viridis_r',
                        show_figure=False, network_graph=ng,
                        top_n_edges=None, network_only=True,
                    )),
                    ('communities', lambda nr=network_results, dl=_dl: create_community_map(
                        nr, title=dl, show_figure=False)[0]),
                    ('connectivity_matrix', lambda cm=_cm, sm=_sm, dl=_dl: create_matrix_with_map(
                        cm, sm, title=dl, show_figure=False)),
                    ('heatmap', lambda nr=network_results, dl=_dl: build_heatmap_figure(
                        nr, title=dl)),
                ]

                # Per-metric hotspot maps
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
                            lambda sm=_sm, ng=_ng, dl=_dl, mc=metric_col: create_network_map(
                                sm, sm.nlargest(10, mc),
                                title=f"{dl} ({mc})",
                                composite_col=mc, cmap='viridis_r',
                                show_figure=False, network_graph=ng, top_n_edges=200,
                            )
                        ))

                for fn, make_fig in save_list:
                    t0 = time.time()
                    try:
                        fig = make_fig()
                        if fig is not None:
                            save_figure(fig, os.path.join(dataset_fig_dir, fn))
                            plt.close(fig)
                    except Exception:
                        pass  # silently skip failures (e.g. louvain missing)

                    dataset_timings[f'fig_{fn}'] = time.time() - t0

        # Save result tables silently
        if save_results:
            dataset_results_dir = os.path.join(project_dirs['results'], dataset_label)
            os.makedirs(dataset_results_dir, exist_ok=True)

            # Type 1: Connectivity metrics CSV (was a bug — never called before)
            t0 = time.time()
            if metrics_by_dataset.get(dataset_label) is not None:
                try:
                    # Passive pathway: export only the summary table, not per-species/community
                    metrics = metrics_by_dataset[dataset_label]
                    if 'community' in metrics:
                        summary_df = pd.DataFrame([metrics['community']])
                        summary_file = os.path.join(dataset_results_dir, 'connectivity_summary.csv')
                        summary_df.to_csv(summary_file, index=False)
                except Exception:
                    pass
            dataset_timings['csv_type1'] = time.time() - t0

            # Type 2: Network results CSVs
            t0 = time.time()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    export_network_results(network_results, output_dir=dataset_results_dir)
            except Exception:
                pass
            dataset_timings['csv_type2'] = time.time() - t0

        timing_log[dataset_label] = dataset_timings

    if timing_log:
        try:
            timing_path = os.path.join(project_dirs['config'], 'analysis_timing.txt')
            all_keys = sorted({k for t in timing_log.values() for k in t})
            with open(timing_path, 'w') as f:
                f.write("Timing summary - total seconds per step across all datasets:\n")
                for k in all_keys:
                    total = sum(t.get(k, 0) for t in timing_log.values())
                    f.write(f"    {k:<22s} {total:7.1f} s\n")
                grand_total = sum(sum(t.values()) for t in timing_log.values())
                f.write(f"    {'GRAND TOTAL':<22s} {grand_total:7.1f} s\n\n")
                f.write("Slowest single dataset per step:\n")
                for k in all_keys:
                    worst = max(timing_log, key=lambda lab: timing_log[lab].get(k, 0))
                    f.write(f"    {k:<22s} {worst} ({timing_log[worst].get(k, 0):.1f} s)\n")
        except Exception:
            pass

    # Step 8: Metadata + summary + interactive results viewer
    metadata_path = os.path.join(project_dirs['config'], 'comcon_outputs.xlsx')

    # Strip Copernicus credentials out before writing metadata
    config['project_dirs'] = project_dirs
    config['save_figures'] = save_figures
    config['save_results'] = save_results
    config_for_metadata = {k: v for k, v in config.items() if not k.startswith('_')}

    pathway_duration = time.time() - pathway_start_time
    run_info = {
        'pathway_duration_seconds': pathway_duration,
        'n_binned_ocean_cells': int(len(binned_ocean_grid_df)) if binned_ocean_grid_df is not None else 0,
        'n_particles_per_season': int(n_particles_per_season),
        'n_datasets': len(network_results_by_dataset),
        'metadata_path': metadata_path,
    }
    try:
        metadata_dict = passive_metadata(config_for_metadata, run_info)
        write_project_metadata(project_dirs, metadata_dict, 'passive', metadata_path)
        print(f"  Metadata saved: {metadata_path}")
    except Exception as e:
        print(f"  Warning: could not write metadata file ({e}).")

    subsection_header("Outputs")
    print(f"  Figures saved to:   {project_dirs['figures']}")
    print(f"  Results saved to:   {project_dirs['results']}")
    print(f"  Metadata saved to:  {metadata_path}")
    print("")
    print(f"  See the metadata file for an explanation of each dataset.\n")

    # Structured analysis results display: Type 1 first, then Type 2 subsections
    section_header("Analysis Results")

    # Type 1
    subsection_header("Type 1: Connectivity Metrics")
    info_box([
        "Type 1 reports community-level summary statistics for each dataset, including:",
        "  - Total particles released (proportion returning within one cell of their release)",
        "  - Self-recruitment rate",
        "  - Dispersal distance in km",
        "  - An animation of particle dispersal",
        "",
        "Use the dropdown below each viewer to switch between datasets.",
    ])
    if metrics_by_dataset:
        print()
        subsection_header("Summary statistics")
        try:
            summary_table_viewer(metrics_by_dataset)
        except Exception as e:
            print(f"  Note: summary table viewer could not be displayed: {e}")
    else:
        print("  No Type 1 results available.")

    if all_trajectory_paths:
        print()
        subsection_header("Dispersal animation")
        try:
            animation_viewer(all_trajectory_paths)
        except Exception as e:
            print(f"  Note: animation viewer could not be displayed: {e}")

    # Type 2
    subsection_header("Type 2: Network Analysis")
    info_box([
        "Type 2 treats the connectivity data as a directed weighted network.",
        "In this network nodes = sites, edges = counts of particles moving between sites.",
        "Four complementary outputs are generated:",
        "",
        "1. Connectivity matrix: ",
        "   - Source-sink heatmap of particle exchanges",
        "",
        "2. Directed network with hotspots: ",
        "   - All sites are coloured by the selected network metric (rank 1 = best)",
        "   - The top 200 strongest site-to-site connections drawn as arrows on top",
        "   - The top 10 connectivity hotspots are indicated",
        "",
        "3. Site × metric heatmap: ",
        "   - Compact overview of rank across all metrics for every site in the dataset",
        "",
        "4. Louvain communities: ",
        "   - Sites are grouped by shared connectivity patterns",
        "   - One colour per community",
        "   - Network-level summary stats: ",
        "       - Density: ",
        "           - The percentage of possible connections that exist (0-1)",
        "           - Low (<0.5) = Most sites unconnected",
        "           - High (>0.5) = Most sites connected",
        "       - Modularity: ",
        "           - How cleanly the network separates into communities (-1 to 1)",
        "           - High (>=0.3) = well-separated communities ",
        "           - Low (<0.3) = weak partition.",
        "       - Number of communities detected by the Louvain algorithm",
        "",
        "Each viewer has a dataset dropdown to switch between datasets.",
    ])

    if network_results_by_dataset:
        print()
        subsection_header("Connectivity matrix")
        try:
            matrix_viewer(network_results_by_dataset)
        except Exception as e:
            print(f"  Note: matrix viewer could not be displayed: {e}")

        print()
        subsection_header("Directed connectivity network with hotspots")
        try:
            metric_viewer(network_results_by_dataset)
        except Exception as e:
            print(f"  Note: metric viewer could not be displayed: {e}")

        print()
        subsection_header("Site × metric heatmap")
        try:
            heatmap_viewer(network_results_by_dataset)
        except Exception as e:
            print(f"  Note: heatmap viewer could not be displayed: {e}")

        print()
        subsection_header("Louvain communities")
        try:
            community_viewer(network_results_by_dataset)
        except Exception as e:
            print(f"  Note: community viewer could not be displayed: {e}")
    else:
        print("  No Type 2 results available.")

    # Completion banner prints last, after all viewers are in place.
    display_passive_config_end(config, run_info)

    return None
# endregion