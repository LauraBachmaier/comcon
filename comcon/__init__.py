"""
ComCon: Community Connectivity Package
=======================================

A package for multi-species marine larval dispersal modeling using
Lagrangian particle tracking with OceanParcels and NEMO hydrodynamic data.
"""

__version__ = '1.0.0'
__author__ = 'Laura Bachmaier'

# Required imports for __init__.py
import warnings
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

warnings.filterwarnings('ignore')

# Suppress Jupyter's IOPub data rate throttle and auto-expand scrolled output cells
# This prevents the confusing "show more output" collapse that hides simulation progress
try:
    _ip = get_ipython()
    _ip.run_line_magic('config', 'Application.iopub_data_rate_limit=1.0e10')
except Exception:
    pass

# Write JupyterLab user settings to disable the "Show more outputs" output cap.
# Takes effect from the next JupyterLab launch. Silently no-ops on non-JupyterLab environments.
try:
    import json as _json, os as _os
    for _nb_plugin in [
        "@jupyterlab/notebook-extension",  # JupyterLab 3.x
        "@jupyterlab/notebook",            # JupyterLab 4.x
    ]:
        _settings_dir = _os.path.expanduser(
            f"~/.jupyter/lab/user-settings/{_nb_plugin}"
        )
        _os.makedirs(_settings_dir, exist_ok=True)
        _settings_file = _os.path.join(_settings_dir, "tracker.jupyterlab-settings")
        _existing = {}
        if _os.path.exists(_settings_file):
            with open(_settings_file) as _f:
                try:
                    _existing = _json.load(_f)
                except Exception:
                    _existing = {}
        if _existing.get("maxNumberOutputs") != 0:
            _existing["maxNumberOutputs"] = 0
            with open(_settings_file, "w") as _f:
                _json.dump(_existing, _f, indent=2)
except Exception:
    pass

# Standardised formatting
# Fallback width for any non-HTML contexts
width = 92


## Full-width horizontal line for Jupyter and terminal output
def display_line(char="═"):
    """Display a line that fills the container width dynamically."""
    try:
        display(HTML(
            f'<pre style="font-family: monospace; margin: 0; overflow: hidden; white-space: pre; width: 100%;">{char * 300}</pre>'))
    except:
        print(char * width)


## Display the ComCon logo on import
def display_logo():
    """Display package logo in Jupyter environments."""
    try:
        import base64
        import os

        logo_path = os.path.join(os.path.dirname(__file__), 'assets', 'logo.png')

        if os.path.exists(logo_path):
            with open(logo_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')

            display(HTML(f'''
                <div style="width: 100%; text-align: center; margin: 15px 0 -10px 0;">
                    <img src="data:image/png;base64,{img_data}" style="width: 100%; max-width: 100%;">
                </div>
            '''))
        else:
            display_line("═")
            print("  ComCon: The Community Connectivity Package v1.0".center(width))
            display_line("═")
    except ImportError:
        display_line("═")
        print("  ComCon: The Community Connectivity Package v1.0".center(width))
        display_line("═")
    except Exception:
        pass


# Check ffmpeg availability (needed for MP4 animation export)
try:
    import subprocess
    _ffmpeg_check = subprocess.run(['ffmpeg', '-version'], capture_output=True)
    if _ffmpeg_check.returncode != 0:
        raise FileNotFoundError
except (FileNotFoundError, OSError):
    try:
        print("  Note: The ffmpeg tool was not found. Animations will display in-notebook but MP4 export is unavailable.")
        print("  To enable MP4 export, run: conda install -c conda-forge ffmpeg\n")
    except Exception:
        pass

# Display logo on import
display_logo()

# region Setup Functions
## Welcome message printed on import
def welcome_message():
    """Display welcome message and package information."""
    display_line("═")
    display(HTML('''
        <div style="font-family: monospace; text-align: center; width: 100%;">ComCon v1.0.0</div>
        <div style="font-family: monospace; text-align: center; width: 100%;">The Community Connectivity Package</div>
    '''))
    display_line("═")

    print("\nWelcome to ComCon!")
    print("\nThis package facilitates multi-species dispersal modelling using Lagrangian particle tracking with Ocean Parcels and NEMO hydrodynamic data.")
    print("\nComCon offers two workflow pathways:")
    print("")
    print("1. Passive dispersal pathway")
    print("   - This is a quick hydrodynamic overview of your area of interest with no species-specific trait data integrated")
    print("   - This pathway is recommended as an easy entry-point with minimal user choice")
    print("   - The approach follows James et al. (2025)")
    print("")
    print("2. Trait-defined pathway")
    print("   - This provides access to the full trait-defined workflow")
    print("   - This pathway allows the use of species-specific planktonic larval durations, spawning timings and habitat preferences")
    print("   - In contrast to Pathway 1, which runs a passive dispersal analysis, Pathway 2 allows for a biologically-informed analysis")
    print("")
    print("For further information on either pathway, please see the User Guide.")
    print("")
    display_line("═")
    print("Please select a pathway below:")


## Pathway and mode selection widget displayed on import
def choose_pathway():
    """Display pathway selection, with a nested mode dropdown for the trait-defined pathway."""
    pathway_dropdown = widgets.Dropdown(
        options=[
            ('Passive dispersal pathway', 'passive'),
            ('Trait-defined pathway', 'trait'),
        ],
        description='Pathway:',
        style={'description_width': 'initial'}
    )

    pathway_button = widgets.Button(
        description="Select",
        button_style='success',
        icon='check'
    )

    pathway_output = widgets.Output(layout={'height': '0px', 'overflow': 'hidden'})

    display(pathway_dropdown, pathway_button, pathway_output)

    ## Print pathway instructions for the passive dispersal option
    def show_passive_instructions():
        """Print the passive dispersal pathway instructions and entry-point commands."""
        display_line("═")
        print("\nPassive dispersal pathway")
        display_line("─")
        print("This runs a quick hydrodynamic overview of your area of interest following James et al. (2025):")
        print("   - Particles are released from all ocean cells")
        print("   - 28-day simulations (summer + winter) are run")
        print("   - Connections are extracted at fixed timepoints (7, 14, 21, 28 days) to represent a range of dispersal capabilities")
        print("   - Analyses are computed automatically")
        print("   - No species-specific trait data is integrated")
        print("")
        print("What requires user input:")
        print("  - Latitude and longitude bounds of your area of interest")
        print("  - A project folder name")
        print("  - Export of figures and result tables")
        print("  - Copernicus Marine Service credentials (first run only and saved for future use)")
        print("")
        print("What is set automatically:")
        print("  - Hydrodynamic dataset")
        print("  - Grid resolution")
        print("  - Simulation time window")
        print("  - Number of particles, runtime, timestep and horizontal diffusivity")
        print("  - Analysis setup")
        print("")
        print("For full methodological details, see Part 4 of the User Guide.")
        print("")
        print("JupyterLab tip: If you see a 'Show more outputs' button during a workflow step, click it to reveal all output.")
        print("→ To disable this permanently: Settings → Settings Editor → Notebook → set 'maxNumberOutputs' to 0.")
        print("")
        print("To begin, run in a new cell:")
        print("from comcon.passive import *")
        print("passive_dispersal_pathway()")
        display_line("═")

    ## Display the trait-defined pathway description and mode selection dropdown
    def show_trait_mode_dropdown():
        """Display the trait-defined pathway overview and present the interactive/automatic/manual mode selector."""
        display_line("═")
        print("\nTrait-defined pathway")
        display_line("─")
        print("This runs the full trait-defined workflow using species-specific trait data.")
        print("Particle release, dispersal and settlement can be constrained by species-specific traits.")
        print("Connectivity estimates can be calculated per species and at the community level.")
        print("")
        print("What requires user input:")
        print("  - A project folder name")
        print("  - Hydrodynamic data and grid resolution")
        print("     - This can be accessed via Copernicus or uploaded as a CSV file")
        print("     - You will need Copernicus Marine Service credentials (first run only and saved for future use)")
        print( "  - Selection of your species of interest and traits: planktonic larval duration, spawning window and habitat preference")
        print("     - This can be uploaded as a CSV or selected from the built-in database of common intertidal rocky shore species")
        print("  - Selection of your study extent")
        print("     - This can be done interactively via the package or uploaded as a GeoJSON file of a spatial polygon")
        print("  - Selection of release and settlement preferences using habitat suitability or species occurrence data")
        print("     - This can be done using GBIF to download occurrence data or uploaded as a CSV")
        print("     - There is the option to include site names to run a site-focused analysis")
        print("     - There is also the option to remove species-specific preferences entirely")
        print("  - Selection of your simulation parameters")
        print("     - This includes the number of particles, timestep, output frequency and horizontal diffusivity")
        print("  - Selection of the analyses you want to run")
        print("     - This include:")
        print("        - Type 1: Connectivity Metrics")
        print("        - Type 2: Network Analysis")
        print("        - Type 3: Trait Sensitivity")
        print("")
        print("All user input requirements have defaults available for easy-access.")
        print("")
        print("You can use the pathway in three modes:")
        print("   1. Interactive Mode: Step-by-step guided workflow")
        print("   2. Automatic Mode: Collect user-choice upfront and run end-to-end")
        print("   3. Manual Mode: Access individual functions for custom workflows")
        print("")
        print("For full methodological details, see Part 3 of the User Guide.")
        print("")
        print("Please select a mode below:")
        print("")

        mode_dropdown = widgets.Dropdown(
            options=[
                ('Interactive Mode', 'interactive'),
                ('Automatic Mode', 'automatic'),
                ('Manual Mode', 'manual'),
            ],
            description='Mode:',
            style={'description_width': 'initial'}
        )

        mode_button = widgets.Button(
            description="Start",
            button_style='success',
            icon='check'
        )

        mode_output = widgets.Output(layout={'height': '0px', 'overflow': 'hidden'})

        display(mode_dropdown, mode_button, mode_output)

        ## Handle mode button click and display the selected mode's instructions
        def on_mode_click(b):
            """Display start instructions for the selected trait-defined pathway mode."""
            selected_mode = mode_dropdown.value
            with mode_output:
                mode_output.layout.height = 'auto'
                mode_output.layout.overflow = 'visible'
                clear_output()
                display_line("═")

                if selected_mode == 'interactive':
                    print("\nInteractive Mode")
                    display_line("─")
                    print("Guides users step-by-step through the complete workflow with detailed explanations, customisation options and visualisations at each step.")
                    print("Recommended for first-time users and for exploratory analysis.\n")
                    print("Step-by-step workflow with 6 main steps:")
                    print("  1. Hydrodynamic and Trait Data")
                    print("  2. Species Occurrence Data")
                    print("  3. Simulation Setup")
                    print("  4. Execute the Simulation")
                    print("  5. Post-Processing")
                    print("  6. Connectivity Analyses")
                    print("")
                    print("JupyterLab tip: If you see a 'Show more outputs' button during a workflow step, click it to reveal all output.")
                    print("→ To disable this permanently: Settings → Settings Editor → Notebook → set 'maxNumberOutputs' to 0.")
                    print("\nTo begin, run in a new cell:")
                    print("from comcon.interactive import *")
                    print("workflow_step(1)")
                    display_line("═")

                elif selected_mode == 'automatic':
                    print("\nAutomatic Mode")
                    display_line("─")
                    print("Collects all parameters upfront, then runs the entire workflow automatically with progress updates only.")
                    print("\nThis will complete the following 6 main steps:")
                    print("  1. Hydrodynamic and Trait Data")
                    print("  2. Species Occurrence Data")
                    print("  3. Simulation Setup")
                    print("  4. Execute the Simulation")
                    print("  5. Post-Processing")
                    print("  6. Connectivity Analyses")
                    print("")
                    print("JupyterLab tip: If you see a 'Show more outputs' button during a workflow step, click it to reveal all output.")
                    print("→ To disable this permanently: Settings → Settings Editor → Notebook → set 'maxNumberOutputs' to 0.")
                    print("\nTo begin, run in a new cell:")
                    print("from comcon.automatic import *")
                    print("start_automatic_mode()")
                    display_line("═")

                else:  # manual
                    print("\nManual Mode")
                    display_line("─")
                    print("Access individual functions for custom workflows. Useful for re-running specific analyses or building custom pipelines.")
                    print("")
                    print("JupyterLab tip: If you see a 'Show more outputs' button during a workflow step, click it to reveal all output.")
                    print("→ To disable this permanently: Settings → Settings Editor → Notebook → set 'maxNumberOutputs' to 0.")
                    print("\nTo access all functions and get started, run in a new cell:")
                    print("from comcon.manual import *")
                    print("\nTo see all available functions, run in a new cell:")
                    print("list_functions()")
                    print("\nTo see example workflows:")
                    print("example_workflow()")
                    display_line("═")

        mode_button.on_click(on_mode_click)

    ## Handle pathway button click and route to the appropriate instructions
    def on_pathway_click(b):
        """Show passive instructions or trait mode dropdown depending on the selected pathway."""
        selected_pathway = pathway_dropdown.value
        with pathway_output:
            pathway_output.layout.height = 'auto'
            pathway_output.layout.overflow = 'visible'
            clear_output()

            if selected_pathway == 'passive':
                show_passive_instructions()
            else:  # trait
                show_trait_mode_dropdown()

    pathway_button.on_click(on_pathway_click)
# endregion

# region Run Setup Function
# Show the welcome message and allow the user to select a pathway
welcome_message()
choose_pathway()

# Define what's available when doing "from comcon import *"
__all__ = [
    'workflow_step',
    'passive_dispersal_pathway',
    'welcome_message',
    'choose_pathway',
    'display_line',
    'width',
]
# endregion