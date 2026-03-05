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

# Standardised formatting
# Fallback width for any non-HTML contexts
width = 92


def display_line(char="═"):
    """Display a line that fills the container width dynamically."""
    try:
        display(HTML(
            f'<pre style="font-family: monospace; margin: 0; overflow: hidden; white-space: pre; width: 100%;">{char * 300}</pre>'))
    except:
        print(char * width)


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


# Display logo on import
display_logo()

# region Setup Functions
def welcome_message():
    """Display welcome message and package information."""
    display_line("═")
    display(HTML('''
        <div style="font-family: monospace; text-align: center; width: 100%;">ComCon v1.0.0</div>
        <div style="font-family: monospace; text-align: center; width: 100%;">The Community Connectivity Package</div>
    '''))
    display_line("═")

    print("\nWelcome to ComCon!")
    print("\nThis package facilitates multi-species dispersal modeling using Lagrangian particle tracking with Ocean Parcels and NEMO hydrodynamic data.")
    print("\nYou can use the package in three modes:")
    print("  1. Interactive Mode: Step-by-step guided workflow")
    print("  2. Automatic Mode: Run entire workflow with preset parameters")
    print("  3. Manual Mode: Access individual functions for custom workflows")
    print("")
    display_line("═")
    print("Please select a mode below:")


def choose_mode():
    """Display mode selection dropdown."""
    dropdown = widgets.Dropdown(
        options=[
            ('Interactive Mode', 'interactive'),
            ('Automatic Mode', 'automatic'),
            ('Manual Mode', 'manual'),
        ],
        description='Mode:',
        style={'description_width': 'initial'}
    )

    button = widgets.Button(
        description="Start",
        button_style='success',
        icon='check'
    )

    output = widgets.Output(layout={'height': '0px', 'overflow': 'hidden'})

    display(dropdown, button, output)

    def on_button_click(b):
        selected_mode = dropdown.value
        with output:
            output.layout.height = 'auto'
            output.layout.overflow = 'visible'
            clear_output()
            display_line("═")

            if selected_mode == 'interactive':
                print("\nInteractive Mode")
                display_line("─")
                print("Guides users step-by-step through the complete workflow with detailed explanations, customisation options and visualisations at each step.")
                print("Recommended for first-time users and for exploratory analysis.\n")
                print("Step-by-step workflow with 6 main steps:")
                print("  1. Hydrodynamic and Trait Data Setup")
                print("  2. Species Occurrence Data Setup")
                print("  3. Release/Settlement Site Setup")
                print("  4. Particle Simulation")
                print("  5. Post-Processing")
                print("  6. Connectivity Analyses")
                print("\nTo begin, run in a new cell:")
                print("from comcon.interactive import *")
                print("workflow_step(1)")

            elif selected_mode == 'automatic':
                print("Automatic Mode")
                display_line("─")
                print("Collects all parameters upfront, then runs the entire workflow automatically with progress updates only.")
                print("\nThis will complete the following 6 main steps:")
                print("  1. Hydrodynamic and Trait Data Setup")
                print("  2. Species Occurrence Data Setup")
                print("  3. Release/Settlement Site Setup")
                print("  4. Particle Simulation")
                print("  5. Post-Processing")
                print("  6. Connectivity Analyses")
                print("\nTo begin, run in a new cell:")
                print("from comcon.automatic import *")
                print("start_automatic_mode()")

            else:  # manual
                print("\nManual Mode")
                display_line("─")
                print("Access individual functions for custom workflows. Useful for re-running specific analyses or building custom pipelines.")
                print("\nTo access all functions and get started, run in a new cell:")
                print("from comcon.manual import *")
                print("\nTo see all available functions, run in a new cell:")
                print("list_functions()")
                print("\nTo see example workflows:")
                print("example_workflow()")

            print("")
            display_line("═")

    button.on_click(on_button_click)
# endregion

# region Run Setup Function
# Show the welcome message and allow the user to select a mode
welcome_message()
choose_mode()

# Define what's available when doing "from comcon import *"
__all__ = [
    'workflow_step',
    'welcome_message',
    'choose_mode',
    'display_line',
    'width',
]
# endregion