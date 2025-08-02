#!/usr/bin/env python3
"""
Example usage of the combined residual plotting script with actual data.
This script shows how to use the combined plotting function with your variables.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from combine_residual_plots import create_combined_residual_figure
import numpy as np
import matplotlib.pyplot as plt

def create_residual_figure_from_notebook_vars():
    """
    Create the combined residual figure using variables from your notebook.
    This assumes you have run the notebook and have the residual variables available.
    """
    
    # This would typically be called from within your notebook environment
    # where these variables are already defined
    try:
        # Prepare data dictionary with your actual residual data
        data_dict = {
            'GS1': (GS1_Tsky_residual, GS1_Tsky_residual_common),
            'GS1_db': (GS1_db_Tsky_residual, GS1_db_Tsky_residual_common),
            'GS5': (GS5_Tsky_residual, GS5_Tsky_residual_common),
            'GS5_db': (GS5_db_Tsky_residual, GS5_db_Tsky_residual_common),
            'GSF5': (GSF5_Tsky_residual, GSF5_Tsky_residual_common),
            'GSF5_db': (GSF5_db_Tsky_residual, GSF5_db_Tsky_residual_common),
        }
        
        # Panel titles (matching your comments in the original code)
        titles = [
            r'1 $\times$ TOD; 1 CalSrc',                    # panel (0,0)
            r'2 $\times$ TOD; 1 CalSrc',                    # panel (0,1) 
            r'1 $\times$ TOD; 5 CalSrc',                    # panel (1,0)
            r'2 $\times$ TOD; 5 CalSrc',                    # panel (1,1)
            r'1 $\times$ TOD; 5 CalSrc + 1/f prior',       # panel (2,0)
            r'2 $\times$ TOD; 5 CalSrc + 1/f prior',       # panel (2,1)
        ]
        
        # Create the combined figure
        fig, axes = create_combined_residual_figure(
            data_dict=data_dict,
            titles=titles,
            save_path='figures/combined_residual_analysis.pdf',
            figsize=(16, 12),
            binwidth=0.05,
            kde=True
        )
        
        plt.show()
        return fig, axes
        
    except NameError as e:
        print(f"Error: {e}")
        print("Make sure you have run the notebook first to define the residual variables.")
        return None, None

if __name__ == "__main__":
    print("This script should be run from within a Jupyter notebook environment")
    print("where the residual variables (GS1_Tsky_residual, etc.) are already defined.")
    print("\nAlternatively, you can copy the create_combined_residual_figure function")
    print("directly into your notebook and use it there.")
