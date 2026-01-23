#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cobra Compartment Duplicator

This script takes a metabolic model and duplicates its reactions based on a set 
of predefined compartment transport rules. It is designed to create a new model 
where reactions are split to represent transport between different compartments explicitly.

The script performs the following steps:
1.  Loads a model from an SBML file.
2.  Uses a custom library function `screen_out_in` to analyze reactions based on 
    compartment mappings (`io_dict`) and a list of insignificant molecules.
3.  Uses `duplicate_model` to create a new model with duplicated reactions.
4.  Corrects the medium for the new duplicated model.
5.  Repairs the new model and saves it to an output SBML file.

This tool requires a local 'Library' folder containing a 'Duplicate_Model.py' module
with the necessary functions (`screen_out_in`, `duplicate_model`, `correct_duplicated_med`).

Usage:
    python <script_name>.py --model-path <path_to_model> [--output-path <path_to_output_file>]
"""

import argparse
import os
import cobra
from cobra.io import read_sbml_model, write_sbml_model

# --- Prerequisite ---
# This script requires the custom library 'Duplicate_Model' to be accessible.
# It is expected to be in a 'Library' sub-directory.
try:
    from Library.Duplicate_Model import screen_out_in, duplicate_model, correct_duplicated_med
except ImportError:
    print("ERROR: Could not import the required custom library.")
    print("Please ensure a 'Library' folder with 'Duplicate_Model.py' exists in the same directory as this script.")
    exit()

# --- CONFIGURATION ---
# These parameters define the core logic of the duplication process.
# They are kept here for easy modification without changing the main script logic.

# io_dict defines the valid compartment transitions for import ('_i') and export ('_o') reactions.
# Format: { suffix: [(source_compartment, destination_compartment), ...] }
# 'None' can be used to represent the exterior.
IO_DICT = {
    "_i": [
        (None, "e"), (None, "c"), ("e", "p"), ("p", "c"), 
        ("e", "c"), ("c", "m"), ("p", "m")
    ],
    "_o": [
        ("c", None), ("e", None), ("p", "e"), ("c", "p"), 
        ("c", "e"), ("m", "c"), ("m", "p")
    ]
}

# unsignificant_mols is a list of metabolites to be ignored during the screening process.
UNSIGNIFICANT_MOLS = ["h_p", "h_c", "pi_c", "pi_p", "adp_c", "h2o_c", "atp_c"]


def main():
    """
    Main function to run the command-line tool.
    """
    parser = argparse.ArgumentParser(
        description="A command-line tool to duplicate reactions in a model based on compartment transport rules.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-m", "--model-path",
        type=str,
        required=True,
        help="Path to the input SBML model file to be processed."
    )
    parser.add_argument(
        "-o", "--output-path",
        type=str,
        required=False,
        help="Optional: Path to save the duplicated SBML model file. \n"
             "If not provided, it defaults to '[model-path]_duplicated.xml'."
    )
    args = parser.parse_args()

    try:
        # 1. Load the input model
        print(f"INFO: Loading model from {args.model_path}...")
        model = read_sbml_model(args.model_path)

        # 2. Screen reactions based on the predefined rules
        print("INFO: Screening reactions based on compartment rules...")
        reac_id_to_io_count_and_way = screen_out_in(model, IO_DICT, UNSIGNIFICANT_MOLS)
        print("INFO: Screening complete.")

        # 3. Duplicate the model
        print("INFO: Duplicating model...")
        new_model = duplicate_model(model, reac_id_to_io_count_and_way)
        print("INFO: Duplication complete.")

        # 4. Correct the medium of the new model
        print("INFO: Correcting the medium for the new model...")
        default_med = model.medium
        new_med = new_model.medium
        correct_med = correct_duplicated_med(default_med, new_med)
        new_model.medium = correct_med
        print("INFO: Medium corrected.")
        # print("DEBUG: New medium is:", new_model.medium) # Uncomment for debugging

        # 5. Repair and save the new model
        print("INFO: Repairing the final model...")
        new_model.repair()

        # Determine output path
        if args.output_path:
            output_path = args.output_path
        else:
            base, ext = os.path.splitext(args.model_path)
            output_path = f"{base}_duplicated{ext}"

        print(f"INFO: Writing duplicated model to {output_path}")
        write_sbml_model(new_model, output_path)

        print("\nSUCCESS: Model duplication process complete.")
        print(f"Original model: {args.model_path}")
        print(f"Duplicated model: {output_path}")

    except FileNotFoundError as e:
        print(f"ERROR: File not found - {e}. Please check the file paths.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()