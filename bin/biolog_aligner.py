#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cobra Biolog ID Aligner (3)

This script aligns reaction IDs in an enzyme-constrained (ec) model with a 
reference list of reaction IDs from a Biolog data CSV file. It is designed to 
be run from the command line.

The primary use case is to rename exchange reactions in the model (e.g., those
ending in '_for' or '_rev') to match the naming convention used in experimental
Biolog data (e.g., ending in '_i' for import or '_o' for output).

The script performs the following steps:
1.  Loads an ec-model from an SBML file.
2.  Loads a reference list of reaction IDs from the column headers of a Biolog CSV file.
3.  Identifies which reaction IDs from the Biolog file are missing from the model.
4.  For each missing ID, it finds the corresponding reaction in the model based on a 
    predefined suffix mapping ('_i' -> '_for', '_o' -> '_rev') and renames it.
5.  Repairs the model to ensure consistency.
6.  Saves the aligned model to a new SBML file.

Usage:
    python <script_name>.py --ec-model-path <path_to_ec_model> \
                            --biolog-csv-path <path_to_biolog_csv> \
                            --output-path <path_to_output_file>
"""

import argparse
import os
import pandas as pd
import cobra
from cobra.io import read_sbml_model, write_sbml_model

def fix_biolog_discrepancies(model: cobra.Model, reactions_to_fix: set):
    """
    Finds and renames reactions in the model to match the Biolog naming convention.

    Args:
        model (cobra.Model): The cobra model to be modified.
        reactions_to_fix (set): A set of reaction IDs from the Biolog file that 
                                need to be renamed in the model.
    """
    print(f"\nINFO: Found {len(reactions_to_fix)} reactions to align with Biolog data.")
    
    # This dictionary maps the suffix in the Biolog file ('i' or 'o')
    # to the current suffix in the model ('for' or 'rev').
    suffix_map = {'i': 'for', 'o': 'rev'}
    
    for biolog_id in reactions_to_fix:
        try:
            # e.g., 'ACALD_i' -> ['ACALD', 'i']
            parts = biolog_id.split('_')
            new_suffix = parts[-1]
            base_id = '_'.join(parts[:-1])
            
            if new_suffix not in suffix_map:
                print(f"  - WARNING: Skipping '{biolog_id}'. Suffix '{new_suffix}' is not in the map ('i', 'o').")
                continue

            # Find the corresponding reaction ID currently in the model
            # e.g., 'ACALD' + 'for' -> 'ACALD_for'
            current_model_id = f'{base_id}_{suffix_map[new_suffix]}'
            
            # Get the reaction from the model and rename it
            reaction = model.reactions.get_by_id(current_model_id)
            reaction.id = biolog_id
            print(f"  - Renamed '{current_model_id}' to '{biolog_id}'")

        except KeyError:
            print(f"  - WARNING: Could not find base reaction '{current_model_id}' in the model. Skipping '{biolog_id}'.")
        except IndexError:
            print(f"  - WARNING: Could not parse reaction ID '{biolog_id}'. Skipping.")

def main():
    """
    Main function to run the command-line tool.
    """
    parser = argparse.ArgumentParser(
        description="A command-line tool to align reaction IDs in an ec-model with a Biolog data file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-e", "--ec-model-path",
        type=str,
        required=True,
        help="Path to the enzyme-constrained (ec) SBML model file to be aligned."
    )
    parser.add_argument(
        "-b", "--biolog-csv-path",
        type=str,
        required=True,
        help="Path to the Biolog data CSV file containing reference reaction IDs in its headers."
    )
    parser.add_argument(
        "-o", "--output-path",
        type=str,
        required=True,
        help="Path to save the aligned ec-model SBML file."
    )
    args = parser.parse_args()

    try:
        # 1. Load Model
        print(f"INFO: Loading model from {args.ec_model_path}...")
        ec_model = read_sbml_model(args.ec_model_path)
        model_reactions = {r.id for r in ec_model.reactions}
        print(f"INFO: Found {len(model_reactions)} reactions in the model.")

        # 2. Load Biolog reference reactions
        print(f"INFO: Loading Biolog data from {args.biolog_csv_path}...")
        biolog_df = pd.read_csv(args.biolog_csv_path)
        # The last column is usually not a reaction, so it's excluded
        biolog_reactions = set(biolog_df.columns[:-1])
        print(f"INFO: Found {len(biolog_reactions)} reference reactions in the Biolog file.")

        # 3. Identify discrepancies
        reactions_to_fix = biolog_reactions - model_reactions
        
        if not reactions_to_fix:
            print("\nINFO: No discrepancies found. All Biolog reactions are already in the model.")
            # Still, we write the output file as a copy of the ec-model.
            write_sbml_model(ec_model, args.output_path)
            print(f"INFO: ec-model copied to {args.output_path} without changes.")
            return

        # 4. Fix reaction names
        fix_biolog_discrepancies(ec_model, reactions_to_fix)
        
        # 5. Repair and check
        print("\nINFO: Repairing the model...")
        ec_model.repair()
        
        print("INFO: Performing final check...")
        model_reactions_fixed = {r.id for r in ec_model.reactions}
        final_check = biolog_reactions - model_reactions_fixed
        
        if not final_check:
            print("INFO: Final check passed. All Biolog reactions are now present in the model.")
        else:
            print(f"WARNING: Final check failed. The following {len(final_check)} Biolog reactions are still missing:")
            print(sorted(list(final_check)))

        # 6. Save the final model
        print(f"\nINFO: Writing aligned model to {args.output_path}")
        write_sbml_model(ec_model, args.output_path)

        print("\nSUCCESS: Model alignment complete.")

    except FileNotFoundError as e:
        print(f"ERROR: File not found - {e}. Please check the file paths.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()