#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cobra Reaction ID Cleaner (2)

This script cleans and standardizes reaction IDs in an enzyme-constrained (ec) 
model, particularly one that has been processed by a preliminary fixing script. 
It is designed to be run from the command line.

The script performs a series of cleaning steps in a specific order:
1.  Loads an ec-model from an SBML file.
2.  Renames duplicated reactions with high-number suffixes (e.g., 'EXP_2', 'REV_EXP_3') 
    to a standard '_for'/'_rev' format.
3.  Renames unpaired reactions (e.g., 'ACALD_EXP_1' without a corresponding 
    'ACALD_REV_EXP_1') by removing the suffix.
4.  Renames the remaining paired reactions (e.g., 'ATPS4r_EXP_1' and 
    'ATPS4r_REV_EXP_1') to the standard '_for'/'_rev' format.
5.  Repairs the model after each major step to ensure consistency.
6.  Saves the fully cleaned model to a new SBML file.

Usage:
    python <script_name>.py --ec-model-path <path_to_ec_model> \
                            --output-path <path_to_output_file>
"""

import argparse
import os
import pandas as pd
import re
import cobra
from cobra.io import read_sbml_model, write_sbml_model

def get_reaction_list(model: cobra.Model, model_name: str) -> pd.DataFrame:
    """
    Extracts a list of all reactions and their properties from a cobra model.

    Args:
        model (cobra.Model): The cobra model object to process.
        model_name (str): A name to identify the model in the output DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing reaction details.
    """
    print(f"INFO: Extracting reactions from model: {model_name}")
    columns = ['Model_Name', 'Reaction_Name', 'Reversibility', 'Lower_Bound', 'Upper_Bound']
    reactions_data = []
    for reaction in model.reactions:
        reactions_data.append({
            "Model_Name": model_name,
            "Reaction_Name": reaction.id,
            "Reversibility": reaction.reversibility,
            "Lower_Bound": reaction.lower_bound,
            "Upper_Bound": reaction.upper_bound
        })
    model_reaction_df = pd.DataFrame(reactions_data, columns=columns)
    print(f"INFO: Found {len(model_reaction_df)} total reactions.")
    return model_reaction_df

def rename_high_number_duplicates(model: cobra.Model):
    """
    Finds and renames reactions with suffixes like '_EXP_[2-9]' or '_REV_EXP_[2-9]'.

    Args:
        model (cobra.Model): The cobra model to be modified.
    """
    print("\n--- Step 1: Renaming high-number duplicate reactions ---")
    pattern_to_find = r'EXP_[2-9]|REV_EXP_[2-9]'
    reactions_to_rename = [r for r in model.reactions if re.search(pattern_to_find, r.id)]
    
    if not reactions_to_rename:
        print("INFO: No high-number duplicate reactions found to rename.")
        return

    print(f"INFO: Found {len(reactions_to_rename)} reactions to rename.")
    for reaction in reactions_to_rename:
        original_id = reaction.id
        # Replace "REV_EXP" first to avoid partial replacement issues
        new_id = re.sub(r'REV_EXP', 'rev', original_id)
        new_id = re.sub(r'EXP', 'for', new_id)
        reaction.id = new_id
        print(f"  - Renamed '{original_id}' to '{new_id}'")
    
    model.repair()
    print("INFO: Model repaired after renaming high-number duplicates.")

def rename_unpaired_reactions(model: cobra.Model):
    """
    Finds and renames unpaired reactions ending in '_EXP_1'.

    An unpaired reaction is one that has an '_EXP_1' version but no
    corresponding '_REV_EXP_1' version. These are renamed by removing the suffix.

    Args:
        model (cobra.Model): The cobra model to be modified.
    """
    print("\n--- Step 2: Renaming unpaired '_EXP_1' reactions ---")
    reaction_df = get_reaction_list(model, "current")
    
    # Extract base names by removing suffixes
    reaction_df['Base_Name'] = reaction_df['Reaction_Name'].str.replace('_REV_EXP_1$', '', regex=True).str.replace('_EXP_1$', '', regex=True)
    
    base_counts = reaction_df['Base_Name'].value_counts()
    
    # Identify rows where the base name appears only once and the reaction ends with '_EXP_1'
    unpaired_mask = (reaction_df['Base_Name'].map(base_counts) == 1) & reaction_df['Reaction_Name'].str.endswith('_EXP_1')
    reactions_to_rename_df = reaction_df[unpaired_mask]

    if reactions_to_rename_df.empty:
        print("INFO: No unpaired '_EXP_1' reactions found to rename.")
        return

    print(f"INFO: Found {len(reactions_to_rename_df)} unpaired reactions to rename.")
    for reaction_id in reactions_to_rename_df['Reaction_Name']:
        try:
            reaction = model.reactions.get_by_id(reaction_id)
            new_id = reaction.id.replace("_EXP_1", "")
            reaction.id = new_id
            print(f"  - Renamed '{reaction_id}' to '{new_id}'")
        except KeyError:
            print(f"  - WARNING: Reaction '{reaction_id}' was identified but not found in model. Skipping.")
            
    model.repair()
    print("INFO: Model repaired after renaming unpaired reactions.")

def rename_remaining_paired_reactions(model: cobra.Model):
    """
    Renames all remaining reactions ending in '_EXP_1' or '_REV_EXP_1'.

    This step assumes that only correctly paired forward/reverse reactions
    with these suffixes are left.

    Args:
        model (cobra.Model): The cobra model to be modified.
    """
    print("\n--- Step 3: Renaming remaining paired '_EXP_1' reactions ---")
    pattern_to_find = r'_EXP_1$'
    reactions_to_rename = [r for r in model.reactions if re.search(pattern_to_find, r.id)]

    if not reactions_to_rename:
        print("INFO: No remaining paired reactions found to rename.")
        return
        
    print(f"INFO: Found {len(reactions_to_rename)} paired reactions to rename.")
    for reaction in reactions_to_rename:
        original_id = reaction.id
        if "_REV_EXP_1" in original_id:
            new_id = original_id.replace("_REV_EXP_1", "_rev")
        else:
            new_id = original_id.replace("_EXP_1", "_for")
        reaction.id = new_id
        print(f"  - Renamed '{original_id}' to '{new_id}'")
        
    model.repair()
    print("INFO: Model repaired after renaming remaining pairs.")

def main():
    """
    Main function to run the command-line tool.
    """
    parser = argparse.ArgumentParser(
        description="A command-line tool to clean and standardize reaction IDs in an ec-model.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-e", "--ec-model-path",
        type=str,
        required=True,
        help="Path to the enzyme-constrained (ec) SBML model file to be cleaned."
    )
    parser.add_argument(
        "-o", "--output-path",
        type=str,
        required=True,
        help="Path to save the cleaned ec-model SBML file."
    )
    args = parser.parse_args()

    try:
        # 1. Load Model
        print(f"INFO: Loading model from {args.ec_model_path}...")
        ec_model = read_sbml_model(args.ec_model_path)

        # 2. Run cleaning steps in sequence
        rename_high_number_duplicates(ec_model)
        rename_unpaired_reactions(ec_model)
        rename_remaining_paired_reactions(ec_model)

        # 3. Final Check
        print("\n--- Final Check ---")
        final_check_list = [r.id for r in ec_model.reactions if "EXP_" in r.id]
        if not final_check_list:
            print("INFO: Final check passed. No 'EXP_' suffixes remain.")
        else:
            print(f"WARNING: Final check found remaining 'EXP_' suffixes: {final_check_list}")

        # 4. Save the final model
        print(f"\nINFO: Writing cleaned model to {args.output_path}")
        write_sbml_model(ec_model, args.output_path)

        print("\nSUCCESS: Model cleaning complete.")

    except FileNotFoundError as e:
        print(f"ERROR: File not found - {e}. Please check the file paths.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()