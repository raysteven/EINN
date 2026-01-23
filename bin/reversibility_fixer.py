#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cobra Model Reversibility Fixer (1)

This script identifies and resolves conflicting reaction reversibility between a 
conventional model and an enzyme-constrained (ec) model.

The script performs the following steps:
1.  Loads a conventional model and an ec-model from SBML files.
2.  Extracts reaction information (ID, reversibility, bounds) from each model.
3.  Identifies reactions that have conflicting reversibility statuses between the two models.
4.  Corrects the conflicting reactions in the ec-model to match the conventional model's logic.
5.  Repairs the modified ec-model to ensure consistency.
6.  Saves the fixed ec-model to a new SBML file.

This tool is useful for ensuring that an enzyme-constrained model is consistent
with its parent conventional model.

Usage:
    python <script_name>.py --conv-model-path <path_to_conventional_model> \
                            --ec-model-path <path_to_ec_model> \
                            --output-path <path_to_output_file>
"""

import argparse
import os
import pandas as pd
import cobra
from cobra import Reaction
from cobra.io import read_sbml_model, write_sbml_model
from typing import List

def get_reaction_list(model: cobra.Model, model_name: str) -> pd.DataFrame:
    """
    Extracts a list of reactions and their properties from a cobra model.

    Args:
        model (cobra.Model): The cobra model object to process.
        model_name (str): A name to identify the model in the output DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing reaction details including
                      'Model_Name', 'Reaction_Name', 'Reversibility',
                      'Lower_Bound', and 'Upper_Bound'.
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
    print(f"INFO: Found {len(model_reaction_df)} reactions in {model_name}.")
    return model_reaction_df

def get_conflicting_reversibility(reaction_list_A: pd.DataFrame, reaction_list_B: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies reactions with conflicting reversibility between two models.

    Args:
        reaction_list_A (pd.DataFrame): DataFrame of reactions from the first model.
        reaction_list_B (pd.DataFrame): DataFrame of reactions from the second model.

    Returns:
        pd.DataFrame: A DataFrame of reactions with conflicting reversibility,
                      indexed by 'Reaction_Name'.
    """
    print("INFO: Identifying conflicting reaction reversibilities...")
    # Merge to find common reactions, using suffixes for clarity
    overlapping_reaction = pd.merge(reaction_list_A, reaction_list_B, on="Reaction_Name", how="inner", suffixes=('_conv', '_ec'))

    # Find reactions where reversibility differs
    conflicting_reversibility = overlapping_reaction[overlapping_reaction["Reversibility_conv"] != overlapping_reaction["Reversibility_ec"]]

    conflicting_reversibility = conflicting_reversibility.set_index('Reaction_Name')
    print(f"INFO: Found {len(conflicting_reversibility)} conflicting reactions.")
    return conflicting_reversibility

def fix_conflicting_reversibility(conflicting_reactions: pd.DataFrame, model_to_fix: cobra.Model) -> cobra.Model:
    """
    Fixes conflicting reaction reversibility in a cobra model.

    For each conflicting reaction, it either renames existing forward/reverse
    reactions or splits a reversible reaction into two irreversible ones.

    Args:
        conflicting_reactions (pd.DataFrame): DataFrame of conflicting reactions.
        model_to_fix (cobra.Model): The cobra model to be modified.

    Returns:
        cobra.Model: The modified cobra model.
    """
    print(f"INFO: Starting to fix {len(conflicting_reactions)} reactions in the target model...")
    reactions_to_add = []
    reactions_to_remove = []

    for reaction_name in conflicting_reactions.index:
        try:
            reaction = model_to_fix.reactions.get_by_id(reaction_name)
        except KeyError:
            print(f"WARNING: Reaction '{reaction_name}' not found in the model to fix. Skipping.")
            continue

        # Case 1: The reaction is already split into forward and reverse, but named incorrectly.
        if reaction.lower_bound >= 0:
            print(f"  - Fixing forward reaction: {reaction_name}")
            # Rename the forward part
            reaction.id = f"{reaction_name}_for"

            # Find and rename the corresponding reverse part
            rev_reaction_name = f"{reaction_name}_REV"
            try:
                rev_reaction = model_to_fix.reactions.get_by_id(rev_reaction_name)
                rev_reaction.id = f"{reaction_name}_rev"
            except KeyError:
                print(f"  - WARNING: Expected reverse reaction '{rev_reaction_name}' not found. Skipping its rename.")

        # Case 2: The reaction is reversible and needs to be split.
        else:
            print(f"  - Splitting reversible reaction: {reaction_name}")
            # Create a new forward reaction
            for_reaction = Reaction(id=f"{reaction.id}_for")
            for_reaction.name = f"{reaction.name} (forward)"
            for_reaction.subsystem = reaction.subsystem
            for_reaction.add_metabolites(reaction.metabolites)
            for_reaction.lower_bound = 0
            for_reaction.upper_bound = reaction.upper_bound

            # Create a new reverse reaction
            rev_reaction = Reaction(id=f"{reaction.id}_rev")
            rev_reaction.name = f"{reaction.name} (reverse)"
            rev_reaction.subsystem = reaction.subsystem
            # Reverse the stoichiometry for the reverse reaction
            rev_stoichiometry = {met: -coeff for met, coeff in reaction.metabolites.items()}
            rev_reaction.add_metabolites(rev_stoichiometry)
            rev_reaction.lower_bound = 0
            rev_reaction.upper_bound = -reaction.lower_bound # Use the absolute value of the original lower bound

            reactions_to_add.extend([for_reaction, rev_reaction])
            reactions_to_remove.append(reaction)

    # Perform additions and removals in batch for efficiency
    if reactions_to_remove:
        print(f"INFO: Removing {len(reactions_to_remove)} original reversible reactions.")
        model_to_fix.remove_reactions(reactions_to_remove)
    if reactions_to_add:
        print(f"INFO: Adding {len(reactions_to_add)} new forward/reverse reactions.")
        model_to_fix.add_reactions(reactions_to_add)

    print("INFO: Finished fixing reactions.")
    return model_to_fix

def main():
    """
    Main function to run the command-line tool.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="A command-line tool to identify and fix conflicting reaction reversibility between a conventional and an ec-model.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-c", "--conv-model-path",
        type=str,
        required=True,
        help="Path to the conventional SBML model file (the reference model)."
    )
    parser.add_argument(
        "-e", "--ec-model-path",
        type=str,
        required=True,
        help="Path to the enzyme-constrained (ec) SBML model file (the model to be fixed)."
    )
    parser.add_argument(
        "-o", "--output-path",
        type=str,
        required=True,
        help="Path to save the fixed ec-model SBML file."
    )
    args = parser.parse_args()

    # --- Core Logic ---
    try:
        # 1. Load Models
        print("INFO: Loading models...")
        conv_model = read_sbml_model(args.conv_model_path)
        ec_model = read_sbml_model(args.ec_model_path)
        
        # Extract model names from file paths for clearer logging
        conv_model_name = os.path.splitext(os.path.basename(args.conv_model_path))[0]
        ec_model_name = os.path.splitext(os.path.basename(args.ec_model_path))[0]

        # 2. Get Reaction Lists
        conv_model_reactions = get_reaction_list(conv_model, conv_model_name)
        ec_model_reactions = get_reaction_list(ec_model, ec_model_name)

        # 3. Find Conflicting Reactions
        conflicting = get_conflicting_reversibility(conv_model_reactions, ec_model_reactions)

        if conflicting.empty:
            print("INFO: No conflicting reactions found. No changes needed.")
            # Still, we write the output file as a copy of the ec-model.
            write_sbml_model(ec_model, args.output_path)
            print(f"INFO: ec-model copied to {args.output_path} without changes.")
            return

        # 4. Fix Conflicts in the ec-model
        fixed_ec_model = fix_conflicting_reversibility(conflicting, ec_model)

        # 5. Repair and Save the Fixed Model
        print("INFO: Repairing the final model...")
        fixed_ec_model.repair()

        print(f"INFO: Writing fixed model to {args.output_path}")
        write_sbml_model(fixed_ec_model, args.output_path)

        print("\nSUCCESS: Model processing complete.")

    except FileNotFoundError as e:
        print(f"ERROR: File not found - {e}. Please check the file paths.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()