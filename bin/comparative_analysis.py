"""
comparative_analysis.py

A comprehensive script for comparative analysis of classical GEMs and enzyme-constrained models (ecModels).
This script generates Figure 3 panels (A-H) for two E. coli models (iML1515 and iAF1260).

INPUTS REQUIRED:
1. For iML1515 analysis:
   - iML1515.xml (classical GEM model)
   - iML1515_ec_duplicated.xml (enzyme-constrained model)
   - ecFVA.csv (flux variability analysis results)
   - kcats.csv (enzyme kcat values)
   - uniprotkb_taxonomy_id_83333_2025_10_14.tsv.gz (UniProt database)

2. For iAF1260 analysis:
   - iAF1260.xml (classical GEM model)
   - iAF1260_ec_duplicated.xml (enzyme-constrained model)
   - ecFVA_iAF1260.csv (flux variability analysis results)
   - kcats_iAF1260.csv (enzyme kcat values)
   - uniprotkb_taxonomy_id_83333_2025_10_14.tsv.gz (UniProt database - same as above)

OUTPUTS GENERATED:
1. Figure 3A: PCA plot comparing iML1515 and eciML1515 flux samples (3A.png, 3A.svg)
2. Figure 3B: Flux variability ECDF for iML1515 models (3B.png)
3. Figure 3C: Scatter plot of pathway usage vs flux variability reduction for iML1515 (3C.png)
4. Figure 3D: Bar plot of high-variability pathways for iML1515 (3D.png)
5. Figure 3E: PCA plot comparing iAF1260 and eciAF1260 flux samples (3E.png)
6. Figure 3F: Flux variability ECDF for iAF1260 models (3F.png)
7. Figure 3G: Scatter plot of pathway usage vs flux variability reduction for iAF1260 (3G.png)
8. Figure 3H: Bar plot of high-variability pathways for iAF1260 (3H.png)

Additional CSV outputs:
- flux_significance_test_eciML1515.csv
- flux_significance_test_eciAF1260.csv
- pathways_progress.csv (iML1515 pathway data)
- pathways_progress_iAF1260.csv (iAF1260 pathway data)
- highlighted_data.csv (iML1515 high-variability pathways)
- highlighted_data_iAF1260.csv (iAF1260 high-variability pathways)
- kegg_hierarchy_2025.json
- kegg_hierarchy_2025_iAF1260.json

Usage: python comparative_analysis.py
"""

import cobra
from cobra.sampling import sample
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu
import seaborn as sns
from matplotlib.ticker import LogLocator
from collections import defaultdict
import time
import os
import json
from bioservices import KEGG


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_flux_samples(samples_gem, samples_ec):
    """Clean and match flux samples between GEM and ecModel."""
    # Match reactions
    common_rxns = list(set(samples_gem.columns).intersection(samples_ec.columns))
    samples_gem = samples_gem[common_rxns]
    samples_ec = samples_ec[common_rxns]
    
    # Remove NaNs or infs
    samples_gem = samples_gem.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")
    samples_ec = samples_ec.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")
    
    # Keep only common clean reactions
    common_rxns = list(set(samples_gem.columns).intersection(samples_ec.columns))
    samples_gem = samples_gem[common_rxns]
    samples_ec = samples_ec[common_rxns]
    
    return samples_gem, samples_ec, common_rxns


def perform_statistical_test(samples_gem, samples_ec, common_rxns, output_csv):
    """Perform Mann-Whitney U test on flux samples."""
    results = []
    for rxn in common_rxns:
        flux_gem = samples_gem[rxn].values
        flux_ec = samples_ec[rxn].values
        
        # Skip any reactions with empty data or NaNs
        if np.isnan(flux_gem).any() or np.isnan(flux_ec).any():
            continue
        if len(flux_gem) == 0 or len(flux_ec) == 0:
            continue
        
        stat, pval = mannwhitneyu(flux_gem, flux_ec, alternative="two-sided")
        mean_diff = abs(np.mean(flux_gem) - np.mean(flux_ec))
        results.append((rxn, pval, mean_diff))
    
    results_df = pd.DataFrame(results, columns=["reaction", "p_value", "mean_diff"])
    results_df["significant"] = results_df["p_value"] < 0.05
    results_df["significant_large"] = (results_df["significant"]) & (results_df["mean_diff"] > 0.1)
    
    # Summary statistics
    n_total = len(results_df)
    n_sig = results_df["significant"].sum()
    n_sig_large = results_df["significant_large"].sum()
    perc_sig = 100 * n_sig / n_total
    perc_sig_large = 100 * n_sig_large / n_total
    
    print(f"Out of {n_total} reactions analyzed:")
    print(f" - {n_sig} ({perc_sig:.1f}%) have significantly different fluxes (p < 0.05)")
    print(f" - {n_sig_large} ({perc_sig_large:.1f}%) are significantly different with Δflux > 0.1 mmol·gDW⁻¹·h⁻¹")
    
    results_df.to_csv(output_csv, index=False)
    return results_df


def run_pca_analysis(samples_gem, samples_ec, model_names, output_prefix):
    """Perform PCA analysis and create plot."""
    samples_gem["model"] = model_names[0]
    samples_ec["model"] = model_names[1]
    all_samples = pd.concat([samples_gem, samples_ec])
    
    flux_matrix = all_samples.drop(columns=["model"])
    labels = all_samples["model"].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(flux_matrix)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Explained variance by PC1 and PC2: {pca.explained_variance_ratio_ * 100}")
    
    # Plot PCA
    plt.figure(figsize=(5.5, 5))
    colors = {model_names[0]: "C0", model_names[1]: "C1"}
    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], s=10, alpha=0.5, c=colors[label], label=label)
    
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png", dpi=1000)
    if output_prefix == "3A":
        plt.savefig("3A.svg")
    plt.close()


def plot_ecFVA_from_csv(csv_path, model_names=("GEM", "ecModel"), output_file="3B.png"):
    """
    Reads ecFVA.csv and:
      • plots cumulative flux variability
      • performs Wilcoxon/Mann–Whitney rank-sum test
      • reports median variability, p-value, and % fully variable reactions
    """
    # Load data
    df = pd.read_csv(csv_path, sep=";")
    
    # Compute flux variability ranges
    flux_range_gem = df["maxFlux"].astype(float) - df["minFlux"].astype(float)
    flux_range_ec = df["ec-maxFlux"].astype(float) - df["ec-minFlux"].astype(float)
    
    # Remove zeros or invalid values
    flux_range_gem = flux_range_gem.replace([np.inf, -np.inf], np.nan).dropna()
    flux_range_ec = flux_range_ec.replace([np.inf, -np.inf], np.nan).dropna()
    
    flux_range_gem = flux_range_gem[flux_range_gem > 1e-10]
    flux_range_ec = flux_range_ec[flux_range_ec > 1e-10]
    
    # Perform statistical test
    stat, pval = mannwhitneyu(flux_range_gem, flux_range_ec, alternative="two-sided")
    
    print("--------------------------------------------------")
    print(f"Flux Variability Comparison ({model_names[0]} vs {model_names[1]})")
    print("--------------------------------------------------")
    print(f"{model_names[0]} median variability: {np.median(flux_range_gem):.4g} mmol·gDW⁻¹·h⁻¹")
    print(f"{model_names[1]} median variability: {np.median(flux_range_ec):.4g} mmol·gDW⁻¹·h⁻¹")
    print(f"Wilcoxon rank-sum test p-value: {pval:.2e}")
    
    # Fraction of fully variable reactions
    n_full_gem = np.sum(np.isclose(flux_range_gem, 1000, atol=1e-6))
    n_full_ec = np.sum(np.isclose(flux_range_ec, 1000, atol=1e-6))
    perc_full_gem = 100 * n_full_gem / len(flux_range_gem)
    perc_full_ec = 100 * n_full_ec / len(flux_range_ec)
    
    print(f"{model_names[0]}: {perc_full_gem:.1f}% reactions have full variability (≈1000)")
    print(f"{model_names[1]}: {perc_full_ec:.1f}% reactions have full variability (≈1000)")
    print("--------------------------------------------------")
    
    # Sort and compute ECDF
    flux_range_gem = np.sort(flux_range_gem)
    flux_range_ec = np.sort(flux_range_ec)
    y_gem = np.linspace(1 / len(flux_range_gem), 1, len(flux_range_gem))
    y_ec = np.linspace(1 / len(flux_range_ec), 1, len(flux_range_ec))
    
    # Plot setup
    plt.figure(figsize=(5.5, 5))
    plt.step(flux_range_gem, y_gem, where="post",
             label=f"{model_names[0]} (median: {np.median(flux_range_gem):.3g})",
             linewidth=2, color="C0")
    plt.step(flux_range_ec, y_ec, where="post",
             label=f"{model_names[1]} (median: {np.median(flux_range_ec):.3g})",
             linewidth=2, color="C1")
    
    # Medians
    med_gem = np.median(flux_range_gem)
    med_ec = np.median(flux_range_ec)
    plt.axvline(med_gem, 0, 0.45, linestyle="--", color="C0", linewidth=1.2)
    plt.axvline(med_ec, 0, 0.45, linestyle="--", color="C1", linewidth=1.2)
    plt.plot(med_gem, 0.5, 'o', markersize=6, markerfacecolor='none', markeredgecolor='C0')
    plt.plot(med_ec, 0.5, 'o', markersize=6, markerfacecolor='none', markeredgecolor='C1')
    
    # Horizontal 0.5 line
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    
    # Axis formatting
    plt.xscale("log")
    plt.xlim(3E-2, 1e3)
    plt.ylim(0.1, 1.0)
    plt.yticks(np.arange(0.1, 1.01, 0.1))
    ax = plt.gca()
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    plt.tick_params(axis='x', which='both', direction='in')
    plt.tick_params(axis='y', which='both', direction='in')
    
    # Labels and legend
    plt.xlabel("Flux variability range (mmol·gDW⁻¹·h⁻¹)")
    plt.ylabel("Cumulative distribution")
    plt.legend(loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig(output_file, dpi=1000)
    plt.close()


def extract_gene_id(kegg_string):
    """Extract gene ID from KEGG string."""
    if isinstance(kegg_string, str):
        for entry in kegg_string.split(';'):
            if entry.startswith('eco:'):
                return entry.replace('eco:', '')
    return np.nan


def parse_kegg_brite(brite_text):
    """Parse KEGG BRITE hierarchy."""
    hierarchy = defaultdict(lambda: defaultdict(list))
    current_A = None
    current_B = None
    
    for line in brite_text.splitlines():
        line = line.strip()
        if not line or line.startswith(("+", "!", "#")):
            continue
        
        # Main category
        if line.startswith("A"):
            current_A = line[1:].strip()
            current_B = None
        
        # Subcategory
        elif line.startswith("B"):
            current_B = line[1:].strip()
        
        # Pathway
        elif line.startswith("C") and current_A and current_B:
            parts = line[1:].strip().split(maxsplit=1)
            if len(parts) == 2:
                pid, name = parts
                hierarchy[current_A][current_B].append({"id": pid, "name": name})
    return hierarchy


def get_pathways(k, gene_id, cache=None):
    """Fetch pathway information from KEGG with optional caching."""
    if cache and gene_id in cache:
        return cache[gene_id]
    
    if pd.isna(gene_id):
        return None
    try:
        entry = k.get(gene_id)
        if not entry:
            return None
        parsed = k.parse(entry)
        if "PATHWAY" in parsed:
            pathways = "; ".join(parsed["PATHWAY"].values())
            if cache is not None:
                cache[gene_id] = pathways
            return pathways
        return None
    except Exception as e:
        print(f"Error for {gene_id}: {e}")
        return None


def map_to_bar_even(pathways, pathway_to_group):
    """Map pathways to Bar-Even grouping categories."""
    if pd.isna(pathways):
        return None
    path_list = [p.strip() for p in pathways.split(";") if p.strip()]
    groups = {pathway_to_group.get(p) for p in path_list if p in pathway_to_group}
    groups.discard(None)
    return "; ".join(sorted(groups)) if groups else "ISO"


def calculate_enzyme_usage(ecmodel_file, kcat_file, uniprot_file, output_prefix):
    """Calculate enzyme usage and process data."""
    # Load ecModel
    model = cobra.io.read_sbml_model(ecmodel_file)
    solution = model.optimize()
    print(f"Growth rate: {solution.objective_value:.3f} h^-1")
    
    # Collect enzyme usage from pseudo-reactions
    enzyme_usage = []
    for rxn in model.reactions:
        if rxn.id.startswith("usage_prot_"):   # pseudoreaction
            flux_mmol_per_gDW_per_h = solution.fluxes[rxn.id]  # mmol/gDW h
            enzyme_id = rxn.id.replace("usage_", "")
            enzyme_usage.append({
                "enzyme_id": enzyme_id,
                "flux_mmol_per_gDW_per_h": flux_mmol_per_gDW_per_h
            })
    
    df_usage = pd.DataFrame(enzyme_usage)
    
    # Load kcat data
    df_kcats = pd.read_csv(kcat_file, sep=";")
    df_kcats_sorted = df_kcats.sort_values(by="kcat", ascending=True)
    df_kcats_unique = df_kcats_sorted.drop_duplicates(subset=['Gene'], keep='first')
    
    # Load UniProt data
    uniprot_df = pd.read_csv(uniprot_file, sep="\t", compression="gzip")
    uniprot_df = uniprot_df.rename(columns={"Entry": "uniprot_id", "Mass": "mass_Da"})
    uniprot_df['Gene'] = uniprot_df['KEGG'].apply(extract_gene_id)
    
    # Prepare and merge
    df_usage["uniprot_id"] = df_usage["enzyme_id"].str.replace("prot_", "", regex=False)
    df_merged = pd.merge(df_usage, uniprot_df[['uniprot_id', 'mass_Da', 'Gene']], on="uniprot_id", how="left")
    df_final = pd.merge(df_merged, df_kcats_unique[['Gene', 'kcat', 'Substrates', 'Reaction']], on="Gene", how="left")
    
    # Perform calculation
    df_final = df_final.rename(columns={"usage_mmol_per_gDW": "flux_mmol_per_gDW_per_h"})
    df_final["MW_g_per_mol"] = df_final["mass_Da"]
    df_final["concentration_mmol_per_gDW"] = df_final["flux_mmol_per_gDW_per_h"] / (df_final["kcat"] * 3600)
    df_final["mass_mg_per_gDW"] = df_final["concentration_mmol_per_gDW"] * df_final["MW_g_per_mol"]
    
    # Clean up
    df_final = df_final.dropna(subset=['mass_mg_per_gDW'])
    
    return df_final


def fetch_pathways(df, output_csv):
    """Fetch pathway information from KEGG with progress saving."""
    k = KEGG()
    
    # Load progress or start fresh
    if os.path.exists(output_csv):
        print(f"Resuming from previously saved progress: {output_csv}")
        df_with_pathways = pd.read_csv(output_csv)
    else:
        print(f"Starting a new run: {output_csv}")
        df_with_pathways = df.copy()
        if 'pathway' not in df_with_pathways.columns:
            df_with_pathways['pathway'] = None
    
    # Fetch pathways
    for index, row in df_with_pathways.iterrows():
        if pd.isna(row['pathway']):
            gene_id = row['Gene']
            print(f"Fetching pathway for Gene: {gene_id} (Row {index})...")
            
            pathway_info = get_pathways(k, f"eco:{gene_id}")
            df_with_pathways.at[index, 'pathway'] = pathway_info
            
            time.sleep(1)  # Be gentle to the API
            
            # Save progress every 50 rows
            if (index + 1) % 50 == 0:
                print(f"--- Progress saved at row {index + 1} ---")
                df_with_pathways.to_csv(output_csv, index=False)
    
    print("All done! Saving final results.")
    df_with_pathways.to_csv(output_csv, index=False)
    
    return df_with_pathways


def create_kegg_hierarchy():
    """Create KEGG hierarchy for Bar-Even grouping."""
    k = KEGG()
    
    # Bar-Even groups mapped by KEGG subcategories
    bar_even_map = {
        "CE": [
            "Carbohydrate metabolism",
            "Energy metabolism",
        ],
        "AFN": [
            "Lipid metabolism",
            "Nucleotide metabolism",
            "Amino acid metabolism",
            "Metabolism of other amino acids",
            "Glycan biosynthesis and metabolism",
        ],
        "ISO": [
            "Metabolism of cofactors and vitamins",
            "Metabolism of terpenoids and polyketides",
            "Biosynthesis of other secondary metabolites",
            "Xenobiotics biodegradation and metabolism",
        ],
    }
    
    kegg_dict = parse_kegg_brite(k.get("br:br08901"))
    
    pathway_to_group = {}
    for group, subcats in bar_even_map.items():
        for maincat, subdict in kegg_dict.items():
            for subcat in subcats:
                if subcat in subdict:
                    for p in subdict[subcat]:
                        pathway_to_group[p["name"]] = group
    
    return kegg_dict, pathway_to_group


def create_scatter_plot(ecfva_file, enzyme_df, output_prefix, pathway_to_group):
    """Create scatter plot (Figure 3C/3G)."""
    # Load ecFVA data
    ecfva = pd.read_csv(ecfva_file, sep=';')
    
    # Calculate flux variability reduction
    ecfva['flux_variability'] = (ecfva['maxFlux'] - ecfva['minFlux']) - (ecfva['ec-maxFlux'] - ecfva['ec-minFlux'])
    
    # Merge on reaction ID
    merged = pd.merge(ecfva, enzyme_df, left_on='rxnIDs', right_on='Reaction', how='inner')
    
    # Split & strip Bar-Even Grouping and pathway
    def split_and_strip(x):
        if pd.isna(x):
            return []
        return [g.strip() for g in x.split(';')]
    
    merged['Bar-Even Grouping'] = merged['Bar-Even Grouping'].apply(split_and_strip)
    merged['pathway'] = merged['pathway'].apply(split_and_strip)
    
    # Explode both columns
    merged_exploded = merged.explode('Bar-Even Grouping').explode('pathway')
    
    # E. coli pathways filter
    ecoli_pathways = [
        'Purine metabolism', 'Nucleotide metabolism', 'Pyrimidine metabolism',
        'Biosynthesis of cofactors', 'Arginine and proline metabolism',
        'Biosynthesis of amino acids', 'Citrate cycle (TCA cycle)',
        'Glyoxylate and dicarboxylate metabolism',
        'Microbial metabolism in diverse environments', 'Carbon metabolism',
        '2-Oxocarboxylic acid metabolism', 'Glutathione metabolism',
        'Lysine degradation', 'Glycolysis / Gluconeogenesis', 'Pyruvate metabolism',
        'Propanoate metabolism', 'Fructose and mannose metabolism',
        'Amino sugar and nucleotide sugar metabolism',
        'Biosynthesis of nucleotide sugars', 'Riboflavin metabolism',
        'Quorum sensing', 'Two-component system', 'Pentose phosphate pathway',
        'Glycerolipid metabolism', 'Starch and sucrose metabolism',
        'Biofilm formation - Escherichia coli', 'Porphyrin metabolism',
        'Valine, leucine and isoleucine biosynthesis',
        'Pantothenate and CoA biosynthesis',
        'Glycine, serine and threonine metabolism',
        'Phenylalanine, tyrosine and tryptophan biosynthesis',
        'Oxidative phosphorylation', 'Butanoate metabolism',
        'Terpenoid backbone biosynthesis', 'Lipopolysaccharide biosynthesis',
        'Peptidoglycan biosynthesis', 'RNA degradation',
        'Aminoacyl-tRNA biosynthesis', 'Thiamine metabolism',
        'Pentose and glucuronate interconversions', 'One carbon pool by folate',
        'Folate transport and metabolism', 'C5-Branched dibasic acid metabolism',
        'Cysteine and methionine metabolism', 'Vitamin B6 metabolism',
        'Alanine, aspartate and glutamate metabolism',
        'Nicotinate and nicotinamide metabolism', 'Fatty acid biosynthesis',
        'Fatty acid metabolism', 'Ubiquinone and other terpenoid-quinone biosynthesis',
        'Biosynthesis of siderophore group nonribosomal peptides',
        'Ascorbate and aldarate metabolism', 'Arginine biosynthesis',
        'Biosynthesis of various nucleotide sugars', 'Lipoic acid metabolism',
        'Valine, leucine and isoleucine degradation', 'Tryptophan metabolism',
        'Phenylalanine metabolism', 'Degradation of aromatic compounds',
        'Galactose metabolism', 'Lysine biosynthesis', 'Folate biosynthesis',
        'Nitrogen metabolism', 'Sulfur metabolism',
        'Phosphotransferase system (PTS)', 'D-Amino acid metabolism',
        'Benzoate degradation', 'beta-Alanine metabolism', 'Biotin metabolism',
        'Histidine metabolism', 'Exopolysaccharide biosynthesis',
        'Inositol phosphate metabolism', 'Taurine and hypotaurine metabolism',
        'Fatty acid degradation', 'Selenocompound metabolism',
        'O-Antigen repeat unit biosynthesis', 'ABC transporters',
        'Flagellar assembly', 'Cationic antimicrobial peptide (CAMP) resistance',
        'Tyrosine metabolism', 'beta-Lactam resistance',
        'Cyanoamino acid metabolism', 'Bacterial chemotaxis',
        'Aminobenzoate degradation', 'Biosynthesis of unsaturated fatty acids',
        'Ether lipid metabolism', 'Sulfur relay system',
        'Cobalamin transport and metabolism', 'Phosphonate and phosphinate metabolism'
    ]
    
    merged_exploded = merged_exploded[
        merged_exploded['pathway'].isin(ecoli_pathways)
    ].reset_index(drop=True)
    
    # Aggregate by pathway
    agg_df = merged_exploded.groupby('pathway').agg({
        'mass_mg_per_gDW': 'sum',           # total enzyme usage
        'flux_variability': 'mean',         # average flux reduction
        'Bar-Even Grouping': 'first'        # for coloring
    }).reset_index()
    
    # Log-transform total enzyme usage for X-axis
    agg_df['log_mass'] = np.log10(agg_df['mass_mg_per_gDW'] + 1e-6)
    
    # Assign colors
    color_map = {
        'CE': sns.color_palette(palette='Set1')[0],
        'AFN': sns.color_palette(palette='Set1')[1],
        'ISO': sns.color_palette(palette='Set1')[2]
    }
    
    # Plot scatter
    plt.figure(figsize=(5.5, 5))
    sns.scatterplot(data=agg_df, x='mass_mg_per_gDW', y='flux_variability',
                    hue='Bar-Even Grouping', palette=color_map, s=50, edgecolor='k', alpha=0.7)
    
    # Calculate all group means
    group_means = []
    highest_mean_y = 0
    highest_mean_x_log = -np.inf
    
    for group in agg_df['Bar-Even Grouping'].unique():
        group_df = agg_df[agg_df['Bar-Even Grouping'] == group]
        mean_x_log = np.mean(np.log10(group_df['mass_mg_per_gDW'] + 1e-6))
        mean_y = np.mean(group_df['flux_variability'])
        group_means.append({'group': group, 'mean_x_log': mean_x_log, 'mean_y': mean_y})
        
        if mean_y > highest_mean_y:
            highest_mean_y = mean_y
        if mean_x_log > highest_mean_x_log:
            highest_mean_x_log = mean_x_log
    
    highest_mean_x = 10**highest_mean_x_log
    
    # Add discontinuous lines and diamond markers
    for item in group_means:
        group = item['group']
        mean_x = 10**item['mean_x_log']
        mean_y = item['mean_y']
        color = color_map[group]
        
        plt.axvline(mean_x, linestyle='--', color=color, alpha=0.7)
        plt.axhline(mean_y, linestyle='--', color=color, alpha=0.7)
        plt.scatter(mean_x, mean_y, marker='D', color='none', s=50, edgecolor=color, zorder=10)
    
    # Axis formatting
    plt.xscale("log")
    plt.yscale("log")
    
    if output_prefix == "3C":
        plt.xlim(1e-5, 1e5)
        plt.ylim(1e-1, 1e3)
    else:  # 3G
        plt.xlim(1e-5, 1e5)
        plt.ylim(1e-5, 1e5)
    
    # Add light grey background region
    ax = plt.gca()
    xlim_max = ax.get_xlim()[1]
    ylim_max = ax.get_ylim()[1]
    facecolor_val = 'lightgrey'
    alpha_val = 0.5
    ax.fill_between([highest_mean_x, xlim_max], [highest_mean_y, highest_mean_y], 
                    [ylim_max, ylim_max], color=facecolor_val, alpha=alpha_val, zorder=0)
    
    # Add minor ticks
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=100))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    plt.tick_params(axis='x', which='both', direction='in')
    plt.tick_params(axis='y', which='both', direction='in')
    
    plt.xlabel('Total pathway usage (mg/gDW)')
    plt.ylabel('Average flux variability reduction (mmol/gDW h)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}.png', dpi=1000)
    plt.close()
    
    # Filter highlighted region
    highlighted_df = agg_df[
        (agg_df['mass_mg_per_gDW'] > highest_mean_x) &
        (agg_df['flux_variability'] > highest_mean_y)
    ]
    
    highlighted_df.to_csv(f'highlighted_data_{output_prefix}.csv', index=False)
    
    return highlighted_df, color_map


def create_bar_plot(highlighted_df, color_map, output_prefix):
    """Create bar plot (Figure 3D/3H)."""
    # Prepare DataFrame for plotting
    plot_df = highlighted_df.sort_values('flux_variability', ascending=False)
    
    # Create horizontal bar plot
    plt.figure(figsize=(5.5, 5))
    ax = sns.barplot(
        x='flux_variability',
        y='pathway',
        data=plot_df,
        palette=color_map,
        hue='Bar-Even Grouping',
        dodge=False,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Add value labels
    for index, value in enumerate(plot_df['flux_variability']):
        ax.text(value + 20, index, f'{value:.1f}', color='black', ha="left", va="center")
    
    # Formatting
    ax.set_title('Average Variability Reduction [mmol/gDWh]', fontsize=9)
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    
    ax.get_xaxis().set_visible(False)
    sns.despine(ax=ax, top=True, right=True, left=True, bottom=True)
    
    # Set background color
    fig = ax.get_figure()
    fig.patch.set_facecolor('#E9E9E9')
    fig.patch.set_alpha(0.5)
    ax.patch.set_alpha(0.0)
    
    # Add white border
    fig.patch.set_edgecolor('white')
    fig.patch.set_linewidth(40)
    fig.patch.set_alpha(1.0)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}.png', dpi=1000, bbox_inches='tight', 
                pad_inches=0.5, facecolor=fig.get_facecolor(), 
                edgecolor='white', transparent=False)
    plt.close()


# ============================================================================
# MAIN ANALYSIS FUNCTIONS
# ============================================================================

def analyze_model(gem_file, ecmodel_file, ecfva_file, kcat_file, uniprot_file, 
                  model_name, sampling_method="achr"):
    """Run complete analysis for one model."""
    print(f"\n{'='*60}")
    print(f"ANALYZING MODEL: {model_name}")
    print(f"{'='*60}")
    
    # Figure prefix mapping
    if model_name == "iML1515":
        figure_prefixes = {"pca": "3A", "ecdf": "3B", "scatter": "3C", "bar": "3D"}
        pathways_csv = "pathways_progress.csv"
        kegg_hierarchy_json = "kegg_hierarchy_2025.json"
        highlighted_prefix = ""
    else:  # iAF1260
        figure_prefixes = {"pca": "3E", "ecdf": "3F", "scatter": "3G", "bar": "3H"}
        pathways_csv = "pathways_progress_iAF1260.csv"
        kegg_hierarchy_json = "kegg_hierarchy_2025_iAF1260.json"
        highlighted_prefix = "_iAF1260"
    
    # 1. Load models
    print("\n1. Loading models...")
    model_gem = cobra.io.read_sbml_model(gem_file)
    model_ec = cobra.io.read_sbml_model(ecmodel_file)
    
    print(f"GEM growth: {model_gem.optimize().objective_value}")
    print(f"ecGEM growth: {model_ec.optimize().objective_value}")
    
    # 2. Flux sampling
    print(f"\n2. Flux sampling ({sampling_method})...")
    n_samples = 10000
    print("Sampling GEM...")
    samples_gem = sample(model_gem, n=n_samples, method=sampling_method, processes=8)
    print("Sampling ecGEM...")
    samples_ec = sample(model_ec, n=n_samples, method=sampling_method, processes=8)
    
    # 3. Clean data
    print("\n3. Cleaning and matching flux samples...")
    samples_gem, samples_ec, common_rxns = clean_flux_samples(samples_gem, samples_ec)
    print(f"Reactions analyzed: {len(common_rxns)}")
    
    # 4. Statistical test
    print("\n4. Performing statistical test...")
    results_df = perform_statistical_test(
        samples_gem, samples_ec, common_rxns, 
        f"flux_significance_test_ec{model_name}.csv"
    )
    
    # 5. PCA analysis
    print("\n5. Running PCA analysis...")
    run_pca_analysis(samples_gem, samples_ec, 
                     (model_name, f"ec{model_name}"), 
                     figure_prefixes["pca"])
    
    # 6. Flux variability ECDF
    print("\n6. Creating flux variability ECDF plot...")
    plot_ecFVA_from_csv(ecfva_file, (model_name, f"ec{model_name}"), 
                       figure_prefixes["ecdf"])
    
    # 7. Calculate enzyme usage
    print("\n7. Calculating enzyme usage...")
    enzyme_df = calculate_enzyme_usage(ecmodel_file, kcat_file, uniprot_file, model_name)
    
    # 8. Fetch pathways
    print("\n8. Fetching pathway information from KEGG...")
    enzyme_df_with_pathways = fetch_pathways(enzyme_df, pathways_csv)
    
    # 9. Create KEGG hierarchy and Bar-Even grouping
    print("\n9. Creating KEGG hierarchy and Bar-Even grouping...")
    kegg_dict, pathway_to_group = create_kegg_hierarchy()
    
    # Save KEGG hierarchy
    with open(kegg_hierarchy_json, "w") as f:
        json.dump(kegg_dict, f, indent=2, ensure_ascii=False)
    
    # Apply Bar-Even grouping
    enzyme_df_with_pathways["Bar-Even Grouping"] = enzyme_df_with_pathways["pathway"].apply(
        lambda x: map_to_bar_even(x, pathway_to_group)
    )
    
    # 10. Create scatter plot
    print("\n10. Creating scatter plot...")
    highlighted_df, color_map = create_scatter_plot(
        ecfva_file, enzyme_df_with_pathways, 
        figure_prefixes["scatter"], pathway_to_group
    )
    
    # 11. Create bar plot
    print("\n11. Creating bar plot...")
    create_bar_plot(highlighted_df, color_map, figure_prefixes["bar"])
    
    print(f"\nAnalysis for {model_name} completed successfully!")
    print(f"Output files saved with prefix: {figure_prefixes}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print(__doc__)
    
    # Check for required inputs
    required_files = [
        ("iML1515.xml", "iML1515 GEM model"),
        ("iML1515_ec_duplicated.xml", "iML1515 ecModel"),
        ("ecFVA.csv", "iML1515 flux variability data"),
        ("kcats.csv", "iML1515 kcat values"),
        ("iAF1260.xml", "iAF1260 GEM model"),
        ("iAF1260_ec_duplicated.xml", "iAF1260 ecModel"),
        ("ecFVA_iAF1260.csv", "iAF1260 flux variability data"),
        ("kcats_iAF1260.csv", "iAF1260 kcat values"),
        ("uniprotkb_taxonomy_id_83333_2025_10_14.tsv.gz", "UniProt database")
    ]
    
    print("\nChecking for required input files...")
    missing_files = []
    for file, description in required_files:
        if not os.path.exists(file):
            missing_files.append((file, description))
            print(f"  ❌ Missing: {file} ({description})")
        else:
            print(f"  ✓ Found: {file}")
    
    if missing_files:
        print(f"\nERROR: {len(missing_files)} required file(s) are missing!")
        print("Please ensure all input files are in the current directory.")
        return
    
    print("\nAll required files found. Starting analysis...")
    
    # Analyze iML1515
    analyze_model(
        gem_file="iML1515.xml",
        ecmodel_file="iML1515_ec_duplicated.xml",
        ecfva_file="ecFVA.csv",
        kcat_file="kcats.csv",
        uniprot_file="uniprotkb_taxonomy_id_83333_2025_10_14.tsv.gz",
        model_name="iML1515",
        sampling_method="achr"
    )
    
    # Analyze iAF1260
    analyze_model(
        gem_file="iAF1260.xml",
        ecmodel_file="iAF1260_ec_duplicated.xml",
        ecfva_file="ecFVA_iAF1260.csv",
        kcat_file="kcats_iAF1260.csv",
        uniprot_file="uniprotkb_taxonomy_id_83333_2025_10_14.tsv.gz",
        model_name="iAF1260",
        sampling_method="optgp"
    )
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated output files:")
    print("  Figures: 3A.png, 3A.svg, 3B.png, 3C.png, 3D.png,")
    print("           3E.png, 3F.png, 3G.png, 3H.png")
    print("  Data files: flux_significance_test_*.csv")
    print("              pathways_progress*.csv")
    print("              highlighted_data*.csv")
    print("              kegg_hierarchy_*.json")


if __name__ == "__main__":
    main()