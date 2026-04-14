# EINN: Enzyme-Informed Neural Network

![Nextflow](https://img.shields.io/badge/Nextflow-22.10+-23aa62.svg)
![MATLAB](https://img.shields.io/badge/MATLAB-R2023b-0076A8.svg)
![Python](https://img.shields.io/badge/Python-3.10-3776AB.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00.svg)
![GECKO](https://img.shields.io/badge/GECKO-3.0-4B9CD3.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🧬 Abstract

**EINN (Enzyme-Informed Neural Network)** is a unified framework that bridges enzyme-constrained genome-scale metabolic models (ecGEMs) with deep learning architectures for biologically grounded metabolic prediction. Based on the manuscript *"Enzyme-informed neural network framework integrating enzyme-constrained metabolism for biologically grounded hybrid modeling"*, this framework demonstrates that integrating enzyme constraints into neural-mechanistic models significantly improves:

- **Prediction accuracy** (78.5% → 13% non-predictive models)
- **Training stability** (compact convergence vs. multimodal distributions)
- **Biological realism** (accurate overflow metabolism recapitulation)
- **Resource allocation** (pathway-specific flux variability reduction)

This repository contains the complete pipeline for constructing and preprocessing enzyme-constrained GEMs (`main.nf`), as well as Jupyter notebook-based training workflows for the **ecAMN** and **ecMINN** models located in the `training/` folder.

---

## ✨ Key Features

### 🔬 ecGEM Construction (`main.nf`)
- Automated GECKO 3.0 pipeline for ecModel construction compatible with hybrid-mechanistic models
- Integrated preprocessing: reversibility fixing, reaction cleaning, and Biolog-based exchange alignment
- Reaction duplication for forward/reverse flux separation (required by AMN/MINN architectures)
- Comparative flux variability analysis (ecFVA) with publication-quality figures

### 🧠 Neural-Mechanistic Integration (`training/`)
- **ecAMN**: Enzyme-constrained Artificial Metabolic Network for growth rate prediction
- **ecMINN**: Enzyme-constrained Metabolic-Informed Neural Network for multi-omics integration
- Multiple data integration strategies (Early, Intermediate, F*)
- Reservoir-based training using synthetic FBA simulation data
- Mechanistic integration of proteomic (enzyme usage) constraints

---

## 🚀 Part 1: ecGEM Construction with Nextflow

The `main.nf` pipeline handles the full construction and preprocessing of enzyme-constrained GEMs for two *E. coli* models: **eciML1515** and **eciAF1260**.

### Pipeline Overview

```
GECKO_PIPELINE
    ↓
COMPARATIVE_ANALYSIS          ← runs in parallel after GECKO
    ↓
FIX_REVERSIBILITY
    ↓
CLEAN_REACTIONS
    ↓
ALIGN_BIOLOG_EC / ALIGN_BIOLOG_CONV    ← eciML1515 only
    ↓
DUPLICATE_MODEL
```

| Process | Description |
|---|---|
| `GECKO_PIPELINE` | Runs GECKO 3.0 via MATLAB to build raw ecGEMs and generate kcats + ecFVA outputs |
| `COMPARATIVE_ANALYSIS` | Generates Figures 3A–3H comparing ec vs. conventional model flux distributions |
| `FIX_REVERSIBILITY` | Aligns reaction reversibility between the ecModel and conventional model |
| `CLEAN_REACTIONS` | Removes artefact reactions incompatible with AMN/MINN stoichiometry |
| `ALIGN_BIOLOG_EC` / `ALIGN_BIOLOG_CONV` | Aligns exchange reactions with experimental Biolog carbon source data |
| `DUPLICATE_MODEL` | Duplicates reactions into forward/reverse pairs for neural network compatibility |

### Prerequisites

- [Nextflow](https://www.nextflow.io/) ≥ 22.10
- MATLAB R2023b with [GECKO Toolbox 3.0](https://github.com/SysBioChalmers/GECKO)
- Gurobi (licensed) — path set via `params.license`
- Python 3.10 with `cobrapy` and dependencies (see `nextflow.config` for conda environments)

### Running the Pipeline

```bash
# Clone the repository
git clone https://github.com/your-org/EINN.git
cd EINN

# Run with default parameters
nextflow run main.nf

# Override training/FVA parameters as needed
nextflow run main.nf \
  --epochs 50 \
  --folds 5 \
  --outer_loops 300 \
  --license /path/to/gurobi.lic \
  --outdir results/
```

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `--input_base` | `./data` | Root directory for all input data |
| `--outdir` | `results` | Directory for pipeline outputs |
| `--ec_params` | `params.csv` | GECKO enzyme parameters CSV |
| `--license` | `.` | Path to Gurobi license file |
| `--epochs` | `20` | Training epochs (passed to downstream use) |
| `--folds` | `5` | Cross-validation folds |
| `--outer_loops` | `200` | Outer reservoir training loops |
| `--timestep` | `4` | ODE integration timestep |
| `--mediumbound` | `UB` | Medium bound type (`UB` or `LB`) |

### Pipeline Outputs

Results are written to `results/` with the following structure:

```
results/
├── eciML1515/
│   ├── eciML1515_final.xml             # Final preprocessed ecGEM (duplicated)
│   ├── eciML1515_conventional_dup.xml  # Duplicated conventional model
│   ├── eciML1515_kcats_final.csv       # Per-reaction kcat values
│   └── eciML1515_ecFVA_final.csv       # Flux variability analysis results
├── eciAF1260/
│   ├── eciAF1260_final.xml
│   ├── eciAF1260_conventional_dup.xml
│   ├── eciAF1260_kcats_final.csv
│   └── eciAF1260_ecFVA_final.csv
└── comparative_analysis/
    ├── 3A.svg / 3A.png                 # ecFVA comparison figures
    ├── 3B.png ... 3H.png
    ├── flux_significance_test_*.csv
    ├── pathways_progress*.csv
    ├── highlighted_data*.csv
    └── kegg_hierarchy_*.json
```

---

## 🧠 Part 2: Model Training (Jupyter Notebooks)

Training of the ecAMN and ecMINN models is performed interactively via Jupyter notebooks located in the `training/` directory. Each model has its own subfolder with a dedicated Conda environment.

> **Note:** The Nextflow pipeline must be completed first. The `*_final.xml` models and `*_kcats_final.csv` outputs from the pipeline are required as inputs to the training notebooks.

---

### ecAMN Training (`training/ecAMN/`)

The **ecAMN (Enzyme-Constrained Artificial Metabolic Network)** predicts microbial growth rates by embedding stoichiometric constraints from the ecGEM directly into the network architecture.

#### Environment Setup

```bash
cd training/ecAMN
conda env create -f amn_env.yml
conda activate amn_env
jupyter notebook
```

#### Notebooks

Open and run the notebooks in `training/ecAMN/` sequentially. They cover:

1. **Data preparation** — loading FBA simulation data and ecGEM stoichiometry
2. **Model construction** — building the enzyme-constrained AMN architecture
3. **Training** — reservoir-based and gradient-based training loops
4. **Evaluation** — cross-validated growth prediction performance and flux analysis

---

### ecMINN Training (`training/ecMINN/`)

The **ecMINN (Enzyme-Constrained Metabolic-Informed Neural Network)** extends the AMN framework to integrate multi-omics data (e.g., transcriptomics, proteomics) with metabolic mechanistic constraints.

#### Environment Setup

```bash
cd training/ecMINN
conda env create -f minn_env.yml
conda activate minn_env
jupyter notebook
```

#### Notebooks

Open and run the notebooks in `training/ecMINN/` sequentially. They cover:

1. **Data integration** — multi-omics preprocessing and alignment with ecGEM reactions
2. **Model construction** — MINN architecture with enzyme usage constraints
3. **Training** — early/intermediate/F* integration strategy comparisons
4. **Evaluation** — prediction accuracy, training stability, and biological validation

---

## 📚 Dependencies

| Tool | Version | Purpose |
|---|---|---|
| [Nextflow](https://www.nextflow.io/) | ≥ 22.10 | Workflow orchestration |
| [GECKO Toolbox](https://github.com/SysBioChalmers/GECKO) | 3.0 | Enzyme-constrained model construction |
| [MATLAB](https://www.mathworks.com/) | R2023b | GECKO pipeline execution |
| [Gurobi](https://www.gurobi.com/) | — | LP/FBA solver (license required) |
| [cobrapy](https://github.com/opencobra/cobrapy) | — | Constraint-based modeling (Python) |
| [TensorFlow](https://github.com/tensorflow/tensorflow) | 2.13 | Neural network implementation |
| [AMN](https://github.com/brsynth/amn_release) | — | Base AMN architecture |
| [MINN](https://github.com/gabrieletaz/MINN) | — | Base MINN architecture |

Python dependencies for training are fully specified in `training/ecAMN/amn_env.yml` and `training/ecMINN/minn_env.yml`.

---

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex

```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

**Ray Steven**  
Graduate School of Natural Science and Technology  
Kanazawa University, Kanazawa 9201192, Japan  
📧 raysteven127@gmail.com
