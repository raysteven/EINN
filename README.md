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

This repository contains the complete pipeline for constructing, preprocessing, and training enzyme-constrained hybrid models, along with comprehensive validation and visualization tools.

## ✨ Key Features

### 🔬 **ecGEM Construction**
- Automated GECKO 3.0 pipeline for ecModel construction that is compatible with hybrid-mechanistic models
- Integrated preprocessing (reversibility fixing, reaction cleaning)
- Comparative flux variability analysis and visualization

### 🧠 **Neural-Mechanistic Integration**
- **ecAMN**: Enzyme-constrained Artificial Metabolic Network for growth prediction
- **ecMINN**: Enzyme-constrained Metabolic-Informed Neural Network for multi-omics integration
- Multiple data integration strategies (Early, Intermediate, F*)
- Reservoir-based training with synthetic FBA data
- Mechanistic integration of proteomic constraints


## 🚀 Dependencies

- **[GECKO Toolbox 3.0](https://github.com/SysBioChalmers/GECKO)**: Enzyme-constrained model construction
- **[cobrapy](https://github.com/opencobra/cobrapy)**: Constraint-based modeling in Python
- **[TensorFlow](https://github.com/tensorflow/tensorflow)**: Neural network implementation
- **[Nextflow](https://github.com/nextflow-io/nextflow)**: Workflow orchestration
- **[AMN](https://github.com/brsynth/amn_release)**: Base AMN code and implementation
- **[MINN](https://github.com/gabrieletaz/MINN)**: Base MINN code and implementation


## 📚 Citation

If you use this framework in your research, please cite:

```bibtex

```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 📬 Contact

**Ray Steven**  
Graduate School of Natural Science and Technology  
Kanazawa University, Kanazawa 9201192, Japan  
📧 raysteven127@gmail.com
