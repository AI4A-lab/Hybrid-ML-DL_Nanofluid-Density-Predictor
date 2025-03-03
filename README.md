# Hybrid Nanofluid Density Prediction via Computational Intelligence

This repository contains the implementation of machine learning and deep learning models for predicting hybrid nanofluid density using advanced data augmentation and meta-heuristic optimization techniques. The methods presented here are detailed in the research paper:

### [A Computational Intelligence Framework Integrating Data Augmentation and Meta-Heuristic Optimization Algorithms for Enhanced Hybrid Nanofluid Density Prediction Through Machine and Deep Learning Paradigms](https://doi.org/10.1109/ACCESS.2025.3543475)

- **Published in:** IEEE Access  
- **Publication Date:** 19 February 2025  

#### Links:

- **[Dataset](https://github.com/AI4A-lab/Hybrid-Nanofluid-Density-Prediction-Dataset)**
- **[Journal Homepage](https://ieeeaccess.ieee.org/)**
- **[Scopus](https://www.scopus.com/sourceid/21100374601)**

#### Abstract
<p align="justify">
This research presents a robust and comprehensive framework for predicting the density of hybrid nanofluids using state-of-the-art machine learning and deep learning techniques. Addressing the limitations of conventional empirical approaches, the study used a curated dataset of 436 samples from the peer-reviewed literature, which includes nine input parameters such as the nanoparticle, base fluid, temperature (°C), volume concentration (φ), base fluid density (ρbf), density of primary and secondary nanoparticles (ρnp1 and ρnp2), and volume mixture ratios of primary and secondary nanoparticles. Data preprocessing involved outlier removal via the Interquartile Range (IQR) method, followed by augmentation using either autoencoder-based or Gaussian noise injection, which preserved statistical integrity and enhanced dataset diversity. The research analyzed fourteen predictive models, employing advanced hyperparameter optimization methods facilitated by Grey Wolf Optimization (GWO) and Particle Swarm Optimization (PSO). In particular, autoencoder-based augmentation combined with hyperparameter optimization consistently improved predictive accuracy across all models. For machine learning models, Gradient Boosting achieved the most remarkable performance, with R2 scores of 0.99999 and minimal MSE values of 0.00091. Among deep learning models, Recurrent Neural Networks (RNN) stacked with Linear Regression achieved superior performance with an R2 of 0.9999, MSE of 0.0014, and MAE of 0.012. The findings underscore the synergy of advanced data augmentation, meta-heuristic optimization, and modern predictive algorithms in modelling hybrid nanofluid density with unprecedented precision. This framework offers a scalable and reliable tool for advancing nanofluid-based applications in thermal engineering and related domains.
</p>



## Table of Contents

- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Augmentation Techniques](#augmentation-techniques)
- [Optimization Algorithms](#optimization-algorithms)
- [Implemented Models](#implemented-models)
- [Results and Discussion](#results-and-discussion)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)


## Repository Structure

```
Hybrid-Nanofluid-Density-Prediction/
├── Code/
│   ├── 1_Baseline                      # Baseline code.
│   ├── 2_Augmentation            
│   │   ├── Augmentation_Check          # Code for various different augmentations.
│   │   ├── 2_A_Augmentation            # First best augmentation (Autoencoder).
│   │   ├── 2_B_Augmentation            # Second best augmentation (Gaussian Noise).
|   |── 3_Optimization
│   │   ├── Optimization_Check          # Code for various different optimizations.
│   │   ├── 3_A_Optimization            # First best augmentation (GWO).
│   │   ├── 3_B_Optimization            # Second best augmentation (PSO).
|   |── Graphs                          # Code for graphs generation.
│   └── requirements.txt                # Python dependencies.
├── Results/                            # Output files: Tables and metrics.
└── README.md                           # This file
```


## Introduction

The goal of this project is to provide a comprehensive framework for predicting hybrid nanofluid density using advanced computational intelligence techniques. The repository includes code for data preprocessing, augmentation, model training, and optimization, supporting both machine learning and deep learning paradigms. This approach addresses the inherent challenges of modeling complex thermophysical properties with high accuracy and robustness.



## Dataset

The experimental dataset consists of **436 samples** extracted from peer-reviewed literature. Each sample is characterized by:

- **Input Parameters:** Nanoparticle type, base fluid, temperature (°C), volume concentration (φ), base fluid density (ρ<sub>bf</sub>), density of primary (ρ<sub>np1</sub>) and secondary nanoparticles (ρ<sub>np2</sub>), and volume mixture ratios.
- **Output Parameter:** Hybrid nanofluid density (ρ<sub>hybrid</sub>).

Dataset is available at: [Hybrid-Nanofluid-Density-Prediction-Dataset](https://github.com/AI4A-lab/Hybrid-Nanofluid-Density-Prediction-Dataset)



## Methodology

The workflow is divided into two key phases:

1. **Data Preprocessing:**  
   - Cleaning and normalization of data using techniques such as outlier removal via the IQR method.  
   - Binary vector encoding for categorical variables.

2. **Model Training and Optimization:**  
   - Implementation of fourteen predictive models spanning machine learning and deep learning architectures.  
   - Integration of advanced data augmentation (autoencoder-based and Gaussian noise injection) to improve training data diversity.  
   - Application of meta-heuristic optimization algorithms (GWO and PSO) to fine-tune model hyperparameters.



## Augmentation Techniques

- **Autoencoder-Based Augmentation:**  
  Utilizes an autoencoder network to generate synthetic samples by reconstructing input data, thereby enhancing the training set while preserving statistical properties.

- **Gaussian Noise Injection:**  
  Expands the dataset by adding controlled Gaussian noise to input samples, improving model robustness against real-world variations.



## Optimization Algorithms

To fine-tune model performance, two meta-heuristic optimization techniques are employed:

- **Grey Wolf Optimization (GWO):**  
  Inspired by the social hierarchy and hunting behavior of grey wolves, this algorithm efficiently explores the hyperparameter space to find optimal configurations.

- **Particle Swarm Optimization (PSO):**  
  Mimics the social behavior of bird flocking and fish schooling, iteratively updating candidate solutions to achieve high accuracy in model predictions.



## Implemented Models

The repository includes implementations of fourteen predictive models and their ensemble variant (stacked with Linear Regression), the models included in the study are as follows:

- **Machine Learning Models:**  
  - Decision Tree  
  - Random Forest  
  - Ridge Regression  
  - Poisson Regression  
  - Gradient Boosting  
  - LightGBM  
  - CatBoost  
  - XGBoost

- **Deep Learning Models:**  
  - Neural Networks  
  - Convolutional Neural Networks (CNN)  
  - Recurrent Neural Networks (RNN)  
  - Gated Recurrent Units (GRU)  
  - Long Short-Term Memory (LSTM)  
  - Autoencoders

Each model and their variant are evaluated based on performance metrics such as R2 Score, Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), MAPE, SMAPE, EVS, and Max Error.



## Results and Discussion

Experimental results show that the integration of autoencoder-based augmentation and meta-heuristic optimization significantly enhances the predictive performance of the models. Particularly, autoencoder-based augmentation combined with hyperparameter optimization consistently improved predictive accuracy across all models. For machine learning models, Gradient Boosting achieved the most remarkable performance, with R2 scores of 0.99999 and minimal MSE values of 0.00091. Among deep learning models, Recurrent Neural Networks (RNN) stacked with Linear Regression achieved superior performance with an R2 of 0.9999, MSE of 0.0014, and MAE of 0.012. Detailed analyses, comparative evaluations, and visualization outputs are provided in the paper's Results section.



## Installation

### Prerequisites

- Python >= 3.10 
- Scikit-Learn >= 1.3.2 (for machine learning implementations)  
- PyTorch >= 2.4.1 (for deep learning implementations)  
- Keras (for deep learning implementations)  
- CUDA-enabled GPU (recommended for acceleration)

For the complete list, please refer to the ```requirements.txt``` file.

### Setup

Clone the repository:

```bash
git clone https://github.com/AI4A-lab/Hybrid-ML-DL_Nanofluid-Density-Predictor.git
cd Hybrid-ML-DL_Nanofluid-Density-Predictor
```


## Citation

If you find this code useful for your research and applications or if you are using the dataset, please cite using this BibTeX:
```bibtex
@ARTICLE{10892114,
  author={Mathur, Priya and Shaikh, Hammad and Sheth, Farhan and Kumar, Dheeraj and Gupta, Amit Kumar},
  journal={IEEE Access}, 
  title={A Computational Intelligence Framework Integrating Data Augmentation and Meta-Heuristic Optimization Algorithms for Enhanced Hybrid Nanofluid Density Prediction Through Machine and Deep Learning Paradigms}, 
  year={2025},
  volume={13},
  number={},
  pages={35750-35779},
  keywords={Autoencoders;Nanoparticles;Predictive models;Machine learning;Fluids;Prediction algorithms;Gaussian noise;Deep learning;Data models;Data augmentation;Density prediction;hybrid nanofluids;machine learning;deep learning;data augmentation;meta-heuristic optimization;optimization algorithms;thermal engineering},
  doi={10.1109/ACCESS.2025.3543475}}
```

## Authors

- FARHAN SHETH ([GitHub](https://github.com/Phantom-fs), [LinkedIn](https://www.linkedin.com/in/farhan-sheth/))
- HAMMAD SHAIKH ([GitHub](https://github.com/Hammad-1105), [LinkedIn](https://www.linkedin.com/in/hammad-shaikh1105/))
- DHEERAJ KUMAR ([GitHub](https://github.com/Dheeraj21K), [LinkedIn](https://www.linkedin.com/in/dheeraj-kumar-bb1184262/))


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.