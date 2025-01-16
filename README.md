# Knoop: Practical Enhancement of Knockoff with Over-Parameterization for Variable Selection

This repository contains the code and data for the paper "Knoop: Practical Enhancement of Knockoff with Over-Parameterization for Variable Selection", which was accepted for publication in Springer Machine Learning on December 12, 2024.

## Table of Contents
- [Background](#background)
- [Data Sources](#data-sources)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)
- [References](#references)

## Background
In this project, we present an enhancement to the Knockoff filter method using over-parameterization for improved variable selection in high-dimensional settings. The details can be found in our [paper](https://openreview.net/forum?id=jWeyv03uXM&noteId=jWeyv03uXM).

## Data Sources
- Alon dataset: [CRAN AlonDS](https://search.r-project.org/CRAN/refmans/HiDimDA/html/AlonDS.html)
- Superconductivity dataset, Energy dataset, Community and Crime dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/datasets)

## Installation
The following Python libraries are required to run the code in this repository:

- choldate==0.1.0
- knockpy==1.3.1
- matplotlib
- pandas
- scikit-learn
- scipy==1.10.1
- seaborn==0.13.2
- skfeature==1.0.0
- statsmodels
- torch==2.3.0
- torchaudio==2.3.0+cu121
- torchvision==0.18.0
- tqdm
- xgboost==2.1.0

You can install these dependencies using:
```bash
pip install -r requirements.txt
```
Additionally, the following R package is required:
- multiknockoffs:  https://github.com/cKarypidis/multiknockoffs

To install the R package, run:
```R
# In an R console
install.packages("devtools")
devtools::install_github("cKarypidis/multiknockoffs")
```

## Usage
To run the simulation experiments, use the following command:
```bash
python scripts/simulations/classical scenarios/p=80,p_real=10,n=100/repeats.py
```
To run the real dataset experiments, use the following command:
```bash
python scripts/real_data_experiments/superconductivity/get_everything_about_BestKFeatures.py
```
Make sure to prepare the data according to the instructions in the data/readme.txt.

## Project Structure
```plaintext
Knoop-Knockoff-Enhancement/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── Alon dataset/
│   ├── community_and_crime_dataset/
│   ├── energy_data_dataset/
│   └── superconductivity_dataset/
├── scripts/
│   ├── simulations/
│   │   ├── classical_scenarios/
│   │   └── HDLSS_scenarios/
│   └── real_data_experiments/
│       ├── Alon_dataset/
│       ├── community_and_crime_dataset/
│       ├── energy_data_dataset/
│       └── superconductivity_dataset/
└── results/
    ├── simulations/
    └── real_data_experiments/
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## References
- Original Paper: Zhang, X., Cai, Y., Xiong, H.: Practical Enhancement of Knockoff with Over-Parameterization for Variable Selection. Accepted for publication in Springer Machine Learning, (2024).


