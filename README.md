<h1 align="center">Stacking Ensemble Learning with Neural Network Meta Learner (SEL-NNML) for Heart Disease Prediction</h1>

<p align="center">
  <a href="https://www.python.org" target="_blank"> <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"></a>
  <a href="https://pandas.pydata.org/" target="_blank"> <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"></a>
  <a href="https://scikit-learn.org/" target="_blank"> <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"></a>
  <a href="https://matplotlib.org/" target="_blank"> <img src="https://img.shields.io/badge/Matplotlib-000000?style=for-the-badge&logo=matplotlib&logoColor=white"></a>
  <a href="https://seaborn.pydata.org/" target="_blank"> <img src="https://img.shields.io/badge/Seaborn-80b6ff?style=for-the-badge&logo=seaborn&logoColor=white"></a>
  <a href="https://optuna.org/" target="_blank"> <img src="https://img.shields.io/badge/Optuna-00468B?style=for-the-badge&logo=optuna&logoColor=white"></a>
  <a href="https://joblib.readthedocs.io/en/latest/" target="_blank"> <img src="https://img.shields.io/badge/Joblib-00468B?style=for-the-badge&logo=joblib&logoColor=white"></a>
  <a href="https://pypi.org/project/kagglehub/" target="_blank"> <img src="https://img.shields.io/badge/KaggleHub-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white"></a>
  <a href="https://pypi.org/project/ucimlrepo/" target="_blank"> <img src="https://img.shields.io/badge/UCIMLRepo-6C63FF?style=for-the-badge&logo=Python&logoColor=white"></a>
</p>

This repository documents a machine learning research project focused on **Heart Disease Prediction** using a **Stacking Ensemble Learning with Neural Network Meta Learner (SEL-NNML)** approach.

The primary objective is to evaluate the performance of the SEL-NNML model by rigorously comparing the impact of five different **Hyperparameter Tuning** techniques on model accuracy and stability.

---

## Project Overview

### Datasets Used
This research utilizes two publicly available datasets:
1.  **Kaggle Heart Failure Prediction Dataset (KHFPD)**: Used for general heart disease and failure analysis.
2.  **UCI Cleveland Heart Disease Dataset (UCHDD)**: A widely recognized benchmark dataset for cardiovascular diagnosis.

### Hyperparameter Tuning Techniques Evaluated
The performance of the base models and meta-learner is optimized and compared across the following state-of-the-art methods:
* **TPE** (Tree-structured Parzen Estimator)
* **GA** (Genetic Algorithm)
* **Bayesian** Optimization
* **GS** (Grid Search)
* **RS** (Random Search)

