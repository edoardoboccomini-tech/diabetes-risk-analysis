# Predictive Analysis of Diabetes Risk

## Project Overview
This project explores the impact of non-clinical factors (lifestyle habits and genetics) on the probability of a diabetes diagnosis. We deliberately excluded direct clinical markers (like blood glucose or insulin levels) to focus strictly on preventive lifestyle choices and genetic predispositions.

Initially developed in **R** as a collaborative university project at UNIVPM, the analysis has now been completely translated, optimized, and expanded into a **Python** pipeline. This dual-language approach showcases cross-language proficiency and the rigorous application of classical Machine Learning workflows (Classification and Clustering).

## Dataset
The data is sourced from the **[Diabetes and Lifestyle Dataset](https://www.kaggle.com/datasets/alamshihab075/health-and-lifestyle-data-for-diabetes-prediction)** on Kaggle.

## Key Insights
* **The Weight of Genetics (Non-Modifiable Risk):** Marginal effects analysis (APE) from our Logistic Regression revealed that a family history of diabetes is the strongest overall predictor, increasing the average probability of a positive diagnosis by approximately [XX]% (replace with exact APE number), holding all other factors constant.
* **Actionable Prevention (Modifiable Risk):** Among all lifestyle variables, weekly physical activity emerged as the most significant factor for prevention. Our models indicate that increasing physical activity mitigates diagnostic risk more effectively than diet score improvements alone.
* **Holistic Risk Profiles:** Unsupervised learning (K-Means Clustering) successfully segmented the population into 4 distinct profiles, demonstrating how the combination of high BMI and sedentary behavior exponentially compounds baseline age and genetic risks.


## Methodology & Tools 
This repository contains two parallel implementations of the same analytical pipeline, demonstrating versatility across different data science ecosystems.

* **Languages & Environments:** * **R** (RStudio)
  * **Python** (Jupyter Notebook, PyCharm)
* **Data Handling & Processing:** * **R:** `dplyr`, `tidyr`, `base R`
  * **Python:** `pandas`, `numpy`, `scikit-learn` (StandardScaler)
* **Exploratory Data Analysis (EDA):** * **R:** `ggplot2`, `corrplot`
  * **Python:** `seaborn`, `matplotlib`
* **Classical Machine Learning:**
  * **R:** `MASS` (LDA/QDA), `stats` (Logistic, PCA, K-Means), `factoextra` (Clustering Viz)
  * **Python:** `statsmodels` (Logistic, Marginal Effects), `scikit-learn` (LDA, QDA, PCA, K-Means), `scipy` (Hierarchical Clustering)

## Authors & Credits
* **Original R Project & Analysis (UNIVPM):** Edoardo Boccomini, Maria Benedetta Del Bianco, Mattia Spinelli
* **Python Translation & ML Pipeline:** Edoardo Boccomini
