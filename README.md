# Master's Dissertation

# Integrating Predictive Models with Operational Policies for Constrained Lead Qualification

**Author:** Michael Thomas

**Institution:** Liverpool John Moores University (LJMU)

**Programme:** MSc Artificial Intelligence & Machine Learning

---

## Executive Summary

This repository contains the supporting implementation, datasets, and experimental notebooks for my MSc dissertation:

> **Integrating Predictive Models with Operational Policies for Constrained Lead Qualification**

The dissertation investigates whether a **post-scoring qualification layer** can improve lead selection when sales teams operate under a fixed follow-up capacity.

Rather than treating predictive lead scoring and lead qualification as competing approaches, the research proposes that they solve different problems. Predictive models estimate conversion probability using information available when a lead is created, while qualification frameworks acquire additional operational information that only becomes available during later sales interactions. The dissertation evaluates whether combining these two stages improves constrained lead-selection decisions.

---

# Research Question

> **Can a qualification layer applied after predictive lead scoring improve lead-selection outcomes under a fixed sales-capacity constraint?**

---

# Why this research matters

Most academic research treats lead scoring as a prediction problem. Machine learning models rank leads according to their estimated probability of conversion, and the ranked list is typically treated as the final decision.

In practice, however, organisations almost always perform an additional qualification stage before committing scarce sales resources. During qualification, sales teams collect information that was unavailable when the predictive model generated its score, while also applying organisational policies and commercial judgement developed through practical experience.

This dissertation investigates whether incorporating this post-scoring qualification stage produces better lead-selection decisions than predictive scoring alone.

---

# Experimental Design

Three lead-selection systems were evaluated using the **X Education Lead Scoring Dataset**.

## System 1 — Machine Learning Baseline

- XGBoost predictive lead-scoring model
- Leads ranked by predicted conversion probability
- Top **K** leads selected for follow-up

## System 2 — Tier-Based Qualification

- Same machine-learning model
- Additional qualification information applied after scoring
- Priority qualification tiers created
- Machine-learning ranking preserved within each tier

## System 3 — Calibrated Blend

- Same qualification information as System 2
- Qualification incorporated as a weighted adjustment to the machine-learning score
- No explicit priority tiers

System 3 acts as a robustness test, allowing the study to distinguish whether any improvement arises from the qualification information itself or from the mechanism used to apply it.

---

# Key Findings

The study found that:

- Tier-based qualification improved **Precision@K** from **87.7%** to **89.4%**.
- This represented **five additional conversions** within the same fixed sales follow-up capacity.
- Applying the same qualification information as a weighted score adjustment produced only a modest improvement (**88.1%**).
- The results demonstrate that **how qualification information is applied matters as much as the information itself**.
- Qualification produced the greatest value when it introduced **genuinely new information that was unavailable when the predictive model generated its initial score**.
- Predictive lead scoring and lead qualification should therefore be viewed as **complementary decision processes rather than competing approaches**.

---

# Research Contribution

The dissertation contributes to the lead-scoring literature in four ways.

First, it distinguishes **predictive lead scoring** from **lead qualification** as separate decision stages rather than treating them as a single process.

Second, it proposes that qualification frameworks such as **BANT**, **MEDDIC**, **CHAMP**, and **SPIN** can be viewed as **structured information-acquisition frameworks**. Rather than simply acting as decision checklists, they guide sales teams towards discovering commercially important information that is unavailable when predictive scores are first generated.

Third, it demonstrates that combining predictive scoring with post-scoring qualification can improve constrained lead-selection decisions under appropriate conditions.

Finally, it identifies the conditions under which qualification adds value:

- new information becomes available after scoring;
- qualification identifies groups with above-baseline conversion rates; and
- machine-learning ranking remains the primary mechanism for selecting leads within those groups.

---

# Repository Contents

```text
/
├── Masters_Constrained_Lead_Qualification_Notebook.ipynb
├── Masters_UCI_Bank_test.ipynb
├── Masters_Stuffmart_Additional_Dataset_Notebook.ipynb
├── 27062026 dissertation_presentation sent to Dr Mayank.pptx
├── bank.csv
├── bank-additional.csv
├── bank-additional-full.csv
├── bank-names.txt
├── bank-additional-names.txt
├── test.csv
├── README.md
└── .gitignore
```

---

# Primary Experimental Notebook

## Masters_Constrained_Lead_Qualification_Notebook.ipynb

This notebook contains the complete implementation reported in the dissertation, including:

- data preparation and preprocessing
- exploratory data analysis
- XGBoost predictive lead-scoring model
- construction of the three lead-selection systems
- Precision@K evaluation
- Recall@K, Lift@K and Wasted Effort Rate
- bootstrap confidence intervals
- sensitivity analyses
- statistical robustness testing
- final experimental results reported in the dissertation

Running the notebook sequentially reproduces all principal experimental results presented in the dissertation.

---

# Validation Dataset

## Masters_UCI_Bank_test.ipynb

The UCI Bank Marketing dataset was investigated as an independent validation dataset.

Although the predictive model performed well, the available variables did not provide a defensible separation between information available at lead creation and information obtained during lead qualification. Because the dissertation specifically investigates **post-scoring qualification**, the dataset was considered unsuitable for testing the research hypothesis and was excluded from the final empirical evaluation.

The notebook has been retained to document the evaluation process and dataset selection decisions.

---

# Additional Dataset

## Masters_Stuffmart_Additional_Dataset_Notebook.ipynb

A second commercial-style dataset (Stuffmart) was evaluated as a potential validation dataset.

The analysis concluded that the available variables did not contain genuine post-scoring qualification information. Apparent qualification signals were either synthetic, represented information leakage, or were already available at the point of prediction.

Consequently, the dataset could not support a meaningful separation between predictive lead scoring and post-scoring qualification and was not used in the final empirical evaluation.

The notebook is included for transparency and to document the dataset evaluation process.

---

# Supporting Data

The repository also contains the publicly available datasets used during experimentation together with supporting documentation required to reproduce the analysis.

The final dissertation results are based on the **X Education Lead Scoring Dataset**, implemented in the primary notebook.

---

# Reproducing the Analysis

Running the primary notebook from beginning to end reproduces:

- data preprocessing
- predictive model training
- system construction
- Precision@K evaluation
- bootstrap confidence intervals
- sensitivity analyses
- dissertation tables
- dissertation figures

---

# Technologies

**Programming Language**

- Python 3

**Primary Libraries**

- XGBoost
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook

---

# Future Research

Potential extensions of this work include:

- validation using live B2B qualification datasets;
- modelling qualification as a structured information-acquisition process;
- identifying which qualification questions produce the greatest incremental information value;
- adaptive qualification frameworks that select questions dynamically;
- integration with uplift modelling and persuadability modelling; and
- optimisation of variable sales-team capacity rather than assuming a fixed follow-up budget.

---

# Citation

If you use this work in academic research, please cite:

> Thomas, M. (2026). *Integrating Predictive Models with Operational Policies for Constrained Lead Qualification*. MSc Dissertation, Liverpool John Moores University.

---

# License

This repository is provided for academic, educational, and research purposes.

© Michael Thomas, 2026.
