# Kalimati Data Analysis

Group assignment for data analytics and machine learning analyzing Kalimati vegetable price data.

---

## Description

This repository contains a complete data analytics workflow built around Kalimati vegetable price data spanning from May 2021 to September 2023. The project focuses on cleaning raw market data, performing exploratory data analysis (EDA), generating visual insights, and building a predictive model based on historical price trends.

The codebase is intentionally simple and script-driven, making it suitable for academic use, rapid experimentation, and reproducible analysis. It demonstrates a practical end-to-end approach to working with real-world tabular datasets.

Key scripts referenced in this document include:

* [`cleandata.py`](./cleandata.py)
* [`edascript.py`](./edascript.py)
* [`Model.py`](./Model.py)

---

## Techniques Used

* **Structured data cleaning and preprocessing**
  Raw CSV files are normalized, cleaned, and transformed into analysis-ready formats using Pandas-style workflows.

* **Exploratory Data Analysis (EDA)**
  Statistical summaries and visualizations are generated to identify trends, seasonality, and anomalies in vegetable pricing data.

* **Programmatic visualization pipelines**
  Plots are generated through scripts rather than notebooks and saved to disk, enabling reproducibility and version control of analytical outputs.

* **Model training and evaluation**
  A dedicated modeling script encapsulates feature preparation, training logic, and result visualization.

* **Separation of concerns**
  Data, scripts, generated figures, and cached files are kept in distinct locations to maintain a clean project structure.

---

## Technologies and Libraries

This project is implemented in Python and relies on common but powerful data science libraries that may be of interest to intermediate developers:

* **Pandas** – tabular data processing and transformation
  [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)

* **NumPy** – numerical computing and array operations
  [https://numpy.org/doc/](https://numpy.org/doc/)

* **Matplotlib** – foundational plotting and figure generation
  [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)

* **Seaborn** – high-level statistical data visualization built on Matplotlib
  [https://seaborn.pydata.org/](https://seaborn.pydata.org/)

These tools form a lightweight but production-proven stack for data analysis and visualization.

---

## Project Structure

```text
/
├── Model_Figs/
├── __pycache__/
├── eda_plots/
├── preprocessed data.daml
├── Kalimati_Tarkari_Price.csv
├── kalimati-tarkari-prices-from-may-2021-to-september-2023.csv
├── Model.py
├── cleandata.py
├── edascript.py
└── tempCodeRunnerFile.py
```

### Directory Notes

* **Model_Figs/**
  Contains figures generated during model training and evaluation.

* **eda_plots/**
  Stores plots created during exploratory data analysis, such as distributions and trend visualizations.

* ****pycache**/**
  Python bytecode cache generated during script execution. This directory can be safely ignored or excluded from version control.

### Key Files

* **`kalimati-tarkari-prices-from-may-2021-to-september-2023.csv`**
  Original raw dataset used as the primary data source.

* **`Kalimati_Tarkari_Price.csv`**
  Alternate or cleaned copy of the dataset.

* **`preprocessed data.daml`**
  Preprocessed data output used directly by the modeling pipeline.

* **`cleandata.py`**
  Handles data cleaning and preprocessing logic.

* **`edascript.py`**
  Performs exploratory data analysis and generates plots.

* **`Model.py`**
  Contains model training, evaluation, and visualization logic.

---

## Usage

1. Ensure Python 3.x and the required libraries are installed.
2. Run `cleandata.py` to preprocess the raw dataset.
3. Execute `edascript.py` to generate exploratory plots.
4. Run `Model.py` to train the model and produce evaluation figures.

---

## License

This project is intended for educational and academic use. Add a license file if redistribution or reuse terms are required.
