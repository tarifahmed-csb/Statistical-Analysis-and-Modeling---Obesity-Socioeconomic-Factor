# Obesity Trends Analysis: The Impact of COVID-19

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview

This project conducts a comprehensive statistical analysis of obesity trends in the United States, specifically examining whether the COVID-19 pandemic accelerated the rise in obesity rates. Using data from the **Behavioral Risk Factor Surveillance System (BRFSS)**, we analyze trends across three distinct time periods:

*   **Early Period (2011–2013)**: Pre-pandemic baseline
*   **Middle Period (2016–2018)**: Pre-COVID reference
*   **Late Period (2021–2023)**: COVID-era data

The analysis investigates whether the rate of increase in obesity was significantly higher during the pandemic period compared to the pre-pandemic trend, using state-level data and demographic breakdowns.

## 👥 Authors

*   **Tarif Ahmed**
*   **VincentCaruso**

## 📊 Key Features

The analysis includes:

1.  **State-Level Comparisons**:
    *   Empirical Cumulative Distribution Function (ECDF) plots.
    *   Histograms of mean difference scores (Late vs. Middle vs. Early).
    *   Identification of states with the largest and smallest changes.

2.  **Statistical Inference**:
    *   **Paired t-tests**: Comparing pre-COVID and COVID-era rate changes.
    *   **Bootstrap Analysis**: Non-parametric confidence intervals (5,000–10,000 resamples) to robustly estimate differences without assuming normality.
    *   **Difference-in-Differences (DiD)**: Assessing changes in the *slope* of obesity prevalence over time.

3.  **Demographic Deep Dives**:
    *   **Income**: Analysis across 6 income brackets.
    *   **Race/Ethnicity**: Trends across 7 racial/ethnic groups.
    *   **age**: Impact on different age cohorts (18–65+).
    *   **Sex**: Male vs. Female comparisons.
    *   **Education**: Trends by educational attainment.

## 📁 Repository Structure

```
Obesity/
├── ahmed_caruso_obesity.py       # Original research code (Google Colab optimized)
├── obesity_analysis_cleaned.py   # Refactored, production-ready local version
├── README.md                     # This file
└── course doc/
    └── Caruso_Ahmed_Presentation.pptx # Project presentation slides
    └── Caruso_Ahmed_ReportBanner.png

```

*   **`ahmed_caruso_obesity.py`**: The original script used for the initial research. It is designed for Google Colab and contains hardcoded paths.
*   **`obesity_analysis_cleaned.py`**: A polished, modularized version of the code designed to run on any local machine. It allows you to select your data file and includes robust error handling.

## 🚀 Getting Started

### Prerequisites

You need Python 3 installed along with the following libraries:

```bash
pip install numpy pandas matplotlib scipy seaborn
```

### Data Access

This project uses the **CDC's Nutrition, Physical Activity, and Obesity - Behavioral Risk Factor Surveillance System** dataset.

1.  **Download the data**:
    *   [CDC Data.gov Link](https://data.cdc.gov/Nutrition-Physical-Activity-and-Obesity/Nutrition-Physical-Activity-and-Obesity-Behavioral/hn4x-zwk7/about_data) (Select the **CSV** format)

2.  **Place the file**: Move the downloaded CSV file into this project directory.

### Running the Analysis

To run the improved analysis script:

1.  Open your terminal/command prompt.
2.  Navigate to the project directory:
    ```bash
    cd path/to/Obesity
    ```
3.  Run the script:
    ```bash
    python obesity_analysis_cleaned.py
    ```
4.  Enter the filename of your downloaded CSV when prompted.

## 📈 Methodology

Our approach focuses on **Difference-in-Differences (DiD)** estimation. We calculate the "slope" of obesity increase for the pre-COVID period ($Slope_{pre} = Middle - Early$) and the COVID-era period ($Slope_{covid} = Late - Middle$). We then test the hypothesis that $Slope_{covid} > Slope_{pre}$.

*   **Significance Testing**: We use bootstrap resampling to generate confidence intervals for the difference in slopes ($Slope_{covid} - Slope_{pre}$), providing a robust measure of statistical significance.
