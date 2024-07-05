# Fraudulent Website Detection

## Project Description

This project focuses on detecting fraudulent websites using machine learning techniques. It leverages a dataset containing various features of websites to train models that can predict the likelihood of a website being fraudulent.

## Dataset

The project uses a dataset that includes various features of websites, such as URL length, presence of certain keywords, and other indicators of potential fraudulence.

## Features

The dataset includes the following features:
- URL Length
- Number of Dots in URL
- Presence of HTTPS
- IP Address
- Keyword Presence (e.g., "login", "secure")
- Web Traffic Rank
- Domain Age
- WHOIS Data
- Page Rank
- Google Index

## Getting Started

### Prerequisites

To run the project, you need the following Python libraries:
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- joblib

You can install them using pip:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn joblib
```

## Running the Project

Clone the repository:

```bash
git clone https://github.com/PadmalathaKasireddy/fraudulent-website-detection.git
cd fraudulent-website-detection
```

Load the dataset. Ensure the dataset (dataset.csv) is in the project directory.

Run the Jupyter notebook:

```bash
jupyter notebook Fraudulent_website_detection.ipynb
```

Follow the steps in the notebook to preprocess the data, train the models, and evaluate their performance.

## Model Training

The project includes training several machine learning models:

- Random Forest Classifier
- Gradient Boosting Classifier
- Logistic Regression
- Neural Network (MLP Classifier)
- Support Vector Machine (SVM)

The models are evaluated based on metrics such as accuracy, ROC AUC score, and confusion matrix.

## Results

The performance of the models is summarized, and the best-performing model is identified based on the evaluation metrics.
