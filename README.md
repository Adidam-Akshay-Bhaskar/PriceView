# 🚗 PriceView

### Machine Learning Based Used Car Price Estimation & Market Segmentation

PriceView is a machine learning project that estimates the **selling price of a used car** and classifies the vehicle into a **price segment** based on its specifications and usage information.

The project combines:

- Exploratory Data Analysis
- Data preprocessing
- Regression
- Classification
- Unsupervised clustering
- Hyperparameter tuning
- Streamlit deployment

The final application provides a simple interface where users can enter vehicle details and receive an estimated selling price and price segment.

---

## 📌 Project Overview

Buying or selling a used vehicle can be difficult because the appropriate resale price depends on multiple factors such as:

- Vehicle age
- Kilometres driven
- Engine capacity
- Mileage
- Maximum power
- Number of seats
- Fuel type
- Transmission
- Seller type
- Brand
- Model

PriceView uses historical used-car data to learn relationships between these features and vehicle prices.

The system performs two main supervised learning tasks:

1. **Regression** → Predict the numerical selling price.
2. **Classification** → Classify the vehicle into Budget, Mid-Range, or Premium.

It also includes:

3. **Clustering** → Discover natural groups of vehicles based on numerical specifications.

---

# 🎯 Objectives

The main objectives of PriceView are:

- Analyze used-car pricing data.
- Identify important patterns and relationships in the dataset.
- Preprocess numerical and categorical features.
- Build multiple regression models for price prediction.
- Build multiple classification models for price segmentation.
- Compare different machine learning algorithms.
- Tune selected models using GridSearchCV.
- Perform vehicle segmentation using K-Means clustering.
- Build a user-friendly Streamlit application.
- Provide an end-to-end machine learning workflow.

---

# 📊 Dataset

## Dataset Used

The project uses the **CarDekho Used Car Dataset**:

```text
cardekho_dataset.csv