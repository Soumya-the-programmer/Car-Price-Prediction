# 🚗 Car Price Prediction using Machine Learning

## 📌 Overview
This project predicts the **price of used cars** based on features like brand, model, engine size, year, fuel type, and transmission type. The model is trained using **Multiple Linear Regression** with **Feature Engineering techniques** for handling categorical and missing data.

## 📊 Actual vs. Predicted Car Prices

**Here’s a visual representation of my model’s predictions compared to actual car prices:**

![Actual vs. Predicted Prices](https://github.com/user-attachments/assets/7f57323d-38d9-408a-a22b-193f6a4a35e8)

## 🤖 Model Performance: Predicted vs Original Prices

*Below is a sample comparison of the predicted car prices and their original prices from the dataset:*

| Original Price | Predicted Price |
|--------------|----------------|
| 5901        | 5923.09        |
| 5425        | 5445.46        |
| 15398       | 15371.08       |
| 4731        | 4754.03        |
| 4912        | 4944.71        |
| ...         | ...            |

- *The model achieved **99.93% accuracy**, demonstrating strong predictive performance.*

## 🔍 Features & Techniques Used
✅ **Data Cleaning & Preprocessing**  
   - Removed **null values & duplicates**  
   - Converted categorical data using **One-Hot Encoding & Label Encoding**  
   - Reindexed data for structured feature alignment  

✅ **Feature Engineering & Model Training**  
   - Applied **Multiple Linear Regression** from `sklearn.linear_model`  
   - Used **80-20 Train-Test Split**  
   - Achieved **99.93% accuracy**  

✅ **Tools & Technologies Used**  
   - **Python, pandas, numpy, matplotlib**  
   - **scikit-learn (Linear Regression, Encoding, Model Evaluation)**  

## 📂 Dataset  
- `car_price_dataset.csv` contains historical data of **car sales with features like brand, mileage, engine size, etc.**  
- Missing values were handled using **mean imputation & category encoding**.

## 📊 Dataset Source  
- **The dataset used in this project was obtained from Kaggle.**  
- **You can find the original dataset here:** **([https://www.kaggle.com/](https://www.kaggle.com/datasets/asinow/car-price-dataset))**  

*Note: This dataset is publicly available for research and learning purposes.* 

## 🚀 How to Run the Project
1️⃣ **Install dependencies:**  
```bash
pip install pandas scikit-learn numpy matplotlib
```
2️⃣ **Run the ML model script:**

```bash
python car_price_prediction_model.py
```

## 📊 Model Performance
- Training Accuracy: 99.93%
- Evaluation Metrics: R² Score, Mean Squared Error (MSE), Mean Absolute Error (MAE) were used to validate predictions.

## 📜 License
- Feel free to use and modify this project for learning purposes. ⭐ If you find this useful, consider giving it a star on GitHub!
