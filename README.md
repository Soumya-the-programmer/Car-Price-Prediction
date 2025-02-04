# 🚗 Car Price Prediction using Machine Learning

## 📌 Overview
This project predicts the **price of used cars** based on features like brand, model, engine size, year, fuel type, and transmission type. The model is trained using **Multiple Linear Regression** with **Feature Engineering techniques** for handling categorical and missing data.

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
   - **Python, pandas, NumPy, Matplotlib**  
   - **scikit-learn (Linear Regression, Encoding, Model Evaluation)**  

## 📂 Dataset  
- `car_price_dataset.csv` contains historical data of **car sales with features like brand, mileage, engine size, etc.**  
- Missing values were handled using **mean imputation & category encoding**.  

## 🚀 How to Run the Project
1️⃣ Install dependencies:  
```bash
pip install numpy pandas scikit-learn matplotlib
```
2️⃣ Run the ML model script:

```bash
python car_price_prediction_model.py
```

## 📊 Model Performance
- Training Accuracy: 99.93%
- Evaluation Metrics: R² Score, Mean Squared Error (MSE), Mean Absolute Error (MAE) were used to validate predictions.

## 📜 License
- Feel free to use and modify this project for learning purposes. ⭐ If you find this useful, consider giving it a star on GitHub!
