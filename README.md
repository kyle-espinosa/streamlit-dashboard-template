# Predicting Sales and Classifying Data using Streamlit

A Streamlit web application that performs **Exploratory Data Analysis (EDA)**, **Data Pre-processing**, and two supervised classification models: **Logistic Regression** and **Random Forest Regressor** to make predictions on sales volume and classification on new unseen data.

![Main Page Screenshot](screenshots/main_page_screenshot.png)

### 🙍🏻‍♂️ Members:
1. Kyle Chrystian Espinosa
2. Renzo Andre Falloran
3. Carlo Luis Leyva
4. Kurt Joseph Pecenio
5. Angelica Young

### 🔗 Links:

- 🌐 [Streamlit Link]()
- 📗 [Google Colab Notebook]( https://colab.research.google.com/drive/1h0pu9_x6SK-1tHLppzMmRK3v-lVIOKiZ?usp=sharing#scrollTo=YqyCyvIg1dm4)

### 📊 Dataset:

- [Amazon Phone Data: Prices, Ratings & Sales Insight (Kaggle)](https://www.kaggle.com/datasets/shreyasur965/phone-search-dataset)

### 📖 Pages:

1. `Dataset` - Brief description of the Iris Flower dataset used in this dashboard.
2. `EDA` - Exploratory Data Analysis of the Amazon Phone Prices, Ratings & Sales dataset. Highlighting the distribution of Iris species and the relationship between the features. Includes graphs such as Pie Chart, Scatter Plots, and Pairwise Scatter Plot Matrix.

   Histogram, Scatter Plot, Pairwise Scatter Plot, and Confusion Matrix. bar plot
4. `Data Cleaning / Pre-processing` - Data cleaning and pre-processing steps such as encoding the species column and splitting the dataset into training and testing sets.
5. `Machine Learning` - Training two supervised classification models: Logistic Regression and Random Forest Regressor. Includes model evaluation, feature importance, and tree plot.
6. `Prediction` - Prediction page where users can input values to predict the Iris species using the trained models.
7. `Conclusion` - Summary of the insights and observations from the EDA and model training.

### 💡 Findings / Insights



logistic regression to predict the whether aproduct was designated as "Amazon Choice". The random forest regressor used to predict product sales volume, 
Overview:





Through exploratory data analysis and training of two classification models (Logistic Regression and Random Forest Regressor) on the Amazon Phone Data dataset, the key insights and observations are:

### 1. 📊 **Dataset Characteristics**:

- The dataset captures valuable e-commerce metrics such as product price, ratings, and sales volume.
Some columns like product_original_price, product_availability, unit_price, and coupon_text had significant missing values, which were either removed or imputed as necessary.
- Feature normalization and encoding allowed us to prepare the dataset for effective modeling.

### 2. 📝 **Feature Distributions and Separability**:

- Visualizations, including histograms, scatter plots, and pairwise scatter plots, revealed relationships between product prices and star ratings.
- EDA highlighted that certain features, such as product_price and product_star_rating, have a strong correlation, suggesting potential for feature engineering in future work.

### 3. 📈 **Model Performance (Logistic Regression for Classification)**:

- The Logistic Regression model achieved high accuracy in predicting whether a product was designated as "Amazon Choice." This simple yet effective model was a good fit for the classification task.
- The confusion matrix and classification report further illustrated the model’s effectiveness in classifying the products accurately based on the provided features.

### 4. 📈 **Model Performance (Random Forest Regressor for Sales Volume Prediction)**:

- The Random Forest Regressor was trained to predict product sales volume, achieving a high R² score on both the training and testing data.
- Feature importance analysis identified product_price and product_star_rating as the most influential features for predicting sales volume, highlighting how customer preferences may affect sales.

#### **Summing up:**

This project successfully demonstrated effective data preparation, exploratory analysis, and modeling techniques on the Amazon Phone Data dataset. Both models achieved high accuracy in their respective tasks, providing insights into product sales prediction and classification.
