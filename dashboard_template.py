#######################
# Import libraries

import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder



#######################
# Page configuration
st.set_page_config(
    page_title="Dashboard Template", # Replace this with your Project's Title
    page_icon="assets/icon.png", # You may replace this with a custom icon or emoji related to your project
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar
with st.sidebar:

    # Sidebar Title (Change this with your project's title)
    st.title('Dashboard Template')

    # Page Button Navigation
    st.subheader("Pages")

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Members
    st.subheader("Members")
    st.markdown("1. KYLE CHRYSTIAN ESPINOSA\n2. RENZO ANDRE FALLORAN\n3. CARLO LUIS LEYVA\n4. KURT JOSEPH PECENIO\n5. ANGELICA YOUNG")

#######################
# Data

# Load data
phonesearch_df = pd.read_csv("data/phone search.csv")

# Plots

# `key` parameter is used to update the plot when the page is refreshed

def histogram(column, width, height, key):
    # Generate a histogram
    histogram_chart = px.histogram(phonesearch_df, x=column)

    # Adjust the height and width
    histogram_chart.update_layout(
        width=width,   # Set the width
        height=height  # Set the height
    )

    st.plotly_chart(histogram_chart, use_container_width=True, key=f"histogram_{key}")

def scatter_plot(x_column, y_column, width, height, key):
    # Generate a scatter plot
    scatter_plot = px.scatter(phonesearch_df, x=phonesearch_df[x_column], y=phonesearch_df[y_column])

    # Adjust the height and width
    scatter_plot.update_layout(
        width=width,  # Set the width
        height=height  # Set the height
    )

    st.plotly_chart(scatter_plot, use_container_width=True, key=f"scatter_plot_{key}")
def pairwise_scatter_plot(key):
    # Generate a pairwise scatter plot matrix
    scatter_matrix = px.scatter_matrix(
        phonesearch_df,
        dimensions=['product_price', 'product_star_rating', 'product_original_price'],  # Choose relevant numeric columns
        color='is_prime'  # Replace with a categorical column if applicable
    )

    # Adjust the layout
    scatter_matrix.update_layout(
        width=500,  # Set the width
        height=500  # Set the height
    )

    st.plotly_chart(scatter_matrix, use_container_width=True, key=f"pairwise_scatter_plot_{key}")

def feature_importance_plot(feature_importance_df, width, height, key):
    # Generate a bar plot for feature importances
    feature_importance_fig = px.bar(
        feature_importance_df,
        x='Importance',
        y='Feature',
        labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
        orientation='h'  # Horizontal bar plot
    )

    # Adjust the height and width
    feature_importance_fig.update_layout(
        width=width,  # Set the width
        height=height  # Set the height
    )

    st.plotly_chart(feature_importance_fig, use_container_width=True, key=f"feature_importance_plot_{key}")


# -------------------------

#######################

# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")

    # Your content for the ABOUT page goes here
   

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")

     # Your content for your DATASET page goes here

    st.markdown("""

    The **Amazon Phone Data dataset** contains real-time information about 340 phone products available on Amazon. It includes prices, reviews, sales volume, and tags such as Best Seller and Amazon Choice. This data is useful for analyzing price trends, forecasting sales volume, and gaining insights into e-commerce.

    For each product in the Amazon Phone Data dataset, several features are measured: asin (Amazon Standard Identification Number), product_title (name of the product), product_price (current price of the product), and product_star_rating (rating given to the product based on customer reviews). This dataset is useful for analyzing product performance, pricing strategies, and consumer preferences in e-commerce. The dataset was uploaded to Kaggle by the user shreyasur965.

    **Content**
    The dataset has **340** rows containing attributes related to phone products on Amazon. The columns include information such as Price, Reviews, Sales Volume, and Tags (e.g., Best Seller, Amazon Choice).

    `Link:` https://www.kaggle.com/datasets/shreyasur965/phone-search-dataset          
                
    """)
    
 # Display the dataset
    st.subheader("Dataset displayed as a Data Frame")
    st.dataframe(phonesearch_df, use_container_width=True, hide_index=True)
    
 # Describe Statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(phonesearch_df.describe(), use_container_width=True)

    st.markdown("""

    The results from `df.describe()` highlights the descriptive statistics about the dataset. The product_star_rating has an average rating of 4.09 out of 5, with a standard deviation of 0.38, indicating that most products are rated positively, close to 4 or above. The product_num_ratings column shows an average of 2,744 ratings per product, though a large standard deviation (7,331.82) suggests high variability; while some products have very few ratings, others are extremely popular with many reviews (up to 64,977). product_num_offers averages around 8 offers per product, but again, there's substantial variability (ranging from 1 to 83), suggesting certain products are offered by many vendors. The unit_count column, with only four recorded values, suggests missing data, limiting its usefulness for analysis.

    Overall, these statistics highlight a need for further data cleaning, particularly in handling missing values in unit_count and potentially examining outliers in product_num_ratings and product_num_offers for better consistency in analysis.
                
    """)
   
    

# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")

    with st.expander('Legend', expanded=True):
        st.write('''
            - Data: [Phone Search Dataset](https://www.kaggle.com/datasets/shreyasur965/phone-search-dataset).
            - :orange[**Histogram**]: Distribution of normalized product prices.
            - :orange[**Scatter Plots**]: Product prices vs. Star Ratings.
            - :orange[**Pairwise Scatter Plot Matrix**]: Highlighting *overlaps* and *differences* among Numerical features.
        ''')

    st.markdown('#### Price Distribution')
    histogram("product_price", 800, 600, 1)
    
    st.markdown('#### Product prices vs. Star Ratings')
    scatter_plot('product_price', 'product_star_rating', 500, 500, 1)
    
    st.markdown('#### Pairwise Scatter Plot Matrix')
    pairwise_scatter_plot(1)

        
    

# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")

    # Your content for the DATA CLEANING / PREPROCESSING page goes here
    st.subheader("Sample of Encoded Data")
    st.write(phonesearch_df[['is_best_seller', 'is_best_seller_encoded', 'is_amazon_choice', 'is_amazon_choice_encoded']].head())

    # Step 2: Show value counts for each encoded column
    st.subheader("Value Counts of Encoded Columns")
    best_seller_counts = phonesearch_df['is_best_seller_encoded'].value_counts()
    amazon_choice_counts = phonesearch_df['is_amazon_choice_encoded'].value_counts()

    st.write("**Best Seller Encoded Value Counts**")
    st.bar_chart(best_seller_counts)

    st.write("**Amazon Choice Encoded Value Counts**")
    st.bar_chart(amazon_choice_counts)



# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    # Your content for the MACHINE LEARNING page goes here

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    # Your content for the PREDICTION page goes here

# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    # Your content for the CONCLUSION page goes here
