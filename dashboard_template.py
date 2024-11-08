#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

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

   
    

# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")


    col = st.columns((1.5, 4.5, 2), gap='medium')

    # Your content for the EDA page goes here

    with col[0]:
        st.markdown('#### Graphs Column 1')


    with col[1]:
        st.markdown('#### Graphs Column 2')
        
    with col[2]:
        st.markdown('#### Graphs Column 3')

# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")

    # Your content for the DATA CLEANING / PREPROCESSING page goes here

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
