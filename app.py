import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset
df_ML = pd.read_csv("ML_data.csv")

# Set page title and favicon
st.set_page_config(page_title="NPRI ML Model", page_icon="📊")

# Set up the Streamlit app with a custom background color and text colors
st.markdown(
    """
    <style>
    body {
        background-color: #d4f4dd; /* Light green */
        color: #333333; /* Dark blue */
    }
    .feature-box {
        background-color: #c7ecee; /* Light blue */
        padding: 10px ;
        border-radius: 10px ;
        box-shadow: 0px 0px 10px 0px rgba(0,0,0.1,0.9);
        margin-right: 10px;
        margin-bottom: 10px;
        display: inline-block;
        width: 150px; /* Adjust the width of the feature boxes */
        text-align: center; /* Center align text */
    }
    .feature-box:nth-child(even) {
        background-color: #a9cce3; /* Light blue for alternate boxes */
    }
    .title-text {
        font-size: 18px;
        color: #004c6d; /* Dark blue */
        font-weight: bold; /* Make title text bold */
    }
    .subheader-text {
        font-size: 16px;
        color: #666666; /* Dark blue */
    }
    .selected-feature {
        font-weight: bold; /* Make selected feature text bold */
    }
    .predicted-quantity {
        font-weight: bold; /* Make predicted quantity text bold */
        color: #000000; /* Black text color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.title("NPRI ML Model")

# Sidebar
st.sidebar.header('Select Features')

# Select box for categorical features
st.subheader('Selected Features')
def user_input_features():
    company_name = st.sidebar.selectbox('Company Name', df_ML['Company name'].unique())
    province = df_ML[df_ML['Company name'] == company_name]['Province'].iloc[0]
    substance_name = st.sidebar.selectbox('Substance Name', df_ML['Substance name'].unique())
    number_of_employees = st.sidebar.number_input('Number of Employees', min_value=1, max_value=5000, value=10, step=1)
    growth_input = st.sidebar.radio('Growth', ('up', 'down'))
    Price_input = st.sidebar.radio('Price', ('up', 'down'))

    user_input_data = {'Company Name': company_name,
                       'Province': province,
                       'Substance Name': substance_name,
                       'Number of Employees': number_of_employees,
                       'Growth': growth_input,
                       'Price': Price_input}

    features = pd.DataFrame(user_input_data, index=[0])

    return features

df_user_input = user_input_features()

st.write(df_user_input.style.set_properties(**{'font-weight': 'bold'}))

# Placeholder for displaying predicted quantities for 2023 to 2027
st.markdown("<hr>", unsafe_allow_html=True)
st.write("<p class='title-text'><b>Predicted Quantities For 2023 to 2027:</b></p>", unsafe_allow_html=True)

# Filter data for the selected substance name and company name
selected_substance_name = df_user_input['Substance Name'].iloc[0]
selected_company_name = df_user_input['Company Name'].iloc[0]
filtered_data = df_ML[(df_ML['Substance name'] == selected_substance_name) & (df_ML['Company name'] == selected_company_name)]

if not filtered_data.empty:
    # Train a Linear Regression model using data from 2014 to 2021
    model = LinearRegression()
    model.fit(df_ML[['Last year quantity', 'Two years ago quantity', 'Three years ago quantity',
                     'Number of employees', 'Growth', 'Price']][df_ML['NPRI_Report_ReportYear'] < 2022],
              df_ML['Current year quantity'][df_ML['NPRI_Report_ReportYear'] < 2022])

    # Initialize input data for prediction
    input_data = {
        'Last year quantity': [filtered_data['Current year quantity'].iloc[0]],
        'Two years ago quantity': [filtered_data['Last year quantity'].iloc[0]],
        'Three years ago quantity': [filtered_data['Two years ago quantity'].iloc[0]],
        'Number of employees': [df_user_input['Number of Employees'].iloc[0]],
        'Growth': [1 if df_user_input['Growth'].iloc[0] == 'up' else 0],
        'Price': [1 if df_user_input['Price'].iloc[0] == 'up' else 0],
    }

    # Predict quantities for the next five years (2023 to 2027)
    predicted_quantities = []
    for year in range(2023, 2028):
        prediction = model.predict(pd.DataFrame(input_data))
        predicted_quantities.append(prediction[0])
        # Update input data for the next prediction
        input_data['Three years ago quantity'] = input_data['Two years ago quantity']
        input_data['Two years ago quantity'] = input_data['Last year quantity']
        input_data['Last year quantity'] = [prediction[0]]

    # Plot the trend graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(2023, 2028), predicted_quantities, marker='o', linestyle='-')
    plt.xticks(range(2023, 2028))  # Set integer ticks on x-axis
    plt.xlabel('Year')
    plt.ylabel('Predicted Quantity')
    plt.title(f'Trend of {selected_substance_name} Quantity for {selected_company_name} from 2023 to 2027')
    plt.grid(True)

    # Show the plot in Streamlit
    st.pyplot(plt)
else:
    st.write(f"No data available for {selected_substance_name} for {selected_company_name} for previous years.")
