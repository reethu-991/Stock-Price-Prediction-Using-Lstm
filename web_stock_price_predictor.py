import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import base64
import os

st.set_page_config(
    page_title="Stock Market Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Function to load and embed CSS
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    else:
        st.warning(f"CSS file '{file_name}' not found. Proceeding without custom styles.")

# Function to load and encode image
def load_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
        return f"data:image/jpeg;base64,{encoded}"
    else:
        st.warning(f"Image file '{image_path}' not found. Proceeding without the image.")
        return ""

# Load CSS
local_css("style.css")

# Initialize session state
if 'show_main_app' not in st.session_state:
    st.session_state.show_main_app = False

# Landing Page
if not st.session_state.show_main_app:
    # Load and encode the image
    stock_image = load_image("stock-image1.jpg")

    # HTML content for the landing page
    landing_html = f"""
    <h1>Stock Market Price Prediction</h1>

    <!-- Section 1 -->
    <div id="section1" class="section">
        <img src="{stock_image}" alt="Stock Market">
        <div class="project-info">
            <h2>About the Project</h2>
            <p>
                The Stock Price Prediction project aims to analyze historical stock prices using advanced machine learning techniques, providing users with insights into stock performance over different time frames (e.g., 200 days back, 100 days back, 50 days back). It features a user-friendly interface where users can input the stock name, triggering the analysis of stock trends and patterns. The project collects historical stock price data from Yahoo Finance, which undergoes rigorous preprocessing to ensure accuracy and relevance. Various machine learning models, including Random Forest, XGBoost, and LSTM with RNN, are employed to enhance analysis capabilities, with model performance evaluated through metrics like AUC, precision, and recall. The system integrates with Power BI for effective data visualization, enabling users to track trends and insights over different time periods, and is deployed for real-time analysis. Overall, this project leverages cutting-edge technology to assist users in making informed investment decisions based on historical data trends.
            </p>
        </div>
    </div>

    <!-- Section 2 -->
   
    """

    # Render the HTML content
    st.markdown(landing_html, unsafe_allow_html=True)

    # Capture the button click using Streamlit's built-in functionality
    # Since embedding JavaScript to communicate with Streamlit is complex,
    # we'll overlay a transparent Streamlit button on top of the HTML button using HTML and CSS.

    # Positioning the Streamlit button over the HTML button
    button_html = """
    <div style="position: relative; width: 100%; height: 50px;">
        <button class="navigate-btn" style="position: absolute; width: 100%; height: 100%; opacity: 0; cursor: pointer;"></button>
    </div>
    """
    st.markdown(button_html, unsafe_allow_html=True)

    # Create a Streamlit button with no visible interface
    if st.button("Get Recommendations", key='get_recommendation_btn', help="Click to proceed to Stock Predictions"):
        st.session_state.show_main_app = True

else:
    # Main App
    st.title("📈 Stock Price Predictor App")

    # User Input
    stock = st.text_input("🔍 Enter the Stock Ticker Symbol (e.g., GOOG)", "GOOG")

    # Date Range Selection (Optional Enhancement)
    with st.expander("📅 Select Date Range"):
        end = datetime.now()
        start_year = end.year - 20
        start = st.date_input("Start Date", datetime(start_year, end.month, end.day))
        end_date = st.date_input("End Date", end)

    # Validate date inputs
    if start > end_date:
        st.error("Error: End Date must fall after Start Date.")
        st.stop()

    # Fetching Stock Data
    try:
        with st.spinner(f"Fetching data for {stock} from Yahoo Finance..."):
            stock_data = yf.download(stock, start=start, end=end_date)
            if stock_data.empty:
                st.error(f"No data found for ticker symbol '{stock}'. Please check the symbol and try again.")
                st.stop()
    except Exception as e:
        st.error(f"Error fetching data for {stock}: {e}")
        st.stop()

    # Display Stock Data
    st.subheader("📊 Stock Data")
    st.write(stock_data)

    # Load the Pre-trained Model
    try:
        with st.spinner("Loading the prediction model..."):
            model = load_model("latest_stock_price_model.keras")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # Data Preparation
    splitting_len = int(len(stock_data) * 0.7)
    x_test = pd.DataFrame(stock_data.Close[splitting_len:])

    # Plotting Function
    def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(values, 'orange', label='Moving Average')
        ax.plot(full_data.Close, 'b', label='Close Price')
        if extra_data and extra_dataset is not None:
            ax.plot(extra_dataset, label='Additional MA')
        ax.legend()
        ax.grid(True)
        return fig

    # Visualization of Moving Averages
    st.subheader('📈 Original Close Price and Moving Averages')

    ma_periods = [250, 200, 100]
    colors = ['#FFA500', '#00FF00', '#FF69B4']  # Orange, Lime, Hot Pink for distinction

    for period, color in zip(ma_periods, colors):
        ma_label = f"MA for {period} days"
        stock_data[ma_label] = stock_data.Close.rolling(period).mean()
        st.pyplot(plot_graph((15, 6), stock_data[ma_label], stock_data, 0))

    # Combined Moving Averages Plot
    st.subheader('📉 Original Close Price with MA for 100 days and MA for 250 days')
    st.pyplot(plot_graph((15, 6), stock_data['MA for 100 days'], stock_data, 1, stock_data['MA for 250 days']))

    # Data Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(x_test[['Close']])

    x_data = []
    y_data = []

    for i in range(100, len(scaled_data)):
        x_data.append(scaled_data[i - 100:i])
        y_data.append(scaled_data[i])

    x_data, y_data = np.array(x_data), np.array(y_data)

    # Model Prediction
    with st.spinner("Predicting stock prices..."):
        predictions = model.predict(x_data)

    # Inverse Scaling to Get Actual Prices
    inv_pre = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_data)

    # Preparing Data for Display
    plotting_data = pd.DataFrame(
        {
            'Original': inv_y_test.reshape(-1),
            'Predicted': inv_pre.reshape(-1)
        },
        index=stock_data.index[splitting_len + 100:]
    )

    # Display Original vs Predicted Values
    st.subheader("✅ Original vs Predicted Stock Prices")
    st.write(plotting_data)

    # Plotting the Results
    st.subheader('📉 Original Close Price vs Predicted Close Price')
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(stock_data.Close[:splitting_len + 100], label='Historical Close Price')
    ax.plot(plotting_data['Original'], label='Original Test Data')
    ax.plot(plotting_data['Predicted'], label='Predicted Close Price')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Optional: Display Metrics or Additional Insights
    # You can add more features here, such as error metrics, future predictions, etc.



