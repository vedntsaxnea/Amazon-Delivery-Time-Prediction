import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# --- Load the Model and Preprocessor ---
# Set cache_resource to True so it only loads once
@st.cache_resource
def load_model_and_preprocessor():
    try:
        model = tf.keras.models.load_model('deep_csat_model.h5')
        preprocessor = joblib.load('data_preprocessor.pkl')
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None

model, preprocessor = load_model_and_preprocessor()

# --- Page Configuration ---
st.set_page_config(page_title="DeepCSAT Score Predictor", layout="wide")
st.title("🛒 Shopzilla CSAT Score Predictor")
st.write("Enter the details of a customer interaction to predict the CSAT score (1-5).")

# --- Define Input Form ---
if model and preprocessor:
    # Get the feature names from the dataset description
    # IMPORTANT: These must match the columns in the DataFrame we trained on
    
    # Column 1: Interaction Details
    with st.container():
        st.header("Interaction Details")
        col1, col2 = st.columns(2)
        with col1:
            # These are the variables that will hold the user's input
            channel_name = st.selectbox("Channel Name", ['Live Chat', 'Email', 'Call', 'Social Media'])
            category = st.selectbox("Category", ['Order Problems', 'Returns', 'Payments', 'Technical Issue'])
            sub_category = st.selectbox("Sub-category", ['Wrong item', 'Delayed Order', 'Refund Status', 'Website Error'])
            product_category = st.selectbox("Product Category", ['Electronics', 'Apparel', 'Home Goods', 'Lifestyle'])

        with col2:
            agent_shift = st.selectbox("Agent Shift", ['Morning', 'Evening', 'Night'])
            tenure_bucket = st.selectbox("Agent Tenure", ['0-3 mon', '3-6 mon', '6-12 mon', '> 12 mon'])
            item_price = st.number_input("Item Price ($)", min_value=0.0, value=50.0)
            connected_handling_time = st.number_input("Handling Time (seconds)", min_value=0.0, value=300.0)

    # Column 2: Time-based Features (We'll simplify for the app)
    with st.container():
        st.header("Engineered Features")
        st.info("In the real model, these are calculated from timestamps. Here, we'll input them directly.")
        col3, col4 = st.columns(2)
        with col3:
            response_time_s = st.number_input("Response Time (seconds)", min_value=0.0, value=60.0, help="Time from 'Issue reported' to 'Issue responded'")
            issue_lag_days = st.number_input("Issue Lag (days)", min_value=0.0, value=1.5, help="Time from 'Order date' to 'Issue reported'")
        with col4:
            issue_hour = st.slider("Hour of Day Issue Reported", 0, 23, 14)
            issue_day_of_week = st.slider("Day of Week Issue Reported", 0, 6, 1, help="0=Monday, 6=Sunday")

    # --- Prediction Button ---
    if st.button("Predict CSAT Score", type="primary"):
        # 1. Collect all inputs into a dictionary
        # *** THIS IS THE CORRECTED SECTION ***
        # The keys (e.g., 'Item_price') now match your training columns
        input_data = {
            # Numerical Features
            'Item_price': item_price,
            'connected_handling_time': connected_handling_time,
            'Response time (s)': response_time_s,
            'Issue lag (days)': issue_lag_days,
            'Issue hour': issue_hour,
            'Issue day_of_week': issue_day_of_week,
            
            # Categorical Features
            'channel_name': channel_name,
            'category': category,
            'Sub-category': sub_category,
            'Product_category': product_category,
            'Tenure Bucket': tenure_bucket,
            'Agent Shift': agent_shift
        }

        # 2. Convert to DataFrame
        # We use index=[0] to create a single-row DataFrame
        input_df = pd.DataFrame(input_data, index=[0])

        # 3. Preprocess the data
        # Use the *loaded* preprocessor to transform the new data
        input_processed = preprocessor.transform(input_df)
        
        # Convert to dense array if it's sparse (from the fix in Step 3)
        if hasattr(input_processed, "toarray"):
            input_processed = input_processed.toarray()

        # 4. Make a prediction
        prediction_probs = model.predict(input_processed)
        
        # 5. Get the predicted class
        # np.argmax gives the index (0-4) of the highest probability
        predicted_class = np.argmax(prediction_probs, axis=1)[0]
        
        # Add 1 to get the actual score (1-5)
        predicted_score = predicted_class + 1
        confidence = prediction_probs[0][predicted_class] * 100

        # 6. Display the result
        st.subheader("Prediction Result")
        st.metric(label="Predicted CSAT Score", value=f"{predicted_score} / 5", delta=f"{confidence:.2f}% Confidence")
        
        # Show probabilities for insight
        st.write("Prediction Probabilities:")
        prob_df = pd.DataFrame(prediction_probs, columns=[f"Score {i+1}" for i in range(5)])
        st.dataframe(prob_df)

else:
    st.error("Model or preprocessor not loaded. Please check file paths.")
