
import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Function to train or load the model
def load_or_train_model(data_path, model_path):
    try:
        # Try loading the model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        st.success("Model loaded successfully.")
    except:
        # Train the model if not already trained
        st.info("Training the model. This might take a while...")
        model = train_model(data_path, model_path)
        st.success("Model trained and saved successfully.")
    return model

# Function to train the model
def train_model(data_path, model_path):
    # Load the dataset
    data = pd.read_csv(data_path)
    data.columns = data.columns.str.strip()  # Strip whitespaces from column names

    # Drop irrelevant columns like loan_id if it exists
    if 'loan_id' in data.columns:
        data = data.drop(columns=['loan_id'])

    # Preprocess categorical columns
    label_encoder = LabelEncoder()
    data['education'] = label_encoder.fit_transform(data['education'])
    data['self_employed'] = label_encoder.fit_transform(data['self_employed'])
    data['loan_status'] = label_encoder.fit_transform(data['loan_status'])  # Approved -> 1, Rejected -> 0

    # Separate features and target
    X = data.drop(columns=['loan_status'])  # Features
    y = data['loan_status']               # Target variable

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Save the model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    return model

# Main Streamlit UI
def main():
    st.title("Loan Approval Prediction System")
    st.write("This system predicts whether a loan will be Approved or Rejected based on input data.")

    # Paths for the dataset and model
    data_path = r"loan_approval_dataset.csv"  # Path to your dataset
    model_path = "loan_model.pkl"  # Path to save/load the model

    # Load or train the model
    model = load_or_train_model(data_path, model_path)

    # User input form
    st.subheader("Enter Loan Details")
    with st.form("loan_form", clear_on_submit=True):
        no_of_dependents = st.number_input("Number of Dependents", min_value=0, value=1)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        income_annum = st.number_input("Annual Income", min_value=0, value=500000)
        loan_amount = st.number_input("Loan Amount", min_value=0, value=1000000)
        loan_term = st.number_input("Loan Term (in months)", min_value=1, value=12)
        cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750)
        residential_assets_value = st.number_input("Residential Assets Value", min_value=0, value=500000)
        commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, value=1000000)
        luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0, value=2000000)
        bank_asset_value = st.number_input("Bank Asset Value", min_value=0, value=1000000)

        submit_button = st.form_submit_button("Predict")
        clear_button = st.form_submit_button("Clear")

    if submit_button:
        # Create input DataFrame
        input_data = pd.DataFrame({
            'no_of_dependents': [no_of_dependents],
            'education': [1 if education == "Graduate" else 0],  # Encode Graduate as 1, Not Graduate as 0
            'self_employed': [1 if self_employed == "Yes" else 0],  # Encode Yes as 1, No as 0
            'income_annum': [income_annum],
            'loan_amount': [loan_amount],
            'loan_term': [loan_term],
            'cibil_score': [cibil_score],
            'residential_assets_value': [residential_assets_value],
            'commercial_assets_value': [commercial_assets_value],
            'luxury_assets_value': [luxury_assets_value],
            'bank_asset_value': [bank_asset_value]
        })

        # Make prediction
        prediction = model.predict(input_data)

        # Display result
        if prediction[0] == 1:
            st.success("Congratulations! Your Loan is Approved. ✅")
        else:
            st.error("Sorry, Your Loan is Rejected. ❌")

if __name__ == "__main__":
    main()
