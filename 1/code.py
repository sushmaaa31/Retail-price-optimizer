import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Streamlit title
st.title("🛒 Retail Price Optimizer Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload your retail price CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Overview")
    st.dataframe(data.head())

    st.write("### 🔍 Columns in the dataset")
    st.write(data.columns.tolist())

    # 1. Total Price Distribution Graph
    if 'total_price' in data.columns:
        st.subheader("📈 Total Price Distribution")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.histplot(data['total_price'], bins=30, kde=True, ax=ax1)
        ax1.set_title('Total Price Distribution')
        ax1.set_xlabel('Total Price')
        ax1.set_ylabel('Count')
        st.pyplot(fig1)

    # 2. Total Price vs Quantity
    if 'qty' in data.columns and 'total_price' in data.columns:
        st.subheader("📉 Total Price vs Quantity Sold")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='qty', y='total_price', data=data, ax=ax2)
        ax2.set_title('Total Price vs Quantity Sold')
        ax2.set_xlabel('Quantity Sold')
        ax2.set_ylabel('Total Price')
        st.pyplot(fig2)

    # 3. Prepare data
    if 'total_price' not in data.columns or 'qty' not in data.columns:
        st.error("Dataset must contain 'total_price' and 'qty' columns.")
    else:
        if 'actual_price' in data.columns:
            X = data.drop(['total_price', 'actual_price'], axis=1)
        else:
            X = data.drop(['total_price'], axis=1)

        y = data['total_price']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        label_encoder = LabelEncoder()
        for col in X_train.select_dtypes(include=['object']).columns:
            X_train[col] = label_encoder.fit_transform(X_train[col])
            X_test[col] = label_encoder.transform(X_test[col])

        # 4. Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 5. Predicted vs Actual
        st.subheader("🔮 Actual vs Predicted Retail Price")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.scatter(y_test, y_pred)
        ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax3.set_title('Actual vs Predicted Price')
        ax3.set_xlabel('Actual Price')
        ax3.set_ylabel('Predicted Price')
        st.pyplot(fig3)

        # 6. Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        accuracy_percentage = r2 * 100

        st.write(f"📌 **Mean Squared Error**: {mse:.2f}")
        st.write(f"📌 **R² Score**: {r2:.2f}")
        st.write(f"✅ **Accuracy Percentage**: {accuracy_percentage:.2f}%")

        # 7. Optimization
        st.subheader("💡 Optimize Prices (+10%)")

        # Select a numeric column to optimize
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Select the price column to optimize", numeric_cols)
            data['optimized_price'] = data[selected_col] * 1.10
            st.dataframe(data[[selected_col, 'optimized_price']].head())
        else:
            st.warning("No numeric columns found to optimize.")
