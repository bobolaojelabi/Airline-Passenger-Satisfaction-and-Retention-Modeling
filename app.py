import streamlit as st
import pandas as pd # to load the dataset
import matplotlib.pyplot as plt
import seaborn as sns

#Dashboard Title
st.title("Interactive Dashboard for my AI Model")

#Description
st.write("This is an interactive dashboard to expore my data and visualiza the models")

#Data upload
file = st.file_uploader("upload your dataset (csv file)", type="csv")
if file is not None:
    data = pd.read_csv(file)
    st.write("Data preview")
    st.write(data.head())

    #checking for missing values
    st.subheader("Data Quality Check")
    st.write("Checking for missing values")
    null_counts = data.isnull().sum()  # Count of null values in each column
    st.write(f"Missing values: {null_counts}")

    if null_counts.sum() > 0:
        st.warning("Data contains missing values")
        data = data.dropna()
        st.success("Missing Values dropped")
        # Recalculate null_counts after dropping missing values
        null_counts = data.isnull().sum()
        st.write(f"Missing values: {null_counts}")
    else:
        st.success("No missing values found.")

    #checking for duplicates
    st.write("Checking for duplicates rows")
    duplicate_count = data.duplicated().sum()
    st.write(f"Duplicate rows: {duplicate_count}")

    if duplicate_count > 0:
        st.warning("Data contains duplicate rows")
        data = data.drop_duplicates()
        st.success("Duplicate rows drop")
        # Recalculate duplicate_count after removing duplicates
        duplicate_count = data.duplicated().sum()
        st.write(f"Duplicate rows: {duplicate_count}")
    else:
        st.success("No duplicate rows found.")

    # Visualization options
    st.subheader("Data Visualization")
    graph = st.selectbox("Select a plot to display", [
        "Satisfaction plot",
        "Age Distribution",
        "Customer Type vs Satisfaction",
        "Type class vs Satisfaction",
        "count of Travel Type"
    ])

    # Plot for the target(satisfaction)
    if graph == "Satisaction plot":
        st.write("count plot for Satisfaction")
        plt.figure(figsize=(10, 5))
        sns.countplot(x = 'Satisfaction', data = data)
        plt.title("Count Plot for Satisfaction", fontsize=16)
        plt.xlabel("Satisfaction", fontsize=14)
        plt.ylabel("Count", fontsize=14)
        plt.tight_layout()
        st.pyplot(plt)

    # Plot histogram for Age
    elif graph == "Histogram for Age Distribution":
        st.write("### Age Distribution")
        plt.figure(figsize=(10, 6))
        sns.histplot(data['Age'], bins=20, kde=True, color='skyblue')
        plt.title("Distribution of Age", fontsize=16)
        plt.xlabel("Age", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(plt)

    # Plot of Relationship Between Customer Type Distribution and Satisfaction
    elif graph == "Customer Type vs Satisfaction":
        st.write("### Relationship Between Customer Type and Satisfaction")
        plt.figure(figsize=(8, 6))
        sns.countplot(x='Customer Type', hue='Satisfaction', data=data, palette='Blues')
        plt.title("Customer Type vs Satisfaction", fontsize=16)
        plt.xlabel("Customer Type", fontsize=14)
        plt.ylabel("Count", fontsize=14)
        plt.tight_layout()
        st.pyplot(plt)

    # Plot of Relationship Between Travel Class Distribution and Satisfaction Levels
    elif graph == "Travel Class vs Satisfaction":
        st.write("### Travel Class vs Satisfaction Levels")
        plt.figure(figsize=(8, 6))
        sns.countplot(x='Class', hue='Satisfaction', data=data, palette='coolwarm')
        plt.title("Travel Class vs Satisfaction Levels", fontsize=16)
        plt.xlabel("Class", fontsize=14)
        plt.ylabel("Count", fontsize=14)
        plt.tight_layout()
        st.pyplot(plt)

    # Plot of Travel Types
    elif graph == "Count of Travel Types":
        st.write("### Count of Travel Types")
        plt.figure(figsize=(8, 6))
        sns.countplot(data=data, x='Type of Travel', palette='viridis')
        plt.title("Count of Type of Travel", fontsize=16)
        plt.xlabel("Type of Travel", fontsize=14)
        plt.ylabel("Count", fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(plt)