import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Cache the dataset loading to avoid re-processing on every interaction
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Function to preprocess data
@st.cache_data
def preprocess_data(df):
    # Dropping missing values (Null values)
    df = df.dropna()

    # Step 2: Remove duplicates
    df = df.drop_duplicates()

    # Step 3: Label Encoding for categorical columns (excluding 'Class')
    label_encoder = {}
    for col in df.select_dtypes(include='object').columns:
        if col != 'Class':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoder[col] = le

    # Step 4: Ordinal Encoding for 'Class' column (if applicable)
    class_order = ['Economy', 'Business', 'Economy Plus']  # Modify if needed based on your data
    ordinal_encoder = OrdinalEncoder(categories=[class_order])
    df['Class'] = ordinal_encoder.fit_transform(df[['Class']])

    return df

# Function to plot graphs
def plot_graph(graph_type, data):
    if graph_type == "Count Plot (Single Column)":
        col = st.selectbox("Select a column for the count plot", data.columns)
        if col:
            plt.figure(figsize=(10, 6))
            sns.countplot(x=data[col], hue=None, palette="viridis", legend=False)
            plt.title(f"Count Plot for {col}", fontsize=16)
            st.pyplot(plt)
            plt.close()

    elif graph_type == "Histogram (Single Numeric Column)":
        numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
        col = st.selectbox("Select a numeric column for the histogram", numeric_columns)
        if col:
            plt.figure(figsize=(10, 6))
            sns.histplot(data[col], bins=20, kde=True, color="skyblue")
            plt.title(f"Histogram for {col}")
            st.pyplot(plt)
            plt.close()

    elif graph_type == "Bar Plot (Two Columns: X vs Hue)":
        x_col = st.selectbox("Select the X-axis column", data.columns)
        hue_col = st.selectbox("Select the Hue column (optional)", ["None"] + list(data.columns))
        if x_col:
            plt.figure(figsize=(10, 6))
            if hue_col != "None":
                sns.countplot(x=data[x_col], hue=data[hue_col], palette="coolwarm")
                plt.title(f"Bar Plot: {x_col} vs {hue_col}")
            else:
                sns.countplot(x=data[x_col], hue=None if hue_col == "None" else hue_col, palette="coolwarm", legend=False)
                plt.title(f"Bar Plot for {x_col}")
            st.pyplot(plt)
            plt.close()

# Define a function to train and evaluate a model
def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get probabilities for ROC
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    return accuracy, y_pred, y_prob

# Function for Retention Analysis
def retention_analysis(df):
    st.subheader("Customer Retention Analysis")
    
    # List of columns to be aggregated for customer ratings
    rating_columns = [
        'Departure and Arrival Time Convenience',
        'Ease of Online Booking',
        'Check-in Service',
        'Online Boarding',
        'Gate Location',
        'On-board Service',
        'Seat Comfort',
        'Leg Room Service',
        'Cleanliness',
        'Food and Drink',
        'In-flight Service',
        'In-flight Wifi Service',
        'In-flight Entertainment',
        'Baggage Handling'
    ]

    # Check if rating columns exist in the dataset
    missing_cols = [col for col in rating_columns if col not in df.columns]
    if missing_cols:
        st.warning(f"Missing rating columns: {', '.join(missing_cols)}")
        return

    # Compute the mean rating for each customer
    df['mean_rating'] = df[rating_columns].mean(axis=1)

    # Get the highest and lowest mean ratings
    highest_mean_rating = df['mean_rating'].max()
    lowest_mean_rating = df['mean_rating'].min()

    st.write(f"### Highest Mean Rating: {highest_mean_rating:.2f}")
    st.write(f"### Lowest Mean Rating: {lowest_mean_rating:.2f}")

    # Compute the 75th percentile of the mean_rating
    threshold = df['mean_rating'].quantile(0.75)

    # Classify customers as retained or not based on the threshold
    df['retained'] = (df['mean_rating'] >= threshold).astype(int)

    st.write(f"### Retention Threshold (75th Percentile): {threshold:.2f}")
    st.write("### First Few Rows of Retention Data")
    st.write(df[['mean_rating', 'retained']].head())

    # Plot the distribution of the 'retained' column
    st.write("### Customer Retention Distribution (Bar Chart)")
    plt.figure(figsize=(6, 4))
    sns.countplot(x='retained', data=df, palette='Set2')
    plt.title('Customer Retention Distribution')
    plt.xlabel('Retention Status')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Not Retained', 'Retained'])
    st.pyplot(plt)

    # Count the occurrences of each category in the 'retained' column
    retained_counts = df['retained'].value_counts()

    # Plot the pie chart
    st.write("### Customer Retention Distribution (Pie Chart)")
    plt.figure(figsize=(6, 6))
    plt.pie(
        retained_counts,
        labels=['Not Retained', 'Retained'],
        autopct='%1.1f%%',
        startangle=90,
        colors=['lightgreen', 'lightblue']
    )
    plt.title('Customer Retention Distribution')
    plt.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.
    st.pyplot(plt)

# Function for Data Exploration Page
def data_exploration(data):
    st.title("Data Exploration")
    st.write("### Data Preview")
    st.write(data.head())

    # Visualization options
    st.subheader("Data Visualization")
    graph = st.selectbox("Select a plot type to display", [
        "Count Plot (Single Column)",
        "Histogram (Single Numeric Column)",
        "Bar Plot (Two Columns: X vs Hue)",
    ])
    
    plot_graph(graph, data)

# Function for Model Training and Evaluation Page
def model_training(data):
    st.title("Train and Evaluate Machine Learning Models")
    target_column = st.selectbox("Select the target column (label)", data.columns)
    if target_column:
        # Preprocess data: encode categorical variables
        data_encoded = preprocess_data(data)

        # Split the data into features and target
        X = data_encoded.drop(target_column, axis=1)
        y = data_encoded[target_column]
        
        # Train-test split and scaling...
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scaling the data
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model selection
        model_option = st.selectbox(
            "Select a machine learning model to train",
            ["All Models", "Logistic Regression", "Random Forest", "XGBoost"]
        )

        if st.button("Train and Compare Models"):
            accuracies = {}  # dictionary to store accuracies of each model
            roc_data = {}  # dictionary to store ROC curve data

            # Defining all models in a dictionary
            models = {
                "Logistic Regression": LogisticRegression(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "XGBoost": xgb.XGBClassifier(random_state=42)
            }

            for model_name, model_instance in models.items():
                # Train and evaluate each model
                accuracy, _, y_prob = train_and_evaluate(model_instance, X_train_scaled, y_train, X_test_scaled, y_test)
                accuracies[model_name] = accuracy

                # Calculate ROC data if probabilities are available
                if y_prob is not None:
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                    roc_data[model_name] = (fpr, tpr, roc_auc)

            # Plot accuracy comparison
            st.write("### Model Accuracy Comparison")
            plot_accuracy_comparison(accuracies)

            # Plot ROC curves
            st.write("### ROC Curve Comparison")
            plot_roc_curves(roc_data)

            # Display detailed metrics for the selected model
            if model_option != "All Models":
                selected_model = models[model_option]
                _, y_pred, _ = train_and_evaluate(selected_model, X_train_scaled, y_train, X_test_scaled, y_test)
                st.write(f"### {model_option} Performance Metrics")
                st.write("Accuracy:", accuracies[model_option])
                st.write("Classification Report:")
                st.text(classification_report(y_test, y_pred))
                st.write("Confusion Matrix:")
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Retained", "Retained"], yticklabels=["Not Retained", "Retained"])
                st.pyplot(plt)


# Function to plot accuracy comparison
def plot_accuracy_comparison(accuracies):
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), hue=None, palette="Set2")
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    st.pyplot(plt)

# Function to plot ROC curves
def plot_roc_curves(roc_data):
    plt.figure(figsize=(8, 6))
    for model_name, (fpr, tpr, roc_auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("Receiver Operating Characteristic (ROC) Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    st.pyplot(plt)

# Define main function to handle the navigation and multipage structure
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page", ["Home", "Model Training", "Retention Analysis"])
    
    file = st.file_uploader("Upload your dataset (CSV file)", type="csv")
    if file is not None:
        data = load_data(file)
        
        if page == "Home":
            st.title("Welcome to the AI Model Dashboard")
            st.write("This dashboard allows you to explore data and train machine learning models.")
            data_exploration(data)
        
        elif page == "Model Training":
            model_training(data)
        
        elif page == "Retention Analysis":
            retention_analysis(data)
        
    else:
        st.info("Please upload a CSV file to get started.")

# Run the main function
if __name__ == "__main__":
    main()
