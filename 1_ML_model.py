# once upon a time
import pandas as pd
import sklearn

# graph
import matplotlib.pyplot as plt
import seaborn as sns

# encoding
from sklearn.preprocessing import LabelEncoder

# models
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# data manage
from sklearn.model_selection import train_test_split

# Importa la data
df_base = pd.read_csv("data/Credit_Card.csv", sep=";")

def frequency_encode_non_numeric(df):
    """
    Apply frequency encoding to non-numeric columns in the DataFrame.

    Parameters:
    - df: pandas DataFrame

    Returns:
    - df_encoded: DataFrame with frequency-encoded non-numeric columns
    """
    label_encoder = LabelEncoder()
    df_encoded = df.copy()

    for column in df.columns:
        if df[column].dtype not in [int, float]:
            # If the column is not numeric, apply frequency encoding
            df_encoded[column] = label_encoder.fit_transform(df[column])

    return df_encoded

# Example usage:
# Load your DataFrame (replace 'your_file.csv' with your actual file name)
# df = pd.read_csv('your_file.csv')

# Apply frequency encoding to non-numeric columns
df = frequency_encode_non_numeric(df_base)

# Fill NaN values with the most common value (mode) of each column
df = df.apply(lambda x: x.fillna(x.mode().iloc[0]))

# Select features (independent variables) and labels (dependent variable)
features = df.drop('label', axis=1)
labels = df['label']

# Convert categorical variables to dummy variables (if necessary)
# features = pd.get_dummies(features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize the random forest classifier
rf_model = RandomForestClassifier(random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model performance
accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
classification_rep_rf = classification_report(y_test, y_pred_rf, zero_division=1)  # Set zero_division to 1 or any other value