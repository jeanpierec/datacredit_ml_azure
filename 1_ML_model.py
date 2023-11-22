# once upon a time
import pandas as pd
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
import pickle
# metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# data manage
from sklearn.model_selection import train_test_split

# Importa la data
df= pd.read_csv("data/Credit_Card.csv", sep=";")

domain_dict = {}

# Inicializar el LabelEncoder
label_encoder = LabelEncoder()

# Iterar sobre las columnas del DataFrame
for column in df.columns:
    if df[column].dtype == 'object':
        # Si la columna es de tipo 'object', aplicar LabelEncoder
        df[column] = label_encoder.fit_transform(df[column])
        # Almacenar el dominio de valores en el diccionario
        domain_dict[column] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    else:
        # Si la columna no es de tipo 'object', simplemente almacenar el dominio de valores
        domain_dict[column] = df[column].unique().tolist()

# Fill NaN values with the most common value (mode) of each column
df = df.apply(lambda x: x.fillna(x.mode().iloc[0]))

# Select features (independent variables) and labels (dependent variable)
features = df.drop(['label','Ind_ID','EMAIL_ID'], axis=1)
labels = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize an trainthe random forest classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Inicializar y entrenar el LabelEncoder
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
label_encoder_dict = dict(zip(labels, encoded_labels))

# Guardar el modelo y el LabelEncoder en archivos
with open('random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)

with open('label_encoder.pkl', 'wb') as encoder_file:
    pickle.dump(label_encoder_dict, encoder_file)