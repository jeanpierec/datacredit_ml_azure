from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Cargar el modelo
model = joblib.load('random_forest_model.pkl')

# Ruta para la página principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para manejar la predicción
@app.route('/predict', methods=['POST'])
def predict():
    # Obtén los valores de las características desde el formulario
    features = []
    for feature in ['GENDER', 'Car_Owner', 'Propert_Owner', 'CHILDREN',
                    'Annual_income', 'Type_Income', 'EDUCATION', 'Marital_status',
                    'Housing_type', 'Birthday_count', 'Employed_days', 'Mobile_phone',
                    'Work_Phone', 'Phone', 'Type_Occupation', 'Family_Members']:
        value = float(request.form[feature])
        features.append(value)

    # Convierte las características a un DataFrame
    input_data = pd.DataFrame([features], columns=['GENDER', 'Car_Owner', 'Propert_Owner', 'CHILDREN',
                                                   'Annual_income', 'Type_Income', 'EDUCATION', 'Marital_status',
                                                   'Housing_type', 'Birthday_count', 'Employed_days', 'Mobile_phone',
                                                   'Work_Phone', 'Phone', 'Type_Occupation', 'Family_Members'])

    # Realiza la predicción
    prediction = model.predict(input_data)

    # Muestra la predicción en el HTML
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)

