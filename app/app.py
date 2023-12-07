from flask import Flask, render_template, request, jsonify
from app.model_loader.model_loader import load_model
import datetime
import torch 
import numpy as np
import pandas as pd
import joblib




app = Flask(__name__)


model = load_model()

scaler = joblib.load('scaler.joblib')



@app.route('/')
def index():
    return render_template('index.html')


@app.route("/predict", methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Extract input data from the form
        mileage = float(request.form['mileage'])
        engineV = float(request.form['engineV'])
        year = int(request.form['year'])
        brand = request.form['brand']
        selected_model = request.form['model']
        engine_type = request.form['engineType'] 

        brand_and_model = f'{brand}_{selected_model}'


        print("Input Data from Form:", mileage, engineV, year, brand_and_model, engine_type)


        brands_and_models_to_encode  = [

        'Volkswagen_Caddy', 'Volkswagen_Passat-B6', 'Volkswagen_Touareg',
        'Volkswagen_Polo', 'Volkswagen_T5-(Transporter)',
        'Volkswagen_Passat-B5', 'Volkswagen_Passat-B7',
        'Volkswagen_T4-(Transporter)', 'Volkswagen_Jetta',
        'Volkswagen_T5-(Transporter) ', 'Volkswagen_Other',
        'Mercedes-Benz_E-Class', 'Mercedes-Benz_Vito', 'Mercedes-Benz_C-Class',
        'Mercedes-Benz_S-500', 'Mercedes-Benz_S-350', 'Mercedes-Benz_ML-350',
        'Mercedes-Benz_GLS-350', 'Mercedes-Benz_Sprinter-313',
        'Mercedes-Benz_S-550', 'Mercedes-Benz_S-320', 'Mercedes-Benz_Other',
        'BMW_X5', 'BMW_520', 'BMW_320', 'BMW_525', 'BMW_530', 'BMW_X6',
        'BMW_318', 'BMW_730', 'BMW_528', 'BMW_535', 'BMW_Other', 'Toyota_Camry',
        'Toyota_LandCruiserPrado', 'Toyota_Rav-4', 'Toyota_Corolla',
        'Toyota_LandCruiser-200', 'Toyota_Avensis', 'Toyota_Auris',
        'Toyota_LandCruiser-100', 'Toyota_Yaris', 'Toyota_Highlander',
        'Toyota_Other', 'Renault_Kangoo', 'Renault_Megane', 'Renault_Trafic',
        'Renault_Laguna', 'Renault_Scenic', 'Renault_Logan',
        'Renault_GrandScenic', 'Renault_Fluence', 'Renault_Symbol',
        'Renault_Master', 'Renault_Other', 'Audi_A6', 'Audi_Q7', 'Audi_A4',
        'Audi_A8', 'Audi_A6-Allroad', 'Audi_100', 'Audi_A5', 'Audi_Q5',
        'Audi_80', 'Audi_A3', 'Audi_Other', 'Mitsubishi_Lancer',
        'Mitsubishi_Lancer-X', 'Mitsubishi_PajeroWagon', 'Mitsubishi_Galant',
        'Mitsubishi_Outlander', 'Mitsubishi_PajeroSport',
        'Mitsubishi_Outlander-XL', 'Mitsubishi_L-200', 'Mitsubishi_Colt',
        'Mitsubishi_Carisma', 'Mitsubishi_Other'

        ]

        brand_and_model_vector = [1 if brand_and_model == bm else 0 for bm in brands_and_models_to_encode]


        engine_type_encode = ['Diesel', 'Gas', 'other', 'Petrol']
        engine_type_vector = [1 if engine_type == et else 0 for et in engine_type_encode]


        current_year = datetime.datetime.now().year  
        age = current_year - year


        numerical_data = [mileage, engineV, age]

        

        numerical_data= np.array(numerical_data, dtype=float)

        engine_type_vector = np.array(engine_type_vector)

        brand_and_model_vector = np.array(brand_and_model_vector)

        print("Input Data before:", numerical_data, engine_type_vector, brand_and_model_vector  )
        
        numerical_data_scaled = scaler.fit_transform(numerical_data.reshape(-1, 1))

        input_data = np.array([*numerical_data_scaled.flatten(), *engine_type_vector, *brand_and_model_vector]).reshape(1, -1)

        print("Input Data after:", input_data)
                                     
                                
        input_tensor = torch.FloatTensor(np.array(input_data))

        prediction = model(input_tensor)

        print("Raw Model Prediction:", prediction.item())

        output = round(prediction.item(), 2)


        return render_template('index.html', prediction_text=f'The predicted price is ${output}K', mileage=mileage, engineV=engineV, year=year, brand=brand, engineType=engine_type,  selected_model=selected_model)
    

    return render_template('index.html')



@app.route("/get_models/<brand>")
def get_models(brand):

    models = []
    if brand == 'Volkswagen':
        models = ['Caddy', 'Touareg', 'Passat-B6', 'T5-(Transporter)', 'Polo', 'Passat-B5', 'Passat-B7', 'Jetta', 'T4-(Transporter)', 'T5-(Transporter)', 'Other']
    elif brand == 'Mercedes-Benz':
        models = ['E-Class', 'Vito', 'C-Class', 'S-500', 'S-350', 'GLS-350', 'ML-350', 'GLE-Class', 'Sprinter-313', 'V-250', 'Other']
    elif brand == 'BMW':
        models = ['X5', '520', '320', '525', '530', 'X6', '318', '730', '535', '528', 'Other']
    elif brand == 'Toyota':
        models = ['Camry', 'LandCruiserPrado', 'Rav-4', 'Corolla', 'LandCruiser-200', 'Auris', 'Avensis', 'LandCruiser-100', 'Highlander', 'Yaris', 'Other']
    elif brand == 'Renault':
        models = ['Kangoo', 'Megane', 'Trafic', 'Laguna', 'Scenic', 'GrandScenic', 'Logan', 'Symbol', 'Master', 'Fluence', 'Other']
    elif brand == 'Audi':
        models = ['A6', 'Q7', 'A4', 'A8', 'A6-Allroad', '100', 'A5', '80', 'Q5', 'A3', 'Other']
    elif brand == 'Mitsubishi':
        models = ['Lancer', 'Lancer-X', 'PajeroWagon', 'Outlander', 'Galant', 'PajeroSport', 'Outlander-XL', 'L-200', 'Colt', 'Carisma', 'Other']

    return jsonify(models)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
    