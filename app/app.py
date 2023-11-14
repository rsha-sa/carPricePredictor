from flask import Flask, render_template, request, url_for
from app.model_loader.model_loader import load_model
import datetime
import torch 

app = Flask(__name__, static_folder='app/static')


model = load_model()

@app.route('/')
def index():
    img_url = url_for('static', filename='app/static/logo.png')
    return render_template('index.html', img_url=img_url)


@app.route("/predict", methods=['POST'])
def predict():
    # Extract input data from the form
    mileage = float(request.form['mileage'])
    engineV = float(request.form['engineV'])
    year = int(request.form['year'])
    brand = request.form['brand']
    engine_type = request.form['engineType']  

    # Convert brand and engine_type to one-hot encoded vectors
    brands_to_encode = ['Volkswagen', 'Mercedes-Benz', 'BMW', 'Toyota', 'Renault', 'Audi', 'Mitsubishi']
    brand_vector = [1 if brand == b else 0 for b in brands_to_encode]

    engine_type_encode = ['Diesel', 'Petrol', 'Gas', 'other']
    engine_type_vector = [1 if engine_type == et else 0 for et in engine_type_encode]

    # Calculate age based on the current year
    current_year = 2023  # Replace with the actual current year
    age = current_year - year

    # Prepare input data for the model
    input_data = [mileage, engineV, age] + brand_vector + engine_type_vector

    # Convert to PyTorch tensor
    input_tensor = torch.FloatTensor(input_data).unsqueeze(0)

    # Make prediction using the model
    prediction = model(input_tensor)

    # Extract the predicted value from the tensor
    output = round(prediction.item(), 2)

    return render_template('index.html', prediction_text=f'The predicted price is ${output}K')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
    