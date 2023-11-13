import torch
from app.ml_model.MLproject.MLproject import CarPricePredictor 

def load_model():
    # Initialize the model
    model = CarPricePredictor(14)  # Replace with the correct input size

    # Load the trained model state
    state_dict = torch.load('carPricePredictor.pth', map_location=torch.device('cpu'))

    # Load the state dict into the model
    model.load_state_dict(state_dict)

    model.eval()

    return model