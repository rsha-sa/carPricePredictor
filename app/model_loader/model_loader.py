import torch
from app.ml_model.MLproject import CarPricePredictor
import os 


def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(current_dir, 'carPricePredictor.pth')

    model = CarPricePredictor(84) 

    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    model.load_state_dict(state_dict)

    model.eval()

    return model

