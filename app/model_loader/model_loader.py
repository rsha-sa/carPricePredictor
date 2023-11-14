import torch
from app.ml_model.MLproject.MLproject import CarPricePredictor 

def load_model():
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Specify the path to the model file
    model_path = os.path.join(current_dir, 'carPricePredictor.pth')

    # Initialize the model
    model = CarPricePredictor(14)  # Replace with the correct input size

    # Load the trained model state
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Load the state dict into the model
    model.load_state_dict(state_dict)

    # Set the model in evaluation mode
    model.eval()

    return model