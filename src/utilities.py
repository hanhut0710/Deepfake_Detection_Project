import torch

def load_model(model_path, device = None):
    if (device is None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model