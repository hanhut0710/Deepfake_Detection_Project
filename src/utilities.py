import torch
import os
from src.preprocessing import process_video_from_outsource
from PIL import Image
import matplotlib.pyplot as plt

def load_model(model_path, model, device = None):
    if (device is None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    history = checkpoint['history']

    model.to(device)
    model.eval()
    return model, history

def predict_frame(model, image, transform, device):

    image = Image.fromarray(image)

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image)
        prob = torch.sigmoid(output)    

    return prob.item()

def predict_video(model, video_path, transform, device):

    model.eval()

    votes_real = 0
    votes_fake = 0
    fakes_prob = []

    faces = process_video_from_outsource(video_path)

    if len(faces) == 0:
        print("No face detected")
        return 0.0

    for face in faces:
        real_prob = predict_frame(model, face, transform, device)

        fakes_prob.append(1 - real_prob)
       
    k = max(1, int(0.3 * len(fakes_prob)))
    k_top = sorted(fakes_prob, reverse=True)[:k]

    score = sum(k_top) / len(k_top)

    label = "Fake" if score > 0.5 else "Real"

    # print("Votes REAL:", votes_real)
    # print("Votes FAKE:", votes_fake)
    print("Average probability", score)
    print("Video prediction:", label)
    # print("Confidence:", confidence)

    return label, score

def plot_history(history):

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # Accuracy
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()