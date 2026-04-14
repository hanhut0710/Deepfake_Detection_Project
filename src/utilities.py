import torch
import numpy as np
import cv2
import os
from src.preprocessing import process_video_from_outsource, extract_face
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from collections import Counter

def load_model(model_path, model, optimizer, scheduler, device = None):
    if (device is None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    history = checkpoint['history']

    model.to(device)
    model.eval()
    return model, optimizer, scheduler, history

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

    #Learning rate
    plt.figure()
    plt.plot(epochs, history['lr'], marker="o", label="Learning rate") 
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.title("Learning rate")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def plot_roc_curve(y_true, y_probs, save_path=None):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,6))

    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1], [0,1], linestyle="--")  # random line

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    print(f"AUC: {roc_auc:.4f}")

def plot_confusion_matrix(y_true, y_pred, class_names=["Fake", "Real"], save_path=None):

    cm = confusion_matrix(y_true, y_pred)
    
    # normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6,6))

    im = ax.imshow(cm_norm)

    # labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    # text inside cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i,
                f"{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)",
                ha="center", va="center"
            )

    plt.colorbar(im)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def print_classification_report(y_true, y_pred):

    report = classification_report(
        y_true,
        y_pred,
        target_names=["Fake", "Real"],
        digits=4
    )

    print(report)

def plot_all_splits(cnf):
    from torchvision import datasets

    splits = {
        "Train": cnf.TRAIN_DIR,
        "Val": cnf.VAL_DIR,
        "Test": cnf.TEST_DIR
    }

    plt.figure(figsize=(12,4))

    for i, (name, path) in enumerate(splits.items(), 1):
        dataset = datasets.ImageFolder(root=path)
        labels = [label for _, label in dataset]
        counter = Counter(labels)

        classes = list(counter.keys())
        counts = list(counter.values())

        plt.subplot(1,3,i)
        bars = plt.bar(classes, counts)
        plt.xticks(classes, ["Fake", "Real"])

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1,
                str(int(height)),
                ha='center',
                va='bottom'
            )

        plt.title(name)

    plt.tight_layout()
    plt.show()

def predict_video_visual(model, video_path, transform, device, output_path="output.mp4"):

    model.eval()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Cannot open video")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps == 0:
        fps = 25

    frame_count = 0

    fake_prob = []
    frame_buffer = []

    fourcc = cv2.VideoWriter_fourcc(*"H264")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (w, h))

        frame_buffer.append(frame)

        result = extract_face(frame)

        if not result:
            continue

        face, (x1, y1, x2, y2) = result

        if face is not None:

            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

            x = transform(face_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(x)
                prob = torch.sigmoid(output).item()
                fake_prob.append(1 - prob)

            color = (0,255,0) if prob > 0.5 else (0,0,255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        else:
            cv2.putText(
                frame,
                "No face",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,255),
                2
            )
        fake_prob.append(0.0)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()

    ### prediction k top

    if len(fake_prob) == 0:
        print("No face detected in entire video")
        return

    k = int(0.3 * len(fake_prob))
    k_top = sorted(fake_prob, reverse=True)[:k]

    score = sum(k_top) / len(k_top)
    confidence = 1 - score

    final_label = "Fake" if score > 0.5 else "Real"

    text = f"Predict: {final_label} | Confidence: {confidence:.2f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9   # nhỏ lại
    thickness = 1

    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # vị trí text: phía trên bounding box
    x = x1
    y = y1 - 10

    # nếu quá sát mép trên thì đẩy xuống dưới box
    if y < text_h:
        y = y1 + text_h + 10

    for idx, frame in enumerate(frame_buffer):

        color = (0, 255, 0) if final_label == "Real" else (0, 0, 255)

        if (idx > int(0.8 * len(frame_buffer))):
            cv2.putText(frame,
                        text,
                        (x, y),
                        font,
                        font_scale,
                        color,
                        thickness,
                        cv2.LINE_AA)
        else:
            cv2.putText(
                frame,
                f"Probability: ({fake_prob[idx]:.2f})",
                (x, y),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA
            )

        out.write(frame)


    out.release()

    print(f"Saved video to {output_path}")