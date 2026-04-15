import torch
from torch.cuda.amp import autocast, GradScaler

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)

        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        correct_predictions += torch.sum(preds == labels)
        total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples

    return epoch_loss, epoch_acc.item()


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, config):

    scaler = GradScaler()

    best_val_acc = 0
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": []
    }

    for epoch in range(num_epochs):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f} | "
            f"Learning rate {current_lr:.9f}"
        )

        history['train_acc'].append(train_acc)
        history["val_acc"].append(val_acc)
        history['train_loss'].append(train_loss)
        history["val_loss"].append(val_loss)
        history['lr'].append(current_lr)

        scheduler.step(val_loss)

        if val_acc > best_val_acc or best_val_loss > val_loss:

            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                "best_val_acc": best_val_acc
            },f"models/{config.MODEL_NAME}.pth")

            print("✅ Best model saved")

        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("⛔ Early stopping triggered")
            break

def evaluate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:

            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            correct_predictions += torch.sum(preds == labels)
            total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples

    return epoch_loss, epoch_acc.item()

def predict_test(model, test_loader, device, model_path = None):

    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

    model.eval()

    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            output = model(inputs)  

            probs = torch.sigmoid(output)
            preds = (probs > 0.5).float()

            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
    
    acc = correct / total

    print(f"Test Accuracy: {acc}")

    return acc, all_preds, all_labels, all_probs