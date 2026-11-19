
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import create_dataloaders
from resnet_face import myResnet20
from losses import ArcFaceLoss


BATCH_SIZE = 16
EPOCHS = 60
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
EMBEDDING_DIM = 128

CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "training_logs"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, arcface, loader, optimizer, device):
    model.train()
    arcface.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        embeddings = model(images)
        loss, accuracy = arcface(embeddings, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += accuracy.item() * batch_size
        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )


def validate(model, arcface, loader, device):
    model.eval()
    arcface.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            embeddings = model(images)
            loss, accuracy = arcface(embeddings, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += accuracy.item() * batch_size
            total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )


def main():
    device = select_device()
    print("Using device:", device)

    train_loader, val_loader, class_to_idx = create_dataloaders(
        batch_size=BATCH_SIZE
    )

    num_classes = len(class_to_idx)

    print("Number of identities:", num_classes)
    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))

    model = myResnet20(
        embedding_dim=EMBEDDING_DIM
    ).to(device)

    arcface = ArcFaceLoss(
        embedding_dim=EMBEDDING_DIM,
        num_classes=num_classes,
        margin=0.5,
        scale=32.0
    ).to(device)

    optimizer = optim.AdamW(
        list(model.parameters()) + list(arcface.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )

    best_val_accuracy = 0.0
    best_val_loss = float("inf")

    history = []

    for epoch in range(EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, arcface, train_loader, optimizer, device
        )

        val_loss, val_acc = validate(
            model, arcface, val_loader, device
        )

        scheduler.step(val_loss)

        elapsed = time.time() - start_time

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc * 100:.2f}% "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc * 100:.2f}% "
            f"Time: {elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

        improved = (
            val_acc > best_val_accuracy
            or (
                val_acc == best_val_accuracy
                and val_loss < best_val_loss
            )
        )

        if improved:
            best_val_accuracy = val_acc
            best_val_loss = val_loss

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": {
                    k: v.detach().cpu()
                    for k, v in model.state_dict().items()
                },
                "arcface_state_dict": {
                    k: v.detach().cpu()
                    for k, v in arcface.state_dict().items()
                },
                "class_to_idx": class_to_idx,
                "embedding_dim": EMBEDDING_DIM,
                "best_val_accuracy": best_val_accuracy,
                "best_val_loss": best_val_loss
            }

            torch.save(
                checkpoint,
                os.path.join(CHECKPOINT_DIR, "best_resnet.pth")
            )

            print("Best checkpoint saved.")

    history_path = os.path.join(LOG_DIR, "training_history.json")

    with open(history_path, "w") as file:
        json.dump(history, file, indent=4)

    with open(
        os.path.join(CHECKPOINT_DIR, "class_mapping.json"), "w"
    ) as file:
        json.dump(class_to_idx, file, indent=4)

    print("\nTraining completed.")
    print(f"Best validation accuracy: {best_val_accuracy * 100:.2f}%")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Checkpoint:", os.path.join(CHECKPOINT_DIR, "best_resnet.pth"))


if __name__ == "__main__":
    main()
