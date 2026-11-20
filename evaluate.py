
import os
import json
import torch
import torch.nn.functional as F
import numpy as np

from dataset import create_dataloaders
from resnet_face import myResnet20
from config import (
    CHECKPOINT_PATH,
    EVALUATION_LOG_PATH as OUTPUT_PATH,
    EMBEDDING_DIM,
)


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(device):
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    model = myResnet20(
        embedding_dim=EMBEDDING_DIM
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def extract_embeddings(model, loader, device):
    all_embeddings = []
    all_labels = []

    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            embeddings = model(images)
            embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

    return (
        torch.cat(all_embeddings).numpy(),
        torch.cat(all_labels).numpy()
    )


def calculate_pair_scores(embeddings, labels):
    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    embeddings = F.normalize(embeddings, p=2, dim=1)

    similarity_matrix = embeddings @ embeddings.T

    genuine_scores = []
    impostor_scores = []

    count = len(labels)

    for i in range(count):
        for j in range(i + 1, count):
            score = similarity_matrix[i, j].item()

            if labels[i] == labels[j]:
                genuine_scores.append(score)
            else:
                impostor_scores.append(score)

    return (
        np.array(genuine_scores),
        np.array(impostor_scores)
    )


def evaluate_thresholds(genuine, impostor):
    thresholds = np.linspace(-1.0, 1.0, 2001)

    results = []

    for threshold in thresholds:
        true_accept = np.mean(genuine >= threshold)
        false_reject = 1.0 - true_accept

        false_accept = np.mean(impostor >= threshold)
        true_reject = 1.0 - false_accept

        balanced_accuracy = (
            true_accept + true_reject
        ) / 2.0

        results.append({
            "threshold": float(threshold),
            "balanced_accuracy": float(balanced_accuracy),
            "false_accept_rate": float(false_accept),
            "false_reject_rate": float(false_reject)
        })

    best = max(
        results,
        key=lambda item: item["balanced_accuracy"]
    )

    eer_candidate = min(
        results,
        key=lambda item: abs(
            item["false_accept_rate"] -
            item["false_reject_rate"]
        )
    )

    return best, eer_candidate


def main():
    device = select_device()
    print("Evaluation device:", device)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    _, val_loader, class_to_idx = create_dataloaders(
        batch_size=16
    )

    model, checkpoint = load_model(device)

    print("Loaded checkpoint from epoch:", checkpoint["epoch"])
    print("Validation images:", len(val_loader.dataset))
    print("Identities:", len(class_to_idx))

    embeddings, labels = extract_embeddings(
        model,
        val_loader,
        device
    )

    genuine, impostor = calculate_pair_scores(
        embeddings,
        labels
    )

    print("\nEmbedding evaluation")
    print("--------------------")
    print("Embedding shape:", embeddings.shape)
    print("Genuine pairs:", len(genuine))
    print("Impostor pairs:", len(impostor))

    print("\nSimilarity statistics")
    print("---------------------")
    print(f"Genuine mean:  {genuine.mean():.4f}")
    print(f"Genuine min:   {genuine.min():.4f}")
    print(f"Genuine max:   {genuine.max():.4f}")
    print(f"Impostor mean: {impostor.mean():.4f}")
    print(f"Impostor min:  {impostor.min():.4f}")
    print(f"Impostor max:  {impostor.max():.4f}")

    best, eer_candidate = evaluate_thresholds(
        genuine,
        impostor
    )

    print("\nBest balanced-accuracy threshold")
    print("--------------------------------")
    print(f"Threshold: {best['threshold']:.4f}")
    print(f"Balanced accuracy: {best['balanced_accuracy'] * 100:.2f}%")
    print(f"False acceptance rate: {best['false_accept_rate'] * 100:.2f}%")
    print(f"False rejection rate: {best['false_reject_rate'] * 100:.2f}%")

    print("\nApproximate EER threshold candidate")
    print("-----------------------------------")
    print(f"Threshold: {eer_candidate['threshold']:.4f}")
    print(f"FAR: {eer_candidate['false_accept_rate'] * 100:.2f}%")
    print(f"FRR: {eer_candidate['false_reject_rate'] * 100:.2f}%")

    report = {
        "checkpoint_epoch": checkpoint["epoch"],
        "embedding_dimension": EMBEDDING_DIM,
        "validation_images": len(labels),
        "genuine_pairs": len(genuine),
        "impostor_pairs": len(impostor),
        "genuine_mean": float(genuine.mean()),
        "impostor_mean": float(impostor.mean()),
        "best_threshold": best,
        "eer_candidate": eer_candidate
    }

    os.makedirs("training_logs", exist_ok=True)

    with open(OUTPUT_PATH, "w") as file:
        json.dump(report, file, indent=4)

    print("\nEvaluation report saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
