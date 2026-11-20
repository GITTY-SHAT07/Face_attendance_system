
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    EMBEDDING_DIM,
    DEFAULT_NUM_CLASSES,
    ARCFACE_MARGIN,
    ARCFACE_SCALE,
)


class ArcFaceLoss(nn.Module):
    def __init__(
        self,
        embedding_dim=EMBEDDING_DIM,
        num_classes=DEFAULT_NUM_CLASSES,
        margin=ARCFACE_MARGIN,
        scale=ARCFACE_SCALE
    ):
        super().__init__()

        self.margin = margin
        self.scale = scale
        self.num_classes = num_classes

        self.weight = nn.Parameter(
            torch.empty(num_classes, embedding_dim)
        )

        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        cosine = F.linear(
            F.normalize(embeddings),
            F.normalize(self.weight)
        )

        sine = torch.sqrt(
            torch.clamp(1.0 - cosine.pow(2), min=1e-7)
        )

        phi = (
            cosine * self.cos_m
            - sine * self.sin_m
        )

        phi = torch.where(
            cosine > self.threshold,
            phi,
            cosine - self.mm
        )

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = (
            one_hot * phi
            + (1.0 - one_hot) * cosine
        )

        logits *= self.scale

        loss = F.cross_entropy(logits, labels)

        predictions = logits.argmax(dim=1)
        accuracy = (predictions == labels).float().mean()

        return loss, accuracy
