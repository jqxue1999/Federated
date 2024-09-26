import torch.nn as nn
from transformers import ViTForImageClassification

def init_model(freeze_layers=0):
    model_name = "google/vit-base-patch16-224"
    model = ViTForImageClassification.from_pretrained(model_name)
    model.classifier = nn.Linear(model.classifier.in_features, 10)

    if freeze_layers > 0:
        for param in model.vit.embeddings.parameters():
            param.requires_grad = False

        for i in range(freeze_layers):
            for param in model.vit.encoder.layer[i].parameters():
                param.requires_grad = False

    return model