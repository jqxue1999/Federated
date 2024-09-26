import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

class FederatedClient:
    def __init__(self, model, data, args, device):
        self.device = device
        self.model = model.to(self.device)
        self.data = data
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.local_epochs = args.local_epochs
        self.quantize_bits = args.quantize_bits
        self.round_start_weights = self.get_model_weights()

    def local_update(self, client_id):
        self.model.train()
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            with tqdm(self.data, desc=f"Client {client_id}, Epoch {epoch + 1}/{self.local_epochs}",
                      leave=False, ncols=100, colour='yellow') as pbar:
                for batch_idx, (data, target) in enumerate(pbar):
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output.logits, target)
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    pbar.set_postfix({'loss': f'{epoch_loss / (batch_idx + 1):.4f}'})

        current_weights = self.get_model_weights()
        updates = {}
        for k in current_weights.keys():
            updates[k] = current_weights[k] - self.round_start_weights[k]

        return updates

    def quantize_updates(self, updates):
        """Quantize weight updates to specified bits."""
        scale_factor = 2 ** (self.quantize_bits - 1) - 1
        quantized_updates = {}
        for k, v in updates.items():
            # Scale the updates
            scaled = v * scale_factor
            # Clip values to the valid range for the specified number of bits
            max_val = 2 ** (self.quantize_bits - 1) - 1
            min_val = -2 ** (self.quantize_bits - 1)
            clipped = np.clip(scaled, min_val, max_val)
            # Convert to int dtype that can hold the quantized values
            if self.quantize_bits <= 8:
                dtype = np.int8
            elif self.quantize_bits <= 16:
                dtype = np.int16
            else:
                dtype = np.int32
            quantized_updates[k] = clipped.astype(dtype)
        return quantized_updates

    def update_local_model(self, aggregated_updates):
        """Update the local model with the aggregated updates."""
        with torch.no_grad():
            for key, value in self.model.state_dict().items():
                if key in aggregated_updates:
                    update_tensor = torch.from_numpy(aggregated_updates[key]).to(self.device)
                    # Dequantize the update
                    update_tensor = update_tensor.float() / (2 ** (self.quantize_bits - 1) - 1)
                    # Apply the update
                    value.add_(update_tensor)
        # Update round start weights
        self.round_start_weights = self.get_model_weights()

    def get_model_weights(self):
        return {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}

    def get_model_state_dict(self):
        """Return the current state dict of the global model."""
        return self.model.state_dict()