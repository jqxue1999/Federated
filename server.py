import torch
import numpy as np

class FederatedServer:
    def __init__(self, model, device, quantize_bits):
        self.device = device
        self.model = model.to(self.device)
        self.quantize_bits = quantize_bits

    def receive_updates(self, client_updates):
        # Aggregate updates from clients using averaging
        aggregated_updates = {}
        num_clients = len(client_updates)

        for key in client_updates[0].keys():
            # Stack updates from all clients
            stacked_updates = np.stack([update[key] for update in client_updates])
            # Compute average update
            aggregated_updates[key] = np.mean(stacked_updates, axis=0)

        return aggregated_updates

    def update_model(self, aggregated_updates):
        # Update server model with aggregated updates
        with torch.no_grad():
            for key, value in self.model.state_dict().items():
                if key in aggregated_updates:
                    # Convert numpy array to torch tensor
                    update_tensor = torch.from_numpy(aggregated_updates[key]).to(self.device)

                    # Dequantize the update
                    update_tensor = update_tensor.float() / (2 ** (self.quantize_bits - 1) - 1)

                    # Apply the update
                    value.add_(update_tensor)

    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy

    def get_model_state_dict(self):
        """Return the current state dict of the global model."""
        return self.model.state_dict()