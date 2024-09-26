import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from models import init_model
from client import FederatedClient
from server import FederatedServer
from utils import create_client_datasets, parse_args

if __name__ == "__main__":
    args = parse_args()

    if args.use_wandb:
        wandb.init(project="federated-learning", config=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = init_model(freeze_layers=args.freeze_layers)
    server = FederatedServer(model, device, args.quantize_bits)

    # Create client datasets
    client_datasets = create_client_datasets(args.num_clients, batch_size=args.bs)

    # Create a test dataset for evaluation
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

    clients = [FederatedClient(model, data, args, device) for data in client_datasets]

    for round in tqdm(range(args.rounds), desc="Federated Rounds", ncols=100, colour='green'):
        client_updates = []

        # Set round start weights for all clients
        global_weights = server.get_model_state_dict()
        for client in clients:
            client.round_start_weights = {k: v.cpu().numpy() for k, v in global_weights.items()}

        # Each client trains and sends updates
        for client_id, client in enumerate(clients):
            local_updates = client.local_update(client_id)
            # Quantize
            quantized_updates = client.quantize_updates(local_updates)
            client_updates.append(quantized_updates)

        # Server aggregates updates and updates model
        aggregated_updates = server.receive_updates(client_updates)
        server.update_model(aggregated_updates)

        # Update each client's local model with the aggregated updates
        for client in clients:
            client.update_local_model(aggregated_updates)

        # Compute size of updates
        update_size = sum([w.nbytes for update in client_updates for w in update.values()])

        # Evaluate the model
        accuracy = server.evaluate(test_loader)

        if args.use_wandb:
            wandb.log({
                "round": round + 1,
                "update_size": update_size / (1024 * 1024),
                "test_accuracy": accuracy
            })

        tqdm.write(f"Round {round + 1}")
        tqdm.write(f"Update size: {update_size / (1024 * 1024):.2f} MB")
        tqdm.write(f"Test Accuracy: {accuracy:.2f}%")
        tqdm.write("")