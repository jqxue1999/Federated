import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning with Vision Transformer')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--bs', type=int, default=64, help='batch size')
    parser.add_argument('--rounds', type=int, default=10, help='number of federated rounds')
    parser.add_argument('--num_clients', type=int, default=3, help='number of federated clients')
    parser.add_argument('--local_epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--freeze_layers', type=int, default=12, help='number of layers to freeze in ViT')
    parser.add_argument('--quantize_bits', type=int, default=8, help='number of bits for quantization')
    parser.add_argument('--use_wandb', action='store_true', help='use wandb for logging')
    return parser.parse_args()


def create_client_datasets(num_clients, batch_size=32):
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ViT requires 224x224 images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load CIFAR-10 dataset
    full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Calculate the size of each client's dataset
    base_samples_per_client = len(full_dataset) // num_clients
    remainder = len(full_dataset) % num_clients

    # Distribute samples to clients
    client_dataset_sizes = [base_samples_per_client + (1 if i < remainder else 0) for i in range(num_clients)]

    # Split dataset into num_clients parts
    client_datasets = random_split(full_dataset, client_dataset_sizes)

    # Create DataLoaders for each client
    client_dataloaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in client_datasets]

    return client_dataloaders