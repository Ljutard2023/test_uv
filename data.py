import torch
from torch.utils.data import TensorDataset

DATA_PATH = "data/corruptmnist"

def corrupt_mnist():
    """Charge et fusionne les données corrompues."""
    train_images, train_target = [], []
    
    # 1. On charge les 6 morceaux de l'entraînement
    for i in range(6):
        train_images.append(torch.load(f"{DATA_PATH}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{DATA_PATH}/train_target_{i}.pt"))
    
    # 2. On colle tout ensemble (concaténation)
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    # 3. On charge le test
    test_images = torch.load(f"{DATA_PATH}/test_images.pt")
    test_target = torch.load(f"{DATA_PATH}/test_target.pt")

    # 4. TRÈS IMPORTANT : On ajoute la dimension "Channel" (Canal)
    # Les CNN veulent du [N, C, H, W] -> [Nombre, 1, 28, 28]
    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    
    train_target = train_target.long()
    test_target = test_target.long()

    # 5. On crée les Datasets
    train_set = TensorDataset(train_images, train_target)
    test_set = TensorDataset(test_images, test_target)

    return train_set, test_set

if __name__ == "__main__":
    train, test = corrupt_mnist()
    print(f"Taille Train: {len(train)}") # Devrait être 30 000 ou plus
    print(f"Taille Test: {len(test)}")   # Devrait être 5 000