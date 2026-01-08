import torch
from torch import nn

class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()
        # 3 couches de convolution pour extraire les formes (traits, courbes, boucles)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # Entrée: 1 canal (noir & blanc)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        
        self.dropout = nn.Dropout(0.5) # Pour éviter le par coeur (overfitting)
        self.fc1 = nn.Linear(128, 10)  # Sortie: 10 chiffres (0-9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Passage dans les couches avec ReLU (activation) et MaxPool (réduction taille)
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        
        x = torch.flatten(x, 1) # On aplatit tout pour la fin
        x = self.dropout(x)
        return self.fc1(x)

if __name__ == "__main__":
    # Petit test pour vérifier que ça ne plante pas
    model = MyAwesomeModel()
    print(f"Architecture du modèle: {model}")
    dummy_input = torch.randn(1, 1, 28, 28)
    print(f"Sortie test: {model(dummy_input).shape}") # Doit afficher [1, 10]