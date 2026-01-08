import torch
import typer
import matplotlib.pyplot as plt
from data import corrupt_mnist
from model import MyAwesomeModel

app = typer.Typer()

# Détection automatique : GPU (cuda), Mac M1/M2 (mps) ou CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@app.command()
def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 5):
    """Entraîne le modèle et sauvegarde les poids."""
    print(f"🚀 Démarrage de l'entraînement sur {DEVICE}...")
    
    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    losses = []

    for epoch in range(epochs):
        model.train()
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()           # Reset gradients
            preds = model(imgs)             # Prédiction
            loss = loss_fn(preds, labels)   # Calcul erreur
            loss.backward()                 # Rétropropagation
            optimizer.step()                # Mise à jour poids
            
            losses.append(loss.item())

        print(f"✅ Epoch {epoch+1}/{epochs} terminée.")

    # Sauvegarde
    torch.save(model.state_dict(), "model.pth")
    print("💾 Modèle sauvegardé dans 'model.pth'")

    # Petit graphique pour frimer
    plt.plot(losses)
    plt.title("Courbe d'apprentissage")
    plt.savefig("training_curve.png")

@app.command()
def evaluate(model_path: str):
    """Charge un modèle et teste sa précision."""
    print(f"🔍 Évaluation du modèle : {model_path}")
    
    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    _, test_set = corrupt_mnist()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32)

    correct = 0
    total = 0

    with torch.no_grad(): # Pas besoin de gradient pour le test (économie mémoire)
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs)
            predicted_class = preds.argmax(dim=1)
            correct += (predicted_class == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"🎯 Précision finale : {accuracy * 100:.2f}%")

if __name__ == "__main__":
    app()