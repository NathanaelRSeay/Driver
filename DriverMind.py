import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from LoadDriverMind import save_model  # assumes you defined this elsewhere

# Device config (for Mac)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Config
img_size = 128
batch_size = 32
num_epochs = 100
best_val_acc = 0.01

# Transforms
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# Load dataset
train_data = datasets.ImageFolder("/Users/nathanaelseay/Documents/DL/training_data/", transform=transform)
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# -----------------------
# Define the CNN with Dropout and BatchNorm
# -----------------------
class BinaryCNN(nn.Module):
    def __init__(self, img_size=128):
        super(BinaryCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(128 * (img_size // 8) * (img_size // 8), 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Dropout here
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

# -----------------------
# Initialization function (Kaiming)
# -----------------------
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Instantiate model
model = BinaryCNN(img_size).to(device)
model.apply(init_weights)

# Loss, optimizer, scheduler
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training loop
train_acc_history, val_acc_history = [], []

for epoch in range(num_epochs):
    model.train()
    correct, total = 0, 0
    running_loss = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.float().to(device).unsqueeze(1)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(outputs)
        preds = probs > 0.5
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item()

    train_acc = correct / total
    train_acc_history.append(train_acc)

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.float().to(device).unsqueeze(1)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = probs > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total
    val_acc_history.append(val_acc)

    # Scheduler step
    scheduler.step()

    print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, LR={scheduler.get_last_lr()[0]:.6f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_model(model, optimizer, epoch + 1, loss.item(), path="best_model.pth")

# Plot results
plt.plot(train_acc_history, label='Train Acc')
plt.plot(val_acc_history, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
