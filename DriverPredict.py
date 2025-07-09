from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn

import torch
import torch.nn as nn

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
            nn.Dropout(p=0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)




# Step 1: Load your model architecture and weights
model = BinaryCNN(img_size=128)
checkpoint = torch.load("best_model.pth", map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Step 2: Define the same transform used during training
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Step 3: Load and preprocess your image
img_path = "/Users/nathanaelseay/Documents/DL/training_data/clean/DL_clean_image21.jpg"
image = Image.open(img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)  # shape: [1, 3, 128, 128]

# Step 4: Run inference
with torch.no_grad():
    output = model(input_tensor)
    prob = torch.sigmoid(output)
    prediction = (prob > 0.5).item()

# Step 5: Display result
print(f"Prediction: {prediction}  (probability: {prob.item():.4f})")

if prediction==False:
    print('The image is clean')

else:
    print('The image is NOT clean')