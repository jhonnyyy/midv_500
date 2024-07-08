import json 
import os 
import torch 
import torch.nn as nn 
import torch.optim as optim 

import config as cf
import data as dt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image 

#downloading the dataset
dt.download_data(cf.dataset_dir)

#Loading the images and the corner cordinates 
print("Loading the dataset")
image_paths,corner_coords=dt.load_data(os.path.join(cf.dataset_dir,"midv500"))


# Custom Dataset
class DocumentDataset(Dataset):
    def __init__(self, image_paths, corner_coords, transform=None):
        self.image_paths = image_paths
        self.corner_coords = corner_coords
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        coords = torch.tensor(self.corner_coords[idx], dtype=torch.float32).view(-1)
        return image, coords
    
#######################################
#Preprocessing and transforms 
print("Transforming the dataset")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=cf.brightness, contrast=cf.contrast, saturation=cf.saturation, hue=cf.hue),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Split the data
train_paths, test_paths, train_coords, test_coords = train_test_split(image_paths, corner_coords, test_size=cf.test_size, random_state=42)
print("Train Test Splitting of data")
#######################################
# Create datasets and dataloaders
train_dataset = DocumentDataset(train_paths, train_coords, transform=transform)
test_dataset = DocumentDataset(test_paths, test_coords, transform=transform)


train_loader = DataLoader(train_dataset, batch_size=cf.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=cf.batch_size, shuffle=False)

#######################################
# Model Definitation 
class DocumentCornerModel(nn.Module):
    def __init__(self):
        super(DocumentCornerModel, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 8)  # 8 coordinates (x1, y1, x2, y2, x3, y3, x4, y4)

    def forward(self, x):
        return self.backbone(x)
    

#######################################

model = DocumentCornerModel()

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=cf.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

#######################################
print("Starting the Training process")
num_epochs=cf.num_epochs

for epoch in range(num_epochs):
    print(epoch)
    model.train()
    running_loss = 0.0
    for images, coords in train_loader:
        images, coords = images, coords
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, coords)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Step the scheduler
    scheduler.step()

print("Training completed")

######################################

# Evaluate the Model
model.eval()
test_loss = 0.0
with torch.no_grad():
    for images, coords in test_loader:
        images, coords = images, coords
        outputs = model(images)
        loss = criterion(outputs, coords)
        test_loss += loss.item() * images.size(0)

    test_loss /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}")

# Save the Model
torch.save(model.state_dict(), cf.output_document_path)
print("Model Saved successfully")