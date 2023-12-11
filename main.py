#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 20:32:55 2023

@author: wcchun
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from PIL import ImageEnhance
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob

# Define the dimensions for a 16:9 aspect ratio image
aspect_ratio = 16 / 9
width = 224
height = int(width / aspect_ratio)

# Check if a GPU is available and set PyTorch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the FusionModel with convolutional layers
class FusionModel(nn.Module):
    def __init__(self, text_feature_size, output_size):
        super(FusionModel, self).__init__()
        # Convolutional layers for image features
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Flatten the output for the fully connected layer
            nn.Flatten()
        )
        # Calculate the flattened size by passing a dummy input through the conv layers
        dummy_input = torch.zeros(1, 3, height, width)
        self.flattened_size = self.conv_layers(dummy_input).numel()
        
        # Fully connected layers for generating output
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size + text_feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_size),
            nn.Tanh()
        )

    def forward(self, text_features, image_features):
        # image_features should be of shape (B, C, H, W)
        image_out = self.conv_layers(image_features)
        # Flatten the output from the convolutional layers
        image_out = image_out.view(-1, self.flattened_size)
        combined_features = torch.cat((image_out, text_features), dim=1)
        output = self.fc_layers(combined_features)
        return output

# Custom Dataset class
class TextImageDataset(Dataset):
    def __init__(self, text_features, image_folder, transform=None):
        self.text_features = text_features
        self.image_paths = sorted(glob(os.path.join(image_folder, '*.jpg')))
        self.transform = transform
        self.match_length()

    def __len__(self):
        return len(self.text_features)

    def __getitem__(self, idx):
        text_feature = self.text_features[idx]
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return torch.tensor(text_feature, dtype=torch.float32), image

    def match_length(self):
        min_length = min(len(self.text_features), len(self.image_paths))
        self.text_features = self.text_features[:min_length]
        self.image_paths = self.image_paths[:min_length]

# Load and preprocess text data
text_data_path = '/Users/wcchun/cityu/SM3750 Machine Learning for Artists/assignment_2/outcome/cumulative_text_database.txt'
with open(text_data_path, 'r') as file:
    texts = file.readlines()

vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
text_features = vectorizer.fit_transform(texts).toarray()

# Image transformations with augmentation
transform = transforms.Compose([
    transforms.Resize((height, width)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Image directory path
image_folder_path = '/Users/wcchun/cityu/SM3750 Machine Learning for Artists/assignment_2/images/'

# Instantiate the dataset and dataloader
dataset = TextImageDataset(text_features, image_folder_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Define the output size of the model (flattened image tensor size)
output_size = 3 * height * width  # for RGB images

# Instantiate model, loss function, and optimizer
model = FusionModel(text_feature_size=100, output_size=output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the number of epochs for training
num_epochs = 50  # Change this to the number of epochs you want to train for

# Move the model to the specified device
model = model.to(device)

# Training loop ...
for epoch in range(num_epochs):
    for text_features, images in dataloader:
        # Move data to the device
        text_features = text_features.to(device)
        images = images.to(device)
        text_features = text_features.to(device)

        optimizer.zero_grad()
        # Forward pass, now images are correctly shaped
        outputs = model(text_features, images)
        # Ensure the target is also shaped correctly
        loss = criterion(outputs, images.view(images.size(0), -1))
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Save the model after training
torch.save(model.state_dict(), '/Users/wcchun/cityu/SM3750 Machine Learning for Artists/assignment_2/outcome/fusion_model.pth')

# Directory for saving generated images
save_image_path = '/Users/wcchun/cityu/SM3750 Machine Learning for Artists/assignment_2/outcome/'
if not os.path.exists(save_image_path):
    os.makedirs(save_image_path)

# Function to convert tensor to PIL image and enhance brightness
def tensor_to_pil_and_enhance(image_tensor, enhancement_factor=3):  # adjust enhancement_factor as needed
    image_tensor = image_tensor.view(3, height, width)
    image_tensor = torch.clamp(image_tensor, 0, 1)  # Assuming image_tensor is normalized to [0, 1]
    image_pil = transforms.ToPILImage()(image_tensor.cpu())
    enhancer = ImageEnhance.Brightness(image_pil)
    enhanced_image_pil = enhancer.enhance(enhancement_factor)
    return enhanced_image_pil

# Generate and save images
model.eval()
for i, (text_features, _) in enumerate(dataset):
    with torch.no_grad():
        text_features = text_features.unsqueeze(0)
        # Generate random image tensor
        random_image_tensor = torch.randn(1, 3, height, width)
        # Get model output
        generated_image_tensor = model(text_features, random_image_tensor)
        # Convert to PIL image and enhance brightness
        enhanced_image = tensor_to_pil_and_enhance(generated_image_tensor, enhancement_factor=1.5)  # Adjust the factor as needed
        # Save image
        file_name = f'generated_image_{i}.png'
        enhanced_image.save(os.path.join(save_image_path, file_name))