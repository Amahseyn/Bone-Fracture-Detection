import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import cv2
import numpy as np
import torchvision.transforms as transforms


main_directory = "/content"
train_directory = os.path.join(main_directory, "train")
val_directory = os.path.join(main_directory, "valid")
test_directory = os.path.join(main_directory, "test")
image_size = (256,256)
num_classes = 7
classes = ['elbow positive', 'fingers positive', 'forearm fracture', 'humerus fracture', 'humerus', 'shoulder fracture', 'wrist positive']
learning_rate = 1e-4
epochs = 10
batchsize = 4
def unconvert(width=image_size[0], height=image_size[0], x, y, w, h):
    xmax = int((x * width) + (w * width) / 2.0)
    xmin = int((x * width) - (w * width) / 2.0)
    ymax = int((y * height) + (h * height) / 2.0)
    ymin = int((y * height) - (h * height) / 2.0)
    return xmin, ymin, xmax, ymax

class ReadDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path,image_size):
        self.dir_path = dir_path
        self.images_dir = os.path.join(self.dir_path, "images")
        self.labels_dir = os.path.join(self.dir_path, "labels")
        self.images = sorted(os.listdir(self.images_dir))
        self.labels = sorted(os.listdir(self.labels_dir))
        self.augs = None
        self.image_size = image_size
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        ip = os.path.join(self.dir_path, 'images', self.images[idx])
        tp = os.path.join(self.dir_path, 'labels', self.labels[idx])

        image = cv2.imread(ip)
        image = image.resize(image,self.image_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W, _ = image.shape

        file = open(tp, 'r')
        target = list(map(float, file.read().split()))
        label = [int(target.pop(0))] 
        bbox = []
        i = 0
        while i < len(target):
            x, y, w, h = target[i:i + 4]
            bbox.append([*unconvert(W, H, x, y, w, h)])
            i += 4

        bbox = np.array(bbox, dtype=np.float32)
        label = np.array(label, dtype=np.long)

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        bbox = torch.tensor(bbox, dtype=torch.float32)
        label = torch.tensor(label)
        return image, {"boxes": bbox, "labels": label}

class ObjectDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super(ObjectDetectionModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_cls = nn.Linear(256 * 28 * 28, num_classes)
        self.fc_bbox = nn.Linear(256 * 28 * 28, 4)  

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  
        cls_scores = self.fc_cls(x)
        bbox_preds = self.fc_bbox(x)
        return cls_scores, bbox_preds

def train_model(model, train_loader, criterion_cls, criterion_bbox, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets['labels'] = targets['labels'].to(device)
            targets['boxes'] = targets['boxes'].to(device)

            optimizer.zero_grad()
            
            cls_scores, bbox_preds = model(images)
            loss_cls = criterion_cls(cls_scores, targets['labels'])
            loss_bbox = criterion_bbox(bbox_preds, targets['boxes'])
            loss = loss_cls + loss_bbox

            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

train_dataset = ReadDataset(train_directory,image_size = image_size)
train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ObjectDetectionModel(num_classes=num_classes).to(device)

criterion_cls = nn.CrossEntropyLoss()
criterion_bbox = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_model(model, train_loader, criterion_cls, criterion_bbox, optimizer, device, num_epochs=epochs)
