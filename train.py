import numpy as np
import pandas as pd
import os
import torch
from torch import nn
import albumentations as A
import cv2
from tqdm import tqdm
import random
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

torch.cuda.empty_cache()

BS = 2
LR = 0.0005
epochs = 20
IS = 256
D = 'cuda' if torch.cuda.is_available() else 'cpu'

num_classes = 7
classes = ['elbow positive', 'fingers positive', 'forearm fracture', 'humerus fracture', 'humerus', 'shoulder fracture', 'wrist positive']
c2l = {k: v for k, v in list(zip(classes, list(range(num_classes))))}
l2c = {v: k for k, v in c2l.items()}

dir_path = '/content/'
train_dir_path = '/content/train'
train_img_paths = sorted(os.listdir('/content/train/images'))
train_target_paths = sorted(os.listdir('/content/train/labels'))

val_dir_path = '/content/valid'
val_img_paths = sorted(os.listdir('/content/valid/images'))
val_target_paths = sorted(os.listdir('/content/valid/labels'))

def unconvert(width, height, x, y, w, h):
    xmax = int((x*width) + (w * width)/2.0)
    xmin = int((x*width) - (w * width)/2.0)
    ymax = int((y*height) + (h * height)/2.0)
    ymin = int((y*height) - (h * height)/2.0)
    return xmin, ymin, xmax, ymax

# Define augmentation
augs = A.Compose([
    A.Resize(IS, IS),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']), is_check_shapes=True)

# Define dataset class
class FractureData(torch.utils.data.Dataset):
    
    def __init__(self, dir_path, img_paths, target_paths, augs=None):
        self.dir_path = dir_path
        self.img_paths = img_paths
        self.target_paths = target_paths
        self.augs = augs
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        ip = os.path.join(self.dir_path, 'images', self.img_paths[idx])
        tp = os.path.join(self.dir_path, 'labels', self.target_paths[idx])
        
        image = cv2.imread(ip)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W, _ = image.shape
        
        file = open(tp, 'r')
        target = list(map(float, file.read().split()))
        
        try:
            label = [target.pop(0)]
            bbox = []    
            i = 0
            while i < len(target):
                x, y, w, h = target[i:i+4]
                bbox.append([*unconvert(W, H, x, y, w, h)])
                i += 4
            label = label * len(bbox)
        
            if self.augs is not None:
                data = self.augs(image=image, bboxes=bbox, class_labels=['None'] * len(label))
                image = data['image']
                bbox = data['bboxes']
        except:
            if idx + 1 < len(self.img_paths):
                return self.__getitem__(idx + 1)
            else:
                return self.__getitem__(0)
            
        image = torch.Tensor(np.transpose(image, (2, 0, 1))) / 255.0
        bbox = torch.Tensor(bbox).long()
        label = torch.Tensor(label).long()
        
        annot = {'boxes': bbox, 'labels': label}
        
        return image, annot

# Function to collate data in DataLoader
def collate_fn(batch):
    return tuple(zip(*batch))

# DataLoader for training and validation sets
trainset = FractureData(train_dir_path, train_img_paths, train_target_paths, augs)
valset = FractureData(val_dir_path, val_img_paths, val_target_paths, augs)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS, collate_fn=collate_fn)
valloader = torch.utils.data.DataLoader(valset, batch_size=BS, collate_fn=collate_fn)

print(f'Training Data: {len(trainset)} images divided into {len(trainloader)} batches')
print(f'Validation Data: {len(valset)} images divided into {len(valloader)} batches')

# Function to compute Intersection over Union (IoU)
def compute_iou(pred_box, true_box):
    x1 = max(pred_box[0], true_box[0])
    y1 = max(pred_box[1], true_box[1])
    x2 = min(pred_box[2], true_box[2])
    y2 = min(pred_box[3], true_box[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    true_area = (true_box[2] - true_box[0]) * (true_box[3] - true_box[1])
    union = pred_area + true_area - intersection

    iou = intersection / union if union > 0 else 0
    return iou

# Function to calculate accuracy
def calculate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            images = [image.to(D) for image in images]
            targets = [{k: v.to(D) for k, v in ele.items()} for ele in targets]

            outputs = model(images)
            
            for output, target in zip(outputs, targets):
                pred_boxes = output['boxes'].cpu().numpy()
                true_boxes = target['boxes'].cpu().numpy()

                if len(pred_boxes) == 0 or len(true_boxes) == 0:
                    continue
                
                for pred_box, true_box in zip(pred_boxes, true_boxes):
                    iou = compute_iou(pred_box, true_box)
                    if iou > 0.5:  # Adjust the threshold as needed
                        correct += 1
                    total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy

# Model setup
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.to(D)

best_val_loss = np.Inf
opt = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    
    for (i,(images, targets)) in enumerate(trainloader):
        images = [image.to(D) for image in images]
        targets = [{k: v.to(D) for k, v in t.items()} for t in targets]

        opt.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        
        loss.backward()
        opt.step()

        train_loss += loss

    train_accuracy = calculate_accuracy(model, trainloader)
    print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss / len(trainloader)}, Train Accuracy: {train_accuracy}")

    model.train()
    val_loss = 0.0
    
    for images, targets in tqdm(valloader):
        images = [image.to(D) for image in images]
        targets = [{k: v.to(D) for k, v in t.items()} for t in targets]
        loss_dict=model(images,targets)
        loss = sum(loss for loss in loss_dict.values())
        val_loss += loss

    val_accuracy = calculate_accuracy(model, valloader)
    print(f"Epoch {epoch + 1}/{epochs}: Validation Loss: {val_loss / len(valloader)}, Validation Accuracy: {val_accuracy}")

    if val_loss < best_val_loss:
        torch.save(model.state_dict(), 'bestmodel.pt')
        print("Model Updated")
        best_val_loss = val_loss

torch.save(model.state_dict(), 'FullyTrainedModel.pt')
print("Fully Trained Model Saved")
print(f"Done. Best Validation Loss: {best_val_loss}")
