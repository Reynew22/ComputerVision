import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from ultralytics import YOLO
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchvision.transforms as T
from faceMaskDataset import FaceMaskDataset
from tqdm import tqdm  # Progress bar
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_ssd_model(num_classes):
    # Load a pre-trained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(weights="COCO_V1")
    # Modify the classifier head for the custom dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model.to(device)

def get_yolo_model():
    model = YOLO('yolov5s.pt')
    return model

def train_ssd(model, train_loader, val_loader=None, num_epochs=10, lr=1e-4):
    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for images, targets in progress_bar:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            losses = model(images, targets)
            loss = sum(loss for loss in losses.values())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1}/{num_epochs} completed. Total Loss: {epoch_loss:.4f}")

def train_yolo(yolo_model, train_images_dir, val_images_dir, epochs=10):
    yolo_model.train(
        data=dict(
            train=train_images_dir,
            val=val_images_dir,
            names=["No Mask", "Incorrect Mask", "Correct Mask"]
        ),
        epochs=epochs,
    )

train_images_dir = r"C:\Users\METE\Desktop\face_mask_dataset\images\train"
train_labels_dir = r"C:\Users\METE\Desktop\face_mask_dataset\labels\train"
val_images_dir = r"C:\Users\METE\Desktop\face_mask_dataset\images\val"
val_labels_dir = r"C:\Users\METE\Desktop\face_mask_dataset\labels\val"

train_transforms = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = FaceMaskDataset(train_images_dir, train_labels_dir, transforms=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

if __name__ == "__main__":
    num_classes = 3  #[No Mask, Incorrect Mask, Correct Mask]

    print("Training SSD Model...")
    ssd_model = get_ssd_model(num_classes)
    train_ssd(ssd_model, train_loader, None, num_epochs=10, lr=1e-4)

    print("Training YOLO Model...")
    yolo_model = get_yolo_model()
    train_yolo(yolo_model, train_images_dir, val_images_dir, epochs=10)
