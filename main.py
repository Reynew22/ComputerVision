import os
import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms, models
import xml.etree.ElementTree as ET
import numpy as np
import cv2

# dataset class
class MaskDataset(torch.utils.data.Dataset):
    def __init__(self, images_folder, annotations_folder, transforms=None):
        self.images_folder = images_folder
        self.annotations_folder = annotations_folder
        self.transforms = transforms
        self.imgs = sorted(os.listdir(images_folder))
        self.annotations = sorted(os.listdir(annotations_folder))
        self.classes = {"with_mask": 2, "without_mask": 1, "mask_weared_incorrect": 3}

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_folder, self.imgs[idx])
        annotation_path = os.path.join(self.annotations_folder, self.annotations[idx])
        img = Image.open(img_path).convert("RGB")

        # annotations
        tree = ET.parse(annotation_path)
        boxes = []
        labels = []
        for obj in tree.findall("object"):
            name = obj.find("name").text
            if name not in self.classes:
                print(f"Unexpected label found: {name}")
                continue
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.classes[name])

        # convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        # Apply transformations
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

# for images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


images_folder = r"C:\Users\METE\Desktop\archive\images"
annotations_folder = r"C:\Users\METE\Desktop\archive\annotations"
dataset = MaskDataset(images_folder, annotations_folder, transforms=transform)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# exploration
def explore_dataset(dataset):
    class_counts = {cls: 0 for cls in dataset.classes.values()}
    for i in range(len(dataset)):
        _, target = dataset[i]
        for label in target["labels"]:
            label = label.item()
            if label in class_counts:
                class_counts[label] += 1
            else:
                print(f"Unexpected label in dataset: {label}")

    print("Class Distribution:")
    for key, value in class_counts.items():
        if key in dataset.classes.values():
            class_name = [k for k, v in dataset.classes.items() if v == key][0]
            print(f"{class_name}: {value}")
        else:
            print(f"Unknown class {key}: {value}")

# visual
def visualize_batch(batch, num_images=4):
    classes_dict = {1: "without_mask", 2: "with_mask", 3: "mask_weared_incorrect"}
    rows = 2  # Fixed number of rows
    cols = (num_images + rows - 1) // rows  # adjust columns

    fig = plt.figure(figsize=(16, 9))
    for i in range(num_images):
        ax = fig.add_subplot(rows, cols, i + 1)
        img = batch[0][i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)  # pixel range
        bboxes = batch[1][i]["boxes"]
        labels = batch[1][i]["labels"]

        ax.imshow(img)
        for j in range(len(bboxes)):
            bbox = bboxes[j]
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                      linewidth=2, edgecolor="r", facecolor="none")
            ax.add_patch(rect)
            ax.text(bbox[0], bbox[1] - 10, classes_dict[labels[j].item()], color="white",
                    fontsize=8, bbox=dict(facecolor="red", alpha=0.5))
        ax.axis("off")
    plt.show()

# metrics
def evaluate_model(model, data_loader, device):
    model.eval()
    iou_threshold = 0.5
    total_iou = 0
    total_objects = 0

    with torch.no_grad():
        for imgs, targets in data_loader:
            imgs = [img.to(device) for img in imgs]
            outputs = model(imgs)

            for target, output in zip(targets, outputs):
                gt_boxes = target["boxes"].cpu().numpy()
                pred_boxes = output["boxes"].cpu().numpy()

                for gt_box in gt_boxes:
                    for pred_box in pred_boxes:
                        iou = compute_iou(gt_box, pred_box)
                        if iou > iou_threshold:
                            total_iou += iou
                            total_objects += 1

    print(f"Mean IoU: {total_iou / total_objects if total_objects > 0 else 0:.2f}")

# IoU
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

# Webcam
def real_time_detection(model, transforms, device):
    model.eval()
    cap = cv2.VideoCapture(0)
    classes_dict = {1: "without_mask", 2: "with_mask", 3: "mask_weared_incorrect"}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # preprocess
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = transforms(img).unsqueeze(0).to(device)

        # predictions
        with torch.no_grad():
            outputs = model(img_tensor)

        # boxes and labels
        for box, label in zip(outputs[0]["boxes"], outputs[0]["labels"]):
            box = box.cpu().numpy().astype(int)
            label = label.cpu().item()
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, classes_dict[label], (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Real-Time Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# exploration
explore_dataset(dataset)

# visual batch
sample_batch = next(iter(data_loader))
visualize_batch(sample_batch)

