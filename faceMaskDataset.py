import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class FaceMaskDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transforms=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms

        self.images = sorted(os.listdir(images_dir))
        self.labels = sorted(os.listdir(labels_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        img = Image.open(img_path).convert("RGB")

        label_path = os.path.join(self.labels_dir, self.labels[idx])
        boxes, labels = self.parse_yolo_label(label_path, img.size)

        if self.transforms:
            img = self.transforms(img)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }
        return img, target

    def parse_yolo_label(self, label_path, img_size):
        boxes = []
        labels = []
        with open(label_path, "r") as file:
            for line in file.readlines():
                cls, x_center, y_center, width, height = map(float, line.split())
                x_min = (x_center - width / 2) * img_size[0]
                y_min = (y_center - height / 2) * img_size[1]
                x_max = (x_center + width / 2) * img_size[0]
                y_max = (y_center + height / 2) * img_size[1]
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(int(cls))
        return boxes, labels

train_transforms = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_images_dir = r"C:\Users\METE\Desktop\face_mask_dataset\images\train"
train_labels_dir = r"C:\Users\METE\Desktop\face_mask_dataset\labels\train"
val_images_dir = r"C:\Users\METE\Desktop\face_mask_dataset\images\val"
val_labels_dir = r"C:\Users\METE\Desktop\face_mask_dataset\labels\val"

train_dataset = FaceMaskDataset(train_images_dir, train_labels_dir, transforms=train_transforms)
val_dataset = FaceMaskDataset(val_images_dir, val_labels_dir, transforms=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))


# visual
def visualize_batch(images, targets):
    fig, ax = plt.subplots(1, len(images), figsize=(20, 5))
    for i, (img, target) in enumerate(zip(images, targets)):
        img = img.permute(1, 2, 0).numpy()
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # Denormalize
        img = img.clip(0, 1)

        ax[i].imshow(img)
        for box in target["boxes"]:
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     linewidth=2, edgecolor="red", facecolor="none")
            ax[i].add_patch(rect)
        ax[i].axis("off")
    plt.show()


# visual
batch = next(iter(train_loader))
visualize_batch(batch[0], batch[1])
