import os
import torch
import torchvision.transforms as T
from PIL import Image

class YoloMaskDataset(torch.utils.data.Dataset):
    def __init__(self, images_folder, labels_folder, img_size=512, transforms=None, is_train=True):
        self.images_folder = images_folder
        self.labels_folder = labels_folder
        self.img_size = img_size
        self.transforms = transforms
        self.is_train = is_train

        self.image_paths = sorted(os.listdir(images_folder))
        self.label_paths = sorted(os.listdir(labels_folder))
        self.classes = {0: "with_mask", 1: "without_mask", 2: "mask_weared_incorrect"}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # load image
        img_path = os.path.join(self.images_folder, self.image_paths[idx])
        label_path = os.path.join(self.labels_folder, self.label_paths[idx])
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # nnotations
        boxes = []
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                cls, x_center, y_center, width, height = map(float, line.split())
                x_center *= orig_w
                y_center *= orig_h
                width *= orig_w
                height *= orig_h
                xmin = x_center - width / 2
                ymin = y_center - height / 2
                xmax = x_center + width / 2
                ymax = y_center + height / 2
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(cls))

        # to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        # transformations
        if self.transforms:
            img = self.transforms(img)
        return img, target

train_transforms = T.Compose([
    T.Resize((512, 512)),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = YoloMaskDataset(
    images_folder=r"data//face_mask_dataset//images//train",
    labels_folder=r"data//face_mask_dataset//labels//train",
    transforms=train_transforms
)

val_dataset = YoloMaskDataset(
    images_folder=r"data//face_mask_dataset//images//val",
    labels_folder=r"data//face_mask_dataset//labels//val",
    transforms=val_transforms
)
