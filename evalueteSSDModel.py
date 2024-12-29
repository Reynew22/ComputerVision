import torch
from faceMaskDataset import FaceMaskDataset
from torch.utils.data import DataLoader
import torchvision.transforms as T


val_images_dir = "C:\\Users\\METE\\Desktop\\face_mask_dataset\\images\\val"
val_labels_dir = "C:\\Users\\METE\\Desktop\\face_mask_dataset\\labels\\val"
model_path = "path_to_model"

#or validation
val_transforms = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
])

# validation dataset
val_dataset = FaceMaskDataset(val_images_dir, val_labels_dir, transforms=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

# trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_path)
model.to(device)
model.eval()

# IoU
def calculate_iou(pred_boxes, true_boxes):

    if len(pred_boxes) == 0 or len(true_boxes) == 0:
        return 0.0

    # intersection
    inter_x_min = torch.max(pred_boxes[:, None, 0], true_boxes[:, 0])
    inter_y_min = torch.max(pred_boxes[:, None, 1], true_boxes[:, 1])
    inter_x_max = torch.min(pred_boxes[:, None, 2], true_boxes[:, 2])
    inter_y_max = torch.min(pred_boxes[:, None, 3], true_boxes[:, 3])

    inter_width = (inter_x_max - inter_x_min).clamp(min=0)
    inter_height = (inter_y_max - inter_y_min).clamp(min=0)
    intersection = inter_width * inter_height

    # union
    pred_areas = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    true_areas = (true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])
    union = pred_areas[:, None] + true_areas - intersection

    iou = intersection / union
    avg_iou = iou.mean().item()
    return avg_iou

# evaluate
@torch.no_grad()
def evaluate(model, loader):

    total_iou = 0
    num_samples = 0

    for images, targets in loader:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)

        # IoU
        for output, target in zip(outputs, targets):
            iou = calculate_iou(output['boxes'], target['boxes'])
            total_iou += iou
            num_samples += 1

    avg_iou = total_iou / num_samples
    print(f"Average IoU: {avg_iou:.4f}")

evaluate(model, val_loader)
