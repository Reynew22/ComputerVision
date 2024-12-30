import torch
from faceMaskDataset import FaceMaskDataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
from sklearn.metrics import precision_recall_fscore_support


val_images_dir = "data\\face_mask_dataset\\images\\val"
val_labels_dir = "data\\face_mask_dataset\\labels\\val"
model_path = "path_to_model"

# for validation
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

@torch.no_grad()
def evaluate_metrics(model, loader):
    all_true_labels = []
    all_pred_labels = []
    all_scores = []

    for images, targets in loader:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)

        for output, target in zip(outputs, targets):
            true_labels = target['labels'].cpu().numpy()
            pred_labels = output['labels'].cpu().numpy()
            scores = output['scores'].cpu().numpy()

            all_true_labels.extend(true_labels)
            all_pred_labels.extend(pred_labels)
            all_scores.extend(scores)

    # precision recall f1 score
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        all_true_labels, all_pred_labels, average='weighted'
    )

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

    mAP = sum(all_scores) / len(all_scores)
    print(f"mAP: {mAP:.4f}")

evaluate_metrics(model, val_loader)
