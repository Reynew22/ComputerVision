import torch
import cv2
import torchvision.transforms as T
from faceMaskDataset import FaceMaskDataset

# Paths
model_path = "path_to_model"

#trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_path)
model.to(device)
model.eval()

real_time_transforms = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
])


# real-time
def real_time_detection(model, transforms, device):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    print("Press 'q' to exit the webcam view.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from the camera.")
            break

        # preprocess
        original_frame = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transforms(Image.fromarray(frame)).unsqueeze(0).to(device)
        outputs = model(frame)

        # predictions
        for box, label, score in zip(outputs[0]['boxes'], outputs[0]['labels'], outputs[0]['scores']):
            if score > 0.5:  # Only display confident predictions
                x1, y1, x2, y2 = box.int().tolist()
                cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(original_frame, f"Label: {label}, Score: {score:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # display the frame
        cv2.imshow("Real-Time Face Mask Detection", original_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

real_time_detection(model, real_time_transforms, device)
