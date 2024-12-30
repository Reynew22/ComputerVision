from ultralytics import YOLO


data_yaml_path = "dataset.yaml"
model_save_dir = "YOLO_FaceMaskDetection"
pretrained_model = "yolov5su.pt"


def train_yolo():

    print("Training YOLO Model...")


    yolo_model = YOLO(pretrained_model)

    # train
    yolo_model.train(
        data=data_yaml_path,
        epochs=80,
        batch=8,
        imgsz=512,
        project=model_save_dir,
        name="exp1",
        device="cpu",
        pretrained=True,
        verbose=True,
    )
    print("YOLO model training completed.")


def evaluate_yolo():

    print("Evaluating YOLO Model...")

    # trained YOLO
    yolo_model = YOLO(f"{model_save_dir}/exp1/weights/best.pt")  #

    # evaluate the model
    results = yolo_model.val(data=data_yaml_path, device="cpu")
    print("Evaluation results:")
    print(results)


if __name__ == "__main__":
    train_yolo()
    evaluate_yolo()
