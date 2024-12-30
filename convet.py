import os
import shutil
import random
import xml.etree.ElementTree as ET
from PIL import Image

archive_images = "archive/images"
archive_annotations = "archive/annotations"
output_dataset = "face_mask_dataset"

os.makedirs(f"{output_dataset}/images/train", exist_ok=True)
os.makedirs(f"{output_dataset}/images/val", exist_ok=True)
os.makedirs(f"{output_dataset}/labels/train", exist_ok=True)
os.makedirs(f"{output_dataset}/labels/val", exist_ok=True)

# split ratio
train_ratio = 0.7

images = sorted(os.listdir(archive_images))
random.shuffle(images)


# VOC XML to YOLO
def convert_voc_to_yolo(annotation_file, img_width, img_height):
    yolo_annotations = []
    tree = ET.parse(annotation_file)
    root = tree.getroot()

    for obj in root.findall("object"):
        label = obj.find("name").text
        bndbox = obj.find("bndbox")

        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        # convert to YOLO
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        label_mapping = {"without_mask": 0, "mask_weared_incorrect": 1, "with_mask": 2}
        if label not in label_mapping:
            print(f"Unknown label '{label}' in {annotation_file}. Skipping.")
            continue

        class_id = label_mapping[label]
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return yolo_annotations

for i, image_file in enumerate(images):
    base_name = os.path.splitext(image_file)[0]
    annotation_file = os.path.join(archive_annotations, f"{base_name}.xml")
    image_file_path = os.path.join(archive_images, image_file)

    if not os.path.exists(annotation_file):
        print(f"Annotation for {image_file} not found. Skipping.")
        continue

    split = "train" if i < len(images) * train_ratio else "val"

    # copy image to new dataset folder
    output_image_path = os.path.join(output_dataset, f"images/{split}", image_file)
    shutil.copy(image_file_path, output_image_path)

    # read image
    with Image.open(image_file_path) as img:
        img_width, img_height = img.size

    # convert annotatio
    yolo_annotations = convert_voc_to_yolo(annotation_file, img_width, img_height)

    # save YOLO
    label_output_path = os.path.join(output_dataset, f"labels/{split}", f"{base_name}.txt")
    with open(label_output_path, "w") as f:
        f.write("\n".join(yolo_annotations))

print("Conversion complete. Dataset is ready in 'face_mask_dataset' folder.")
