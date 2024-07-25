import json
import os
import shutil


def create_text_file(line: dict, new_file_name: str) -> None:
    """Takes in our annotations and outputs in yolo format.
    Yolo format is where each image has a text file with
    each line corresponding to a different object in the image.
    The format is:
    <object-class> <cx> <cy> <width> <height>
    where cx, cy, width, and height are relative to the image size.
    """
    txt = ""
    for box in line["bboxes"]:
        if len(box) == 0:
            continue
        cx, cy, width, height = box
        txt += f"{0} {cx} {cy} {width} {height}\n"
    with open(new_file_name, "w") as f:
        f.write(txt)


def create_yaml_file(
    file_path: str, splits: list[str], class_mapping: dict[int, str]
) -> None:
    """Writes a yolo yaml file for the dataset.


    Args:
        file_path: The path to write the yaml file to.
        splits: The splits of the data. Ex. ["train", "valid"]
            These correspond to the directories in the dataset.
        class_mapping: The mapping of class index to class name.
            Ex. {0: "drone", 1: "person"}
    """
    classes = [class_mapping[i] for i in range(len(class_mapping))]
    with open(file_path, "w") as f:
        for split in splits:
            f.write(f"{split}: {split}/images\n")
        f.write(f"\nnc: {len(class_mapping)}\n")
        f.write("names: " + str(classes))


def make_new_directory(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)


def convert_to_yolo(
    current_annotation_path: str,
    new_dataset_dir: str,
) -> None:
    """This function takes in a flat directory of x amount of images with
    an associated jsonl file. The jsonl file will have two keys: "image_path"
    and "bboxes". The image_path is the path to the image within the diretory
    and the bboxes are cx, cy, width, and height of all bounding boxes in the image.

    The function will output a yolo format dataset with the following structure:
    - split
        - images
            - image1.jpg
            - image2.jpg
        - labels
            - image1.txt
            - image2.txt

    Args:
        current_annotation_path: The path to the current annotation file.
        new_dataset_dir: The path to the new directory where we will have
            two subdirectories images and labels. We assume that the last
            part of the path is the split name.
        class_mapping: The mapping of class index to class name.
            Ex. {0: "drone", 1: "person"}
    """
    split = new_dataset_dir.split("/")[-1]
    assert split in ["train", "valid", "test"], "The last part of the path should be the split name."
    make_new_directory(new_dataset_dir)
    image_dir = os.path.join(new_dataset_dir, "images")
    label_dir = os.path.join(new_dataset_dir, "labels")

    if os.path.exists(image_dir):
        os.system(f"rm -r {image_dir}")
    if os.path.exists(label_dir):
        os.system(f"rm -r {label_dir}")

    make_new_directory(image_dir)
    make_new_directory(label_dir)

    with open(current_annotation_path, "r") as f:
        lines = [json.loads(line) for line in f.readlines()]

    for line in lines:
        # create the label file
        image_path = line["image_path"]
        file_name = os.path.basename(image_path).split(".")[0]
        new_file_name = f"{file_name}.txt"
        new_file_directory = os.path.join(label_dir, new_file_name)
        create_text_file(line, new_file_directory)

        # now move the image to the new directory
        image_path = line["image_path"]
        file_name = os.path.basename(image_path)
        new_file_directory = os.path.join(image_dir, file_name)
        shutil.move(image_path, new_file_directory)


if __name__ == "__main__":
    """
    Inputs should be:
    image_directory
    annotation_file
    new_image_directory
    new_anntoation_directory
    """
    convert_to_yolo(
        current_annotation_path="Drone-detection-github/_train_annotation_custom.jsonl",
        new_dataset_dir="roboflow_format/train",
    )
    create_yaml_file(
        file_path="roboflow_format/data.yaml",
        splits=["train"],
        class_mapping={0: "drone"},
    )
