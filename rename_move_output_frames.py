import json
import os
import shutil
from collections import defaultdict


def count_jsonl_lines(file_path: str) -> int:
    with open(file_path, "r") as f:
        return len(f.readlines())


def list_all_files(directory: str) -> None:
    """
    Check to make sure the number of files in the directory matches
    the number of lines in the jsonl file
    """
    possible = ["test", "train", "valid"]
    for p in possible:
        if not os.path.exists(os.path.join(directory, p)):
            continue

        jsonl_dir = os.path.join(directory, f"_{p}_annotation_custom.jsonl")
        num_lines = count_jsonl_lines(jsonl_dir)
        p_path = os.path.join(directory, p)
        num_files = len(
            [f for f in os.listdir(p_path) if f.endswith((".png", ".jpg", ".jpeg"))]
        )
        print(f"For {p}, files: {num_files}, jsonl lines: {num_lines}")


def move_and_rename_images(root_dir: str, new_dir: str) -> None:
    os.makedirs(new_dir, exist_ok=True)

    for subdir, _, files in os.walk(root_dir):
        subdir_name = os.path.basename(subdir)

        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                old_file_path = os.path.join(subdir, file)
                new_file_name = f"{subdir_name}_{file}"
                new_file_path = os.path.join(new_dir, new_file_name)
                shutil.copy(old_file_path, new_file_path)


def translate_jsonl(jsonl_file: str, new_path: str, output_jsonl_file: str) -> None:
    """
    Translate the jsonl file to the new image path
    """
    with open(jsonl_file, "r") as f:
        data = f.readlines()

    new_lines = []
    for i, line in enumerate(data):
        line = json.loads(line)
        old_image_path = line["image_path"]
        file_name = old_image_path.split("/")[-1]
        subdir_name = old_image_path.split("/")[-2]
        new_name = f"{subdir_name}_{file_name}"
        new_image_path = os.path.join(new_path, new_name)

        new_line = {"image_path": new_image_path, "bbox": [line["boxes"]]}
        new_lines.append(json.dumps(new_line) + "\n")

    with open(output_jsonl_file, "w") as f:
        f.writelines(new_lines)
        
def remove_make_new_directory(directory: str) -> None:
    if os.path.exists(directory):
        os.system(f"rm -r {directory}")
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":
    root_directory = "output_frames"
    new_image_directory = "Drone-detection-github/train"
    new_directory = "Drone-detection-github"
    jsonl_file = "output.jsonl"
    output_jsonl_file = "_train_annotation_custom.jsonl"
    
    remove_make_new_directory(new_directory)
    remove_make_new_directory(new_image_directory)

    move_and_rename_images(root_directory, new_image_directory)
    translate_jsonl(jsonl_file, new_image_directory, output_jsonl_file)

    with open(output_jsonl_file, "r") as f:
        all_lines = [json.loads(line) for line in f.readlines()]

    image_path_to_boxes = defaultdict(list)
    for line in all_lines:
        image_path_to_boxes[line["image_path"]].extend(line["bbox"])
    new_lines = []
    for image_path, boxes in image_path_to_boxes.items():
        new_line = {"image_path": image_path, "bboxes": boxes}
        new_lines.append(json.dumps(new_line) + "\n")

    with open(os.path.join(new_directory, output_jsonl_file), "w") as f:
        for line in new_lines:
            f.write(line)

    list_all_files(new_directory)
