from collections import defaultdict
from typing import Optional

import jsonlines
import torchvision
import tqdm
from groundingdino.util.inference import batch_predict, load_model
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class ImageFolderWithFilenames(datasets.ImageFolder):
    def __getitem__(self, index):
        # Get the original tuple (image, label)
        original_tuple = super().__getitem__(index)

        # Get the image path
        path, _ = self.samples[index]

        # Append the filename to the tuple
        tuple_with_filename = (*original_tuple, path)

        return tuple_with_filename


def create_dataloader(
    data_dir: str,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 4,
    transform: Optional[torchvision.transforms.Compose] = None,
) -> DataLoader:
    """
    Creates a PyTorch DataLoader for images stored in subdirectories.

    Parameters:
    - data_dir (str): Path to the main directory containing subdirectories of images.
    - batch_size (int): Number of samples per batch to load.
    - shuffle (bool): Whether to shuffle the dataset.
    - num_workers (int): How many subprocesses to use for data loading.
    - transform (torchvision.transforms.Compose): Transformations to apply to the images.

    Returns:
    - DataLoader: PyTorch DataLoader.
    """

    # Load the dataset from the directory with subdirectories
    dataset = ImageFolderWithFilenames(root=data_dir, transform=transform)

    # Create the DataLoader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return dataloader

def label_with_dino(
    output_file_path: str,
    model_path: str,
    model_config: str,
    dataloader: DataLoader,
    text_prompt: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    device: str = "cuda",
) -> None:
    """
    Takes in a dataloader of images and labels them with DINO.
    
    The dataloader should return a tuple of (images, labels, filenames).
    
    Args:
        output_file_path (str): The path to write the output JSONL file.
        model_path (str): The path to the DINO model checkpoint.
        model_config (str): The path to the DINO model configuration.
        data_loader (DataLoader): The DataLoader containing the images.
        text_prompt (str): The text prompt to use for DINO.
        box_threshold (float): The threshold for the bounding box.
        text_threshold (float): The threshold for the text.
        device (str): The device to run the model on.
    """
    # Load the model
    model = load_model(
        model_config, model_path,
    ).to(device)

    with jsonlines.open(output_file_path, mode="w") as writer:
        for idx, batch in enumerate(tqdm.tqdm(dataloader)):
            boxes, logits, boxes_to_im = batch_predict(
                model=model,
                preprocessed_images=batch[0],
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=device,
            )
            numbers_seen = set()

            for idx, im_num in enumerate(boxes_to_im):

                # Prepare the data to write to JSONL
                data = {
                    "image_name": batch[2][im_num],
                    "boxes": boxes[idx].tolist(),
                    "logits": logits[
                        idx
                    ].tolist(),  # Convert to list for JSON serialization
                }
                numbers_seen.add(im_num)
                writer.write(data)

            numbers_not_seen = set(range(len(batch[0]))) - numbers_seen
            for num in numbers_not_seen:
                data = {
                    "image_name": batch[2][num],
                    "boxes": [],
                    "logits": [],
                }
                writer.write(data)
    
    


if __name__ == "__main__":
    DEFAULT_TRANSFORM = transforms.Compose(
        [
            transforms.Resize([800, 1000]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    TEXT_PROMPT = "a flying object"
    DEVICE = "CPU"
    
    image_data_dir = "output_frames"
    dataloader = create_dataloader(data_dir=image_data_dir, transform=DEFAULT_TRANSFORM)
    
    model_path = "../GroundingDINO/groundingdino_swint_ogc.pth"
    model_config = "../GroundingDINO/GroundingDINO_SwinT_OGC.py"
    label_with_dino(
        output_file_path="output.jsonl",
        model_path=model_path,
        model_config=model_config,
        dataloader=dataloader,
        text_prompt=TEXT_PROMPT,
        device="cpu",
    )