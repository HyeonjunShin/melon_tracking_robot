import torch
from torchvision.io import read_image, ImageReadMode
from torchvision import tv_tensors
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2
import numpy as np
import os
import glob


def color2bbox_filename(color_filename):
    return color_filename.replace("rgb_", "bounding_box_2d_tight_").replace(
        ".png", ".npy"
    )


def get_data(dir_path="/home/second/melon_set720v2"):
    color_dir = os.path.join(dir_path, "rgb")
    color_files = glob.glob(os.path.join(color_dir, "*.png"))
    print(f"Fonud the number of {len(color_files)} color files.")

    bbox_dir = os.path.join(dir_path, "bbox_2d_tight")
    matched = 0
    ret = []
    for color_file in color_files:
        target_file = color2bbox_filename(os.path.basename(color_file))
        target_file = os.path.join(bbox_dir, target_file)
        if os.path.exists(target_file):
            matched += 1
            ret.append((color_file, target_file))
    print(f"Found the number of {matched} matched label files.")

    if len(color_files) != matched:
        raise Exception("Mismatched the number of color and label files.")

    return ret


def split_data(paired_path, ratio=0.8):
    from sklearn.model_selection import train_test_split

    train_path, val_path = train_test_split(
        paired_path, train_size=ratio, random_state=42, shuffle=True
    )
    return train_path, val_path


def chamae_collate_fn(batch):
    images = []
    targets = []
    for img, tgs in batch:
        images.append(img)
        targets.append(tgs)
    return torch.stack(images, dim=0), targets


class ChamaeDataset(Dataset):
    def __init__(
        self,
        path_list,
        is_train=True,
        height=720,
        width=1280,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.path_list = path_list
        self.is_train = is_train

        if is_train:
            self.transform = v2.Compose(
                [
                    v2.Resize(size=(self.height // 2, self.width // 2)),
                    v2.Pad(padding=[0, 12, 0, 12], fill=0),
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    v2.RandomApply(
                        [v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))], p=0.5
                    ),
                    v2.RandomRotation(degrees=(-30, 30)),
                    v2.ToDtype(torch.float32, scale=True),
                    # v2.RandomErasing(
                    # p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0
                    # ),
                ]
            )
        else:
            self.transform = v2.Compose(
                [
                    v2.Resize(size=(self.height // 2, self.width // 2)),
                    v2.Pad(padding=[0, 12, 0, 12], fill=0),
                    v2.ToDtype(torch.float32, scale=True),
                ]
            )

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        color_path, target_path = self.path_list[index]
        color = read_image(color_path, ImageReadMode.RGB)

        raw_data = np.load(target_path)
        bboxes = [
            raw_data["x_min"],
            raw_data["y_min"],
            raw_data["x_max"],
            raw_data["y_max"],
        ]
        cls_ids = raw_data["semanticId"].astype(np.float32)
        bboxes = np.stack(bboxes, axis=1)
        bboxes = torch.from_numpy(bboxes).float()
        bboxes = tv_tensors.BoundingBoxes(
            bboxes, format="XYXY", canvas_size=(self.height, self.width)
        )

        # color, bboxes = self.transform((color, bboxes))
        transformed = self.transform({"image": color, "boxes": bboxes})
        color = transformed["image"]
        bboxes = transformed["boxes"]

        if bboxes.numel() > 0:
            target_labels = torch.as_tensor(
                cls_ids, dtype=torch.float32, device=bboxes.device
            ).unsqueeze(1)
            final_bboxes = torch.cat([bboxes, target_labels], dim=1)
        else:
            final_bboxes = torch.zeros((0, 5), dtype=torch.float32)

        return color, final_bboxes


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from torchvision.utils import draw_bounding_boxes
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    paired_path = get_data()
    train_path, val_path = split_data(paired_path)

    train_dataset = ChamaeDataset(train_path, True)
    val_dataset = ChamaeDataset(val_path, False)

    train_loader = DataLoader(
        train_dataset,
        32,
        True,
        collate_fn=chamae_collate_fn,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        32,
        False,
        collate_fn=chamae_collate_fn,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    # for batch in train_loader:
    #     color, target = batch

    #     print(color.shape)

    # vis = draw_bounding_boxes(color[0], target[0][:, :4], colors="red", width=3)
    # pil_img = v2.ToPILImage()(vis)
    # print(color.shape, resized_color.shape)
    # print(pil_img.size)
    # plt.imshow(pil_img)
    # plt.show()
    # print(data[1])

    for batch in val_loader:
        color, target = batch
        print(color.shape)

    # vis = draw_bounding_boxes(color[0], target[0][:, :4], colors="red", width=3)
    # pil_img = v2.ToPILImage()(vis)
    # print(color.shape, resized_color.shape)
    # print(pil_img.size)
    # plt.imshow(pil_img)
    # plt.show()
    # print(data[1])
