import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class ImagesDataset(Dataset):
    def __init__(self, image_paths, formulas, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.formulas = formulas

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        formula = self.formulas[index]

        if self.transform:
            image = self.transform(image)

        return image, formula


class Im2LatexDataset:
    def __init__(self, batch_size=32, device=torch.device("cpu"), path_to_data="./"):
        self.path_to_data = path_to_data
        self.batch_size = batch_size
        dfs = (train_df, val_df, test_df) = self.load_dataframes()
        train_dataset, val_dataset, test_dataset = self.load_datasets(*dfs)

        self.train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        self.test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    def load_datasets(
        self, train_df, val_df, test_df
    ) -> tuple[ImagesDataset, ImagesDataset, ImagesDataset]:
        transform = transforms.Compose(
            [
                transforms.Resize((224, 600)),  # Resize to a specific size
                transforms.ToTensor(),  # Convert to tensor
            ]
        )

        train_dataset = ImagesDataset(
            train_df["image"], train_df["formula"], transform=transform
        )
        val_dataset = ImagesDataset(
            val_df["image"], val_df["formula"], transform=transform
        )
        test_dataset = ImagesDataset(
            test_df["image"], test_df["formula"], transform=transform
        )

        train_dataset = ImagesDataset(
            train_df["image"], train_df["formula"], transform=transform
        )
        val_dataset = ImagesDataset(
            val_df["image"], val_df["formula"], transform=transform
        )
        test_dataset = ImagesDataset(
            test_df["image"], test_df["formula"], transform=transform
        )

        return train_dataset, val_dataset, test_dataset

    def load_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_handwritten_df = pd.read_csv(self.path_to_data + "train_handwritten.csv")
        val_handwritten_df = pd.read_csv(self.path_to_data + "val_handwritten.csv")
        train_df = pd.read_csv(self.path_to_data + "im2latex_train.csv")
        val_df = pd.read_csv(self.path_to_data + "im2latex_validate.csv")
        test_df = pd.read_csv(self.path_to_data + "im2latex_test.csv")
        dataframes = [train_df, val_df, test_df]

        for df in dataframes:
            df["image"] = df["image"].map(lambda x: self.fix_path(x))

        train_df = pd.concat([train_df, train_handwritten_df])
        val_df = pd.concat([val_df, val_handwritten_df])

        return train_df, val_df, test_df

    def fix_path(self, path):
        return self.path_to_data + "images/" + path
