import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
from torch.nn.utils.rnn import pad_sequence
import os


class ImagesDataset(Dataset):
    def __init__(self, image_paths, formulas, classification=False, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.formulas = formulas
        self.classification = classification

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths.iloc[index]

        image = Image.open(image_path)

        formula = self.formulas.iloc[index]

        if self.transform:
            image = self.transform(image)

        if self.classification:
            label = 'handwritten' in image_path 
            return image, label

        return image, formula


class Im2LatexDataset:
    def __init__(
        self,
        block_size=216,
        batch_size=32,
        img_dims=[224,640],
        device=torch.device("cpu"),
        path_to_data="./",
        tokenizer="./tokenizer.json",
        classification=False,
    ):
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer)
        self.tokenizer.pad_token = "[PAD]"
        self.path_to_data = path_to_data
        self.batch_size = batch_size
        self.block_size = block_size
        dfs = (train_df, val_df, test_df) = self.load_dataframes()
        train_dataset, val_dataset, test_dataset = self.load_datasets(*dfs, img_dims, classification=classification)
        H, W = img_dims


        #define what happens when batches are collected
        def collate_fn(batch):
            images, labels = zip(*batch)
            images = torch.cat(images).view(-1, 1, H, W) # convert from tuple into batched tensor (B, C, H, W)

            if classification:
                labels = torch.tensor([float(val) for val in labels]).reshape(-1, 1)
                return images, labels

            tokens = self.tokenizer.batch_encode_plus(
                labels,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False,
                max_length=self.block_size,
                pad_to_max_length=True,
            )
            return images, tokens

        self.train = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, 
        )
        self.val = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        )
        self.test = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, 
        )

    def load_datasets(
        self, train_df, val_df, test_df, img_dims, classification=False
    ) -> tuple[ImagesDataset, ImagesDataset, ImagesDataset]:
        transform = transforms.Compose(
            [
                transforms.Resize(img_dims),  # Resize to a specific size
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),  # Convert to tensor
            ]
        )

        train_dataset = ImagesDataset(
            train_df["image"], train_df["formula"], transform=transform, classification=classification
        )
        val_dataset = ImagesDataset(
            val_df["image"], val_df["formula"], transform=transform, classification=classification
        )
        test_dataset = ImagesDataset(
            test_df["image"], test_df["formula"], transform=transform, classification=classification
        )

        return train_dataset, val_dataset, test_dataset

    def load_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_handwritten_df = pd.read_csv(
            self.path_to_data + "train_handwritten.csv"
        ).dropna()
        val_handwritten_df = pd.read_csv(
            self.path_to_data + "val_handwritten.csv"
        ).dropna()
        train_df = pd.read_csv(self.path_to_data + "im2latex_train.csv")
        val_df = pd.read_csv(self.path_to_data + "im2latex_validate.csv")
        test_df = pd.read_csv(self.path_to_data + "im2latex_test.csv")
        dataframes = [train_df, val_df, test_df]

        def fix_path(path):
            return self.path_to_data + "images/" + path

        for df in dataframes:
            df["image"] = df["image"].map(lambda x: fix_path(x))


        for _ in range(4):
            train_df = pd.concat(
                [train_df, train_handwritten_df], axis=0
            )  # combine handwritten and non handwritten dataframes
            val_df = pd.concat([val_df, val_handwritten_df], axis=0)

        return train_df, val_df, test_df

    def generate_tokenizer(equations, output, vocab_size):
        from tokenizers import Tokenizer, pre_tokenizers
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer

        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = BpeTrainer(
            special_tokens=[
                "[PAD]",
                "[BOS]",
                "[EOS]",
            ],
            vocab_size=vocab_size,
            show_progress=True,
        )
        tokenizer.pad_token = "[PAD]"
        tokenizer.train([equations], trainer)
        tokenizer.save(path=output, pretty=False)
