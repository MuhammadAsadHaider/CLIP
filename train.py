import clip
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image


df = pd.read_csv("dataset/captions.txt")
df['id'] = [id_ for id_ in range(df.shape[0] // 5) for _ in range(5)]
df.to_csv("dataset/captions.csv", index=False)
df = pd.read_csv("dataset/captions.csv")


batch_size = 200
EPOCH = 10


device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
print("Using device: ", device)
model, preprocess = clip.load("ViT-B/16",device=device,jit=False) #Must set jit=False for training


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names
        """

        self.image_filenames = image_filenames
        self.captions = list(captions)

    def __getitem__(self, idx):
        item = {}

        image = Image.open(f"dataset/Images/{self.image_filenames[idx]}").convert("RGB")
        image = preprocess(image)
        item['image'] = image
        item['caption'] = self.captions[idx]
        return item

    def __len__(self):
        return len(self.captions)


def make_train_valid_dfs():
    dataframe = pd.read_csv(f"dataset/captions.csv")
    max_id = dataframe["id"].max() + 1
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe


def build_loaders(dataframe, mode):
    dataset = CLIPDataset(
        dataframe["image"].values,
        dataframe["caption"].values
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if mode == "train" else False,
    )
    return dataloader


def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()


train_dataframe , valid_dataframe = make_train_valid_dfs()
train_dataloader = build_loaders(train_dataframe, mode="train")
valid_dataloader = build_loaders(valid_dataframe, mode="valid")


loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.visual.pacl_embedder.parameters(), lr=5e-4,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH * len(train_dataloader) // batch_size, eta_min=1e-6)


for epoch in tqdm(range(EPOCH)):
    epoch_loss = 0
    for batch in tqdm(train_dataloader) :
        optimizer.zero_grad()

        images= batch['image'].to(device)
        texts = clip.tokenize(batch['caption']).to(device)

        logits_per_image, logits_per_text = model(images, texts)

        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        print('Batch Loss : ', total_loss.item())
        epoch_loss += total_loss.item()
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else: 
            convert_models_to_fp32(model)
        optimizer.step()
        lr_scheduler.step()
        clip.model.convert_weights(model)
    total_loss = epoch_loss / len(train_dataloader)
    print(f"Epoch {epoch} : {total_loss}")
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
            }, f"model_checkpoint/model_epoch_{epoch}.pt")





