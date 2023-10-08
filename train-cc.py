import clip
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import PIL

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import urllib
from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent


batch_size = 64
EPOCH = 10
num_threads = 4
USER_AGENT = get_datasets_user_agent()


def fetch_single_image(image_url, timeout=None, retries=0):
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = PIL.Image.open(io.BytesIO(req.read()))
            break
        except Exception:
            image = None
    return image


def fetch_images(batch, num_threads, timeout=None, retries=0):
    fetch_single_image_with_args = partial(fetch_single_image, timeout=timeout, retries=retries)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch["image"] = list(executor.map(fetch_single_image_with_args, batch["image_url"]))
    return batch

dset = load_dataset("conceptual_captions", split='train', streaming=True)
dset = dset.with_format("torch")

device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
print("Using device: ", device)
model, preprocess = clip.load("ViT-B/16",device=device,jit=False) #Must set jit=False for training


def build_loaders(dset, mode):
    dataloader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size
    )
    return dataloader


def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()


train_dataloader = build_loaders(dset, mode="train")

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.visual.pacl_embedder.parameters(), lr=5e-4,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH * 3318333 // batch_size, eta_min=1e-6)


for epoch in tqdm(range(EPOCH)):
    epoch_loss = 0
    for batch in tqdm(train_dataloader):
        batch = fetch_images(batch, num_threads=num_threads)
        images = batch["image"]
        captions = batch["caption"]
        valid_images = []
        valid_captions = []
        for i in range(len(images)):
            try:
                image = preprocess(images[i])
                valid_images.append(image)
                valid_captions.append(captions[i])
            except:
                continue

        images = torch.stack(valid_images).to(device)
        texts = clip.tokenize(valid_captions).to(device)

        optimizer.zero_grad()

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
            }, f"model_checkpoint_cc/model_epoch_{epoch}.pt")





