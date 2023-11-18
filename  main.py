import numpy as np
import torch
from pkg_resources import packaging

seed = 0
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.manual_seed(seed)

print("Torch version:", torch.__version__)

import clip

clip.available_models()
model, preprocess = clip.load("ViT-B/32")
model.cuda().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

import argparse
import os
from time import time
from matplotlib import pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from bindsnet.conversion import ann_to_snn
from bindsnet.encoding import RepeatEncoder
from bindsnet.datasets import ImageNet, CIFAR100, DataLoader
import torchvision.transforms as transforms
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ann.features = torch.nn.DataParallel(ann.features)
# ann.cuda()

time_rep = 50
root = 'data/'
batch_size = 32
one_step = False
percentile = 99.7

input_shape=(3, 32, 32)
print('==> Using Pytorch CIFAR-100 Dataset')
normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                    std=[0.267, 0.256, 0.276])
val_loader = CIFAR100(
    image_encoder=RepeatEncoder(time=time_rep, dt=1.0),
    label_encoder=None,
    root=root,
    download=True,
    train=False,
    transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        normalize, ]
    )
)

dataloader = DataLoader(
    val_loader,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

norm_loader = CIFAR100(
    image_encoder=RepeatEncoder(time=time_rep, dt=1.0),
    label_encoder=None,
    root=root,
    download=True,
    train=True,
    transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        normalize, ]
    )
)

eval_size = len(val_loader)


for step, batch in enumerate(torch.utils.data.DataLoader(norm_loader, batch_size=128)):
        data = batch['image']
        break

snn = ann_to_snn(model.visual.to(device), input_shape=input_shape, data=data, percentile=percentile)

torch.cuda.empty_cache()
snn = snn.to(device)

correct = 0
accuracies = np.zeros((time_rep, (eval_size//batch_size)+1), dtype=np.float32)
for step, batch in enumerate(tqdm(dataloader)):
    if (step+1)*batch_size > eval_size:
        break
    # Prep next input batch.
    inputs = batch["encoded_image"]
    labels = batch["label"]
    inpts = {"Input": inputs.to(device)}
    # inpts = {k: v.cuda() for k, v in inpts.items()}

    snn.run(inpts=inpts, time=time_rep, step=step, acc= accuracies, labels=labels,one_step=one_step)
    last_layer = list(snn.layers.keys())[-1]
    output_voltages = snn.layers[last_layer].summed
    prediction = torch.softmax(output_voltages, dim=1).argmax(dim=1)
    correct += (prediction.cpu() == labels).sum().item()
    snn.reset_()

final = accuracies.sum(axis=1) / eval_size
accuracy = 100 * correct / eval_size

print(f"SNN accuracy: {accuracy:.2f}")