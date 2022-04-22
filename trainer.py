import glob
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from torchvision import transforms as t
from torchvision import models
import random
import torch.onnx
import onnx

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


# paths to kaggle notebook
# PROJECT_DIR = '../input/stanford-dogs-dataset/'
# IMAGES_DIR = PROJECT_DIR + 'images/Images'


def get_images_paths():
    images_dir = (Path(os.environ.get("HOME"))
                  .joinpath("PycharmProjects")
                  .joinpath("StanfordDogs")
                  .joinpath("data")
                  .joinpath("Images")
                  .joinpath("n02092002-Scottish_deerhound")
                  )
    os.chdir(images_dir)
    return glob.glob('./*.jpg')


class DatasetSD(Dataset):
    def __init__(self, paths, class_names, trg, transform=None):
        self.images_paths = paths
        self.class_names = class_names
        self.targets = trg
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img = Image.open(self.images_paths[index])
        return {
            'img': self.transform(img) if self.transform else img,
            'trg': self.targets[index]
        }


def collate_fn(data):
    images, trg = zip(*list(map(
        lambda i:
        (i['img'] if i['img'].shape[0] == 3 else i['img'][[0, 1, 2], :, :],
         i['trg']),
        data)))
    return torch.stack(images).float(), torch.tensor(trg)


def compute_f1(model, loader, dvc):
    model.eval()

    f1_accum = 0
    i_step = 0
    for i_step, batch in enumerate(loader):
        data = batch[0].to(dvc)
        ground_truth = batch[1].to(dvc)
        pred = model(data)
        f1_accum += f1_score(ground_truth.view(-1).cpu(),
                             torch.max(pred, -1)[1].view(-1).cpu(),
                             average='weighted')
    return f1_accum / (i_step+1)


def train(n_epochs, train_data, val_data, model,
          loss_func, optimizer, dvc, bs, n_breeds, scheduler):
    for epoch in range(n_epochs):
        train_loader = DataLoader(
            dataset=train_data,
            collate_fn=collate_fn,
            batch_size=bs,
            shuffle=True,
            drop_last=True
        )
        val_loader = DataLoader(
            dataset=val_data,
            collate_fn=collate_fn,
            batch_size=bs,
            shuffle=True,
            drop_last=True
        )

        loss_accum = 0
        train_f1_accum = 0
        i_step = 0
        for i_step, batch in enumerate(train_loader):
            model.train()
            data = batch[0].to(dvc)
            trg = batch[1].to(dvc)
            pred = model(data).logits
            loss = loss_func(pred.view(-1, n_breeds), trg.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_accum += loss
            train_f1_accum += f1_score(trg.view(-1).cpu(),
                                       torch.max(pred, -1)[1].view(-1).cpu(),
                                       average='weighted')

        ave_loss = loss_accum / (i_step+1)
        train_f1 = train_f1_accum / (i_step+1)
        val_f1 = compute_f1(model, val_loader, dvc)
        scheduler.step(val_f1)

        print(f'Ave loss: {ave_loss}, Train f1: {train_f1}, Val f1: {val_f1}')


if __name__ == '__main__':
    img_paths = get_images_paths()
    dog_breeds = [path.split('-')[-1].split('/')[0] for path in img_paths]
    dog_breeds = np.array(dog_breeds)
    targets = LabelEncoder().fit_transform(dog_breeds)

    split = list(StratifiedKFold(5).split(img_paths, targets))[0]
    train_indices, val_indices = split[0], split[1]

    transforms = t.Compose([
                            t.Resize((299, 299)),
                            t.ToTensor()
                            ])
    train_set = DatasetSD(
        img_paths[train_indices],
        dog_breeds[train_indices],
        targets[train_indices],
        transform=transforms
        )
    val_set = DatasetSD(
        img_paths[val_indices],
        dog_breeds[val_indices],
        targets[val_indices],
        transform=transforms
        )

    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
    n_classes = len(set(targets))

    net = models.inception_v3(pretrained=True)
    for param in list(net.parameters()):
        param.requires_grad = False
    net.fc = nn.Linear(2048, n_classes)
    net.to(device)

    num_epochs = 10
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters(), 0.0015)
    lr_scheduler = ReduceLROnPlateau(optim,
                                     patience=1,
                                     mode='max',
                                     factor=0.75,
                                     verbose=True,
                                     threshold=0.01)
    batch_size = 128

    train(num_epochs, train_set, val_set, net,
          criterion, optim, device, batch_size, n_classes, lr_scheduler)

    x = torch.randn(1, 3, 299, 299, requires_grad=True).to(device)
    torch.save(x, 'x.pt')
    torch_out = net(x)
    torch.save(torch_out, 'torch_out.pt')

    torch.onnx.export(net,
                      x,
                      "StanfordDogs.onnx",
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})

    onnx_model = onnx.load("StanfordDogs.onnx")
    onnx.checker.check_model(onnx_model)
