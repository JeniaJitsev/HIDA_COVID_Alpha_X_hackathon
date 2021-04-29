import numpy as np
from skimage.io import imread
import pandas as pd
from timm.data import resolve_data_config
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from skimage.io import imsave

import torch.nn as nn
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import time
import os
import argparse
from torch.utils.data import Dataset
import torch.nn.functional as F
from evaluation import full_evaluation
from sklearn.ensemble import RandomForestClassifier

import timm
from timm.data.transforms_factory import create_transform

from evaluation import COLS, Submission
from tabular_baseline import TabularBaseline

LABEL = "Prognosis"
classname_to_index = {"SEVERE": 1, "MILD": 0}
WORKERS = 16
XRV = ['chex', 'all']

class CustomDataSet(Dataset):
    def __init__(self, filenames, transform=None, labels=None):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if args.model in XRV:
            image = imread(self.filenames[idx]) / 255.0
            image = image.reshape((1, image.shape[0], image.shape[1]))
        else:
            image = Image.open(self.filenames[idx]).convert("RGB")
        label = self.labels[idx] if self.labels is not None else 0
        if self.transform:
            tensor_image = self.transform(image)
        return tensor_image, label

def build_dataset(df, folder):
  filenames = [os.path.join(folder, name) for name in df.ImageFile]
  labels = [classname_to_index.get(name,0) for name in df[LABEL]]
  dataset = CustomDataSet(filenames, transform=transform, labels=labels)
  return dataset

import torchxrayvision as xrv

def features(model, x, avg_pool=True):
    if model.type in XRV:
        return model.features2(x)
    else:
        x = model.forward_features(x)
        if avg_pool:
            x = F.adaptive_avg_pool2d(x, (1,1))
        return x

def load_model(model_type):
    if model_type in XRV:
        model = xrv.models.DenseNet(weights=model_type)
        model.classifier = nn.Linear(1024, 1)
    else:
        model = timm.create_model(args.model, pretrained=True, num_classes=1)
    return model

def build_transforms(model):
    if model.type in XRV:
        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])
        return transform
    else:
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        return transform

def pred(model, X):
    if model.type in XRV:
        outputs = model.classifier(model.features2(X))[:,0]
    else:
        outputs = model(X)[:,0]
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pytorch Image Models where we fine-tune only last layer')
    parser.add_argument('--model', default="resnet50", type=str, required=True)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--no-avg-pool', default=False, action="store_true")
    parser.add_argument('--img-version', default="normalizedImg")
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("Device:", device)

    index_to_classname = {i:n for n, i in classname_to_index.items()}
    model = load_model(args.model)
    model.type = args.model
    transform = build_transforms(model)

    class FullFineTuning(Submission):
        """
        Simple fine-tuneing of last layer of pre-trained models
        from TIMM's repo (https://github.com/rwightman/pytorch-image-models).

        At test time:
        - if the image is available, use the learned model for predicting prognosis
        - If the image is not available, use the simple tabular baseline (tabular_baseline.py)
          for predicting prognosis
        """
        def fit(self, df_train):
            self.tabular = TabularBaseline(train_path=self.train_path, test_path=self.test_path)
            self.tabular.fit(df_train)

            # build image dataset
            train_dataset = build_dataset(df_train, folder=os.path.join(self.train_path, args.img_version))
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=WORKERS,
            )
            model = load_model(args.model)
            model = model.to(device)
            model.train()
            model.type = args.model
            self.model = model
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01,step_size_up=5,mode="triangular2")
            # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
            print("Train")
            for epoch in range(5):
                for X, Y in train_loader:
                    X = X.to(device)
                    Y = Y.float().to(device)
                    optimizer.zero_grad()
                    outputs = pred(model, X)
                    loss = criterion(outputs, Y)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                acc = ((outputs.sigmoid()>0.5) == Y).float().mean()
                print(loss.item(), acc.item())
            print("Finish train")
            
        def predict(self, df_test):
            torch.cuda.empty_cache()
            self.model.eval()
            # construct test dataset using non missing images
            df_test = df_test.copy()
            missing_images = pd.isna(df_test.ImageFile)
            test_dataset = build_dataset(df_test[~missing_images], folder=os.path.join(self.test_path, args.img_version))
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=WORKERS,
            )
            Y_pred_test = []
            with torch.no_grad():
                for X, _ in test_loader:
                    X = X.to(device)
                    outputs = pred(self.model, X)
                    y = (outputs.sigmoid() > 0.5).cpu().numpy().astype(int)
                    Y_pred_test.append(y)
            Y_pred_test = np.concatenate(Y_pred_test).tolist()
            # predict the prognosis label
            Y_pred_test = [index_to_classname[y] for y in Y_pred_test]
            
            # use tabular model to impute
            df_test_imputed = self.tabular.predict(df_test)
            # if images available, use vision model to predict the label
            df_test_imputed.loc[~missing_images, LABEL] = Y_pred_test
            # if images not available, use tabular model to predict the label
            if np.any(missing_images):
                df_test_imputed.loc[missing_images, LABEL] = self.tabular.predict(df_test[missing_images])
            return df_test_imputed
    full_evaluation(FullFineTuning)
