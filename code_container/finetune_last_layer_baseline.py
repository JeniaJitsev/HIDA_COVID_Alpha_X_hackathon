import numpy as np
from skimage.io import imread
import pandas as pd
from timm.data import resolve_data_config
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


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

@torch.no_grad()
def extract_features(model, dataloader, avg_pool=False, device="cpu"):
    Flist = []
    Ylist = []
    for i, (X, Y) in enumerate(dataloader):
        X = X.to(device)
        x = features(model, X, avg_pool=avg_pool)
        x = x.view(x.size(0), -1)
        x = x.data.cpu()
        Flist.append(x)
        Ylist.append(Y)
    return torch.cat(Flist).numpy(), torch.cat(Ylist).numpy()


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
    else:
        model = timm.create_model(args.model, pretrained=True)
    return model

def build_transforms(model):
    if model.type in XRV:
        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])
        return transform
    else:
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        return transform

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
    model = load_model(args.model)
    model = model.to(device)
    model.eval()
    model.type = args.model

    index_to_classname = {i:n for n, i in classname_to_index.items()}
    transform = build_transforms(model)

    class FineTuneLastLayer(Submission):
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
                shuffle=False,
                num_workers=WORKERS,
            )
            # extract features of last layer
            X_train, y_train = extract_features(model, train_loader, avg_pool=not args.no_avg_pool, device=device)
            print(X_train.shape, y_train.shape)
            # fit linear model to predict prognosis
            # clf = LogisticRegression(class_weight="balanced", n_jobs=-1)
            # clf = RandomForestClassifier(n_jobs=-1)
            clf = GradientBoostingClassifier()
            clf.fit(X_train, y_train)
            print("train acc", (clf.predict(X_train) == y_train).mean())
            self.clf = clf
            
        def predict(self, df_test):

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
            # extract features of last layer
            X_test, _ = extract_features(model, test_loader, avg_pool=not args.no_avg_pool, device=device)
            Y_pred_test = self.clf.predict(X_test)
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
    full_evaluation(FineTuneLastLayer)
