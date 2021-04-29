import numpy as np
import pandas as pd
from timm.data import resolve_data_config
from sklearn.linear_model import LogisticRegression
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

@torch.no_grad()
def extract_features(model, dataloader, avg_pool=False, device="cpu"):
    Flist = []
    Ylist = []
    for i, (X, Y) in enumerate(dataloader):
        X = X.to(device)
        x = model.forward_features(X)
        if avg_pool:
            x = F.adaptive_avg_pool2d(x, (1,1))
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

parser = argparse.ArgumentParser(description='Pytorch Image Models where we fine-tune only last layer')
parser.add_argument('--model', default="resnet50", type=str, required=True)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--avg-pool', default=False, action="store_true")
parser.add_argument('--img-version', default="rawImg")
args = parser.parse_args()


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("Device:", device)
# parent folder should contain train.csv and submission_valid.csv

print("Init model")
model = timm.create_model(args.model, pretrained=True)
model = model.to(device)
model.eval()
config = resolve_data_config({}, model=model)
transform = create_transform(**config)
index_to_classname = {i:n for n, i in classname_to_index.items()}

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
        X_train, y_train = extract_features(model, train_loader, avg_pool=args.avg_pool, device=device)
        # fit linear model to predict prognosis
        clf = LogisticRegression(max_iter=1000,class_weight="balanced")
        clf.fit(X_train, y_train)
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
        X_test, _ = extract_features(model, test_loader, avg_pool=args.avg_pool, device=device)
        Ypred_test = self.clf.predict(X_test)
        # predict the prognosis label
        Ypred_test = [index_to_classname[y] for y in Ypred_test]
        
        # use tabular model to impute
        df_test_imputed = self.tabular.predict(df_test)
        # if images available, use vision model to predict the label
        df_test_imputed.loc[~missing_images, LABEL] = Ypred_test
        # if images not available, use tabular model to predict the label
        if np.any(missing_images):
            df_test_imputed.loc[missing_images, LABEL] = self.tabular.predict(df_test[missing_images])
        return df_test_imputed

if __name__ == "__main__":
    full_evaluation(FineTuneLastLayer)
