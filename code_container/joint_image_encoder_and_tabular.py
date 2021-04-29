import numpy as np
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

from evaluation import COLS, Submission, MISSING_IMAGES_RATE
from tabular_baseline import TabularBaseline

LABEL = "Prognosis"
classname_to_index = {"SEVERE": 1, "MILD": 0}
WORKERS = 16
MISSING_VALUE = 999
XRV = ['chex', 'all']

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

def features(model, x, avg_pool=True):
    if model.type in XRV:
        return model.features2(x)
    else:
        x = model.forward_features(x)
        if avg_pool:
            x = F.adaptive_avg_pool2d(x, (1,1))
        return x
def build_dataset(df, folder):
  filenames = [os.path.join(folder, name) for name in df.ImageFile]
  labels = [classname_to_index.get(name,0) for name in df[LABEL]]
  dataset = CustomDataSet(filenames, transform=transform, labels=labels)
  return dataset

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

    class JointImageEncoderAndTabular(Submission):
        """
        Simple fine-tuneing of last layer of pre-trained models
        from TIMM's repo (https://github.com/rwightman/pytorch-image-models).

        At test time:
        - if the image is available, use the learned model for predicting prognosis
        - If the image is not available, use the simple tabular baseline (tabular_baseline.py)
          for predicting prognosis
        """

        def tabular_features(self, df):
            df_imputed = self.tabular.predict(df)
            df_imputed = df_imputed.mask(~pd.isna(df), df)
            return df_imputed[COLS].values

        def fit(self, df_train):
            df_train = df_train.copy()
            
            # Fit imputation model on training
            self.tabular = TabularBaseline(train_path=self.train_path, test_path=self.test_path)
            self.tabular.fit(df_train)
        

            # get tabular data cols
            X_train_tabular = self.tabular_features(df_train)

            # build image dataset
            train_dataset = build_dataset(df_train, folder=os.path.join(self.train_path, args.img_version))
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=WORKERS,
            )
            # extract features of last layer
            X_train_image_features, y_train = extract_features(model, train_loader, avg_pool=not args.no_avg_pool, device=device)

            # artificially make some images missing, because we have that in the test phase
            missing_images = (np.random.uniform(size=len(X_train_image_features)) <= MISSING_IMAGES_RATE)
            X_train_image_features[missing_images] = MISSING_VALUE
            
            # Concat tabular values and image features
            X_train = np.concatenate((X_train_tabular, X_train_image_features), axis=1)

            # fit a model on the concatenated features
            clf = GradientBoostingClassifier()
            # clf = RandomForestClassifier()
            clf.fit(X_train, y_train)
            self.clf = clf
            
        def predict(self, df_test):

            # construct test dataset using non missing images
            df_test = df_test.copy()

            # Get tabular cols
            X_test_tabular = self.tabular_features(df_test)
            
            # Get non missing images
            missing_images = pd.isna(df_test.ImageFile)
            test_dataset = build_dataset(df_test[~missing_images], folder=os.path.join(self.test_path, args.img_version))
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=WORKERS,
            )
            # Compute image features for non missing images
            X_test_image_features_non_missing, _ = extract_features(model, test_loader, avg_pool=not args.no_avg_pool, device=device)
            X_test_image_features = np.zeros((len(df_test), X_test_image_features_non_missing.shape[1]  ))
            X_test_image_features[:] = MISSING_VALUE
            X_test_image_features[~missing_images] = X_test_image_features_non_missing
            
            X_test = np.concatenate((X_test_tabular, X_test_image_features), axis=1)
            Y_pred_test = self.clf.predict(X_test)
            Y_pred_test = [index_to_classname[y] for y in Y_pred_test]
            df_test_imputed = self.tabular.predict(df_test)
            df_test_imputed[LABEL] = Y_pred_test
            return df_test_imputed

    full_evaluation(JointImageEncoderAndTabular)
