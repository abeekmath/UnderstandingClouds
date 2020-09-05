import os 
import cv2
import numpy as np
import pandas as pd 
from sklearn.model_selection import StratifiedKFold


if __name__ == "__main__":
    in_path = r"..\data"
    class_names_dict = {'Fish':1, 'Flower':2, 'Gravel':3, 'Sugar':4}

    df = pd.read_csv(os.path.join(in_path, "train.csv"))
    df['exists'] = df['EncodedPixels'].notnull().astype(int)
    df["image_name"] = df["Image_Label"].map(lambda x: x.split('_')[0].strip())
    df['class_name'] = df['Image_Label'].map(lambda x: x.split('_')[-1])
    df['class_id'] = df['class_name'].map(class_names_dict)
    df['class_id'] = [row.class_id if row.exists else 0 for row in df.itertuples()]

    df.kfold = -1
    df = df.sample(frac=1).reset_index(drop = True)
    y = df.class_id.values
    kf = StratifiedKFold(n_splits = 5)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=y)):
        df.loc[val_idx, 'kfold'] = fold
    df.to_csv(os.path.join(in_path, "train_folds.csv"), index=False)