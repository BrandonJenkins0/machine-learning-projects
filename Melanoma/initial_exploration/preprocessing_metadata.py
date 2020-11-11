# Loading modules
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# Current directory
os.getcwd()

# Loading in data
train = pd.read_csv('Melanoma/data/train.csv')
test = pd.read_csv("Melanoma/data/test.csv")

# Replacing missing data. Based off intuition gained in the eda process

# Mole location replacing nas with 'missing'
train['anatom_site_general_challenge'].fillna('missing', inplace=True)
test['anatom_site_general_challenge'].fillna('missing', inplace=True)

# Imputing age with average age per target group
train['age_approx'] = train.groupby('target')['age_approx'].transform(lambda x: x.fillna(x.mean()))

# Imputing sex with 'missing'
train['sex'].fillna('missing', inplace=True)

# Dropping features that arent in test set
train_dropped = train.drop(['image_name', 'patient_id', 'diagnosis', 'benign_malignant', 'target'], axis=1)
test_dropped = test.drop(['image_name', 'patient_id'], axis=1)

# Encoding categorical features
train_encoded = pd.get_dummies(train_dropped, drop_first=True)
test_encoded = pd.get_dummies(test_dropped, drop_first=True)
test_encoded['sex_missing'] = 0
test_encoded = test_encoded[train_encoded.columns]

# Scaling data
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_encoded)
test_scaled = scaler.transform(test_encoded)

# Saving data to .npz file
np.savez_compressed("Melanoma/data/preprocessed_meta.npz",
                    train_meta=train_scaled,
                    test_meta=test_scaled,
                    meta_labels=train_encoded.columns.values)