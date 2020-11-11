# Importing libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
from imgaug import augmenters as iaa
import imgaug as ia

# Loading in data
X_train = np.load("Melanoma/data/x_train_224.npy")
y_train = pd.read_csv("Melanoma/data/train.csv")['target'].values

# Getting malignant images to augment
mal = y_train == 1
mal_imgs = X_train[mal]
ben_imgs = X_train[~mal]

# Augmenting malignant images to create bigger dataset
seq = iaa.Sequential([
    iaa.Affine(rotate=(-25, 25)),
    iaa.AdditiveGaussianNoise(scale=.02 * 255),
    iaa.Crop(percent=(0, 0.2)),
    iaa.HorizontalFlip(),
    iaa.Cutout(nb_iterations=(1, 5), cval=255),
], random_order=True)

# Getting augmented data
mal_aug = [seq(images=mal_imgs) for _ in tqdm(range(57))]
ben_aug = seq(images=ben_imgs)

# Concatenating lists of augmenting mal images into a np array
mal_aug_conc = np.vstack(mal_aug)

# Creating complete dataset
aug_data = np.vstack([mal_aug_conc, ben_aug])
aug_labels = np.array([1] * len(mal_aug_conc) + [0] * len(ben_aug))


# Shuffling data function
def shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# Shuffled data
aug_shuffle_imgs, aug_shuffle_labels = shuffle(aug_data, aug_labels)

# Saving out data
np.savez_compressed("Melanoma/data/augmented_224.npz",
                    aug_images=aug_shuffle_imgs,
                    aug_labels=aug_shuffle_labels)
