# Loading in the modules
import pandas as pd
pd.set_option('display.max_columns', 500)

# Loading in the train, val, test labels
train_labels = pd.read_json("Hate Memes/data/train.jsonl", lines=True)
val_labels = pd.read_json("Hate Memes/data/dev.jsonl", lines=True)
test_labels = pd.read_json("Hate-Memes/data/test.jsonl", lines=True)

# Writing text files to create the desired folder structure
train_labels.loc[train_labels['label'] == 1, 'img'].to_csv("Hate Memes/data/train_hate.txt", sep=' ', index=False)
train_labels.loc[train_labels['label'] == 0, 'img'].to_csv("Hate Memes/data/train_nohate.txt", sep=' ', index=False)
test_labels['img'].to_csv("Hate Memes/data/test.txt", sep=' ', index=False)
val_labels.loc[train_labels['label'] == 1, 'img'].to_csv("Hate Memes/data/val_hate.txt", sep=' ', index=False)
val_labels.loc[train_labels['label'] == 0, 'img'].to_csv("Hate Memes/data/val_nohate.txt", sep=' ', index=False)