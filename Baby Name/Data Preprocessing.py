# Importing modules
import pandas as pd

# Reading in the data
# Data was downloaded from https://www.ssa.gov/oact/babynames/limits.html. Within the zip file are txt files for each
# year. I am looping through all the text files and concatenating them into one dataset. I removed the txt files to
# free up some space.
pieces = []
columns = ['name', 'sex', 'births']

for year in range(1880, 2019):
    file_path = f'Baby Name/data/names/yob{year}.txt'
    data = pd.read_csv(file_path, names=columns)
    data['year'] = year
    pieces.append(data)

full_data = pd.concat(pieces, ignore_index=True)

# Writing out full dataset
full_data.to_csv('Baby Name/data/full_names_dataset2018.csv', index=False)