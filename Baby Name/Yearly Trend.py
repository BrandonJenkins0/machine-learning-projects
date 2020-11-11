# Importing modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Reading in the data that was created in "Data Preprocessing.py"
data = pd.read_csv('Baby Name/data/full_names_dataset2018.csv')

sns.set_style('white')
plt.style.use('fivethirtyeight')

mapping = {'Penrod': ['Zachary', 'Lani', 'Halley', 'Kadence', 'Jacob'],
           'Knudsen': ['Tyler', 'Makayla', 'Sage'],
           'Harston': ['David', 'Nisha', 'Britnie', 'Jacob'],
           'Our': ['Valencia', 'Zachary', 'Caleb', 'Makayla', 'Brandon', 'Deniro']}

for key, values in mapping.items():
    dat = data.loc[data['name'].isin(values)]

    x = sns.relplot(x='year', y='births', row='name', kind='line', hue='name', linewidth=5, palette='Set2',
                    ci=None, height=5, aspect=3, facet_kws=dict(sharex=False, sharey=False),  data=dat)
    x.fig.suptitle(f'How popular are the names in {key} Family? \n Note: x & y scales vary', size=20)
    x.fig.subplots_adjust(top=.92)
    x.set_titles("{row_name}", size=15)
    x._legend.remove()
    x.set_ylabels('')
    x.savefig(f'Baby Name/Finished Charts/{key}_name_pop')

