import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from recommendation import CRITERIA

df = pd.read_csv(sys.argv[1]) 
df[CRITERIA] = (df[CRITERIA] - df[CRITERIA].min())/(df[CRITERIA].max() - df[CRITERIA].min())

coverage = pd.read_csv(sys.argv[2])

over_selected = coverage.loc[coverage['rg_l=1/10*m_t=80']>0.2, ['uid', 'rg_l=1/10*m_t=80', 'rank']]

f, axs = plt.subplots(3, 4, figsize=(13, 7))

print(df.loc[df['uid'].isin(over_selected['uid']), ['name', 'uploader']])

#Plot
for i in range(len(CRITERIA)):
    sns.barplot(
        data=df.loc[
            df['uid'].isin(over_selected['uid']), 
            CRITERIA+['uid'],
        ],
        x='uid', 
        y=CRITERIA[i], 
        ax=axs[i % 3, i % 4]
    )

sns.barplot(
    data=over_selected, x='uid', y='rg_l=1/10*m_t=80', ax=axs[2, 2]
)

sns.barplot(
    data=over_selected, x='uid', y='rank', ax=axs[2, 3]
)


plt.subplots_adjust(
    left=0.05, bottom=0.074, right=0.998, top=0.976, wspace=0.212, hspace=0.264
)

plt.show()
