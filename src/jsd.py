#%%
import numpy as np 
from scipy.spatial.distance import jensenshannon, pdist, squareform
import seaborn as sns

K = 7

topic_dists = np.asarray([ np.random.rand(12) for _ in range(K) ])

M = squareform(pdist(topic_dists, jensenshannon)) + np.diag(np.ones(7))

sns.heatmap(M, square=True, annot=True, cmap="Reds")


# %%
