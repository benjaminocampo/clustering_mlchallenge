# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils import (get_embedding, get_vocab_by_action, get_action_vec,
                   get_act_schs, get_objects, get_embedding_2DTSNE,
                   plot_vocabulary_kmeans)
# %%
TRAIN_URL = "https://www.famaf.unc.edu.ar/~nocampo043/training-instances.parquet.gzip"
TEST_URL = "https://www.famaf.unc.edu.ar/~nocampo043/evaluation-instances.parquet.gzip"

df_train = pd.read_parquet(TRAIN_URL)
df_test = pd.read_parquet(TEST_URL)
# %%
corpus = df_train["relaxed_plan"].to_numpy()
# %%
act_schs = get_act_schs(corpus)
objects = get_objects(corpus)
vocab = get_vocab_by_action(corpus)
longest_interface = max([len(action.split()) - 1 for action in vocab])
# %%
sch_enc = {sch: i for i, sch in enumerate(act_schs, start=1)}
obj_enc = {obj: i for i, obj in enumerate(objects, start=1)}
# %%
vector_dim = longest_interface * 2 + 1
model = {
    action: get_action_vec(action, vector_dim, sch_enc, obj_enc)
    for action in vocab
}
# %%
embedding = get_embedding(vocab, model)
embedding_TSNE = get_embedding_2DTSNE(vocab, model)
X = embedding.drop(columns=["word"]).to_numpy()
X_TSNE = embedding_TSNE.drop(columns=["word"]).to_numpy()
# %%
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
# %% [markdown]
# ## Plots
# %%
_, ax_kmeans = plt.subplots(figsize=(20, 10))
plot_vocabulary_kmeans(X_TSNE, kmeans, ax_kmeans)
ax_kmeans.grid()
# %%
