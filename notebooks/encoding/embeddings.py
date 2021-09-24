# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from adjustText import adjust_text
from collections import Counter
from gensim.models import FastText
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

TRAIN_URL = "https://www.famaf.unc.edu.ar/~nocampo043/training-instances.parquet.gzip"
TEST_URL = "https://www.famaf.unc.edu.ar/~nocampo043/evaluation-instances.parquet.gzip"

df_train = pd.read_parquet(TRAIN_URL)
df_test = pd.read_parquet(TEST_URL)


def get_embedding(vocab, model):
    return (
        pd.DataFrame({word: model[word] for word in vocab})
        .transpose()
        .reset_index()
        .rename(columns={"index": "word"})
    )


def get_embedding_2DTSNE(vocab, model):
    embedding = get_embedding(vocab, model)
    X = embedding.drop(columns=["word"])
    X_TSNE = TSNE(n_components=2).fit_transform(X)
    embedding_TSNE = pd.concat(
        [pd.DataFrame(vocab, columns=["word"]),
         pd.DataFrame(X_TSNE)], axis=1)
    return embedding_TSNE


def get_object_types(vocab):
    nof_words = len(vocab)
    obj_types = np.zeros(nof_words)
    obj_indices = {"Actions/Separators": 0}
    for i, word in enumerate(vocab):
        match = re.search(r"([A-Za-z]+)\d", word)
        if match:
            if match[1] not in obj_indices:
                obj_indices[match[1]] = len(obj_indices)
            obj_types[i] = obj_indices[match[1]]
    return obj_types, obj_indices


def tokenize_plan(plan):
    begin_token = "("
    end_token = ")"
    wrap = lambda action: begin_token + " " + action + " " + end_token
    splited_actions = np.concatenate([wrap(action).split() for action in plan])
    return splited_actions.tolist()


def plot_vocabulary_2DTSNE(X_TSNE,
                           vocab,
                           obj_t,
                           objs,
                           ax,
                           point_size=50,
                           marker_size=2,
                           legend_size=20,
                           tick_size=20,
                           annotation_size=20):
    nof_words = len(X_TSNE)
    cm = plt.get_cmap('tab20')
    idxs = [1. * i / 19 for i in range(20)]
    ax.set_prop_cycle(color=[cm(idx) for idx in idxs[:len(objs)]])
    for obj, i in objs.items():
        ax.scatter(X_TSNE[obj_t == i, 0],
                   X_TSNE[obj_t == i, 1],
                   s=point_size,
                   label=obj.capitalize())
    ax.legend(bbox_to_anchor=(1.02, 1.02),
              loc='upper left',
              markerscale=marker_size,
              prop={"size": legend_size})
    ax.set_title("t-SNE Plot", fontsize=legend_size)
    annot_list = []
    for i in range(nof_words):
        if obj_t[i] == 0:
            a = ax.annotate(vocab[i], (X_TSNE[i, 0], X_TSNE[i, 1]),
                            size=annotation_size)
            annot_list.append(a)
    adjust_text(annot_list)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)


def plot_vocabulary_kmeans(X_TSNE,
                           km_model,
                           ax,
                           point_size=50,
                           marker_size=2,
                           legend_size=20,
                           tick_size=20):
    cluster_df = pd.DataFrame(X_TSNE)
    cluster_df = cluster_df.assign(label=km_model.labels_)
    for k in range(km_model.n_clusters):
        ax.scatter(X_TSNE[km_model.labels_ == k, 0],
                   X_TSNE[km_model.labels_ == k, 1],
                   s=point_size,
                   label=k)
    ax.legend(bbox_to_anchor=(1.02, 1.02),
              loc='upper left',
              markerscale=marker_size,
              prop={"size": legend_size})
    ax.set_title("KMeans Plot", fontsize=legend_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
# %% [markdown]
# ## FastText
# %%
sentences = df_train["relaxed_plan"].to_numpy()
fasttext = FastText(sentences=[tokenize_plan(s) for s in sentences],
                    min_count=1,
                    vector_size=100,
                    window=7)
# %% [markdown]
# ## TSNE
# %%
model = fasttext.wv
vocab = fasttext.wv.index_to_key

embedding = get_embedding(vocab, model)
embedding_TSNE = get_embedding_2DTSNE(vocab, model)
X = embedding.drop(columns=["word"]).to_numpy()
X_TSNE = embedding_TSNE.drop(columns=["word"]).to_numpy()
# %% [markdown]
# ## KMeans
# %%
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
# %% [markdown]
# ## Object type
# %%
obj_types, obj_indices = get_object_types(vocab)
# %% [markdown]
# ## Plots
# %%
_, (ax_kmeans, ax_tsne) = plt.subplots(1, 2, figsize=(30, 10))
plot_vocabulary_2DTSNE(X_TSNE, vocab, obj_types, obj_indices, ax_tsne)
plot_vocabulary_kmeans(X_TSNE, kmeans, ax_kmeans)
ax_kmeans.grid()
ax_tsne.grid()
# %%

# %%
