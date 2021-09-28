import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from adjustText import adjust_text
from collections import Counter
from sklearn.manifold import TSNE


def get_vocab_by_exp(corpus):
    all_words = np.concatenate([tokenize_plan(plan) for plan in corpus])
    return np.array(list(Counter(all_words).keys()))

def get_vocab_by_action(corpus):
    all_words = np.concatenate(corpus)
    return np.array(list(Counter(all_words).keys()))

def get_action_vec(action, vector_dim, sch_enc, obj_enc):
    vec = np.zeros(vector_dim)
    schema, *instances = action.split()
    vec[0] = sch_enc[schema]
    for i, instance in zip(range(1, vector_dim - 1, 2), instances):
        instance_number = int(re.search(r"\d+", instance)[0]) + 1
        object = re.search(r"[A-Za-z]+", instance)[0]
        vec[i] = obj_enc[object]
        vec[i + 1] = instance_number
    return vec

def get_custom_model(vocab):
    pass

def get_coo_ocurrence_model(vocab):
    pass

def get_act_schs(corpus):
    all_words = np.concatenate(
        [tokenize_plan(plan, with_wrapping=False)
        for plan in corpus])
    act_schs = set()
    for w in all_words:
        match = re.search(r"[A-Za-z]+\d", w)
        if match is None:
            act_schs.add(w)
    return list(act_schs)

def get_objects(corpus):
    all_words = np.concatenate(
        [tokenize_plan(plan, with_wrapping=False)
        for plan in corpus])
    objects = set()
    for w in all_words:
        match = re.search(r"([A-Za-z]+)\d", w)
        if match:
            objects.add(match[1])
    return list(objects)


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
    obj_indices = {"actions": 0, "separators": 1}
    for i, word in enumerate(vocab):
        match_sep = re.search(r"[\(\)]", word)
        if match_sep:
            obj_types[i] = 1
        match = re.search(r"([A-Za-z]+)\d", word)
        if match:
            if match[1] not in obj_indices:
                obj_indices[match[1]] = len(obj_indices)
            obj_types[i] = obj_indices[match[1]]
    return obj_types, obj_indices


def tokenize_plan(plan, with_wrapping=True):
    begin_token = "("
    end_token = ")"
    wrap = lambda action: begin_token + " " + action + " " + end_token
    if with_wrapping:
        tokenized_plans = [wrap(action).split() for action in plan]
    else:
        tokenized_plans = [action.split() for action in plan]
    return np.concatenate(tokenized_plans).tolist()


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