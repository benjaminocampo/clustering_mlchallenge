# %% [markdown]
# ## Imports
# %%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from nltk import FreqDist


def most_frequent_actions(plans, n):
    actions = np.concatenate(plans)
    counter = FreqDist(actions)
    return pd.DataFrame(counter.most_common(n), columns=["action", "count"])


def most_frequent_objects(plans, n):
    objects = np.concatenate(
        plans.apply(lambda plan:
                    np.concatenate([action.split()[1:]
                    for action in plan]))
    )
    counter = FreqDist(objects)
    return pd.DataFrame(counter.most_common(n), columns=["object", "count"])
# %%
TRAIN_URL = "https://www.famaf.unc.edu.ar/~nocampo043/training-instances.parquet.gzip"
TEST_URL = "https://www.famaf.unc.edu.ar/~nocampo043/evaluation-instances.parquet.gzip"

df_train = pd.read_parquet(TRAIN_URL)
df_test = pd.read_parquet(TEST_URL)
df_train["type_of_instance"] = "training"
df_test["type_of_instance"] = "testing"
df = pd.concat([df_train, df_test], ignore_index=True)
# %% [markdown]
# ## Common length of the plans
# %%
df["plan_length"] = df["relaxed_plan"].apply(lambda plan: len(plan))
plt.figure(figsize=(20,10))
sns.countplot(data=df, x="plan_length", hue="type_of_instance")
# %% [markdown]
# ## Frequent actions
# %%
plt.figure(figsize=(20, 10))
sns.barplot(data=most_frequent_actions(
    df.loc[df["type_of_instance"] == "training", "relaxed_plan"], 25),
            x="action",
            y="count")
plt.xticks(rotation=90)
# %%
plt.figure(figsize=(20,10))
sns.barplot(data=most_frequent_actions(df_test, 25), x="action", y="count")
plt.xticks(rotation=90)
# %% [markdown]
# ## Frequent objects
# %%
plt.figure(figsize=(20,10))
sns.barplot(data=most_frequent_objects(df_train, 25), x="object", y="count")
plt.xticks(rotation=90)
# %%
plt.figure(figsize=(20,10))
sns.barplot(data=most_frequent_objects(df_test, 25), x="object", y="count")
plt.xticks(rotation=90)
# %% [markdown]
# ## Good actions that are contained in relaxed plans
# %%
(
    df_train[["good_operators", "relaxed_plan"]]
        .apply(lambda row: sum(action in row[0] for action in row[1]), axis=1)
        .to_frame()
        .rename(columns={0: "good_ops_in_relaxed"})
        .join(df_train["relaxed_plan"].apply(lambda plan: len(plan)))
        .describe()
)

(
    df_test[["good_operators", "relaxed_plan"]]
        .apply(lambda row: sum(action in row[0] for action in row[1]), axis=1)
        .to_frame()
        .rename(columns={0: "good_ops_in_relaxed"})
        .join(df_test["relaxed_plan"].apply(lambda plan: len(plan)))
        .describe()
)
# %% [markdown]
# ##
