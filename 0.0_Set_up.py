# Databricks notebook source
# MAGIC %pip install pandas_profiling matplotlib scikit-learn cloudpickle imbalanced-learn shap fairlearn

# COMMAND ----------

import pandas as pd
# from pandas_profiling import ProfileReport
# import seaborn as sns
# import matplotlib.pyplot as plt
# %matplotlib inline
# color = sns.color_palette()
# import mlflow
# import mlflow.pyfunc
# import mlflow.sklearn
# import sklearn
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# from sklearn.metrics import accuracy_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn import model_selection
# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import precision_recall_curve
# from mlflow.models.signature import infer_signature
# from mlflow.utils.environment import _mlflow_conda_env
# import cloudpickle
# import time
# import numpy as np
# from collections import Counter
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve
# import shap

# COMMAND ----------

df = pd.read_csv("/dbfs/FileStore/shared_uploads/remy.chen@neudesic.com/Sample_IssueDataset.csv")
df.columns = [c.lower().replace(' ', '_') for c in df.columns]
df.columns = [c.lower().replace(',', '_') for c in df.columns]
df.columns = [c.lower().replace('\'', '_') for c in df.columns]

# COMMAND ----------

df.rename(columns={'activity_on_company_forums': 'forum_activity', 'hired_through_smtp': 'smtp_hired','negative_review_in_past_5_years': 'negative_review','survey__relative__attitude_toward_peers': 'attitude_toward_peers','survey__relative__peer_s_average_review_of_employee': 'peers_review','national_origin_(code)':'nation'}, inplace=True)
df.columns = df.columns.str.replace('survey__relative__peer_s_average_attitude_toward_', '')
df.columns.values
