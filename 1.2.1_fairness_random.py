# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ##Initial set up

# COMMAND ----------

# MAGIC  %run ./0.0_Set_up

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##Import Libraries

# COMMAND ----------

import copy
import numpy as np
import pandas as pd

import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env

import cloudpickle
import time
from collections import Counter
import shap

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##Define Function to get feature name from Sklearn pipeline 

# COMMAND ----------

def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """
    
    
    # Get columns and bin edges
    def pair_bin_edges_columns(bins_edges, columns):
        all_edge_column = []
        for index in range(len(columns)):
            bin_edge = bins_edges[index]
            column = columns[index]
            
            pair_bin_edge = []
            # match consecutive bin edges together
            for first, second in zip(bin_edge, bin_edge[1:]):
                pair_bin_edge.append((first, second))
            
            bin_edge_column = [column +": ["+ str(edge[0]) +", "+ str(edge[1])+")"  for edge in pair_bin_edge]
            all_edge_column.extend(bin_edge_column)
        return all_edge_column
    
    
    
    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
        # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                                 "provide get_feature_names. "
                                 "Will return input column names if available"
                                 % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            
            # For transformers that create bins
            if hasattr(trans, 'bin_edges_'):
                bins_edges = trans.bin_edges_
                pairs_edges_columns = pair_bin_edges_columns(bins_edges, column)
            if column is None:
                return []
            else:
                return [name + "__" + f for f in pairs_edges_columns]
        return [name + "__" + f for f in trans.get_feature_names(column)]
    
    ### Start of processing
    feature_names = []
    
    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))
    
    for name, trans, column, _ in l_transformers:
        if type(trans) == Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names)==0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))
    
    return feature_names

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Convert category columns

# COMMAND ----------

for col in ['university','nation']:
    df[col] = df[col].astype('category')

# COMMAND ----------

# MAGIC %md
# MAGIC ##Split Train/Test dataset

# COMMAND ----------

y  = df['employeeleft']
X = df.drop(['employeeleft'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# COMMAND ----------

#Set Experiment Name 
experiment_name = 'EDSP_Sample_Experiment'
mlflow.set_experiment(f'/Shared/{experiment_name}')

# COMMAND ----------

def fetch_logged_data(run_id):
    client = MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts


# COMMAND ----------

# MAGIC %md 
# MAGIC ##Build preprocessor pipeline

# COMMAND ----------

numeric_transformer = Pipeline(
    steps=[
        ("impute", SimpleImputer()),
        ("scaler", StandardScaler()),
    ]
)
categorical_transformer = Pipeline(
    [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, selector(dtype_exclude="category")),
        ("cat", categorical_transformer, selector(dtype_include="category")),
    ]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Construct Logistic regression pipeline

# COMMAND ----------

rf_predictor = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(),
        ),
    ]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Fit model

# COMMAND ----------

# enable autologging
mlflow.sklearn.autolog()

with mlflow.start_run() as run:
    
    params = {
          "eval_metric":"auc"
            }
    
    rf_predictor.fit(X_train, y_train)
    
    mlflow.sklearn.log_model(rf_predictor, 'EDSP_rf_predictor')

    
    mlflow.log_params(params)
    
    auc = roc_auc_score(y_train,rf_predictor.predict(X_train))
    print("Train AUC:",auc)
    auc = roc_auc_score(y_test,rf_predictor.predict(X_test))
    print("Test AUC:",auc)
    
    
mlflow.end_run()

# COMMAND ----------

rf_predictor.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Get feature names from preprocessor

# COMMAND ----------

import warnings
feature_names = get_feature_names(preprocessor)
feature_names

# COMMAND ----------

# MAGIC %md
# MAGIC ##Assign column name back to transformed data

# COMMAND ----------

Train = rf_predictor.named_steps['preprocessor'].fit_transform(X_train)
Test = rf_predictor.named_steps['preprocessor'].fit_transform(X_test)
X_tran = pd.DataFrame(Train, columns= feature_names)
X_tes = pd.DataFrame(Test, columns= feature_names)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Confusion matrix

# COMMAND ----------

from sklearn.metrics import classification_report
print(classification_report(y_test, rf_predictor.predict(X_test)))

# COMMAND ----------

# MAGIC %md
# MAGIC # Evaluating bias using fairlearn library 

# COMMAND ----------

# MAGIC %md
# MAGIC ## visualize selection rate on different nation origin

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC from fairlearn.metrics import selection_rate, MetricFrame
# MAGIC 
# MAGIC 
# MAGIC selection_rates = MetricFrame(
# MAGIC     metrics=selection_rate, y_true=y_test, y_pred=rf_predictor.predict(X_test), sensitive_features=X_test['nation']
# MAGIC )
# MAGIC 
# MAGIC fig = selection_rates.by_group.plot.bar(
# MAGIC     legend=False, rot=0, title="Fraction of Employees Predicted to Resign per National Origin")
# MAGIC plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize more factors by national origin

# COMMAND ----------

from fairlearn.metrics import (
    MetricFrame,
    count,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
)
from sklearn.metrics import accuracy_score, precision_score
# Analyze metrics using MetricFrame
metrics = {
    "Accuracy": accuracy_score,
    "Precision": precision_score,
    "False Positive Rate (FPR)": false_positive_rate,
    "False Negative Rate (FNR)": false_negative_rate,
    "Selection Rate": selection_rate,
    "Count of Employees": count,
}
metric_frame = MetricFrame(
    metrics=metrics, y_true=y_test, y_pred=rf_predictor.predict(X_test), sensitive_features=X_test['nation']
)
metric_frame.by_group.plot.bar(
    subplots=True,
    layout=[3, 3],
    legend=False,
    figsize=[12, 11],
    title="Fairness Unaware Predictor",
)
plt.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ##Calculate shap value for each feature

# COMMAND ----------

# %matplotlib inline
# explainer = shap.Explainer(rf_predictor.named_steps['classifier'].predict_proba, X_tran)
# # Calculates the SHAP values 
# shap_values = explainer(X_tes)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Visualize shap values by importance

# COMMAND ----------

# shap.plots.bar(shap_values[:, :, 1],max_display=10)
# plt.show()

# COMMAND ----------

# %matplotlib inline
# shap.plots.beeswarm(shap_values[:, :, 1],max_display=15)
# plt.show()

# COMMAND ----------

from fairlearn.reductions import DemographicParity, ExponentiatedGradient, EqualizedOdds
np.random.seed(0)  # set seed for consistent results with ExponentiatedGradient


# COMMAND ----------

# MAGIC %md
# MAGIC #Correcting bias

# COMMAND ----------

# MAGIC %md
# MAGIC ##Demographic parity: the prediction should be independent from the sensitive features (for instance independent from gender). It states that all categories from the protected feature should receive the positive outcome at the same rate (it plays on selection rate)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Fit ExponentiatedGradient model to minimize selection rate on different groups 

# COMMAND ----------

exponentiated_gradientMitigatedrf_model = ExponentiatedGradient(
    estimator=rf_predictor,
    constraints=DemographicParity(),
    sample_weight_name="classifier__sample_weight",
)

# COMMAND ----------

Mitigatedrf_model = ExponentiatedGradient(
    estimator=rf_predictor,
    constraints=DemographicParity(),
    sample_weight_name="classifier__sample_weight",
)
Mitigatedrf_model.fit(X_train, y_train, sensitive_features=X_train['nation'])
# print(Mitigatedlr_model.predict(X_test))

# COMMAND ----------

# mlflow.sklearn.autolog()
# with mlflow.start_run() as run:
    
#     params = {
#           "eval_metric":"auc"
#             }
    


#     Mitigatedrf_model.fit(X_train, y_train, sensitive_features=X_train['nation'])
    
#     mlflow.sklearn.log_model(Mitigatedrf_model, 'EDSP_Mitigated_RF')
#     mlflow.log_params(params)
    
    
# mlflow.end_run()

# COMMAND ----------

pred = Mitigatedrf_model.predict(X_test)
pred

# COMMAND ----------

# MAGIC %md 
# MAGIC ##confusion matrix for mitigated model

# COMMAND ----------

print(classification_report(y_test, Mitigatedrf_model.predict(X_test)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Selection rate on mitigated model

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC selection_rates = MetricFrame(
# MAGIC     metrics=selection_rate, y_true=y_test, y_pred=Mitigatedrf_model.predict(X_test), sensitive_features=X_test['nation']
# MAGIC )
# MAGIC 
# MAGIC fig = selection_rates.by_group.plot.bar(
# MAGIC     legend=False, rot=0, title="Fraction of Employees Predicted to Resign per National Origin")
# MAGIC plt.show()

# COMMAND ----------

# Analyze metrics using MetricFrame
metrics = {
    "Accuracy": accuracy_score,
    "Precision": precision_score,
    "False Positive Rate (FPR)": false_positive_rate,
    "False Negative Rate (FNR)": false_negative_rate,
    "Selection Rate": selection_rate,
    "Count of Employees": count,
}
metric_frame = MetricFrame(
    metrics=metrics, y_true=y_test, y_pred=Mitigatedrf_model.predict(X_test), sensitive_features=X_test['nation']
)
metric_frame.by_group.plot.bar(
    subplots=True,
    layout=[3, 3],
    legend=False,
    figsize=[12, 11],
    title="Fairness aware Predictor",
)
plt.show()

# COMMAND ----------


