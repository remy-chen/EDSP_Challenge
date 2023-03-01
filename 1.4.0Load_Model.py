# Databricks notebook source
# MAGIC %pip install fairlearn

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json

def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
    ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    url = 'https://adb-2893114107964037.17.azuredatabricks.net/model/EDSP_Mitigated_LR/13/invocations'
    headers = {'Authorization': f'Bearer {ctx.apiToken().get()}', 'Content-Type': 'application/json'}
    ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()

# COMMAND ----------

df = pd.read_csv("/dbfs/FileStore/shared_uploads/remy.chen@neudesic.com/Sample_HoldoutDataset.csv")

# COMMAND ----------

df

# COMMAND ----------

predictions = score_model(df)#['predictions']
#predictions_df = pd.DataFrame(predictions)
#predictions_df.head()


# COMMAND ----------

predictdf=pd.json_normalize(predictions,record_path =['predictions'])
predictdf['employeeleft']=predictdf['employeeleft'].astype(int)
predictdf

# COMMAND ----------

from sklearn.metrics import roc_auc_score, classification_report

pred = predictdf['predictions']
actu = predictdf['employeeleft']

print("\nClassification Report:\n", classification_report(actu, pred))

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC from fairlearn.metrics import selection_rate, MetricFrame
# MAGIC 
# MAGIC 
# MAGIC selection_rates = MetricFrame(
# MAGIC     metrics=selection_rate, y_true=actu, y_pred=pred, sensitive_features=predictdf['nation']
# MAGIC )
# MAGIC 
# MAGIC fig = selection_rates.by_group.plot.bar(
# MAGIC     legend=False, rot=0, title="Fraction of Employees Predicted to Resign per National Origin")

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
    metrics=metrics, y_true=actu, y_pred=pred, sensitive_features=predictdf['nation']
)
metric_frame.by_group.plot.bar(
    subplots=True,
    layout=[3, 3],
    legend=False,
    figsize=[12, 11],
    title="Fairness Unaware Predictor",
)

# COMMAND ----------


