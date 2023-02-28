# Databricks notebook source
# MAGIC  %run ./0.0_Set_up

# COMMAND ----------

import pandas as pd
from pandas_profiling import ProfileReport
import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

profile = ProfileReport(df, title="Report",minimal=True)
data = profile.to_html()
displayHTML(data)

# COMMAND ----------

df

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC df.employeeleft.value_counts().plot(kind='pie',
# MAGIC                                                 autopct='%1.0f%%',
# MAGIC                                                figsize=(8, 6))

# COMMAND ----------

df.groupby('employeeleft').mean()

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC corr = df.corr()
# MAGIC ax = sns.heatmap(
# MAGIC     corr, 
# MAGIC     vmin=-1, vmax=1, center=0,
# MAGIC     cmap=sns.diverging_palette(20, 220, n=200),
# MAGIC     square=True
# MAGIC )
# MAGIC ax.set_xticklabels(
# MAGIC     ax.get_xticklabels(),
# MAGIC     rotation=45,
# MAGIC     horizontalalignment='right'
# MAGIC );
# MAGIC plt.show()

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC pd.crosstab(df.university,df.employeeleft).plot(kind='bar')
# MAGIC plt.title('Turnover Frequency for Univesity')
# MAGIC plt.xlabel('Univesity')
# MAGIC plt.ylabel('Frequency of Turnover')

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC pd.crosstab(df.smtp_hired,df.employeeleft).plot(kind='bar')
# MAGIC plt.title('Turnover Frequency for smtp')
# MAGIC plt.xlabel('smtp')
# MAGIC plt.ylabel('Frequency of Turnover')

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC pd.crosstab(df.attitude_toward_peers,df.employeeleft).plot(kind='bar')
# MAGIC plt.title('Turnover Frequency for attitude_toward_peers')
# MAGIC plt.xlabel('survey__relative__attitude_toward_peers')
# MAGIC plt.ylabel('Frequency of Turnover')

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC pd.crosstab(df.environment,df.employeeleft).plot(kind='bar')
# MAGIC plt.title('Turnover Frequency for environment')
# MAGIC plt.xlabel('survey_toward_environment')
# MAGIC plt.ylabel('Frequency of Turnover')

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC pd.crosstab(df.resources,df.employeeleft).plot(kind='bar')
# MAGIC plt.title('Turnover Frequency for resources')
# MAGIC plt.xlabel('survey_resources')
# MAGIC plt.ylabel('Frequency of Turnover')

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC pd.crosstab(df.worktype,df.employeeleft).plot(kind='bar')
# MAGIC plt.title('Turnover Frequency for worktype')
# MAGIC plt.xlabel('survey_toward_worktype')
# MAGIC plt.ylabel('Frequency of Turnover')

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC pd.crosstab(df.workload,df.employeeleft).plot(kind='bar')
# MAGIC plt.title('Turnover Frequency for worktload')
# MAGIC plt.xlabel('survey_toward_workload')
# MAGIC plt.ylabel('Frequency of Turnover')

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC pd.crosstab(df.peers_review,df.employeeleft).plot(kind='bar')
# MAGIC plt.title('Turnover Frequency for peers_review')
# MAGIC plt.xlabel('survey_peers_review')
# MAGIC plt.ylabel('Frequency of Turnover')

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC pd.crosstab(df.attitude_toward_peers,df.employeeleft).plot(kind='bar')
# MAGIC plt.title('Turnover Frequency for attitude_toward_peers')
# MAGIC plt.xlabel('survey_attitude_toward_peers')
# MAGIC plt.ylabel('Frequency of Turnover')

# COMMAND ----------


