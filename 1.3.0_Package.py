# Databricks notebook source
# MAGIC  %run ./0.0_Set_up

# COMMAND ----------

import mlflow

# COMMAND ----------

eval_path = ""

# COMMAND ----------

run_id = "ab7daabc2a294b6c964b046e9827845a"
model = mlflow.sklearn.load_model(f'runs:/{run_id}/EDSP_Mitigated_LR')

class pyfunc_model(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    # Prediction function for the node model:
    def predict(self, context, df):
        

        predicted_df = df.copy()
        #change column types if necessary
        predicted_df.columns = [c.lower().replace(' ', '_') for c in predicted_df.columns]
        predicted_df.columns = [c.lower().replace(',', '_') for c in predicted_df.columns]
        predicted_df.columns = [c.lower().replace('\'', '_') for c in predicted_df.columns]
        predicted_df.rename(columns={'activity_on_company_forums': 'forum_activity', 'hired_through_smtp': 'smtp_hired','negative_review_in_past_5_years': 'negative_review','survey__relative__attitude_toward_peers': 'attitude_toward_peers','survey__relative__peer_s_average_review_of_employee': 'peers_review','national_origin_(code)':'nation'}, inplace=True)
        predicted_df.columns = predicted_df.columns.str.replace('survey__relative__peer_s_average_attitude_toward_', '')
        predicted_df['university'].astype('category')
        predicted_df['nation'].astype('category')

        
        #separating features & target
        y  = predicted_df['employeeleft']
        X = predicted_df.drop(['employeeleft'],axis=1)

        predicted_df['predictions']  = self.model.predict(X)
        
        return predicted_df

# COMMAND ----------

my_pyfunc = pyfunc_model(model)

# COMMAND ----------

## import data to test ##
df = pd.read_csv("/dbfs/FileStore/shared_uploads/remy.chen@neudesic.com/Sample_IssueDataset.csv")
df.head()

# COMMAND ----------

pred_df = my_pyfunc.predict(None, df)
pred_df.head()

# COMMAND ----------

import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env

environment = {
    "name": "mlflow-env",
    "channels": ["conda-forge"],
    "dependencies": [
        "python=3.9.5",
        "pip",
        {
            "pip": [
                    'mlflow == 2.1.1',
                    'cloudpickle==2.0.0',
                    'psutil==5.8.0',
                    'scikit-learn==1.0.2',
                    'typing-extensions==4.1.1',
                    'fairlearn == 0.8.0'
            ],
      
        },
    ],
}
print(environment)

# COMMAND ----------

experiment_name = 'EDSP_Sample_Experiment'
mlflow.set_experiment(f'/Shared/{experiment_name}')

with mlflow.start_run(run_id=run_id):
    signature = infer_signature(df, my_pyfunc.predict(None, df))
    mlflow.pyfunc.log_model('logi_test', python_model=my_pyfunc, input_example=df, conda_env=environment, signature=signature ,registered_model_name="EDSP_Mitigated_LR") 

# COMMAND ----------


