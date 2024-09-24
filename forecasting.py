import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
import mlflow
import mlflow.pyfunc

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def prep_store_data(df: pd.DataFrame, store_id: int = 4, store_open: int = 1) -> pd.DataFrame:
    df['Date'] = pd.to_datetime(df['Date'])
    df.rename(columns= {'Date': 'ds', 'Sales': 'y'}, inplace=True)
    df_store = df[
        (df['Store'] == store_id) &\
        (df['Open'] == store_open)
    ].reset_index(drop=True)
    return df_store.sort_values('ds', ascending=True) 
    
    
class ProphetWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        super().__init__()

    def load_context(self, context):
        from prophet import Prophet

        return

    def predict(self, context, model_input):
        future = self.model.make_future_dataframe(periods=model_input["periods"][0])
        return self.model.predict(future)


seasonality = {
    'yearly': True,
    'weekly': True,
    'daily': True
}

def train_predict(df_all_data, df_all_train_index, seasonality_params=seasonality):
    # grab split data
    df_train = df_all_data.copy().iloc[0:df_all_train_index]
    df_test = df_all_data.copy().iloc[df_all_train_index:]

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    #mlflow.set_experiment("check-localhost-connection")

    # Useful for multiple runs (only doing one run in this sample notebook)
    with mlflow.start_run():
        # create Prophet model
        model = Prophet(
            yearly_seasonality=seasonality_params['yearly'],
            weekly_seasonality=seasonality_params['weekly'],
            daily_seasonality=seasonality_params['daily']
        )
        # train and predict
        model.fit(df_train)

        # Evaluate Metrics
        df_cv = cross_validation(model, initial="366 days", period="180 days", horizon="365 days")
        df_p = performance_metrics(df_cv)

        # Print out metrics
        print("  CV: \n%s" % df_cv.head())
        print("  Perf: \n%s" % df_p.head())

        # Log parameter, metrics, and model to MLflow
        mlflow.log_metric("rmse", df_p.loc[0, "rmse"])

        mlflow.pyfunc.log_model("model", python_model=ProphetWrapper(model))
        print(
            "Logged model with URI: runs:/{run_id}/model".format(
                run_id=mlflow.active_run().info.run_id
            )
        )

    predicted = model.predict(df_test)
    return predicted, df_train, df_test


if __name__ == "__main__":
    # Read in Data
    df = pd.read_csv('train_data.csv')
    df = prep_store_data(df)
    df.rename(columns={'Date': 'ds', 'Sales': 'y'}, inplace=True)

    train_index = int(0.8 * df.shape[0])
    train_predict(
        df_all_data=df,
        df_all_train_index=train_index,
        seasonality_params=seasonality
    )
