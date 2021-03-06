import pickle
from datetime import datetime, timedelta

import pandas as pd
from prefect import flow, get_run_logger, task
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        # print(f"The mean duration of training is {mean_duration}")
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        # print(f"The mean duration of validation is {mean_duration}")
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task(name='train model')
def train_model(df, categorical):
    logger = get_run_logger()
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    # print(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")
    # print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    # print(f"The MSE of training is: {mse}")
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    # print(f"The MSE of validation is: {mse}")
    logger.info(f"The MSE of validation is: {mse}")
    return

@task
def get_paths(date):

    train_date, valid_date = date - timedelta(days=62),  date - timedelta(days=31)
    train_path, valid_path = f'./data/fhv_tripdata_{train_date.year}-0{train_date.month}.parquet', f'./data/fhv_tripdata_{valid_date.year}-0{valid_date.month}.parquet'

    return(train_path, valid_path)

@flow()
def main(date=None):
    if date is None:
        date = datetime.date.today()
    date = datetime.strptime(date, '%Y-%m-%d')

    train_path, val_path = get_paths(date).result()

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)
    with open(f'./models/model-{date.year}-0{date.month}-{date.day}.bin', 'wb') as f_out:
        pickle.dump(lr, f_out)
    with open(f'./models/dv-{date.year}-0{date.month}-{date.day}.b', 'wb') as f_out:
        pickle.dump(dv, f_out)    

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

DeploymentSpec(
    flow=main,
    name="model_training_homework",
    schedule=CronSchedule(cron='0 9 15 * *'),
    flow_runner=SubprocessFlowRunner(),
    tags=["mlops-zoomcamp", "homework"])


