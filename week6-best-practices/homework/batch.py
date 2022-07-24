#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd
import os


S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:4566")

with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)

def get_input_path(year, month):
    default_input_pattern = 's3://nyc-duration/in/{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration/out/{year:04d}-{month:02d}.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)

def read_data(input_path):

    if S3_ENDPOINT_URL is not None:
        options = {
            'client_kwargs': {
                'endpoint_url': S3_ENDPOINT_URL
            }
        }

        df = pd.read_parquet(input_path, storage_options=options)
    else:
        df = pd.read_parquet(input_path)

    return(df)

def save_data(data, output_path):

    if S3_ENDPOINT_URL is not None:
        options = {
            'client_kwargs': {
                'endpoint_url': S3_ENDPOINT_URL
            }
        }

        data.to_parquet(
            output_path,
            engine='pyarrow',
            compression=None,
            index=False,
            storage_options=options
        )
    else:
        data.to_parquet(
            output_path,
            engine='pyarrow',
            compression=None,
            index=False
        )

def prepare_data(df, categorical):
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def main(year, month):

    input_path = get_input_path(year, month)
    df = read_data(input_path)
    df = prepare_data(df, ['PUlocationID', 'DOlocationID'])
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


    dicts = df[['PUlocationID', 'DOlocationID']].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)


    print('predicted mean duration:', y_pred.mean())


    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    output_path= get_output_path(year, month)
    save_data(df, output_path)

if __name__ == "__main__":
    
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)
