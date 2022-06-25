import pickle
import pandas as pd
import sys
import os
import wget

with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


year = int(sys.argv[1])
month = int(sys.argv[2])

categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    if not os.path.isfile(filename):
        filename = wget.download(f'https://nyc-tlc.s3.amazonaws.com/trip data/fhv_tripdata_{year}-{month:02d}.parquet')
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    return df

df = read_data(f'data/fhv_tripdata_{year}-{month:02d}.parquet')


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)

print(y_pred.mean())

df_result = df[['ride_id']].copy()
df_result['Result'] = y_pred


output_file = 'df_result.csv'


df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)



