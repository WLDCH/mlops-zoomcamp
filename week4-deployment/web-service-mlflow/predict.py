import pickle
from flask import Flask, request, jsonify

import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
RUN_ID = '0c1a244d996943d786c5efc7c232e571'
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

logged_model = f'runs:/{RUN_ID}/model'
model = mlflow.pyfunc.load_model(logged_model)

# path = client.download_artifacts(run_id=RUN_ID, path='dict_vectorizer.bin')
# print(f'downloading the dict vectorizer to {path}')

# with open(path, 'rb') as f_out:
#     dv = pickle.load(f_out)

def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features


def predict(features):
    # X = dv.transform(features)
    preds = model.predict(features)
    return(preds[0])


app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred,
        'model_version': RUN_ID
    }

    return(jsonify(result))

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)