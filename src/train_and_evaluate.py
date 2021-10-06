import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from get_data import read_params
import argparse
import joblib
import json

def eval_metrics(actual,predicted):
    rmse=mean_squared_error(actual,predicted)
    mae=mean_absolute_error(actual,predicted)
    r2=r2_score(actual,predicted)
    return rmse,mae,r2

def train_and_evaluate(config_path):
    config=read_params(config_path)
    test_data_path=config['split_data']['test_path']
    train_data_path=config['split_data']['train_path']
    random_state=config['base']['random_state']
    model_dir=config['model_dir']
    
    alpha=config['estimators']['ElasticNet']['params']['alpha']
    l1_ratio=config['estimators']['ElasticNet']['params']['l1_ratio']
    
    target=config['base']['target_col']
    
    train=pd.read_csv(train_data_path,sep=',')
    test=train=pd.read_csv(test_data_path,sep=',')
    
    y_train=train['TARGET']
    y_test=test['TARGET']

    x_train=train.drop(target,axis=1)
    x_test=test.drop(target,axis=1)
    
   
    lr=ElasticNet(alpha=alpha,l1_ratio=l1_ratio,random_state=random_state)
    lr.fit(x_train,y_train)
    
    predicted_qualities=lr.predict(x_test)
    y_test=np.array(y_test)
    y_test=y_test.reshape(len(y_test),1)
    predicted_qualities=predicted_qualities.reshape(len(y_test),1)
    (rmse,mae,r2) = eval_metrics(y_test,predicted_qualities)

    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    with open(scores_file, "w") as f:
        scores = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        json.dump(scores, f,indent=2)

    with open(params_file, "w") as f:
        params = {
            "alpha": alpha,
            "l1_ratio": l1_ratio,
        }
        json.dump(params, f,indent=2)



    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(lr, model_path)


if __name__ == '__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--config',default='params.yaml')
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)


