# usage:
# python3 test_model.py --model /Users/joshgardner/Documents/UM-Graduate/UMSI/LED_Lab/s17/model_build_infrastructure/job_runner/mod.pkl \
# --test_data /Users/joshgardner/Documents/UM-Graduate/UMSI/LED_Lab/s17/model_build_infrastructure/job_runner/1496931470-josh_gardner-clinicalskills-003/week_3/week_3_sum_feats.csv \

from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, cohen_kappa_score, confusion_matrix
import argparse
import pandas as pd
import numpy as np
import pickle
import os


def fetch_model_test_preds(mod_fp, data_fp, label_col = 'dropout_current_week'):
    # load model
    mod = joblib.load(mod_fp)
    # load training data
    data = pd.read_csv(data_fp).drop(['userID', 'week'], axis=1)
    # get X and Y from data
    x_cols = [c for c in data.columns if c != label_col]
    X = data[x_cols].as_matrix()
    Y = data[label_col].as_matrix()
    # predict on training data
    preds = mod.predict(X)
    # todo: apply log transformation when using xing models
    return (preds, Y)


def fetch_model_metrics(preds, labs):
    """
    Create a dictionary of model metrics
    :param preds: numpy array of predictions.
    :param labs: numpy array of (true) labels.
    :return: Python dictionary with model metrics.
    """
    output = {}
    output['accuracy'] = accuracy_score(labs, preds)
    output['precision'] = precision_score(labs, preds)
    output['recall'] = recall_score(labs, preds)
    output['auc'] = roc_auc_score(labs, preds)
    output['kappa'] = cohen_kappa_score(labs, preds)
    return output


def main(model, test_data, output_loc = './data/output'):
    preds, labs = fetch_model_test_preds(model, test_data)
    metrics = fetch_model_metrics(preds, labs)
    # write .csv of predictions and labels
    results_df = pd.DataFrame.from_dict({'pred':preds, 'label': labs})
    results_df.to_csv(os.path.join(output_loc, 'results.csv'), index=False, header=True)
    # write file of summary statistics
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index').reset_index()
    metrics_df.columns = ['metric', 'value']
    metrics_df.to_csv(os.path.join(output_loc, 'metrics.csv'), index=False, header=True)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Test predictive model on new data.')
    parser.add_argument('--model', required=True, help='path to saved .pkl model')
    parser.add_argument('--test_data', required=True, help='path to testing data')
    parser.add_argument('--output_loc', required=False, help='path to output directory for trained model',
                        default='./data/output')
    args = parser.parse_args()
    main(model=args.model, test_data=args.test_data, output_loc=args.output_loc)