# usage:
# python3 train_model.py --train_data /Users/joshgardner/Documents/UM-Graduate/UMSI/LED_Lab/s17/model_build_infrastructure/job_runner/1496853720-josh_gardner-clinicalskills/week_3/week_3_sum_feats.csv --output_loc .

from sklearn import linear_model
import argparse
import pandas as pd
from sklearn.externals import joblib
import os


def train_model(data_fp, label_col = 'dropout_current_week'):
    # trains logistic regression from dataframe
    data = pd.read_csv(data_fp).drop(['userID', 'week'], axis = 1)
    # get X and Y from data
    x_cols = [c for c in data.columns if c != label_col]
    X = data[x_cols].as_matrix()
    Y = data[label_col].as_matrix()
    # train model
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X, Y)
    return logreg

def main(data_fp, output_loc = './data/output'):
    # reads data, trains model, and writes result to specified path
    # call train_model() to build
    mod = train_model(data_fp=data_fp)
    # write output; can be loaded with joblib.load()
    joblib.dump(mod, os.path.join(output_loc, 'mod.pkl'))

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='Train predictive model.')
    parser.add_argument('--train_data', required=True, help='path to training data')
    parser.add_argument('--output_loc', required=False, help='path to output directory for trained model',
                        default='./data/output')
    args = parser.parse_args()
    main(data_fp = args.train_data, output_fp = args.output_loc)