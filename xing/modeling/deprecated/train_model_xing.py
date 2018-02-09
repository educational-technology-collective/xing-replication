# train model to replicate [Xing 2016]
# usage:
# python train_model_xing.py \
# --train_data ./data/1496931175-josh_gardner-clinicalskills-002/week_3/week_3_sum_feats.csv \
# --output_loc .

from sklearn import tree
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
import argparse
import pandas as pd
from sklearn.externals import joblib
import os
import math
import numpy as np
from train_bn import *


def preprocess_xing_data(data_fp, label_col, drop_cols = ['userID', 'week']):
    """
    Create preprocessed (log-transformed) X and Y vectors from input data.
    :param data_fp: path to data (string).
    :param label_col: name of column containing outcome (string)
    :param drop_cols: names of columns (if any) to drop from input data (list).
    :return: (X, Y) of pd.Dataframes.
    """
    data = pd.read_csv(data_fp).drop(drop_cols, axis=1)
    # get X and Y from data
    x_cols = [c for c in data.columns if c != label_col]
    X = data[x_cols].applymap(np.log1p)
    # apply log transformation to data
    Y = data[label_col]
    return (X,Y)


def train_tree(X, Y, max_tree_depth = 30):
    """
    Using 10-fold cross-validation, fit a CART tree to X and Y.
    :param X: pd.DataFrame of predictors, matching output from preprocess_xing_data().
    :param Y: pd.DataFrame of labels, matching output from preprocess_xing_data().
    :param max_tree_depth: max depth to consider; this controls complexity/model training time but in practice can probably be reduced far below 30.
    :return: sklearn DecisionTreeClassifier.
    """
    # tree model using log transform
    X_train = X.as_matrix()
    Y_train = Y.as_matrix()
    # train tree model using 10x10-fold CV
    parameters = {'max_depth': list(range(1, max_tree_depth))}  # TODO: verify that max_depth of 30 is reasonable search space (may want to allow for deeper models)
    mod = GridSearchCV(tree.DecisionTreeClassifier(), parameters, cv=10, n_jobs=10, refit=True)
    mod.fit(X, Y)
    best_mod = mod.best_estimator_
    print (mod.best_score_, mod.best_params_)
    return best_mod


def predict_tree(tree_mod, X):
    """
    Given a model and set of test cases, returns a vector of predictions.
    :param tree_mod: sklearn DecisionTreeClassifier.
    :param X: pd.DataFrame of predictors.
    :return: vector of predictions.
    """
    preds = tree_mod.predict(X)
    return preds


def make_oof_df(X, Y, n_splits=5):
    """
    Generate dataframe of out-of-fold predictions for tree and bayesian network classifiers.
    :param X: input pd.DataFrame of predictors.
    :param Y: input pd.DataFrame of response labels.
    :param n_splits: number of splits to use in cross-validation (k).
    :return: pd.DataFrame with columns for out-of-fold predictions (for bn and tree) and true label.
    """
    oof_df = pd.DataFrame()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X.iloc[train_index.tolist()], X.iloc[test_index.tolist()]
        Y_train, Y_test = Y.iloc[train_index.tolist()], Y[test_index.tolist()]
        # train base learners on in-fold data
        print("training tree")
        tree_mod = train_tree(X_train, Y_train)
        print("training bayesian network")
        bn_mod = train_bn(X_train, Y_train)
        ml = Y_train.value_counts().idxmax()
        # collect out-of-fold prediction for each model
        print("fetching tree preds")
        tree_preds = predict_tree(tree_mod, X_test)
        print("fetching bayesian network preds")
        bn_preds = predict_bn(X=X_test, Y=Y_test, bn=bn_mod, ml=ml, preds_only=True)
        # build dataframe from base model predictions and labels
        k_df = pd.DataFrame.from_dict({'bn_pred': bn_preds, 'tree_pred': tree_preds.tolist(), 'label': Y_test.tolist()})
        oof_df = pd.concat([oof_df, k_df], axis=0)
    return oof_df


def train_stacked_ensemble(X, Y):
    """
    Trains a simple stacked ensemble classifier from results of make_oof_df().
    :param X: input pd.DataFrame of predictors.
    :param Y: input pd.DataFrame of response labels.
    :return: sklearn.LogisticRegressionClassifier
    """
    mod_stack = linear_model.LogisticRegression()
    mod_stack.fit(X, Y)
    return mod_stack

def train_xing_model(data_fp, label_col = 'dropout_current_week'):
    """
    Trains stacked ensemble of decision tree and bayesian network to replicate (insofar as possible) [xing 2016] with logistic regression as level-1 learner.
    :param data_fp: path to data (string).
    :param label_col: column with target label of prediction.
    :return: Python dictionary of base learners (bayesian network, tree) and meta-learner (logistic regression).
    """
    X,Y = preprocess_xing_data(data_fp = data_fp, label_col=label_col)
    # create training folds using stratified sampling (preserves relative class balance in each of the K splits)
    oof_df = make_oof_df(X=X, Y=Y)
    # train "stacked" meta-learner using out-of-fold prediction from each model
    X_oof = oof_df.drop('label', axis = 1).as_matrix()
    Y_oof = oof_df['label'].as_matrix()
    mod_stack = train_stacked_ensemble(X_oof, Y_oof)
    # construct base models for test data, using full training dataset
    base_bn = train_bn(X,Y)
    base_tree = train_tree(X, Y)
    # return all models; predicting on new data requires 2-stage prediction process (first base, then meta-learner)
    mod_dict = {'base_bn': base_bn, 'base_tree': base_tree, 'stacked':mod_stack}
    return mod_dict


def main():
    # reads data, trains model, and writes result to specified path
    # parse args
    parser = argparse.ArgumentParser(description='Train predictive model.')
    parser.add_argument('--train_data', required=True, help='path to training data')
    parser.add_argument('--output_loc', required=False, help='path to output directory for trained model', default = './data/output')
    args = parser.parse_args()
    # call train_model() to build
    mod = train_xing_model(data_fp = args.train_data)
    #todo: save model and implement a prediction function
    # write output; can be loaded with joblib.load()
    joblib.dump(mod, os.path.join(args.output_loc, 'mod.pkl'))

if __name__ == '__main__':
    main()