# todo: consider exploring pgmpy for model construction http://pgmpy.org/ ; http://pgmpy.org/estimators.html#hill-climb-search ; might have better model-building (i.e., hill-climb search for structure)
# todo: is this running in python2?
# see libpgm documentation at:
#   https://github.com/CyberPoint/libpgm/blob/master/examples/examples.py
#   https://github.com/CyberPoint/libpgm/blob/master/examples/examples.py
# usage: python train_bn.py

import json
import sys
import pandas as pd
import numpy as np
from libpgm.nodedata import NodeData
from libpgm.graphskeleton import GraphSkeleton
from libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
from libpgm.lgbayesiannetwork import LGBayesianNetwork
from libpgm.hybayesiannetwork import HyBayesianNetwork
from libpgm.dyndiscbayesiannetwork import DynDiscBayesianNetwork
from libpgm.tablecpdfactorization import TableCPDFactorization
from libpgm.sampleaggregator import SampleAggregator
from libpgm.pgmlearner import PGMLearner
from sklearn.metrics import accuracy_score

TRAIN_DATA_FP = "/Users/joshgardner/Documents/UM-Graduate/UMSI/LED_Lab/s17/model_build_infrastructure/xing-feature-extractor/modeling/data/1496931175-josh_gardner-clinicalskills-002/week_3/week_3_sum_feats.csv"
TEST_DATA_FP = "/Users/joshgardner/Documents/UM-Graduate/UMSI/LED_Lab/s17/model_build_infrastructure/xing-feature-extractor/modeling/data/1497282771-josh_gardner-clinicalskills-003-extract/week_3/week_3_sum_feats.csv"

def read_and_preproc_data(fp, label = 'dropout_current_week'):
    """
    Read and preprocess data by applying np.log1p transormation as in [Xing 2016].
    :param fp: path to data (string).
    :param label: name of outcome column.
    :return: tuple of (X, Y) pd.DataFrames of predictors and response, respectively.
    """
    # load and preprocess data, return X and Y series
    df = pd.read_csv(fp).drop(['userID', 'week'], axis=1)
    X = df.drop(label, axis = 1).applymap(np.log1p)
    Y = df[label]
    return (X, Y)


def make_bn_input_data(X, Y):
    """
    Reformat X and Y vectors to create observation-level dictionaries as input for BN learner.
    :param X: pd.DataFrame of predictors from read_and_preproc_data().
    :param Y: pd.DataFrames and response from read_and_preproc_data().
    :return: list of dicts, one per observation, where entries in each dict represent variables.
    """
    df_in = pd.concat([X, Y], axis=1)
    data = df_in.to_dict('records')
    return data


def fetch_majority_label(fp, label = 'dropout_current_week'):
    df = pd.read_csv(fp)
    majority_label = df.groupby(label).size().idxmax()
    return majority_label


def train_bn(X, Y):
    """
    Trains bayesian network on X and Y vectors.
    :param X: pd.DataFrame of predictors from read_and_preproc_data().
    :param Y: pd.DataFrames of response from read_and_preproc_data().
    :return: libpgm.discretebayesiannetwork.DiscreteBayesianNetwork
    """
    data = make_bn_input_data(X=X, Y=Y)
    # instantiate learner
    learner = PGMLearner()
    # learn structure and parameters for discrete bn TODO: determine whether to use discrete or linear gaussian network...
    # todo: tuning of discrete_estimatebn
    bn = learner.discrete_estimatebn(data)
    # bn = learner.lg_estimatebn(data)
    return bn


def predict_dropout_prob(obs, bn, ml, preds_only):
    """
    Predict probability of dropout for a given observation.
    :param obs: dictionary entry, in format from make_bn_input_data.
    :param bn: libpgm.discretebayesiannetwork.DiscreteBayesianNetwork
    :param ml: majority class label, used for values not observed in training data.
    :param preds_only: boolean; return predicted label only or return dict of pred, lab, and prob.
    :return: prediction (if preds_only == True) or Python dictionary of {pred, lab, prob}.
    """
    #fetch ground truth label and remove
    lab = obs['dropout_current_week']
    # Compute the exact probability dropout
    query = {'dropout_current_week' : [1]}
    evidence = {k:v for k,v in obs.items() if k != 'dropout_current_week'}
    # compute using conditional probability; this is for discrete network
    # load factorization
    fn = TableCPDFactorization(bn)
    # calculate probability distribution
    try:
        prob = fn.specificquery(query, evidence)
        pred = 1 if prob > 0.5 else 0
    except:
        prob = np.nan
        pred = ml
    # # # compute using sampling; this is for linear gaussian network -- NOTE: there is some type of error here; predictions with this methods appear random...
    # # todo: can probably reduce number of samples; make sure this is correct sampling method
    # samp = bn.randomsample(n=1000, evidence = evidence)
    # prob = sum(x['dropout_current_week'] for x in samp)/len(samp)
    if preds_only == True:
        return pred
    else:
        return {'pred':pred, 'lab':lab, 'prob':prob}


def predict_bn(X, Y, bn, ml, preds_only = False):
    """
    Make predictions for observations from X and Y.
    :param X: pd.DataFrame of predictors from read_and_preproc_data().
    :param Y: pd.DataFrame of response from read_and_preproc_data().
    :param bn: libpgm.discretebayesiannetwork.DiscreteBayesianNetwork
    :param ml: majority class label.
    :param preds_only: boolean; return predicted label only or return dict of pred, lab, and prob.
    :return: list of results, with format conditional on preds_only.
    """
    data = make_bn_input_data(X=X, Y=Y)
    bad_obs = 0
    results = []
    for x in range(len(data)):
        try:
            prob = predict_dropout_prob(data[x], bn = bn, ml=ml, preds_only = preds_only)
            results.append(prob)
        except:
            bad_obs += 1
            print("Error predicting on test observation {}".format(x))
    return results


def main():
    print("training model")
    X,Y = read_and_preproc_data(fp = TRAIN_DATA_FP)
    bn = train_bn(X = X, Y = Y)
    ml = fetch_majority_label(TRAIN_DATA_FP)
    print("testing")
    X_test, Y_test = read_and_preproc_data(fp = TEST_DATA_FP)
    test_results = predict_bn(X_test, Y_test, bn = bn, ml = ml)
    preds = [x['pred'] for x in test_results]
    labs = [x['lab'] for x in test_results]
    acc = accuracy_score(labs, preds)


if __name__ == "__main__":
    main()