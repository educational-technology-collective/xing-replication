import argparse
import subprocess
from feature_extraction.xing_feature_extractor import main as extract_xing_features
from feature_extraction.sql_utils import extract_coursera_sql_data
import os
import re
import pandas as pd
from multiprocessing import Pool


def run_train_job(feat_type, week, course_id):
    """
    Create directories for output and run R script to train models (which will deposit trained models in out_dir).
    :param feat_type: one of {appended, only, sum} (string).
    :param week: week number (string).
    :param course_id: course id (string).
    :return: None.
    """
    out_dir = '/output/{}/{}/'.format(feat_type, week)
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    data_fp = '/input/week_{}/week_{}_{}_traindata.csv'.format(week, week, feat_type)
    out_fp = out_dir + '{}_{}_{}_xing_mod.Rdata'.format(course_id, feat_type, week)
    command = ['Rscript', './modeling/train_model_xing.R', '--file', data_fp, '--out', out_fp, '--week', week]
    res = subprocess.call(command)
    return res 


def run_test_job(feat_type, week, course_id):
    """
    Create directories for output and run R script to train models (which will deposit trained models in out_dir).
    :param feat_type: one of {appended, only, sum} (string).
    :param week: week number (string).
    :param course_id: course id (string).
    :return: None.
    """
    out_dir = '/output/{}/{}/'.format(feat_type, week)
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    data_fp = '/input/week_{}/week_{}_{}_testdata.csv'.format(week, week, feat_type)
    model_fp = '/input/{}/{}/{}_{}_{}_xing_mod.Rdata'.format(feat_type, week, course_id, feat_type, week)
    pred_fp = '/output/{}_{}_{}_test_preds.csv'.format(course_id, feat_type, week)
    summary_fp = '/output/{}_{}_{}_test_summary.csv'.format(course_id, feat_type, week)
    if os.path.exists(model_fp):
        print("[INFO] testing mode for week {} feat type {}".format(week, feat_type))
        command = ['Rscript', './modeling/test_model_xing.R', '--file', data_fp, '--model', model_fp, '--pred', pred_fp,
                   '--summary', summary_fp, '--week', week, '--feat_type', feat_type]
        res = subprocess.call(command)
    else:
        msg = "[WARNING] No trained model for course {} week {} feat_type {}; skipping.".format(args.course_id, week, feat_type)
        print(msg)
        res = msg
    return res


def create_master_test_summary_df(course_id):
    """
    aggregate all summary results into single file and remove individual files.
    :param course_id: course id (string)
    :return: None
    """
    summary_files = ['/output/{}'.format(f) for f in os.listdir('/output/') if '_test_summary.csv' in f]
    summary_df_list = [pd.read_csv(f) for f in summary_files]
    master_summary_df = pd.concat(summary_df_list, axis=0)
    for f in summary_files:
        os.remove(f)
    master_summary_fp = '/output/{}_test_summary.csv'.format(course_id)
    print("[INFO] writing model performance summary to {}".format(master_summary_fp))
    master_summary_df.to_csv(master_summary_fp, index=False, header=True)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='execute feature extraction, training, or testing.')
    parser.add_argument('-c', '--course_id', required=True, help='an s3 pointer to a course')
    parser.add_argument('-r', '--run_number', required=False, help='3-digit course run number')
    parser.add_argument('-m', '--mode', required=True, help='mode to run image in; {extract, train, test}')
    args = parser.parse_args()
    feat_types = ['only', 'appended', 'sum']
    # fetch weeks from filenames
    weeks = [re.search('week_([0-9]+)', x).group(1) for x in os.listdir('/input/') if 'week_' in x]
    weeks = list(sorted(set(weeks), key = lambda x: int(x)))
    if args.mode == 'extract':
        # setup the mysql database
        extract_coursera_sql_data(args.course_id, args.run_number)
        extract_xing_features(course_name = args.course_id, run_number = args.run_number)
    if args.mode == 'train':
        with Pool() as pool:
            for feat_type in feat_types:
                for week in weeks:
                    pool.apply_async(run_train_job, [feat_type, week, args.course_id])
            pool.close()
            pool.join()
    if args.mode == 'test':
        # don't predict on final week; by definition all remaining students drop out this week
        try:
            weeks.pop(-1)
        except IndexError:
            print("[WARNING] no course weeks detected for course {} run {}".format(args.course_id, args.run_number))
        with Pool() as pool:
            for feat_type in feat_types:
                for week in weeks:
                    pool.apply_async(run_test_job, [feat_type, week, args.course_id])
            pool.close()
            pool.join()
        create_master_test_summary_df(args.course_id)





