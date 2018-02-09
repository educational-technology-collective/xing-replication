import boto3
from morf.utils import *
import subprocess
import os
import shutil
import pandas as pd
import tempfile


def run_job(job_runner_url, docker_url, course, user_id, job_id, run, data_bucket, mode):
    ipy_cmd = "ipython3 {} -- --docker_url {} --course_id {} --user_id {} --job_id {} --run_number {} --raw_data_bucket {} --mode {} --level session".format(job_runner_url, docker_url, course, user_id, job_id, run, data_bucket, mode)
    print("running {}\n".format(ipy_cmd))
    subprocess.call(ipy_cmd, shell=True)
    return


def make_xing_train_test_data(s3, data_bucket, proc_data_bucket, user_id, job_id, course, data_dir='morf-data'):
    print("[INFO] Building training dataset for course {}".format(course))
    # create course_dir if doesn't exist; this is necessary if feature extraction was skipped
    course_dir = './{}/'.format(course)
    if not os.path.exists(course_dir):
        os.makedirs(course_dir)
    # get runs
    runs = fetch_runs(s3, data_bucket, data_dir, course)
    test_run = fetch_runs(s3, data_bucket, data_dir, course, fetch_holdout_run_only=True)
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as temp_data_dir:
        # download data for each run
        for run in runs + test_run:
            if run == test_run[0]:
                mode = 'extract-holdout'
            else:
                mode = 'extract'
            run_dir = '{}/{}'.format(temp_data_dir, run)
            if not os.path.exists(run_dir): os.makedirs(run_dir)
            # download and untar file for each run
            train_data_file = '{}-{}-{}-{}-{}.tgz'.format(user_id, job_id, mode, course, run)
            train_fp = 's3://{}/{}/{}/{}/{}/{}/{}'.format(proc_data_bucket, user_id, job_id, mode, course, run,
                                                          train_data_file)
            print("\t[INFO] Downloading data course {} run {} from {}".format(course, run, train_fp))
            train_tar_path = initialize_tar(train_fp, s3, dest_dir=run_dir)
            tar = tarfile.open(train_tar_path)
            tar.extractall(run_dir)
            tar.close()
            # TODO: remove run_dir if contents of tar are empty
            if len(os.listdir(run_dir)) <= 1:
                print('[WARNING] no extracted features for course {} run {}; skipping'.format(course, run))
        # check weeks
        weeks_list = []
        for run in runs + test_run:
            weeks = [re.search('week_([0-9]+)', x).group(1) for x in os.listdir(os.path.join(temp_data_dir, run)) if
                     'week_' in x]
            weeks_list.append(weeks)
        if not weeks_list[1:] == weeks_list[
                                 :-1]:  # if course has different numbers of weeks in different runs, use the minimum number of weeks
            print(
            "[WARNING] different weeks in iterations of course {}; setting to shortest number of weeks".format(course))
            weeks = min(weeks_list, key=len)
        # create test and train data: copy data for test_run to testdata.csv file; concatenate data for each train_run into single dataset and write into run_dir
        for eval_week in weeks:
            for feat_type in ['only', 'sum', 'appended']:
                week_out_dir = course_dir + 'week_{}/'.format(eval_week)
                if not os.path.exists(week_out_dir): os.makedirs(week_out_dir)
                # test data
                print(
                "[INFO] Writing test data for course {} week {} feat type {}".format(course, eval_week, feat_type))
                test_data_path = '{}/{}/week_{}/week_{}_{}_feats.csv'.format(temp_data_dir, test_run[0], eval_week, eval_week,
                                                                                 feat_type)
                test_dropout_path = '{}/{}/user_dropout_weeks.csv'.format(temp_data_dir, test_run[0])
                test_outpath = '{}week_{}_{}_testdata.csv'.format(week_out_dir, eval_week, feat_type)
                test_df = pd.read_csv(test_data_path)
                test_dropout_df = pd.read_csv(test_dropout_path)
                test_df = test_df.merge(test_dropout_df, how='inner', on='userID')
                test_df.to_csv(test_outpath, index=False, header=True)
                # train data
                train_df_list = []
                print(
                    "[INFO] Concatenating training data for course {} week {} feat type {}".format(course, eval_week,
                                                                                                   feat_type))
                for run in runs:
                    run_dir = '{}/{}/'.format(temp_data_dir, run)
                    train_data_path = '{}week_{}/week_{}_{}_feats.csv'.format(run_dir, eval_week, eval_week, feat_type)
                    run_df = pd.read_csv(train_data_path)
                    dropout_data_path = run_dir + 'user_dropout_weeks.csv'
                    dropout_df = pd.read_csv(dropout_data_path)
                    run_df = run_df.merge(dropout_df, how='inner', on='userID')
                    train_df_list.append(run_df)
                # write to output file in course_dir, remove data
                train_df = pd.concat(train_df_list, axis=0)
                train_outpath = '{}week_{}_{}_traindata.csv'.format(week_out_dir, eval_week, feat_type)
                train_df.to_csv(train_outpath, index=False, header=True)
    # compress and push to s3
    dest_file = '{}-{}-{}-{}.tgz'.format(user_id, job_id, 'train-test-data', course)
    cmd = 'tar -cvf {} -C {} .'.format(dest_file, course_dir)
    subprocess.call(cmd, shell=True, stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))
    bucket = proc_data_bucket
    key = '{}/{}/{}/{}/{}'.format(user_id, job_id, 'extract', course, dest_file)
    print('[INFO] pushing {} to s3://{}/{}'.format(dest_file, bucket, key))
    session = boto3.Session()
    s3_client = session.client('s3')
    tc = boto3.s3.transfer.TransferConfig()
    t = boto3.s3.transfer.S3Transfer(client=s3_client, config=tc)
    t.upload_file(dest_file, bucket, key)
    subprocess.call('rm {}'.format(dest_file), shell=True)
    shutil.rmtree(course_dir)
    return
