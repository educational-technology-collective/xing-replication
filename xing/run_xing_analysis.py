# script to extract/train/test across all courses

import boto3
from morf.utils import *
import subprocess
from multiprocessing import Pool
from xing_extract_utils import make_xing_train_test_data, run_job
from morf.utils.config import get_config_properties

data_dir = 'morf-data'
proc_data_bucket = get_config_properties()['proc_data_bucket']
docker_url = get_config_properties()['docker_url']
user_id = get_config_properties()['user_id']
job_id = get_config_properties()['job_id']
job_runner_url = get_config_properties()['job_runner_url']

#TODO: remove these later; used to control mode for development
EXTRACT = True
TRAIN = True
TEST = True

#TODO: single call to fetch_data_buckets_from_config() here to ensure all loops use same buckets
#raw_data_buckets = fetch_data_buckets_from_config()

s3 = boto3.client('s3', aws_access_key_id=get_config_properties()['aws_access_key_id'], aws_secret_access_key=get_config_properties()['aws_secret_access_key'])

# first iteration: feature extraction
# extract all 3 feature types (only, summed, appended) for each week of all courses/runs
if EXTRACT:
    proc_data_dir = '{}/{}/{}'.format(user_id, job_id, 'extract')
    raw_data_buckets = fetch_data_buckets_from_config()
    with Pool(processes = min(os.cpu_count(), int(get_config_properties()['max_threads']))) as pool:
        for data_bucket in raw_data_buckets:
            courses = fetch_complete_courses(s3, data_bucket, n_train=2)
            for course in courses:
                # extract for training runs
                train_runs = fetch_runs(s3, data_bucket, data_dir, course)
                test_run = fetch_runs(s3, data_bucket, data_dir, course, fetch_holdout_run_only=True)
                for run in train_runs + test_run:
                    if run == test_run[0]:
                        mode = 'extract-holdout'
                    else:
                        mode = 'extract'
                    pool.apply_async(run_job, [job_runner_url, docker_url, course, user_id, job_id, run, data_bucket, mode])
        pool.close()
        pool.join()
    ## from extracted data, create single dataset for model training and testing
    for data_bucket in raw_data_buckets:
        for course in fetch_complete_courses(s3, data_bucket, n_train=2):
            make_xing_train_test_data(s3, data_bucket, proc_data_bucket, user_id, job_id, course)

# second iteration: model training
if TRAIN:
    raw_data_buckets = fetch_data_buckets_from_config()
    for data_bucket in raw_data_buckets:
        for course in fetch_complete_courses(s3, data_bucket, n_train = 2):
            ipy_cmd = "ipython3 {} -- --docker_url {} --course_id {} --user_id {} --job_id {} --mode train --raw_data_bucket {}".format(job_runner_url, docker_url, course, user_id, job_id, data_bucket)
            print("[INFO] running {}".format(ipy_cmd))
            subprocess.call(ipy_cmd, shell=True)

# third iteration: model testing; this will only happen once per course
if TEST:
    all_courses = []
    raw_data_buckets = fetch_data_buckets_from_config()
    for data_bucket in raw_data_buckets:
        courses = fetch_complete_courses(s3, data_bucket, n_train = 2)
        all_courses += courses
        for course in courses:
            ipy_cmd = "ipython3 {} -- --docker_url {} --course_id {} --user_id {} --job_id {} --mode test --raw_data_bucket {}".format(job_runner_url, docker_url, course, user_id, job_id, data_bucket)
            print("[INFO] running {}".format(ipy_cmd))
            subprocess.call(ipy_cmd, shell=True)
    compile_test_results(s3 = s3, courses = all_courses, bucket = proc_data_bucket, user_id=user_id, job_id=job_id)

