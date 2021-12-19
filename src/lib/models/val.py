import logging
import wget
import json
import gc
import cloudpickle
import numpy as np
import multiprocessing as mp
from joblib import Parallel, delayed
from sklearn.utils import gen_batches
from os.path import join, exists, isfile
from os import makedirs, remove, chdir, system

from lib import aws_handler, ZipHandler
from lib.models.train import train_pipe
from lib.models.metric import Metrics
from lib.utils import get_digest
from lib import root, config, zip_type


def spam_val(t_json, v_json, model_file_name, model_file_name2,
             model_file_hash, s3_link, s3_link2):

    """It is a validating pipeline to validate a spam filtering model"""

    try: 
        # variables
        logging.info("get variables")
        model_folder, bucket = s3_link.split("/")[-2], s3_link.split("/")[-3]
        model_dir = join(root, model_folder)
        mdir = join(model_dir, model_file_name)

        # check if mdir exist
        if not exists(mdir):
            makedirs(mdir)
            logging.info("created mdir: %s" % mdir)

        l_data = [None] * 2
        for index, j in enumerate([t_json, v_json]):
            filename = wget.download(j, out=mdir)
            json_file = join(mdir, filename)
            l_data[index] = json.load(open(json_file))
            remove(json_file)

        # zip_handler
        local_path, s3_path = mdir + zip_type, s3_link.split("/", 3)[-1]
        zip_handler = ZipHandler(model_file_name, zip_type, "")

        # check if files exist
        mfile_path = join(model_dir, model_file_name + zip_type)
        if not isfile(mfile_path):
            logging.info("download Spam model from s3 if model file not exist")
            aws_handler.download_fromS3(s3_path, local_path)
        else:
            hashword = get_digest(mfile_path)
            logging.info(f"hashword is: {hashword}")
            if hashword != model_file_hash:
                logging.info(f"spam model hash not match")
                aws_handler.download_fromS3(s3_path, local_path)

        # unzip Spam model
        logging.info("unzip Spam model")
        zip_handler.decompressor(model_dir, model_dir)

        # training a new Spam model
        logging.info("validating")
        return_metric = validating(l_data[0], l_data[1], mdir, model_file_name, config)
        metric_flag, l_component = return_metric[0], return_metric[1]
        if metric_flag == 0:

            # garbage collection
            gc.collect()

            # chdir root
            chdir(root)

            return {"hash": None, "result": metric_flag}

        # save new model files
        model_folder2 = s3_link2.split("/")[-2]
        model_dir2 = join(root, model_folder2)
        mdir2 = join(model_dir2, model_file_name2)
        local_path2 = join(model_dir2, model_file_name2 + zip_type)
        s3_path2 = s3_link2.split("/", 3)[-1]

        if not exists(mdir2):
            makedirs(mdir2)
            logging.info(f"Created mdir path with path {mdir2}")

        with open("{0}/{1}_pipe.pkl".format(mdir2, model_file_name2), "wb") as f:
            cloudpickle.dump(l_component, f)
        system(f"rm -r {mdir} && rm {mdir}{zip_type}")

        # zip model files
        logging.info("zip file")
        zip_handler2 = ZipHandler(model_file_name2, zip_type, "")
        zip_handler2.compressor(mdir2, model_dir2)

        # hash
        logging.info("hashing")
        hashword = get_digest(join(model_dir2, model_file_name2 + zip_type))
        logging.info(f"The directory before hash: {model_file_name2}{zip_type}, hashword is: {hashword}")

        # upload to s3
        logging.info("upload to s3")
        aws_handler.upload_2S3(s3_path2, local_path2)

        # garbage collection
        gc.collect()

        # chdir root
        chdir(root)

        return {"hash": hashword, "result": metric_flag}
    except Exception as e:
        logging.error(f"Error message: {e}")
        return json.dumps({"error": str(e)})

def validating(d_data, v_data, model_path, model_file_name, **kwargs):

    """Objective: to train a spam model and validating v_data
    input: An array of training data, v_data, model_path and model_file_name"""

    # function
    def _predict(method, X, sl):
        return method(X[sl])

    arr_sent, arr_token, arr_medium, arr_ytrue = np.array(d_data["post_message"]), np.array(d_data["token"]), np.array(d_data["medium"]), np.array(d_data[y_col])

    # train model
    logging.info("train pipeline model")
    pipe1 = train_pipe(v_data, **kwargs)

    # generate X test features
    logging.info("X test features")
    arr_token = arr_token.reshape((len(arr_token), 1))
    arr_medium = arr_medium.reshape((len(arr_medium), 1))
    arr_x = np.hstack(
        (arr_token, arr_medium)
    )

    # multiprocessing
    logging.info("start prediction")
    cpu = mp.cpu_count() - 1
    n_samples = len(arr_sent)
    batch_size = n_samples//cpu
    y_pred = Parallel(cpu)(delayed(_predict)(pipe1.predict, arr_x, sl) for sl in gen_batches(n_samples, batch_size))
    ret1 = np.concatenate(y_pred).ravel()

    # load old model and do prediction
    logging.info("prediction using old model")
    pipe2 = cloudpickle.load(open(f"{model_path}/{model_file_name}_pipe.pkl", "rb"))
    y_pred2 = Parallel(cpu)(delayed(_predict)(pipe2.predict, arr_x, sl) for sl in gen_batches(n_samples, batch_size))
    ret2 = np.concatenate(y_pred2).ravel()
    gc.collect()

    # Metrics
    logging.info("calculate f1 score")
    metric1 = Metrics(arr_ytrue, ret1)
    metric2 = Metrics(arr_ytrue, ret2)

    logging.info("fit Metrics")
    cm1 = metric1.confusion_matrix
    cm2 = metric2.confusion_matrix
    logging.info(f"cm1 is: \n {cm1}")
    logging.info(f"cm2 is: \n {cm2}")

    logging.info("metric fit")
    precision1, recall1 = metric1.fit(cm1)
    precision2, recall2 = metric2.fit(cm2)

    if kwargs["m_flag"] == "macro":
        logging.info("macro")
        f1_1 = metric1.f1(arr_ytrue, precision1, recall1)
        f1_2 = metric2.f1(arr_ytrue, precision2, recall2)
        logging.info(f"macro f1:\n new: {f1_1}, old: {f1_2}")
    else:
        logging.info("weighted")
        f1_1 = metric1.f1(arr_ytrue, precision1, recall1, "weighted")
        f1_2 = metric2.f1(arr_ytrue, precision2, recall2, "weighted")
        logging.info(f"weighted f1:\n new: {f1_1}, old: {f1_2}")

    if f1_1 > f1_2:
        return (1, pipe1)
    else:
        return (0, None)

