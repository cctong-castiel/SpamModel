import logging
import json
import gc
import cloudpickle
import multiprocessing as mp
from joblib import Parallel, delayed
from sklearn.utils import gen_batches
import numpy as np
from os.path import join, exists, isfile
from os import makedirs, chdir

from lib import aws_handler, ZipHandler
from lib.utils import get_digest
from lib import root, zip_type

def spam_predict(d_data, model_file_name, model_file_hash, s3_link):

    """It is a predicting pipeline to predict a spam filtering model"""

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

        # zip_handler
        local_path, s3_path = mdir + zip_type, s3_link.split("/", 3)[-1]
        zip_handler = ZipHandler(model_file_name, zip_type, "")
        zip_handler.compressor(mdir, model_dir)

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

        # apply models
        logging.info("model prediction")
        l_rets = next(predicting(d_data, mdir, model_file_name))

        for k in l_rets:
            del k["post_message"]

        # garbage collection
        logging.info("garbage collection")
        gc.collect()

        # chdir root
        logging.info("change root dir")
        chdir(root)

        return {"result": l_rets}
    except Exception as e:
        logging.error(f"Error message: {e}")
        return json.dumps({"error": str(e)})

def predicting(d_data, model_path, model_file_name):

    """Objective: to predict on unlabeled data
    input: An array of testing data, model_path and model_file_name"""

    # function
    def _predict(method, X, sl):
        return method(X[sl])[:,1]

    # var
    arr_sent, arr_id, arr_token, arr_medium = np.array(d_data["post_message"]), np.array(d_data["live_sid"]), np.array(d_data["token"]), np.array(d_data["medium"])
    logging.info(f"length of arr_sent is {len(arr_sent)}")

    # generate X test features
    logging.info("X test features")
    arr_token = arr_token.reshape((len(arr_token), 1))
    arr_medium = arr_medium.reshape((len(arr_medium), 1))
    arr_x = np.hstack(
        (arr_token, arr_medium)
    )

    # load pipeline components
    logging.info("load pipeline components")
    pipe = cloudpickle.load(open(f"{model_path}/{model_file_name}_pipe.pkl", "rb"))

    logging.info("start prediction")
    cpu = mp.cpu_count() - 1
    n_samples = len(arr_sent)
    batch_size = n_samples//cpu
    y_pred = Parallel(cpu)(delayed(_predict)(pipe.predict_proba, arr_x, sl) for sl in gen_batches(n_samples, batch_size))
    ret = np.concatenate(y_pred).ravel()
    logging.info(f"length of arr_id: {len(arr_id)}, arr_sent: {len(arr_sent)}, ret: {len(ret)}")

    gc.collect()

    # sort in ascending order
    logging.info("sorted array")
    idret = np.random.permutation(len(arr_id))
    arr_id, arr_sent, ret = arr_id[idret].tolist(), arr_sent[idret], ret[idret]
    logging.info(f"length of arr_id: {len(arr_id)}, ret: {len(ret)}")

    # final join the result
    logging.info("arr_results")
    l_results = [{"live_sid": i, "post_message": k, "pred": j} for i, k, j in zip(arr_id, arr_sent, ret)]

    yield l_results