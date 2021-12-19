import logging
import json
import wget
import gc
import cloudpickle
import numpy as np
from os.path import join, basename, exists
from os import makedirs, remove

# pipeline
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, PolynomialFeatures
from imblearn.under_sampling import RandomUnderSampler
from scripts.selector import MSelector

from lib import aws_handler, ZipHandler
from lib.utils import get_digest
from lib import root, config, zip_type


def spam_train(json_link, s3_link):

    """It is a training pipeline to train a spam filtering model"""

    try:
        # variables
        hashword = None
        json_file_name = basename(json_link).split(".", 1)[0]
        model_file_name, model_folder = s3_link.split("/")[-1], s3_link.split("/")[-2]
        model_dir = join(root, model_folder)
        mdir = join(model_dir, model_file_name)

        # check if mdir exist
        if not exists(mdir):
            makedirs(mdir)
            logging.info("created mdir: %s" % mdir)

        # wget json file
        filename = wget.download(json_link, out=mdir)
        json_file = join(mdir, filename)
        d_data = json.load(open(json_file))
        
        # remove json file
        remove(json_file)

        # create model
        logging.info("train model")
        training(d_data, mdir, model_file_name, config)

        # zip model and vocan file
        logging.info("zip file")
        zip_handler = ZipHandler(model_file_name, zip_type, "")
        zip_handler.compressor(mdir, model_dir)

        # hash
        logging.info("hashing")
        hashword = get_digest(join(model_dir, model_file_name + zip_type))

        # upload to S3
        logging.info("upload to s3")
        local_path = join(model_dir, model_file_name + zip_type)
        s3_path = s3_link.split("/", 3)[-1]
        aws_handler.upload_2S3(s3_path, local_path)

        return {"hash": hashword}
    except Exception as e:
        logging.error(f"Error message: {e}")
        return json.dumps({"error": str(e)})

def train_pipe(d_data, **kwargs):

    max_num = kwargs["max_num"]; dimension = kwargs["dimension"]; sampler = kwargs["sampler"]; y_col = kwargs["y_col"]

    # var
    logging.info("var")
    arr_sent, arr_token, arr_medium, arr_ylabel = np.array(d_data["post_message"]), np.array(d_data["token"]), np.array(d_data["medium"]), np.array(d_data[y_col])
    cnt = arr_sent.size
    num_0, num_1 = np.unique(arr_ylabel)[0], np.unique(arr_ylabel)[1]
    cnt_0, cnt_1 = np.count_nonzero(arr_ylabel==num_0), np.count_nonzero(arr_ylabel==num_1)
    logging.info(f"label_0: {num_0} label_1: {num_1} cnt_0: {cnt_0} cnt_1: {cnt_1}")

    # generate X features
    logging.info("X train features")
    arr_token = arr_token.reshape((len(arr_token), 1))
    arr_medium = arr_medium.reshape((len(arr_medium), 1))
    arr_X = np.hstack(
        (arr_token, arr_medium)
    )

    # model selector
    logging.info("Selector")
    selector = MSelector(cnt, cnt_1, cnt_0)
    clf = selector.fit_transform

    # pipelines
    logging.info("create Pipeline")
    reshape_func = FunctionTransformer(lambda x: x.reshape(-1), validate=False)
    dense_func = FunctionTransformer(lambda x: x.toarray(), validate=False)

    pipe_num = Pipeline([
        ("onehot", OneHotEncoder(
            handle_unknown="ignore"
        )),
        ("poly", PolynomialFeatures(
            dimension
        ))
    ])

    pipe_text = Pipeline([
        ("reshape", reshape_func),
        ("vect", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_num
        )),
        ("dense", dense_func)
    ])

    if (cnt_0 / cnt_1 > 3.0) or (cnt_1 / cnt_0 > 3.0):

        pipe = Pipeline([
            ("sampler", RandomUnderSampler(
                sampling_strategy=0.5,
                random_state=1118
            )),
            ("preprocessor", ColumnTransformer(
                transformers = [
                    ("numeric_features", pipe_num, [1]),
                    ("text_features", pipe_text, [0])
                ]
            )),
            ("clf", clf)
        ])

    else:

        pipe = Pipeline([
            ("preprocessor", ColumnTransformer(
                transformers = [
                    ("numeric_features", pipe_num, [1]),
                    ("text_features", pipe_text, [0])
                ]
            )),
            ("clf", clf)
        ])

    logging.info("train pipe")
    pipe.fit(arr_X, arr_ylabel)

    return pipe

def training(d_data, model_path, model_file_name, **kwargs):

    """Objective: to train a fast train model and upload to s3
    input: An array of training data, model_path and model_file_name"""

    # train model
    logging.info("train pipeline model")
    l_component = train_pipe(d_data, **kwargs)

    logging.info("train finished")
    gc.collect()

    with open("{0}/{1}_pipe.pkl".format(model_path, model_file_name), "wb") as f:
        cloudpickle.dump(l_component, f)

    logging.info(f"finished training and model is saved as {model_path}/{model_file_name}_pipe.pkl")
