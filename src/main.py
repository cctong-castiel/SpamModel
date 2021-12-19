from os import getenv
import logging
import numpy as np
from sanic import Sanic
from sanic.response import json as Json, text
from dotenv import load_dotenv
from lib.models.train import spam_train
from lib.models.pred import spam_predict
from lib.models.val import spam_val

# app
app = Sanic(__name__)

# load env
load_dotenv()

# logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename="./log/spam.log", filemode="a", level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# route
@app.route("/ping", methods=["GET"])
async def home(request):
    return text("IMU pong")

@app.route("/train", methods=["POST"])
async def runTrain(request):

    data = request.data
    json_link, s3_link = data["json_link"], data["s3_link"]

    logging.info("run train")
    result = spam_train(
        json_link, 
        s3_link
    )

    return Json(result)

@app.route("/predict", methods=["POST"])
async def runPred(request):

    data = request.data
    d_data = data["d_data"]
    model_file_name, model_file_hash = data["model_file_name"], data["model_file_hash"]
    s3_link = data["s3_link"]

    logging.info("run predict")
    result = spam_predict(
        d_data,
        model_file_name,
        model_file_hash,
        s3_link
    )

    return Json(result)

@app.route("/val", methods=["POST"])
async def runVal(request):

    data = request.data
    t_json, v_json = data["t_data"], data["v_json"]
    model_file_name, model_file_name2, model_file_hash = data["model_file_name"], data["model_file_name2"], data["model_file_hash"]
    s3_link, s3_link2 = data["s3_link"], data["s3_link2"]

    result = spam_val(
        t_json, v_json,
        model_file_name, model_file_name2,
        model_file_hash,
        s3_link, s3_link2
    )

    return Json(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", threaded=True, port=getenv("PORT"), debug=True)