import json
from os import getcwd, getenv
from os.path import join
from lib.handlers.awshandler import AWSHandler
from lib.handlers.ziphandler import ZipHandler

# root
root = getcwd()

# get config value
config = json.load(open(join(root, "config/config.json")))
zip_type = config.get("zip_type")

# load aws env
accessKey = getenv("AWS_ACCESS")
secretKey = getenv("AWS_SECRET")
region = getenv("AWS_REGION")
bucket = getenv("AWS_BUCKET")

# create ZipHandler and AWSHandler objects
aws_handler = AWSHandler(accessKey, secretKey, region, bucket)