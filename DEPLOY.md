## Train API🤖
- call /train
- structure of file.json: {"live_sid":[1,2,3], "post_message":["...","...","..."],"token":["...","...","..."],"medium":["...","...","..."],"ylabel":[1,0,0]}
```
{
  "json_link": "https://abc.s3.amazonaws.com/folder/file.json",
  "s3_link": "s3://bucket/folder/object"
}
```
- Response
```
{
  "hash": "hash_number"
}
```

## Predict API✂
- call /predict
```
{
  "d_data": {"live_sid":[1,2,3], "post_message":["...","...","..."],"token":["...","...","..."],"medium":["...","...","..."]},
  "model_file_name": "model_name",
  "model_file_hash": "hash",
  "s3_link": "s3://bucket/folder/object"
}
```
- Response
```
[{"live_sid":1, "pred":0}, {"live_sid":2, "pred":0}, {"live_sid":3, "pred":1}]
```

## Validation API🤹
- call /val
- structure of file.json: {"live_sid":[1,2,3], "post_message":["...","...","..."],"token":["...","...","..."],"medium":["...","...","..."],"ylabel":[1,0,0]}
```
{
  "t_data": "s3://bucket_t/folder_t/object_t",
  "v_data": "s3://bucket_v/folder_v/object_v",
  "model_file_name": "model_name",
  "model_file_name2": "model_name2",
  "model_file_hash": "hash",
  "s3_link": "s3://bucket/folder/object",
  "s3_link2": "s3://bucket_2/folder_2/object_2"
}
```
- Response
- 1 means update model, 0 means retrain model
- response for replacing model
```
{
  "hash": "hash_number",
  "result": 1  
}
```
- response not replacing model
```
{
  "hash": None,
  "result": 0
}
```

## start a docker host server
prerequisites:
- install docker desktop

start a docker host server
- in command line, type 
```
docker-compose up --build
```
- it needs time to start a docker server. Please wait patiently.
- http://0.0.0.0:8001/ping (GET)
- http://0.0.0.0:8001/train (POST)
- http://0.0.0.0:8001/predict (POST)
- http://0.0.0.0:8001/val (POST)

Stop a docker host server
- in command line, type
```
docker-compose down
```