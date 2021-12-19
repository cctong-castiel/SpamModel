# SpamModel

## Importants for when to trigger the SpamModel
#### 1. Train and Validation Pipeline
- initially allow users to do tagging in an interface(total data size about 300)
- reached 66% of the total data size(300) in a specific task
- each spam or not spam label count must be >= 50

#### 2. Prediction Pipeline
- run by batch with max 300 records

#### 3. Re-train Pipeline
- when to send reminder
  - for sudden changing condition, event driven according to rate of change of total volume count(over 150%)
  - for constant condition, event driven according to whether number of records reach 1200 per week
- trigger requirements
  - or trigger by user. The trigger requirement is same as that in train pipeline

#### 4. Sentencepiece Model used
- after retrain a model, use the latest sentencepiece model and cut all previous records again

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


# SpamModel
