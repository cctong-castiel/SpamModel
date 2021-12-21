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