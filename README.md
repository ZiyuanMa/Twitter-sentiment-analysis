# Ensemble model for sentiment analysis in Twitter

This is a SFU cmpt413 nlp final project,the project is based on SemEval 2017 task 4, for the details about model please look at project.ipynb

## Files Distribution
data: dataset used for training and testing
models: code for two models 
output: output of testing data from each model
Twitter_analysis: module package that combine Twitter API and and our model to do real-time tweets sentiment analysis

## Data Processing
Read "project.ipynb" Data part to find where to download all of data, put all of data(there should be 14 files) in "data/raw_data", then run "clean_data.ipynb" in "data" folder to do data preprocessing. The output data should be in "data/modified_data".

## Models
In this file and our code, model 1 represents BiLSTM with attention, model 2 represents fine-tuned pre-trained model and model 3 represents ensemble model.

## Train Model
In "models" folder, run "model1.py" by "python3 model1.py" and use jupyter notebook to run "model2.ipynb" to train these two models. Our ensemble model is a combination of these two models, so we don't need to train it. model data is saved in "models/model_data"

## Test Model
In "models" folder, run "test_score.py" by "python3 test_score.py" would create output of models and save it to output folder, it would also print the test score of baseline and three models.

## Play with our Twitter analysis system
There's a example in "example.ipynb" shows how to use our system. To use this first copy "model1.pth", "model2.pkl" in "models/model_data" and "word_to_idx.pkl" in "data/modified_data" these three files into "Twitter_analysis/model_data", then our module package is ready to use.