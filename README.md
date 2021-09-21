# Disaster Response Pipeline Project

## Table of Contents
* [Project Overview](#project-overview)
* [Dependencies](#dependencies)
* [Data and Metrics](#data-and-metrics)
* [Screenshots](#screenshots)

## Project Overview

The Disaster Response Pipeline classifies messages received during humanitarian disasters into relevant categories in order to facilitate rapid response from responders. Different stages of the pipeline clean and tokenize messages and train the classifier. A web interface allows a user to interact with the application.

## Dependencies
A list of dependencies may be found in the [pyproject.toml](https://github.com/sunnykan/disaster_response_pipeline_project/blob/main/pyproject.toml?raw=true) file.

## Usage
* ETL: [process_data.py](https://github.com/sunnykan/disaster_response_pipeline_project/blob/main/data/process_data.py?raw=True) reads and cleans the data which is then inserted into a sqlite database. Run ```python data/process_data.py -h``` for instructions.
* Classification: The classifier [train_classifier.py](https://github.com/sunnykan/disaster_response_pipeline_project/blob/main/models/train_classifier.py?raw=True) pulls the data from the database and feeds it through a pipeline that processes the messages into tokens and classifies them. Different classifiers may be used in the pipeline. Run ```python models/train_classifier.py -h``` for instructions.
* Flask application: The app allows a user enter a message to have it classified into relevant categories. To run it locally, clone the repository, open the ```app``` folder and type ```python run.py```. 
* Deployment on Heroku: A modified version of the app using a Logistic regression can be found on [Heroku](https://disaster-response-ks.herokuapp.com/).

## Data and Metrics
* Imbalance: Most of the categories in the target are severely imbalanced. To address the issues, all classifiers were run with the parameter  ```class_weight``` set to ```'balanced'```. Counts of the target variable are used to automatically adjust weights inversely proportional to class frequencies in the categories. For more information, see the relevant documentation for classifiers in the scikit-learn package.
* Assessing model performance: Accuracy is not a useful metric given the severe imbalance in the target categories. A case may be made for preferring either precision or recall. Given the need to act quickly in emergencies and other constraints such as money and transportation, one may be more interested in precision. At the same time, higher recall may cause less harm. The threshold in the decision function can be changed to favour precision or recall. The f1-score which is a harmonic mean of precision and recall was used as the scoring function in the grid search. ```model_results.json``` contains [metrics](https://github.com/sunnykan/disaster_response_pipeline_project/blob/main/models/model_results.json?raw=true) from the classification report generated from the best model selected by grid search. 

## Screenshots
* Main page with some graphs of summary statistics of the training data. 
![Graph 1 of summary statistics of training data](https://github.com/sunnykan/disaster_response_pipeline_project/blob/main/images/index-page-graph1.png?raw=true)

![Graphs 2 and 3 of summary statistics of training data](https://github.com/sunnykan/disaster_response_pipeline_project/blob/main/images/index-page-graph-2-3.png?raw=true)

* Sample results from classifying a message.
![Results from classification of sample text](https://github.com/sunnykan/disaster_response_pipeline_project/blob/main/images/classification-page.png?raw=true)
