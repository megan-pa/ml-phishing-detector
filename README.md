# Phishing Detector
## Overview
This project uses a trained machine learning model to form a phishing email detector. Given a pre-created data, the model classifies emails as either phishing or legitimate. 

## Dataset
This model is trained on a collection of phishing emails available at [the following site](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset). Data includes the email text and binary labels. Raw datasets are not included in this repository due to file size limits. 

## Rule-Based Detection
Alongside the ML model, this system makes use of a rule-system to assist in classifying legitimate emails. A risk score is computed using factors surrounding links, language and formatting of emails, with each having a different weighted score. These scores are then combined and when exceeding a certain threshold (e.g. >= 6), the email is automatically classified as phishing regardless of the model's prediction. 

## Tech/Framework Used
* Python 3
* pandas
* FastAPI
* scikit-learn
* Regular Expressions (re) 

## API Usage
The phishing detector has been exposed via an API built with FastAPI. 

### Running the API
From the root directory of the project file, run the following command:
<pre>uvicorn api.main:app --reload --host 0.0.0.0 --port 8001</pre>

## Installation
Refer to requirements.txt for all dependencies needed to run this project. To install all libraries, run the following command:
<pre>pip install -r requirements.txt</pre>

## Tests
Tests not currently added.

## Authors
Megan Parfitt (za23370@bristol.ac.uk)
