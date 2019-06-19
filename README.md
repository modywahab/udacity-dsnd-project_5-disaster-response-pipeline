# Disaster Response Pipeline
**Mohammed Alhussain - June 19th, 2019**

*Disclaimer: The author developed this project as part of Udacity's Nanodegree program in Data Science. The specifications for this project and the HTML files were provided by [Udacity](udacity.com), and the datasets were provided by [Figure-Eight](https://www.figure-eight.com/)*

--------------------------

## Project Overview 
*(From [Udacity.com](udacity.com))*

In this course, you've learned and built on your data engineering skills to expand your opportunities and potential as a data scientist. In this project, you'll apply these skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

In the Project Workspace, you'll find a data set containing real messages that were sent during disaster events. You will be creating a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

Your project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. This project will show off your software skills, including your ability to create basic data pipelines and write clean, organized code!

Below are a few screenshots of the web app.


<img src="https://s3.amazonaws.com/video.udacity-data.com/topher/2018/September/5b967bef_disaster-response-project1/disaster-response-project1.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />
<img src="https://s3.amazonaws.com/video.udacity-data.com/topher/2018/September/5b967cda_disaster-response-project2/disaster-response-project2.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 20px;" />
     
 ## Project Components
There are three components you'll need to complete for this project.

### 1. ETL Pipeline
In a Python script, process_data.py, write a data cleaning pipeline that:

 - Loads the messages and categories datasets
 - Merges the two datasets
 - Cleans the data
 - Stores it in a SQLite database
 
 
 
### 2. ML Pipeline
In a Python script, `train_classifier.py`, write a machine learning pipeline that:

 - Loads data from the SQLite database
 - Splits the dataset into training and test sets
 - Builds a text processing and machine learning pipeline
 - Trains and tunes a model using GridSearchCV
 - Outputs results on the test set
 - Exports the final model as a pickle file



### 3. Flask Web App
We are providing much of the flask web app for you, but feel free to add extra features depending on your knowledge of flask, html, css and javascript. For this part, you'll need to:

 - Modify file paths for database and model as needed
 - Add data visualizations using Plotly in the web app. One example is provided for you
 
 
--------------------------

## Instructions

## Required Packages

 - Pandas
 - Numpy
 - Sci-kit Learn
 - NLTK
 - CloudPickle
 - Pickle
 - Sqlalchemy
 - Flask
 - Plotly
## CML Instructions

Before you launch the Flask server you need to run the two following instructions
 - ETL the database: 
  
  `python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/processed_messages.db`
  
 - Train the model: 
 
  `python3 models/process_data.py data/processed_messages.db models/model.pkl`
  
 - Then launch the server by running: 
 
  `python3 app/run.py`
