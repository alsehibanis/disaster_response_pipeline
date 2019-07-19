# Disaster Response Pipeline Project

## Contents
1. [Overview](#overview)
2. [Project Structure](#projectStructure)
3. [Instructions](#instructions)
4. [Screenshots](#screenshots)

<a name="overview"></a>
### Overview
This project is part of Udacity's Data Science Nanodegree.  This project aims to apply the data engineering skills to analyze the disater data from Figure Eight and to build a model that classifies disaster messages.

<a name="projectStructure"></a>
### Project Structure
- `app/`
  - `template/`
    - `master.html`  -  Main page of web app.
    - `go.html`  -  Classification result page of web app.
  - `run.py`  - Flask file that runs the app.

- `data/`
  - `disaster_categories.csv`  - Disaster categories dataset.
  - `disaster_messages.csv`  - Disaster Messages dataset.
  - `process_data.py` - The script to process and clean the data script.
  - `DisasterResponse.db`   - An SQLite database that stores the processed data.

- `models/`
  - `train_classifier.py` - The pipeline script to train and evaluate the messages.

 - `screenshots/` 
  	- `ExampleOfMessageClassification.JPG` - Example of Message Classification
 	- `DistributionOfGenres.JPG` - Example of the Distribution of Genres Bar Chart
	- `Top10Categories.JPG` - Example of the Distribution of Top 10 Categories Bar Chart
 	- `DistributionOfCategories.JPG` - Example of the Distribution of Categories Bar Chart

<a name="instructions"></a>
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

<a name="screenshots"></a>
### Screenshots
1.This screenshot shows an example of message classification:
<img src='screenshot/ExampleOfMessageClassification.JPG' width="800" height="500" />
<br>

2.This screenshot shows the distribution of message genres bar chart displayed by the appliation
<img src='screenshot/DistributionOfGenres.JPG' width="800" height="500" />
<br>

3.This screenshot shows the distribution of top 10 message categories bar chart displayed by the
appliation
<img src='screenshot/Top10Categories.JPG' width="800" height="500" />
<br>


4.This screenshot shows the distribution of message categories bar chart displayed by the
appliation
<img src='screenshot/DistributionOfCategories.JPG' width="800" height="500" />
<br>