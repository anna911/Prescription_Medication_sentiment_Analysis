## Introduction: 
There are various formulations and brands for Prescription drugs available in the market for every condition(illness)     Physicians/Prescribers choose a formulation based on factors such drug interactions with other drugs the patient is already    taking , effectiveness of the formulation, more importantly side effects.Some Prescription drugs have very adverse side        effects.For example most of the Prescription drugs have side effects such as migrane. sometimes it may lead to worst and      lead to bi-polar disorder.

### Goal:
  I believe Technology should be incorporated  to Healthcare to improve quality care. 
  Based on the Reviews of the Prescription Drugs.
  1) To Predict the patients sentiment(experience)  with a "Rx" if it is Positive,Negative or Neutral based on the reviews 
  2) To Predict the severity of side effects based on the patients reviews/comments. 
  1 - No side effects
  2 - less or Moderate Side effects 
  3 - High or Severe side effects 

## Sentiment Analysis of the Rx Drug based on the Patients Reviews 
 
### Overview of the Data: 
The dataset has been webscraped from http://www.druglib.com/ by Surya Kallumadi, Felix Gräßer.
	    https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29
The reviews are collected from Feb 2008 to 2017 December. 
Total Number of reviews for all the drugs 215,063 for 3671 drugs and 917 different conditions.

## Include 2 drugs with many conditions and reviews .Most Reviewed conditions pie chart 

### Exploratory Data Analysis: 

#### Distibution of the reviews , positive,Negative or Neutral 
![alt text](https://github.com/anna911/Prescription_Medication_sentiment_Analysis/blob/master/piechart.png)
#### Most Reviewed  Drugs 
![alt text](https://github.com/anna911/Prescription_Medication_sentiment_Analysis/blob/master/Most_reviewed_drugs.png)
#### Most Reviewed  Conditions
![alt text](https://github.com/anna911/Prescription_Medication_sentiment_Analysis/blob/master/Most_reviwed_conditions.png)


   

### Data Pipeline : 

Created NLP pipeline for Textscrubbing.The size of the each review varied between  700 
1) Created word tokens for each review.
2) Removed Stop words and punctuation marks from the tokens  
3) Stemming and Lemmatization on word tokens
4) Standardise text 
The code to do this can be found here
5) Created TF-IDF sparse Matrix for the reviews

### Model Selection:

Tested Various Models the accuracy score for these models.

Models and their accuracy scores 

Logistic Regression 
Accuracy score - 0.77 

Random Forest Classifier 
Accuracy Score - 0.88

Random Forest Classifier performed the best of all the models chosen. 

Neural Networks and LSTM Model architecture to Predict the classes 










