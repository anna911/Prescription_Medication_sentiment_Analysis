## Introduction: 
There are various formulations and brands for Prescription drugs available in the market for every condition(illness)     Physicians/Prescribers choose a formulation based on factors such drug interactions with other drugs the patient is already    taking , effectiveness of the formulation, more importantly side effects.Some Prescription drugs have very adverse side        effects.For example most of the Prescription drugs have side effects such as migrane. sometimes it may lead to worst and      lead to bi-polar disorder.

### Goal:
  I believe Technology should be incorporated  to Healthcare to improve quality care. 
  Based on the Reviews of the Prescription Drugs.
  * To Predict the patients sentiment(experience)  with a "Rx" if it is Positive,Negative or Neutral based on the reviews 


## Sentiment Analysis of the Rx Drug based on the Patients Reviews 
 
### Overview of the Data: 
The dataset has been webscraped from http://www.druglib.com/ by Surya Kallumadi, Felix Gräßer.
	    https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29
The reviews are collected from Feb 2008 to 2017 December. 
Total Number of reviews for all the drugs 215,063 for 3671 drugs and 917 different conditions.

example of the reviews
##### review 1
![alt text](https://github.com/anna911/Prescription_Medication_sentiment_Analysis/blob/master/L1.png)
##### review 2
![alt text](https://github.com/anna911/Prescription_Medication_sentiment_Analysis/blob/master/L-2.png)


### Exploratory Data Analysis: 

#### Distibution of the reviews,Positive,Negative or Neutral 
![alt text](https://github.com/anna911/Prescription_Medication_sentiment_Analysis/blob/master/piechart.png)
#### Most Reviewed  Drugs 
![alt text](https://github.com/anna911/Prescription_Medication_sentiment_Analysis/blob/master/Most_reviewed_drugs.png)
#### Most Reviewed  Conditions
![alt text](https://github.com/anna911/Prescription_Medication_sentiment_Analysis/blob/master/Most_reviwed_conditions.png)
#### Most Reviewed Conditions and Drugs 
#### Most common words for High Rated Drugs 
![alt text](https://github.com/anna911/Prescription_Medication_sentiment_Analysis/blob/master/WCHighestRateddrugs.png)
#### Most common words for Low Rated Drugs  
![alt text](https://github.com/anna911/Prescription_Medication_sentiment_Analysis/blob/master/lowest_rated.png)

### Data Pipeline : 

Created NLP pipeline for Textscrubbing.The size of the each review varied between  10 TO ~800 words 
1) Created word tokens for each review.
2) Removed Stop words and punctuation marks from the tokens  
3) Stemming and Lemmatization on word tokens
4) Standardise text 
The code to do this can be found here
5) Created TF-IDF sparse Matrix for the reviews

  #### Add pictutre for Data Pipeline 


### Model Selection:

Tested Various Models the accuracy score for these models.

Models and their accuracy scores 

| Logistic Regression |  Gradient Booting | Random Forest   |  LSTM       |
|---------------------|-------------------|-----------------|-------------|                 
| 0.76972436112041    | 0.659152624335081 |     0.8869      | 0.853       |                   


Confusion Matrix for the Highest Performing Dta 

![alt text](https://github.com/anna911/Prescription_Medication_sentiment_Analysis/blob/master/Random_forest_confusion_matrix.png)![alt text](https://github.com/anna911/Prescription_Medication_sentiment_Analysis/blob/master/LSTM-sentiment-analysis.png)
![alt text](https://github.com/anna911/Prescription_Medication_sentiment_Analysis/blob/master/lg-cm.png)
![alt text](https://github.com/anna911/Prescription_Medication_sentiment_Analysis/blob/master/gb-cm.png)






