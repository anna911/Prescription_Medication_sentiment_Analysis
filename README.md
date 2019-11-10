## Introduction: 
There are various formulations and brands for Prescription drugs available in the market for every condition(illness)     Physicians/Prescribers choose a formulation based on factors such drug interactions with other drugs the patient is already    taking , effectiveness of the formulation, more importantly side effects.Some Prescription drugs have very adverse side        effects.For example most of the Prescription drugs have side effects such as migrane. sometimes it may lead to worst and      lead to bi-polar disorder.

### Goal:
  I believe Technology should be incorporated  to Healthcare to improve quality care. 
  Based on the Reviews of the Prescription Drugs.
  I want to predict the patients experience with a "Rx" it is Positive,Negative or Neutral based on the reviews 
  
### Overview of the Data: 
The dataset has been webscraped from http://www.druglib.com/ by Surya Kallumadi, Felix Gräßer.
	    https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29
The reviews are collected from Feb 2008 to 2017 December. 
Total Number of reviews for all the drugs 215063 for 3671 drugs and 917 different conditions.

## Include 2 drugs with many conditions and reviews .Most Reviewed conditions:

### Exploratory Data Analysis : 
![alt text](https://github.com/anna911/Prescription_Medication_sentiment_Analysis/blob/master/piechart.png)
   

### DataPipeline : 

Created NLP pipeline for Textscrubbing.The size of the each review varied between  700 
1) Created word tokens for each review.
2) Removed Stop words and punctuation marks from the tokens  
3) Stemming and Lemmatization on word tokens
4) Standardise text 
The code to do this can be found here

### Model Selection:

Went through different machine learning algorithms in order to find a model that can predict the personalities. Random would be 1/16 or 0.0625. That is really low, so for our model let's aim to achiece results higher than 50%. The code for this can be found here

We will use the following models:

Random Forest - Accuracy = 0.3614985590778098
Gradient Boosting Classifier - Accuracy = 0.650787552823665
Naive Bayes - Accuracy = 0.22051479062620052
Logistic Regression - Accuracy = 0.6300422589320015
Support Vector Machine - Accuracy = 0.6699961582789089
Back to top

Deep Learning

Creating a Neural Network gives us a much higher accuracy score. The code for this can be found here

Accuracy = 0.9865539761813292
Back to top

Emotional Analysis

Next let's dive into the emotions by each personality type. The code for this can be found here.

Extroverted	Introverted
	
Back to top

WordClouds

Now let's go back to the data and see what we can derive

Created another dictionary with high frequency words by Personality Type
This can help us make some word clouds but first we need to clean our data
Created a list of the 30 most common words among all personality types
Removed the words in that list from our dataset
Let's get a bit fancy, instead of the default wordclouds, we can use a template for them, since we are talking about the mind, let's use a head.

Extroverted	Introverted
ENTP	INTP
	
Back to top

Conclusion and Next Steps

Took the datasets and performed Exploratory Data Analysis
Created a data pipeline
Built several models and picked support vector machine with stochastic gradient descent due to it's high accuracy and precision
Built a Neural Network which improved gave great accuracy but was overfit to the over represnted classes
Performed emotional analysis for each personality type
Created Word Clouds based on the frequancy of words used by each personality type.
Next step would be to gather data from another place like twitter or facebook and see if we can predict personalities based on that text
