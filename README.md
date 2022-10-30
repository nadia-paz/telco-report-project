# telco-report-project
Project: Find drivers for customer churn at Telco. 
 

 
## Project Goal
 
* **Part I** *Find the drivers of customers churn*.
* **Part II** *Build the machine learning model that can predict customer's churn with accuracy > 75%* 
* *Drow conclusions and define the next steps*

## Steps to Reproduce
1) Clone this repo into your computer.
2) Acquire the data from Codeup using your ```env.py``` file
3) Put the data in the file containing the cloned repo.
4) Run the ```telco_project.ipynb``` file.

 
## Initial Thoughts
 
My initial hypothesis is that main churning factors are are connected with the monthly charges customers have and the time they have spent with the company.
 
## The Plan
 

#### Part I
1. Acquire the data from the ```telco``` database. Transform the data to a Pandas data frame to make it easy to use and manipulate in the Jupyter Notebook.
2. Prepare the data for exploration and analysis. Find out if there are some values missing and find a way to handle those missing values.
3. Explore the data through visualizations and statistical tests. Find which features are connected with the customers' churn and which ones are not. 
4. Make the exploration summary and leave the recommendation, where the company has to pay more attention


#### Part II
1. Pick the features that can help to build a good predicting model.
2. Pick the algorithms for creating the predicting model.
3. Encode the categorical variables 
4. Split the data into 3 data sets: train, validate and test data (56%, 24%, and 20% respectively)
5. Create the models and evaluate them using the accuracy score on the train data sets.
6. Pick the models with the best accuracy score and evaluate them on the validation set.
7. Find out which model has the best performance: relatively high predicting power on the validation test and slight difference in the train and validation prediction results.
8. Apply the predictions to the test data set. Show the final accuracy scores. Save the predictions to the ```*.csv``` file

*Drow conclusions and define the next steps*
 
## Data Dictionary


| Feature | Definition |Values|
|:--------|:-----------|:-----------|
|<img width=150/>|<img width=550/>|
|**Categorical Data**
|*gender*| The gender of the customer  | 'Female', 'Male'
|*senior_citizen*| Gives the information if the customer is a senior citizen| 0 -not senior, 1 - is senior citizen
|*partner*| Shows if the customer has a partner| 'Yes', 'No'
|*dependents*| Information if the customer has dependents | 'Yes', 'No'
|**Phone services**
|*phone_service*| Phone service connected | 'Yes', 'No'
|*multiple_lines*| Multiple phone lines connected | 'Yes', 'No', 'No phone service'
|**Internet services**
|*internet_service_type*|  Type of internet service customer has | 'DSL', 'Fiber optic', 'None'
|*online_security*| Online security included | 'Yes', 'No', 'No internet service'
|*online_backup*| Online backup included | 'Yes', 'No', 'No internet service'
|*device_protection*| Device protection included | 'Yes', 'No', 'No internet service'
|*tech_support*|  Tech support included | 'Yes', 'No', 'No internet service'
|*streaming_tv*|  Streaming TV included| 'Yes', 'No', 'No internet service'
|*streaming_movies*|  Streaming movies | 'Yes', 'No', 'No internet service'
|**Financial categories**
|*paperless_billing*|  Does customer use paperless billing | 'Yes', 'No'
|*contract_type*|  What type of contract has the customer | 'One year', <br>'Month-to-month', <br>'Two year'
|*payment_type*|  How does the customer pay for the service | 'Mailed check', <br>'Electronic check', <br>'Credit card (automatic)', <br>'Bank transfer (automatic)'
|**Numerical Data**
|*monthly_charges*|  Monthly payment of the customer | Amount in USD
|*total_charges*|  Total amount paid to the company during all the period | Amount in USD
|*tenure*| How long did the customer stay with the company | Number of months
||<img width=150/>|<img width=550/>|
|**Other Data**
|*customer_id*| Unique customer's id number|
||<img width=150/>|<img width=550/>|
|**Target Data**
|**churn** | **Did the customer churn?** | **'Yes', 'No'**

 
 
## Part I - Exploration

## Takeaways and recommendations
1. Having or not having a phone service doesn't affect the customer’s churn much.
3. Fiber optic customers churn more.
    - ***Recommendation:*** research the competitor's market and reconsider the payment for the fiber optic internet or include more additional services for free for those customers.
4. Most of the churn happens within the first few months. My guess is that those customers are travelers that use the phone/internet services only for a short period of time.
    - ***Recommendation:***
        * Create special attractive 'travelers packages’ and track them as well. This will help to identify if the churn is linked to travel reasons or if the company has something that makes new customers unhappy.
        * Offer some additional service discounts at the end of the first half-year period.
5. Most customers that churn have month-to-month contracts.
    - ***Recommendation:*** give more incentives for signing one-year or two-year contracts
6. Churned customers also had higher monthly payments. Monthly payments correlate with additional services. We can not recommend cutting the services, but we can recommend making more attractive service packages.
    - ***Recommendation:*** create more attractive service packages with better quality/price relationship
 
## Part II - Modeling
- Almost all features in the data set significantly affect the customer's churn. My choice was to create feature combinations and test how various algorithms perform.
- Logistic Regression performs much better in terms of showing similar results with ```train```, ```validate``` and ```test``` data sets. It won the first two positions in my 'best model' criterion.
- The 3rd best model was Desicion Tree with ```max_depth``` hyperparameter set to ```3```.
- Overall, my goal was to create a model that has a prediction power > 75%, and my 'winner' successfully coped with the task.

## Conclusions and the next steps
- Most of the data presented in the ```telco``` data set affects the customer's churn. After exploring data I can name the biggest drivers of churn at the company, so the company can pay more attention to the customer that are high risk of churn. The drivers are:
- contract type, 
- the internet service type,
- the payment type,
- monthly charges. 

The model that have got the highest accuracy score included all of them. 
 
**Next steps** 
The analysis provided in this project is just a first step on the way to deeper explorations and predictions. The next step in the exploration would be a multivariate analysis. Do the customers with dependents prefer Streaming Tv or Device protection? Do their preferences make them stay with the company or leave? Who are the people that sign Two-year contracts? Senior citizens, married couples?  What services do they prefer to include? What contracts do fiber optic customers sign? What payment types do they use?
- The multivariate analysis will help to see the churn drivers better and possibly will help to build a better prediction model.