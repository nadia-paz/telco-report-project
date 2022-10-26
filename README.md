# telco-report-project
Project: Find drivers for customer churn at Telco. 

## Project Description
 
Chess is widely renowned as one of the most skill intensive games ever invented. Both sides begin the game with identical pieces in an identical position, there are no random elements (aside from assigning the first move), and the movement of those pieces during a game can result in over 121 million possible combinations in just the fist three moves. Because of this, the player with the most skill is likely to win the grand majority of chess games. I have decided to look into the different elements of a chess game to determine if any of them increase or decrease the chance of a player with lower skill defeating a player with greater skill.
 
## Project Goal
 
* Find drivers of customers churn at Telco. Answer the question: Why do customers churn at Telco?
* Find the important features and build the model that can predict the customer churn.
* Give the recommendations how to decrease the number of churned customers.

 
## Initial Thoughts
 
My initial hypothesis is that main churning factors are are connected with the monthly charges customers have and the time they have spent with the company.
 
## The Plan
 
* Aquire data from the database
 
* Prepare data
   * steps of preparation
       * drop the columns that are not relative to the report(payment id, internet service id, contract type id)
       * drop duplicates if available, fix the columns that have incorrect data types, null values
       * encode the categorical variables 
       * split the data into 3 data sets: train, validate, test
 
* Explore data in search of drivers of upsets
   * Answer the following initial questions:
       * Do the monthly charges affect the churn?
       * Does the contract type affect the churn?
       * Do additional services affect the churn?
       
      
      
* Develop a Model to predict if a chess game will end in an upset
   * Use drivers identified in explore to build predictive models of different types
   * Evaluate models on train and validate data
   * Select the best model based on highest accuracy
   * Evaluate the best model on test data
 
* Draw conclusions
 
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
|
|
|**Numerical Data**
|*monthly_charges*|  Monthly payment of the customer | Amount in USD
|*total_charges*|  Total amount paid to the company during all the period | Amount in USD
|*tenure*| How long did the customer stay with the company | Number of months
|
|**Other Data**
|*customer_id*| Unique customer's id number|
|
|**Target Data**
|
|**churn** | **Did the customer churn?** | **'Yes', 'No'**

 
## Steps to Reproduce
1) Clone this repo.
2) Acquire the data using your ```env.py``` file
3) Put the data in the file containing the cloned repo.
4) Run notebook.
 
## Takeaways and Conclusions
* 
* 
* 
* 
* 
* 
* 
* 
 
## Recommendations
* 
* 