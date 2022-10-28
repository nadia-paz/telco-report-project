# Data handling
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns


# Sklearn modules
#Machine Learning imports
import sklearn.preprocessing

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay

import wrangle as wr

seed = 2912
models_to_pick = 2
number_of_features = 13


f1 =['streaming_tv',
 'device_protection',
 'online_backup',
 'tech_support',
 'streaming_movies',
 'online_security',
 'multiple_lines']

f2 = ['tenure', 'internet_service_type', 'monthly_charges']
f3 = ['tenure', 'contract_type', 'monthly_charges']
f4 = ['contract_type', 'internet_service_type',\
             'payment_type_Credit card (automatic)', 'payment_type_Electronic check', 'payment_type_Mailed check']
f5 = ['tenure', 'contract_type', 'payment_type_Credit card (automatic)']
f6 = ['contract_type', 'online_security', 'tech_support', 'internet_service_type'] # 4 most powerful categorical features
f7 = ['contract_type', 'online_security', 'tech_support', 'internet_service_type', 'tenure'] # same + tenure
f8 = ['tenure', 'contract_type', 'monthly_charges']
f9 = ['tenure', 'contract_type', 'paperless_billing', 'monthly_charges']

f10 = ['add_services', 'tenure', 'paperless_billing', 'payment_type_Credit card (automatic)']

f11 = ['add_services', 'tenure', 'internet_service_type']

f12 = ['tenure', 'add_services',
 'paperless_billing',
 'monthly_charges',
 'contract_type',
 'internet_service_type',
 'payment_type_Credit card (automatic)',
 'payment_type_Electronic check',
 'payment_type_Mailed check']

f13 = ['tenure', 'add_services',
 'paperless_billing',
 'monthly_charges',
 'contract_type', 'dependents', 'partner']

f14 = ['tenure', 'contract_type', 'dependents', 'partner', 'monthly_charges']

f15 = ['tenure', 'internet_service_type', 'monthly_charges', 'dependents', 'partner']

f16 = ['tenure',
 'online_security',
 'device_protection',
 'tech_support',
 'paperless_billing',
 'monthly_charges',
 'contract_type',
 'internet_service_type',
 'payment_type_Credit card (automatic)',
 'payment_type_Electronic check',
 'payment_type_Mailed check'] 

def get_features(train):
    return train.columns.tolist()



features_dict= {
              1:f1, 
              2:f2, 
              3:f3, 
              4:f4, 
              5:f5,
              6:f6,
              7:f7,
              8:f8,
              9:f9,
              10:f10,
              11:f11,
              12:f12,
              13:f13,
              14:f14,
              15:f15,
              16:f16
            }

prediction_dictionary = {
    'model_name': [],
    'accuracy_score': [],
    'validate_score': []
}
best_models = pd.DataFrame()

scores_dt = {
    'model_name': [],
    'feature_name': [],
    'features': [],
    'parameters': [],
    'accuracy_score': [],
    'validate_score': []
}

scores_rf = {
    'model_name': [],
    'feature_name': [],
    'features': [],
    'parameters': [],
    'accuracy_score': [],
    'validate_score': []
}

scores_knn = {
    'model_name': [],
    'feature_name': [],
    'features': [],
    'parameters': [],
    'accuracy_score': [],
    'validate_score': []
}

scores_lr = {
    'model_name': [],
    'feature_name': [],
    'features': [],
    'parameters': [],
    'accuracy_score': [],
    'validate_score': []
}


def gen_decision_trees(X_train, X_validate, y_train, y_validate):
    for key in range(1,number_of_features):
        for i in range(1, 4):
            model = DecisionTreeClassifier(max_depth = i, random_state=seed)
            model.fit(X_train[features_dict[key]], y_train)
            
            #calculate scores
            score = round(model.score(X_train[features_dict[key]], y_train), 3)
            validate = round(model.score(X_validate[features_dict[key]], y_validate), 3)


            #save the information about the model and it's score to a dictionary
            scores_dt['model_name'].append('Decision Tree')
            scores_dt['feature_name'].append('f'+str(key))
            scores_dt['accuracy_score'].append(score)
            scores_dt['features'].append(features_dict[key])
            scores_dt['parameters'].append(i)
            scores_dt['validate_score'].append(validate)

            # create a data frame with models_to_pick number of models
            # that perform the best on the training set
            # return this data frame
            best_dt = pd.DataFrame(scores_dt).\
                    sort_values(by='accuracy_score', ascending=False).\
                    head(models_to_pick)
            
    return best_dt


def gen_random_forest(X_train, X_validate, y_train, y_validate):
    for key in range(1,number_of_features):
        for i in range(1, 6):
            #build the model and fit X_train, y_train into it
            model = RandomForestClassifier(max_depth = i, random_state=seed)
            model.fit(X_train[features_dict[key]], y_train)

            #calculate scores
            score = round(model.score(X_train[features_dict[key]], y_train), 3)
            validate = round(model.score(X_validate[features_dict[key]], y_validate), 3)
            
            
            #save the information about the model and it's score to a dictionary
            scores_rf['model_name'].append('Random Forest')
            scores_rf['feature_name'].append('f'+str(key))
            scores_rf['accuracy_score'].append(score)
            scores_rf['features'].append(features_dict[key])
            scores_rf['parameters'].append(i)
            scores_rf['validate_score'].append(validate)

            # create a data frame with models_to_pick number of models
            # that perform the best on the training set
            # return this data frame
            best_rf = pd.DataFrame(scores_rf).\
                    sort_values(by='accuracy_score', ascending=False).\
                    head(models_to_pick)
        
    return best_rf

def gen_knn(X_train, X_validate, y_train, y_validate):
    for key in range(1,number_of_features):
        for i in range(1, 21):
            #build the model and fit X_train, y_train into it
            model = KNeighborsClassifier(n_neighbors=i, weights='uniform')
            model.fit(X_train[features_dict[key]], y_train)
            
            #calculate scores
            score = round(model.score(X_train[features_dict[key]], y_train), 3)            
            validate = round(model.score(X_validate[features_dict[key]], y_validate), 3)

            
            #save the information about the model and it's score to a dictionary
            scores_knn['model_name'].append('KNN')
            scores_knn['feature_name'].append('f'+str(key))
            scores_knn['accuracy_score'].append(score)
            scores_knn['features'].append(features_dict[key])
            scores_knn['parameters'].append(i)
            scores_knn['validate_score'].append(validate)

            # create a data frame with models_to_pick number of models
            # that perform the best on the training set
            # return this data frame
            best_knn = pd.DataFrame(scores_knn).\
                    sort_values(by='accuracy_score', ascending=False).\
                    head(models_to_pick)
    return best_knn

def gen_logistic_regression(X_train, X_validate, y_train, y_validate):
    for key in features_dict:
        #build the model and fit X_train, y_train into it
        model = LogisticRegression(random_state=seed)
        model.fit(X_train[features_dict[key]], y_train)

        #calculate scores
        score = round(model.score(X_train[features_dict[key]], y_train), 3)
        validate = round(model.score(X_validate[features_dict[key]], y_validate), 3)
        
        #save the information about the model and it's score to a dictionary
        scores_lr['model_name'].append('Logistic Regression')
        scores_lr['feature_name'].append('f'+str(key))
        scores_lr['accuracy_score'].append(score)
        scores_lr['features'].append(features_dict[key])
        scores_lr['parameters'].append(0)
        scores_lr['validate_score'].append(validate)

        # create a data frame with models_to_pick number of models
        # that perform the best on the training set
        # return this data frame
    return pd.DataFrame(scores_lr).sort_values(by='accuracy_score', ascending=False).\
                    head(models_to_pick)

def get_best_train_models(X_train, X_validate, y_train, y_validate):
    
    #run all models 
    #functions return best performing models
    #save them to 1 data fram and return this data frame
    results = gen_decision_trees(X_train, X_validate, y_train, y_validate)
    results = results.append(gen_random_forest(X_train, X_validate, y_train, y_validate))
    results = results.append(gen_knn(X_train, X_validate, y_train, y_validate))
    results = results.append(gen_logistic_regression(X_train, X_validate, y_train, y_validate))
    
    return results

def run_the_final_test(results, X_train, X_test, y_train, y_test):
    '''
    accepts a data frame with 3 best models
    creates the 
    '''
    #create a data frame with 1  row - model winner
    best_model = results.head(1)
    #build the model
    model = LogisticRegression()
    #fit X-train and y_train
    model.fit(X_train[f16], y_train)
    #create predictions from the X_test
    predictions = model.predict(X_test[f16])
    #save probalities of churn from the X_test
    probabilities = model.predict_proba(X_test[f16])[:,1]
    #count the accuracy score on the test sample
    score = round(model.score(X_test[f16], y_test), 3)
    #save the score to the best_model data frame
    best_model['test_score'] = score

    return best_model, probabilities, predictions

def save_to_csv(customer_ids, probabilities, predictions):
    '''
    accepts 3 arrays as parameters
    creates a data frame 
    save this data frame into *.csv file
    '''
    final_csv = pd.DataFrame({'customer_id':customer_ids, 
                         'probability_of_churn':probabilities, 
                         'predictions_of_churn':predictions})
    final_csv.to_csv('predictions.csv', index=False)

####################################

def get_baseline_scocre(y_train):
    '''
    accepts y_train series, calculates and returns baseline score
    '''
    baseline_score = (y_train == int(y_train.mode())).mean()
    return baseline_score

def get_customers_ids(X_train, X_validate, X_test):
    '''
    accepts train, validate, test data sets
    saves customer ids to the variable
    drops the customer_id column from all data sets
    returns a Series with customer ids
    '''
    #get the customer ids from the test data set. We'll need it for the *.csv file
    customer_ids = X_test.customer_id

    #drop customer_id from all columns
    X_train.drop(columns = 'customer_id', inplace=True)
    X_validate.drop(columns = 'customer_id', inplace=True)
    X_test.drop(columns = 'customer_id', inplace=True)

    return customer_ids



def count_differnce(results):
    '''
    accepts a dataframe with the test results as a parameter
    for every row counts the absolute value of the 
    difference in scores of train and validate data sets
    creates a new column that holds the obtained values
    sorts the values by max validate_score and min score_difference
    returns back the result data frame with new column
    
    '''
    results['score_difference'] = abs(results.accuracy_score - results.validate_score)
    return results.sort_values(by=['validate_score', 'score_difference'], ascending=[False, True]).head(3)


####################################


def count_scores(confusion_matrix, model_name_str):
    '''
    returns a dataframe with scores of the model performing
    '''
    TN, FP, FN, TP = confusion_matrix.ravel()
    ALL = TP + FP + FN + TN

    accuracy = round((TP + TN)/ALL, 2)
    true_positive_rate = sensitivity = recall = power = round(TP/(TP+FN), 2)
    false_positive_rate = false_alarm_ratio = fallout = round(FP/(FP+TN), 2)
    true_negative_rate = specificity = selectivity = round(TN/(TN+FP), 2)
    false_negative_rate = miss_rate = round(FN/(FN+TP), 2)
    precision = PPV = round(TP/(TP+FP), 2)
    f1_score = round(2*(precision*recall)/(precision+recall), 2)
    support_pos = int(TP + FN)
    support_neg = int(FP + TN)
    
    rates = ['Accuracy', 
         'True Positive Rate', 
         'False Positive Rate', 
         'True Negative Rate', 
         'False Negative Rate',
         'Precision', 
         'F1 Score',
         'Support Positive',
         'Support Negative']
    scores = pd.Series([accuracy, true_positive_rate, false_positive_rate, true_negative_rate, 
                        false_negative_rate, precision, f1_score, support_pos, support_neg])
    
    return pd.DataFrame({'Score Name':rates, model_name_str:scores})

def display_cm(confusion_matrix):
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    disp.plot()
