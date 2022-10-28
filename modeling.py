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
 #'dependents', 
 #'partner',
 #'multiple_lines',
 'online_security',
 #'online_backup',
 'device_protection',
 'tech_support',
 #'streaming_tv',
 #'streaming_movies',
 'paperless_billing',
 'monthly_charges',
 'contract_type',
 'internet_service_type',
 'payment_type_Credit card (automatic)',
 'payment_type_Electronic check',
 'payment_type_Mailed check'] #everything

def get_features(train):
    return train.columns.tolist()

number_of_features = 13

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
            model = DecisionTreeClassifier(max_depth = i, random_state=2912)
            model.fit(X_train[features_dict[key]], y_train)
            #predictions = model.predict(X_train[features_dict[key]])
            score = round(model.score(X_train[features_dict[key]], y_train), 3)
            validate = round(model.score(X_validate[features_dict[key]], y_validate), 3)
            #name = 'F' + str(key) + '-DT max_depth='+str(i)
            
            #prediction_dictionary['model_name'].append(name)
            #prediction_dictionary['accuracy_score'].append(score)

            scores_dt['model_name'].append('Decision Tree')
            scores_dt['feature_name'].append(key)
            scores_dt['accuracy_score'].append(score)
            scores_dt['features'].append(features_dict[key])
            scores_dt['parameters'].append(i)
            scores_dt['validate_score'].append(validate)

            best_dt = pd.DataFrame(scores_dt).\
                    sort_values(by=['accuracy_score', 'validate_score'], ascending=[False, False]).\
                    head(number_of_features)
            #best_models = pd.concat(best_models, best_dt)
    return best_dt


def gen_random_forest(X_train, X_validate, y_train, y_validate):
    for key in range(1,number_of_features):
        for i in range(1, 6):
            model = RandomForestClassifier(max_depth = i, random_state=2912)
            model.fit(X_train[features_dict[key]], y_train)
            #predictions = model.predict(X_train[features_dict[key]])
            score = round(model.score(X_train[features_dict[key]], y_train), 3)
            #name = 'F' + str(key) + '-RF max_depth='+str(i)
            validate = round(model.score(X_validate[features_dict[key]], y_validate), 3)
            
            #prediction_dictionary['model_name'].append(name)
            #prediction_dictionary['accuracy_score'].append(score)
            
            scores_rf['model_name'].append('Random Forest')
            scores_rf['feature_name'].append(key)
            scores_rf['accuracy_score'].append(score)
            scores_rf['features'].append(features_dict[key])
            scores_rf['parameters'].append(i)
            scores_rf['validate_score'].append(validate)

            best_rf = pd.DataFrame(scores_rf).\
                    sort_values(by=['accuracy_score', 'validate_score'], ascending=[False, False]).\
                    head(number_of_features)
            #best_models = pd.concat(best_models, best_rf)
    return best_rf

def gen_knn(X_train, X_validate, y_train, y_validate):
    for key in range(1,number_of_features):
        for i in range(1, 21):
            model = KNeighborsClassifier(n_neighbors=i, weights='uniform')
            model.fit(X_train[features_dict[key]], y_train)
            #predictions = model.predict(X_train[features_dict[key]])
            score = round(model.score(X_train[features_dict[key]], y_train), 3)
            #name = 'F' + str(key) + '-KNN n_neigh='+str(i)
            validate = round(model.score(X_validate[features_dict[key]], y_validate), 3)
            #prediction_dictionary['model_name'].append(name)
            #prediction_dictionary['accuracy_score'].append(score)
            
            scores_knn['model_name'].append('KNN')
            scores_knn['feature_name'].append(key)
            scores_knn['accuracy_score'].append(score)
            scores_knn['features'].append(features_dict[key])
            scores_knn['parameters'].append(i)
            scores_knn['validate_score'].append(validate)

            best_knn = pd.DataFrame(scores_knn).\
                    sort_values(by=['accuracy_score', 'validate_score'], ascending=[False, False]).\
                    head(number_of_features)
    return best_knn

def gen_logistic_regression(X_train, X_validate, y_train, y_validate):
    for key in features_dict:
        model = LogisticRegression(random_state=2912)
        model.fit(X_train[features_dict[key]], y_train)


        #predictions = model.predict(X_train[features_dict[key]])
        score = round(model.score(X_train[features_dict[key]], y_train), 3)
        #name = 'F' + str(key) + '-LR='+str(key)
        validate = round(model.score(X_validate[features_dict[key]], y_validate), 3)
        #prediction_dictionary['model_name'].append(name)
        #prediction_dictionary['accuracy_score'].append(score)
        
        
        scores_lr['model_name'].append('Logistic Regression')
        scores_lr['feature_name'].append(key)
        scores_lr['accuracy_score'].append(score)
        scores_lr['features'].append(features_dict[key])
        scores_lr['parameters'].append(0)
        scores_lr['validate_score'].append(validate)
    
    return pd.DataFrame(scores_lr)

def get_best_train_models(X_train, X_validate, y_train, y_validate):
    results = gen_decision_trees(X_train, X_validate, y_train, y_validate)
    results = results.append(gen_random_forest(X_train, X_validate, y_train, y_validate))
    results = results.append(gen_knn(X_train, X_validate, y_train, y_validate))
    results = results.append(gen_logistic_regression(X_train, X_validate, y_train, y_validate))
    
    return results

def parse_model(results):
    validate_models = results.model_name
    values = []
    for name in validate_models:
        key = int(name[1])
        
        model_name = name[3:5]
        
        parameter = 0
        #do not get parameters for logistic regression
        if model_name != 'LR':
            parameter = int(name.split('=')[-1])
            
        values.append({'key': key, 'model_name':model_name, 'parameter':parameter})
    return values

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
