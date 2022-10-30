''' 
STATISTICAL TESTS AND VISUALIZATIONS
'''

import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

import wrangle as wr

#set the significance level to 0.05
alpha = 0.05

def services_features(df):
    '''
    accepts a telco data frame as a parameter
    returns list of names of columns with additional services
    'online_security', 'tech_support', 'streaming_movies', 
    'device_protection', 'multiple_lines', 'streaming_tv', 'online_backup'
    '''
    services = []

    #create a pattern to look for in the unique values
    pattern = ['No ', ' service'] #include spaces!!!

    # get the categorical variables excluding customer id
    categorical_data = wr.get_cat_variables(df)
    #for loop to find a pattern
    #identify the columns that have unique values 
    #'No phone service' and 'No internet service'
    for col in categorical_data:
        for value in df[col].unique().tolist():
            for p in pattern:
                if re.findall(p, value):
                    services.append(col)  
    
    return list(set(services))

############ run statistical tests ##########

def check_variances(churned, not_churned, numerical_column):
    '''
    accepts 2 data subsets from telco data frame and a numerical column name as parameters
    runs the Levene test to check if the variances are equal in both subsets
    '''
    #get the p-value
    p = stats.levene(churned[numerical_column], not_churned[numerical_column])[1]
    
    #check if the p-value is less than a significance level
    if p<alpha:
        print('Variances are not equal')
    else:
        print('Variances are equal')

def phone_service_test(df):
    '''
    accepts a telco data frame as a parameter and runs the Chi squared stat test
    comaring having a phone service and churn
    '''
    phone_observed = pd.crosstab(df.phone_service, df.churn)
    p = stats.chi2_contingency(phone_observed)[1]
    if p < alpha:
        print(f'P-value is {p}')
        print('There is enough evidence that having a phone service connceted is associated with customer churn')
    else:
        print(f'P-value is {p}')
        print('There is not enough evidence that having a phone service connected is associated with customer churn')

def internet_service_test(df):
    '''
    accepts a telco data frame as a parameter and runs the Chi squared stat test
    comaring having a internet service type and churn
    '''
    internet_observed = pd.crosstab(df.internet_service_type, df.churn)
    p = stats.chi2_contingency(internet_observed)[1]
    if p < alpha:
        print(f'P-value is {p}')
        print('There is enough evidence that Internet service type is associated with customers churn')
    else:
        print(f'P-value is {p}')
        print('There is not enough evidence that Internet service type is associated with customer churn')

def test_tenure(churned, not_churned):
    '''
    accepts 2 data subsets from telco data frame and a numerical column name as parameters
    runs the Mann Whtney-U test to check if the variances are equal in both subsets
    '''
    
    #p = stats.ttest_ind(churned.tenure, not_churned.tenure, equal_var=False)[1]
    p = stats.mannwhitneyu(churned.tenure, not_churned.tenure)[1]


    if(p < alpha):
        print(f'P-value is {p}')
        print('There is a significant difference in the average number of months churned and current customers stay with the company')
    else:
        print(f'P-value is {p}')
        print('There is no significant difference in the average number of months churned and current customers stay with the company')

def test_monthly_charges(churned, not_churned):
    '''
    accepts 2 data subsets from telco data frame and a numerical column name as parameters
    runs the Mann Whtney-U test to check if the variances are equal in both subsets
    '''
    t, p = stats.ttest_ind(churned.monthly_charges, not_churned.monthly_charges, equal_var=False)
    #p = stats.mannwhitneyu(churned.monthly_charges, not_churned.monthly_charges)[1]
    if p / 2 < alpha  and t > 0:
        print(f'P-value is {p}')
        print('The average monthly charges of churned customers <= The average monthly charges of customers who haven\'t churned')
    else:
        print('The average monthly charges of churned customers > The average monthly charges of customers who haven\'t churned')

def get_p_values(df, cat_vars):
    '''
    accepts a data frame and the list of categorical column names
    returns a data frame with p_values of all categorical variables
    '''

    #dictionary to hold names of the column and a p_value assotiated with it
    p_v = {}
    #for every column in category variables run a chi2 test
    for col in cat_vars:
        #create a crosstable
        observed = pd.crosstab(df[col], df.churn)
        #run a chi squared test fot categorical data
        test = stats.chi2_contingency(observed)
        p_value = test[1]
        #add the result to the dictionary
        p_v[col] = p_value
        
        #transform a dictionary to Series and then to Data Frame
        p_values = pd.Series(p_v).reset_index()
        p_values.rename(columns = {'index':'Feature', 0:'P_value'}, inplace = True)
        p_values = p_values.sort_values(by='P_value')

        #add the column that shows if the result is significant
        p_values['is_significant'] = p_values['P_value'] < alpha
    
    return p_values


####### VISUALISATIONS ######





def service_visuals(df):
    '''
    accepts a telco data frame as a parameter
    creates visuals for tech_support, online_security, online_backup, device_protection
    highlight customers that churned and didn't churn
    '''
    
    plt.figure(figsize = (20, 8))

    #using subplots to show visuals 1 row and 4 columns

    #subplot 1
    plt.subplot(141)
    sns.histplot(data = df, x="tech_support", hue="churn", stat="percent", multiple="dodge", shrink=.8)
    plt.title('Tech support', fontsize=16)

    #subplot 2
    plt.subplot(142)
    sns.histplot(data = df, x="online_security", hue="churn", stat="percent", multiple="dodge", shrink=.8)
    plt.title('Online security', fontsize=16)

    #subplot 3
    plt.subplot(143)
    sns.histplot(data = df, x="online_backup", hue="churn", stat="percent", multiple="dodge", shrink=.8)
    plt.title('Online backup', fontsize=16)

    #subplot 4
    plt.subplot(144)
    sns.histplot(data = df, x="device_protection", hue="churn", stat="percent", multiple="dodge", shrink=.8)
    plt.title('Device protection', fontsize=16)

    plt.show()

def visualize_phone_internet_services(churned, not_churned):
    '''
    accepts churned and not_churned subsets of telco data frame
    shows the churned/current customers and if they use phone/internet services

    '''
    plt.figure(figsize = (20, 8))

    plt.suptitle('Phone and internet services', fontsize = 20)

    #using subplots to show visuals 1 row and 4 columns

    #phone service

    #subplot 1
    plt.subplot(141)
    sns.histplot(data = churned, x="phone_service", hue='phone_service', stat="percent")
    plt.title('Phone    Churned', fontsize = 16)

    #subplot 2
    plt.subplot(142)
    sns.histplot(data = not_churned, x="phone_service", stat="percent", hue='phone_service')
    plt.title('Phone    Current', fontsize = 16)

    #internet service types

    #subplot 3
    plt.subplot(143)
    sns.histplot(data = churned, x="internet_service_type", hue='internet_service_type', stat="percent")
    plt.title('Internet    Churned', fontsize =16)

    #subplot 4
    plt.subplot(144)
    sns.histplot(data = not_churned, x="internet_service_type", hue='internet_service_type', stat="percent")
    plt.title('Internet    Current', fontsize = 16)
    plt.show()

def visualize_contract_type(churned, not_churned):  
    '''
    accepts churned and not_churned subsets of telco data frame
    shows the contract types of churned/current customers
    '''

    plt.figure(figsize = (20, 6))
    plt.suptitle('Contract types', fontsize=20)

    #data is shown if 1 row and 2 columns: churned and not churned(current customers)

    #subplot 1 - churned
    plt.subplot(121)
    sns.histplot(data = churned, x="contract_type", hue='contract_type', stat="percent")
    plt.title('Churned customers')

    #subplot 2 - not churned
    plt.subplot(122)
    sns.histplot(data = not_churned, x="contract_type", hue='contract_type', stat="percent")
    plt.title('Current customers')
    plt.show()

def visualize_tenure(churned, not_churned):
    '''
    accepts churned and not_churned subsets of telco data frame
    visualize 2 histgrams showing how long the customers stay/stayed with the company
    '''
    plt.figure(figsize = (20, 6))
    plt.suptitle('Tenure of churned and current customers', fontsize = 20)

    #subplot 1 - churned
    plt.subplot(121)
    plt.title('Churned customers', fontsize=16)
    sns.histplot(data = churned, x = 'tenure', kde = True)

    plt.subplot(122)
    plt.title('Current customers', fontsize=16)
    sns.histplot(data = not_churned, x = 'tenure', kde = True)

    plt.show()

def visualize_monthly_charges(churned, not_churned):
    '''
    accepts churned and not_churned subsets of telco data frame
    visualize 2 histgrams showing monthly charges of churned and current customers
    '''
    
    plt.figure(figsize = (20, 6))
    plt.suptitle('Monthly charges of churned and current customers', fontsize = 20)

    #subplot 1
    plt.subplot(121)
    sns.histplot(data=churned, x = 'monthly_charges', stat='percent', kde=True)
    plt.title('Churned customers', fontsize = 16)

    #subplot 2
    plt.subplot(122)
    sns.histplot(data=not_churned, x = 'monthly_charges', stat='percent', kde=True)
    plt.title('Current customers', fontsize = 16)

    plt.show()

def charges_services_corr(df):
    '''
    creates a scatter plot that shows a relation between
    monthly charges and number of additiona services
    '''
    plt.figure(figsize = (12, 10))
    sns.scatterplot(data = df, x = 'monthly_charges', y='add_services', hue='churn')
    plt.show()

####### COMBINED ######
def explore_tenure(churned, not_churned):
    test_tenure(churned, not_churned)
    visualize_tenure(churned, not_churned)

def explore_monthly_charges(churned, not_churned):
    test_monthly_charges(churned, not_churned)
    visualize_monthly_charges(churned, not_churned)

