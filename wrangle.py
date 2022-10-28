'''Acquire and prepare TELCO data'''

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from env import get_db_url

#service_features holds the names of columns with additional services
service_features = ['streaming_tv',\
                'device_protection',\
                'online_backup',\
                'multiple_lines',\
                'online_security',\
                'streaming_movies',\
                'tech_support']

######## ACQUIRE TELCO DATA ########

def get_telco_data():
    '''
    Returns a dataframe from the telco database or csv file.

    The returned DataFrame includes the customer information 
    plus contract, payment and internet service information of each customer.

    If file telco.csv is not available this function reads the database,
    and creates a *.csv file in the current repo.

    To run this function you need to have your own env.py file that gives you an access to the data base.
    
    '''
    
    filename = 'telco.csv'

    sql = '''
    SELECT * FROM customers
    JOIN contract_types USING (contract_type_id)
    JOIN internet_service_types USING (internet_service_type_id)
    JOIN payment_types USING (payment_type_id)
    '''
    url = get_db_url('telco_churn')

    if os.path.isfile(filename):
        #if file exists, read the data from *.csv
        return pd.read_csv(filename)
    else:
        #if file doesn't exist read the data from the database and save it into the *.csv file
        df = pd.read_sql(sql, url)
        df.to_csv(filename, index_label = False)
        return df


######## FUNCTIONS TO PREPARE TELCO DATA FOR THE EXPLORATION AND MODELING ########



def prep_telco(df):
    '''
    This function prepares the telco data for the exploration and analysis.
    Drop duplicates and id columns.
    It leaves the customer_id column as it's needed to the *.csv file with predictions,
    so the customer_id should be dropped right before the modeling
    '''

    # Drop duplicates
    df.drop_duplicates(inplace = True)
    df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id'], inplace=True)
       
    # Drop null values stored as whitespace    
    df['total_charges'] = df['total_charges'].str.strip()
    df = df[df.total_charges != '']
    
    # Convert total charges to float datatype
    df['total_charges'] = df.total_charges.astype(float)

    #optimize the memory usage with type converting to category type

    #convert objects to categories
    df = to_category(df)
    
    #convert senior_citizen from int64 to uint8
    df.loc[:, 'senior_citizen'] = df.loc[:, 'senior_citizen'].astype('uint8')

    return df


def dummies_telco(df):
    '''
    creates dummies and
    drops not numerical columns 
    '''
    
    # Convert categorical variables to numeric
    
    #replace 'Yes'/'No' with 1 and 0
    #Month-to-month with 0, One year with 1, Two year with 2
    #None(internet service) with 0, DSL with 1 and Fiber optic with 2
    df['paperless_billing'] = df.paperless_billing.map({'Yes': 1, 'No': 0}).astype('uint8')
    df['churn'] = df.churn.map({'Yes': 1, 'No': 0}).astype('uint8')
    df['contract_type'] = df.contract_type.map({'Month-to-month':0, 'One year':1, 'Two year':2}).astype('uint8')
    df['internet_service_type'] = df.internet_service_type.map({'None':0, 'DSL':1, 'Fiber optic':2}).astype('uint8')
    df['dependents'] = df.dependents.map({'Yes': 1, 'No': 0}).astype('uint8')
    df['partner'] = df.partner.map({'Yes': 1, 'No': 0}).astype('uint8')

   # for col in service_features:
        #dataframe[col] = dataframe[col].map({'Yes': 1, 'No': 0}).astype('uint8')
        #print(col)
    
    # Get dummies for non-binary categorical variables
    dummy_df = pd.get_dummies(df[['payment_type']], dummy_na=False, drop_first=True)
    
    # Concatenate dummy dataframe with the original 
    df = pd.concat([df, dummy_df], axis=1)

    #drop columns with low significance and columns where dummies where created aut
    df.drop(columns = ['gender', 'senior_citizen', \
                     'phone_service', 'total_charges', 'payment_type'],
                   inplace = True)
    
    return df

### SPLIT FUNCTION ###


def full_split(df, target):
    '''
    accepts a data frame and the name of the target variable as a parameter
    splits the data frame into:
    X_train, X_validate, X_test, y_train, y_validate, y_test
    '''
    train, validate, test = train_validate_test_split(df, target)

    #save target column
    y_train = train[target]
    y_validate = validate[target]
    y_test = test[target]

    #remove target column from the sets
    train.drop(columns = target, inplace=True)
    validate.drop(columns = target, inplace=True)
    test.drop(columns = target, inplace=True)

    return train, validate, test, y_train, y_validate, y_test

def train_validate_test_split(df, target, seed=2912):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    #split_db class verision with random seed
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test


######### HELPERS ##########
def get_cat_variables(df):
    '''
    this functions accepts a data frame as a parameter
    returns a list with categorical columns of the data frame
    '''
    #create a list
    cat_vars = []

    #find all object categories, exclude customer_id
    for col in df.columns:
        if (df[col].dtype == 'O' or df[col].dtype == 'category') and col != 'customer_id':
            cat_vars.append(col)
    return cat_vars

def to_category(df):
    '''
    this function transforms data type=object
    into data type = category
    returns a data frame with the new data type
    '''
    cat = get_cat_variables(df)
    for col in cat:
        df[col] = df[col].astype('category')
    return df

def transform_services(df):
    '''
    this functions accepts a data frame as a parameter
    goes through the additional services and replaces 
    values 'No phone service', 'No internet service' with 'No'
    '''
    for col in service_features:
        df[col] = df[col].replace([ 'No phone service', 'No internet service'], 'No')
    

def add_services_number(df):
    '''
    this functions accepts a data frame as a parameter
    counts the number of additional services per customer and 
    creates a new column with the number the total number of 
    additional service
    '''
    for col in service_features:
        df[col] = df[col].map({'Yes': 1, 'No': 0}).astype('uint8')
    df['add_services'] = df[service_features].sum(axis=1)