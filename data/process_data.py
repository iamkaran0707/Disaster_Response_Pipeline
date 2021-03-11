import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    """ Here we will have to load and merge the 2 files/dataset for the project used!!
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = pd.merge(messages,categories,on = 'id')
    return df


def clean_data(df):
    
    """Here we have to clean the dataset: which includes creating the columns for categories dataset that we will achieve by splitting the values
    """
    
    categories = df.categories.str.split(';',expand = True)
    row = categories.iloc[1,:]
    columns = row.apply(lambda x : x.split('-')[0]).tolist()
    categories.columns = columns
    
    ### now convert all values of categories in only numbers
    for i in columns:
        categories[i] = categories[i].str.split('-').str[1]
        categories[i] = categories[i].astype(int)
    
    index_to_drop = categories[categories['related']==2].index
    categories = categories.iloc[~categories.index.isin(index_to_drop)]
    
    df.drop('categories',axis=1,inplace=True)
    df_new = pd.concat([df,categories],axis=1, join = 'inner')
    df_new = df_new.drop_duplicates()
    return df_new


def save_data(df, database_filename):
    """Save the work that we have done in sql format"""
    
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql('Disaster',con = engine, index=False, if_exists = 'replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()