import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    data_file = [messages_filepath, categories_filepath]
    print('Loading data from',data_file[0],'and',data_file[1],':')
    # read in file
    ##load messages 
    messages = pd.read_csv(data_file[0])
    #load categories
    categories = pd.read_csv(data_file[1])
    ##merge dfs
    df = messages.merge(categories,on='id')
    return df


def clean_data(df):
    ##split cates
    cates_names = list(pd.Series(df['categories'][0].split(';'))
        .apply(lambda x: (x.split('-')[0])))
    cates = df['categories'].apply(lambda y: pd.Series(y.split(';'))
        .apply(lambda x: (x.split('-')[1])))
    cates.columns = cates_names
    ## concate splitted cates
    df = pd.concat([df,cates],1)
    ##drop categories column
    df = df.drop(['categories'],1)

    ##remove duplicates
    pd.Series(df.groupby('message')['id'].count().sort_values(ascending=False).values).value_counts()
    message_count = pd.Series(df.groupby('message')['id'].count().sort_values(ascending=False))
    dublicate_messages = message_count[message_count>1]

    duplicates_index = []
    for msg in dublicate_messages.index.values:
        duplicates_index = duplicates_index + list(df[df['message'] == msg].index.values[1:])
    df.drop(duplicates_index,inplace=True);
    return df


def save_data(df, database_filename):
    try:
      engine = create_engine('sqlite:///processed_messages.db')
      df.to_sql('messages', engine, if_exists='replace', index=False)
      return True
    except:
      print('Error saving the data into the database')
      return False


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