'''                                      Number of code lines in this Jupyternotebook
                   Source: https://stackoverflow.com/questions/56419105/get-the-total-number-of-code-lines-in-python                               '''
from datetime import datetime
import pandas as pd
pd.options.mode.chained_assignment = None # avoid error 
import numpy as np 

def count_code_lines(filename):
    with open(filename) as f:
        cnt = sum(1 for line in f)
        print(f'There are {cnt} code lines in {filename}')

def gdr(source_df, new_df):
    """Returns just the rows from the new dataframe that differ from the source dataframe"""
    merged_df = source_df.merge(new_df, indicator=True, how='outer')
    changed_rows_df = merged_df[merged_df['_merge'] == 'right_only']
    return (changed_rows_df.drop('_merge', axis=1),source_df.equals(new_df))

def N_Na(df):
    '''N° of null'''
    Null = df.isnull().sum(axis = 0)
    '''N° of no-null values'''
    No_Null = df.notnull().sum()
    print(f'There are {Null} Na values and {No_Null} values')
    
    
'''                                          Data Preparation: 1-> Collected data, 0-> GLMmodel                         '''

def N_Na_array(array):
    N = np.count_nonzero(np.isnan(array)) # How many nan are there
    return N


def full_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        # more options can be specified also
        print(df)