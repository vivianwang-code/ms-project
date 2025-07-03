import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

#load data
def load_processed_data(file_path='data_after_preprocessing.csv'):

    print("=== loading data ===")

    df = pd.read_csv(file_path)

    if len(df) == 0:
        print("No data found in the file.")

    else:
        print(f"total : {len(df)} records")
        print(f"time range : {df['timestamp'].min()} to {df['timestamp'].max()}")

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df
    
#control target
def control_target(df):

    print("==== control target ====")
    
    idle_state_mapping = {
        'non use': 'no control',         
        'phantom load': 'target', # main target !!!
        'light use': 'no control',          
        'regular use': 'no control'  
    }


    #create target field
    df['control_target'] = df['power_state'].map(idle_state_mapping)

    # Check if there is any unmapped state
    unmapped = df[df['control_target'].isna()]
    if len(unmapped) > 0:
        print(f"find {len(unmapped)} data are unmapped:")
        print(unmapped['power_state'].value_counts())
    else:
        print("All states have been mapped successfully")

    # mapping result distribution
    print("mapping result statistics:")
    idle_distribution = df['control_target'].value_counts()
    for state, count in idle_distribution.items():
        percentage = count / len(df) * 100
        print(f"  {state}: {count:,} records ({percentage:.1f}%)")
    
    return df

#analyze power consumption by state
def analyze_power_consumption_by_state(df):
    print("==== analyze power consumption by state ====")

    df['']



if __name__ == "__main__":
    # Load the processed data
    df = load_processed_data()

    control_target(df)
