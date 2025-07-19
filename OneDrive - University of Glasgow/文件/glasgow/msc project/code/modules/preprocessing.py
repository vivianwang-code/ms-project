import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from . import kmeans  

def load_data(file_path):
    print('loading file')

    if file_path == None:
        print('no file')
        return None
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            print("loading file successfully")
            lines = f.readlines()

        print("first ten lines:")
        for i, line in enumerate(lines[:10]):
            print(f"line{i+1}:{line.strip()}")

        ##find where to start
        start_data = 0
        for i, line in enumerate(lines):
            if not line.startswith('#') and ',result' in line:
                start_data = i
        print(f"start with {start_data + 1}")
        print(lines[start_data+1])

        ##load the data
        df = pd.read_csv(file_path, skiprows=start_data)
        print('\n==== original data ====')
        print(f"shape of the data : {df.shape}")
        print(f"colums : {list(df.columns)}")
        print("the first 5 lines")
        print(df.head())

        print('\n==== checking missing data ====')
        print(f"missing value : {df['timestamp'].isna().sum()}")
        print(f"missing time : {df['power_value'].isna().sum()}")
        #ideally : all equals to 0


        ######## timestamp ##########

        print('\n==== timestamp ====')
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
        print(df['timestamp'])
        print(f"\ntime range : from {df['timestamp'].min()} to {df['timestamp'].max()}")
        total_days = (df['timestamp'].max()-df['timestamp'].min()).days + 1
        print(f"total : {total_days}days") 


        ####### power value ##########
        
        print('\n==== power ====')
        df['power'] = pd.to_numeric(df['power_value'])
        print(f"min : {df['power'].min():.2f}W") 
        print(f"max : {df['power'].max():.2f}W") 
        print(f"mean : {df['power'].mean():.2f}W")

        df = kmeans_analysis(df)

        print("\n==== power distribution =====")
        state_counts = df['power_state'].value_counts()
        for state, count in state_counts.items():
            percentage = count / len(df['power']) * 100
            print(f'{state} : {count} (percentage : {percentage:.2f}%)')

        df = binary_classification(df)

        return df, total_days

#kmeans analysis
def kmeans_analysis(df):
    
    print('\n==== kmeans analysis ====')
    df['power_state'] = kmeans.get_power_classification(df)
    df_kmeans = df.copy()
    df_kmeans['power_category'] = df['power_state']

    return df

def binary_classification(df):

    print('==== power binary classification ====')
    
    #off
    df['is_off'] = (df['power_state'] == 'non use')
    
    #phantom load
    df['is_phantom_load'] = (df['power_state'] == 'phantom load')
    
    #on : light use / regular use
    df['is_light_use'] = (df['power_state'] == 'light use')
    df['is_regular_use'] = (df['power_state'] == 'regular use')
    df['is_on'] = (df['power'] > 0)

    #print statistics
    print(f"Off: {df['is_off'].sum()} ({df['is_off'].mean()*100:.1f}%)")
    print(f"Phantom Load: {df['is_phantom_load'].sum()} ({df['is_phantom_load'].mean()*100:.1f}%)")
    print(f"Light Use: {df['is_light_use'].sum()} ({df['is_light_use'].mean()*100:.1f}%)")
    print(f"Regular Use: {df['is_regular_use'].sum()} ({df['is_regular_use'].mean()*100:.1f}%)")
    print(f"On (any power): {df['is_on'].sum()} ({df['is_on'].mean()*100:.1f}%)")
    
    return df


def power_distribution(df):

    print('\n==== power distribution ====')
    power_statistic = df['power'].describe()
    print("Power Statistics:")
    print(power_statistic)

    state_distribution = df['power_state'].value_counts()
    state_dis_percentage = df['power_state'].value_counts(normalize=True)*100

    print('\nUsage Distribution:')
    for state in state_distribution.index:
        print(f"{state} : {state_distribution[state]} times ({state_dis_percentage[state]:.2f}%)")

    ##phantom load analysis
    phantom_data = df[df['is_phantom_load']]
    if len(phantom_data) > 0:
        print('\nPhantom Load Analysis:')
        print(f'Number of phantom load readings: {len(phantom_data)}')
        print(f'Phantom load percentage: {len(phantom_data)/len(df)*100:.2f}%')
        print(f'Phantom load average: {phantom_data['power'].mean():.2f}W')
        print(f'Phantom load range: {phantom_data['power'].min():.2f}W ~ {phantom_data['power'].max():.2f}W')

        average_phantom_power = phantom_data['power'].mean()
        phantom_hours_per_day = len(phantom_data) / len(df) * 24
        annual_phantom_load = average_phantom_power * phantom_hours_per_day * 365 / 1000

        print(f"Estimated phantom hours per day: {phantom_hours_per_day:.2f} hours")
        print(f"Estimated annual phantom load: {annual_phantom_load:.2f} kWh")
    else:
        print('No phantom load detected.')
    
    return df

def save_data_csv(df, output_path='data/data_after_preprocessing.csv'):
    
    print('\n==== Saving Processed Data ====')

    output_columns = [
        'timestamp', 'date', 
        'power', 'power_state', 'is_phantom_load', 'is_off', 'is_on', 
        'is_light_use', 'is_regular_use', 'time_diff_seconds'
    ]

    # Only include columns that exist in the DataFrame
    available_columns = [col for col in output_columns if col in df.columns]

    df_output = df[available_columns].copy()
    df_output.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'Data saved to {output_path}')
    print(f'Saved {len(df_output)} rows and {len(available_columns)} columns')
    
    return output_path

def data_preprocessing(file_path):
    df, total_days = load_data(file_path)

    df = power_distribution(df)

    output_path = save_data_csv(df)

    return total_days, output_path