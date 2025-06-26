import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import kmeans  

#load file
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
        print(f"missing value : {df['_value'].isna().sum()}")
        print(f"missing time : {df['_time'].isna().sum()}")
        print(f"missing field : {df['_field'].isna().sum()}")
        #ideally : all equals to 0


        ######## timestamp ##########

        print('\n==== timestamp ====')
        df['timestamp'] = pd.to_datetime(df['_time'])
        print(df['timestamp'])
        print(f"\ntime range : from {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"total : {(df['timestamp'].max()-df['timestamp'].min()).days + 1}days") 

        # time intervals check
        df['time_diff'] = df['timestamp'].diff()  # calculate time differences
        df['time_diff_seconds'] = df['time_diff'].dt.total_seconds()  #convert to seconds
        intervals = df['time_diff_seconds'].dropna()  #analyze intervals

        print(intervals)

        print((intervals.max())//60)
        print((intervals.min())//60)

        if ((intervals.max())//60) == ((intervals.min())//60):
            print('\ninterval check pass !!!')
            print(f'interval : {intervals/60:.2f} min')

        else:
            unequal_interval_plot(df)
            print('\ninterval check fail !!!')
            print(f"Mean interval: {intervals.mean():.1f} seconds")
            print(f"Median interval: {intervals.median():.1f} seconds")
            print(f"Min interval: {intervals.min():.1f} seconds")
            print(f"Max interval: {intervals.max():.1f} seconds")
            


        ######## power value ##########
        
        print('\n==== power ====')
        df['power'] = pd.to_numeric(df['_value'])
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

        return df

#kmeans analysis
def kmeans_analysis(df):
    
    print('\n==== kmeans analysis ====')
    df['power_state'] = kmeans.get_power_classification(df)
    df_kmeans = df.copy()
    df_kmeans['power_category'] = df['power_state']
    df_kmeans_result = kmeans.kmeans_power(df_kmeans)
    kmeans.plot_kmeans_power(df_kmeans_result)
    kmeans.plot_power_classification(df_kmeans_result)

    return df

#kmeans classification
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

#power distribution
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

#unequal interval plot
def unequal_interval_plot(df):
    print('\n==== unequal interval plot ====')

    intervals = df['time_diff_seconds'].dropna()
    
    plt.figure(figsize=(12, 6))
    
    x_values = range(len(intervals))
    plt.scatter(x_values, intervals, alpha=0.6, s=20, c='blue')
    
    mean_interval = intervals.mean()
    
    plt.axhline(y=mean_interval, color='red', linestyle='--', linewidth=2, 
                label=f'average: {mean_interval:.1f}sec')
    
    plt.title('Time Interval Scatter Plot', fontsize=14, fontweight='bold')
    plt.xlabel('Data Point Index')
    plt.ylabel('Time Interval (seconds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def quick_find_outliers(df, missing_data_file='missing_times.txt', fuzzy_data_file='fuzzy_input.csv', threshold_seconds=1800):
    print('\n==== outliers ====')
    
    large_gaps = df[df['time_diff_seconds'] > threshold_seconds]
    
    if len(large_gaps) == 0:
        print("no outliers")
        return
    
    print(f"found {len(large_gaps)} interval outliers:")
    
    # Collect all missing time points
    missing_times = []
    
    # Collect data for fuzzy logic
    fuzzy_data = []
    
    for idx, row in large_gaps.iterrows():
        minutes = row['time_diff_seconds'] / 60  # Convert to minutes
        
        print(f"\noutlier index {idx} - interval {minutes:.1f}min:")
        print("-" * 50)
        
        if idx-1 >= 0 and idx-1 in df.index:
            prev_data = df.loc[idx-1]
            prev_minutes = prev_data['time_diff_seconds'] / 60 if pd.notna(prev_data['time_diff_seconds']) else 0
            print(f"index {idx-1}: {prev_data['timestamp']} - power {prev_data['power']:.2f}W - interval {prev_minutes:.1f}min")

        print(f"index {idx}: {row['timestamp']} - power {row['power']:.2f}W - interval {minutes:.1f}min >>> outlier")
         
        if idx+1 < len(df) and idx+1 in df.index:
            next_data = df.loc[idx+1]
            next_minutes = next_data['time_diff_seconds'] / 60 if pd.notna(next_data['time_diff_seconds']) else 0
            print(f"index {idx+1}: {next_data['timestamp']} - power {next_data['power']:.2f}W - interval {next_minutes:.1f}min")
        
        # Calculate missing time points
        missing_count = 0
        power_before = None
        power_after = row['power']
        
        if idx-1 >= 0 and idx-1 in df.index:
            start_time = pd.to_datetime(df.loc[idx-1]['timestamp'])
            end_time = pd.to_datetime(row['timestamp'])
            power_before = df.loc[idx-1]['power']
            
            # Assume normal interval is 15 minutes, find missing time points in between
            current_time = start_time + timedelta(minutes=15)
            while current_time < end_time:
                missing_times.append(current_time.strftime('%Y-%m-%d %H:%M'))
                missing_count += 1
                current_time += timedelta(minutes=15)
            
            # Collect fuzzy logic data
            power_change = abs(power_after - power_before) if power_before is not None else 0
            power_change_ratio = power_change / max(power_before, 1) if power_before is not None else 0
            
            fuzzy_record = {
                'gap_index': idx,
                'gap_start_time': start_time.strftime('%Y-%m-%d %H:%M'),
                'gap_end_time': end_time.strftime('%Y-%m-%d %H:%M'),
                'gap_duration_minutes': minutes,
                'missing_records_count': missing_count,
                'power_before': power_before,
                'power_after': power_after,
                'power_change': round(power_change, 2),
                'power_change_ratio': round(power_change_ratio, 3),
                'flag': 1  # Simple flag to indicate data gap
            }
            fuzzy_data.append(fuzzy_record)
        
        print()
    
    # Output missing time points to file
    if missing_times:
        with open(missing_data_file, 'w', encoding='utf-8') as f:
            f.write("Missing data time points:\n")
            f.write("=" * 30 + "\n")
            for i, missing_time in enumerate(missing_times, 1):
                f.write(f"{i}. {missing_time}\n")
                print(f"Missing: {missing_time}")
        
        print(f"\nTotal missing records: {len(missing_times)}")
        print(f"Missing time points saved to: {missing_data_file}")
    
    # Output fuzzy logic data to CSV file
    if fuzzy_data:
        fuzzy_df = pd.DataFrame(fuzzy_data)
        fuzzy_df.to_csv(fuzzy_data_file, index=False)
        print(f"Fuzzy logic data saved to: {fuzzy_data_file}")
        print(f"Fuzzy data summary:")
        print(f"  - Total gaps: {len(fuzzy_data)}")
        print(f"  - Average gap duration: {fuzzy_df['gap_duration_minutes'].mean():.1f} min")
        print(f"  - Total missing records: {fuzzy_df['missing_records_count'].sum()}")
        print(f"  - All gaps flagged: {fuzzy_df['flag'].sum()}")
    
    return missing_times, fuzzy_data

#output file
def save_data_csv(df, output_path='data_after_preprocessing.csv'):
    
    print('\n==== Saving Processed Data ====')

    output_columns = [
        'timestamp', 'date', 
        'power', 'power_state', 'is_phantom_load', 'is_off', 'is_on', 
        'is_light_use', 'is_regular_use', 'time_diff_seconds', 'outlier_flag'
    ]

    # Only include columns that exist in the DataFrame
    available_columns = [col for col in output_columns if col in df.columns]

    df_output = df[available_columns].copy()
    df_output.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'Data saved to {output_path}')
    print(f'Saved {len(df_output)} rows and {len(available_columns)} columns')
    
    return df_output

if __name__ == "__main__":

    file_path = 'C:/Users/王俞文/Documents/glasgow/msc project/data/6CL-S8 television 30days_15min.csv'

    print('====================== data preprocessing ======================')

    df = load_data(file_path)

    outliers = quick_find_outliers(df)

    df = power_distribution(df)

    df_output = save_data_csv(df)
