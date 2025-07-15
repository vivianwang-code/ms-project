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


def quick_find_outliers(df, threshold_seconds=1800):
    import pandas as pd
    from datetime import timedelta
    
    print('\n==== outliers ====')
    
    large_gaps = df[df['time_diff_seconds'] > threshold_seconds]
    
    if len(large_gaps) == 0:
        print("no outliers")
        return
    
    print(f"found {len(large_gaps)} interval outliers:")
    
    # 用來統計各時間點出現次數的字典
    time_count = {}
    
    for idx, row in large_gaps.iterrows():
        if idx-1 >= 0 and idx-1 in df.index:
            start_time = df.loc[idx-1, 'timestamp']
            end_time = row['timestamp']
            
            # 確保是 datetime 格式
            if not isinstance(start_time, pd.Timestamp):
                start_time = pd.to_datetime(start_time)
            if not isinstance(end_time, pd.Timestamp):
                end_time = pd.to_datetime(end_time)
            
            # 計算缺失的時間點（假設正常間隔是15分鐘）
            current_time = start_time + timedelta(minutes=15)
            while current_time < end_time:
                time_str = current_time.strftime('%H:%M')
                
                # 統計這個時間點出現的次數
                if time_str in time_count:
                    time_count[time_str] += 1
                else:
                    time_count[time_str] = 1
                    
                current_time += timedelta(minutes=15)
    
    # 按時間順序排序並顯示統計結果
    if time_count:
        print('\nmissing time point statistics:')
        sorted_times = sorted(time_count.items())
        for time_str, count in sorted_times:
            print(f"{time_str}: {count}counts")
        
        print(f'\n total: {sum(time_count.values())}counts of missing data')
        print(f'time points: {len(time_count)}counts of time points')
        
        # 匯出CSV檔案
        time_stats_df = pd.DataFrame(sorted_times, columns=['time', 'missing_time'])
        csv_filename = 'missing_time_statistics.csv'
        time_stats_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f'\n missing data saved to : {csv_filename}')
    
    return large_gaps

#output file
def save_data_csv(df, output_path='data_after_preprocessing.csv'):
    
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
    
    return df_output

if __name__ == "__main__":

    # file_path = "C:/Users/王俞文/OneDrive - University of Glasgow/文件/glasgow/msc project/data/6CL-S8 television 15min.csv"

    # file_path = "C:/Users/王俞文/OneDrive - University of Glasgow/文件/glasgow/msc project/data/6CL-S2_washing_machine_15min.csv"
    file_path = "C:/Users/王俞文/OneDrive - University of Glasgow/文件/glasgow/msc project/data/iotproject_D8.csv"
    

    print('====================== data preprocessing ======================')

    df = load_data(file_path)

    outliers = quick_find_outliers(df)

    df = power_distribution(df)

    df_output = save_data_csv(df)
