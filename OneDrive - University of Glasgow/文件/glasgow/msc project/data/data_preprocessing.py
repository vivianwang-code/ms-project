import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import kmeans  # Import kmeans module

def load_data(file_path):
    print("start")

    if file_path == None:
        print("no file")
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
        print(f"shape of the data : {df.shape}")
        print(f"colums : {list(df.columns)}")
        print("the first 5 lines")
        print(df.head())

        ##checking the missing data
        print("checking the missing data")
        print(f"missing value : {df['_value'].isna().sum()}")
        print(f"missing time : {df['_time'].isna().sum()}")
        print(f"missing field : {df['_field'].isna().sum()}")
        #ideal : all equals to 0

        ##
        print(df['_field'].value_counts())

        ##timestamp
        print("timestamp")
        df['timestamp'] = pd.to_datetime(df['_time'])
        print(df['timestamp'])

        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['weekend'] = df['day_of_week'].isin([5,6])

        print(f"time range : from {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"total : {(df['timestamp'].max()-df['timestamp'].min()).days + 1}days") 

        ##power
        df['power'] = pd.to_numeric(df['_value'])

        print(f"min : {df['power'].min():.2f}W") 
        print(f"max : {df['power'].max():.2f}W") 
        print(f"mean : {df['power'].mean():.2f}W")

        ##finding negative power
        negative_power = df[df['power']<0]
        if len(negative_power) == 0:
            print("there is no negative power")
        else:
            print("Negative power found:")
            print(negative_power)        

        ##distribution
        print("distribution(hour) : ")
        hour_counts = df['hour'].value_counts().sort_index()
        print(f"with the most data : {hour_counts.idxmax()} o'clock ({hour_counts.max()})")
        print(f"with the less data : {hour_counts.idxmin()} o'clock ({hour_counts.min()})")

        ##usage time
        def usage_time(hour):
            if 6 <= hour <12 :
                return 'morning'
            elif 12 <= hour < 18:
                return 'afternoon'
            elif 18<= hour <22:
                return 'evening'
            else:
                return 'night'
            
        df['usage_time_state'] = df['hour'].apply(usage_time)

        # Run complete K-means analysis with all visualizations
        df = run_complete_kmeans_analysis(df, show_analysis=True)

        print("power distribution : ")
        state_counts = df['power_state'].value_counts()
        for state, count in state_counts.items():
            percentage = count / len(df['power']) * 100
            print(f'{state} : {count} (percentage : {percentage:.2f}%)')
        
        # Fix binary classification to match K-means results
        df = update_binary_classification(df)

        display_columns = ['timestamp', 'hour', 'weekend', 'usage_time_state', 'power', 'power_state', 'is_phantom_load']
        print("\nSample processed data:")
        print(df[display_columns].head(10))

        return df

def run_complete_kmeans_analysis(df, show_analysis=True):

    if show_analysis:
        print("\n=== Starting Complete K-means Power Analysis ===")
        
        # Get power classification results
        df['power_state'] = kmeans.get_power_classification(df)
        
        # Create a DataFrame with power_category column for kmeans visualization functions
        df_kmeans = df.copy()
        df_kmeans['power_category'] = df['power_state']
        
        # Run complete kmeans analysis and visualization
        print("\n--- Running Complete K-means Analysis ---")
        df_kmeans_result = kmeans.kmeans_power(df_kmeans)
        
        print("\n--- Generating K-means Power Distribution Plots ---")
        kmeans.plot_kmeans_power(df_kmeans_result)
        
        print("\n--- Generating Power Classification Scatter Plot ---")
        kmeans.plot_power_classification(df_kmeans_result)
        
        print("=== Complete K-means Analysis Finished ===\n")
    else:
        # Just get classification without plots
        print("\n=== Getting K-means Classification Only ===")
        df['power_state'] = kmeans.get_power_classification(df)
        print("=== Classification Complete ===\n")
    
    return df

def update_binary_classification(df):
    """
    Update binary classification to match K-means results
    """
    print("\n=== Updating Binary Classification to Match K-means Results ===")
    
    # Create binary columns based on power_state instead of fixed thresholds
    df['is_off'] = (df['power_state'] == 'non use')
    df['is_phantom_load'] = (df['power_state'] == 'phantom load')
    df['is_light_use'] = (df['power_state'] == 'light use')
    df['is_regular_use'] = (df['power_state'] == 'regular use')
    
    # Create simplified is_on column (any non-zero power)
    df['is_on'] = (df['power'] > 0)
    
    # Print updated statistics
    print("Updated Binary Classification Statistics:")
    print(f"Off: {df['is_off'].sum()} ({df['is_off'].mean()*100:.1f}%)")
    print(f"Phantom Load: {df['is_phantom_load'].sum()} ({df['is_phantom_load'].mean()*100:.1f}%)")
    print(f"Light Use: {df['is_light_use'].sum()} ({df['is_light_use'].mean()*100:.1f}%)")
    print(f"Regular Use: {df['is_regular_use'].sum()} ({df['is_regular_use'].mean()*100:.1f}%)")
    print(f"On (any power): {df['is_on'].sum()} ({df['is_on'].mean()*100:.1f}%)")
    
    return df

def analyze_time_intervals(df):
    """
    Analyze the time intervals in the data
    """
    print("\n=== Analyzing Time Intervals ===")
    
    # Convert timestamp to datetime if it's not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate time differences
    df['time_diff'] = df['timestamp'].diff()
    
    # Convert to seconds
    df['time_diff_seconds'] = df['time_diff'].dt.total_seconds()
    
    # Analyze intervals
    intervals = df['time_diff_seconds'].dropna()
    
    print("Time Interval Analysis:")
    print(f"Mean interval: {intervals.mean():.1f} seconds")
    print(f"Median interval: {intervals.median():.1f} seconds")
    print(f"Min interval: {intervals.min():.1f} seconds")
    print(f"Max interval: {intervals.max():.1f} seconds")
    print(f"Standard deviation: {intervals.std():.1f} seconds")
    
    # Most common intervals
    print("\nMost common intervals:")
    common_intervals = intervals.value_counts().head(10)
    for interval, count in common_intervals.items():
        print(f"{interval:.0f} seconds: {count} times")
    
    # Check for irregular intervals (gaps > 1 minute)
    large_gaps = intervals[intervals > 60]
    if len(large_gaps) > 0:
        print(f"\nFound {len(large_gaps)} gaps larger than 1 minute:")
        print(f"Largest gap: {large_gaps.max():.0f} seconds ({large_gaps.max()/60:.1f} minutes)")
    
    return df

def power_distribution(df):
    print('\n=== Power Distribution Analysis ===')

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

def generate_usage_summary(df):
    """
    Generate a comprehensive usage summary
    """
    print("\n=== Device Usage Summary ===")
    
    # Daily usage patterns
    daily_usage = df.groupby('date').agg({
        'power': ['mean', 'max', 'count'],
        'is_on': 'sum',
        'is_phantom_load': 'sum',
        'is_regular_use': 'sum'
    }).round(2)
    
    print("Daily Usage Statistics:")
    print(f"Total days analyzed: {len(daily_usage)}")
    print(f"Average power per day: {df.groupby('date')['power'].mean().mean():.2f}W")
    print(f"Peak power recorded: {df['power'].max():.2f}W")
    
    # Hourly patterns
    hourly_patterns = df.groupby('hour').agg({
        'power': 'mean',
        'is_on': 'mean',
        'is_phantom_load': 'mean',
        'is_regular_use': 'mean'
    }).round(3)
    
    print("\nPeak Usage Hours:")
    peak_hours = hourly_patterns['power'].nlargest(5)
    for hour, avg_power in peak_hours.items():
        print(f"Hour {hour:02d}: {avg_power:.1f}W average")
    
    # Weekend vs Weekday
    weekend_comparison = df.groupby('weekend').agg({
        'power': 'mean',
        'is_on': 'mean',
        'is_phantom_load': 'mean',
        'is_regular_use': 'mean'
    }).round(3)
    
    print("\nWeekend vs Weekday Usage:")
    print("Weekday average power:", weekend_comparison.loc[False, 'power'], "W")
    print("Weekend average power:", weekend_comparison.loc[True, 'power'], "W")
    
    return df

def save_data_csv(df, output_path='data_after_preprocessing.csv'):
    print(f'\n=== Saving Processed Data ===')
    
    output_columns = [
        'timestamp', 'date', 'hour', 'usage_time_state', 'day_of_week', 'weekend', 
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

def test_kmeans_integration(df):
    """
    Test function: Verify kmeans classification results
    """
    print("\n=== Testing K-means Integration Results ===")
    
    # Check if power_state column was successfully created
    if 'power_state' in df.columns:
        print("✓ power_state column successfully created")
        print(f"power_state data type: {df['power_state'].dtype}")
        print(f"power_state unique values: {df['power_state'].unique()}")
        print(f"power_state distribution:")
        print(df['power_state'].value_counts())
        
        # Check for missing values
        missing_count = df['power_state'].isna().sum()
        print(f"power_state missing values: {missing_count}")
        
        if missing_count == 0:
            print("✓ No missing values")
        else:
            print("⚠ Missing values found, need to check")
        
        # Check classification consistency
        power_ranges = df.groupby('power_state')['power'].agg(['min', 'max', 'mean']).round(2)
        print("\nPower ranges by classification:")
        print(power_ranges)
            
    else:
        print("✗ power_state column creation failed")
    
    print("=== Test Complete ===\n")

def plot_additional_analysis(df):
    """
    Create additional analysis plots
    """
    print("\n=== Generating Additional Analysis Plots ===")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Hourly power consumption
    hourly_power = df.groupby('hour')['power'].mean()
    axes[0, 0].plot(hourly_power.index, hourly_power.values, marker='o', linewidth=2)
    axes[0, 0].set_title('Average Power Consumption by Hour')
    axes[0, 0].set_xlabel('Hour of Day')
    axes[0, 0].set_ylabel('Average Power (W)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(range(0, 24, 2))
    
    # Plot 2: Weekend vs Weekday comparison
    weekend_data = df[df['weekend'] == True]['power']
    weekday_data = df[df['weekend'] == False]['power']
    
    axes[0, 1].hist([weekday_data, weekend_data], bins=20, alpha=0.7, 
                   label=['Weekday', 'Weekend'], color=['blue', 'orange'])
    axes[0, 1].set_title('Power Distribution: Weekday vs Weekend')
    axes[0, 1].set_xlabel('Power (W)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Usage time state distribution
    usage_time_counts = df['usage_time_state'].value_counts()
    axes[1, 0].pie(usage_time_counts.values, labels=usage_time_counts.index, 
                   autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title('Distribution by Time of Day')
    
    # Plot 4: Power state timeline (sample)
    sample_df = df.head(100)  # Use first 100 rows for timeline
    colors = {'non use': 'red', 'phantom load': 'orange', 'light use': 'blue', 'regular use': 'green'}
    
    for state in sample_df['power_state'].unique():
        state_data = sample_df[sample_df['power_state'] == state]
        axes[1, 1].scatter(range(len(state_data)), state_data['power'], 
                          c=colors.get(state, 'gray'), label=state, alpha=0.7)
    
    axes[1, 1].set_title('Power Timeline (First 100 readings)')
    axes[1, 1].set_xlabel('Reading Number')
    axes[1, 1].set_ylabel('Power (W)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("Additional analysis plots generated successfully")

if __name__ == "__main__":
    # file_path = '6CL-S8 television 30days'
    file_path = 'C:/Users/王俞文/Documents/glasgow/msc project/data/6CL-S8 television 30days_5s.csv'
    
    print("="*60)
    print("SMART DEVICE POWER CONSUMPTION ANALYSIS")
    print("="*60)
    
    # Load and process data with complete K-means analysis
    df = load_data(file_path)
    
    if df is not None:
        # Analyze time intervals
        df = analyze_time_intervals(df)
        
        # Test kmeans integration
        test_kmeans_integration(df)
        
        # Additional power distribution analysis
        df = power_distribution(df)
        
        # Generate comprehensive usage summary
        df = generate_usage_summary(df)
        
        # Create additional analysis plots
        plot_additional_analysis(df)
        
        # Save processed data
        df_output = save_data_csv(df)
        
        # Final data summary
        print(f"\n📊 FINAL DATA SUMMARY:")
        print(f"Total readings: {len(df):,}")
        print(f"Time period: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Total days: {(df['timestamp'].max() - df['timestamp'].min()).days + 1}")
        print(f"Power range: {df['power'].min():.1f}W - {df['power'].max():.1f}W")
        print(f"Average power: {df['power'].mean():.1f}W")
        
        print(f"\n🔌 DEVICE USAGE BREAKDOWN:")
        for state in df['power_state'].value_counts().index:
            count = df['power_state'].value_counts()[state]
            percentage = count / len(df) * 100
            avg_power = df[df['power_state'] == state]['power'].mean()
            print(f"{state}: {count:,} readings ({percentage:.1f}%) - Avg: {avg_power:.1f}W")
    
    else:
        print("❌ Failed to load data. Please check the file path.")