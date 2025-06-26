import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_processed_data(file_path='data_after_preprocessing.csv'):

    print("=== loading data ===")
    
    try:
        df = pd.read_csv(file_path)
        print(f"total : {len(df)} records")
        print(f"time range : {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        #ensure the timestamp column
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        #checking columns
        required_columns = ['timestamp', 'power', 'power_state']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"missing columns: {missing_columns}")
            return None
        
        print(f"data completeness check passed")
        return df
        
    except FileNotFoundError:
        print(f"file not found error : {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return None

def create_idle_state_mapping(df):

    print("\n=== idle state mapping ===")
    
    #define state mapping rules
    idle_state_mapping = {
        'non use': 'deep_idle',         
        'phantom load': 'standby_idle', # main target !!!
        'light use': 'active',          
        'regular use': 'active'         
    }
    
    #create idle state field
    df['idle_state'] = df['power_state'].map(idle_state_mapping)
    
    # Check if there is any unmapped state
    unmapped = df[df['idle_state'].isna()]
    if len(unmapped) > 0:
        print(f"find {len(unmapped)} data are unmapped:")
        print(unmapped['power_state'].value_counts())
    else:
        print("All states have been mapped successfully")
    
    # mapping result distribution
    print("mapping result statistics:")
    idle_distribution = df['idle_state'].value_counts()
    for state, count in idle_distribution.items():
        percentage = count / len(df) * 100
        print(f"  {state}: {count:,} records ({percentage:.1f}%)")
    
    return df

def create_control_target(df):
    
    print("\n=== create control target ===")
    
    # control strategies
    control_strategies = {
        'deep_idle': 'no_control',      
        'standby_idle': 'primary_target',  #main target
        'active': 'no_control'          
    }
    
    # control target field
    df['control_target'] = df['idle_state'].map(control_strategies)
    
    # Create binary control labels
    df['needs_control'] = (df['control_target'] == 'primary_target')
    
    # control statistics
    control_stats = df['control_target'].value_counts()
    needs_control_count = df['needs_control'].sum()
    needs_control_percentage = needs_control_count / len(df) * 100
    
    print("Control target statistics:")
    for target, count in control_stats.items():
        percentage = count / len(df) * 100
        print(f"  {target}: {count:,} records ({percentage:.1f}%)")
    
    print(f"\n Time point that needs to be controlled: {needs_control_count:,} records ({needs_control_percentage:.1f}%)")
    
    return df

def analyze_power_consumption_by_state(df):
    
    print("\n=== Power consumption analysis ===")
    
    power_analysis = df.groupby(['idle_state', 'power_state'])['power'].agg([
        'count', 'min', 'max', 'mean', 'std'
    ]).round(2)
    
    print(power_analysis)
    
    # calculate standby power
    standby_data = df[df['idle_state'] == 'standby_idle']
    if len(standby_data) > 0:
        avg_standby_power = standby_data['power'].mean()
        standby_hours = len(standby_data) * 5 / 3600  # 5 sec
        daily_standby_energy = avg_standby_power * standby_hours * 24 / standby_hours if standby_hours > 0 else 0
        annual_standby_energy = daily_standby_energy * 365 / 1000  # kWh
        
        print(f"  Standby power analysis:")
        print(f"  Average standby power: {avg_standby_power:.1f}W")
        print(f"  Daily standby time: {standby_hours:.1f} hour")
        print(f"  Annual standby power consumption: {annual_standby_energy:.1f} kWh")
        print(f"  Annual electricity bill (assume $0.15/kWh): ${annual_standby_energy * 0.15:.0f}")
    
    return power_analysis

def analyze_time_patterns(df):
    
    print("\n=== Temporal pattern analysis ===")
    
    # Hourly status distribution
    hourly_patterns = df.groupby(['hour', 'idle_state']).size().unstack(fill_value=0)
    hourly_percentage = hourly_patterns.div(hourly_patterns.sum(axis=1), axis=0) * 100
    
    print("Hourly status distribution (%):")
    print(hourly_percentage.round(1))
    
    # Find out the high idle time
    standby_hours = hourly_percentage['standby_idle'].sort_values(ascending=False)
    print(f"The most idle time:")
    for hour, percentage in standby_hours.head(5).items():
        print(f"  {hour:02d} o'clock: {percentage:.1f}%")
    
    # Weekend vs Weekday Analysis
    weekend_analysis = df.groupby(['weekend', 'idle_state']).size().unstack(fill_value=0)
    weekend_percentage = weekend_analysis.div(weekend_analysis.sum(axis=1), axis=0) * 100
    
    print(f"\nWeekend vs weekday status distribution (%):")
    print(weekend_percentage.round(1))
    
    return hourly_patterns, weekend_analysis

def detect_state_transitions(df):

    print("\n=== detect state transition ===")
    
    # Calculating state changes
    df['prev_idle_state'] = df['idle_state'].shift(1)
    df['state_changed'] = df['idle_state'] != df['prev_idle_state']
    
    # Finding state transitions
    transitions = df[df['state_changed'] & df['prev_idle_state'].notna()]
    
    if len(transitions) > 0:
        # Analyze conversion types
        transition_patterns = transitions.groupby(['prev_idle_state', 'idle_state']).size()
        
        print("State transition statistics:")
        for (prev_state, current_state), count in transition_patterns.items():
            print(f"  {prev_state} → {current_state}: {count} times")
        
        # key transition：active → standby_idle
        critical_transitions = transitions[
            (transitions['prev_idle_state'] == 'active') & 
            (transitions['idle_state'] == 'standby_idle')
        ]
        
        if len(critical_transitions) > 0:
            print(f"\nkey transition (active → standby_idle): {len(critical_transitions)} times")
            
            transition_hours = critical_transitions['hour'].value_counts().sort_index()
            print("transition time:")
            for hour, count in transition_hours.items():
                print(f"  {hour:02d} o'clock : {count} times")
    
    return transitions

def visualize_state_analysis(df):

    print("\n=== visualize state analysis ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # figure 1
    state_counts = df['idle_state'].value_counts()
    colors = ['red', 'orange', 'lightblue']
    axes[0, 0].pie(state_counts.values, labels=state_counts.index, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
    axes[0, 0].set_title('Idle State Distribution')
    
    # figure 2
    control_counts = df['control_target'].value_counts()
    axes[0, 1].bar(control_counts.index, control_counts.values, 
                   color=['lightcoral', 'lightgreen'])
    axes[0, 1].set_title('Control Target Distribution')
    axes[0, 1].set_ylabel('Count')
    plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
    
    # figure 3
    hourly_data = df.groupby(['hour', 'idle_state']).size().unstack(fill_value=0)
    hourly_data.plot(kind='bar', stacked=True, ax=axes[1, 0], 
                     color=['red', 'orange', 'lightblue'])
    axes[1, 0].set_title('Hourly State Distribution')
    axes[1, 0].set_xlabel('Hour of Day')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend(title='Idle State')
    
    # figure 4
    for state in df['idle_state'].unique():
        state_data = df[df['idle_state'] == state]
        axes[1, 1].scatter(state_data['hour'], state_data['power'], 
                          label=state, alpha=0.6, s=10)
    axes[1, 1].set_title('Power vs Time by State')
    axes[1, 1].set_xlabel('Hour of Day')
    axes[1, 1].set_ylabel('Power (W)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def save_results(df, output_file='data_with_control_features.csv'):

    print(f"\n=== save results ===")
    
    output_columns = [
        'timestamp', 'date', 'hour', 'day_of_week', 'weekend',
        'power', 'power_state', 
        'idle_state', 'control_target', 'needs_control',
        'time_diff_seconds'
    ]
    
    available_columns = [col for col in output_columns if col in df.columns]
    
    df_output = df[available_columns].copy()
    df_output.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"saving file to : {output_file}")
    
    return df_output

def generate_summary_report(df):

    print('generate summary report')

    # statistics
    total_records = len(df)
    time_span = (df['timestamp'].max() - df['timestamp'].min()).days + 1
    
    print(f"  data :")
    print(f"  total records : {total_records:,}")
    print(f"  time span : {time_span} days")
    print(f"  Average daily record : {total_records/time_span:.0f} records")
    
    # state distribution
    print(f"\n state distribution:")
    for state in df['idle_state'].value_counts().index:
        count = df['idle_state'].value_counts()[state]
        percentage = count / total_records * 100
        print(f"  {state}: {count:,} records ({percentage:.1f}%)")
    
    # control opportunities
    control_opportunities = df['needs_control'].sum()
    control_percentage = control_opportunities / total_records * 100
    
    print(f"\n control opportunities :")
    print(f"  Time point that needs to be controlled : {control_opportunities:,} records ({control_percentage:.1f}%)")
    
    standby_data = df[df['idle_state'] == 'standby_idle']
    if len(standby_data) > 0:
        avg_standby_power = standby_data['power'].mean()
        print(f"  Average standby power: {avg_standby_power:.1f}W")

def main():

    df = load_processed_data()

    df = create_idle_state_mapping(df)

    df = create_control_target(df)

    power_analysis = analyze_power_consumption_by_state(df)

    hourly_patterns, weekend_analysis = analyze_time_patterns(df)

    transitions = detect_state_transitions(df)

    visualize_state_analysis(df)

    df_output = save_results(df)

    generate_summary_report(df)
    
    return df_output

if __name__ == "__main__":
    result_df = main()