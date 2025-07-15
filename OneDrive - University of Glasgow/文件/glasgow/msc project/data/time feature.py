import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta


def load_control_features_data(file_path='data_with_control_features.csv'):

    print("=== Loading Control Features Data ===")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded: {len(df)} records")
        print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Ensure timestamp is datetime format
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp for proper sequential analysis
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Check required columns
        required_columns = ['timestamp', 'power', 'idle_state', 'needs_control']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return None
        
        print("Data integrity check passed")
        return df
        
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        print("Please run Task 1 (smart_control_preprocessing.py) first")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_improved_duration_features(df):

    print("\n=== Calculating Duration Features (Improved) ===")
    
    # Initialize duration features
    df['standby_duration'] = 0.0
    df['active_duration'] = 0.0 
    df['off_duration'] = 0.0
    df['time_since_last_active'] = 0.0
    
    # Ensure we have time_diff_seconds for calculation
    if 'time_diff_seconds' not in df.columns:
        df['time_diff_seconds'] = df['timestamp'].diff().dt.total_seconds()
        df['time_diff_seconds'].fillna(0, inplace=True)
    
    print(f"Time interval statistics:")
    print(f"   Min interval: {df['time_diff_seconds'].min():.0f} seconds")
    print(f"   Max interval: {df['time_diff_seconds'].max():.0f} seconds") 
    print(f"   Mean interval: {df['time_diff_seconds'].mean():.0f} seconds")
    
    # Method 1: Calculate consecutive state durations using cumulative time
    current_standby_duration = 0.0
    current_active_duration = 0.0
    current_off_duration = 0.0
    
    for i in range(len(df)):
        current_state = df.loc[i, 'idle_state']
        time_diff_minutes = df.loc[i, 'time_diff_seconds'] / 60.0  # Convert to minutes
        
        # Check if state changed from previous record
        if i > 0:
            prev_state = df.loc[i-1, 'idle_state']
            
            if current_state == prev_state:
                # Same state, accumulate time
                if current_state == 'standby_idle':
                    current_standby_duration += time_diff_minutes
                elif current_state == 'active':
                    current_active_duration += time_diff_minutes
                elif current_state == 'deep_idle':
                    current_off_duration += time_diff_minutes
            else:
                # State changed, reset counters
                current_standby_duration = 0.0
                current_active_duration = 0.0
                current_off_duration = 0.0
        
        # Assign current duration to the record
        df.loc[i, 'standby_duration'] = current_standby_duration
        df.loc[i, 'active_duration'] = current_active_duration
        df.loc[i, 'off_duration'] = current_off_duration
    
    # Method 2: Calculate time since last active state
    last_active_time = None
    
    for i in range(len(df)):
        current_time = df.loc[i, 'timestamp']
        current_state = df.loc[i, 'idle_state']
        
        if current_state == 'active':
            # Update last active time
            last_active_time = current_time
            df.loc[i, 'time_since_last_active'] = 0.0
        else:
            # Calculate time since last active
            if last_active_time is not None:
                time_diff = (current_time - last_active_time).total_seconds() / 60.0
                df.loc[i, 'time_since_last_active'] = time_diff
            else:
                # No previous active state found
                df.loc[i, 'time_since_last_active'] = 999.0  # Large number
    
    # Validate and print statistics
    print("\nDuration features calculated successfully:")
    
    # Statistics for standby duration
    standby_data = df[df['idle_state'] == 'standby_idle']
    if len(standby_data) > 0:
        print(f"   Standby Duration Statistics:")
        print(f"     - Records in standby: {len(standby_data)}")
        print(f"     - Average duration: {standby_data['standby_duration'].mean():.1f} minutes")
        print(f"     - Max duration: {standby_data['standby_duration'].max():.1f} minutes")
        print(f"     - Records with >30min standby: {(standby_data['standby_duration'] > 30).sum()}")
        print(f"     - Records with >60min standby: {(standby_data['standby_duration'] > 60).sum()}")
    
    # Statistics for time since last active
    non_active_data = df[df['idle_state'] != 'active']
    if len(non_active_data) > 0:
        print(f"   Time Since Last Active Statistics:")
        print(f"     - Average: {non_active_data['time_since_last_active'].mean():.1f} minutes")
        print(f"     - Max: {non_active_data['time_since_last_active'].max():.1f} minutes")
        print(f"     - Records with >2 hours since active: {(non_active_data['time_since_last_active'] > 120).sum()}")
    
    return df

def calculate_usage_pattern_features(df):

    print("\n=== Calculating Usage Pattern Features ===")
    
    # Calculate hourly usage probability
    hourly_active_count = df[df['idle_state'] == 'active'].groupby('hour').size()
    hourly_total_count = df.groupby('hour').size()
    hourly_usage_prob = (hourly_active_count / hourly_total_count).fillna(0)
    
    # Map hourly usage probability to each record
    df['hourly_usage_probability'] = df['hour'].map(hourly_usage_prob)
    
    # Calculate daily usage frequency (rolling 24-hour window)
    df['daily_usage_frequency'] = 0
    df['recent_usage_count'] = 0
    
    for i in range(len(df)):
        current_time = df.loc[i, 'timestamp']
        
        # Look back 24 hours
        lookback_time = current_time - timedelta(hours=24)
        recent_data = df[(df['timestamp'] >= lookback_time) & (df['timestamp'] < current_time)]
        
        # Count active periods in last 24 hours
        if len(recent_data) > 0:
            active_transitions = recent_data['idle_state'].eq('active') & recent_data['idle_state'].shift(1).ne('active')
            df.loc[i, 'daily_usage_frequency'] = active_transitions.sum()
            df.loc[i, 'recent_usage_count'] = (recent_data['idle_state'] == 'active').sum()
    
    # Calculate weekend usage ratio
    weekend_usage = df[(df['weekend'] == True) & (df['idle_state'] == 'active')].groupby('hour').size()
    weekday_usage = df[(df['weekend'] == False) & (df['idle_state'] == 'active')].groupby('hour').size()
    
    weekend_ratio = {}
    for hour in range(24):
        weekend_count = weekend_usage.get(hour, 0)
        weekday_count = weekday_usage.get(hour, 0)
        total_count = weekend_count + weekday_count
        weekend_ratio[hour] = weekend_count / total_count if total_count > 0 else 0
    
    df['weekend_usage_ratio'] = df['hour'].map(weekend_ratio)
    
    print("Usage pattern features calculated:")
    print(f"   - hourly_usage_probability: Range {df['hourly_usage_probability'].min():.3f} - {df['hourly_usage_probability'].max():.3f}")
    print(f"   - daily_usage_frequency: Average {df['daily_usage_frequency'].mean():.1f} times/day")
    
    return df

def analyze_improved_data_distribution(df):

    print("=== Analyzing Data Distribution for Threshold Setting ===")
    
    thresholds = {}
    
    # 1. Analyze standby duration distribution (data-driven)
    standby_data = df[df['idle_state'] == 'standby_idle']['standby_duration']
    
    if len(standby_data) > 0:
        duration_stats = standby_data.describe()
        print(f"\nStandby Duration Distribution:")
        print(f"   Count: {duration_stats['count']:.0f}")
        print(f"   Mean: {duration_stats['mean']:.1f} minutes")
        print(f"   25th percentile: {duration_stats['25%']:.1f} minutes")
        print(f"   50th percentile (median): {duration_stats['50%']:.1f} minutes")
        print(f"   75th percentile: {duration_stats['75%']:.1f} minutes")
        print(f"   90th percentile: {standby_data.quantile(0.9):.1f} minutes")
        
        # Set thresholds based on percentiles
        thresholds['standby_short'] = standby_data.quantile(0.33)
        thresholds['standby_medium'] = standby_data.quantile(0.67)
        thresholds['standby_long'] = standby_data.quantile(0.85)
    
    # 2. Analyze time since last active distribution (data-driven)
    non_active_data = df[df['idle_state'] != 'active']['time_since_last_active']
    
    if len(non_active_data) > 50:  # Sufficient data
        time_since_stats = non_active_data.describe()
        print(f"\nTime Since Last Active Distribution:")
        print(f"   Mean: {time_since_stats['mean']:.1f} minutes")
        print(f"   25th percentile: {time_since_stats['25%']:.1f} minutes")
        print(f"   50th percentile: {time_since_stats['50%']:.1f} minutes")
        print(f"   75th percentile: {time_since_stats['75%']:.1f} minutes")
        print(f"   90th percentile: {non_active_data.quantile(0.9):.1f} minutes")
        
        # Set data-driven thresholds with reasonable bounds
        data_33 = non_active_data.quantile(0.33)
        data_67 = non_active_data.quantile(0.67)
        data_85 = non_active_data.quantile(0.85)
        
        # Ensure thresholds are within reasonable ranges
        thresholds['time_since_short'] = max(30, min(data_33, 90))   # 30-90 minutes
        thresholds['time_since_medium'] = max(60, min(data_67, 180)) # 60-180 minutes  
        thresholds['time_since_long'] = max(120, min(data_85, 300))  # 120-300 minutes
        
        print(f"Data-driven time_since thresholds:")
        print(f"   Short: {thresholds['time_since_short']:.0f} minutes")
        print(f"   Medium: {thresholds['time_since_medium']:.0f} minutes")
        print(f"   Long: {thresholds['time_since_long']:.0f} minutes")
    else:
        # Insufficient data, use behavioral science defaults
        thresholds['time_since_short'] = 60   # 1 hour
        thresholds['time_since_medium'] = 120  # 2 hours
        thresholds['time_since_long'] = 180   # 3 hours
        print(f"Insufficient data for time_since, using defaults: 60, 120, 180 minutes")
    
    # 3. Analyze hourly usage probability distribution (data-driven)
    if 'hourly_usage_probability' in df.columns:
        usage_prob_stats = df['hourly_usage_probability'].describe()
        print(f"\nHourly Usage Probability Distribution:")
        print(f"   Mean: {usage_prob_stats['mean']:.3f}")
        print(f"   25th percentile: {usage_prob_stats['25%']:.3f}")
        print(f"   50th percentile: {usage_prob_stats['50%']:.3f}")
        print(f"   75th percentile: {usage_prob_stats['75%']:.3f}")
        
        # Set thresholds based on distribution
        thresholds['usage_very_low'] = df['hourly_usage_probability'].quantile(0.15)
        thresholds['usage_low'] = df['hourly_usage_probability'].quantile(0.35)
        thresholds['usage_medium'] = df['hourly_usage_probability'].quantile(0.65)
    
    print(f"\n=== Data-Driven Thresholds Summary ===")
    for key, value in thresholds.items():
        print(f"   {key}: {value:.1f}")
    
    return thresholds

def calculate_urgency_score(df, thresholds=None):

    print("\n=== Calculating Improved Control Urgency Score ===")
    
    # If no thresholds provided, calculate from data
    if thresholds is None:
        thresholds = analyze_improved_data_distribution(df)
    
    # Define factor weights based on energy savings priority and user experience
    weights = {
        'standby_duration': 0.45,      # Highest weight - direct energy waste
        'time_since_active': 0.35,     # High weight - indicates user disengagement  
        'usage_probability': 0.20      # Lower weight - context modifier
    }
    
    print(f"Factor weights: {weights}")
    
    df['control_urgency_score'] = 0.0
    urgency_breakdown = {
        'standby_factor': [],
        'time_since_factor': [],
        'usage_prob_factor': []
    }
    
    for i in range(len(df)):
        if df.loc[i, 'idle_state'] == 'standby_idle':
            total_score = 0.0
            
            # Factor 1: Standby Duration (45% weight)
            standby_duration = df.loc[i, 'standby_duration']
            standby_factor = 0.0
            
            if standby_duration >= thresholds.get('standby_long', 60):
                standby_factor = 1.0    # Very long standby - urgent
            elif standby_duration >= thresholds.get('standby_medium', 35):
                standby_factor = 0.7    # Medium standby - moderate urgency
            elif standby_duration >= thresholds.get('standby_short', 15):
                standby_factor = 0.4    # Short standby - low urgency
            else:
                standby_factor = 0.1    # Very short standby - minimal urgency
            
            urgency_breakdown['standby_factor'].append(standby_factor)
            total_score += standby_factor * weights['standby_duration']
            
            # Factor 2: Time Since Last Active (35% weight)
            time_since_active = df.loc[i, 'time_since_last_active']
            time_since_factor = 0.0
            
            if time_since_active >= thresholds.get('time_since_long', 180):
                time_since_factor = 1.0    # >3 hours - user clearly disengaged
            elif time_since_active >= thresholds.get('time_since_medium', 120):
                time_since_factor = 0.7    # >2 hours - likely disengaged
            elif time_since_active >= thresholds.get('time_since_short', 60):
                time_since_factor = 0.4    # >1 hour - possibly disengaged
            else:
                time_since_factor = 0.1    # <1 hour - recently active
            
            urgency_breakdown['time_since_factor'].append(time_since_factor)
            total_score += time_since_factor * weights['time_since_active']
            
            # Factor 3: Usage Probability (20% weight)
            usage_prob = df.loc[i, 'hourly_usage_probability'] if 'hourly_usage_probability' in df.columns else 0.5
            usage_prob_factor = 0.0
            
            if usage_prob <= thresholds.get('usage_very_low', 0.1):
                usage_prob_factor = 1.0    # Very low usage time - safe to control
            elif usage_prob <= thresholds.get('usage_low', 0.2):
                usage_prob_factor = 0.7    # Low usage time - probably safe
            elif usage_prob <= thresholds.get('usage_medium', 0.4):
                usage_prob_factor = 0.4    # Medium usage time - moderate risk
            else:
                usage_prob_factor = 0.1    # High usage time - risky to control
            
            urgency_breakdown['usage_prob_factor'].append(usage_prob_factor)
            total_score += usage_prob_factor * weights['usage_probability']
            
            # Apply safety constraints
            # Reduce urgency during peak hours
            if 'is_peak_hours' in df.columns and df.loc[i, 'is_peak_hours']:
                total_score *= 0.8  # 20% reduction during peak hours
            
            # Reduce urgency if recent usage pattern is unstable
            if 'usage_pattern_stability' in df.columns:
                stability = df.loc[i, 'usage_pattern_stability']
                if stability < 0.3:  # Low stability
                    total_score *= 0.9  # 10% reduction for uncertainty
            
            df.loc[i, 'control_urgency_score'] = min(total_score, 1.0)
        else:
            # Non-standby states have zero urgency
            urgency_breakdown['standby_factor'].append(0)
            urgency_breakdown['time_since_factor'].append(0)
            urgency_breakdown['usage_prob_factor'].append(0)
    
    # Calculate and print statistics
    standby_urgency = df[df['idle_state'] == 'standby_idle']['control_urgency_score']
    
    if len(standby_urgency) > 0:
        print(f"\nUrgency Score Statistics (standby records only):")
        print(f"   Mean: {standby_urgency.mean():.3f}")
        print(f"   Std: {standby_urgency.std():.3f}")
        print(f"   Min: {standby_urgency.min():.3f}")
        print(f"   Max: {standby_urgency.max():.3f}")
        
        # Urgency level distribution
        high_urgency = (standby_urgency >= 0.7).sum()
        medium_urgency = ((standby_urgency >= 0.4) & (standby_urgency < 0.7)).sum()
        low_urgency = (standby_urgency < 0.4).sum()
        
        print(f"\nUrgency Level Distribution:")
        print(f"   High urgency (>=0.7): {high_urgency} records ({high_urgency/len(standby_urgency)*100:.1f}%)")
        print(f"   Medium urgency (0.4-0.7): {medium_urgency} records ({medium_urgency/len(standby_urgency)*100:.1f}%)")
        print(f"   Low urgency (<0.4): {low_urgency} records ({low_urgency/len(standby_urgency)*100:.1f}%)")
        
        # Factor contribution analysis
        print(f"\nFactor Contribution Analysis:")
        if urgency_breakdown['standby_factor']:
            avg_standby_contribution = np.mean(urgency_breakdown['standby_factor']) * weights['standby_duration']
            avg_time_contribution = np.mean(urgency_breakdown['time_since_factor']) * weights['time_since_active']
            avg_usage_contribution = np.mean(urgency_breakdown['usage_prob_factor']) * weights['usage_probability']
            
            print(f"   Average standby duration contribution: {avg_standby_contribution:.3f}")
            print(f"   Average time since active contribution: {avg_time_contribution:.3f}")
            print(f"   Average usage probability contribution: {avg_usage_contribution:.3f}")
    
    print(f"Improved urgency score calculated with data-driven thresholds")
    
    return df

def validate_urgency_score(df):

    print("\n=== Validating Urgency Score Logic ===")
    
    # Test case 1: Very long standby during low usage time should have high urgency
    test_case_1 = df[
        (df['standby_duration'] > 60) & 
        (df['idle_state'] == 'standby_idle') &
        (df['hourly_usage_probability'] < 0.1)
    ]
    
    if len(test_case_1) > 0:
        avg_urgency_1 = test_case_1['control_urgency_score'].mean()
        print(f"Test 1 - Long standby + Low usage: Average urgency = {avg_urgency_1:.3f} (should be high)")
    
    # Test case 2: Short standby during peak hours should have low urgency
    test_case_2 = df[
        (df['standby_duration'] < 20) & 
        (df['idle_state'] == 'standby_idle') &
        (df.get('is_peak_hours', False) == True)
    ]
    
    if len(test_case_2) > 0:
        avg_urgency_2 = test_case_2['control_urgency_score'].mean()
        print(f"Test 2 - Short standby + Peak hours: Average urgency = {avg_urgency_2:.3f} (should be low)")
    
    # Test case 3: Medium standby with medium usage should have medium urgency
    test_case_3 = df[
        (df['standby_duration'] >= 20) & (df['standby_duration'] <= 60) &
        (df['idle_state'] == 'standby_idle') &
        (df['hourly_usage_probability'] >= 0.2) & (df['hourly_usage_probability'] <= 0.5)
    ]
    
    if len(test_case_3) > 0:
        avg_urgency_3 = test_case_3['control_urgency_score'].mean()
        print(f"Test 3 - Medium standby + Medium usage: Average urgency = {avg_urgency_3:.3f} (should be medium)")
    
    # Overall sanity check
    all_urgency = df[df['idle_state'] == 'standby_idle']['control_urgency_score']
    if len(all_urgency) > 0:
        print(f"\nOverall Sanity Check:")
        print(f"   Urgency scores range: {all_urgency.min():.3f} - {all_urgency.max():.3f}")
        print(f"   Standard deviation: {all_urgency.std():.3f}")
        
        if all_urgency.std() > 0.1:
            print("   Good variation in urgency scores")
        else:
            print("   Low variation - consider adjusting thresholds")

def calculate_comprehensive_context_features(df):

    print("\n=== Calculating Comprehensive Context Features ===")
    
    def get_time_period(hour):
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    
    df['time_period'] = df['hour'].apply(get_time_period)
    
    if 'hourly_usage_probability' not in df.columns:
        print("   Calculating hourly usage probability...")
        hourly_active_count = df[df['idle_state'] == 'active'].groupby('hour').size()
        hourly_total_count = df.groupby('hour').size()
        hourly_usage_prob = (hourly_active_count / hourly_total_count).fillna(0)
        df['hourly_usage_probability'] = df['hour'].map(hourly_usage_prob)
    
    peak_hours = df.groupby('hour')['hourly_usage_probability'].first().nlargest(6).index
    df['is_peak_hours'] = df['hour'].isin(peak_hours)
    
    sleep_hours = [0, 1, 2, 3, 4, 5]
    df['is_sleep_hours'] = df['hour'].isin(sleep_hours)
    
    print("   Calculating usage pattern stability...")
    df['usage_pattern_stability'] = 0.0
    
    for i in range(288, len(df)):  # Need at least 24 hours of history
        # Calculate consistency of usage in same hour over past days
        current_hour = df.loc[i, 'hour']
        lookback_hours = min(288, i)  # Look back up to 24 hours
        past_data = df.loc[max(0, i-lookback_hours):i]
        same_hour_data = past_data[past_data['hour'] == current_hour]
        
        if len(same_hour_data) > 2:  # Need at least 3 data points
            active_ratio = (same_hour_data['idle_state'] == 'active').mean()
            # Stability is higher when usage pattern is consistent
            df.loc[i, 'usage_pattern_stability'] = abs(active_ratio - 0.5) * 2
        else:
            df.loc[i, 'usage_pattern_stability'] = 0.5  # Default moderate stability
    
    print("   Calculating consecutive idle episodes...")
    df['consecutive_idle_episodes'] = 0
    
    for i in range(len(df)):
        current_date = df.loc[i, 'timestamp'].date()
        current_time = df.loc[i, 'timestamp']
        
        # Get all data for current date up to current time
        today_mask = (df['timestamp'].dt.date == current_date) & (df['timestamp'] <= current_time)
        today_data = df[today_mask]
        
        # Count idle state transitions (transitions INTO idle states)
        if len(today_data) > 1:
            # Create a boolean mask for idle states
            is_idle = today_data['idle_state'].isin(['standby_idle', 'deep_idle'])
            was_not_idle = ~today_data['idle_state'].shift(1).isin(['standby_idle', 'deep_idle'])
            
            # Count transitions from non-idle to idle
            idle_transitions = (is_idle & was_not_idle).sum()
            df.loc[i, 'consecutive_idle_episodes'] = idle_transitions
    
    print("   Calculating improved control urgency score...")
    df = calculate_urgency_score(df)
    
    print("   Calculating additional context features...")
    
    # Time-based features
    df['is_weekend_evening'] = (df['weekend'] == True) & (df['time_period'] == 'evening')
    df['is_weekday_morning'] = (df['weekend'] == False) & (df['time_period'] == 'morning')
    df['is_late_night'] = df['hour'].isin([23, 0, 1, 2])
    
    # Usage intensity features
    df['usage_intensity'] = 'medium'  # Default value
    for i in range(len(df)):
        usage_prob = df.loc[i, 'hourly_usage_probability']
        
        if usage_prob >= 0.7:
            df.loc[i, 'usage_intensity'] = 'high'
        elif usage_prob >= 0.4:
            df.loc[i, 'usage_intensity'] = 'medium'
        elif usage_prob >= 0.15:
            df.loc[i, 'usage_intensity'] = 'low'
        else:
            df.loc[i, 'usage_intensity'] = 'very_low'
    
    # Control opportunity windows
    df['control_opportunity_window'] = False
    
    # Define control opportunity windows (high urgency + safe conditions)
    control_mask = (
        (df['control_urgency_score'] >= 0.6) &  # High urgency
        (df['idle_state'] == 'standby_idle') &  # Currently in standby
        (~df['is_peak_hours']) &  # Not during peak hours
        (df['usage_pattern_stability'] >= 0.3)  # Reasonably stable pattern
    )
    df.loc[control_mask, 'control_opportunity_window'] = True
    
    # Calculate control confidence score
    df['control_confidence'] = 0.0
    
    for i in range(len(df)):
        if df.loc[i, 'idle_state'] == 'standby_idle':
            confidence = 0.0
            
            # Base confidence from urgency score
            urgency = df.loc[i, 'control_urgency_score']
            confidence += urgency * 0.5  
            
            # Stability factor
            stability = df.loc[i, 'usage_pattern_stability']
            confidence += stability * 0.2  
            
            # Time period factor
            if df.loc[i, 'is_sleep_hours']:
                confidence += 0.2  
            elif df.loc[i, 'is_peak_hours']:
                confidence -= 0.1  
            
            # Duration factor
            standby_duration = df.loc[i, 'standby_duration']
            if standby_duration > 90:  # Very long standby
                confidence += 0.1
            elif standby_duration < 10:  # Very short standby
                confidence -= 0.1
            
            df.loc[i, 'control_confidence'] = max(0.0, min(1.0, confidence))
    
    print("\nContext features calculated successfully:")
    
    # Time period distribution
    time_period_dist = df['time_period'].value_counts()
    print(f"   Time period distribution: {time_period_dist.to_dict()}")
    
    # Peak hours
    print(f"   Peak usage hours: {sorted(peak_hours)}")
    
    # Usage intensity distribution
    if 'usage_intensity' in df.columns:
        intensity_dist = df['usage_intensity'].value_counts()
        print(f"   Usage intensity distribution: {intensity_dist.to_dict()}")
    
    # Control opportunities
    control_opportunities = df['control_opportunity_window'].sum()
    standby_records = (df['idle_state'] == 'standby_idle').sum()
    if standby_records > 0:
        opportunity_percentage = control_opportunities / standby_records * 100
        print(f"   Control opportunities: {control_opportunities} out of {standby_records} standby records ({opportunity_percentage:.1f}%)")
    
    # Confidence statistics
    standby_confidence = df[df['idle_state'] == 'standby_idle']['control_confidence']
    if len(standby_confidence) > 0:
        print(f"   Average control confidence: {standby_confidence.mean():.3f}")
        print(f"   High confidence records (>0.7): {(standby_confidence > 0.7).sum()}")
    
    # Quality validation
    validate_urgency_score(df)
    
    return df

def calculate_transition_features(df):

    print("\n=== Calculating Transition Features ===")
    
    # Calculate state change indicators
    df['state_changed'] = (df['idle_state'] != df['idle_state'].shift(1)).astype(int)
    df['prev_state'] = df['idle_state'].shift(1)
    
    # Identify critical transitions
    df['transition_to_standby'] = ((df['prev_state'] == 'active') & 
                                  (df['idle_state'] == 'standby_idle')).astype(int)
    df['transition_to_active'] = ((df['prev_state'] == 'standby_idle') & 
                                 (df['idle_state'] == 'active')).astype(int)
    
    # Calculate typical standby duration before returning to active
    typical_standby_durations = []
    
    # Find all standby periods that ended with return to active
    standby_starts = df[df['transition_to_standby'] == 1].index
    
    for start_idx in standby_starts:
        # Find when this standby period ended
        subsequent_data = df.loc[start_idx:]
        active_return = subsequent_data[subsequent_data['transition_to_active'] == 1]
        
        if len(active_return) > 0:
            end_idx = active_return.index[0]
            duration = (df.loc[end_idx, 'timestamp'] - df.loc[start_idx, 'timestamp']).total_seconds() / 60
            typical_standby_durations.append(duration)
    
    # Calculate average return time for context
    avg_return_time = np.mean(typical_standby_durations) if typical_standby_durations else 60
    df['typical_return_time'] = avg_return_time
    
    print("Transition features calculated:")
    print(f"   - Total state changes: {df['state_changed'].sum()}")
    print(f"   - Transitions to standby: {df['transition_to_standby'].sum()}")
    print(f"   - Typical return time: {avg_return_time:.1f} minutes")
    
    return df

def analyze_feature_importance(df):

    print("\n=== Analyzing Feature Importance ===")
    
    # Select features for analysis
    feature_columns = [
        'standby_duration', 'time_since_last_active', 'hourly_usage_probability',
        'daily_usage_frequency', 'weekend_usage_ratio', 'is_peak_hours',
        'consecutive_idle_episodes', 'control_urgency_score', 'usage_pattern_stability'
    ]
    
    # Calculate correlation with control need
    feature_correlations = {}
    for feature in feature_columns:
        if feature in df.columns:
            correlation = df[feature].corr(df['needs_control'].astype(int))
            feature_correlations[feature] = correlation
    
    print("Feature correlations with control need:")
    for feature, corr in sorted(feature_correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"   {feature}: {corr:.3f}")
    
    return feature_correlations

def visualize_features(df):

    print("\n=== Generating Feature Visualizations ===")
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    
    # Plot 1: Standby duration distribution
    standby_data = df[df['idle_state'] == 'standby_idle']
    if len(standby_data) > 0:
        axes[0, 0].hist(standby_data['standby_duration'], bins=30, alpha=0.7, color='orange')
        axes[0, 0].set_title('Standby Duration Distribution')
        axes[0, 0].set_xlabel('Duration (minutes)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Hourly usage probability
    hourly_prob = df.groupby('hour')['hourly_usage_probability'].first()
    axes[0, 1].plot(hourly_prob.index, hourly_prob.values, marker='o', linewidth=2)
    axes[0, 1].set_title('Hourly Usage Probability')
    axes[0, 1].set_xlabel('Hour of Day')
    axes[0, 1].set_ylabel('Usage Probability')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(range(0, 24, 2))
    
    # Plot 3: Control urgency score by time period
    urgency_by_period = df.groupby('time_period')['control_urgency_score'].mean()
    axes[1, 0].bar(urgency_by_period.index, urgency_by_period.values, color='red', alpha=0.7)
    axes[1, 0].set_title('Average Control Urgency by Time Period')
    axes[1, 0].set_ylabel('Control Urgency Score')
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 4: Weekend vs Weekday usage patterns
    weekend_pattern = df[df['weekend'] == True].groupby('hour')['hourly_usage_probability'].first()
    weekday_pattern = df[df['weekend'] == False].groupby('hour')['hourly_usage_probability'].first()
    
    axes[1, 1].plot(weekend_pattern.index, weekend_pattern.values, label='Weekend', marker='o')
    axes[1, 1].plot(weekday_pattern.index, weekday_pattern.values, label='Weekday', marker='s')
    axes[1, 1].set_title('Weekend vs Weekday Usage Patterns')
    axes[1, 1].set_xlabel('Hour of Day')
    axes[1, 1].set_ylabel('Usage Probability')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # # Plot 5: Feature correlation heatmap
    # feature_cols = ['standby_duration', 'time_since_last_active', 'hourly_usage_probability',
    #                'daily_usage_frequency', 'control_urgency_score', 'needs_control']
    # available_cols = [col for col in feature_cols if col in df.columns]
    
    # if len(available_cols) > 1:
    #     corr_matrix = df[available_cols].corr()
    #     sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
    #                ax=axes[2, 0], fmt='.3f')
    #     axes[2, 0].set_title('Feature Correlation Matrix')
    
    # # Plot 6: Control opportunities over time
    # hourly_control_need = df.groupby('hour')['needs_control'].mean()
    # axes[2, 1].bar(hourly_control_need.index, hourly_control_need.values, 
    #                color='green', alpha=0.7)
    # axes[2, 1].set_title('Control Opportunities by Hour')
    # axes[2, 1].set_xlabel('Hour of Day')
    # axes[2, 1].set_ylabel('Proportion Needing Control')
    # axes[2, 1].set_xticks(range(0, 24, 2))
    
    plt.tight_layout(pad=2.5)
    plt.show()
    
    print("Feature visualizations generated")

def save_engineered_features(df, output_file='data_with_engineered_features.csv'):

    print(f"\n=== Saving Engineered Features ===")
    
    # Select columns for output
    output_columns = [
        # Original columns
        'timestamp', 'date', 'hour', 'day_of_week', 'weekend',
        'power', 'power_state', 'idle_state', 'control_target', 'needs_control',
        
        # Duration features
        'standby_duration', 'active_duration', 'time_since_last_active',
        
        # Usage pattern features
        'hourly_usage_probability', 'daily_usage_frequency', 'weekend_usage_ratio',
        
        # Context features
        'time_period', 'is_peak_hours', 'is_sleep_hours', 'consecutive_idle_episodes',
        'control_urgency_score', 'usage_intensity', 'control_opportunity_window', 'control_confidence',
        
        # Transition features
        'state_changed', 'transition_to_standby', 'usage_pattern_stability'
    ]
    
    # Only include columns that exist in the dataframe
    available_columns = [col for col in output_columns if col in df.columns]
    
    df_output = df[available_columns].copy()
    df_output.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"Data saved to: {output_file}")
    print(f"Records: {len(df_output)}, Features: {len(available_columns)}")
    
    return df_output

def generate_feature_report(df):
    
    # Basic statistics
    total_records = len(df)
    control_needed = df['needs_control'].sum()
    control_percentage = control_needed / total_records * 100
    
    print(f"\nDataset Overview:")
    print(f"   Total records: {total_records:,}")
    print(f"   Records needing control: {control_needed:,} ({control_percentage:.1f}%)")
    
    # Feature quality assessment
    standby_data = df[df['idle_state'] == 'standby_idle']
    if len(standby_data) > 0:
        print(f"\nStandby Analysis:")
        print(f"   Average standby duration: {standby_data['standby_duration'].mean():.1f} minutes")
        print(f"   Max standby duration: {standby_data['standby_duration'].max():.1f} minutes")
        print(f"   High urgency periods (score > 0.5): {(standby_data['control_urgency_score'] > 0.5).sum()}")
    
    # Usage pattern insights
    peak_hours = df[df['is_peak_hours'] == True]['hour'].unique()
    print(f"\nUsage Patterns:")
    print(f"   Peak usage hours: {sorted(peak_hours)}")
    print(f"   Weekend usage difference: {df['weekend_usage_ratio'].mean():.3f}")
    
    # Control opportunities
    high_urgency = (df['control_urgency_score'] > 0.5).sum()
    print(f"\nControl Opportunities:")
    print(f"   High urgency control points: {high_urgency:,}")
    print(f"   Average daily usage frequency: {df['daily_usage_frequency'].mean():.1f}")
    
    print(f"\nReady for Task 3: Fuzzy Logic Rule Development")
    print("="*60)

def main():

    print("Starting Task 2: Temporal Feature Engineering")
    print("="*60)
    
    # Step 1: Load processed data from Task 1
    df = load_control_features_data()
    if df is None:
        return None
    
    # Step 2: Calculate improved duration features
    df = calculate_improved_duration_features(df)
    
    # Step 3: Calculate usage pattern features
    df = calculate_usage_pattern_features(df)
    
    # Step 4: Calculate comprehensive context features (includes improved urgency score)
    df = calculate_comprehensive_context_features(df)
    
    # Step 5: Calculate transition features
    df = calculate_transition_features(df)
    
    # Step 6: Analyze feature importance
    feature_correlations = analyze_feature_importance(df)
    
    # Step 7: Generate visualizations
    visualize_features(df)
    
    # Step 8: Save engineered features
    df_output = save_engineered_features(df)
    
    # Step 9: Generate comprehensive report
    generate_feature_report(df)
    
    return df_output

if __name__ == "__main__":
    # Execute Task 2
    result_df = main()