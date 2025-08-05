import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy import stats
import warnings
import os
import requests
from collections import deque

warnings.filterwarnings('ignore')

# Import your existing modules
try:
    import fuzzy_logic_control
    from fuzzy_logic_control import AntiOscillationFilter
    HAS_FUZZY_SYSTEM = True
    print("✅ Successfully imported existing fuzzy_logic_control module")
except ImportError:
    HAS_FUZZY_SYSTEM = False
    print("❌ Unable to import fuzzy_logic_control module")

try:
    from fuzzy_logic_control import DecisionTreeSmartPowerAnalysis
    HAS_DECISION_SYSTEM = True
    print("✅ Successfully imported DecisionTreeSmartPowerAnalysis")
except ImportError:
    HAS_DECISION_SYSTEM = False
    print("❌ Unable to import DecisionTreeSmartPowerAnalysis")

def get_octopus_latest_unit_rate():

    product_code = "VAR-22-11-01"
    tariff_code = "E-1R-VAR-22-11-01-K"
    
    url = f"https://api.octopus.energy/v1/products/{product_code}/electricity-tariffs/{tariff_code}/standard-unit-rates/"

    try:
        response = requests.get(url)
        data = response.json()

        for result in data['results']:
            if result['valid_from'] <= datetime.now().isoformat():
                unit_rate_pence = result['value_inc_vat']  # 單位: pence
                unit_rate_gbp = unit_rate_pence / 100       # 換成英鎊
                return round(unit_rate_gbp, 4)

    except Exception as e:
        print(f"⚠️ 無法抓取 Octopus 電價: {e}")
        return 0.30  # fallback 預設值

# 使用範例
uk_electricity_rate = get_octopus_latest_unit_rate()
print(f"✅ 使用最新英國電價：£{uk_electricity_rate}/kWh")

class CompletePowerAnalyzer:
    """
    Complete power analysis for ALL energy consumption levels
    (not just phantom load)
    """
    
    def __init__(self, data_file_path):
        self.data_file = data_file_path
        # self.uk_electricity_rate = 0.30
        
        # Initialize your existing system
        if HAS_DECISION_SYSTEM:
            print("🔄 Initializing your existing decision system...")
            self.decision_system = DecisionTreeSmartPowerAnalysis()
            print("✅ Decision system initialization completed")
        else:
            self.decision_system = None
            print("❌ Cannot initialize decision system")
        
        # Result storage
        self.analysis_results = []
        self.original_powers = []
        self.controlled_powers = []
        
        print("✅ CompletePowerAnalyzer initialization completed")

    
    
    def _generate_all_power_opportunities(self, df):
        """Generate opportunities for ALL power levels (not just phantom load)"""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        print(f'Processing ALL power data: {len(df)} data points')
        print(f'Power range: {df["power"].min():.1f}W - {df["power"].max():.1f}W')
        
        opportunities = []
        
        # Create opportunities from consecutive time periods
        window_size = 4  # Use 4 consecutive records (1 hour) as one opportunity
        
        for i in range(0, len(df) - window_size + 1, window_size):
            window_data = df.iloc[i:i+window_size]
            
            if len(window_data) == window_size:
                avg_power = window_data['power'].mean()
                start_time = window_data.iloc[0]['timestamp']
                end_time = window_data.iloc[-1]['timestamp']
                
                opportunities.append({
                    'device_id': 'smart_device',
                    'start_time': start_time,
                    'end_time': end_time,
                    'power_watt': avg_power,
                    'records': window_data.to_dict('records')
                })
        
        return opportunities
    
    def _make_decision_for_all_power(self, opportunity):
        """Make decisions for ALL power levels using your existing system"""
        if not self.decision_system:
            return self._fallback_decision_all_power(opportunity)
        
        try:
            # Use your existing system's decision logic
            features = self.decision_system._extract_enhanced_features(opportunity, None)
            timestamp = opportunity['start_time']
            
            # Get three scores
            if self.decision_system.device_activity_model:
                try:
                    activity_result = self.decision_system.device_activity_model.calculate_activity_score(timestamp)
                    activity_score = activity_result['activity_score']
                except:
                    activity_score = self.decision_system._fallback_activity_score(features, timestamp)
            else:
                activity_score = self.decision_system._fallback_activity_score(features, timestamp)
            
            if self.decision_system.user_habit_model:
                try:
                    habit_result = self.decision_system.user_habit_model.calculate_habit_score(timestamp)
                    habit_score = habit_result['habit_score']
                except:
                    habit_score = self.decision_system._fallback_habit_score(features, timestamp)
            else:
                habit_score = self.decision_system._fallback_habit_score(features, timestamp)
            
            if self.decision_system.confidence_model:
                try:
                    confidence_result = self.decision_system.confidence_model.calculate_confidence_score(timestamp)
                    confidence_score = confidence_result['confidence_score']
                except:
                    confidence_score = self.decision_system._fallback_confidence_score(features, timestamp)
            else:
                confidence_score = self.decision_system._fallback_confidence_score(features, timestamp)
            
            # Use your existing decision logic
            decision, debug_info = self.decision_system._make_intelligent_decision(
                activity_score, habit_score, confidence_score, features
            )
            
            # Apply anti-oscillation filter
            filter_result = self.decision_system.anti_oscillation_filter.filter_decision(
                original_decision=decision,
                power_value=features['power_watt'],
                timestamp=timestamp,
                scores={
                    'activity': activity_score,
                    'habit': habit_score,
                    'confidence': confidence_score
                }
            )
            
            final_decision = filter_result['filtered_decision']
            
            return {
                'decision': final_decision,
                'original_decision': decision,
                'activity_score': activity_score,
                'habit_score': habit_score,
                'confidence_score': confidence_score,
                'filter_applied': filter_result['should_use_filtered'],
                'filter_reason': filter_result['filter_reason'],
                'debug_info': debug_info
            }
            
        except Exception as e:
            print(f"⚠️ Error in decision process: {e}")
            return self._fallback_decision_all_power(opportunity)
    
    def _fallback_decision_all_power(self, opportunity):
        """Fallback decision method for all power levels"""
        power = opportunity.get('power_watt', 50)
        hour = opportunity['start_time'].hour
        
        # Decisions based on ALL power levels
        if power < 20:  # Very low power
            decision = 'suggest_shutdown'
        elif power < 60 and (hour < 7 or hour > 23):  # Low power during sleep hours
            decision = 'suggest_shutdown'
        elif power < 60:  # Low power during active hours
            decision = 'send_notification'
        elif power < 100:  # Medium power
            decision = 'delay_decision'
        else:  # High power
            decision = 'keep_on'
        
        return {
            'decision': decision,
            'original_decision': decision,
            'activity_score': 0.5,
            'habit_score': 0.5,
            'confidence_score': 0.5,
            'filter_applied': False,
            'filter_reason': 'fallback method',
            'debug_info': {}
        }
    
    def _calculate_controlled_power_all_levels(self, original_power, decision_result, opportunity):
        """Calculate controlled power for ALL power levels"""
        decision = decision_result['decision']
        
        # Power control for all levels
        if decision == 'suggest_shutdown':
            # Direct shutdown
            controlled_power = 0
            power_reduction = original_power
            
        elif decision == 'send_notification':

            controlled_power = 0
            # if original_power < 60:
            #     controlled_power = 0  # Shutdown low power devices
            # else:
            #     controlled_power = original_power * 0.3  # Reduce medium power devices
            power_reduction = original_power - controlled_power
            
        elif decision == 'delay_decision':

            controlled_power = original_power
            power_reduction = 0

            # # Moderate reduction based on scores
            # activity_score = decision_result.get('activity_score', 0.5)
            # habit_score = decision_result.get('habit_score', 0.5)
            # confidence_score = decision_result.get('confidence_score', 0.5)
            
            # # Calculate control factor
            # control_factor = 0.3 * (1 - activity_score) + 0.3 * (1 - habit_score) + 0.4 * confidence_score
            # control_factor = max(0.2, min(0.8, control_factor))
            
            # controlled_power = original_power * control_factor
            # power_reduction = original_power - controlled_power
            
        elif decision == 'keep_on':
            # Slight optimization only

            controlled_power = original_power
            power_reduction = 0

            # controlled_power = original_power * 0.95
            # power_reduction = original_power - controlled_power
            
        else:
            # Unknown decision, maintain original power
            controlled_power = original_power
            power_reduction = 0
        
        return {
            'controlled_power': max(0, controlled_power),
            'power_reduction': power_reduction,
            'reduction_percentage': (power_reduction / original_power * 100) if original_power > 0 else 0
        }
    
    def process_all_power_data(self, df):
        """Process ALL power data (not just phantom load)"""
        print(f"🔄 Starting complete power data processing...")

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 🔧 修正：先排序確保數據正確
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 🔧 修正：獲取最新日期（確保是 datetime 格式）
        latest_date = df['timestamp'].max()
        print(f"📅 Latest date in data: {latest_date}")

        start_date = pd.to_datetime('2025-07-14').tz_localize('UTC')
        end_date = pd.to_datetime('2025-07-20').tz_localize('UTC')
        
        # # 設定時間範圍
        # start_date = latest_date - pd.Timedelta(days=7)  # 最近25天
        # end_date = latest_date

        df_filtered = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)].copy()
        
        # 過濾數據
        # df_filtered = df[df['timestamp'] >= start_date].copy()
        
        print(f"📅 Analysis period: {start_date.date()} to {end_date.date()}")
        print(f"📊 Original data points: {len(df)}")
        print(f"📊 Filtered data points: {len(df_filtered)} (last 25 days)")
        
        # 使用過濾後的數據
        opportunities = self._generate_all_power_opportunities(df_filtered)
        print(f"✅ Generated {len(opportunities)} opportunities from all power levels")
        
        if len(opportunities) == 0:
            print("❌ No opportunities found")
            return
        
        # Process each opportunity
        for i, opportunity in enumerate(opportunities):
            try:
                original_power = opportunity['power_watt']
                
                # Use your existing system for decisions
                decision_result = self._make_decision_for_all_power(opportunity)
                
                # Calculate controlled power
                power_control = self._calculate_controlled_power_all_levels(
                    original_power, decision_result, opportunity
                )
                
                # Store results
                result = {
                    'timestamp': opportunity['start_time'],
                    'end_timestamp': opportunity['end_time'],
                    'duration_minutes': (opportunity['end_time'] - opportunity['start_time']).total_seconds() / 60,
                    'original_power': original_power,
                    'controlled_power': power_control['controlled_power'],
                    'power_reduction': power_control['power_reduction'],
                    'reduction_percentage': power_control['reduction_percentage'],
                    'decision': decision_result['decision'],
                    'original_decision': decision_result['original_decision'],
                    'activity_score': decision_result['activity_score'],
                    'habit_score': decision_result['habit_score'],
                    'confidence_score': decision_result['confidence_score'],
                    'filter_applied': decision_result['filter_applied'],
                    'filter_reason': decision_result['filter_reason'],
                    'power_category': self._categorize_power(original_power)
                }
                
                self.analysis_results.append(result)
                self.original_powers.append(original_power)
                self.controlled_powers.append(power_control['controlled_power'])
                
                # Show detailed information for first 5 results
                if i < 5:
                    print(f"\n--- Sample {i+1} ---")
                    print(f"Time: {opportunity['start_time'].strftime('%Y-%m-%d %H:%M')}")
                    print(f"Original Power: {original_power:.1f}W → Controlled: {power_control['controlled_power']:.1f}W")
                    print(f"Category: {result['power_category']}")
                    print(f"Decision: {decision_result['decision']}")
                    print(f"Savings: {power_control['power_reduction']:.1f}W ({power_control['reduction_percentage']:.1f}%)")
                
            except Exception as e:
                print(f"⚠️ Error processing opportunity {i+1}: {e}")
                continue
        
        print(f"\n✅ Successfully processed {len(self.analysis_results)} opportunities")
    
    def _categorize_power(self, power):
        """Categorize power levels"""
        if power < 37:
            return 'Very Low (Phantom)'
        elif power < 82:
            return 'Low (Standby)'
        elif power < 223:
            return 'Medium (Active)'
        else:
            return 'High (Peak)'
        
    def analyze_power_categories(self, df):
        """分析不同功率類別的消耗和占比 - 純print版本"""
        print(f"\n{'='*70}")
        print("📊 Power Category Analysis - Phantom Load, Light Use, Regular Use")
        print(f"{'='*70}")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 計算時間間隔（假設是15分鐘）
        time_interval_hours = 15 / 60  # 15分鐘 = 0.25小時
        
        # 功率分類定義（根據您的前兩個代碼的分類邏輯）
        def categorize_power(power):
            if power == 0:
                return 'no-use'
            elif power <= 1.5:  # 根據您的threshold
                return 'phantom'
            elif power <= 36:   # 根據您的閾值設定
                return 'phantom'
            elif power <= 81:   # phantom load | light use 閾值
                return 'light'
            else:
                return 'regular'
        
        # 應用分類
        df['power_category'] = df['power'].apply(categorize_power)
        
        # 計算各類別統計
        category_stats = {}
        total_energy = 0
        
        print(f"🔍 Analysis Settings:")
        print(f"   Time interval: {time_interval_hours*60:.0f} minutes per data point")
        print(f"   Total data points: {len(df):,}")
        print(f"   Analysis period: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
        print(f"   Power classification thresholds:")
        print(f"     - No use: 0W")
        print(f"     - Phantom load: 0.1W - 36W") 
        print(f"     - Light use: 36.1W - 81W")
        print(f"     - Regular use: >81W")
        
        for category in ['no-use', 'phantom', 'light', 'regular']:
            category_data = df[df['power_category'] == category]
            
            if len(category_data) > 0:
                # 基本統計
                count = len(category_data)
                percentage = count / len(df) * 100
                
                # 功率統計
                min_power = category_data['power'].min()
                max_power = category_data['power'].max()
                mean_power = category_data['power'].mean()
                total_power = category_data['power'].sum()
                
                # 能耗計算 (kWh)
                energy_kwh = total_power * time_interval_hours / 1000
                total_energy += energy_kwh
                
                category_stats[category] = {
                    'count': count,
                    'percentage': percentage,
                    'min_power': min_power,
                    'max_power': max_power,
                    'mean_power': mean_power,
                    'total_power': total_power,
                    'energy_kwh': energy_kwh
                }
            else:
                category_stats[category] = {
                    'count': 0, 'percentage': 0, 'min_power': 0,
                    'max_power': 0, 'mean_power': 0, 'total_power': 0,
                    'energy_kwh': 0
                }
        
        # 計算能耗占比
        for category in category_stats:
            if total_energy > 0:
                category_stats[category]['energy_percentage'] = (
                    category_stats[category]['energy_kwh'] / total_energy * 100
                )
            else:
                category_stats[category]['energy_percentage'] = 0
        
        # 顯示詳細結果
        print(f"\n📋 Power Category Breakdown:")
        print(f"{'='*70}")
        
        # 中文標籤映射
        category_labels = {
            'no-use': 'No Use (關閉)',
            'phantom': 'Phantom Load (待機)',
            'light': 'Light Use (輕度使用)',
            'regular': 'Regular Use (正常使用)'
        }
        
        for category in ['no-use', 'phantom', 'light', 'regular']:
            stats = category_stats[category]
            label = category_labels.get(category, category)
            
            if stats['count'] > 0:
                print(f"\n🔸 {label}:")
                print(f"   📊 Data points: {stats['count']:,} ({stats['percentage']:.1f}%)")
                print(f"   ⚡ Power range: {stats['min_power']:.1f}W - {stats['max_power']:.1f}W")
                print(f"   📈 Average power: {stats['mean_power']:.1f}W")
                print(f"   🔋 Total consumption: {stats['energy_kwh']:.3f} kWh ({stats['energy_percentage']:.1f}%)")
                print(f"   💰 Cost: £{stats['energy_kwh'] * uk_electricity_rate:.3f}")
            else:
                print(f"\n🔸 {label}: No data found")
        
        # 總計
        print(f"\n📊 TOTAL SUMMARY:")
        print(f"{'='*40}")
        print(f"   🔋 Total energy consumption: {total_energy:.3f} kWh")
        print(f"   💰 Total cost: £{total_energy * uk_electricity_rate:.3f}")
        print(f"   📊 Total data points: {len(df):,}")
        print(f"   ⏱️  Analysis duration: {len(df) * time_interval_hours:.1f} hours")
        print(f"   📅 Days analyzed: {len(df) * time_interval_hours / 24:.1f} days")
        
        return category_stats, total_energy
    
    def perform_paired_t_test(self):
        """Perform paired t-test analysis for all power levels"""
        print(f"\n{'='*60}")
        print("📊 Performing Paired T-Test Analysis (ALL Power Levels)")
        print(f"{'='*60}")
        
        if len(self.original_powers) == 0:
            print("❌ No data available for analysis")
            return None
        
        # Perform paired t-test
        original = np.array(self.original_powers)
        controlled = np.array(self.controlled_powers)
        differences = original - controlled
        
        # Statistical test
        t_stat, p_value = stats.ttest_rel(original, controlled)
        
        # Basic statistics
        n = len(differences)
        mean_original = np.mean(original)
        mean_controlled = np.mean(controlled)
        mean_difference = np.mean(differences)
        std_difference = np.std(differences, ddof=1)

        sum_original = np.sum(original)/1000
        sum_controlled = np.sum(controlled)/1000
        sum_difference = sum_original - sum_controlled
        sum_percentage = sum_difference / sum_original *100
        print(f"original total power (kw):{sum_original:.2f} kw")
        print(f"controlled total power (kw):{sum_controlled:.2f} kw")
        print(f"total save : {sum_difference:.2f}w, reduce {sum_percentage:.2f}%")
        cost_original = sum_original * uk_electricity_rate
        cost_controlled = sum_controlled * uk_electricity_rate
        cost_difference = cost_original - cost_controlled
        cost_percentage = cost_difference / cost_original *100
        print(f"original total cost : £{cost_original:.2f}")
        print(f"controlled total cost : £{cost_controlled:.2f}")
        print(f"total save : £{cost_difference:.2f}, reduce {cost_percentage:.2f}%")

        
        # Confidence interval
        t_critical = stats.t.ppf(0.975, n-1)
        margin_error = t_critical * (std_difference / np.sqrt(n))
        ci_lower = mean_difference - margin_error
        ci_upper = mean_difference + margin_error
        
        # Effect size
        pooled_std = np.sqrt((np.var(original, ddof=1) + np.var(controlled, ddof=1)) / 2)
        cohens_d = mean_difference / pooled_std if pooled_std > 0 else 0
        
        # Energy savings
        total_energy_saved_kwh = np.sum(differences) / 1000
        percentage_reduction = (mean_difference / mean_original * 100) if mean_original > 0 else 0
        
        results = {
            'n_samples': n,
            'mean_original': mean_original,
            'mean_controlled': mean_controlled,
            'mean_difference': mean_difference,
            'std_difference': std_difference,
            't_statistic': t_stat,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'cohens_d': cohens_d,
            'total_energy_saved_kwh': total_energy_saved_kwh,
            'percentage_reduction': percentage_reduction
        }
        
        # Print results
        self._print_t_test_results(results)
        return results
    
    def _print_t_test_results(self, results):
        """Print t-test results"""
        print(f"\n🔍 Basic Statistics (ALL Power Levels):")
        print(f"   Sample size: {results['n_samples']}")
        print(f"   Mean original power: {results['mean_original']:.2f} W")
        print(f"   Mean controlled power: {results['mean_controlled']:.2f} W")
        print(f"   Mean power difference: {results['mean_difference']:.2f} W")
        
        print(f"\n📈 Statistical Test:")
        print(f"   t-statistic: {results['t_statistic']:.4f}")
        print(f"   p-value: {results['p_value']:.6f}")
        print(f"   95% Confidence Interval: [{results['ci_lower']:.2f}, {results['ci_upper']:.2f}] W")
        print(f"   Cohen's d: {results['cohens_d']:.4f}")
        
        print(f"\n💡 Energy Savings:")
        print(f"   Total energy saved: {results['total_energy_saved_kwh']:.3f} kWh")
        print(f"   Energy reduction percentage: {results['percentage_reduction']:.1f}%")
        print(f"   Cost savings: £{results['total_energy_saved_kwh'] * uk_electricity_rate:.3f}")
        
        # Statistical significance assessment
        if results['p_value'] < 0.001:
            significance = "Highly significant (p < 0.001)"
        elif results['p_value'] < 0.01:
            significance = "Very significant (p < 0.01)"
        elif results['p_value'] < 0.05:
            significance = "Significant (p < 0.05)"
        else:
            significance = "Not significant (p ≥ 0.05)"
        
        print(f"\n🎯 Conclusion:")
        print(f"   Statistical significance: {significance}")
        
        if results['p_value'] < 0.05:
            print(f"   ✅ Your fuzzy logic control system significantly reduces TOTAL power consumption")
        else:
            print(f"   ⚠️ Control effect does not reach statistical significance")
    
    def create_complete_visualization(self):
        """Create separate visualizations for ALL power levels - each chart displayed individually"""
        if len(self.analysis_results) == 0:
            print("❌ No data for visualization")
            return
        
        print("\n📊 Generating individual charts for COMPLETE power analysis...")
        
        # Prepare data
        timestamps = [r['timestamp'] for r in self.analysis_results]
        original_powers = [r['original_power'] for r in self.analysis_results]
        controlled_powers = [r['controlled_power'] for r in self.analysis_results]
        df_results = pd.DataFrame(self.analysis_results)
        
        # Chart 1: Time series comparison (ALL power levels)
        plt.figure(figsize=(15, 8))
        plt.plot(timestamps, original_powers, 'b-', linewidth=2, label='Before Control', alpha=1.0)
        plt.plot(timestamps, controlled_powers, 'r-', linewidth=1, label='After Control', alpha=1.0)
        plt.fill_between(timestamps, original_powers, controlled_powers, alpha=0.3, color='green', label='Energy Saved')
        plt.xlabel('Time', fontsize=16)
        plt.ylabel('Power (W)', fontsize=16)
        plt.title('Complete Power Control: Before vs After (ALL Levels)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Chart 2: Power distribution comparison (ALL levels)
        plt.figure(figsize=(12, 8))
        plt.hist(original_powers, bins=30, alpha=0.6, color='blue', label='Before Control', density=True)
        plt.hist(controlled_powers, bins=30, alpha=0.6, color='red', label='After Control', density=True)
        plt.xlabel('Power (W)', fontsize=12)
        plt.ylabel('Probability Density', fontsize=12)
        plt.title('Complete Power Distribution Comparison (ALL Levels)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Chart 3: Before vs After scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(original_powers, controlled_powers, alpha=0.6, c='purple', s=50)
        max_power = max(max(original_powers), max(controlled_powers))
        plt.plot([0, max_power], [0, max_power], 'k--', alpha=0.5, label='No Change Line')
        plt.xlabel('Original Power (W)', fontsize=12)
        plt.ylabel('Controlled Power (W)', fontsize=12)
        plt.title('Before vs After Power Relationship (ALL Levels)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Chart 4: Decision distribution
        plt.figure(figsize=(10, 8))
        decisions = [r['decision'] for r in self.analysis_results]
        decision_counts = pd.Series(decisions).value_counts()
        
        colors = {'suggest_shutdown': '#FF6B6B', 'send_notification': '#4ECDC4', 
                 'delay_decision': '#45B7D1', 'keep_on': '#96CEB4'}
        pie_colors = [colors.get(d, '#95A5A6') for d in decision_counts.index]
        
        plt.pie(decision_counts.values, labels=decision_counts.index, 
               colors=pie_colors, autopct='%1.1f%%', startangle=90)
        plt.title('Control Decision Distribution (ALL Levels)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Chart 5: Power category analysis
        plt.figure(figsize=(12, 8))
        category_savings = df_results.groupby('power_category').agg({
            'power_reduction': 'mean',
            'original_power': 'count'
        }).round(2)
        
        categories = category_savings.index
        avg_savings = category_savings['power_reduction']
        counts = category_savings['original_power']
        
        bars = plt.bar(range(len(categories)), avg_savings, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        plt.xlabel('Power Category', fontsize=12)
        plt.ylabel('Average Power Reduction (W)', fontsize=12)
        plt.title('Average Energy Savings by Power Category (ALL Levels)', fontsize=14, fontweight='bold')
        plt.xticks(range(len(categories)), categories, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, saving, count) in enumerate(zip(bars, avg_savings, counts)):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(avg_savings)*0.02,
                    f'{saving:.1f}W\n({count} samples)', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        # Chart 6: Control effectiveness by power level
        plt.figure(figsize=(12, 8))
        power_bins = [0, 20, 60, 100, 200]
        bin_labels = ['<20W', '20-60W', '60-100W', '>100W']
        
        df_results['power_bin'] = pd.cut(df_results['original_power'], bins=power_bins, labels=bin_labels)
        effectiveness = df_results.groupby('power_bin').agg({
            'reduction_percentage': 'mean',
            'original_power': 'count'
        }).round(1)
        
        bins = effectiveness.index
        percentages = effectiveness['reduction_percentage']
        bin_counts = effectiveness['original_power']
        
        bars = plt.bar(range(len(bins)), percentages, 
                      color=['#FF9999', '#99CCFF', '#99FF99', '#FFCC99'])
        plt.xlabel('Power Range', fontsize=12)
        plt.ylabel('Average Reduction Percentage (%)', fontsize=12)
        plt.title('Control Effectiveness by Power Range (ALL Levels)', fontsize=14, fontweight='bold')
        plt.xticks(range(len(bins)), bins)
        plt.grid(True, alpha=0.3)
        
        # Add percentage labels
        for bar, percentage, count in zip(bars, percentages, bin_counts):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(percentages)*0.02,
                    f'{percentage:.1f}%\n({count} samples)', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        # Chart 7: Cumulative energy consumption comparison
        plt.figure(figsize=(15, 8))
        cumulative_original = np.cumsum([r['original_power'] for r in self.analysis_results]) / 1000
        cumulative_controlled = np.cumsum([r['controlled_power'] for r in self.analysis_results]) / 1000
        
        plt.plot(timestamps, cumulative_original, 'b-', linewidth=2, label='Original Consumption')
        plt.plot(timestamps, cumulative_controlled, 'r-', linewidth=2, label='Controlled Consumption')
        plt.fill_between(timestamps, cumulative_original, cumulative_controlled, 
                        alpha=0.3, color='green', label='Total Energy Saved')
        
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Cumulative Energy (kWh)', fontsize=12)
        plt.title('Cumulative Energy Consumption Comparison (ALL Levels)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Chart 8: Hourly analysis
        plt.figure(figsize=(15, 8))
        df_results['hour'] = pd.to_datetime(df_results['timestamp']).dt.hour
        hourly_analysis = df_results.groupby('hour').agg({
            'original_power': 'mean',
            'controlled_power': 'mean',
            'power_reduction': 'mean'
        }).round(2)
        
        hours = hourly_analysis.index
        plt.plot(hours, hourly_analysis['original_power'], 'b-', linewidth=2, label='Original Power', marker='o')
        plt.plot(hours, hourly_analysis['controlled_power'], 'r-', linewidth=2, label='Controlled Power', marker='s')
        plt.plot(hours, hourly_analysis['power_reduction'], 'g-', linewidth=2, label='Power Reduction', marker='^')
        
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Power (W)', fontsize=12)
        plt.title('Hourly Power Analysis (ALL Levels)', fontsize=14, fontweight='bold')
        plt.xticks(range(0, 24, 2))
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def create_phantom_load_visualization(self):
        """Create separate visualizations for PHANTOM LOAD ONLY - each chart displayed individually"""
        # Filter only phantom load data
        phantom_results = [r for r in self.analysis_results if r['original_power'] < 37]  # Adjust threshold as needed
        
        if len(phantom_results) == 0:
            print("❌ No phantom load data for visualization")
            return
        
        print(f"\n📊 Generating individual charts for PHANTOM LOAD ONLY analysis ({len(phantom_results)} samples)...")
        
        # Prepare phantom load data
        phantom_timestamps = [r['timestamp'] for r in phantom_results]
        phantom_original = [r['original_power'] for r in phantom_results]
        phantom_controlled = [r['controlled_power'] for r in phantom_results]
        phantom_df = pd.DataFrame(phantom_results)
        
        # Phantom Chart 1: Time series comparison (PHANTOM LOAD ONLY)
        plt.figure(figsize=(15, 8))
        plt.plot(phantom_timestamps, phantom_original, 'b-', linewidth=2, label='Before Control (Phantom)', alpha=0.7)
        plt.plot(phantom_timestamps, phantom_controlled, 'r-', linewidth=2, label='After Control (Phantom)', alpha=0.7)
        plt.fill_between(phantom_timestamps, phantom_original, phantom_controlled, alpha=0.3, color='green', label='Energy Saved')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Power (W)', fontsize=12)
        plt.title('Phantom Load Power Control: Before vs After (<60W Only)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Phantom Chart 2: Power distribution comparison (PHANTOM LOAD ONLY)
        plt.figure(figsize=(12, 8))
        plt.hist(phantom_original, bins=20, alpha=0.6, color='blue', label='Before Control (Phantom)', density=True)
        plt.hist(phantom_controlled, bins=20, alpha=0.6, color='red', label='After Control (Phantom)', density=True)
        plt.xlabel('Power (W)', fontsize=12)
        plt.ylabel('Probability Density', fontsize=12)
        plt.title('Phantom Load Power Distribution Comparison (<60W Only)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Phantom Chart 3: Before vs After scatter plot (PHANTOM LOAD ONLY)
        plt.figure(figsize=(10, 8))
        plt.scatter(phantom_original, phantom_controlled, alpha=0.6, c='purple', s=50)
        max_phantom = max(max(phantom_original), max(phantom_controlled))
        plt.plot([0, max_phantom], [0, max_phantom], 'k--', alpha=0.5, label='No Change Line')
        plt.xlabel('Original Power (W)', fontsize=12)
        plt.ylabel('Controlled Power (W)', fontsize=12)
        plt.title('Phantom Load: Before vs After Power Relationship (<60W Only)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Phantom Chart 4: Decision distribution (PHANTOM LOAD ONLY)
        plt.figure(figsize=(10, 8))
        phantom_decisions = [r['decision'] for r in phantom_results]
        phantom_decision_counts = pd.Series(phantom_decisions).value_counts()
        
        colors = {'suggest_shutdown': '#FF6B6B', 'send_notification': '#4ECDC4', 
                 'delay_decision': '#45B7D1', 'keep_on': '#96CEB4'}
        pie_colors = [colors.get(d, '#95A5A6') for d in phantom_decision_counts.index]
        
        plt.pie(phantom_decision_counts.values, labels=phantom_decision_counts.index, 
               colors=pie_colors, autopct='%1.1f%%', startangle=90)
        plt.title('Phantom Load Control Decision Distribution (<60W Only)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Phantom Chart 5: Power reduction distribution (PHANTOM LOAD ONLY)
        plt.figure(figsize=(12, 8))
        phantom_reductions = [r['power_reduction'] for r in phantom_results]
        plt.hist(phantom_reductions, bins=20, alpha=0.7, color='green', edgecolor='black')
        plt.axvline(np.mean(phantom_reductions), color='red', linestyle='--', linewidth=2,
                   label=f'Average Savings: {np.mean(phantom_reductions):.1f}W')
        plt.xlabel('Power Reduction (W)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Phantom Load Power Reduction Distribution (<60W Only)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Phantom Chart 6: Cumulative energy savings (PHANTOM LOAD ONLY)
        plt.figure(figsize=(15, 8))
        phantom_cumulative_savings = np.cumsum(phantom_reductions) / 1000
        
        plt.plot(phantom_timestamps, phantom_cumulative_savings, 'g-', linewidth=2)
        plt.fill_between(phantom_timestamps, 0, phantom_cumulative_savings, alpha=0.3, color='green')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Cumulative Energy Saved (kWh)', fontsize=12)
        plt.title('Phantom Load Cumulative Energy Savings Trend (<60W Only)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Phantom Chart 7: Hourly phantom load analysis
        plt.figure(figsize=(15, 8))
        phantom_df['hour'] = pd.to_datetime(phantom_df['timestamp']).dt.hour
        phantom_hourly = phantom_df.groupby('hour').agg({
            'original_power': 'mean',
            'controlled_power': 'mean',
            'power_reduction': 'mean'
        }).round(2)
        
        hours = phantom_hourly.index
        plt.plot(hours, phantom_hourly['original_power'], 'b-', linewidth=2, label='Original Power', marker='o')
        plt.plot(hours, phantom_hourly['controlled_power'], 'r-', linewidth=2, label='Controlled Power', marker='s')
        plt.plot(hours, phantom_hourly['power_reduction'], 'g-', linewidth=2, label='Power Reduction', marker='^')
        
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Power (W)', fontsize=12)
        plt.title('Hourly Phantom Load Analysis (<60W Only)', fontsize=14, fontweight='bold')
        plt.xticks(range(0, 24, 2))
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Phantom Chart 8: Decision effectiveness for phantom load
        plt.figure(figsize=(12, 8))
        phantom_decision_avg = phantom_df.groupby('decision').agg({
            'power_reduction': 'mean',
            'original_power': 'count'
        }).round(2)
        
        decisions = phantom_decision_avg.index
        avg_reductions = phantom_decision_avg['power_reduction']
        decision_counts = phantom_decision_avg['original_power']
        
        bars = plt.bar(range(len(decisions)), avg_reductions, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(decisions)])
        plt.xlabel('Decision Type', fontsize=12)
        plt.ylabel('Average Power Reduction (W)', fontsize=12)
        plt.title('Phantom Load: Average Energy Savings by Decision Type (<60W Only)', fontsize=14, fontweight='bold')
        plt.xticks(range(len(decisions)), decisions, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, reduction, count in zip(bars, avg_reductions, decision_counts):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(avg_reductions)*0.02,
                    f'{reduction:.1f}W\n({count} samples)', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def _create_detailed_complete_analysis(self):
        """This method is no longer needed as all charts are now displayed individually"""
        pass
    
    def run_complete_analysis(self):
        """Run complete analysis for ALL power levels"""
        print(f"\n{'='*80}")
        print("🚀 Complete Power Analysis - ALL Energy Consumption Levels")
        print(f"{'='*80}")
        
        if not HAS_FUZZY_SYSTEM or not HAS_DECISION_SYSTEM:
            print("❌ Missing required modules, cannot perform analysis")
            return
        
        try:
            # 1. Load data
            print("📁 Loading data...")
            df = pd.read_csv(self.data_file)
            print(f"✅ Loaded {len(df)} data points")
            print(f"📊 Power range: {df['power'].min():.1f}W - {df['power'].max():.1f}W")

            # 🆕 添加這些代碼
            print("\n🕒 Filtering data for 0714-0720 range...")
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            start_date = pd.to_datetime('2025-07-14').tz_localize('UTC')
            end_date = pd.to_datetime('2025-07-20').tz_localize('UTC')
            
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            
            df_filtered = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)].copy()
            
            print(f"📅 Using 0714-0720 data: {len(df_filtered)} points")

            category_stats, total_energy = self.analyze_power_categories(df_filtered)
            
            # 2. Process all power data
            self.process_all_power_data(df_filtered)
            
            # 3. Statistical analysis
            t_test_results = self.perform_paired_t_test()
            
            # 4. Visualization for both complete and phantom load analysis
            print("\n📊 Generating complete power analysis charts...")
            self.create_complete_visualization()
            
            print("\n📊 Generating phantom load only analysis charts...")
            self.create_phantom_load_visualization()
            
            # 5. Final summary
            self._print_final_summary(t_test_results, category_stats, total_energy) 
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def _print_final_summary(self, t_test_results, category_stats=None, total_energy=None):
        print(f"\n{'='*80}")
        print("🎉 Complete Power Analysis - Final Summary")
        print(f"{'='*80}")
        
        # 🆕 首先顯示原始數據分類統計
        if category_stats and total_energy:
            print(f"\n📋 Original Data Summary by Category:")
            print(f"{'-'*50}")
            
            category_labels = {
                'no-use': 'No Use (關閉)',
                'phantom': 'Phantom Load (待機)', 
                'light': 'Light Use (輕度使用)',
                'regular': 'Regular Use (正常使用)'
            }
            
            for category in ['no-use', 'phantom', 'light', 'regular']:
                if category in category_stats and category_stats[category]['count'] > 0:
                    stats = category_stats[category]
                    label = category_labels[category]
                    print(f"   🔸 {label}:")
                    print(f"      Points: {stats['count']:,} ({stats['percentage']:.1f}%)")
                    print(f"      Energy: {stats['energy_kwh']:.3f} kWh ({stats['energy_percentage']:.1f}%)")
                    print(f"      Cost: £{stats['energy_kwh'] * uk_electricity_rate:.3f}")
            
            print(f"\n   🔋 Original Total: {total_energy:.3f} kWh")
            print(f"   💰 Original Cost: £{total_energy * uk_electricity_rate:.3f}")
        
        # 原有的控制系統分析結果
        if t_test_results:
            total_savings = np.sum([r['power_reduction'] for r in self.analysis_results]) / 1000
            
            print(f"\n📊 Control System Performance:")
            print(f"{'-'*50}")
            print(f"   ✅ Analyzed opportunities: {len(self.analysis_results)}")
            print(f"   📊 Power range: {min(self.original_powers):.1f}W - {max(self.original_powers):.1f}W")
            print(f"   💡 Energy saved: {total_savings:.3f} kWh")
            print(f"   💰 Cost savings: £{total_savings * uk_electricity_rate:.3f}")
            print(f"   📈 Average reduction: {t_test_results['percentage_reduction']:.1f}%")
            
            if t_test_results['p_value'] < 0.05:
                print(f"\n   🎯 Statistical Result: Significant reduction achieved!")
                print(f"       (p = {t_test_results['p_value']:.6f}, Cohen's d = {t_test_results['cohens_d']:.3f})")
            else:
                print(f"\n   ⚠️ Statistical Result: Not statistically significant")
                print(f"       (p = {t_test_results['p_value']:.6f})")
        
        print(f"{'='*80}")


def main():
    """Main program for complete power analysis"""
    # Your data file path
    data_file_path = "C:/Users/王俞文/OneDrive - University of Glasgow/文件/glasgow/msc project/data/complete_power_data_with_history.csv"
    
    # Check if file exists
    if not os.path.exists(data_file_path):
        print(f"❌ Cannot find data file: {data_file_path}")
        print("Please confirm the file path is correct")
        return
    
    # Check required modules
    if not HAS_FUZZY_SYSTEM or not HAS_DECISION_SYSTEM:
        print("❌ Please ensure the following files are in the same directory:")
        print("   - fuzzy_logic_control.py (your existing module)")
        print("   - related decision system modules")
        return
    
    # Create complete analyzer
    analyzer = CompletePowerAnalyzer(data_file_path)
    
    # Run complete analysis
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    print("🔧 COMPLETE Power Analysis - ALL Energy Consumption Levels")
    print("📊 Using Your Existing Fuzzy Logic System")
    print("📈 Paired T-Test & Effect Evaluation for ALL Power Ranges")
    print("="*70)
    
    main()