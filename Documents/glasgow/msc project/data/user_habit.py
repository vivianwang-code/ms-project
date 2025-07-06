# user habit - improved version
# usage probability, usage stability, time of the day

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ImprovedUserHabitScoreModule:

    def __init__(self):
        self.time_slots = 96  # 96個時段，每15分鐘一個 (24*4)
        self.transition_data = None
        self.usage_probability_matrix = {}
        self.stability_matrix = {}
        self.time_factor_matrix = {}
        self.habit_rules = []
        self.membership_parameters = {}
        self.data_quality_report = {}

    def validate_data_quality(self, df):
        """驗證輸入數據的質量"""
        print("==== Data Quality Validation ====")
        issues = []
        
        # 檢查必要欄位
        required_columns = ['timestamp', 'power_state', 'is_on', 'is_off']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing columns: {missing_columns}")
        
        # 檢查數據類型
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except:
                issues.append("Timestamp column cannot be converted to datetime")
        
        # 檢查數據範圍
        if len(df) < 100:
            issues.append(f"Insufficient data: only {len(df)} records")
        
        # 檢查狀態一致性
        if 'is_on' in df.columns and 'is_off' in df.columns:
            inconsistent_states = df[(df['is_on'] == True) & (df['is_off'] == True)]
            if len(inconsistent_states) > 0:
                issues.append(f"Found {len(inconsistent_states)} inconsistent on/off states")
        
        # 檢查時間序列完整性
        if 'timestamp' in df.columns:
            df_sorted = df.sort_values('timestamp')
            time_gaps = df_sorted['timestamp'].diff()
            large_gaps = time_gaps[time_gaps > pd.Timedelta(hours=4)]
            if len(large_gaps) > 10:
                issues.append(f"Found {len(large_gaps)} large time gaps > 4 hours")
        
        self.data_quality_report = {
            'total_records': len(df),
            'issues': issues,
            'quality_score': max(0, 1.0 - len(issues) * 0.2)
        }
        
        print(f"Data quality score: {self.data_quality_report['quality_score']:.2f}")
        if issues:
            print("Issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✓ No major data quality issues found")
        
        return issues

    def preprocess_data_enhanced(self, df):
        """增強的數據預處理"""
        print("==== Enhanced Data Preprocessing ====")
        
        # 複製數據避免修改原始數據
        df = df.copy()
        
        # 處理缺失值
        original_nan_count = df.isnull().sum().sum()
        if original_nan_count > 0:
            print(f"Handling {original_nan_count} missing values...")
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 確保時間格式正確
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 處理異常的時間差值
        if 'time_diff_seconds' in df.columns:
            # 異常值檢測和處理
            Q1 = df['time_diff_seconds'].quantile(0.25)
            Q3 = df['time_diff_seconds'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = max(0, Q1 - 1.5 * IQR)  # 確保下界不為負
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (df['time_diff_seconds'] < lower_bound) | (df['time_diff_seconds'] > upper_bound)
            outliers_count = outliers_mask.sum()
            
            if outliers_count > 0:
                print(f"Handling {outliers_count} time_diff outliers...")
                df.loc[outliers_mask, 'time_diff_seconds'] = df['time_diff_seconds'].median()
        
        # 確保狀態欄位的數據類型正確
        boolean_columns = ['is_on', 'is_off', 'is_phantom_load', 'is_light_use', 'is_regular_use']
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        print(f"✓ Preprocessing completed. Final dataset: {len(df)} records")
        return df

    def load_data(self, file_path):
        print("==== Loading Usage Data for User Habit Score ====")
        
        try:
            # 讀取數據
            df = pd.read_csv(file_path)
            
            # 數據質量驗證
            quality_issues = self.validate_data_quality(df)
            if len(quality_issues) > 3:
                print("⚠️  Warning: Multiple data quality issues detected. Results may be unreliable.")
            
            # 增強預處理
            df = self.preprocess_data_enhanced(df)
            
            # 轉換時間格式並添加特徵
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            df['weekday'] = df['timestamp'].dt.weekday
            df['is_weekend'] = df['weekday'] >= 5
            df['day_type'] = df['is_weekend'].map({True: 'weekend', False: 'weekday'})
            df['time_slot'] = df['hour'] * 4 + df['minute'] // 15  # 0-95個時段
            
            print(f"✓ Loaded usage data: {len(df)} records")
            print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"Weekdays: {sum(~df['is_weekend'])}, Weekends: {sum(df['is_weekend'])}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None

    def triangular_membership(self, x, a, b, c):

        # 處理輸入為數組的情況
        if isinstance(x, (list, np.ndarray)):
            x = np.array(x)
            return np.array([self.triangular_membership(xi, a, b, c) for xi in x])
        
        # 檢查NaN值
        if np.isnan(x) or np.isnan(a) or np.isnan(b) or np.isnan(c):
            return 0.0
        
        # 檢查特殊情況
        if a == b == c:
            return 1.0 if x == a else 0.0
        
        # 確保參數順序正確
        if a > b or b > c:
            print(f"Warning: Invalid triangular parameters: a={a}, b={b}, c={c}")
            return 0.0
        
        try:
            if x <= a or x >= c:
                return 0.0
            elif x == b:
                return 1.0
            elif a < x < b:
                return (x - a) / (b - a) if (b - a) > 0 else 0.0
            else:  # b < x < c
                return (c - x) / (c - b) if (c - b) > 0 else 0.0
        except:
            return 0.0

    def state_transitions(self, df):
        print("==== State Transitions Analysis ====")

        print("Available state columns:")
        state_cols = ['power_state', 'is_on', 'is_off', 'is_phantom_load', 'is_light_use', 'is_regular_use']
        for col in state_cols:
            if col in df.columns:
                print(f"  {col}: {df[col].value_counts().to_dict()}")
        
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)

        turn_off_events = pd.DataFrame()
        turn_on_events = pd.DataFrame()

        if 'is_on' in df.columns and 'is_off' in df.columns:
            df_sorted['is_on_prev'] = df_sorted['is_on'].shift(1)

            # 關機事件：上一個是開機，現在是關機
            turn_off_events = df_sorted[
                (df_sorted['is_on_prev'] == True) & 
                (df_sorted['is_off'] == True)
            ].copy()

            # 開機事件：上一個是關機，現在是開機
            turn_on_events = df_sorted[
                (df_sorted['is_on_prev'] == False) & 
                (df_sorted['is_on'] == True)
            ].copy()
            
            print(f"\nTransition Analysis:")
            print(f"Turn-off events: {len(turn_off_events)}")
            print(f"Turn-on events: {len(turn_on_events)}")

        # 建立轉換記錄
        transition_records = []
        for idx, turn_off in turn_off_events.iterrows():
            turn_off_time = turn_off['timestamp']
            turn_off_slot = turn_off['time_slot']
            turn_off_day_type = turn_off['day_type']
            
            # 找到下一個開機事件
            next_turn_on = turn_on_events[
                turn_on_events['timestamp'] > turn_off_time
            ]
            
            if len(next_turn_on) > 0:
                next_on_time = next_turn_on.iloc[0]['timestamp']
                interval_minutes = (next_on_time - turn_off_time).total_seconds() / 60.0
                
                # 過濾異常的間隔時間
                if 5 <= interval_minutes <= 2880:  # 5分鐘到48小時之間
                    transition_records.append({
                        'turn_off_time': turn_off_time,
                        'turn_on_time': next_on_time,
                        'interval_minutes': interval_minutes,
                        'time_slot': turn_off_slot,
                        'day_type': turn_off_day_type,
                        'turn_off_hour': turn_off_time.hour
                    })
        
        self.transition_data = pd.DataFrame(transition_records)
        
        print(f"\nCreated {len(self.transition_data)} valid transition records")
        
        if len(self.transition_data) > 0:
            intervals = self.transition_data['interval_minutes']
            print(f"Interval statistics:")
            print(f"  Min: {intervals.min():.1f} minutes")
            print(f"  Max: {intervals.max():.1f} minutes") 
            print(f"  Mean: {intervals.mean():.1f} minutes")
            print(f"  Median: {intervals.median():.1f} minutes")
        else:
            print("⚠️  No valid transitions found - using fallback probability calculation")
        
        return self.transition_data

    def calculate_usage_probability(self):
        print("==== Calculating Usage Probability ====")

        if self.transition_data is None or len(self.transition_data) == 0:
            print("⚠️  No transition data - using fallback probability calculation")
            self._calculate_fallback_probability()
            return self.usage_probability_matrix
        
        valid_entries = 0
        
        for day_type in ['weekday', 'weekend']:
            for time_slot in range(self.time_slots):
                mask = (self.transition_data['day_type'] == day_type) & \
                       (self.transition_data['time_slot'] == time_slot)
                
                slot_data = self.transition_data[mask]

                if len(slot_data) < 1:  # 降低最低要求
                    # 使用默認值
                    self.usage_probability_matrix[(day_type, time_slot)] = {
                        'short_prob': 0.3,
                        'medium_prob': 0.4,
                        'long_prob': 0.3,
                        'immediate_time': 30.0,
                        'short_time': 120.0,
                        'long_time': 360.0,
                        'sample_size': 0
                    }
                    continue

                intervals = slot_data['interval_minutes'].values
                
                # 安全的百分位數計算
                try:
                    # 使用更保守的百分位數
                    min_interval = max(5.0, np.min(intervals))
                    q25 = np.percentile(intervals, 25)
                    q50 = np.percentile(intervals, 50)
                    q75 = np.percentile(intervals, 75)
                    max_interval = min(2880.0, np.max(intervals))  # 限制最大值
                    
                    # 確保順序正確
                    if not (min_interval <= q25 <= q50 <= q75 <= max_interval):
                        print(f"⚠️  Invalid percentiles for {day_type} slot {time_slot}, using defaults")
                        self.usage_probability_matrix[(day_type, time_slot)] = {
                            'short_prob': 0.3,
                            'medium_prob': 0.4,
                            'long_prob': 0.3,
                            'immediate_time': 30.0,
                            'short_time': 120.0,
                            'long_time': 360.0,
                            'sample_size': len(slot_data)
                        }
                        continue
                    
                    # 定義模糊集合
                    fuzzy_sets = {
                        'short': (min_interval, q25, q50),
                        'medium': (q25, q50, q75), 
                        'long': (q50, q75, max_interval)
                    }

                    # 計算模糊機率
                    fuzzy_probs = {}
                    for category, (a, b, c) in fuzzy_sets.items():
                        memberships = []
                        for interval in intervals:
                            membership = self.triangular_membership(interval, a, b, c)
                            memberships.append(membership)
                        
                        avg_membership = np.mean(memberships) if memberships else 0.0
                        fuzzy_probs[category] = max(0.0, min(1.0, avg_membership))
                    
                    # 正規化機率
                    total_prob = sum(fuzzy_probs.values())
                    if total_prob > 0:
                        normalization_factor = 1.0 / total_prob
                        for category in fuzzy_probs:
                            fuzzy_probs[category] *= normalization_factor
                    else:
                        # 如果總機率為0，使用默認分布
                        fuzzy_probs = {'short': 0.33, 'medium': 0.34, 'long': 0.33}
                    
                    self.usage_probability_matrix[(day_type, time_slot)] = {
                        'short_prob': fuzzy_probs['short'],
                        'medium_prob': fuzzy_probs['medium'],
                        'long_prob': fuzzy_probs['long'],
                        'immediate_time': q25,
                        'short_time': q50,
                        'long_time': q75,
                        'sample_size': len(slot_data)
                    }
                    valid_entries += 1
                    
                except Exception as e:
                    print(f"⚠️  Error calculating probabilities for {day_type} slot {time_slot}: {e}")
                    self.usage_probability_matrix[(day_type, time_slot)] = {
                        'short_prob': 0.3,
                        'medium_prob': 0.4,
                        'long_prob': 0.3,
                        'immediate_time': 30.0,
                        'short_time': 120.0,
                        'long_time': 360.0,
                        'sample_size': len(slot_data)
                    }
                
        print(f"✓ Calculated usage probability for {valid_entries} time slots")
        return self.usage_probability_matrix

    def _calculate_fallback_probability(self):
        """當沒有轉換數據時的後備機率計算"""
        print("Using fallback probability calculation based on typical usage patterns...")
        
        # 基於一般使用模式的預設機率
        typical_patterns = {
            'weekday': {
                # 工作日模式：早上和晚上使用率較高
                'morning': (6, 10, 0.7),    # 6-10點，高使用機率
                'work': (10, 18, 0.4),      # 10-18點，中等使用機率  
                'evening': (18, 23, 0.8),   # 18-23點，高使用機率
                'night': (23, 6, 0.2)       # 23-6點，低使用機率
            },
            'weekend': {
                # 週末模式：較為分散的使用時間
                'morning': (8, 12, 0.6),
                'afternoon': (12, 18, 0.7),
                'evening': (18, 24, 0.8),
                'night': (0, 8, 0.3)
            }
        }
        
        for day_type in ['weekday', 'weekend']:
            for time_slot in range(self.time_slots):
                hour = time_slot // 4
                
                # 根據時段確定基礎機率
                base_prob = 0.3  # 默認機率
                patterns = typical_patterns[day_type]
                
                for period, (start_hour, end_hour, prob) in patterns.items():
                    if start_hour <= end_hour:
                        if start_hour <= hour < end_hour:
                            base_prob = prob
                            break
                    else:  # 跨夜的情況
                        if hour >= start_hour or hour < end_hour:
                            base_prob = prob
                            break
                
                # 添加隨機變化
                variation = np.random.normal(0, 0.1)
                adjusted_prob = max(0.1, min(0.9, base_prob + variation))
                
                self.usage_probability_matrix[(day_type, time_slot)] = {
                    'short_prob': adjusted_prob,
                    'medium_prob': 1.0 - adjusted_prob,
                    'long_prob': 0.1,
                    'immediate_time': 30.0,
                    'short_time': 120.0,
                    'long_time': 360.0,
                    'sample_size': 0
                }

    def calculate_usage_stability(self, df):
        print("==== Calculating Usage Stability ====")

        raw_stability_data = []
        
        for day_type in ['weekday', 'weekend']:
            for time_slot in range(self.time_slots):
                mask = (df['day_type'] == day_type) & (df['time_slot'] == time_slot)
                slot_data = df[mask]

                if len(slot_data) > 1:  # 至少需要2個數據點
                    try:
                        time_stability = self._calculate_time_consistency(slot_data)
                        duration_stability = self._calculate_duration_consistency(slot_data)
                        intensity_stability = self._calculate_intensity_consistency(slot_data)
                        
                        # 確保穩定性值在合理範圍內
                        time_stability = max(0.0, min(1.0, time_stability))
                        duration_stability = max(0.0, min(1.0, duration_stability))
                        intensity_stability = max(0.0, min(1.0, intensity_stability))
                        
                        raw_stability_data.append({
                            'day_type': day_type,
                            'time_slot': time_slot,
                            'time_stability': time_stability,
                            'duration_stability': duration_stability,
                            'intensity_stability': intensity_stability,
                            'sample_size': len(slot_data)
                        })
                    except Exception as e:
                        print(f"⚠️  Error calculating stability for {day_type} slot {time_slot}: {e}")
                        continue

        if not raw_stability_data:
            print("⚠️  No stability data available - using defaults")
            self._create_default_stability_matrix()
            return self.stability_matrix

        stability_df = pd.DataFrame(raw_stability_data)
        
        # 為每種穩定性類型計算模糊隸屬度
        stability_types = ['time_stability', 'duration_stability', 'intensity_stability']
        
        for stability_type in stability_types:
            values = stability_df[stability_type].values
            if len(values) > 0:
                # 使用安全的百分位數計算
                try:
                    min_val = max(0.0, np.min(values))
                    q25 = np.percentile(values, 25)
                    q50 = np.percentile(values, 50)
                    q75 = np.percentile(values, 75)
                    max_val = min(1.0, np.max(values))
                    
                    # 確保順序正確
                    if not (min_val <= q25 <= q50 <= q75 <= max_val):
                        print(f"⚠️  Invalid percentiles for {stability_type}, adjusting...")
                        min_val, q25, q50, q75, max_val = 0.0, 0.25, 0.5, 0.75, 1.0
                    
                    stability_fuzzy_sets = {
                        'low': (min_val, q25, q50),
                        'medium': (q25, q50, q75),
                        'high': (q50, q75, max_val)
                    }
                    
                    # 計算每個時段的模糊穩定性
                    for idx, row in stability_df.iterrows():
                        day_type = row['day_type']
                        time_slot = row['time_slot']
                        stability_value = row[stability_type]
                        
                        fuzzy_memberships = {}
                        for level, (a, b, c) in stability_fuzzy_sets.items():
                            membership = self.triangular_membership(stability_value, a, b, c)
                            fuzzy_memberships[f'{stability_type}_{level}'] = membership
                        
                        key = (day_type, time_slot)
                        if key not in self.stability_matrix:
                            self.stability_matrix[key] = {'sample_size': row['sample_size']}
                        
                        self.stability_matrix[key].update(fuzzy_memberships)
                        self.stability_matrix[key][f'{stability_type}_raw'] = stability_value
                
                except Exception as e:
                    print(f"⚠️  Error processing {stability_type}: {e}")
                    continue

        # 計算綜合穩定性
        self._calculate_overall_stability()
        
        print(f"✓ Calculated stability for {len(self.stability_matrix)} time slots")
        return self.stability_matrix

    def _create_default_stability_matrix(self):
        """創建默認的穩定性矩陣"""
        for day_type in ['weekday', 'weekend']:
            for time_slot in range(self.time_slots):
                self.stability_matrix[(day_type, time_slot)] = {
                    'time_stability_raw': 0.5,
                    'duration_stability_raw': 0.5,
                    'intensity_stability_raw': 0.5,
                    'overall_stability_raw': 0.5,
                    'time_stability_low': 0.5,
                    'time_stability_medium': 0.5,
                    'time_stability_high': 0.0,
                    'duration_stability_low': 0.5,
                    'duration_stability_medium': 0.5,
                    'duration_stability_high': 0.0,
                    'intensity_stability_low': 0.5,
                    'intensity_stability_medium': 0.5,
                    'intensity_stability_high': 0.0,
                    'overall_stability_low': 0.5,
                    'overall_stability_medium': 0.5,
                    'overall_stability_high': 0.0,
                    'sample_size': 0
                }

    def _calculate_overall_stability(self):
        """計算綜合穩定性"""
        for key in self.stability_matrix:
            if 'sample_size' in self.stability_matrix[key]:
                stability_scores = []
                
                for stability_type in ['time_stability', 'duration_stability', 'intensity_stability']:
                    raw_key = f'{stability_type}_raw'
                    if raw_key in self.stability_matrix[key]:
                        stability_scores.append(self.stability_matrix[key][raw_key])
                
                if stability_scores:
                    overall_stability = np.mean(stability_scores)
                    self.stability_matrix[key]['overall_stability_raw'] = overall_stability
                    
                    # 為整體穩定性計算模糊隸屬度（使用固定範圍）
                    self.stability_matrix[key]['overall_stability_low'] = self.triangular_membership(
                        overall_stability, 0.0, 0.25, 0.5
                    )
                    self.stability_matrix[key]['overall_stability_medium'] = self.triangular_membership(
                        overall_stability, 0.25, 0.5, 0.75
                    )
                    self.stability_matrix[key]['overall_stability_high'] = self.triangular_membership(
                        overall_stability, 0.5, 0.75, 1.0
                    )

    def _calculate_time_consistency(self, slot_data):
        """計算時間一致性"""
        if len(slot_data) < 2:
            return 0.3
        
        try:
            # 按日期分組，檢查每天該時段的使用情況
            daily_usage = slot_data.groupby(slot_data['timestamp'].dt.date)['is_on'].max()
            
            if len(daily_usage) < 2:
                return 0.4
            
            # 計算使用一致性
            usage_rate = daily_usage.mean()
            usage_variance = daily_usage.var()
            
            # 標準化方差（相對於最大可能方差）
            max_variance = usage_rate * (1 - usage_rate)  # 伯努利分布的方差
            if max_variance > 0:
                normalized_variance = min(1.0, usage_variance / max_variance)
                consistency = 1.0 - normalized_variance
            else:
                consistency = 1.0 if usage_rate == 1.0 or usage_rate == 0.0 else 0.5
            
            return max(0.0, min(1.0, consistency))
        except:
            return 0.3

    def _calculate_duration_consistency(self, slot_data):
        """計算持續時間一致性"""
        if len(slot_data) < 2:
            return 0.3
        
        try:
            # 檢查是否有時間差數據
            if 'time_diff_seconds' not in slot_data.columns:
                return 0.4
            
            # 按日期分組，計算每天該時段的總使用時長
            daily_duration = slot_data.groupby(slot_data['timestamp'].dt.date)['time_diff_seconds'].sum() / 60.0
            daily_duration = daily_duration[daily_duration > 0]  # 只考慮有使用的天數
            
            if len(daily_duration) < 2:
                return 0.4
            
            # 計算變異係數
            mean_duration = daily_duration.mean()
            std_duration = daily_duration.std()
            
            if mean_duration > 0:
                cv = std_duration / mean_duration
                stability = 1.0 / (1.0 + cv)  # CV越小，穩定性越高
            else:
                stability = 0.0
            
            return max(0.0, min(1.0, stability))
        except:
            return 0.3

    def _calculate_intensity_consistency(self, slot_data):
        """計算強度一致性"""
        if len(slot_data) < 2:
            return 0.3
        
        try:
            # 定義使用強度
            slot_data = slot_data.copy()
            slot_data['intensity'] = 0
            
            if 'is_regular_use' in slot_data.columns:
                slot_data.loc[slot_data['is_regular_use'] == 1, 'intensity'] = 3
            if 'is_light_use' in slot_data.columns:
                slot_data.loc[slot_data['is_light_use'] == 1, 'intensity'] = 2
            
            slot_data.loc[(slot_data['is_on'] == 1) & (slot_data['intensity'] == 0), 'intensity'] = 1
            
            # 按日期分組，計算每天該時段的平均強度
            daily_intensity = slot_data.groupby(slot_data['timestamp'].dt.date)['intensity'].mean()
            daily_intensity = daily_intensity[daily_intensity > 0]
            
            if len(daily_intensity) < 2:
                return 0.4
            
            # 計算強度變異（標準化）
            intensity_std = daily_intensity.std()
            max_std = 1.5  # 強度範圍0-3，理論最大標準差約為1.5
            
            if max_std > 0:
                normalized_variance = min(1.0, intensity_std / max_std)
                stability = 1.0 - normalized_variance
            else:
                stability = 1.0
            
            return max(0.0, min(1.0, stability))
        except:
            return 0.3

    def calculate_time_factor(self, df):
        print("==== Calculating Time Factor ====")

        raw_usage_data = []
        
        for day_type in ['weekday', 'weekend']:
            for time_slot in range(self.time_slots):
                mask = (df['day_type'] == day_type) & (df['time_slot'] == time_slot)
                slot_data = df[mask]
                
                if len(slot_data) > 0:
                    try:
                        total_records = len(slot_data)
                        usage_records = len(slot_data[slot_data['is_on'] == 1])
                        usage_rate = usage_records / total_records if total_records > 0 else 0
                        
                        # 計算加權使用率
                        regular_weight = 1.0
                        light_weight = 0.6
                        basic_weight = 0.3
                        
                        weighted_usage = 0
                        if 'is_regular_use' in slot_data.columns:
                            weighted_usage += len(slot_data[slot_data['is_regular_use'] == 1]) * regular_weight
                        if 'is_light_use' in slot_data.columns:
                            weighted_usage += len(slot_data[slot_data['is_light_use'] == 1]) * light_weight
                        
                        # 其他開機但非專門使用的情況
                        other_on = slot_data[
                            (slot_data['is_on'] == 1) & 
                            (slot_data.get('is_regular_use', False) == False) & 
                            (slot_data.get('is_light_use', False) == False)
                        ]
                        weighted_usage += len(other_on) * basic_weight
                        
                        weighted_usage_rate = weighted_usage / total_records if total_records > 0 else 0
                        weighted_usage_rate = max(0.0, min(1.0, weighted_usage_rate))  # 確保在[0,1]範圍內
                        
                        raw_usage_data.append({
                            'day_type': day_type,
                            'time_slot': time_slot,
                            'usage_rate': usage_rate,
                            'weighted_usage_rate': weighted_usage_rate,
                            'sample_size': total_records
                        })
                    except Exception as e:
                        print(f"⚠️  Error calculating time factor for {day_type} slot {time_slot}: {e}")
                        continue

        if not raw_usage_data:
            print("⚠️  No time factor data available - using defaults")
            self._create_default_time_factor_matrix()
            return self.time_factor_matrix

        usage_df = pd.DataFrame(raw_usage_data)
        
        # 為每種日期類型計算模糊集合
        for day_type in ['weekday', 'weekend']:
            day_data = usage_df[usage_df['day_type'] == day_type]
            
            if len(day_data) > 0:
                try:
                    weighted_rates = day_data['weighted_usage_rate'].values
                    
                    min_rate = max(0.0, np.min(weighted_rates))
                    q25_rate = np.percentile(weighted_rates, 25)
                    q50_rate = np.percentile(weighted_rates, 50)
                    q75_rate = np.percentile(weighted_rates, 75)
                    max_rate = min(1.0, np.max(weighted_rates))
                    
                    # 確保順序正確
                    if not (min_rate <= q25_rate <= q50_rate <= q75_rate <= max_rate):
                        print(f"⚠️  Invalid percentiles for {day_type} time factor, using defaults")
                        min_rate, q25_rate, q50_rate, q75_rate, max_rate = 0.0, 0.25, 0.5, 0.75, 1.0
                    
                    usage_fuzzy_sets = {
                        'low': (min_rate, q25_rate, q50_rate),
                        'medium': (q25_rate, q50_rate, q75_rate),
                        'high': (q50_rate, q75_rate, max_rate)
                    }
                    
                    # 對每個時段計算模糊使用率
                    for _, row in day_data.iterrows():
                        time_slot = row['time_slot']
                        usage_rate = row['usage_rate']
                        weighted_usage = row['weighted_usage_rate']
                        
                        fuzzy_memberships = {}
                        for level, (a, b, c) in usage_fuzzy_sets.items():
                            membership = self.triangular_membership(weighted_usage, a, b, c)
                            fuzzy_memberships[f'usage_{level}'] = membership
                        
                        self.time_factor_matrix[(day_type, time_slot)] = {
                            'usage_rate': usage_rate,
                            'weighted_usage_rate': weighted_usage,
                            'usage_low': fuzzy_memberships['usage_low'],
                            'usage_medium': fuzzy_memberships['usage_medium'],
                            'usage_high': fuzzy_memberships['usage_high'],
                            'sample_size': row['sample_size']
                        }
                except Exception as e:
                    print(f"⚠️  Error processing {day_type} time factor: {e}")
                    continue

        # 填補缺失的時段
        for day_type in ['weekday', 'weekend']:
            for time_slot in range(self.time_slots):
                if (day_type, time_slot) not in self.time_factor_matrix:
                    self.time_factor_matrix[(day_type, time_slot)] = {
                        'usage_rate': 0.0,
                        'weighted_usage_rate': 0.0,
                        'usage_low': 1.0,
                        'usage_medium': 0.0,
                        'usage_high': 0.0,
                        'sample_size': 0
                    }

        print(f"✓ Calculated time factor for {len(self.time_factor_matrix)} time slots")
        return self.time_factor_matrix

    def _create_default_time_factor_matrix(self):
        """創建默認的時間因子矩陣"""
        for day_type in ['weekday', 'weekend']:
            for time_slot in range(self.time_slots):
                # 基於時段給出合理的默認值
                hour = time_slot // 4
                
                if day_type == 'weekday':
                    if 7 <= hour <= 9 or 18 <= hour <= 22:
                        usage_rate = 0.7  # 工作日早晚高峰
                    elif 10 <= hour <= 17:
                        usage_rate = 0.3  # 工作時間
                    else:
                        usage_rate = 0.1  # 其他時間
                else:  # weekend
                    if 9 <= hour <= 23:
                        usage_rate = 0.6  # 週末白天和晚上
                    else:
                        usage_rate = 0.2  # 週末夜間
                
                self.time_factor_matrix[(day_type, time_slot)] = {
                    'usage_rate': usage_rate,
                    'weighted_usage_rate': usage_rate,
                    'usage_low': self.triangular_membership(usage_rate, 0.0, 0.25, 0.5),
                    'usage_medium': self.triangular_membership(usage_rate, 0.25, 0.5, 0.75),
                    'usage_high': self.triangular_membership(usage_rate, 0.5, 0.75, 1.0),
                    'sample_size': 0
                }

    def calculate_membership_parameters(self):
        """計算統計隸屬參數"""
        print("==== Calculating Statistical Membership Parameters ====")
        
        self.membership_parameters = {
            'usage_probability': {},
            'stability': {},
            'time_factor': {}
        }
        
        # 收集 Usage Probability 數據
        usage_prob_values = []
        for key in self.usage_probability_matrix:
            data = self.usage_probability_matrix[key]
            if 'short_prob' in data and not np.isnan(data['short_prob']):
                usage_prob_values.append(data['short_prob'])
        
        if usage_prob_values:
            usage_prob_values = np.array(usage_prob_values)
            self.membership_parameters['usage_probability'] = {
                'p0': float(np.min(usage_prob_values)),
                'p25': float(np.percentile(usage_prob_values, 25)),
                'p50': float(np.percentile(usage_prob_values, 50)),
                'p75': float(np.percentile(usage_prob_values, 75)),
                'p100': float(np.max(usage_prob_values))
            }
            print(f"Usage Probability Statistics (n={len(usage_prob_values)}): ✓")
        else:
            print("⚠️  No valid usage probability data, using defaults")
            self.membership_parameters['usage_probability'] = {
                'p0': 0.0, 'p25': 0.25, 'p50': 0.5, 'p75': 0.75, 'p100': 1.0
            }
        
        # 收集 Stability 數據
        stability_values = []
        for key in self.stability_matrix:
            data = self.stability_matrix[key]
            if 'overall_stability_raw' in data and not np.isnan(data['overall_stability_raw']):
                stability_values.append(data['overall_stability_raw'])
        
        if stability_values:
            stability_values = np.array(stability_values)
            self.membership_parameters['stability'] = {
                'p0': float(np.min(stability_values)),
                'p25': float(np.percentile(stability_values, 25)),
                'p50': float(np.percentile(stability_values, 50)),
                'p75': float(np.percentile(stability_values, 75)),
                'p100': float(np.max(stability_values))
            }
            print(f"Stability Statistics (n={len(stability_values)}): ✓")
        else:
            print("⚠️  No valid stability data, using defaults")
            self.membership_parameters['stability'] = {
                'p0': 0.0, 'p25': 0.25, 'p50': 0.5, 'p75': 0.75, 'p100': 1.0
            }
        
        # 收集 Time Factor 數據
        time_factor_values = []
        for key in self.time_factor_matrix:
            data = self.time_factor_matrix[key]
            if 'weighted_usage_rate' in data and not np.isnan(data['weighted_usage_rate']):
                time_factor_values.append(data['weighted_usage_rate'])
        
        if time_factor_values:
            time_factor_values = np.array(time_factor_values)
            self.membership_parameters['time_factor'] = {
                'p0': float(np.min(time_factor_values)),
                'p25': float(np.percentile(time_factor_values, 25)),
                'p50': float(np.percentile(time_factor_values, 50)),
                'p75': float(np.percentile(time_factor_values, 75)),
                'p100': float(np.max(time_factor_values))
            }
            print(f"Time Factor Statistics (n={len(time_factor_values)}): ✓")
        else:
            print("⚠️  No valid time factor data, using defaults")
            self.membership_parameters['time_factor'] = {
                'p0': 0.0, 'p25': 0.25, 'p50': 0.5, 'p75': 0.75, 'p100': 1.0
            }
        
        return self.membership_parameters

    def define_habit_rules(self):
        """定義使用習慣模糊規則"""
        print("==== Defining User Habit Rules ====")
        
        self.habit_rules = [
            # 高習慣分數規則：強烈建議不要關機
            ('high', 'high', 'peak', 'high', 1.0),
            ('high', 'high', 'possible', 'high', 0.9),
            ('high', 'medium', 'peak', 'high', 0.85),
            ('medium', 'high', 'peak', 'high', 0.8),
            
            # 中高習慣分數規則
            ('high', 'low', 'peak', 'medium', 0.75),
            ('high', 'medium', 'possible', 'medium', 0.7),
            ('medium', 'high', 'possible', 'medium', 0.75),
            ('medium', 'medium', 'peak', 'medium', 0.65),
            
            # 中等習慣分數規則
            ('high', 'high', 'non_use', 'medium', 0.6),
            ('low', 'high', 'peak', 'medium', 0.55),
            ('medium', 'low', 'peak', 'medium', 0.5),
            ('medium', 'medium', 'possible', 'medium', 0.55),
            
            # 低習慣分數規則：可以關機節能
            ('low', 'high', 'possible', 'low', 0.6),
            ('low', 'medium', 'peak', 'low', 0.5),
            ('medium', 'low', 'possible', 'low', 0.5),
            ('high', 'low', 'non_use', 'low', 0.45),
            
            # 很低習慣分數規則：強烈建議關機
            ('low', 'high', 'non_use', 'low', 0.8),
            ('low', 'medium', 'non_use', 'low', 0.75),
            ('medium', 'low', 'non_use', 'low', 0.7),
            ('low', 'low', 'possible', 'low', 0.6),
            ('low', 'low', 'non_use', 'low', 0.9),
        ]
        
        print(f"✓ Defined {len(self.habit_rules)} habit rules")
        return self.habit_rules

    def calculate_fuzzy_memberships(self, timestamp):
        """計算指定時間點的模糊隸屬度"""
        # 提取時間特徵
        hour = timestamp.hour
        minute = timestamp.minute
        is_weekend = timestamp.weekday() >= 5
        day_type = 'weekend' if is_weekend else 'weekday'
        time_slot = hour * 4 + minute // 15
        
        result = {
            'day_type': day_type,
            'time_slot': time_slot,
            'hour': hour,
            'minute': minute
        }
        
        # 獲取使用機率隸屬度
        if (day_type, time_slot) in self.usage_probability_matrix:
            prob_data = self.usage_probability_matrix[(day_type, time_slot)]
            short_prob = prob_data.get('short_prob', 0.3)
            
            if not np.isnan(short_prob):
                usage_params = self.membership_parameters['usage_probability']
                result['usage_prob_low'] = self.triangular_membership(
                    short_prob, usage_params['p0'], usage_params['p25'], usage_params['p50']
                )
                result['usage_prob_medium'] = self.triangular_membership(
                    short_prob, usage_params['p25'], usage_params['p50'], usage_params['p75']
                )
                result['usage_prob_high'] = self.triangular_membership(
                    short_prob, usage_params['p50'], usage_params['p75'], usage_params['p100']
                )
                result['usage_probability'] = short_prob
            else:
                result.update({
                    'usage_prob_low': 0.6, 'usage_prob_medium': 0.4, 'usage_prob_high': 0.0,
                    'usage_probability': 0.3
                })
        else:
            result.update({
                'usage_prob_low': 0.7, 'usage_prob_medium': 0.3, 'usage_prob_high': 0.0,
                'usage_probability': 0.2
            })
        
        # 獲取穩定性隸屬度
        if (day_type, time_slot) in self.stability_matrix:
            stability_data = self.stability_matrix[(day_type, time_slot)]
            overall_stability = stability_data.get('overall_stability_raw', 0.5)
            
            if not np.isnan(overall_stability):
                stability_params = self.membership_parameters['stability']
                result['stability_low'] = self.triangular_membership(
                    overall_stability, stability_params['p0'], stability_params['p25'], stability_params['p50']
                )
                result['stability_medium'] = self.triangular_membership(
                    overall_stability, stability_params['p25'], stability_params['p50'], stability_params['p75']
                )
                result['stability_high'] = self.triangular_membership(
                    overall_stability, stability_params['p50'], stability_params['p75'], stability_params['p100']
                )
                result['stability'] = overall_stability
            else:
                result.update({
                    'stability_low': 0.6, 'stability_medium': 0.4, 'stability_high': 0.0,
                    'stability': 0.4
                })
        else:
            result.update({
                'stability_low': 0.8, 'stability_medium': 0.2, 'stability_high': 0.0,
                'stability': 0.2
            })
        
        # 獲取時間因子隸屬度
        if (day_type, time_slot) in self.time_factor_matrix:
            time_data = self.time_factor_matrix[(day_type, time_slot)]
            usage_rate = time_data.get('weighted_usage_rate', 0.2)
            
            if not np.isnan(usage_rate):
                time_params = self.membership_parameters['time_factor']
                result['time_non_use'] = self.triangular_membership(
                    usage_rate, time_params['p0'], time_params['p25'], time_params['p50']
                )
                result['time_possible'] = self.triangular_membership(
                    usage_rate, time_params['p25'], time_params['p50'], time_params['p75']
                )
                result['time_peak'] = self.triangular_membership(
                    usage_rate, time_params['p50'], time_params['p75'], time_params['p100']
                )
                result['time_factor'] = usage_rate
            else:
                result.update({
                    'time_non_use': 0.7, 'time_possible': 0.3, 'time_peak': 0.0,
                    'time_factor': 0.2
                })
        else:
            result.update({
                'time_non_use': 0.6, 'time_possible': 0.4, 'time_peak': 0.0,
                'time_factor': 0.2
            })
        
        return result

    def calculate_habit_score(self, timestamp):
        """計算改進版習慣分數"""
        try:
            memberships = self.calculate_fuzzy_memberships(timestamp)
            
            # 使用改進的規則激活計算
            low_activation = 0.0
            medium_activation = 0.0
            high_activation = 0.0
            valid_rules = 0
            
            for rule in self.habit_rules:
                usage_level, stability_level, time_level, output_level, weight = rule
                
                # 獲取隸屬度
                usage_membership = memberships.get(f'usage_prob_{usage_level}', 0.0)
                stability_membership = memberships.get(f'stability_{stability_level}', 0.0)
                time_membership = memberships.get(f'time_{time_level}', 0.0)
                
                # 檢查是否有有效的隸屬度
                if any(np.isnan([usage_membership, stability_membership, time_membership])):
                    continue
                
                # 使用加權幾何平均而非最小值
                if all(m >= 0 for m in [usage_membership, stability_membership, time_membership]):
                    # 避免零值的幾何平均
                    memberships_adj = [max(0.01, m) for m in [usage_membership, stability_membership, time_membership]]
                    activation = (memberships_adj[0] * memberships_adj[1] * memberships_adj[2]) ** (1/3) * weight
                    
                    # 累積激活
                    if output_level == 'low':
                        low_activation += activation
                    elif output_level == 'medium':
                        medium_activation += activation
                    elif output_level == 'high':
                        high_activation += activation
                    
                    valid_rules += 1
            
            # 限制激活強度
            low_activation = min(low_activation, 1.0)
            medium_activation = min(medium_activation, 1.0) 
            high_activation = min(high_activation, 1.0)
            
            # 計算最終分數
            total_activation = low_activation + medium_activation + high_activation
            
            if total_activation > 0 and valid_rules > 0:
                # 使用加權重心法
                habit_score = (
                    low_activation * 0.2 + 
                    medium_activation * 0.5 + 
                    high_activation * 0.8
                ) / total_activation
                
                # 添加基於實際數據的微調
                time_factor = memberships.get('time_factor', 0.5)
                usage_prob = memberships.get('usage_probability', 0.5)
                stability = memberships.get('stability', 0.5)
                
                # 綜合調整（更保守的調整範圍）
                data_adjustment = (time_factor + usage_prob + stability - 1.5) * 0.05
                habit_score += data_adjustment
                
                confidence = min(1.0, total_activation * valid_rules / len(self.habit_rules))
            else:
                # 後備計算
                habit_score = 0.3
                confidence = 0.1
            
            # 確保分數在合理範圍內
            habit_score = max(0.05, min(0.95, habit_score))
            
            return {
                'habit_score': habit_score,
                'low_activation': low_activation,
                'medium_activation': medium_activation,
                'high_activation': high_activation,
                'memberships': memberships,
                'confidence': confidence,
                'valid_rules': valid_rules
            }
            
        except Exception as e:
            print(f"⚠️  Error calculating habit score: {e}")
            return {
                'habit_score': 0.3,
                'low_activation': 0.0,
                'medium_activation': 0.5,
                'high_activation': 0.0,
                'memberships': {},
                'confidence': 0.1,
                'valid_rules': 0
            }

    def plot_triangular_membership_functions(self):
        
        # 確保有 membership_parameters
        if not hasattr(self, 'membership_parameters') or not self.membership_parameters:
            print("❌ 沒有找到 membership_parameters，請先運行完整分析")
            return
        
        # 設置圖形
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Triangular Membership Functions', fontsize=16, fontweight='bold')
        
        # 定義顏色和樣式
        colors = ['red', 'orange', 'green']
        line_styles = ['-', '--', '-.']
        
        # 1. Usage Probability 隸屬函數
        ax1 = axes[0]
        x_usage = np.linspace(0, 1, 1000)
        
        if 'usage_probability' in self.membership_parameters:
            usage_params = self.membership_parameters['usage_probability']
            
            # 檢查參數有效性
            if all(key in usage_params for key in ['p0', 'p25', 'p50', 'p75', 'p100']):
                p0, p25, p50, p75, p100 = [usage_params[k] for k in ['p0', 'p25', 'p50', 'p75', 'p100']]
                
                # 計算三個三角函數
                usage_low = np.array([self.triangular_membership(x, p0, p25, p50) for x in x_usage])
                usage_medium = np.array([self.triangular_membership(x, p25, p50, p75) for x in x_usage])
                usage_high = np.array([self.triangular_membership(x, p50, p75, p100) for x in x_usage])
                
                # 繪製
                ax1.plot(x_usage, usage_low, color=colors[0], linewidth=3, 
                        label=f'Low ({p0:.3f}, {p25:.3f}, {p50:.3f})', linestyle=line_styles[0])
                ax1.plot(x_usage, usage_medium, color=colors[1], linewidth=3, 
                        label=f'Medium ({p25:.3f}, {p50:.3f}, {p75:.3f})', linestyle=line_styles[1])
                ax1.plot(x_usage, usage_high, color=colors[2], linewidth=3, 
                        label=f'High ({p50:.3f}, {p75:.3f}, {p100:.3f})', linestyle=line_styles[2])
                
                # 標記關鍵點
                key_points = [p25, p50, p75]
                for i, point in enumerate(key_points):
                    ax1.axvline(point, color='gray', linestyle=':', alpha=0.6, linewidth=1)
                    ax1.text(point, 1.05, f'P{25*(i+1)}', ha='center', va='bottom', fontsize=8)
                
                # 填充區域
                ax1.fill_between(x_usage, 0, usage_low, alpha=0.2, color=colors[0])
                ax1.fill_between(x_usage, 0, usage_medium, alpha=0.2, color=colors[1])
                ax1.fill_between(x_usage, 0, usage_high, alpha=0.2, color=colors[2])
            else:
                ax1.text(0.5, 0.5, 'Usage Probability\n unsufficent parameters', ha='center', va='center', fontsize=12)
        
        ax1.set_xlabel('Usage Probability')
        ax1.set_ylabel('Membership Degree')
        ax1.set_title('Usage Probability', fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1.1)
        
        # 2. Stability 隸屬函數
        ax2 = axes[1]
        x_stability = np.linspace(0, 1, 1000)
        
        if 'stability' in self.membership_parameters:
            stability_params = self.membership_parameters['stability']
            
            if all(key in stability_params for key in ['p0', 'p25', 'p50', 'p75', 'p100']):
                p0, p25, p50, p75, p100 = [stability_params[k] for k in ['p0', 'p25', 'p50', 'p75', 'p100']]
                
                stability_low = np.array([self.triangular_membership(x, p0, p25, p50) for x in x_stability])
                stability_medium = np.array([self.triangular_membership(x, p25, p50, p75) for x in x_stability])
                stability_high = np.array([self.triangular_membership(x, p50, p75, p100) for x in x_stability])
                
                ax2.plot(x_stability, stability_low, color=colors[0], linewidth=3, 
                        label=f'Low ({p0:.3f}, {p25:.3f}, {p50:.3f})', linestyle=line_styles[0])
                ax2.plot(x_stability, stability_medium, color=colors[1], linewidth=3, 
                        label=f'Medium ({p25:.3f}, {p50:.3f}, {p75:.3f})', linestyle=line_styles[1])
                ax2.plot(x_stability, stability_high, color=colors[2], linewidth=3, 
                        label=f'High ({p50:.3f}, {p75:.3f}, {p100:.3f})', linestyle=line_styles[2])
                
                # 標記關鍵點
                key_points = [p25, p50, p75]
                for i, point in enumerate(key_points):
                    ax2.axvline(point, color='gray', linestyle=':', alpha=0.6, linewidth=1)
                    ax2.text(point, 1.05, f'P{25*(i+1)}', ha='center', va='bottom', fontsize=8)
                
                # 填充區域
                ax2.fill_between(x_stability, 0, stability_low, alpha=0.2, color=colors[0])
                ax2.fill_between(x_stability, 0, stability_medium, alpha=0.2, color=colors[1])
                ax2.fill_between(x_stability, 0, stability_high, alpha=0.2, color=colors[2])
            else:
                ax2.text(0.5, 0.5, 'Stability\n參數不完整', ha='center', va='center', fontsize=12)
        
        ax2.set_xlabel('Stability')
        ax2.set_ylabel('Membership Degree')
        ax2.set_title('Usage Stability', fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1.1)
        
        # 3. Time Factor 隸屬函數
        ax3 = axes[2]
        x_time = np.linspace(0, 1, 1000)
        
        if 'time_factor' in self.membership_parameters:
            time_params = self.membership_parameters['time_factor']
            
            if all(key in time_params for key in ['p0', 'p25', 'p50', 'p75', 'p100']):
                p0, p25, p50, p75, p100 = [time_params[k] for k in ['p0', 'p25', 'p50', 'p75', 'p100']]
                
                time_non_use = np.array([self.triangular_membership(x, p0, p25, p50) for x in x_time])
                time_possible = np.array([self.triangular_membership(x, p25, p50, p75) for x in x_time])
                time_peak = np.array([self.triangular_membership(x, p50, p75, p100) for x in x_time])
                
                ax3.plot(x_time, time_non_use, color=colors[0], linewidth=3, 
                        label=f'Non-use ({p0:.3f}, {p25:.3f}, {p50:.3f})', linestyle=line_styles[0])
                ax3.plot(x_time, time_possible, color=colors[1], linewidth=3, 
                        label=f'Possible ({p25:.3f}, {p50:.3f}, {p75:.3f})', linestyle=line_styles[1])
                ax3.plot(x_time, time_peak, color=colors[2], linewidth=3, 
                        label=f'Peak ({p50:.3f}, {p75:.3f}, {p100:.3f})', linestyle=line_styles[2])
                
                # 標記關鍵點
                key_points = [p25, p50, p75]
                for i, point in enumerate(key_points):
                    ax3.axvline(point, color='gray', linestyle=':', alpha=0.6, linewidth=1)
                    ax3.text(point, 1.05, f'P{25*(i+1)}', ha='center', va='bottom', fontsize=8)
                
                # 填充區域
                ax3.fill_between(x_time, 0, time_non_use, alpha=0.2, color=colors[0])
                ax3.fill_between(x_time, 0, time_possible, alpha=0.2, color=colors[1])
                ax3.fill_between(x_time, 0, time_peak, alpha=0.2, color=colors[2])
            else:
                ax3.text(0.5, 0.5, 'Time Factor\n參數不完整', ha='center', va='center', fontsize=12)
        
        ax3.set_xlabel('Time Factor / Usage Rate')
        ax3.set_ylabel('Membership Degree')
        ax3.set_title('Time Factor', fontweight='bold')
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.show()
        
        # 打印詳細參數信息
        print("="*60)
        print("三角隸屬函數參數詳情")
        print("="*60)
        
        for param_type, params in self.membership_parameters.items():
            print(f"\n📊 {param_type.replace('_', ' ').title()}:")
            if all(key in params for key in ['p0', 'p25', 'p50', 'p75', 'p100']):
                p0, p25, p50, p75, p100 = [params[k] for k in ['p0', 'p25', 'p50', 'p75', 'p100']]
                print(f"  資料範圍: [{p0:.3f}, {p100:.3f}]")
                print(f"  Low 三角函數:    ({p0:.3f}, {p25:.3f}, {p50:.3f})")
                print(f"  Medium 三角函數: ({p25:.3f}, {p50:.3f}, {p75:.3f})")
                print(f"  High 三角函數:   ({p50:.3f}, {p75:.3f}, {p100:.3f})")
                
                # 計算重疊度
                overlap_low_med = max(0, p50 - p25) / (p50 - p0) if p50 > p0 else 0
                overlap_med_high = max(0, p75 - p50) / (p100 - p50) if p100 > p50 else 0
                print(f"  重疊度: Low-Medium: {overlap_low_med:.2f}, Medium-High: {overlap_med_high:.2f}")


    def test_habit_score_calculation(self, num_tests=5):
        """測試習慣分數計算功能"""
        print("==== Testing Habit Score Calculation ====")
        
        test_times = [
            datetime(2024, 1, 15, 9, 0),   # 週一早上9點
            datetime(2024, 1, 15, 14, 30), # 週一下午2:30
            datetime(2024, 1, 15, 21, 0),  # 週一晚上9點
            datetime(2024, 1, 13, 10, 15), # 週六上午10:15
            datetime(2024, 1, 13, 20, 45), # 週六晚上8:45
        ]
        
        test_results = []
        
        for i, test_time in enumerate(test_times[:num_tests]):
            try:
                result = self.calculate_habit_score(test_time)
                
                day_type = "Weekend" if test_time.weekday() >= 5 else "Weekday"
                print(f"\nTest {i+1}: {test_time.strftime('%Y-%m-%d %H:%M')} ({day_type})")
                print(f"  Habit Score: {result['habit_score']:.3f}")
                print(f"  Confidence: {result['confidence']:.3f}")
                print(f"  Valid Rules: {result['valid_rules']}")
                print(f"  Activations - Low: {result['low_activation']:.3f}, "
                      f"Medium: {result['medium_activation']:.3f}, "
                      f"High: {result['high_activation']:.3f}")
                
                test_results.append({
                    'time': test_time,
                    'score': result['habit_score'],
                    'confidence': result['confidence']
                })
                
            except Exception as e:
                print(f"⚠️  Error in test {i+1}: {e}")
                test_results.append({
                    'time': test_time,
                    'score': 0.3,
                    'confidence': 0.0
                })
        
        return test_results

    def comprehensive_evaluation(self):
        """完整的系統評估"""
        print("\n" + "="*60)
        print("COMPREHENSIVE SYSTEM EVALUATION")
        print("="*60)
        
        # 1. 數據質量評估
        print(f"\n1. Data Quality Assessment:")
        print(f"   Quality Score: {self.data_quality_report.get('quality_score', 0):.2f}")
        print(f"   Issues Count: {len(self.data_quality_report.get('issues', []))}")
        
        # 2. 矩陣完整性檢查
        print(f"\n2. Matrix Completeness:")
        matrices = {
            'Usage Probability': self.usage_probability_matrix,
            'Stability': self.stability_matrix,
            'Time Factor': self.time_factor_matrix
        }
        
        for name, matrix in matrices.items():
            total_slots = 192  # 2 day types * 96 time slots
            coverage = len(matrix) / total_slots * 100
            nan_count = 0
            
            for v in matrix.values():
                if isinstance(v, dict):
                    for val in v.values():
                        if isinstance(val, (int, float)) and np.isnan(val):
                            nan_count += 1
                            break
            
            print(f"   {name}: {coverage:.1f}% coverage, {nan_count} entries with NaN")
        
        # 3. 習慣分數測試
        print(f"\n3. Habit Score Tests:")
        test_scenarios = [
            (datetime(2024, 1, 15, 9, 0), (0.4, 0.8), '工作日早上'),
            (datetime(2024, 1, 15, 14, 0), (0.3, 0.7), '工作日下午'),
            (datetime(2024, 1, 15, 22, 0), (0.1, 0.5), '工作日深夜'),
            (datetime(2024, 1, 13, 10, 0), (0.3, 0.7), '週末上午'),
            (datetime(2024, 1, 13, 2, 0), (0.0, 0.3), '週末凌晨'),
        ]
        
        passed_tests = 0
        for test_time, expected_range, desc in test_scenarios:
            try:
                result = self.calculate_habit_score(test_time)
                score = result['habit_score']
                confidence = result['confidence']
                
                is_reasonable = expected_range[0] <= score <= expected_range[1]
                has_confidence = confidence > 0.1
                
                if is_reasonable and has_confidence:
                    passed_tests += 1
                    status = '✓'
                else:
                    status = '❌'
                
                print(f"   {status} {desc}: {score:.3f} (期望: {expected_range})")
            except Exception as e:
                print(f"   ❌ {desc}: Error - {e}")
        
        # 4. 系統穩定性測試
        print(f"\n4. System Stability:")
        try:
            base_time = datetime(2024, 1, 15, 14, 0)
            base_result = self.calculate_habit_score(base_time)
            base_score = base_result['habit_score']
            
            sensitivity_diffs = []
            for minutes in [-30, -15, 15, 30]:
                test_time = base_time + timedelta(minutes=minutes)
                result = self.calculate_habit_score(test_time)
                diff = abs(result['habit_score'] - base_score)
                sensitivity_diffs.append(diff)
            
            avg_sensitivity = np.mean(sensitivity_diffs)
            print(f"   Time sensitivity: {avg_sensitivity:.3f}")
            print(f"   Stability: {'Good' if avg_sensitivity < 0.15 else 'Needs improvement'}")
        except Exception as e:
            print(f"   ❌ Stability test failed: {e}")
            avg_sensitivity = 1.0
        
        # 5. 最終評分
        print(f"\n=== FINAL ASSESSMENT ===")
        quality_score = self.data_quality_report.get('quality_score', 0)
        test_pass_rate = passed_tests / len(test_scenarios)
        stability_score = max(0, 1 - avg_sensitivity / 0.2)
        
        overall_score = (quality_score + test_pass_rate + stability_score) / 3
        
        print(f"Data Quality: {quality_score:.2f}")
        print(f"Test Pass Rate: {test_pass_rate:.2f}")
        print(f"Stability Score: {stability_score:.2f}")
        print(f"Overall System Quality: {overall_score:.2f}")
        
        if overall_score >= 0.8:
            print("🎉 System Quality: Excellent")
        elif overall_score >= 0.6:
            print("✅ System Quality: Good") 
        elif overall_score >= 0.4:
            print("⚠️  System Quality: Acceptable")
        else:
            print("❌ System Quality: Needs Improvement")

    def run_complete_analysis(self, file_path):
        """運行完整分析"""
        print("="*80)
        print("IMPROVED USER HABIT SCORE MODULE - COMPLETE ANALYSIS")
        print("="*80)

        # 1. 載入數據
        df = self.load_data(file_path)
        if df is None:
            print('❌ Cannot load data')
            return None
        
        # 2. 分析狀態轉換
        print("\n" + "-"*50)
        self.state_transitions(df)
        
        # 3. 計算三個核心指標
        print("\n" + "-"*50)
        self.calculate_usage_probability()
        
        print("\n" + "-"*50)
        self.calculate_usage_stability(df)
        
        print("\n" + "-"*50)
        self.calculate_time_factor(df)

        # 4. 計算隸屬參數
        print("\n" + "-"*50)
        self.calculate_membership_parameters()
    
        # 5. 定義規則
        print("\n" + "-"*50)
        self.define_habit_rules()

        # 6. 測試計算
        print("\n" + "-"*50)
        test_results = self.test_habit_score_calculation()

        # 7. 綜合評估
        print("\n" + "-"*50)
        self.comprehensive_evaluation()

        print("\n" + "-"*50)
        print("==== Plotting Triangular Membership Functions ====")
        self.plot_triangular_membership_functions()

        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE - System ready for production use!")
        print("="*80)

        return {
            'usage_probability': self.usage_probability_matrix,
            'stability': self.stability_matrix,
            'time_factor': self.time_factor_matrix,
            'membership_parameters': self.membership_parameters,
            'habit_rules': self.habit_rules,
            'test_results': test_results,
            'data_quality': self.data_quality_report
        }

# 使用示例
if __name__ == "__main__":
    # 初始化改進版模組
    improved_habit_module = ImprovedUserHabitScoreModule()
    
    # 檔案路徑
    file_path = "C:/Users/王俞文/Documents/glasgow/msc project/data/data_after_preprocessing.csv"
    
    # 運行完整分析
    result = improved_habit_module.run_complete_analysis(file_path)
    
    # 單獨測試習慣分數計算
    if result:
        print("\n" + "="*50)
        print("TESTING INDIVIDUAL HABIT SCORE CALCULATIONS")
        print("="*50)
        
        # 測試幾個特定時間點
        test_times = [
            datetime(2024, 6, 15, 8, 30),   # 週六早上8:30
            datetime(2024, 6, 17, 19, 45),  # 週一晚上7:45
            datetime(2024, 6, 20, 14, 15),  # 週四下午2:15
        ]
        
        for test_time in test_times:
            result = improved_habit_module.calculate_habit_score(test_time)
            day_type = "Weekend" if test_time.weekday() >= 5 else "Weekday"
            
            print(f"\n時間: {test_time.strftime('%Y-%m-%d %H:%M')} ({day_type})")
            print(f"習慣分數: {result['habit_score']:.3f}")
            print(f"信心度: {result['confidence']:.3f}")
            
            # 提供建議
            score = result['habit_score']
            if score >= 0.7:
                suggestion = "🔴 強烈建議保持開機"
            elif score >= 0.5:
                suggestion = "🟡 建議謹慎評估是否關機"
            elif score >= 0.3:
                suggestion = "🟠 可以考慮關機節能"
            else:
                suggestion = "🟢 建議關機節能"
            
            print(f"建議: {suggestion}")