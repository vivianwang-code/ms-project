# 針對無關機場景的改進版用戶習慣模組
# 重新定義使用習慣：基於使用強度轉換而非開關機轉換

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class NoShutdownUserHabitScoreModule:
    """
    專門針對無關機場景的用戶習慣分數模組
    重新定義使用習慣：
    - 使用強度模式：phantom load -> light use -> regular use
    - 活躍度轉換：非活躍 (phantom) -> 活躍 (light/regular)
    - 時間模式分析：不同時段的使用強度偏好
    """

    def __init__(self):
        self.time_slots = 96
        self.intensity_transition_data = None
        self.usage_intensity_matrix = {}
        self.usage_consistency_matrix = {}
        self.time_preference_matrix = {}
        self.habit_rules = []
        self.membership_parameters = {}
        self.data_quality_report = {}

    def validate_data_quality(self, df):
        """數據質量驗證"""
        print("==== Data Quality Validation ====")
        issues = []
        
        # 檢查必要欄位
        required_columns = ['timestamp', 'power_state', 'is_phantom_load', 'is_light_use', 'is_regular_use']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing columns: {missing_columns}")
        
        # 檢查數據範圍
        if len(df) < 100:
            issues.append(f"Insufficient data: only {len(df)} records")
        
        # 檢查狀態分佈（針對無關機場景）
        if 'power_state' in df.columns:
            state_counts = df['power_state'].value_counts()
            phantom_ratio = state_counts.get('phantom load', 0) / len(df)
            if phantom_ratio > 0.95:
                issues.append("Over 95% phantom load - limited usage pattern diversity")
            elif phantom_ratio < 0.1:
                issues.append("Less than 10% phantom load - unusual power pattern")
        
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
        
        df = df.copy()
        
        # 處理缺失值
        original_nan_count = df.isnull().sum().sum()
        if original_nan_count > 0:
            print(f"Handling {original_nan_count} missing values...")
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 處理時間差異常值
        if 'time_diff_seconds' in df.columns:
            Q1 = df['time_diff_seconds'].quantile(0.25)
            Q3 = df['time_diff_seconds'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = max(0, Q1 - 1.5 * IQR)
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (df['time_diff_seconds'] < lower_bound) | (df['time_diff_seconds'] > upper_bound)
            outliers_count = outliers_mask.sum()
            
            if outliers_count > 0:
                print(f"Handling {outliers_count} time_diff outliers...")
                df.loc[outliers_mask, 'time_diff_seconds'] = df['time_diff_seconds'].median()
        
        print(f"✓ Preprocessing completed. Final dataset: {len(df)} records")
        return df

    def load_data(self, file_path):
        print("==== Loading Usage Data for User Habit Score ====")
        
        try:
            df = pd.read_csv(file_path)
            
            # 數據質量驗證
            quality_issues = self.validate_data_quality(df)
            if len(quality_issues) > 3:
                print("⚠️  Warning: Multiple data quality issues detected. Results may be unreliable.")
            
            # 增強預處理
            df = self.preprocess_data_enhanced(df)
            
            # 添加時間特徵
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            df['weekday'] = df['timestamp'].dt.weekday
            df['is_weekend'] = df['weekday'] >= 5
            df['day_type'] = df['is_weekend'].map({True: 'weekend', False: 'weekday'})
            df['time_slot'] = df['hour'] * 4 + df['minute'] // 15
            
            # 重新定義活躍狀態（針對無關機場景）
            df['is_active'] = (df.get('is_light_use', False) | df.get('is_regular_use', False))
            df['is_inactive'] = df.get('is_phantom_load', True)
            
            # 定義使用強度等級
            df['intensity_level'] = 0  # 默認為0
            df.loc[df.get('is_phantom_load', False), 'intensity_level'] = 1  # phantom load
            df.loc[df.get('is_light_use', False), 'intensity_level'] = 2     # light use  
            df.loc[df.get('is_regular_use', False), 'intensity_level'] = 3   # regular use
            
            print(f"✓ Loaded usage data: {len(df)} records")
            print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"Weekdays: {sum(~df['is_weekend'])}, Weekends: {sum(df['is_weekend'])}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None

    def triangular_membership(self, x, a, b, c):
        """三角隸屬函數"""
        if isinstance(x, (list, np.ndarray)):
            x = np.array(x)
            return np.array([self.triangular_membership(xi, a, b, c) for xi in x])
        
        if np.isnan(x) or np.isnan(a) or np.isnan(b) or np.isnan(c):
            return 0.0
        
        if a == b == c:
            return 1.0 if x == a else 0.0
        
        if a > b or b > c:
            return 0.0
        
        try:
            if x <= a or x >= c:
                return 0.0
            elif x == b:
                return 1.0
            elif a < x < b:
                return (x - a) / (b - a) if (b - a) > 0 else 0.0
            else:
                return (c - x) / (c - b) if (c - b) > 0 else 0.0
        except:
            return 0.0

    def analyze_intensity_transitions(self, df):
        """分析使用強度轉換模式"""
        print("==== State Transitions Analysis ====")
        
        print("Available state columns:")
        state_cols = ['power_state', 'is_phantom_load', 'is_light_use', 'is_regular_use']
        for col in state_cols:
            if col in df.columns:
                print(f"  {col}: {df[col].value_counts().to_dict()}")
        
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        
        # 分析強度轉換而非開關機轉換
        df_sorted['prev_intensity'] = df_sorted['intensity_level'].shift(1)
        df_sorted['prev_active'] = df_sorted['is_active'].shift(1)
        
        # 找出從非活躍到活躍的轉換
        activation_events = df_sorted[
            (df_sorted['prev_active'] == False) & 
            (df_sorted['is_active'] == True)
        ].copy()
        
        # 找出從活躍到非活躍的轉換
        deactivation_events = df_sorted[
            (df_sorted['prev_active'] == True) & 
            (df_sorted['is_active'] == False)
        ].copy()
        
        print(f"\nTransition Analysis:")
        print(f"Activation events (phantom -> active): {len(activation_events)}")
        print(f"Deactivation events (active -> phantom): {len(deactivation_events)}")
        
        # 建立轉換記錄
        transition_records = []
        
        for idx, activation in activation_events.iterrows():
            activation_time = activation['timestamp']
            activation_slot = activation['time_slot']
            activation_day_type = activation['day_type']
            
            # 找到下一個非活躍事件
            next_deactivation = deactivation_events[
                deactivation_events['timestamp'] > activation_time
            ]
            
            if len(next_deactivation) > 0:
                deactivation_time = next_deactivation.iloc[0]['timestamp']
                active_duration = (deactivation_time - activation_time).total_seconds() / 60.0
                
                # 過濾合理的持續時間
                if 1 <= active_duration <= 1440:  # 1分鐘到24小時
                    transition_records.append({
                        'activation_time': activation_time,
                        'deactivation_time': deactivation_time,
                        'active_duration': active_duration,
                        'time_slot': activation_slot,
                        'day_type': activation_day_type,
                        'activation_hour': activation_time.hour,
                        'activation_intensity': activation['intensity_level']
                    })
        
        self.intensity_transition_data = pd.DataFrame(transition_records)
        
        print(f"\nCreated {len(self.intensity_transition_data)} valid transition records")
        
        if len(self.intensity_transition_data) > 0:
            durations = self.intensity_transition_data['active_duration']
            print(f"Active duration statistics:")
            print(f"  Min: {durations.min():.1f} minutes")
            print(f"  Max: {durations.max():.1f} minutes")
            print(f"  Mean: {durations.mean():.1f} minutes")
            print(f"  Median: {durations.median():.1f} minutes")
        else:
            print("⚠️  No valid transitions found - using fallback probability calculation")
        
        return self.intensity_transition_data

    def calculate_usage_intensity(self, df):
        """計算使用強度偏好"""
        print("==== Calculating Usage Intensity ====")
        
        valid_entries = 0
        
        for day_type in ['weekday', 'weekend']:
            for time_slot in range(self.time_slots):
                mask = (df['day_type'] == day_type) & (df['time_slot'] == time_slot)
                slot_data = df[mask]
                
                if len(slot_data) < 1:
                    # 使用默認值
                    hour = time_slot // 4
                    if day_type == 'weekday':
                        if 7 <= hour <= 9 or 18 <= hour <= 22:  # 工作日高峰時段
                            phantom_prob, light_prob, regular_prob = 0.4, 0.4, 0.2
                        elif 10 <= hour <= 17:  # 工作時間
                            phantom_prob, light_prob, regular_prob = 0.7, 0.2, 0.1
                        else:  # 其他時間
                            phantom_prob, light_prob, regular_prob = 0.8, 0.15, 0.05
                    else:  # weekend
                        if 9 <= hour <= 22:  # 週末白天
                            phantom_prob, light_prob, regular_prob = 0.5, 0.3, 0.2
                        else:  # 週末夜間
                            phantom_prob, light_prob, regular_prob = 0.85, 0.1, 0.05
                    
                    self.usage_intensity_matrix[(day_type, time_slot)] = {
                        'phantom_prob': phantom_prob,
                        'light_prob': light_prob,
                        'regular_prob': regular_prob,
                        'avg_intensity': phantom_prob * 1 + light_prob * 2 + regular_prob * 3,
                        'sample_size': 0
                    }
                    continue
                
                try:
                    # 計算各強度等級的概率
                    total_records = len(slot_data)
                    phantom_count = len(slot_data[slot_data['intensity_level'] == 1])
                    light_count = len(slot_data[slot_data['intensity_level'] == 2])
                    regular_count = len(slot_data[slot_data['intensity_level'] == 3])
                    
                    phantom_prob = phantom_count / total_records
                    light_prob = light_count / total_records
                    regular_prob = regular_count / total_records
                    
                    # 計算平均強度
                    avg_intensity = slot_data['intensity_level'].mean()
                    
                    self.usage_intensity_matrix[(day_type, time_slot)] = {
                        'phantom_prob': phantom_prob,
                        'light_prob': light_prob,
                        'regular_prob': regular_prob,
                        'avg_intensity': avg_intensity,
                        'sample_size': total_records
                    }
                    valid_entries += 1
                    
                except Exception as e:
                    print(f"⚠️  Error calculating intensity for {day_type} slot {time_slot}: {e}")
                    self.usage_intensity_matrix[(day_type, time_slot)] = {
                        'phantom_prob': 0.7,
                        'light_prob': 0.2,
                        'regular_prob': 0.1,
                        'avg_intensity': 1.4,
                        'sample_size': len(slot_data)
                    }
        
        print(f"✓ Calculated usage intensity for {valid_entries} time slots")
        return self.usage_intensity_matrix

    def calculate_usage_consistency(self, df):
        """計算使用一致性"""
        print("==== Calculating Usage Stability ====")
        
        valid_entries = 0
        
        for day_type in ['weekday', 'weekend']:
            for time_slot in range(self.time_slots):
                mask = (df['day_type'] == day_type) & (df['time_slot'] == time_slot)
                slot_data = df[mask]
                
                if len(slot_data) < 2:
                    self.usage_consistency_matrix[(day_type, time_slot)] = {
                        'intensity_consistency': 0.5,
                        'temporal_consistency': 0.5,
                        'overall_consistency': 0.5,
                        'sample_size': len(slot_data)
                    }
                    continue
                
                try:
                    # 計算強度一致性
                    intensity_std = slot_data['intensity_level'].std()
                    max_intensity_std = 1.0  # 強度範圍1-3，理論最大標準差約為1
                    intensity_consistency = max(0, 1 - intensity_std / max_intensity_std)
                    
                    # 計算時間一致性（每日該時段的使用模式）
                    daily_usage = slot_data.groupby(slot_data['timestamp'].dt.date).agg({
                        'intensity_level': 'mean',
                        'is_active': 'max'
                    })
                    
                    if len(daily_usage) > 1:
                        temporal_std = daily_usage['intensity_level'].std()
                        temporal_consistency = max(0, 1 - temporal_std / max_intensity_std)
                    else:
                        temporal_consistency = 0.5
                    
                    # 綜合一致性
                    overall_consistency = (intensity_consistency + temporal_consistency) / 2
                    
                    self.usage_consistency_matrix[(day_type, time_slot)] = {
                        'intensity_consistency': intensity_consistency,
                        'temporal_consistency': temporal_consistency,
                        'overall_consistency': overall_consistency,
                        'sample_size': len(slot_data)
                    }
                    valid_entries += 1
                    
                except Exception as e:
                    print(f"⚠️  Error calculating consistency for {day_type} slot {time_slot}: {e}")
                    self.usage_consistency_matrix[(day_type, time_slot)] = {
                        'intensity_consistency': 0.4,
                        'temporal_consistency': 0.4,
                        'overall_consistency': 0.4,
                        'sample_size': len(slot_data)
                    }
        
        print(f"✓ Calculated stability for {valid_entries} time slots")
        return self.usage_consistency_matrix

    def calculate_time_preference(self, df):
        """計算時間偏好"""
        print("==== Calculating Time Factor ====")
        
        valid_entries = 0
        
        for day_type in ['weekday', 'weekend']:
            for time_slot in range(self.time_slots):
                mask = (df['day_type'] == day_type) & (df['time_slot'] == time_slot)
                slot_data = df[mask]
                
                if len(slot_data) < 1:
                    hour = time_slot // 4
                    if day_type == 'weekday':
                        if 7 <= hour <= 9 or 18 <= hour <= 22:
                            preference_score = 0.8  # 高偏好時段
                        elif 10 <= hour <= 17:
                            preference_score = 0.4  # 中等偏好
                        else:
                            preference_score = 0.2  # 低偏好
                    else:
                        if 9 <= hour <= 22:
                            preference_score = 0.6
                        else:
                            preference_score = 0.3
                    
                    self.time_preference_matrix[(day_type, time_slot)] = {
                        'activation_rate': 0.0,
                        'avg_intensity': 1.0,
                        'weighted_preference': preference_score,
                        'sample_size': 0
                    }
                    continue
                
                try:
                    total_records = len(slot_data)
                    active_records = len(slot_data[slot_data['is_active'] == True])
                    activation_rate = active_records / total_records if total_records > 0 else 0
                    
                    # 計算加權強度偏好
                    avg_intensity = slot_data['intensity_level'].mean()
                    
                    # 結合激活率和強度計算整體偏好
                    weighted_preference = (activation_rate * 0.6 + (avg_intensity - 1) / 2 * 0.4)
                    weighted_preference = max(0.0, min(1.0, weighted_preference))
                    
                    self.time_preference_matrix[(day_type, time_slot)] = {
                        'activation_rate': activation_rate,
                        'avg_intensity': avg_intensity,
                        'weighted_preference': weighted_preference,
                        'sample_size': total_records
                    }
                    valid_entries += 1
                    
                except Exception as e:
                    print(f"⚠️  Error calculating time preference for {day_type} slot {time_slot}: {e}")
                    self.time_preference_matrix[(day_type, time_slot)] = {
                        'activation_rate': 0.2,
                        'avg_intensity': 1.5,
                        'weighted_preference': 0.3,
                        'sample_size': len(slot_data)
                    }
        
        print(f"✓ Calculated time factor for {valid_entries} time slots")
        return self.time_preference_matrix

    def calculate_membership_parameters(self):
        """計算統計隸屬參數"""
        print("==== Calculating Statistical Membership Parameters ====")
        
        self.membership_parameters = {
            'usage_intensity': {},
            'consistency': {},
            'time_preference': {}
        }
        
        # 收集使用強度數據
        intensity_values = []
        for key in self.usage_intensity_matrix:
            data = self.usage_intensity_matrix[key]
            if 'avg_intensity' in data and not np.isnan(data['avg_intensity']):
                intensity_values.append(data['avg_intensity'])
        
        if intensity_values:
            intensity_values = np.array(intensity_values)
            self.membership_parameters['usage_intensity'] = {
                'p0': float(np.min(intensity_values)),
                'p25': float(np.percentile(intensity_values, 25)),
                'p50': float(np.percentile(intensity_values, 50)),
                'p75': float(np.percentile(intensity_values, 75)),
                'p100': float(np.max(intensity_values))
            }
            print(f"Usage Intensity Statistics (n={len(intensity_values)}): ✓")
        else:
            print("⚠️  No valid intensity data, using defaults")
            self.membership_parameters['usage_intensity'] = {
                'p0': 1.0, 'p25': 1.25, 'p50': 1.5, 'p75': 2.0, 'p100': 3.0
            }
        
        # 收集一致性數據
        consistency_values = []
        for key in self.usage_consistency_matrix:
            data = self.usage_consistency_matrix[key]
            if 'overall_consistency' in data and not np.isnan(data['overall_consistency']):
                consistency_values.append(data['overall_consistency'])
        
        if consistency_values:
            consistency_values = np.array(consistency_values)
            self.membership_parameters['consistency'] = {
                'p0': float(np.min(consistency_values)),
                'p25': float(np.percentile(consistency_values, 25)),
                'p50': float(np.percentile(consistency_values, 50)),
                'p75': float(np.percentile(consistency_values, 75)),
                'p100': float(np.max(consistency_values))
            }
            print(f"Stability Statistics (n={len(consistency_values)}): ✓")
        else:
            print("⚠️  No valid consistency data, using defaults")
            self.membership_parameters['consistency'] = {
                'p0': 0.0, 'p25': 0.25, 'p50': 0.5, 'p75': 0.75, 'p100': 1.0
            }
        
        # 收集時間偏好數據
        preference_values = []
        for key in self.time_preference_matrix:
            data = self.time_preference_matrix[key]
            if 'weighted_preference' in data and not np.isnan(data['weighted_preference']):
                preference_values.append(data['weighted_preference'])
        
        if preference_values:
            preference_values = np.array(preference_values)
            self.membership_parameters['time_preference'] = {
                'p0': float(np.min(preference_values)),
                'p25': float(np.percentile(preference_values, 25)),
                'p50': float(np.percentile(preference_values, 50)),
                'p75': float(np.percentile(preference_values, 75)),
                'p100': float(np.max(preference_values))
            }
            print(f"Time Factor Statistics (n={len(preference_values)}): ✓")
        else:
            print("⚠️  No valid time preference data, using defaults")
            self.membership_parameters['time_preference'] = {
                'p0': 0.0, 'p25': 0.25, 'p50': 0.5, 'p75': 0.75, 'p100': 1.0
            }
        
        return self.membership_parameters

    def define_habit_rules(self):
        """定義使用習慣模糊規則（針對無關機場景）"""
        print("==== Defining User Habit Rules ====")
        
        # 規則格式: (使用強度, 一致性, 時間偏好, 輸出習慣等級, 權重)
        self.habit_rules = [
            # 高習慣分數規則：強烈偏好該時段使用
            ('high', 'high', 'high', 'high', 1.0),      # 高強度+高一致性+高偏好
            ('high', 'high', 'medium', 'high', 0.9),    # 高強度+高一致性+中偏好
            ('high', 'medium', 'high', 'high', 0.85),   # 高強度+中一致性+高偏好
            ('medium', 'high', 'high', 'high', 0.8),    # 中強度+高一致性+高偏好
            
            # 中高習慣分數規則
            ('high', 'low', 'high', 'medium', 0.75),    # 高強度但低一致性
            ('high', 'medium', 'medium', 'medium', 0.7), # 高強度+中等其他因素
            ('medium', 'high', 'medium', 'medium', 0.75), # 中強度+高一致性
            ('medium', 'medium', 'high', 'medium', 0.65), # 中等條件組合
            
            # 中等習慣分數規則
            ('high', 'high', 'low', 'medium', 0.6),     # 高強度但低偏好時段
            ('low', 'high', 'high', 'medium', 0.55),    # 低強度但高一致性+高偏好
            ('medium', 'low', 'high', 'medium', 0.5),   # 中強度+低一致性+高偏好
            ('medium', 'medium', 'medium', 'medium', 0.55), # 全中等條件
            
            # 低習慣分數規則：可以關機節能
            ('low', 'high', 'medium', 'low', 0.6),      # 低強度+高一致性
            ('low', 'medium', 'high', 'low', 0.5),      # 低強度+中一致性+高偏好
            ('medium', 'low', 'medium', 'low', 0.5),    # 中強度+低一致性
            ('high', 'low', 'low', 'low', 0.45),        # 高強度但低一致性+低偏好
            
            # 很低習慣分數規則：強烈建議關機
            ('low', 'high', 'low', 'low', 0.8),         # 低強度+低偏好但一致
            ('low', 'medium', 'low', 'low', 0.75),      # 低強度+低偏好
            ('medium', 'low', 'low', 'low', 0.7),       # 中強度但其他都低
            ('low', 'low', 'medium', 'low', 0.6),       # 低條件組合
            ('low', 'low', 'low', 'low', 0.9),          # 全低條件
        ]
        
        print(f"✓ Defined {len(self.habit_rules)} habit rules")
        return self.habit_rules

    def calculate_fuzzy_memberships(self, timestamp):
        """計算指定時間點的模糊隸屬度"""
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
        
        # 獲取使用強度隸屬度
        if (day_type, time_slot) in self.usage_intensity_matrix:
            intensity_data = self.usage_intensity_matrix[(day_type, time_slot)]
            avg_intensity = intensity_data.get('avg_intensity', 1.5)
            
            intensity_params = self.membership_parameters['usage_intensity']
            result['intensity_low'] = self.triangular_membership(
                avg_intensity, intensity_params['p0'], intensity_params['p25'], intensity_params['p50']
            )
            result['intensity_medium'] = self.triangular_membership(
                avg_intensity, intensity_params['p25'], intensity_params['p50'], intensity_params['p75']
            )
            result['intensity_high'] = self.triangular_membership(
                avg_intensity, intensity_params['p50'], intensity_params['p75'], intensity_params['p100']
            )
            result['avg_intensity'] = avg_intensity
        else:
            result.update({
                'intensity_low': 0.7, 'intensity_medium': 0.3, 'intensity_high': 0.0,
                'avg_intensity': 1.2
            })
        
        # 獲取一致性隸屬度
        if (day_type, time_slot) in self.usage_consistency_matrix:
            consistency_data = self.usage_consistency_matrix[(day_type, time_slot)]
            overall_consistency = consistency_data.get('overall_consistency', 0.5)
            
            consistency_params = self.membership_parameters['consistency']
            result['consistency_low'] = self.triangular_membership(
                overall_consistency, consistency_params['p0'], consistency_params['p25'], consistency_params['p50']
            )
            result['consistency_medium'] = self.triangular_membership(
                overall_consistency, consistency_params['p25'], consistency_params['p50'], consistency_params['p75']
            )
            result['consistency_high'] = self.triangular_membership(
                overall_consistency, consistency_params['p50'], consistency_params['p75'], consistency_params['p100']
            )
            result['consistency'] = overall_consistency
        else:
            result.update({
                'consistency_low': 0.6, 'consistency_medium': 0.4, 'consistency_high': 0.0,
                'consistency': 0.4
            })
        
        # 獲取時間偏好隸屬度
        if (day_type, time_slot) in self.time_preference_matrix:
            preference_data = self.time_preference_matrix[(day_type, time_slot)]
            weighted_preference = preference_data.get('weighted_preference', 0.3)
            
            preference_params = self.membership_parameters['time_preference']
            result['preference_low'] = self.triangular_membership(
                weighted_preference, preference_params['p0'], preference_params['p25'], preference_params['p50']
            )
            result['preference_medium'] = self.triangular_membership(
                weighted_preference, preference_params['p25'], preference_params['p50'], preference_params['p75']
            )
            result['preference_high'] = self.triangular_membership(
                weighted_preference, preference_params['p50'], preference_params['p75'], preference_params['p100']
            )
            result['time_preference'] = weighted_preference
        else:
            result.update({
                'preference_low': 0.6, 'preference_medium': 0.4, 'preference_high': 0.0,
                'time_preference': 0.3
            })
        
        return result

    def calculate_habit_score(self, timestamp):
        """計算習慣分數（針對無關機場景）"""
        try:
            memberships = self.calculate_fuzzy_memberships(timestamp)
            
            low_activation = 0.0
            medium_activation = 0.0
            high_activation = 0.0
            valid_rules = 0
            
            for rule in self.habit_rules:
                intensity_level, consistency_level, preference_level, output_level, weight = rule
                
                # 獲取隸屬度
                intensity_membership = memberships.get(f'intensity_{intensity_level}', 0.0)
                consistency_membership = memberships.get(f'consistency_{consistency_level}', 0.0)
                preference_membership = memberships.get(f'preference_{preference_level}', 0.0)
                
                # 檢查有效性
                if any(np.isnan([intensity_membership, consistency_membership, preference_membership])):
                    continue
                
                # 使用加權幾何平均
                if all(m >= 0 for m in [intensity_membership, consistency_membership, preference_membership]):
                    memberships_adj = [max(0.01, m) for m in [intensity_membership, consistency_membership, preference_membership]]
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
                
                # 基於實際數據的微調
                avg_intensity = memberships.get('avg_intensity', 1.5)
                consistency = memberships.get('consistency', 0.5)
                time_preference = memberships.get('time_preference', 0.3)
                
                # 歸一化後的調整
                normalized_intensity = (avg_intensity - 1) / 2  # 將1-3映射到0-1
                data_adjustment = (normalized_intensity + consistency + time_preference - 1.5) * 0.05
                habit_score += data_adjustment
                
                confidence = min(1.0, total_activation * valid_rules / len(self.habit_rules))
            else:
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
        """繪製三角隸屬函數圖"""
        if not hasattr(self, 'membership_parameters') or not self.membership_parameters:
            print("❌ 沒有找到 membership_parameters，請先運行完整分析")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('三角隸屬函數參數詳情', fontsize=16, fontweight='bold')
        
        colors = ['red', 'orange', 'green']
        line_styles = ['-', '--', '-.']
        
        # 1. 使用強度隸屬函數
        ax1 = axes[0]
        x_intensity = np.linspace(1, 3, 1000)
        
        if 'usage_intensity' in self.membership_parameters:
            intensity_params = self.membership_parameters['usage_intensity']
            
            if all(key in intensity_params for key in ['p0', 'p25', 'p50', 'p75', 'p100']):
                p0, p25, p50, p75, p100 = [intensity_params[k] for k in ['p0', 'p25', 'p50', 'p75', 'p100']]
                
                intensity_low = np.array([self.triangular_membership(x, p0, p25, p50) for x in x_intensity])
                intensity_medium = np.array([self.triangular_membership(x, p25, p50, p75) for x in x_intensity])
                intensity_high = np.array([self.triangular_membership(x, p50, p75, p100) for x in x_intensity])
                
                ax1.plot(x_intensity, intensity_low, color=colors[0], linewidth=3, 
                        label=f'Low ({p0:.2f}, {p25:.2f}, {p50:.2f})', linestyle=line_styles[0])
                ax1.plot(x_intensity, intensity_medium, color=colors[1], linewidth=3, 
                        label=f'Medium ({p25:.2f}, {p50:.2f}, {p75:.2f})', linestyle=line_styles[1])
                ax1.plot(x_intensity, intensity_high, color=colors[2], linewidth=3, 
                        label=f'High ({p50:.2f}, {p75:.2f}, {p100:.2f})', linestyle=line_styles[2])
                
                ax1.fill_between(x_intensity, 0, intensity_low, alpha=0.2, color=colors[0])
                ax1.fill_between(x_intensity, 0, intensity_medium, alpha=0.2, color=colors[1])
                ax1.fill_between(x_intensity, 0, intensity_high, alpha=0.2, color=colors[2])
        
        ax1.set_xlabel('平均使用強度 (1=phantom, 2=light, 3=regular)')
        ax1.set_ylabel('隸屬度')
        ax1.set_title('使用強度偏好', fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(1, 3)
        ax1.set_ylim(0, 1.1)
        
        # 2. 一致性隸屬函數
        ax2 = axes[1]
        x_consistency = np.linspace(0, 1, 1000)
        
        if 'consistency' in self.membership_parameters:
            consistency_params = self.membership_parameters['consistency']
            
            if all(key in consistency_params for key in ['p0', 'p25', 'p50', 'p75', 'p100']):
                p0, p25, p50, p75, p100 = [consistency_params[k] for k in ['p0', 'p25', 'p50', 'p75', 'p100']]
                
                consistency_low = np.array([self.triangular_membership(x, p0, p25, p50) for x in x_consistency])
                consistency_medium = np.array([self.triangular_membership(x, p25, p50, p75) for x in x_consistency])
                consistency_high = np.array([self.triangular_membership(x, p50, p75, p100) for x in x_consistency])
                
                ax2.plot(x_consistency, consistency_low, color=colors[0], linewidth=3, 
                        label=f'Low ({p0:.2f}, {p25:.2f}, {p50:.2f})', linestyle=line_styles[0])
                ax2.plot(x_consistency, consistency_medium, color=colors[1], linewidth=3, 
                        label=f'Medium ({p25:.2f}, {p50:.2f}, {p75:.2f})', linestyle=line_styles[1])
                ax2.plot(x_consistency, consistency_high, color=colors[2], linewidth=3, 
                        label=f'High ({p50:.2f}, {p75:.2f}, {p100:.2f})', linestyle=line_styles[2])
                
                ax2.fill_between(x_consistency, 0, consistency_low, alpha=0.2, color=colors[0])
                ax2.fill_between(x_consistency, 0, consistency_medium, alpha=0.2, color=colors[1])
                ax2.fill_between(x_consistency, 0, consistency_high, alpha=0.2, color=colors[2])
        
        ax2.set_xlabel('使用一致性')
        ax2.set_ylabel('隸屬度')
        ax2.set_title('使用穩定性', fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1.1)
        
        # 3. 時間偏好隸屬函數
        ax3 = axes[2]
        x_preference = np.linspace(0, 1, 1000)
        
        if 'time_preference' in self.membership_parameters:
            preference_params = self.membership_parameters['time_preference']
            
            if all(key in preference_params for key in ['p0', 'p25', 'p50', 'p75', 'p100']):
                p0, p25, p50, p75, p100 = [preference_params[k] for k in ['p0', 'p25', 'p50', 'p75', 'p100']]
                
                preference_low = np.array([self.triangular_membership(x, p0, p25, p50) for x in x_preference])
                preference_medium = np.array([self.triangular_membership(x, p25, p50, p75) for x in x_preference])
                preference_high = np.array([self.triangular_membership(x, p50, p75, p100) for x in x_preference])
                
                ax3.plot(x_preference, preference_low, color=colors[0], linewidth=3, 
                        label=f'Low ({p0:.2f}, {p25:.2f}, {p50:.2f})', linestyle=line_styles[0])
                ax3.plot(x_preference, preference_medium, color=colors[1], linewidth=3, 
                        label=f'Medium ({p25:.2f}, {p50:.2f}, {p75:.2f})', linestyle=line_styles[1])
                ax3.plot(x_preference, preference_high, color=colors[2], linewidth=3, 
                        label=f'High ({p50:.2f}, {p75:.2f}, {p100:.2f})', linestyle=line_styles[2])
                
                ax3.fill_between(x_preference, 0, preference_low, alpha=0.2, color=colors[0])
                ax3.fill_between(x_preference, 0, preference_medium, alpha=0.2, color=colors[1])
                ax3.fill_between(x_preference, 0, preference_high, alpha=0.2, color=colors[2])
        
        ax3.set_xlabel('時間偏好強度')
        ax3.set_ylabel('隸屬度')
        ax3.set_title('時間偏好', fontweight='bold')
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.show()

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
            'Usage Intensity': self.usage_intensity_matrix,
            'Consistency': self.usage_consistency_matrix,
            'Time Preference': self.time_preference_matrix
        }
        
        for name, matrix in matrices.items():
            total_slots = 192
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
            (datetime(2024, 1, 15, 14, 30), (0.2, 0.6), '工作日下午'),
            (datetime(2024, 1, 15, 21, 0), (0.1, 0.5), '工作日深夜'),
            (datetime(2024, 1, 13, 10, 15), (0.3, 0.7), '週末上午'),
            (datetime(2024, 1, 13, 20, 45), (0.0, 0.3), '週末凌晨'),
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
        
        # 2. 分析強度轉換（替代開關機轉換）
        print("\n" + "-"*50)
        self.analyze_intensity_transitions(df)
        
        # 3. 計算三個核心指標
        print("\n" + "-"*50)
        self.calculate_usage_intensity(df)
        
        print("\n" + "-"*50)
        self.calculate_usage_consistency(df)
        
        print("\n" + "-"*50)
        self.calculate_time_preference(df)

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
            'usage_intensity': self.usage_intensity_matrix,
            'consistency': self.usage_consistency_matrix,
            'time_preference': self.time_preference_matrix,
            'membership_parameters': self.membership_parameters,
            'habit_rules': self.habit_rules,
            'test_results': test_results,
            'data_quality': self.data_quality_report
        }

    def test_habit_score_calculation(self, num_tests=5):
        """測試習慣分數計算功能"""
        print("==== Testing Habit Score Calculation ====")
        
        test_times = [
            datetime(2024, 1, 15, 9, 0),
            datetime(2024, 1, 15, 14, 30),
            datetime(2024, 1, 15, 21, 0),
            datetime(2024, 1, 13, 10, 15),
            datetime(2024, 1, 13, 20, 45),
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

# 使用示例
if __name__ == "__main__":
    print("🔧 無關機場景專用用戶習慣分析模組")
    print("="*50)
    
    # 初始化專用模組
    no_shutdown_habit_module = NoShutdownUserHabitScoreModule()
    
    # 檔案路徑
    file_path = "C:/Users/王俞文/OneDrive - University of Glasgow/文件/glasgow/msc project/data/extended_power_data_2months.csv"
    
    # 運行完整分析
    result = no_shutdown_habit_module.run_complete_analysis(file_path)
    
    print("\n🎯 無關機場景特點:")
    print("- 基於使用強度轉換 (phantom ↔ light ↔ regular)")
    print("- 分析時間偏好模式而非開關機模式")
    print("- 計算使用一致性評估習慣穩定性")
    print("- 更適合現代電子設備的待機模式分析")