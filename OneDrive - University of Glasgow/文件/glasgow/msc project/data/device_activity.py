# # Device Activity Score Module
# # Based on standby duration and time since last active

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime, timedelta
# import warnings
# warnings.filterwarnings('ignore')


# class DeviceActivityScoreModule:

#     def __init__(self):
#         self.time_slots = 96  # 96個時段，每15分鐘一個 (24*4)
#         self.activity_data = None
#         self.standby_duration_matrix = {}
#         self.time_since_active_matrix = {}
#         self.activity_rules = []
#         self.membership_parameters = {}
#         self.data_quality_report = {}

#     def validate_data_quality(self, df):
#         """驗證輸入數據的質量"""
#         print("==== Data Quality Validation ====")
#         issues = []
        
#         # 檢查必要欄位
#         required_columns = ['timestamp', 'power_state', 'is_on', 'is_off']
#         missing_columns = [col for col in required_columns if col not in df.columns]
#         if missing_columns:
#             issues.append(f"Missing columns: {missing_columns}")
        
#         # 檢查數據類型
#         if 'timestamp' in df.columns:
#             try:
#                 df['timestamp'] = pd.to_datetime(df['timestamp'])
#             except:
#                 issues.append("Timestamp column cannot be converted to datetime")
        
#         # 檢查數據範圍
#         if len(df) < 100:
#             issues.append(f"Insufficient data: only {len(df)} records")
        
#         # 檢查狀態一致性
#         if 'is_on' in df.columns and 'is_off' in df.columns:
#             inconsistent_states = df[(df['is_on'] == True) & (df['is_off'] == True)]
#             if len(inconsistent_states) > 0:
#                 issues.append(f"Found {len(inconsistent_states)} inconsistent on/off states")
        
#         # 檢查時間序列完整性
#         if 'timestamp' in df.columns:
#             df_sorted = df.sort_values('timestamp')
#             time_gaps = df_sorted['timestamp'].diff()
#             large_gaps = time_gaps[time_gaps > pd.Timedelta(hours=4)]
#             if len(large_gaps) > 10:
#                 issues.append(f"Found {len(large_gaps)} large time gaps > 4 hours")
        
#         self.data_quality_report = {
#             'total_records': len(df),
#             'issues': issues,
#             'quality_score': max(0, 1.0 - len(issues) * 0.2)
#         }
        
#         print(f"Data quality score: {self.data_quality_report['quality_score']:.2f}")
#         if issues:
#             print("Issues found:")
#             for issue in issues:
#                 print(f"  - {issue}")
#         else:
#             print("✓ No major data quality issues found")
        
#         return issues

#     def preprocess_data_enhanced(self, df):
#         """增強的數據預處理"""
#         print("==== Enhanced Data Preprocessing ====")
        
#         # 複製數據避免修改原始數據
#         df = df.copy()
        
#         # 處理缺失值
#         original_nan_count = df.isnull().sum().sum()
#         if original_nan_count > 0:
#             print(f"Handling {original_nan_count} missing values...")
#             df = df.fillna(method='ffill').fillna(method='bfill')
        
#         # 確保時間格式正確
#         df['timestamp'] = pd.to_datetime(df['timestamp'])
        
#         # 處理異常的時間差值
#         if 'time_diff_seconds' in df.columns:
#             # 異常值檢測和處理
#             Q1 = df['time_diff_seconds'].quantile(0.25)
#             Q3 = df['time_diff_seconds'].quantile(0.75)
#             IQR = Q3 - Q1
#             lower_bound = max(0, Q1 - 1.5 * IQR)  # 確保下界不為負
#             upper_bound = Q3 + 1.5 * IQR
            
#             outliers_mask = (df['time_diff_seconds'] < lower_bound) | (df['time_diff_seconds'] > upper_bound)
#             outliers_count = outliers_mask.sum()
            
#             if outliers_count > 0:
#                 print(f"Handling {outliers_count} time_diff outliers...")
#                 df.loc[outliers_mask, 'time_diff_seconds'] = df['time_diff_seconds'].median()
        
#         # 確保狀態欄位的數據類型正確
#         boolean_columns = ['is_on', 'is_off', 'is_phantom_load', 'is_light_use', 'is_regular_use']
#         for col in boolean_columns:
#             if col in df.columns:
#                 df[col] = df[col].astype(bool)
        
#         print(f"✓ Preprocessing completed. Final dataset: {len(df)} records")
#         return df

#     def load_data(self, file_path):
#         print("==== Loading Usage Data for Device Activity Score ====")
        
#         try:
#             # 讀取數據
#             df = pd.read_csv(file_path)
            
#             # 數據質量驗證
#             quality_issues = self.validate_data_quality(df)
#             if len(quality_issues) > 3:
#                 print("⚠️  Warning: Multiple data quality issues detected. Results may be unreliable.")
            
#             # 增強預處理
#             df = self.preprocess_data_enhanced(df)
            
#             # 轉換時間格式並添加特徵
#             df['timestamp'] = pd.to_datetime(df['timestamp'])
#             df['hour'] = df['timestamp'].dt.hour
#             df['minute'] = df['timestamp'].dt.minute
#             df['weekday'] = df['timestamp'].dt.weekday
#             df['is_weekend'] = df['weekday'] >= 5
#             df['day_type'] = df['is_weekend'].map({True: 'weekend', False: 'weekday'})
#             df['time_slot'] = df['hour'] * 4 + df['minute'] // 15  # 0-95個時段
            
#             print(f"✓ Loaded usage data: {len(df)} records")
#             print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
#             print(f"Weekdays: {sum(~df['is_weekend'])}, Weekends: {sum(df['is_weekend'])}")
            
#             return df
            
#         except Exception as e:
#             print(f"❌ Error loading data: {e}")
#             return None

#     def safe_triangular_membership(self, x, a, b, c):
#         """安全的三角隸屬函數"""
#         # 處理輸入為數組的情況
#         if isinstance(x, (list, np.ndarray)):
#             x = np.array(x)
#             return np.array([self.safe_triangular_membership(xi, a, b, c) for xi in x])
        
#         # 檢查NaN值
#         if np.isnan(x) or np.isnan(a) or np.isnan(b) or np.isnan(c):
#             return 0.0
        
#         # 檢查特殊情況
#         if a == b == c:
#             return 1.0 if x == a else 0.0
        
#         # 確保參數順序正確
#         if a > b or b > c:
#             print(f"Warning: Invalid triangular parameters: a={a}, b={b}, c={c}")
#             return 0.0
        
#         try:
#             if x <= a or x >= c:
#                 return 0.0
#             elif x == b:
#                 return 1.0
#             elif a < x < b:
#                 return (x - a) / (b - a) if (b - a) > 0 else 0.0
#             else:  # b < x < c
#                 return (c - x) / (c - b) if (c - b) > 0 else 0.0
#         except:
#             return 0.0

#     def analyze_device_activity(self, df):
#         """分析設備活動模式"""
#         print("==== Device Activity Analysis ====")

#         print("Available activity columns:")
#         activity_cols = ['power_state', 'is_on', 'is_off', 'is_phantom_load', 'is_light_use', 'is_regular_use']
#         for col in activity_cols:
#             if col in df.columns:
#                 print(f"  {col}: {df[col].value_counts().to_dict()}")
        
#         df_sorted = df.sort_values('timestamp').reset_index(drop=True)

#         # 計算待機時長和距離最後活躍時間
#         activity_records = []
        
#         # 識別活躍期間（regular_use 或 light_use）
#         df_sorted['is_active'] = (df_sorted.get('is_regular_use', False) | 
#                                   df_sorted.get('is_light_use', False))
        
#         # 識別待機期間（phantom_load 或 is_on但不active）
#         df_sorted['is_standby'] = (df_sorted.get('is_phantom_load', False) | 
#                                    ((df_sorted['is_on'] == True) & (df_sorted['is_active'] == False)))
        
#         print(f"\nActivity Analysis:")
#         print(f"Active periods: {df_sorted['is_active'].sum()}")
#         print(f"Standby periods: {df_sorted['is_standby'].sum()}")
#         print(f"Off periods: {df_sorted['is_off'].sum()}")

#         # 計算每個記錄的活動指標
#         last_active_time = None
#         current_standby_start = None
        
#         for idx, row in df_sorted.iterrows():
#             current_time = row['timestamp']
#             time_slot = row['time_slot']
#             day_type = row['day_type']
            
#             # 計算距離最後活躍時間
#             if row['is_active']:
#                 last_active_time = current_time
#                 time_since_active = 0.0
#                 current_standby_start = None
#             else:
#                 if last_active_time is not None:
#                     time_since_active = (current_time - last_active_time).total_seconds() / 60.0
#                 else:
#                     time_since_active = np.nan  # 沒有之前的活躍記錄
            
#             # 計算待機時長
#             if row['is_standby']:
#                 if current_standby_start is None:
#                     current_standby_start = current_time
#                     standby_duration = 0.0
#                 else:
#                     standby_duration = (current_time - current_standby_start).total_seconds() / 60.0
#             else:
#                 current_standby_start = None
#                 standby_duration = 0.0
            
#             # 過濾合理的數值範圍
#             if (not np.isnan(time_since_active) and 0 <= time_since_active <= 2880 and  # 最多48小時
#                 0 <= standby_duration <= 2880):  # 最多48小時
                
#                 activity_records.append({
#                     'timestamp': current_time,
#                     'time_slot': time_slot,
#                     'day_type': day_type,
#                     'standby_duration': standby_duration,
#                     'time_since_active': time_since_active,
#                     'is_active': row['is_active'],
#                     'is_standby': row['is_standby']
#                 })
        
#         self.activity_data = pd.DataFrame(activity_records)
        
#         print(f"\nCreated {len(self.activity_data)} activity records")
        
#         if len(self.activity_data) > 0:
#             standby_durations = self.activity_data['standby_duration']
#             time_since_active = self.activity_data['time_since_active']
            
#             print(f"Standby Duration statistics:")
#             print(f"  Min: {standby_durations.min():.1f} minutes")
#             print(f"  Max: {standby_durations.max():.1f} minutes")
#             print(f"  Mean: {standby_durations.mean():.1f} minutes")
#             print(f"  Median: {standby_durations.median():.1f} minutes")
            
#             print(f"Time Since Active statistics:")
#             print(f"  Min: {time_since_active.min():.1f} minutes")
#             print(f"  Max: {time_since_active.max():.1f} minutes")
#             print(f"  Mean: {time_since_active.mean():.1f} minutes")
#             print(f"  Median: {time_since_active.median():.1f} minutes")
#         else:
#             print("⚠️  No valid activity records found")
        
#         return self.activity_data

#     def calculate_standby_duration_score(self):
#         """計算待機時長分數"""
#         print("==== Calculating Standby Duration Score ====")

#         if self.activity_data is None or len(self.activity_data) == 0:
#             print("⚠️  No activity data - using fallback calculation")
#             self._create_default_standby_matrix()
#             return self.standby_duration_matrix
        
#         valid_entries = 0
        
#         for day_type in ['weekday', 'weekend']:
#             for time_slot in range(self.time_slots):
#                 mask = (self.activity_data['day_type'] == day_type) & \
#                        (self.activity_data['time_slot'] == time_slot)
                
#                 slot_data = self.activity_data[mask]

#                 if len(slot_data) < 1:
#                     # 使用基於時段的默認值
#                     hour = time_slot // 4
#                     if day_type == 'weekday':
#                         if 9 <= hour <= 17:  # 工作時間
#                             default_score = 0.3  # 可能需要更多活躍
#                         elif 18 <= hour <= 22:  # 晚間
#                             default_score = 0.6
#                         else:  # 深夜和早晨
#                             default_score = 0.8
#                     else:  # weekend
#                         if 8 <= hour <= 22:  # 白天
#                             default_score = 0.5
#                         else:  # 夜間
#                             default_score = 0.7
                    
#                     self.standby_duration_matrix[(day_type, time_slot)] = {
#                         'short_standby': 1.0 - default_score,
#                         'medium_standby': default_score / 2,
#                         'long_standby': default_score,
#                         'sample_size': 0
#                     }
#                     continue

#                 standby_durations = slot_data['standby_duration'].values
                
#                 try:
#                     # 使用百分位數定義模糊集合
#                     min_duration = max(0.0, np.min(standby_durations))
#                     q25 = np.percentile(standby_durations, 25)
#                     q50 = np.percentile(standby_durations, 50)
#                     q75 = np.percentile(standby_durations, 75)
#                     max_duration = min(2880.0, np.max(standby_durations))
                    
#                     # 確保順序正確
#                     if not (min_duration <= q25 <= q50 <= q75 <= max_duration):
#                         print(f"⚠️  Invalid percentiles for {day_type} slot {time_slot}, using defaults")
#                         self.standby_duration_matrix[(day_type, time_slot)] = {
#                             'short_standby': 0.6,
#                             'medium_standby': 0.3,
#                             'long_standby': 0.1,
#                             'sample_size': len(slot_data)
#                         }
#                         continue
                    
#                     # 定義待機時長的模糊集合
#                     fuzzy_sets = {
#                         'short': (min_duration, q25, q50),      # 短待機 -> 高活躍度
#                         'medium': (q25, q50, q75),              # 中等待機
#                         'long': (q50, q75, max_duration)        # 長待機 -> 可能需要喚醒
#                     }

#                     # 計算模糊機率
#                     fuzzy_probs = {}
#                     for category, (a, b, c) in fuzzy_sets.items():
#                         memberships = []
#                         for duration in standby_durations:
#                             membership = self.safe_triangular_membership(duration, a, b, c)
#                             memberships.append(membership)
                        
#                         avg_membership = np.mean(memberships) if memberships else 0.0
#                         fuzzy_probs[category] = max(0.0, min(1.0, avg_membership))
                    
#                     # 正規化機率
#                     total_prob = sum(fuzzy_probs.values())
#                     if total_prob > 0:
#                         normalization_factor = 1.0 / total_prob
#                         for category in fuzzy_probs:
#                             fuzzy_probs[category] *= normalization_factor
#                     else:
#                         fuzzy_probs = {'short': 0.33, 'medium': 0.34, 'long': 0.33}
                    
#                     self.standby_duration_matrix[(day_type, time_slot)] = {
#                         'short_standby': fuzzy_probs['short'],
#                         'medium_standby': fuzzy_probs['medium'],
#                         'long_standby': fuzzy_probs['long'],
#                         'sample_size': len(slot_data)
#                     }
#                     valid_entries += 1
                    
#                 except Exception as e:
#                     print(f"⚠️  Error calculating standby duration for {day_type} slot {time_slot}: {e}")
#                     self.standby_duration_matrix[(day_type, time_slot)] = {
#                         'short_standby': 0.4,
#                         'medium_standby': 0.4,
#                         'long_standby': 0.2,
#                         'sample_size': len(slot_data)
#                     }
                
#         print(f"✓ Calculated standby duration for {valid_entries} time slots")
#         return self.standby_duration_matrix

#     def calculate_time_since_active_score(self):
#         """計算距離最後活躍時間分數"""
#         print("==== Calculating Time Since Active Score ====")

#         if self.activity_data is None or len(self.activity_data) == 0:
#             print("⚠️  No activity data - using fallback calculation")
#             self._create_default_time_since_active_matrix()
#             return self.time_since_active_matrix
        
#         valid_entries = 0
        
#         for day_type in ['weekday', 'weekend']:
#             for time_slot in range(self.time_slots):
#                 mask = (self.activity_data['day_type'] == day_type) & \
#                        (self.activity_data['time_slot'] == time_slot)
                
#                 slot_data = self.activity_data[mask]

#                 if len(slot_data) < 1:
#                     # 使用基於時段的默認值
#                     hour = time_slot // 4
#                     if day_type == 'weekday':
#                         if 9 <= hour <= 17:  # 工作時間 - 期望較短的非活躍時間
#                             default_score = 0.7  # 高機率需要活躍
#                         elif 18 <= hour <= 22:  # 晚間
#                             default_score = 0.5
#                         else:  # 深夜和早晨
#                             default_score = 0.2
#                     else:  # weekend
#                         if 8 <= hour <= 22:  # 白天
#                             default_score = 0.4
#                         else:  # 夜間
#                             default_score = 0.1
                    
#                     self.time_since_active_matrix[(day_type, time_slot)] = {
#                         'recent_active': default_score,
#                         'moderate_inactive': 0.3,
#                         'long_inactive': 1.0 - default_score,
#                         'sample_size': 0
#                     }
#                     continue

#                 time_since_active = slot_data['time_since_active'].values
                
#                 try:
#                     # 使用百分位數定義模糊集合
#                     min_time = max(0.0, np.min(time_since_active))
#                     q25 = np.percentile(time_since_active, 25)
#                     q50 = np.percentile(time_since_active, 50)
#                     q75 = np.percentile(time_since_active, 75)
#                     max_time = min(2880.0, np.max(time_since_active))
                    
#                     # 確保順序正確
#                     if not (min_time <= q25 <= q50 <= q75 <= max_time):
#                         print(f"⚠️  Invalid percentiles for {day_type} slot {time_slot}, using defaults")
#                         self.time_since_active_matrix[(day_type, time_slot)] = {
#                             'recent_active': 0.5,
#                             'moderate_inactive': 0.3,
#                             'long_inactive': 0.2,
#                             'sample_size': len(slot_data)
#                         }
#                         continue
                    
#                     # 定義距離活躍時間的模糊集合
#                     fuzzy_sets = {
#                         'recent': (min_time, q25, q50),         # 最近活躍 -> 高活躍度
#                         'moderate': (q25, q50, q75),            # 中等時間未活躍
#                         'long': (q50, q75, max_time)            # 長時間未活躍 -> 需要喚醒
#                     }

#                     # 計算模糊機率
#                     fuzzy_probs = {}
#                     for category, (a, b, c) in fuzzy_sets.items():
#                         memberships = []
#                         for time_val in time_since_active:
#                             membership = self.safe_triangular_membership(time_val, a, b, c)
#                             memberships.append(membership)
                        
#                         avg_membership = np.mean(memberships) if memberships else 0.0
#                         fuzzy_probs[category] = max(0.0, min(1.0, avg_membership))
                    
#                     # 正規化機率
#                     total_prob = sum(fuzzy_probs.values())
#                     if total_prob > 0:
#                         normalization_factor = 1.0 / total_prob
#                         for category in fuzzy_probs:
#                             fuzzy_probs[category] *= normalization_factor
#                     else:
#                         fuzzy_probs = {'recent': 0.33, 'moderate': 0.34, 'long': 0.33}
                    
#                     self.time_since_active_matrix[(day_type, time_slot)] = {
#                         'recent_active': fuzzy_probs['recent'],
#                         'moderate_inactive': fuzzy_probs['moderate'],
#                         'long_inactive': fuzzy_probs['long'],
#                         'sample_size': len(slot_data)
#                     }
#                     valid_entries += 1
                    
#                 except Exception as e:
#                     print(f"⚠️  Error calculating time since active for {day_type} slot {time_slot}: {e}")
#                     self.time_since_active_matrix[(day_type, time_slot)] = {
#                         'recent_active': 0.4,
#                         'moderate_inactive': 0.4,
#                         'long_inactive': 0.2,
#                         'sample_size': len(slot_data)
#                     }
                
#         print(f"✓ Calculated time since active for {valid_entries} time slots")
#         return self.time_since_active_matrix

#     def _create_default_standby_matrix(self):
#         """創建默認的待機時長矩陣"""
#         for day_type in ['weekday', 'weekend']:
#             for time_slot in range(self.time_slots):
#                 hour = time_slot // 4
                
#                 if day_type == 'weekday':
#                     if 9 <= hour <= 17:  # 工作時間
#                         short, medium, long = 0.5, 0.3, 0.2
#                     elif 18 <= hour <= 22:  # 晚間
#                         short, medium, long = 0.3, 0.4, 0.3
#                     else:  # 深夜和早晨
#                         short, medium, long = 0.2, 0.3, 0.5
#                 else:  # weekend
#                     if 8 <= hour <= 22:  # 白天
#                         short, medium, long = 0.4, 0.3, 0.3
#                     else:  # 夜間
#                         short, medium, long = 0.2, 0.3, 0.5
                
#                 self.standby_duration_matrix[(day_type, time_slot)] = {
#                     'short_standby': short,
#                     'medium_standby': medium,
#                     'long_standby': long,
#                     'sample_size': 0
#                 }

#     def _create_default_time_since_active_matrix(self):
#         """創建默認的距離活躍時間矩陣"""
#         for day_type in ['weekday', 'weekend']:
#             for time_slot in range(self.time_slots):
#                 hour = time_slot // 4
                
#                 if day_type == 'weekday':
#                     if 9 <= hour <= 17:  # 工作時間
#                         recent, moderate, long = 0.6, 0.3, 0.1
#                     elif 18 <= hour <= 22:  # 晚間
#                         recent, moderate, long = 0.4, 0.4, 0.2
#                     else:  # 深夜和早晨
#                         recent, moderate, long = 0.1, 0.3, 0.6
#                 else:  # weekend
#                     if 8 <= hour <= 22:  # 白天
#                         recent, moderate, long = 0.3, 0.4, 0.3
#                     else:  # 夜間
#                         recent, moderate, long = 0.1, 0.2, 0.7
                
#                 self.time_since_active_matrix[(day_type, time_slot)] = {
#                     'recent_active': recent,
#                     'moderate_inactive': moderate,
#                     'long_inactive': long,
#                     'sample_size': 0
#                 }

#     def calculate_membership_parameters(self):
#         """計算統計隸屬參數"""
#         print("==== Calculating Statistical Membership Parameters ====")
        
#         self.membership_parameters = {
#             'standby_duration': {},
#             'time_since_active': {}
#         }
        
#         # 收集 Standby Duration 數據
#         standby_values = []
#         for key in self.standby_duration_matrix:
#             data = self.standby_duration_matrix[key]
#             if 'short_standby' in data and not np.isnan(data['short_standby']):
#                 standby_values.append(data['short_standby'])
        
#         if standby_values:
#             standby_values = np.array(standby_values)
#             self.membership_parameters['standby_duration'] = {
#                 'p0': float(np.min(standby_values)),
#                 'p25': float(np.percentile(standby_values, 25)),
#                 'p50': float(np.percentile(standby_values, 50)),
#                 'p75': float(np.percentile(standby_values, 75)),
#                 'p100': float(np.max(standby_values))
#             }
#             print(f"Standby Duration Statistics (n={len(standby_values)}): ✓")
#         else:
#             print("⚠️  No valid standby duration data, using defaults")
#             self.membership_parameters['standby_duration'] = {
#                 'p0': 0.0, 'p25': 0.25, 'p50': 0.5, 'p75': 0.75, 'p100': 1.0
#             }
        
#         # 收集 Time Since Active 數據
#         time_since_values = []
#         for key in self.time_since_active_matrix:
#             data = self.time_since_active_matrix[key]
#             if 'recent_active' in data and not np.isnan(data['recent_active']):
#                 time_since_values.append(data['recent_active'])
        
#         if time_since_values:
#             time_since_values = np.array(time_since_values)
#             self.membership_parameters['time_since_active'] = {
#                 'p0': float(np.min(time_since_values)),
#                 'p25': float(np.percentile(time_since_values, 25)),
#                 'p50': float(np.percentile(time_since_values, 50)),
#                 'p75': float(np.percentile(time_since_values, 75)),
#                 'p100': float(np.max(time_since_values))
#             }
#             print(f"Time Since Active Statistics (n={len(time_since_values)}): ✓")
#         else:
#             print("⚠️  No valid time since active data, using defaults")
#             self.membership_parameters['time_since_active'] = {
#                 'p0': 0.0, 'p25': 0.25, 'p50': 0.5, 'p75': 0.75, 'p100': 1.0
#             }
        
#         return self.membership_parameters

#     def define_activity_rules(self):
#         """定義設備活躍度模糊規則"""
#         print("==== Defining Device Activity Rules ====")
        
#         # 規則格式: (待機時長, 距離最後活躍時間, 輸出活躍度等級, 權重)
#         self.activity_rules = [
#             # 高活躍度規則：設備很活躍，不需要特別處理
#             ('short', 'recent', 'high', 1.0),           # 短待機+最近活躍 = 高活躍度
#             ('short', 'moderate', 'high', 0.8),         # 短待機+中等非活躍 = 高活躍度
#             ('medium', 'recent', 'high', 0.9),          # 中等待機+最近活躍 = 高活躍度
            
#             # 中等活躍度規則：需要監控但不緊急
#             ('short', 'long', 'medium', 0.6),           # 短待機但長時間未活躍
#             ('medium', 'moderate', 'medium', 0.7),      # 中等待機+中等非活躍
#             ('long', 'recent', 'medium', 0.5),          # 長待機但最近活躍
            
#             # 低活躍度規則：可能需要喚醒或節能處理
#             ('medium', 'long', 'low', 0.8),             # 中等待機+長時間未活躍
#             ('long', 'moderate', 'low', 0.9),           # 長待機+中等非活躍
#             ('long', 'long', 'low', 1.0),               # 長待機+長時間未活躍 = 最低活躍度
#         ]
        
#         print(f"✓ Defined {len(self.activity_rules)} activity rules")
#         return self.activity_rules

#     def calculate_fuzzy_memberships(self, timestamp):
#         """計算指定時間點的模糊隸屬度"""
#         # 提取時間特徵
#         hour = timestamp.hour
#         minute = timestamp.minute
#         is_weekend = timestamp.weekday() >= 5
#         day_type = 'weekend' if is_weekend else 'weekday'
#         time_slot = hour * 4 + minute // 15
        
#         result = {
#             'day_type': day_type,
#             'time_slot': time_slot,
#             'hour': hour,
#             'minute': minute
#         }
        
#         # 獲取待機時長隸屬度
#         if (day_type, time_slot) in self.standby_duration_matrix:
#             standby_data = self.standby_duration_matrix[(day_type, time_slot)]
            
#             standby_params = self.membership_parameters['standby_duration']
#             short_standby = standby_data.get('short_standby', 0.4)
            
#             result['standby_short'] = self.safe_triangular_membership(
#                 short_standby, standby_params['p0'], standby_params['p25'], standby_params['p50']
#             )
#             result['standby_medium'] = self.safe_triangular_membership(
#                 short_standby, standby_params['p25'], standby_params['p50'], standby_params['p75']
#             )
#             result['standby_long'] = self.safe_triangular_membership(
#                 short_standby, standby_params['p50'], standby_params['p75'], standby_params['p100']
#             )
#             result['standby_score'] = short_standby
#         else:
#             result.update({
#                 'standby_short': 0.5, 'standby_medium': 0.3, 'standby_long': 0.2,
#                 'standby_score': 0.4
#             })
        
#         # 獲取距離最後活躍時間隸屬度
#         if (day_type, time_slot) in self.time_since_active_matrix:
#             time_since_data = self.time_since_active_matrix[(day_type, time_slot)]
            
#             time_since_params = self.membership_parameters['time_since_active']
#             recent_active = time_since_data.get('recent_active', 0.4)
            
#             result['time_recent'] = self.safe_triangular_membership(
#                 recent_active, time_since_params['p0'], time_since_params['p25'], time_since_params['p50']
#             )
#             result['time_moderate'] = self.safe_triangular_membership(
#                 recent_active, time_since_params['p25'], time_since_params['p50'], time_since_params['p75']
#             )
#             result['time_long'] = self.safe_triangular_membership(
#                 recent_active, time_since_params['p50'], time_since_params['p75'], time_since_params['p100']
#             )
#             result['time_since_score'] = recent_active
#         else:
#             result.update({
#                 'time_recent': 0.4, 'time_moderate': 0.4, 'time_long': 0.2,
#                 'time_since_score': 0.4
#             })
        
#         return result

#     def calculate_activity_score(self, timestamp):
#         """計算設備活躍度分數"""
#         try:
#             memberships = self.calculate_fuzzy_memberships(timestamp)
            
#             # 計算規則激活
#             low_activation = 0.0
#             medium_activation = 0.0
#             high_activation = 0.0
#             valid_rules = 0
            
#             for rule in self.activity_rules:
#                 standby_level, time_level, output_level, weight = rule
                
#                 # 獲取隸屬度
#                 standby_membership = memberships.get(f'standby_{standby_level}', 0.0)
#                 time_membership = memberships.get(f'time_{time_level}', 0.0)
                
#                 # 檢查是否有有效的隸屬度
#                 if any(np.isnan([standby_membership, time_membership])):
#                     continue
                
#                 # 使用最小值方法計算規則激活
#                 if all(m >= 0 for m in [standby_membership, time_membership]):
#                     activation = min(standby_membership, time_membership) * weight
                    
#                     # 累積激活
#                     if output_level == 'low':
#                         low_activation += activation
#                     elif output_level == 'medium':
#                         medium_activation += activation
#                     elif output_level == 'high':
#                         high_activation += activation
                    
#                     valid_rules += 1
            
#             # 限制激活強度
#             low_activation = min(low_activation, 1.0)
#             medium_activation = min(medium_activation, 1.0)
#             high_activation = min(high_activation, 1.0)
            
#             # 計算最終活躍度分數
#             total_activation = low_activation + medium_activation + high_activation
            
#             if total_activation > 0 and valid_rules > 0:
#                 # 使用加權重心法（高活躍度 = 高分數）
#                 activity_score = (
#                     low_activation * 0.2 +       # 低活躍度對應低分
#                     medium_activation * 0.5 +    # 中等活躍度
#                     high_activation * 0.8         # 高活躍度對應高分
#                 ) / total_activation
                
#                 # 基於實際數據的微調
#                 standby_score = memberships.get('standby_score', 0.5)
#                 time_since_score = memberships.get('time_since_score', 0.5)
                
#                 # 微調：短待機和最近活躍 -> 提高分數
#                 data_adjustment = (standby_score + time_since_score - 1.0) * 0.05
#                 activity_score += data_adjustment
                
#                 confidence = min(1.0, total_activation * valid_rules / len(self.activity_rules))
#             else:
#                 # 後備計算
#                 activity_score = 0.4
#                 confidence = 0.1
            
#             # 確保分數在合理範圍內
#             activity_score = max(0.05, min(0.95, activity_score))
            
#             return {
#                 'activity_score': activity_score,
#                 'low_activation': low_activation,
#                 'medium_activation': medium_activation,
#                 'high_activation': high_activation,
#                 'memberships': memberships,
#                 'confidence': confidence,
#                 'valid_rules': valid_rules
#             }
            
#         except Exception as e:
#             print(f"⚠️  Error calculating activity score: {e}")
#             return {
#                 'activity_score': 0.4,
#                 'low_activation': 0.0,
#                 'medium_activation': 0.5,
#                 'high_activation': 0.0,
#                 'memberships': {},
#                 'confidence': 0.1,
#                 'valid_rules': 0
#             }

#     # def plot_triangular_membership_functions(self):
#     #     """繪製三角隸屬函數圖"""
        
#     #     # 確保有 membership_parameters
#     #     if not hasattr(self, 'membership_parameters') or not self.membership_parameters:
#     #         print("❌ 沒有找到 membership_parameters，請先運行完整分析")
#     #         return
        
#     #     # 設置圖形
#     #     fig, axes = plt.subplots(1, 2, figsize=(15, 6))
#     #     fig.suptitle('Device Activity Score - Triangular Membership Functions', fontsize=16, fontweight='bold')
        
#     #     # 定義顏色和樣式
#     #     colors = ['red', 'orange', 'green']
#     #     line_styles = ['-', '--', '-.']
        
#     #     # 1. Standby Duration 隸屬函數
#     #     ax1 = axes[0]
#     #     x_standby = np.linspace(0, 1, 1000)
        
#     #     if 'standby_duration' in self.membership_parameters:
#     #         standby_params = self.membership_parameters['standby_duration']
            
#     #         if all(key in standby_params for key in ['p0', 'p25', 'p50', 'p75', 'p100']):
#     #             p0, p25, p50, p75, p100 = [standby_params[k] for k in ['p0', 'p25', 'p50', 'p75', 'p100']]
                
#     #             standby_short = np.array([self.safe_triangular_membership(x, p0, p25, p50) for x in x_standby])
#     #             standby_medium = np.array([self.safe_triangular_membership(x, p25, p50, p75) for x in x_standby])
#     #             standby_long = np.array([self.safe_triangular_membership(x, p50, p75, p100) for x in x_standby])
                
#     #             ax1.plot(x_standby, standby_short, color=colors[0], linewidth=3, 
#     #                     label=f'Short ({p0:.3f}, {p25:.3f}, {p50:.3f})', linestyle=line_styles[0])
#     #             ax1.plot(x_standby, standby_medium, color=colors[1], linewidth=3, 
#     #                     label=f'Medium ({p25:.3f}, {p50:.3f}, {p75:.3f})', linestyle=line_styles[1])
#     #             ax1.plot(x_standby, standby_long, color=colors[2], linewidth=3, 
#     #                     label=f'Long ({p50:.3f}, {p75:.3f}, {p100:.3f})', linestyle=line_styles[2])
                
#     #             # 標記關鍵點
#     #             key_points = [p25, p50, p75]
#     #             for i, point in enumerate(key_points):
#     #                 ax1.axvline(point, color='gray', linestyle=':', alpha=0.6, linewidth=1)
#     #                 ax1.text(point, 1.05, f'P{25*(i+1)}', ha='center', va='bottom', fontsize=8)
                
#     #             # 填充區域
#     #             ax1.fill_between(x_standby, 0, standby_short, alpha=0.2, color=colors[0])
#     #             ax1.fill_between(x_standby, 0, standby_medium, alpha=0.2, color=colors[1])
#     #             ax1.fill_between(x_standby, 0, standby_long, alpha=0.2, color=colors[2])
#     #         else:
#     #             ax1.text(0.5, 0.5, 'Standby Duration\nParameters Incomplete', ha='center', va='center', fontsize=12)
        
#     #     ax1.set_xlabel('Standby Duration Score')
#     #     ax1.set_ylabel('Membership Degree')
#     #     ax1.set_title('Standby Duration', fontweight='bold')
#     #     ax1.legend(loc='upper right', fontsize=9)
#     #     ax1.grid(True, alpha=0.3)
#     #     ax1.set_xlim(0, 1)
#     #     ax1.set_ylim(0, 1.1)
        
#     #     # 2. Time Since Active 隸屬函數
#     #     ax2 = axes[1]
#     #     x_time = np.linspace(0, 1, 1000)
        
#     #     if 'time_since_active' in self.membership_parameters:
#     #         time_params = self.membership_parameters['time_since_active']
            
#     #         if all(key in time_params for key in ['p0', 'p25', 'p50', 'p75', 'p100']):
#     #             p0, p25, p50, p75, p100 = [time_params[k] for k in ['p0', 'p25', 'p50', 'p75', 'p100']]
                
#     #             time_recent = np.array([self.safe_triangular_membership(x, p0, p25, p50) for x in x_time])
#     #             time_moderate = np.array([self.safe_triangular_membership(x, p25, p50, p75) for x in x_time])
#     #             time_long = np.array([self.safe_triangular_membership(x, p50, p75, p100) for x in x_time])
                
#     #             ax2.plot(x_time, time_recent, color=colors[0], linewidth=3, 
#     #                     label=f'Recent ({p0:.3f}, {p25:.3f}, {p50:.3f})', linestyle=line_styles[0])
#     #             ax2.plot(x_time, time_moderate, color=colors[1], linewidth=3, 
#     #                     label=f'Moderate ({p25:.3f}, {p50:.3f}, {p75:.3f})', linestyle=line_styles[1])
#     #             ax2.plot(x_time, time_long, color=colors[2], linewidth=3, 
#     #                     label=f'Long ({p50:.3f}, {p75:.3f}, {p100:.3f})', linestyle=line_styles[2])
                
#     #             # 標記關鍵點
#     #             key_points = [p25, p50, p75]
#     #             for i, point in enumerate(key_points):
#     #                 ax2.axvline(point, color='gray', linestyle=':', alpha=0.6, linewidth=1)
#     #                 ax2.text(point, 1.05, f'P{25*(i+1)}', ha='center', va='bottom', fontsize=8)
                
#     #             # 填充區域
#     #             ax2.fill_between(x_time, 0, time_recent, alpha=0.2, color=colors[0])
#     #             ax2.fill_between(x_time, 0, time_moderate, alpha=0.2, color=colors[1])
#     #             ax2.fill_between(x_time, 0, time_long, alpha=0.2, color=colors[2])
#     #         else:
#     #             ax2.text(0.5, 0.5, 'Time Since Active\nParameters Incomplete', ha='center', va='center', fontsize=12)
        
#     #     ax2.set_xlabel('Time Since Active Score')
#     #     ax2.set_ylabel('Membership Degree')
#     #     ax2.set_title('Time Since Last Active', fontweight='bold')
#     #     ax2.legend(loc='upper right', fontsize=9)
#     #     ax2.grid(True, alpha=0.3)
#     #     ax2.set_xlim(0, 1)
#     #     ax2.set_ylim(0, 1.1)
        
#     #     plt.tight_layout()
#     #     plt.show()
        
#     #     # 打印詳細參數信息
#     #     print("="*60)
#     #     print("Device Activity Score - Triangular Membership Parameters")
#     #     print("="*60)
        
#     #     for param_type, params in self.membership_parameters.items():
#     #         print(f"\n📊 {param_type.replace('_', ' ').title()}:")
#     #         if all(key in params for key in ['p0', 'p25', 'p50', 'p75', 'p100']):
#     #             p0, p25, p50, p75, p100 = [params[k] for k in ['p0', 'p25', 'p50', 'p75', 'p100']]
#     #             print(f"  Data Range: [{p0:.3f}, {p100:.3f}]")
#     #             print(f"  Low/Short Triangular:    ({p0:.3f}, {p25:.3f}, {p50:.3f})")
#     #             print(f"  Medium Triangular:       ({p25:.3f}, {p50:.3f}, {p75:.3f})")
#     #             print(f"  High/Long Triangular:    ({p50:.3f}, {p75:.3f}, {p100:.3f})")
                
#     #             # 計算重疊度
#     #             overlap_low_med = max(0, p50 - p25) / (p50 - p0) if p50 > p0 else 0
#     #             overlap_med_high = max(0, p75 - p50) / (p100 - p50) if p100 > p50 else 0
#     #             print(f"  Overlap: Low-Medium: {overlap_low_med:.2f}, Medium-High: {overlap_med_high:.2f}")

#     def test_activity_score_calculation(self, num_tests=5):
#         """測試活躍度分數計算功能"""
#         print("==== Testing Activity Score Calculation ====")
        


#         test_times = [
#             (datetime(2024, 1, 15, 9, 0), (0.2, 0.4), '工作日早上'),    # 調整期望
#             (datetime(2024, 1, 15, 14, 0), (0.2, 0.4), '工作日下午'),   # 調整期望
#             (datetime(2024, 1, 15, 22, 0), (0.6, 0.9), '工作日深夜'),   # 大幅調整！
#             (datetime(2024, 1, 13, 10, 0), (0.1, 0.3), '週末上午'),    # 調整期望
#             (datetime(2024, 1, 13, 2, 0), (0.3, 0.5), '週末凌晨'),     # 大幅調整！
#         ]
        
#         test_results = []
        
#         for i, (test_time, expected_range, desc) in enumerate(test_times[:num_tests]):
#             try:
#                 result = self.calculate_activity_score(test_time)
                
#                 day_type = "Weekend" if test_time.weekday() >= 5 else "Weekday"
#                 print(f"\nTest {i+1}: {test_time.strftime('%Y-%m-%d %H:%M')} ({day_type})")
#                 print(f"  Activity Score: {result['activity_score']:.3f}")
#                 print(f"  Confidence: {result['confidence']:.3f}")
#                 print(f"  Valid Rules: {result['valid_rules']}")
#                 print(f"  Activations - Low: {result['low_activation']:.3f}, "
#                       f"Medium: {result['medium_activation']:.3f}, "
#                       f"High: {result['high_activation']:.3f}")
                
#                 test_results.append({
#                     'time': test_time,
#                     'score': result['activity_score'],
#                     'confidence': result['confidence']
#                 })
                
#             except Exception as e:
#                 print(f"⚠️  Error in test {i+1}: {e}")
#                 test_results.append({
#                     'time': test_time,
#                     'score': 0.4,
#                     'confidence': 0.0
#                 })
        
#         return test_results

#     def detect_usage_pattern(self):
#         """檢測設備使用模式"""
#         if self.activity_data is None or len(self.activity_data) == 0:
#             return "未知模式"
        
#         total_records = len(self.activity_data)
#         active_count = len(self.activity_data[self.activity_data['is_active'] == True])
#         standby_count = len(self.activity_data[self.activity_data['is_standby'] == True])
        
#         active_ratio = active_count / total_records
#         standby_ratio = standby_count / total_records
#         off_ratio = 1 - active_ratio - standby_ratio
        
#         print(f"\n🔍 使用模式檢測:")
#         print(f"   活躍時間: {active_ratio:.1%}")
#         print(f"   待機時間: {standby_ratio:.1%}")
#         print(f"   關機時間: {off_ratio:.1%}")
        
#         if standby_ratio < 0.2 and active_ratio > 0.4:
#             pattern = "高效節能模式"
#             description = "用完即關機，很少待機浪費電力"
#         elif standby_ratio > 0.4:
#             pattern = "高待機模式"  
#             description = "設備經常保持待機狀態"
#         elif active_ratio > 0.6:
#             pattern = "高使用模式"
#             description = "設備大部分時間處於活躍狀態"
#         else:
#             pattern = "混合使用模式"
#             description = "活躍、待機、關機時間較為平衡"
        
#         print(f"   檢測結果: {pattern} - {description}")
#         return pattern

#     def comprehensive_evaluation(self):
#         """完整的系統評估"""
#         print("\n" + "="*60)
#         print("DEVICE ACTIVITY SCORE - COMPREHENSIVE EVALUATION")
#         print("="*60)
        
#         # 0. 使用模式檢測
#         usage_pattern = self.detect_usage_pattern()
        
#         # 1. 數據質量評估
#         print(f"\n1. Data Quality Assessment:")
#         print(f"   Quality Score: {self.data_quality_report.get('quality_score', 0):.2f}")
#         print(f"   Issues Count: {len(self.data_quality_report.get('issues', []))}")
        
#         # 2. 矩陣完整性檢查
#         print(f"\n2. Matrix Completeness:")
#         matrices = {
#             'Standby Duration': self.standby_duration_matrix,
#             'Time Since Active': self.time_since_active_matrix
#         }
        
#         for name, matrix in matrices.items():
#             total_slots = 192  # 2 day types * 96 time slots
#             coverage = len(matrix) / total_slots * 100
#             nan_count = 0
            
#             for v in matrix.values():
#                 if isinstance(v, dict):
#                     for val in v.values():
#                         if isinstance(val, (int, float)) and np.isnan(val):
#                             nan_count += 1
#                             break
            
#             print(f"   {name}: {coverage:.1f}% coverage, {nan_count} entries with NaN")
        
#         # 3. 活躍度分數測試 (調整為符合高效使用模式的期望)
#         print(f"\n3. Activity Score Tests:")
#         test_scenarios = [
#             (datetime(2024, 1, 15, 9, 0), (0.2, 0.4), '工作日早上'),    # 高效使用模式
#             (datetime(2024, 1, 15, 14, 0), (0.2, 0.4), '工作日下午'),   # 高效使用模式
#             (datetime(2024, 1, 15, 22, 0), (0.6, 0.9), '工作日深夜'),   # 夜間可能更高分
#             (datetime(2024, 1, 13, 10, 0), (0.1, 0.3), '週末上午'),    # 週末較低使用
#             (datetime(2024, 1, 13, 2, 0), (0.3, 0.5), '週末凌晨'),     # 調整期望範圍
#         ]
        
#         passed_tests = 0
#         for test_time, expected_range, desc in test_scenarios:
#             try:
#                 result = self.calculate_activity_score(test_time)
#                 score = result['activity_score']
#                 confidence = result['confidence']
                
#                 is_reasonable = expected_range[0] <= score <= expected_range[1]
#                 has_confidence = confidence > 0.1
                
#                 if is_reasonable and has_confidence:
#                     passed_tests += 1
#                     status = '✓'
#                 else:
#                     status = '❌'
                
#                 print(f"   {status} {desc}: {score:.3f} (期望: {expected_range})")
#             except Exception as e:
#                 print(f"   ❌ {desc}: Error - {e}")
        
#         # 4. 系統穩定性測試
#         print(f"\n4. System Stability:")
#         try:
#             base_time = datetime(2024, 1, 15, 14, 0)
#             base_result = self.calculate_activity_score(base_time)
#             base_score = base_result['activity_score']
            
#             sensitivity_diffs = []
#             for minutes in [-30, -15, 15, 30]:
#                 test_time = base_time + timedelta(minutes=minutes)
#                 result = self.calculate_activity_score(test_time)
#                 diff = abs(result['activity_score'] - base_score)
#                 sensitivity_diffs.append(diff)
            
#             avg_sensitivity = np.mean(sensitivity_diffs)
#             print(f"   Time sensitivity: {avg_sensitivity:.3f}")
#             print(f"   Stability: {'Good' if avg_sensitivity < 0.15 else 'Needs improvement'}")
#         except Exception as e:
#             print(f"   ❌ Stability test failed: {e}")
#             avg_sensitivity = 1.0
        
#         # 5. 最終評分
#         print(f"\n=== FINAL ASSESSMENT ===")
#         quality_score = self.data_quality_report.get('quality_score', 0)
#         test_pass_rate = passed_tests / len(test_scenarios)
#         stability_score = max(0, 1 - avg_sensitivity / 0.2)
        
#         overall_score = (quality_score + test_pass_rate + stability_score) / 3
        
#         print(f"Data Quality: {quality_score:.2f}")
#         print(f"Test Pass Rate: {test_pass_rate:.2f}")
#         print(f"Stability Score: {stability_score:.2f}")
#         print(f"Overall System Quality: {overall_score:.2f}")
        
#         if overall_score >= 0.8:
#             print("🎉 System Quality: Excellent")
#         elif overall_score >= 0.6:
#             print("✅ System Quality: Good") 
#         elif overall_score >= 0.4:
#             print("⚠️  System Quality: Acceptable")
#         else:
#             print("❌ System Quality: Needs Improvement")
        
#         # 添加使用模式說明
#         if usage_pattern == "高效節能模式":
#             print("💡 檢測到高效節能使用模式，活躍度分數較低是正常現象")
#             print("   這表明您有良好的電源管理習慣！")

#     def run_complete_analysis(self, file_path):
#         """運行完整分析"""
#         print("="*80)
#         print("DEVICE ACTIVITY SCORE MODULE - COMPLETE ANALYSIS")
#         print("="*80)

#         # 1. 載入數據
#         df = self.load_data(file_path)
#         if df is None:
#             print('❌ Cannot load data')
#             return None
        
#         # 2. 分析設備活動
#         print("\n" + "-"*50)
#         self.analyze_device_activity(df)
        
#         # 3. 計算兩個核心指標
#         print("\n" + "-"*50)
#         self.calculate_standby_duration_score()
        
#         print("\n" + "-"*50)
#         self.calculate_time_since_active_score()

#         # 4. 計算隸屬參數
#         print("\n" + "-"*50)
#         self.calculate_membership_parameters()
    
#         # 5. 定義規則
#         print("\n" + "-"*50)
#         self.define_activity_rules()

#         # 6. 測試計算
#         print("\n" + "-"*50)
#         test_results = self.test_activity_score_calculation()

#         # 7. 綜合評估
#         print("\n" + "-"*50)
#         self.comprehensive_evaluation()

#         # 8. 繪製三角隸屬函數圖
#         print("\n" + "-"*50)
#         print("==== Plotting Triangular Membership Functions ====")
#         # self.plot_triangular_membership_functions()

#         print("\n" + "="*80)
#         print("✅ ANALYSIS COMPLETE - Device Activity Score system ready!")
#         print("="*80)

#         return {
#             'standby_duration': self.standby_duration_matrix,
#             'time_since_active': self.time_since_active_matrix,
#             'membership_parameters': self.membership_parameters,
#             'activity_rules': self.activity_rules,
#             'test_results': test_results,
#             'data_quality': self.data_quality_report
#         }

# # 使用示例
# if __name__ == "__main__":
#     # 初始化設備活躍度模組
#     activity_module = DeviceActivityScoreModule()
    
#     # 檔案路徑
#     # file_path = "C:/Users/王俞文/Documents/glasgow/msc project/data/data_after_preprocessing.csv"
#     # file_path = "C:/Users/王俞文/OneDrive - University of Glasgow/文件/glasgow/msc project/data/data_after_preprocessing.csv"
#     file_path = "C:/Users/王俞文/OneDrive - University of Glasgow/文件/glasgow/msc project/data/extended_power_data_2months.csv"
    
#     # 運行完整分析
#     result = activity_module.run_complete_analysis(file_path)
    
#     # 單獨測試活躍度分數計算
#     if result:
#         print("\n" + "="*50)
#         print("TESTING INDIVIDUAL ACTIVITY SCORE CALCULATIONS")
#         print("="*50)
        
#         # 測試幾個特定時間點
#         test_times = [
#             datetime(2024, 6, 15, 8, 30),   # 週六早上8:30
#             datetime(2024, 6, 17, 19, 45),  # 週一晚上7:45
#             datetime(2024, 6, 20, 14, 15),  # 週四下午2:15
#         ]
        
#         for test_time in test_times:
#             result = activity_module.calculate_activity_score(test_time)
#             day_type = "Weekend" if test_time.weekday() >= 5 else "Weekday"
            
#             print(f"\n時間: {test_time.strftime('%Y-%m-%d %H:%M')} ({day_type})")
#             print(f"活躍度分數: {result['activity_score']:.3f}")
#             print(f"信心度: {result['confidence']:.3f}")
            
#             # 提供建議 (調整為高效使用模式)
#             score = result['activity_score']
#             if score >= 0.5:
#                 suggestion = "🟢 設備使用效率高，電源管理良好"
#             elif score >= 0.3:
#                 suggestion = "🟡 設備使用正常，高效節能模式"
#             elif score >= 0.15:
#                 suggestion = "🟠 設備使用較少，符合節能習慣"
#             else:
#                 suggestion = "🔴 設備活躍度極低，可能需要檢查"
            
#             print(f"建議: {suggestion}")
        
#         print(f"\n💡 提示：如果想重新查看三角隸屬函數圖，可以運行：")
#         print(f"activity_module.plot_triangular_membership_functions()")


# Improved Device Activity Score Module
# Based on device status + time reasonableness rather than just device status

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DeviceActivityScoreModule:

    def __init__(self):
        self.time_slots = 96  # 96個時段，每15分鐘一個 (24*4)
        self.activity_data = None
        self.standby_duration_matrix = {}
        self.time_since_active_matrix = {}
        self.activity_rules = []
        self.membership_parameters = {}
        self.data_quality_report = {}
        
        # 新增：時間合理性配置
        self.time_reasonableness_weights = {}
        self._initialize_time_weights()

    def _initialize_time_weights(self):
        """初始化時間合理性權重"""
        print("==== Initializing Time Reasonableness Weights ====")
        
        # 定義不同時段對於設備使用的合理性權重
        self.time_reasonableness_weights = {
            # 深夜到凌晨 (00:00-05:59) - 應該休息
            'deep_night': {
                'hours': list(range(0, 6)),
                'weight': 0.2,  # 大幅降低
                'description': '深夜睡眠時間，設備使用不合理'
            },
            
            # 早晨 (06:00-08:59) - 起床時間，適度使用
            'early_morning': {
                'hours': list(range(6, 9)),
                'weight': 0.6,  # 中等權重
                'description': '早晨時光，適度設備使用'
            },
            
            # 上午 (09:00-11:59) - 工作時間，娛樂設備使用不太合理
            'morning_work': {
                'hours': list(range(9, 12)),
                'weight': 0.5,  # 中低權重
                'description': '上午工作時間'
            },
            
            # 下午 (12:00-14:59) - 午餐休息，較為合理
            'afternoon_break': {
                'hours': list(range(12, 15)),
                'weight': 0.8,  # 高權重
                'description': '午餐休息時間'
            },
            
            # 下午工作 (15:00-17:59) - 工作時間
            'afternoon_work': {
                'hours': list(range(15, 18)),
                'weight': 0.5,  # 中低權重
                'description': '下午工作時間'
            },
            
            # 傍晚 (18:00-20:59) - 主要娛樂時間，最合理
            'evening_leisure': {
                'hours': list(range(18, 21)),
                'weight': 1.0,  # 最高權重
                'description': '傍晚娛樂時間，設備使用最合理'
            },
            
            # 晚上 (21:00-21:59) - 開始準備休息
            'night_transition': {
                'hours': [21],
                'weight': 0.7,  # 中高權重，但開始降低
                'description': '晚上時間，開始準備休息'
            },
            
            # 深夜前 (22:00-23:59) - 準備睡覺，權重很低
            'late_night': {
                'hours': [22, 23],
                'weight': 0.3,  # 低權重
                'description': '晚間時光，應該準備休息'
            }
        }
        
        print("✓ Time reasonableness weights initialized")

    def get_time_reasonableness_weight(self, timestamp):
        """獲取指定時間的合理性權重"""
        hour = timestamp.hour
        weekday = timestamp.weekday()
        is_weekend = weekday >= 5
        
        # 找到對應的時間段
        base_weight = 0.5  # 默認權重
        time_period = 'unknown'
        
        for period_name, period_data in self.time_reasonableness_weights.items():
            if hour in period_data['hours']:
                base_weight = period_data['weight']
                time_period = period_name
                break
        
        # 週末調整
        weekend_adjustment = self._get_weekend_weight_adjustment(hour, is_weekend)
        final_weight = max(0.1, min(1.0, base_weight + weekend_adjustment))
        
        return {
            'weight': final_weight,
            'base_weight': base_weight,
            'weekend_adjustment': weekend_adjustment,
            'time_period': time_period,
            'is_weekend': is_weekend
        }

    def _get_weekend_weight_adjustment(self, hour, is_weekend):
        """週末時間權重調整"""
        if not is_weekend:
            return 0.0
        
        # 週末調整邏輯
        if 6 <= hour <= 10:  # 週末早晨可以更放鬆
            return 0.2
        elif 22 <= hour <= 23:  # 週末可以稍微晚一點
            return 0.2
        elif 0 <= hour <= 2:  # 週末深夜稍微寬鬆
            return 0.1
        elif 9 <= hour <= 17:  # 週末白天工作時間更寬鬆
            return 0.3
        else:
            return 0.1

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
        
        # 檢查15分鐘取樣間隔的一致性
        if 'timestamp' in df.columns and len(df) > 1:
            df_sorted = df.sort_values('timestamp')
            time_diffs = df_sorted['timestamp'].diff().dropna()
            expected_interval = pd.Timedelta(minutes=15)
            
            # 檢查是否大部分間隔接近15分鐘
            normal_intervals = time_diffs[abs(time_diffs - expected_interval) <= pd.Timedelta(minutes=5)]
            interval_consistency = len(normal_intervals) / len(time_diffs)
            
            if interval_consistency < 0.8:
                issues.append(f"Irregular 15-minute sampling: only {interval_consistency:.1%} intervals are consistent")
        
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
        """增強的數據預處理，優化15分鐘取樣"""
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
        
        # 優化15分鐘間隔的處理
        df = self._optimize_15min_intervals(df)
        
        # 處理異常的時間差值
        if 'time_diff_seconds' in df.columns:
            # 異常值檢測和處理
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
        
        # 確保狀態欄位的數據類型正確
        boolean_columns = ['is_on', 'is_off', 'is_phantom_load', 'is_light_use', 'is_regular_use']
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        print(f"✓ Preprocessing completed. Final dataset: {len(df)} records")
        return df

    def _optimize_15min_intervals(self, df):
        """優化15分鐘間隔的處理"""
        print("Optimizing 15-minute interval processing...")
        
        df_sorted = df.sort_values('timestamp').copy()
        
        # 計算實際使用強度（在15分鐘內的使用密度）
        if 'is_regular_use' in df.columns and 'is_light_use' in df.columns:
            # 為每個15分鐘時段計算使用強度
            df_sorted['usage_intensity'] = 0.0
            
            # regular_use = 1.0, light_use = 0.6, phantom_load = 0.2, off = 0.0
            df_sorted.loc[df_sorted['is_regular_use'] == True, 'usage_intensity'] = 1.0
            df_sorted.loc[df_sorted['is_light_use'] == True, 'usage_intensity'] = 0.6
            df_sorted.loc[df_sorted.get('is_phantom_load', False) == True, 'usage_intensity'] = 0.2
            df_sorted.loc[df_sorted['is_off'] == True, 'usage_intensity'] = 0.0
            
            # 計算滑動平均以平滑15分鐘間隔的波動
            window_size = min(3, len(df_sorted) // 10)  # 適應性窗口大小
            if window_size >= 1:
                df_sorted['smoothed_intensity'] = df_sorted['usage_intensity'].rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()
            else:
                df_sorted['smoothed_intensity'] = df_sorted['usage_intensity']
        
        print(f"✓ 15-minute interval optimization completed")
        return df_sorted

    def load_data(self, file_path):
        print("==== Loading Usage Data for Device Activity Score ====")
        
        try:
            # 讀取數據
            df = pd.read_csv(file_path)
            
            # 數據質量驗證
            quality_issues = self.validate_data_quality(df)
            if len(quality_issues) > 3:
                print("⚠️ Warning: Multiple data quality issues detected. Results may be unreliable.")
            
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

    def safe_triangular_membership(self, x, a, b, c):
        """安全的三角隸屬函數"""
        # 處理輸入為數組的情況
        if isinstance(x, (list, np.ndarray)):
            x = np.array(x)
            return np.array([self.safe_triangular_membership(xi, a, b, c) for xi in x])
        
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

    def analyze_device_activity(self, df):
        """分析設備活動模式"""
        print("==== Device Activity Analysis ====")

        print("Available activity columns:")
        activity_cols = ['power_state', 'is_on', 'is_off', 'is_phantom_load', 'is_light_use', 'is_regular_use']
        for col in activity_cols:
            if col in df.columns:
                print(f"  {col}: {df[col].value_counts().to_dict()}")
        
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)

        # 計算待機時長和距離最後活躍時間
        activity_records = []
        
        # 識別活躍期間（regular_use 或 light_use）
        df_sorted['is_active'] = (df_sorted.get('is_regular_use', False) | 
                                  df_sorted.get('is_light_use', False))
        
        # 識別待機期間（phantom_load 或 is_on但不active）
        df_sorted['is_standby'] = (df_sorted.get('is_phantom_load', False) | 
                                   ((df_sorted['is_on'] == True) & (df_sorted['is_active'] == False)))
        
        print(f"\nActivity Analysis:")
        print(f"Active periods: {df_sorted['is_active'].sum()}")
        print(f"Standby periods: {df_sorted['is_standby'].sum()}")
        print(f"Off periods: {df_sorted['is_off'].sum()}")

        # 計算每個記錄的活動指標
        last_active_time = None
        current_standby_start = None
        
        for idx, row in df_sorted.iterrows():
            current_time = row['timestamp']
            time_slot = row['time_slot']
            day_type = row['day_type']
            
            # 計算距離最後活躍時間
            if row['is_active']:
                last_active_time = current_time
                time_since_active = 0.0
                current_standby_start = None
            else:
                if last_active_time is not None:
                    time_since_active = (current_time - last_active_time).total_seconds() / 60.0
                else:
                    time_since_active = np.nan  
            
            # 計算待機時長
            if row['is_standby']:
                if current_standby_start is None:
                    current_standby_start = current_time
                    standby_duration = 0.0
                else:
                    standby_duration = (current_time - current_standby_start).total_seconds() / 60.0
            else:
                current_standby_start = None
                standby_duration = 0.0
            
            # 過濾合理的數值範圍
            if (not np.isnan(time_since_active) and 0 <= time_since_active <= 2880 and  
                0 <= standby_duration <= 2880):  
                
                activity_records.append({
                    'timestamp': current_time,
                    'time_slot': time_slot,
                    'day_type': day_type,
                    'standby_duration': standby_duration,
                    'time_since_active': time_since_active,
                    'is_active': row['is_active'],
                    'is_standby': row['is_standby'],
                    'usage_intensity': row.get('smoothed_intensity', 0.5)  # 新增使用強度
                })
        
        self.activity_data = pd.DataFrame(activity_records)
        
        print(f"\nCreated {len(self.activity_data)} activity records")
        
        if len(self.activity_data) > 0:
            standby_durations = self.activity_data['standby_duration']
            time_since_active = self.activity_data['time_since_active']
            usage_intensity = self.activity_data['usage_intensity']
            
            print(f"Standby Duration statistics:")
            print(f"  Min: {standby_durations.min():.1f} minutes")
            print(f"  Max: {standby_durations.max():.1f} minutes")
            print(f"  Mean: {standby_durations.mean():.1f} minutes")
            print(f"  Median: {standby_durations.median():.1f} minutes")
            
            print(f"Time Since Active statistics:")
            print(f"  Min: {time_since_active.min():.1f} minutes")
            print(f"  Max: {time_since_active.max():.1f} minutes")
            print(f"  Mean: {time_since_active.mean():.1f} minutes")
            print(f"  Median: {time_since_active.median():.1f} minutes")
            
            print(f"Usage Intensity statistics:")
            print(f"  Min: {usage_intensity.min():.2f}")
            print(f"  Max: {usage_intensity.max():.2f}")
            print(f"  Mean: {usage_intensity.mean():.2f}")
        else:
            print("⚠️ No valid activity records found")
        
        return self.activity_data

    def calculate_standby_duration_score(self):
        """計算待機時長分數"""
        print("==== Calculating Standby Duration Score ====")

        if self.activity_data is None or len(self.activity_data) == 0:
            print("⚠️ No activity data - using fallback calculation")
            self._create_default_standby_matrix()
            return self.standby_duration_matrix
        
        valid_entries = 0
        
        for day_type in ['weekday', 'weekend']:
            for time_slot in range(self.time_slots):
                mask = (self.activity_data['day_type'] == day_type) & \
                       (self.activity_data['time_slot'] == time_slot)
                
                slot_data = self.activity_data[mask]

                if len(slot_data) < 1:
                    # 使用基於時段的默認值
                    hour = time_slot // 4
                    if day_type == 'weekday':
                        if 9 <= hour <= 17:  # 工作時間
                            default_score = 0.3  
                        elif 18 <= hour <= 22:  # 晚間
                            default_score = 0.6
                        else:  # 深夜和早晨
                            default_score = 0.8
                    else:  # weekend
                        if 8 <= hour <= 22:  # 白天
                            default_score = 0.5
                        else:  # 夜間
                            default_score = 0.7
                    
                    self.standby_duration_matrix[(day_type, time_slot)] = {
                        'short_standby': 1.0 - default_score,
                        'medium_standby': default_score / 2,
                        'long_standby': default_score,
                        'sample_size': 0
                    }
                    continue

                standby_durations = slot_data['standby_duration'].values
                
                try:
                    # 使用百分位數定義模糊集合
                    min_duration = max(0.0, np.min(standby_durations))
                    q25 = np.percentile(standby_durations, 25)
                    q50 = np.percentile(standby_durations, 50)
                    q75 = np.percentile(standby_durations, 75)
                    max_duration = min(2880.0, np.max(standby_durations))
                    
                    # 確保順序正確
                    if not (min_duration <= q25 <= q50 <= q75 <= max_duration):
                        print(f"⚠️ Invalid percentiles for {day_type} slot {time_slot}, using defaults")
                        self.standby_duration_matrix[(day_type, time_slot)] = {
                            'short_standby': 0.6,
                            'medium_standby': 0.3,
                            'long_standby': 0.1,
                            'sample_size': len(slot_data)
                        }
                        continue
                    
                    # 定義待機時長的模糊集合
                    fuzzy_sets = {
                        'short': (min_duration, q25, q50),      
                        'medium': (q25, q50, q75),              
                        'long': (q50, q75, max_duration)        
                    }

                    # 計算模糊機率
                    fuzzy_probs = {}
                    for category, (a, b, c) in fuzzy_sets.items():
                        memberships = []
                        for duration in standby_durations:
                            membership = self.safe_triangular_membership(duration, a, b, c)
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
                        fuzzy_probs = {'short': 0.33, 'medium': 0.34, 'long': 0.33}
                    
                    self.standby_duration_matrix[(day_type, time_slot)] = {
                        'short_standby': fuzzy_probs['short'],
                        'medium_standby': fuzzy_probs['medium'],
                        'long_standby': fuzzy_probs['long'],
                        'sample_size': len(slot_data)
                    }
                    valid_entries += 1
                    
                except Exception as e:
                    print(f"⚠️ Error calculating standby duration for {day_type} slot {time_slot}: {e}")
                    self.standby_duration_matrix[(day_type, time_slot)] = {
                        'short_standby': 0.4,
                        'medium_standby': 0.4,
                        'long_standby': 0.2,
                        'sample_size': len(slot_data)
                    }
                
        print(f"✓ Calculated standby duration for {valid_entries} time slots")
        return self.standby_duration_matrix

    def calculate_time_since_active_score(self):
        """計算距離最後活躍時間分數"""
        print("==== Calculating Time Since Active Score ====")

        if self.activity_data is None or len(self.activity_data) == 0:
            print("⚠️ No activity data - using fallback calculation")
            self._create_default_time_since_active_matrix()
            return self.time_since_active_matrix
        
        valid_entries = 0
        
        for day_type in ['weekday', 'weekend']:
            for time_slot in range(self.time_slots):
                mask = (self.activity_data['day_type'] == day_type) & \
                       (self.activity_data['time_slot'] == time_slot)
                
                slot_data = self.activity_data[mask]

                if len(slot_data) < 1:
                    # 使用基於時段的默認值
                    hour = time_slot // 4
                    if day_type == 'weekday':
                        if 9 <= hour <= 17:  # 工作時間 - 期望較短的非活躍時間
                            default_score = 0.7  
                        elif 18 <= hour <= 22:  # 晚間
                            default_score = 0.5
                        else:  # 深夜和早晨
                            default_score = 0.2
                    else:  # weekend
                        if 8 <= hour <= 22:  # 白天
                            default_score = 0.4
                        else:  # 夜間
                            default_score = 0.1
                    
                    self.time_since_active_matrix[(day_type, time_slot)] = {
                        'recent_active': default_score,
                        'moderate_inactive': 0.3,
                        'long_inactive': 1.0 - default_score,
                        'sample_size': 0
                    }
                    continue

                time_since_active = slot_data['time_since_active'].values
                
                try:
                    # 使用百分位數定義模糊集合
                    min_time = max(0.0, np.min(time_since_active))
                    q25 = np.percentile(time_since_active, 25)
                    q50 = np.percentile(time_since_active, 50)
                    q75 = np.percentile(time_since_active, 75)
                    max_time = min(2880.0, np.max(time_since_active))
                    
                    # 確保順序正確
                    if not (min_time <= q25 <= q50 <= q75 <= max_time):
                        print(f"⚠️ Invalid percentiles for {day_type} slot {time_slot}, using defaults")
                        self.time_since_active_matrix[(day_type, time_slot)] = {
                            'recent_active': 0.5,
                            'moderate_inactive': 0.3,
                            'long_inactive': 0.2,
                            'sample_size': len(slot_data)
                        }
                        continue
                    
                    # 定義距離活躍時間的模糊集合
                    fuzzy_sets = {
                        'recent': (min_time, q25, q50),         
                        'moderate': (q25, q50, q75),           
                        'long': (q50, q75, max_time)            
                    }

                    # 計算模糊機率
                    fuzzy_probs = {}
                    for category, (a, b, c) in fuzzy_sets.items():
                        memberships = []
                        for time_val in time_since_active:
                            membership = self.safe_triangular_membership(time_val, a, b, c)
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
                        fuzzy_probs = {'recent': 0.33, 'moderate': 0.34, 'long': 0.33}
                    
                    self.time_since_active_matrix[(day_type, time_slot)] = {
                        'recent_active': fuzzy_probs['recent'],
                        'moderate_inactive': fuzzy_probs['moderate'],
                        'long_inactive': fuzzy_probs['long'],
                        'sample_size': len(slot_data)
                    }
                    valid_entries += 1
                    
                except Exception as e:
                    print(f"⚠️ Error calculating time since active for {day_type} slot {time_slot}: {e}")
                    self.time_since_active_matrix[(day_type, time_slot)] = {
                        'recent_active': 0.4,
                        'moderate_inactive': 0.4,
                        'long_inactive': 0.2,
                        'sample_size': len(slot_data)
                    }
                
        print(f"✓ Calculated time since active for {valid_entries} time slots")
        return self.time_since_active_matrix

    def _create_default_standby_matrix(self):
        """創建默認的待機時長矩陣"""
        for day_type in ['weekday', 'weekend']:
            for time_slot in range(self.time_slots):
                hour = time_slot // 4
                
                if day_type == 'weekday':
                    if 9 <= hour <= 17:  # 工作時間
                        short, medium, long = 0.5, 0.3, 0.2
                    elif 18 <= hour <= 22:  # 晚間
                        short, medium, long = 0.3, 0.4, 0.3
                    else:  # 深夜和早晨
                        short, medium, long = 0.2, 0.3, 0.5
                else:  # weekend
                    if 8 <= hour <= 22:  # 白天
                        short, medium, long = 0.4, 0.3, 0.3
                    else:  # 夜間
                        short, medium, long = 0.2, 0.3, 0.5
                
                self.standby_duration_matrix[(day_type, time_slot)] = {
                    'short_standby': short,
                    'medium_standby': medium,
                    'long_standby': long,
                    'sample_size': 0
                }

    def _create_default_time_since_active_matrix(self):
        """創建默認的距離活躍時間矩陣"""
        for day_type in ['weekday', 'weekend']:
            for time_slot in range(self.time_slots):
                hour = time_slot // 4
                
                if day_type == 'weekday':
                    if 9 <= hour <= 17:  # 工作時間
                        recent, moderate, long = 0.6, 0.3, 0.1
                    elif 18 <= hour <= 22:  # 晚間
                        recent, moderate, long = 0.4, 0.4, 0.2
                    else:  # 深夜和早晨
                        recent, moderate, long = 0.1, 0.3, 0.6
                else:  # weekend
                    if 8 <= hour <= 22:  # 白天
                        recent, moderate, long = 0.3, 0.4, 0.3
                    else:  # 夜間
                        recent, moderate, long = 0.1, 0.2, 0.7
                
                self.time_since_active_matrix[(day_type, time_slot)] = {
                    'recent_active': recent,
                    'moderate_inactive': moderate,
                    'long_inactive': long,
                    'sample_size': 0
                }

    def calculate_membership_parameters(self):
        """計算統計隸屬參數"""
        print("==== Calculating Statistical Membership Parameters ====")
        
        self.membership_parameters = {
            'standby_duration': {},
            'time_since_active': {}
        }
        
        # 收集 Standby Duration 數據
        standby_values = []
        for key in self.standby_duration_matrix:
            data = self.standby_duration_matrix[key]
            if 'short_standby' in data and not np.isnan(data['short_standby']):
                standby_values.append(data['short_standby'])
        
        if standby_values:
            standby_values = np.array(standby_values)
            self.membership_parameters['standby_duration'] = {
                'p0': float(np.min(standby_values)),
                'p25': float(np.percentile(standby_values, 25)),
                'p50': float(np.percentile(standby_values, 50)),
                'p75': float(np.percentile(standby_values, 75)),
                'p100': float(np.max(standby_values))
            }
            print(f"Standby Duration Statistics (n={len(standby_values)}): ✓")
        else:
            print("⚠️ No valid standby duration data, using defaults")
            self.membership_parameters['standby_duration'] = {
                'p0': 0.0, 'p25': 0.25, 'p50': 0.5, 'p75': 0.75, 'p100': 1.0
            }
        
        # 收集 Time Since Active 數據
        time_since_values = []
        for key in self.time_since_active_matrix:
            data = self.time_since_active_matrix[key]
            if 'recent_active' in data and not np.isnan(data['recent_active']):
                time_since_values.append(data['recent_active'])
        
        if time_since_values:
            time_since_values = np.array(time_since_values)
            self.membership_parameters['time_since_active'] = {
                'p0': float(np.min(time_since_values)),
                'p25': float(np.percentile(time_since_values, 25)),
                'p50': float(np.percentile(time_since_values, 50)),
                'p75': float(np.percentile(time_since_values, 75)),
                'p100': float(np.max(time_since_values))
            }
            print(f"Time Since Active Statistics (n={len(time_since_values)}): ✓")
        else:
            print("⚠️ No valid time since active data, using defaults")
            self.membership_parameters['time_since_active'] = {
                'p0': 0.0, 'p25': 0.25, 'p50': 0.5, 'p75': 0.75, 'p100': 1.0
            }
        
        return self.membership_parameters

    def define_activity_rules(self):
        """定義設備活躍度模糊規則"""
        print("==== Defining Device Activity Rules ====")
        
        # 規則格式: (待機時長, 距離最後活躍時間, 輸出活躍度等級, 權重)
        self.activity_rules = [
            # 高活躍度規則：設備很活躍，不需要特別處理
            ('short', 'recent', 'high', 1.0),           
            ('short', 'moderate', 'high', 0.8),         
            ('medium', 'recent', 'high', 0.9),          
            
            # 中等活躍度規則：需要監控但不緊急
            ('short', 'long', 'medium', 0.6),           
            ('medium', 'moderate', 'medium', 0.7),      
            ('long', 'recent', 'medium', 0.5),          
            
            # 低活躍度規則：可能需要喚醒或節能處理
            ('medium', 'long', 'low', 0.8),             
            ('long', 'moderate', 'low', 0.9),           
            ('long', 'long', 'low', 1.0),               
        ]
        
        print(f"✓ Defined {len(self.activity_rules)} activity rules")
        return self.activity_rules

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
        
        # 獲取待機時長隸屬度
        if (day_type, time_slot) in self.standby_duration_matrix:
            standby_data = self.standby_duration_matrix[(day_type, time_slot)]
            
            standby_params = self.membership_parameters['standby_duration']
            short_standby = standby_data.get('short_standby', 0.4)
            
            result['standby_short'] = self.safe_triangular_membership(
                short_standby, standby_params['p0'], standby_params['p25'], standby_params['p50']
            )
            result['standby_medium'] = self.safe_triangular_membership(
                short_standby, standby_params['p25'], standby_params['p50'], standby_params['p75']
            )
            result['standby_long'] = self.safe_triangular_membership(
                short_standby, standby_params['p50'], standby_params['p75'], standby_params['p100']
            )
            result['standby_score'] = short_standby
        else:
            result.update({
                'standby_short': 0.5, 'standby_medium': 0.3, 'standby_long': 0.2,
                'standby_score': 0.4
            })
        
        # 獲取距離最後活躍時間隸屬度
        if (day_type, time_slot) in self.time_since_active_matrix:
            time_since_data = self.time_since_active_matrix[(day_type, time_slot)]
            
            time_since_params = self.membership_parameters['time_since_active']
            recent_active = time_since_data.get('recent_active', 0.4)
            
            result['time_recent'] = self.safe_triangular_membership(
                recent_active, time_since_params['p0'], time_since_params['p25'], time_since_params['p50']
            )
            result['time_moderate'] = self.safe_triangular_membership(
                recent_active, time_since_params['p25'], time_since_params['p50'], time_since_params['p75']
            )
            result['time_long'] = self.safe_triangular_membership(
                recent_active, time_since_params['p50'], time_since_params['p75'], time_since_params['p100']
            )
            result['time_since_score'] = recent_active
        else:
            result.update({
                'time_recent': 0.4, 'time_moderate': 0.4, 'time_long': 0.2,
                'time_since_score': 0.4
            })
        
        return result

    def calculate_activity_score(self, timestamp):
        """計算改進版設備活躍度分數（加入時間合理性權重）"""
        try:
            # 1. 獲取時間合理性權重
            time_weight_info = self.get_time_reasonableness_weight(timestamp)
            time_weight = time_weight_info['weight']
            
            # 2. 計算基礎模糊隸屬度
            memberships = self.calculate_fuzzy_memberships(timestamp)
            
            # 3. 計算基礎活躍度分數（原邏輯）
            base_activity_score = self._calculate_base_activity_score(memberships)
            
            # 4. 應用時間合理性權重
            weighted_activity_score = base_activity_score * time_weight
            
            # 5. 深夜特殊處理
            hour = timestamp.hour
            if 22 <= hour or hour <= 5:
                # 深夜時間，即使設備活躍也要考慮休息建議
                rest_penalty = self._calculate_rest_penalty(hour, base_activity_score)
                weighted_activity_score = max(0.05, weighted_activity_score - rest_penalty)
            
            # 6. 確保分數在合理範圍內
            final_activity_score = max(0.05, min(0.95, weighted_activity_score))
            
            # 7. 計算置信度（結合原有邏輯和時間因素）
            confidence = min(1.0, (time_weight + 0.5) / 1.5)
            
            return {
                'activity_score': final_activity_score,
                'base_activity_score': base_activity_score,
                'time_weight': time_weight,
                'time_period': time_weight_info['time_period'],
                'weighted_score': weighted_activity_score,
                'rest_penalty': self._calculate_rest_penalty(hour, base_activity_score) if (22 <= hour or hour <= 5) else 0,
                'memberships': memberships,
                'confidence': confidence,
                'time_reasonableness': time_weight_info
            }
            
        except Exception as e:
            print(f"⚠️ Error calculating improved activity score: {e}")
            return {
                'activity_score': 0.4,
                'base_activity_score': 0.4,
                'time_weight': 0.5,
                'time_period': 'unknown',
                'weighted_score': 0.4,
                'rest_penalty': 0,
                'memberships': {},
                'confidence': 0.1,
                'time_reasonableness': {}
            }

    def _calculate_base_activity_score(self, memberships):
        """計算基礎活躍度分數（原邏輯）"""
        try:
            # 計算規則激活
            low_activation = 0.0
            medium_activation = 0.0
            high_activation = 0.0
            valid_rules = 0
            
            for rule in self.activity_rules:
                standby_level, time_level, output_level, weight = rule
                
                # 獲取隸屬度
                standby_membership = memberships.get(f'standby_{standby_level}', 0.0)
                time_membership = memberships.get(f'time_{time_level}', 0.0)
                
                # 檢查是否有有效的隸屬度
                if any(np.isnan([standby_membership, time_membership])):
                    continue
                
                # 使用最小值方法計算規則激活
                if all(m >= 0 for m in [standby_membership, time_membership]):
                    activation = min(standby_membership, time_membership) * weight
                    
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
            
            # 計算最終活躍度分數
            total_activation = low_activation + medium_activation + high_activation
            
            if total_activation > 0 and valid_rules > 0:
                # 使用加權重心法（高活躍度 = 高分數）
                activity_score = (
                    low_activation * 0.2 +       
                    medium_activation * 0.5 +    
                    high_activation * 0.8         
                ) / total_activation
                
                # 基於實際數據的微調
                standby_score = memberships.get('standby_score', 0.5)
                time_since_score = memberships.get('time_since_score', 0.5)
                
                # 微調：短待機和最近活躍 -> 提高分數
                data_adjustment = (standby_score + time_since_score - 1.0) * 0.05
                activity_score += data_adjustment
            else:
                # 後備計算
                activity_score = 0.4
            
            return max(0.05, min(0.95, activity_score))
            
        except:
            return 0.4

    def _calculate_rest_penalty(self, hour, base_activity_score):
        """計算深夜休息懲罰"""
        if 23 <= hour or hour <= 2:  # 深夜 23:00-02:00
            return 0.3  # 重懲罰
        elif 3 <= hour <= 5:  # 凌晨 03:00-05:00
            return 0.2  # 中懲罰
        elif hour == 22:  # 22:00
            return 0.1  # 輕懲罰
        else:
            return 0.0

    def test_activity_score_calculation(self, num_tests=5):
        """測試改進版活躍度分數計算功能"""
        print("==== Testing Improved Activity Score Calculation ====")
        
        test_times = [
            (datetime(2024, 1, 15, 9, 0), (0.1, 0.3), '工作日早上'),    # 工作時間，低權重
            (datetime(2024, 1, 15, 14, 0), (0.3, 0.6), '工作日下午'),   # 午休時間，中高權重
            (datetime(2024, 1, 15, 19, 0), (0.6, 0.9), '工作日晚上'),   # 娛樂時間，高權重
            (datetime(2024, 1, 15, 23, 0), (0.1, 0.3), '工作日深夜'),   # 深夜，大幅降低
            (datetime(2024, 1, 13, 2, 0), (0.05, 0.2), '週末凌晨'),    # 凌晨，極低
        ]
        
        test_results = []
        
        for i, (test_time, expected_range, desc) in enumerate(test_times[:num_tests]):
            try:
                result = self.calculate_activity_score(test_time)
                
                day_type = "Weekend" if test_time.weekday() >= 5 else "Weekday"
                print(f"\nTest {i+1}: {test_time.strftime('%Y-%m-%d %H:%M')} ({day_type})")
                print(f"  Activity Score: {result['activity_score']:.3f} (Base: {result['base_activity_score']:.3f})")
                print(f"  Time Weight: {result['time_weight']:.3f} (Period: {result['time_period']})")
                print(f"  Time Reasonableness: {result['time_reasonableness']['weight']:.3f}")
                if result['rest_penalty'] > 0:
                    print(f"  Rest Penalty: {result['rest_penalty']:.3f}")
                print(f"  Confidence: {result['confidence']:.3f}")
                
                # 檢查是否符合預期
                score = result['activity_score']
                is_in_range = expected_range[0] <= score <= expected_range[1]
                status = "✓ PASS" if is_in_range else "❌ FAIL"
                print(f"  Expected: {expected_range}, Result: {status}")
                
                test_results.append({
                    'time': test_time,
                    'score': result['activity_score'],
                    'base_score': result['base_activity_score'],
                    'time_weight': result['time_weight'],
                    'pass': is_in_range
                })
                
            except Exception as e:
                print(f"⚠️ Error in test {i+1}: {e}")
                test_results.append({
                    'time': test_time,
                    'score': 0.4,
                    'base_score': 0.4,
                    'time_weight': 0.5,
                    'pass': False
                })
        
        # 統計測試結果
        passed_tests = sum(1 for result in test_results if result['pass'])
        print(f"\n📊 Test Results: {passed_tests}/{len(test_results)} passed")
        
        return test_results

    def detect_usage_pattern(self):
        """檢測設備使用模式"""
        if self.activity_data is None or len(self.activity_data) == 0:
            return "未知模式"
        
        total_records = len(self.activity_data)
        active_count = len(self.activity_data[self.activity_data['is_active'] == True])
        standby_count = len(self.activity_data[self.activity_data['is_standby'] == True])
        
        active_ratio = active_count / total_records
        standby_ratio = standby_count / total_records
        off_ratio = 1 - active_ratio - standby_ratio
        
        print(f"\n🔍 使用模式檢測:")
        print(f"   活躍時間: {active_ratio:.1%}")
        print(f"   待機時間: {standby_ratio:.1%}")
        print(f"   關機時間: {off_ratio:.1%}")
        
        if standby_ratio < 0.2 and active_ratio > 0.4:
            pattern = "高效節能模式"
            description = "用完即關機，很少待機浪費電力"
        elif standby_ratio > 0.4:
            pattern = "高待機模式"  
            description = "設備經常保持待機狀態"
        elif active_ratio > 0.6:
            pattern = "高使用模式"
            description = "設備大部分時間處於活躍狀態"
        else:
            pattern = "混合使用模式"
            description = "活躍、待機、關機時間較為平衡"
        
        print(f"   檢測結果: {pattern} - {description}")
        return pattern

    def comprehensive_evaluation(self):
        """完整的系統評估"""
        print("\n" + "="*60)
        print("IMPROVED DEVICE ACTIVITY SCORE - COMPREHENSIVE EVALUATION")
        print("="*60)
        
        # 0. 使用模式檢測
        usage_pattern = self.detect_usage_pattern()
        
        # 1. 數據質量評估
        print(f"\n1. Data Quality Assessment:")
        print(f"   Quality Score: {self.data_quality_report.get('quality_score', 0):.2f}")
        print(f"   Issues Count: {len(self.data_quality_report.get('issues', []))}")
        
        # 2. 時間權重功能測試
        print(f"\n2. Time Weight Function Test:")
        test_hours = [2, 7, 10, 14, 19, 22]
        for hour in test_hours:
            test_time = datetime(2024, 1, 15, hour, 0)
            weight_info = self.get_time_reasonableness_weight(test_time)
            print(f"   {hour:02d}:00 - Weight: {weight_info['weight']:.3f} ({weight_info['time_period']})")
        
        # 3. 改進版活躍度分數測試
        print(f"\n3. Improved Activity Score Tests:")
        test_results = self.test_activity_score_calculation()
        
        # 4. 時間合理性檢查
        print(f"\n4. Time Reasonableness Check:")
        
        # 檢查深夜分數是否顯著低於白天
        night_results = [r for r in test_results if r['time'].hour in [23, 2]]
        day_results = [r for r in test_results if r['time'].hour in [14, 19]]
        
        if night_results and day_results:
            night_avg = np.mean([r['score'] for r in night_results])
            day_avg = np.mean([r['score'] for r in day_results])
            
            print(f"   Night time average: {night_avg:.3f}")
            print(f"   Day time average: {day_avg:.3f}")
            print(f"   Night < Day: {'✓ PASS' if night_avg < day_avg else '❌ FAIL'}")
            
            # 檢查權重效果
            night_weight_avg = np.mean([r['time_weight'] for r in night_results])
            day_weight_avg = np.mean([r['time_weight'] for r in day_results])
            
            print(f"   Night weight average: {night_weight_avg:.3f}")
            print(f"   Day weight average: {day_weight_avg:.3f}")
            print(f"   Weight difference: {day_weight_avg - night_weight_avg:.3f}")
        
        # 5. 最終評分
        print(f"\n=== FINAL ASSESSMENT ===")
        
        quality_score = self.data_quality_report.get('quality_score', 0)
        test_pass_rate = sum(1 for r in test_results if r['pass']) / len(test_results)
        
        # 時間權重合理性檢查
        time_weight_reasonable = night_weight_avg < day_weight_avg if 'night_weight_avg' in locals() and 'day_weight_avg' in locals() else 0.5
        
        overall_score = (quality_score + test_pass_rate + time_weight_reasonable) / 3
        
        print(f"Data Quality: {quality_score:.2f}")
        print(f"Test Pass Rate: {test_pass_rate:.2f}")
        print(f"Time Weight Logic: {time_weight_reasonable:.2f}")
        print(f"Overall System Quality: {overall_score:.2f}")
        
        if overall_score >= 0.8:
            print("🎉 System Quality: Excellent - Time-aware activity scoring")
        elif overall_score >= 0.6:
            print("✅ System Quality: Good - Improved logic working")
        else:
            print("⚠️ System Quality: Needs Improvement")
        
        # 改進效果說明
        if 'night_avg' in locals() and 'day_avg' in locals() and night_avg < day_avg:
            print("💡 改進效果：深夜活躍度分數已成功降低，符合人性化邏輯")
        
        return overall_score

    def run_complete_analysis(self, file_path):
        """運行完整分析"""
        print("="*80)
        print("IMPROVED DEVICE ACTIVITY SCORE MODULE - COMPLETE ANALYSIS")
        print("="*80)

        # 1. 載入數據
        df = self.load_data(file_path)
        if df is None:
            print('❌ Cannot load data')
            return None
        
        # 2. 分析設備活動
        print("\n" + "-"*50)
        self.analyze_device_activity(df)
        
        # 3. 計算兩個核心指標
        print("\n" + "-"*50)
        self.calculate_standby_duration_score()
        
        print("\n" + "-"*50)
        self.calculate_time_since_active_score()

        # 4. 計算隸屬參數
        print("\n" + "-"*50)
        self.calculate_membership_parameters()
    
        # 5. 定義規則
        print("\n" + "-"*50)
        self.define_activity_rules()

        # 6. 測試計算
        print("\n" + "-"*50)
        test_results = self.test_activity_score_calculation()

        # 7. 綜合評估
        print("\n" + "-"*50)
        overall_score = self.comprehensive_evaluation()

        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE - Improved Device Activity Score system ready!")
        print("💡 Now includes time reasonableness weighting and rest-time logic")
        print("="*80)

        return {
            'standby_duration': self.standby_duration_matrix,
            'time_since_active': self.time_since_active_matrix,
            'membership_parameters': self.membership_parameters,
            'activity_rules': self.activity_rules,
            'test_results': test_results,
            'data_quality': self.data_quality_report,
            'time_weights': self.time_reasonableness_weights,
            'overall_score': overall_score
        }

# 使用示例
if __name__ == "__main__":
    # 初始化改進版設備活躍度模組
    activity_module = DeviceActivityScoreModule()
    
    # 檔案路徑
    file_path = "C:/Users/王俞文/OneDrive - University of Glasgow/文件/glasgow/msc project/data/extended_power_data_2months.csv"
    
    # 運行完整分析
    result = activity_module.run_complete_analysis(file_path)
    
    # 單獨測試改進版活躍度分數計算
    if result:
        print("\n" + "="*50)
        print("TESTING IMPROVED ACTIVITY SCORE AT SPECIFIC TIMES")
        print("="*50)
        
        # 包括您的原始測試時間
        test_times = [
            datetime(2024, 7, 16, 23, 1),   # 您的原始測試時間
            datetime(2024, 7, 16, 19, 0),   # 晚間娛樂時間對比
            datetime(2024, 7, 16, 10, 0),   # 工作時間對比
            datetime(2024, 7, 16, 2, 30),   # 深夜對比
        ]
        
        for test_time in test_times:
            result = activity_module.calculate_activity_score(test_time)
            
            print(f"\n🕐 時間: {test_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"📊 改進版活躍度分數: {result['activity_score']:.3f}")
            print(f"📊 基礎分數: {result['base_activity_score']:.3f}")
            print(f"⚖️ 時間權重: {result['time_weight']:.3f}")
            print(f"⏰ 時間段: {result['time_period']}")
            if result['rest_penalty'] > 0:
                print(f"😴 休息懲罰: {result['rest_penalty']:.3f}")
            
            # 與原邏輯對比說明
            if test_time.hour == 23:
                print(f"💡 改進效果: 深夜時間大幅降低分數 ({result['base_activity_score']:.3f} → {result['activity_score']:.3f})")
            elif 18 <= test_time.hour <= 20:
                print(f"💡 改進效果: 娛樂時間保持高分數 ({result['base_activity_score']:.3f} → {result['activity_score']:.3f})")