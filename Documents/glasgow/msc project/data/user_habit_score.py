import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class UserHabitScoreModule:
    
    def __init__(self):
        self.time_slots = 96  # 96個時段，每15分鐘一個 (24*4)
        self.transition_data = None
        self.usage_probability_matrix = {}
        self.stability_matrix = {}
        self.time_factor_matrix = {}
        self.habit_rules = []
        
    def load_usage_data(self, usage_file_path):
        """載入使用記錄數據"""
        print("==== Loading Usage Data for User Habit Score ====")
        
        if usage_file_path is None:
            print(f"Cannot find {usage_file_path}")
            return None
            
        # 讀取數據
        df = pd.read_csv(usage_file_path)
        
        # 轉換時間格式
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 新增時間特徵
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['weekday'] = df['timestamp'].dt.weekday
        df['is_weekend'] = df['weekday'] >= 5
        df['day_type'] = df['is_weekend'].map({True: 'weekend', False: 'weekday'})
        df['time_slot'] = df['hour'] * 4 + df['minute'] // 15  # 0-95個時段，每15分鐘一個
        
        print(f"Loaded usage data: {len(df)} records")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Weekdays: {sum(~df['is_weekend'])}, Weekends: {sum(df['is_weekend'])}")
        
        return df
    
    def extract_state_transitions(self, df):
        """提取狀態轉換事件"""
        print("==== Extracting State Transitions ====")
        
        # 檢查數據中的實際狀態欄位
        print("Available state columns:")
        state_cols = ['power_state', 'is_on', 'is_off', 'is_phantom_load', 'is_light_use', 'is_regular_use']
        for col in state_cols:
            if col in df.columns:
                print(f"  {col}: {df[col].value_counts().to_dict()}")
        
        print(f"\nPower state values: {df['power_state'].unique()}")
        if 'is_on' in df.columns:
            print(f"Is_on values: {df['is_on'].unique()}")
        if 'is_off' in df.columns:
            print(f"Is_off values: {df['is_off'].unique()}")

        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        
        # 檢查是否有真正的開關機狀態
        has_real_onoff = False
        turn_off_events = pd.DataFrame()
        turn_on_events = pd.DataFrame()
        
        # 方法1：檢查 is_on/is_off 欄位
        if 'is_on' in df.columns:
            df_sorted['is_on_prev'] = df_sorted['is_on'].shift(1)
            
            # 轉換：is_on=True -> is_on=False (關機)
            turn_off_events = df_sorted[
                (df_sorted['is_on_prev'] == True) & 
                (df_sorted['is_on'] == False)
            ].copy()

            # 轉換：is_on=False -> is_on=True (開機)
            turn_on_events = df_sorted[
                (df_sorted['is_on_prev'] == False) & 
                (df_sorted['is_on'] == True)
            ].copy()
            
            print(f"\nMethod 1 - Real on/off transitions:")
            print(f"Found {len(turn_off_events)} turn-off events (True->False)")
            print(f"Found {len(turn_on_events)} turn-on events (False->True)")
            
            if len(turn_off_events) > 0 or len(turn_on_events) > 0:
                has_real_onoff = True
        
        # 方法2：如果沒有真正的開關機，使用使用強度轉換
        if not has_real_onoff:
            print("\nNo real on/off transitions found. Using activity transitions...")
            
            # 定義使用強度：phantom load 視為 "low activity", light/regular use 視為 "active"
            df_sorted['is_active'] = (df_sorted['is_light_use'] == True) | (df_sorted['is_regular_use'] == True)
            df_sorted['is_active_prev'] = df_sorted['is_active'].shift(1)
            
            # 活動 -> 非活動 (相當於"關機")
            turn_off_events = df_sorted[
                (df_sorted['is_active_prev'] == True) & 
                (df_sorted['is_active'] == False)
            ].copy()

            # 非活動 -> 活動 (相當於"開機")  
            turn_on_events = df_sorted[
                (df_sorted['is_active_prev'] == False) & 
                (df_sorted['is_active'] == True)
            ].copy()
            
            print(f"Method 2 - Activity transitions:")
            print(f"Found {len(turn_off_events)} activity-off events (active->phantom)")
            print(f"Found {len(turn_on_events)} activity-on events (phantom->active)")
            
            if len(turn_off_events) > 0:
                print("Sample turn-off events:")
                sample_off = turn_off_events[['timestamp', 'power_state', 'is_active_prev', 'is_active']].head(3)
                print(sample_off)
            
            if len(turn_on_events) > 0:
                print("Sample turn-on events:")
                sample_on = turn_on_events[['timestamp', 'power_state', 'is_active_prev', 'is_active']].head(3)
                print(sample_on)

        # 建立轉換記錄
        transition_records = []
        for idx, turn_off in turn_off_events.iterrows():
            turn_off_time = turn_off['timestamp']
            turn_off_slot = turn_off['time_slot']
            turn_off_day_type = turn_off['day_type']
            
            # 找到這次"關機"後的第一個"開機"事件
            next_turn_on = turn_on_events[
                turn_on_events['timestamp'] > turn_off_time
            ]
            
            if len(next_turn_on) > 0:
                next_on_time = next_turn_on.iloc[0]['timestamp']
                interval_minutes = (next_on_time - turn_off_time).total_seconds() / 60.0
                
                transition_records.append({
                    'turn_off_time': turn_off_time,
                    'turn_on_time': next_on_time,
                    'interval_minutes': interval_minutes,
                    'time_slot': turn_off_slot,
                    'day_type': turn_off_day_type,
                    'turn_off_hour': turn_off_time.hour
                })
        
        self.transition_data = pd.DataFrame(transition_records)
        
        print(f"\nCreated {len(self.transition_data)} transition records")
        
        if len(self.transition_data) > 0:
            print(f"Interval statistics:")
            print(f"  Min: {self.transition_data['interval_minutes'].min():.1f} minutes")
            print(f"  Max: {self.transition_data['interval_minutes'].max():.1f} minutes")
            print(f"  Mean: {self.transition_data['interval_minutes'].mean():.1f} minutes")
            print(f"  Median: {self.transition_data['interval_minutes'].median():.1f} minutes")
            print(f"Sample transitions:")
            print(self.transition_data[['turn_off_time', 'turn_on_time', 'interval_minutes']].head(3))
        else:
            print("No valid transition pairs found!")
            print("This suggests:")
            print("  1. Device stayed consistently ON throughout the data period")
            print("  2. Device stayed consistently OFF throughout the data period") 
            print("  3. No activity transitions (phantom <-> active) found")
            print("Will use default probability values based on time-of-day patterns.")
        
        return self.transition_data
        
    def calculate_usage_probability(self):
        """計算使用機率矩陣"""
        print("==== Calculating Usage Probability ====")
        
        if self.transition_data is None or len(self.transition_data) == 0:
            print("No transition data available - using default probability values")
            
            # 使用默認值填充所有時段
            for day_type in ['weekday', 'weekend']:
                for time_slot in range(self.time_slots):
                    self.usage_probability_matrix[(day_type, time_slot)] = {
                        'immediate_prob': 0.15,
                        'short_prob': 0.35,
                        'medium_prob': 0.60,
                        'long_prob': 0.80,
                        'immediate_time': 15.0,
                        'short_time': 45.0,
                        'medium_time': 120.0,
                        'sample_size': 0
                    }
            
            print("Applied default probability values to all time slots")
            return self.usage_probability_matrix
        
        # 為每個(day_type, time_slot)組合計算使用機率
        for day_type in ['weekday', 'weekend']:
            for time_slot in range(self.time_slots):
                # 篩選該情境下的轉換數據
                mask = (self.transition_data['day_type'] == day_type) & \
                       (self.transition_data['time_slot'] == time_slot)
                
                slot_data = self.transition_data[mask]
                
                if len(slot_data) >= 3:  # 至少需要3個數據點才有意義
                    intervals = slot_data['interval_minutes'].values
                    
                    # 計算動態百分比劃分
                    percentiles = {
                        'immediate': np.percentile(intervals, 20),  # 20%分位數
                        'short': np.percentile(intervals, 50),     # 50%分位數  
                        'medium': np.percentile(intervals, 80),    # 80%分位數
                    }
                    
                    # 計算累積使用機率
                    immediate_prob = 0.20  # 20%的事件在immediate時間內
                    short_prob = 0.50      # 50%的事件在short時間內
                    medium_prob = 0.80     # 80%的事件在medium時間內
                    long_prob = 1.00       # 100%的事件在更長時間內
                    
                    self.usage_probability_matrix[(day_type, time_slot)] = {
                        'immediate_prob': immediate_prob,
                        'short_prob': short_prob,
                        'medium_prob': medium_prob,
                        'long_prob': long_prob,
                        'immediate_time': percentiles['immediate'],
                        'short_time': percentiles['short'],
                        'medium_time': percentiles['medium'],
                        'sample_size': len(slot_data)
                    }
                    
                    hour = time_slot // 4
                    minute = (time_slot % 4) * 15
                    print(f"{day_type} {hour:02d}:{minute:02d}: "
                          f"immediate={percentiles['immediate']:.1f}min, "
                          f"short={percentiles['short']:.1f}min, "
                          f"medium={percentiles['medium']:.1f}min "
                          f"(n={len(slot_data)})")
                else:
                    # 數據不足時使用默認值
                    self.usage_probability_matrix[(day_type, time_slot)] = {
                        'immediate_prob': 0.15,
                        'short_prob': 0.35,
                        'medium_prob': 0.60,
                        'long_prob': 0.80,
                        'immediate_time': 15.0,
                        'short_time': 45.0,
                        'medium_time': 120.0,
                        'sample_size': len(slot_data)
                    }
        
        return self.usage_probability_matrix
    
    def calculate_usage_stability(self, df):
        """計算使用穩定性矩陣"""
        print("==== Calculating Usage Stability ====")
        
        # 為每個(day_type, time_slot)組合計算穩定性
        for day_type in ['weekday', 'weekend']:
            for time_slot in range(self.time_slots):
                # 篩選該情境下的數據
                mask = (df['day_type'] == day_type) & (df['time_slot'] == time_slot)
                slot_data = df[mask]
                
                if len(slot_data) > 0:
                    # 計算時間一致性穩定性
                    time_stability = self._calculate_time_consistency(slot_data)
                    
                    # 計算使用時長穩定性
                    duration_stability = self._calculate_duration_consistency(slot_data)
                    
                    # 計算使用強度穩定性  
                    intensity_stability = self._calculate_intensity_consistency(slot_data)
                    
                    # 綜合穩定性
                    overall_stability = (
                        time_stability * 0.4 +
                        duration_stability * 0.3 +
                        intensity_stability * 0.3
                    )
                    
                    self.stability_matrix[(day_type, time_slot)] = {
                        'time_stability': time_stability,
                        'duration_stability': duration_stability,
                        'intensity_stability': intensity_stability,
                        'overall_stability': overall_stability,
                        'sample_size': len(slot_data)
                    }
                else:
                    # 數據不足時使用低穩定性
                    self.stability_matrix[(day_type, time_slot)] = {
                        'time_stability': 0.2,
                        'duration_stability': 0.2,
                        'intensity_stability': 0.2,
                        'overall_stability': 0.2,
                        'sample_size': 0
                    }
        
        return self.stability_matrix
    
    def _calculate_time_consistency(self, slot_data):
        """計算時間一致性"""
        if len(slot_data) < 2:
            return 0.2
        
        # 按日期分組，檢查每天該時段的使用情況
        daily_usage = slot_data.groupby(slot_data['timestamp'].dt.date)['is_on'].max()
        
        if len(daily_usage) < 2:
            return 0.5
        
        # 計算使用一致性
        usage_variance = daily_usage.var()
        consistency = 1.0 - min(usage_variance / 0.25, 1.0)  # 0.25是最大方差(50%使用率時)
        
        return max(0.0, min(1.0, consistency))
    
    def _calculate_duration_consistency(self, slot_data):
        """計算使用時長一致性"""
        if len(slot_data) < 2:
            return 0.2
        
        # 按日期分組，計算每天該時段的總使用時長
        daily_duration = slot_data.groupby(slot_data['timestamp'].dt.date)['time_diff_seconds'].sum() / 60.0
        daily_duration = daily_duration[daily_duration > 0]  # 只考慮有使用的天數
        
        if len(daily_duration) < 2:
            return 0.3
        
        # 計算變異係數
        cv = daily_duration.std() / daily_duration.mean() if daily_duration.mean() > 0 else 2.0
        stability = 1.0 / (1.0 + cv)
        
        return max(0.0, min(1.0, stability))
    
    def _calculate_intensity_consistency(self, slot_data):
        """計算使用強度一致性"""
        if len(slot_data) < 2:
            return 0.2
        
        # 定義使用強度
        slot_data = slot_data.copy()
        slot_data['intensity'] = 0
        slot_data.loc[slot_data['is_regular_use'] == 1, 'intensity'] = 3
        slot_data.loc[slot_data['is_light_use'] == 1, 'intensity'] = 2
        slot_data.loc[(slot_data['is_on'] == 1) & (slot_data['intensity'] == 0), 'intensity'] = 1
        
        # 按日期分組，計算每天該時段的平均強度
        daily_intensity = slot_data.groupby(slot_data['timestamp'].dt.date)['intensity'].mean()
        daily_intensity = daily_intensity[daily_intensity > 0]
        
        if len(daily_intensity) < 2:
            return 0.3
        
        # 計算強度變異
        intensity_variance = daily_intensity.var()
        max_variance = 2.25  # 最大理論方差(強度0-3的方差)
        stability = 1.0 - min(intensity_variance / max_variance, 1.0)
        
        return max(0.0, min(1.0, stability))
    
    def calculate_time_factor(self, df):
        """計算時間因子矩陣 - 每個15分鐘時段的使用率"""
        print("==== Calculating Time Factor (15-min slots) ====")
        
        # 為每個(day_type, time_slot)組合計算使用率
        for day_type in ['weekday', 'weekend']:
            print(f"\nCalculating {day_type} time factors...")
            
            for time_slot in range(self.time_slots):  # 0-95個時段
                # 篩選該情境下的數據
                mask = (df['day_type'] == day_type) & (df['time_slot'] == time_slot)
                slot_data = df[mask]
                
                if len(slot_data) > 0:
                    # 計算該時段的總體使用率
                    total_records = len(slot_data)
                    usage_records = len(slot_data[slot_data['is_on'] == 1])
                    usage_rate = usage_records / total_records if total_records > 0 else 0
                    
                    # 計算加權使用率(考慮使用強度)
                    weighted_usage = (
                        len(slot_data[slot_data['is_regular_use'] == 1]) * 1.0 +
                        len(slot_data[slot_data['is_light_use'] == 1]) * 0.6 +
                        len(slot_data[(slot_data['is_on'] == 1) & 
                                     (slot_data['is_regular_use'] == 0) & 
                                     (slot_data['is_light_use'] == 0)]) * 0.3
                    ) / total_records
                    
                    self.time_factor_matrix[(day_type, time_slot)] = {
                        'usage_rate': usage_rate,
                        'weighted_usage_rate': weighted_usage,
                        'sample_size': total_records
                    }
                    
                    # 只打印有明顯使用的時段，避免輸出過多
                    if usage_rate > 0.1 or time_slot % 16 == 0:  # 每4小時或有使用的時段才打印
                        hour = time_slot // 4
                        minute = (time_slot % 4) * 15
                        print(f"  {hour:02d}:{minute:02d} - usage_rate={usage_rate:.3f}, "
                              f"weighted_rate={weighted_usage:.3f} (n={total_records})")
                else:
                    self.time_factor_matrix[(day_type, time_slot)] = {
                        'usage_rate': 0.0,
                        'weighted_usage_rate': 0.0,
                        'sample_size': 0
                    }
        
        # 統計摘要
        print(f"\nTime Factor Summary:")
        for day_type in ['weekday', 'weekend']:
            usage_rates = [self.time_factor_matrix[(day_type, slot)]['weighted_usage_rate'] 
                          for slot in range(self.time_slots)]
            non_zero_rates = [rate for rate in usage_rates if rate > 0]
            
            print(f"  {day_type.capitalize()}:")
            print(f"    Active slots: {len(non_zero_rates)}/{self.time_slots}")
            print(f"    Max usage rate: {max(usage_rates):.3f}")
            print(f"    Mean usage rate: {np.mean(usage_rates):.3f}")
        
        return self.time_factor_matrix
    
    def triangular_membership(self, x, a, b, c):
        """三角隸屬函數"""
        return np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))
    
    def calculate_fuzzy_memberships(self, timestamp):
        """計算指定時間點的模糊隸屬度"""
        # 提取時間特徵
        hour = timestamp.hour
        minute = timestamp.minute
        is_weekend = timestamp.weekday() >= 5
        day_type = 'weekend' if is_weekend else 'weekday'
        time_slot = hour * 4 + minute // 15  # 0-95個時段
        
        result = {
            'day_type': day_type,
            'time_slot': time_slot,
            'hour': hour,
            'minute': minute
        }
        
        # 獲取使用機率
        if (day_type, time_slot) in self.usage_probability_matrix:
            prob_data = self.usage_probability_matrix[(day_type, time_slot)]
            
            # 使用短期機率作為主要指標
            short_prob = prob_data['short_prob']
            
            # 使用機率模糊化
            result['usage_prob_low'] = self.triangular_membership(short_prob, 0, 0.15, 0.3)
            result['usage_prob_medium'] = self.triangular_membership(short_prob, 0.3, 0.5, 0.7)
            result['usage_prob_high'] = self.triangular_membership(short_prob, 0.7, 0.85, 1.0)
            result['usage_probability'] = short_prob
        else:
            result['usage_prob_low'] = 0.7
            result['usage_prob_medium'] = 0.3
            result['usage_prob_high'] = 0.0
            result['usage_probability'] = 0.2
        
        # 獲取使用穩定性
        if (day_type, time_slot) in self.stability_matrix:
            stability_data = self.stability_matrix[(day_type, time_slot)]
            overall_stability = stability_data['overall_stability']
            
            # 穩定性模糊化
            result['stability_low'] = self.triangular_membership(overall_stability, 0, 0.2, 0.4)
            result['stability_medium'] = self.triangular_membership(overall_stability, 0.4, 0.6, 0.8)
            result['stability_high'] = self.triangular_membership(overall_stability, 0.8, 0.9, 1.0)
            result['stability'] = overall_stability
        else:
            result['stability_low'] = 0.8
            result['stability_medium'] = 0.2
            result['stability_high'] = 0.0
            result['stability'] = 0.2
        
        # 獲取時間因子
        if (day_type, time_slot) in self.time_factor_matrix:
            time_data = self.time_factor_matrix[(day_type, time_slot)]
            usage_rate = time_data['weighted_usage_rate']
            
            # 時間因子模糊化
            result['time_non_use'] = self.triangular_membership(usage_rate, 0, 0.15, 0.3)
            result['time_possible'] = self.triangular_membership(usage_rate, 0.3, 0.5, 0.7)
            result['time_peak'] = self.triangular_membership(usage_rate, 0.7, 0.85, 1.0)
            result['time_factor'] = usage_rate
        else:
            result['time_non_use'] = 0.6
            result['time_possible'] = 0.4
            result['time_peak'] = 0.0
            result['time_factor'] = 0.2
        
        return result
    
    def define_habit_rules(self):
        """定義使用習慣模糊規則"""
        print("==== Defining User Habit Rules ====")
        
        # 規則格式: (使用機率, 穩定性, 時間因子, 輸出習慣等級, 權重)
        self.habit_rules = [
            # === 高習慣分數規則：強烈建議不要關機 ===
            ('high', 'high', 'peak', 'high', 1.0),          # 高機率+高穩定+高峰時段
            ('high', 'high', 'possible', 'high', 0.95),     # 高機率+高穩定+可能時段
            ('high', 'medium', 'peak', 'high', 0.9),        # 高機率+中穩定+高峰時段
            ('medium', 'high', 'peak', 'high', 0.85),       # 中機率+高穩定+高峰時段
            
            # === 中習慣分數規則：需要謹慎判斷 ===
            ('high', 'low', 'peak', 'medium', 0.8),         # 高機率但不穩定+高峰時段
            ('high', 'medium', 'possible', 'medium', 0.75), # 高機率+中穩定+可能時段
            ('medium', 'high', 'possible', 'medium', 0.8),  # 中機率+高穩定+可能時段
            ('medium', 'medium', 'peak', 'medium', 0.7),    # 中機率+中穩定+高峰時段
            ('high', 'high', 'non_use', 'medium', 0.6),     # 高機率+高穩定但非使用時段
            
            # === 中低習慣分數規則：可以考慮節能 ===
            ('low', 'high', 'peak', 'medium', 0.6),         # 低機率但高穩定+高峰
            ('medium', 'low', 'peak', 'medium', 0.5),       # 中機率+低穩定+高峰
            ('medium', 'medium', 'possible', 'medium', 0.6), # 中機率+中穩定+可能
            ('high', 'low', 'possible', 'medium', 0.5),     # 高機率+低穩定+可能
            
            # === 低習慣分數規則：可以關機節能 ===
            ('low', 'high', 'possible', 'low', 0.7),        # 低機率+高穩定+可能時段
            ('low', 'medium', 'peak', 'low', 0.6),          # 低機率+中穩定+高峰時段
            ('medium', 'low', 'possible', 'low', 0.6),      # 中機率+低穩定+可能時段
            ('low', 'low', 'peak', 'low', 0.5),             # 低機率+低穩定+高峰時段
            
            # === 很低習慣分數規則：強烈建議關機 ===
            ('low', 'high', 'non_use', 'low', 1.0),         # 低機率+高穩定+非使用時段
            ('low', 'medium', 'non_use', 'low', 0.9),       # 低機率+中穩定+非使用時段
            ('medium', 'low', 'non_use', 'low', 0.8),       # 中機率+低穩定+非使用時段
            ('low', 'low', 'possible', 'low', 0.7),         # 低機率+低穩定+可能時段
            ('low', 'low', 'non_use', 'low', 1.0),          # 低機率+低穩定+非使用時段
        ]
        
        print(f"Defined {len(self.habit_rules)} habit rules")
        return self.habit_rules
    
    def calculate_habit_score(self, timestamp):
        """計算指定時間點的習慣分數 - 改進版"""
        # 獲取模糊隸屬度
        memberships = self.calculate_fuzzy_memberships(timestamp)
        
        # 初始化規則激活
        low_activation = 0.0
        medium_activation = 0.0
        high_activation = 0.0
        
        # 計算規則激活強度
        for rule in self.habit_rules:
            usage_level, stability_level, time_level, output_level, weight = rule
            
            # 獲取前件隸屬度
            usage_membership = memberships[f'usage_prob_{usage_level}']
            stability_membership = memberships[f'stability_{stability_level}']
            time_membership = memberships[f'time_{time_level}']
            
            # 計算規則激活(最小值方法)
            activation = min(usage_membership, stability_membership, time_membership) * weight
            
            # 累積到對應的輸出等級
            if output_level == 'low':
                low_activation += activation
            elif output_level == 'medium':
                medium_activation += activation
            elif output_level == 'high':
                high_activation += activation
        
        # 限制激活強度上限
        low_activation = min(low_activation, 1.0)
        medium_activation = min(medium_activation, 1.0)
        high_activation = min(high_activation, 1.0)
        
        # 改進的最終習慣分數計算 - 增加分數多樣性
        total_activation = low_activation + medium_activation + high_activation
        
        if total_activation > 0:
            # 加權平均，但加入更多變化
            base_score = (
                low_activation * 0.15 + 
                medium_activation * 0.50 + 
                high_activation * 0.85
            ) / total_activation
            
            # 添加基於時間和使用率的微調
            time_factor = memberships.get('time_factor', 0.5)
            usage_prob = memberships.get('usage_probability', 0.5)
            
            # 微調係數 (±0.1範圍)
            time_adjustment = (time_factor - 0.5) * 0.1
            usage_adjustment = (usage_prob - 0.5) * 0.05
            
            habit_score = base_score + time_adjustment + usage_adjustment
        else:
            habit_score = 0.3  # 默認中低分數
        
        # 確保分數在合理範圍內，但允許更寬的範圍
        habit_score = max(0.05, min(0.95, habit_score))
        
        return {
            'habit_score': habit_score,
            'low_activation': low_activation,
            'medium_activation': medium_activation,
            'high_activation': high_activation,
            'memberships': memberships,
            'confidence': min(total_activation, 1.0)
        }
    
    def visualize_habit_analysis(self):
        """視覺化習慣分析結果"""
        print("==== Creating User Habit Visualizations ====")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. 使用機率熱力圖 (聚合為每小時顯示)
        prob_matrix = np.zeros((2, 24))  # 24小時顯示
        for i, day_type in enumerate(['weekday', 'weekend']):
            for hour in range(24):
                hour_probs = []
                for quarter in range(4):  # 每小時4個15分鐘時段
                    slot = hour * 4 + quarter
                    if (day_type, slot) in self.usage_probability_matrix:
                        hour_probs.append(self.usage_probability_matrix[(day_type, slot)]['short_prob'])
                prob_matrix[i, hour] = np.mean(hour_probs) if hour_probs else 0
        
        im1 = axes[0, 0].imshow(prob_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[0, 0].set_title('Usage Probability by Hour')
        axes[0, 0].set_ylabel('Day Type')
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_yticks([0, 1])
        axes[0, 0].set_yticklabels(['Weekday', 'Weekend'])
        axes[0, 0].set_xticks(range(0, 24, 4))
        axes[0, 0].set_xticklabels([f'{i}h' for i in range(0, 24, 4)])
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 2. 穩定性熱力圖 (聚合為每小時顯示)
        stability_matrix = np.zeros((2, 24))
        for i, day_type in enumerate(['weekday', 'weekend']):
            for hour in range(24):
                hour_stabilities = []
                for quarter in range(4):
                    slot = hour * 4 + quarter
                    if (day_type, slot) in self.stability_matrix:
                        hour_stabilities.append(self.stability_matrix[(day_type, slot)]['overall_stability'])
                stability_matrix[i, hour] = np.mean(hour_stabilities) if hour_stabilities else 0
        
        im2 = axes[0, 1].imshow(stability_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        axes[0, 1].set_title('Usage Stability by Hour')
        axes[0, 1].set_ylabel('Day Type')
        axes[0, 1].set_xlabel('Hour of Day')
        axes[0, 1].set_yticks([0, 1])
        axes[0, 1].set_yticklabels(['Weekday', 'Weekend'])
        axes[0, 1].set_xticks(range(0, 24, 4))
        axes[0, 1].set_xticklabels([f'{i}h' for i in range(0, 24, 4)])
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 3. 時間因子熱力圖 (聚合為每小時顯示)
        time_matrix = np.zeros((2, 24))
        for i, day_type in enumerate(['weekday', 'weekend']):
            for hour in range(24):
                hour_rates = []
                for quarter in range(4):
                    slot = hour * 4 + quarter
                    if (day_type, slot) in self.time_factor_matrix:
                        hour_rates.append(self.time_factor_matrix[(day_type, slot)]['weighted_usage_rate'])
                time_matrix[i, hour] = np.mean(hour_rates) if hour_rates else 0
        
        im3 = axes[0, 2].imshow(time_matrix, cmap='Oranges', aspect='auto', vmin=0, vmax=1)
        axes[0, 2].set_title('Time Factor (Usage Rate) by Hour')
        axes[0, 2].set_ylabel('Day Type')
        axes[0, 2].set_xlabel('Hour of Day')
        axes[0, 2].set_yticks([0, 1])
        axes[0, 2].set_yticklabels(['Weekday', 'Weekend'])
        axes[0, 2].set_xticks(range(0, 24, 4))
        axes[0, 2].set_xticklabels([f'{i}h' for i in range(0, 24, 4)])
        plt.colorbar(im3, ax=axes[0, 2])
        
        # 4. 轉換間隔分布
        if self.transition_data is not None and len(self.transition_data) > 0:
            intervals = self.transition_data['interval_minutes']
            axes[1, 0].hist(intervals[intervals <= 300], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 0].set_xlabel('Interval Minutes (≤300)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Turn-on Interval Distribution')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No transition data available', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].set_title('Turn-on Interval Distribution (No Data)')
            axes[1, 0].set_xlabel('Interval Minutes')
            axes[1, 0].set_ylabel('Frequency')
        
        # 5. 每日習慣分數示例 (每小時計算)
        sample_times = pd.date_range('2024-01-01', periods=24, freq='H')
        habit_scores = []
        for time in sample_times:
            result = self.calculate_habit_score(time)
            habit_scores.append(result['habit_score'])
        
        axes[1, 1].plot(range(24), habit_scores, 'o-', linewidth=2, markersize=4)
        axes[1, 1].set_xlabel('Hour of Day')
        axes[1, 1].set_ylabel('Habit Score')
        axes[1, 1].set_title('Daily Habit Score Pattern (Sample)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim(0, 23)
        axes[1, 1].set_ylim(0, 1)
        
        # 6. 週末vs平日對比 (每小時計算)
        weekday_scores = []
        weekend_scores = []
        
        for hour in range(24):
            # 週三的習慣分數
            weekday_time = datetime(2024, 1, 3, hour, 0)  # Wednesday
            weekday_result = self.calculate_habit_score(weekday_time)
            weekday_scores.append(weekday_result['habit_score'])
            
            # 週六的習慣分數
            weekend_time = datetime(2024, 1, 6, hour, 0)  # Saturday
            weekend_result = self.calculate_habit_score(weekend_time)
            weekend_scores.append(weekend_result['habit_score'])
        
        axes[1, 2].plot(range(24), weekday_scores, 'o-', label='Weekday', linewidth=2)
        axes[1, 2].plot(range(24), weekend_scores, 's-', label='Weekend', linewidth=2)
        axes[1, 2].set_xlabel('Hour of Day')
        axes[1, 2].set_ylabel('Habit Score')
        axes[1, 2].set_title('Weekday vs Weekend Habit Patterns')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_xlim(0, 23)
        axes[1, 2].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
        
        # 額外顯示詳細的15分鐘時段使用率
        self._plot_detailed_usage_rates()
        
        # 顯示三角隸屬函數
        self._plot_triangular_memberships()
    
    def _plot_detailed_usage_rates(self):
        """繪製詳細的15分鐘時段使用率"""
        print("==== Creating Detailed 15-min Usage Rate Plot ====")
        
        fig, axes = plt.subplots(2, 1, figsize=(20, 8))
        
        # 準備數據
        hours = []
        weekday_rates = []
        weekend_rates = []
        
        for slot in range(self.time_slots):
            hour = slot // 4
            minute = (slot % 4) * 15
            time_label = hour + minute/60.0
            
            hours.append(time_label)
            
            # 平日使用率
            if ('weekday', slot) in self.time_factor_matrix:
                weekday_rates.append(self.time_factor_matrix[('weekday', slot)]['weighted_usage_rate'])
            else:
                weekday_rates.append(0)
            
            # 週末使用率  
            if ('weekend', slot) in self.time_factor_matrix:
                weekend_rates.append(self.time_factor_matrix[('weekend', slot)]['weighted_usage_rate'])
            else:
                weekend_rates.append(0)
        
        # 繪製平日使用率
        axes[0].plot(hours, weekday_rates, 'o-', linewidth=1, markersize=2, color='blue')
        axes[0].set_title('Weekday Usage Rate by 15-min Slots')
        axes[0].set_ylabel('Usage Rate')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, 24)
        axes[0].set_ylim(0, max(max(weekday_rates), max(weekend_rates)) * 1.1 if weekday_rates else 1)
        
        # 設置x軸刻度
        axes[0].set_xticks(range(0, 25, 2))
        axes[0].set_xticklabels([f'{i}:00' for i in range(0, 25, 2)])
        
        # 繪製週末使用率
        axes[1].plot(hours, weekend_rates, 'o-', linewidth=1, markersize=2, color='red')
        axes[1].set_title('Weekend Usage Rate by 15-min Slots')
        axes[1].set_xlabel('Time of Day')
        axes[1].set_ylabel('Usage Rate')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0, 24)
        axes[1].set_ylim(0, max(max(weekday_rates), max(weekend_rates)) * 1.1 if weekend_rates else 1)
        
        # 設置x軸刻度
        axes[1].set_xticks(range(0, 25, 2))
        axes[1].set_xticklabels([f'{i}:00' for i in range(0, 25, 2)])
        
        plt.tight_layout()
        plt.show()
        
        # 顯示高使用率時段
        print("\nHigh Usage Rate Time Slots (>0.3):")
        for day_type in ['weekday', 'weekend']:
            print(f"\n{day_type.capitalize()}:")
            high_usage_slots = []
            for slot in range(self.time_slots):
                if (day_type, slot) in self.time_factor_matrix:
                    rate = self.time_factor_matrix[(day_type, slot)]['weighted_usage_rate']
                    if rate > 0.3:
                        hour = slot // 4
                        minute = (slot % 4) * 15
                        high_usage_slots.append(f"{hour:02d}:{minute:02d} (rate={rate:.3f})")
            
            if high_usage_slots:
                for slot_info in high_usage_slots:
                    print(f"  {slot_info}")
            else:
                print("  No high usage time slots found")
    
    def _plot_triangular_memberships(self):
        """顯示三角隸屬函數"""
        print("==== Creating Triangular Membership Functions Plot ====")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Usage Probability 隸屬函數
        x_usage = np.linspace(0, 1, 1000)
        
        usage_low = self.triangular_membership(x_usage, 0, 0.15, 0.3)
        usage_medium = self.triangular_membership(x_usage, 0.3, 0.5, 0.7)
        usage_high = self.triangular_membership(x_usage, 0.7, 0.85, 1.0)
        
        axes[0].plot(x_usage, usage_low, 'r-', linewidth=2, label='Low (0-0.15-0.3)')
        axes[0].plot(x_usage, usage_medium, 'orange', linewidth=2, label='Medium (0.3-0.5-0.7)')
        axes[0].plot(x_usage, usage_high, 'g-', linewidth=2, label='High (0.7-0.85-1.0)')
        
        axes[0].set_xlabel('Usage Probability')
        axes[0].set_ylabel('Membership Degree')
        axes[0].set_title('Usage Probability Membership Functions')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1.1)
        
        # 標記關鍵點
        axes[0].axvline(0.15, color='gray', linestyle='--', alpha=0.5)
        axes[0].axvline(0.3, color='gray', linestyle='--', alpha=0.5)
        axes[0].axvline(0.5, color='gray', linestyle='--', alpha=0.5)
        axes[0].axvline(0.7, color='gray', linestyle='--', alpha=0.5)
        axes[0].axvline(0.85, color='gray', linestyle='--', alpha=0.5)
        
        # 2. Usage Stability 隸屬函數
        x_stability = np.linspace(0, 1, 1000)
        
        stability_low = self.triangular_membership(x_stability, 0, 0.2, 0.4)
        stability_medium = self.triangular_membership(x_stability, 0.4, 0.6, 0.8)
        stability_high = self.triangular_membership(x_stability, 0.8, 0.9, 1.0)
        
        axes[1].plot(x_stability, stability_low, 'r-', linewidth=2, label='Low (0-0.2-0.4)')
        axes[1].plot(x_stability, stability_medium, 'orange', linewidth=2, label='Medium (0.4-0.6-0.8)')
        axes[1].plot(x_stability, stability_high, 'g-', linewidth=2, label='High (0.8-0.9-1.0)')
        
        axes[1].set_xlabel('Usage Stability')
        axes[1].set_ylabel('Membership Degree')
        axes[1].set_title('Usage Stability Membership Functions')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1.1)
        
        # 標記關鍵點
        axes[1].axvline(0.2, color='gray', linestyle='--', alpha=0.5)
        axes[1].axvline(0.4, color='gray', linestyle='--', alpha=0.5)
        axes[1].axvline(0.6, color='gray', linestyle='--', alpha=0.5)
        axes[1].axvline(0.8, color='gray', linestyle='--', alpha=0.5)
        axes[1].axvline(0.9, color='gray', linestyle='--', alpha=0.5)
        
        # 3. Time Factor 隸屬函數
        x_time = np.linspace(0, 1, 1000)
        
        time_non_use = self.triangular_membership(x_time, 0, 0.15, 0.3)
        time_possible = self.triangular_membership(x_time, 0.3, 0.5, 0.7)
        time_peak = self.triangular_membership(x_time, 0.7, 0.85, 1.0)
        
        axes[2].plot(x_time, time_non_use, 'r-', linewidth=2, label='Non-use (0-0.15-0.3)')
        axes[2].plot(x_time, time_possible, 'orange', linewidth=2, label='Possible (0.3-0.5-0.7)')
        axes[2].plot(x_time, time_peak, 'g-', linewidth=2, label='Peak (0.7-0.85-1.0)')
        
        axes[2].set_xlabel('Time Factor (Usage Rate)')
        axes[2].set_ylabel('Membership Degree')
        axes[2].set_title('Time Factor Membership Functions')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1.1)
        
        # 標記關鍵點
        axes[2].axvline(0.15, color='gray', linestyle='--', alpha=0.5)
        axes[2].axvline(0.3, color='gray', linestyle='--', alpha=0.5)
        axes[2].axvline(0.5, color='gray', linestyle='--', alpha=0.5)
        axes[2].axvline(0.7, color='gray', linestyle='--', alpha=0.5)
        axes[2].axvline(0.85, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        
        # 顯示隸屬函數的數值定義
        print("\nMembership Function Definitions:")
        print("="*40)
        print("Usage Probability:")
        print("  Low:    triangular(x, 0, 0.15, 0.3)")
        print("  Medium: triangular(x, 0.3, 0.5, 0.7)")
        print("  High:   triangular(x, 0.7, 0.85, 1.0)")
        print("\nUsage Stability:")
        print("  Low:    triangular(x, 0, 0.2, 0.4)")
        print("  Medium: triangular(x, 0.4, 0.6, 0.8)")
        print("  High:   triangular(x, 0.8, 0.9, 1.0)")
        print("\nTime Factor:")
        print("  Non-use:  triangular(x, 0, 0.15, 0.3)")
        print("  Possible: triangular(x, 0.3, 0.5, 0.7)")
        print("  Peak:     triangular(x, 0.7, 0.85, 1.0)")
    
    def get_habit_score(self, timestamp):
        """外部調用介面"""
        result = self.calculate_habit_score(timestamp)
        return {
            'score': result['habit_score'],
            'confidence': result['confidence'],
            'factors': {
                'usage_probability': result['memberships']['usage_probability'],
                'stability': result['memberships']['stability'],
                'time_factor': result['memberships']['time_factor'],
                'day_type': result['memberships']['day_type'],
                'time_slot': result['memberships']['time_slot']
            }
        }
    
    # ========== 評估功能 ==========
    
    def evaluate_accuracy(self):
        """評估系統準確性"""
        
        # 創建測試案例
        test_cases = [
            # (時間, 預期習慣分數範圍, 場景描述)
            (datetime(2024, 1, 3, 14, 0), (0.6, 0.9), "平日下午高峰期"),
            (datetime(2024, 1, 3, 2, 0), (0.1, 0.4), "平日深夜"),
            (datetime(2024, 1, 6, 10, 0), (0.4, 0.8), "週末上午"),
            (datetime(2024, 1, 3, 22, 30), (0.2, 0.5), "平日晚睡時間"),
        ]
        
        accuracy_results = []
        for timestamp, expected_range, description in test_cases:
            result = self.get_habit_score(timestamp)
            score = result['score']
            is_accurate = expected_range[0] <= score <= expected_range[1]
            
            accuracy_results.append({
                'scenario': description,
                'expected': expected_range,
                'actual': score,
                'accurate': is_accurate,
                'confidence': result['confidence']
            })
        
        accuracy_rate = sum(r['accurate'] for r in accuracy_results) / len(accuracy_results)
        return accuracy_rate, accuracy_results

    def evaluate_stability(self):
        """評估系統穩定性"""
        
        # 相似時間點的分數變化
        similar_times = [
            datetime(2024, 1, 3, 14, 0),   # 週三14:00
            datetime(2024, 1, 3, 14, 15),  # 週三14:15
            datetime(2024, 1, 3, 14, 30),  # 週三14:30
        ]
        
        scores = [self.get_habit_score(t)['score'] for t in similar_times]
        stability = 1 - np.std(scores)  # 標準差越小越穩定
        
        return stability, scores

    def evaluate_weekday_weekend_logic(self):
        """檢查週末平日邏輯"""
        
        results = {}
        for hour in [8, 14, 20]:  # 早中晚測試
            weekday_score = self.get_habit_score(datetime(2024, 1, 3, hour))['score']  # 週三
            weekend_score = self.get_habit_score(datetime(2024, 1, 6, hour))['score']  # 週六
            
            results[f'{hour}h'] = {
                'weekday': weekday_score,
                'weekend': weekend_score,
                'difference': abs(weekday_score - weekend_score)
            }
        
        return results

    def evaluate_time_logic(self):
        """檢查時間邏輯合理性"""
        
        # 檢查深夜分數是否確實很低
        night_scores = []
        for hour in [0, 1, 2, 3, 4, 5]:
            score = self.get_habit_score(datetime(2024, 1, 3, hour))['score']
            night_scores.append(score)
        
        # 檢查下午分數是否確實較高
        afternoon_scores = []
        for hour in [12, 13, 14, 15, 16, 17]:
            score = self.get_habit_score(datetime(2024, 1, 3, hour))['score']
            afternoon_scores.append(score)
        
        logic_check = {
            'night_avg': np.mean(night_scores),
            'afternoon_avg': np.mean(afternoon_scores),
            'logic_correct': np.mean(afternoon_scores) > np.mean(night_scores)
        }
        
        return logic_check

    def evaluate_score_distribution(self):
        """評估分數分布合理性"""
        
        all_scores = []
        for day in range(7):  # 一週
            for hour in range(24):  # 每小時
                timestamp = datetime(2024, 1, 1 + day, hour)
                score = self.get_habit_score(timestamp)['score']
                all_scores.append(score)
        
        distribution_stats = {
            'min': np.min(all_scores),
            'max': np.max(all_scores),
            'mean': np.mean(all_scores),
            'std': np.std(all_scores),
            'unique_values': len(np.unique(np.round(all_scores, 2))),
            'full_range_used': (np.max(all_scores) - np.min(all_scores)) > 0.5
        }
        
        return distribution_stats, all_scores

    def simulate_energy_saving(self):
        """模擬節能效果"""
        
        # 模擬一週的決策
        decisions = []
        for day in range(7):
            for hour in range(24):
                timestamp = datetime(2024, 1, 1 + day, hour)
                habit_result = self.get_habit_score(timestamp)
                
                # 修正的決策邏輯 - 更激進的節能策略
                if habit_result['score'] < 0.4:  # 降低閾值
                    decision = 'power_off'
                    energy_saved = 50  # 瓦特
                elif habit_result['score'] < 0.6:  # 降低閾值
                    decision = 'standby'
                    energy_saved = 30
                else:
                    decision = 'keep_on'
                    energy_saved = 0
                
                decisions.append({
                    'timestamp': timestamp,
                    'habit_score': habit_result['score'],
                    'decision': decision,
                    'energy_saved': energy_saved
                })
        
        total_energy_saved = sum(d['energy_saved'] for d in decisions)
        power_off_hours = sum(1 for d in decisions if d['decision'] == 'power_off')
        standby_hours = sum(1 for d in decisions if d['decision'] == 'standby')
        
        return {
            'decisions': decisions,  # 返回決策列表
            'total_energy_saved_wh': total_energy_saved,
            'power_off_hours': power_off_hours,
            'standby_hours': standby_hours,
            'energy_saving_rate': (power_off_hours + standby_hours) / len(decisions)
        }

    def comprehensive_evaluation(self):
        """綜合系統評估"""
        print("="*60)
        print("COMPREHENSIVE SYSTEM EVALUATION")
        print("="*60)
        
        # 1. 準確性評估
        accuracy_rate, accuracy_details = self.evaluate_accuracy()
        print(f"✅ Accuracy Score: {accuracy_rate:.2%}")
        
        # 2. 穩定性評估  
        stability_score, _ = self.evaluate_stability()
        print(f"✅ Stability Score: {stability_score:.3f}")
        
        # 3. 邏輯檢查
        logic_results = self.evaluate_time_logic()
        print(f"✅ Time Logic Correct: {logic_results['logic_correct']}")
        
        # 4. 分數分布
        dist_stats, _ = self.evaluate_score_distribution()
        print(f"✅ Score Range Used: {dist_stats['full_range_used']}")
        print(f"✅ Unique Values: {dist_stats['unique_values']}")
        
        # 5. 節能效果
        energy_results = self.simulate_energy_saving()
        print(f"✅ Energy Saving Rate: {energy_results['energy_saving_rate']:.2%}")
        
        # 綜合評分
        scores = [
            accuracy_rate,
            stability_score,
            1.0 if logic_results['logic_correct'] else 0.0,
            1.0 if dist_stats['full_range_used'] else 0.5,
            min(dist_stats['unique_values'] / 50, 1.0),  # 歸一化
            energy_results['energy_saving_rate'] * 2  # 加權
        ]
        
        overall_score = np.mean(scores)
        
        print(f"\n🎯 OVERALL SYSTEM SCORE: {overall_score:.3f}/1.000")
        
        # 評級
        if overall_score >= 0.8:
            grade = "EXCELLENT 🌟"
        elif overall_score >= 0.7:
            grade = "GOOD ✅"
        elif overall_score >= 0.6:
            grade = "ACCEPTABLE ⚠️"
        else:
            grade = "NEEDS IMPROVEMENT ❌"
        
        print(f"📊 System Grade: {grade}")
        
        return overall_score, {
            'accuracy': accuracy_rate,
            'stability': stability_score,
            'logic_correct': logic_results['logic_correct'],
            'distribution': dist_stats,
            'energy_saving': energy_results
        }

    def create_evaluation_report(self):
        """創建評估報告"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. 24小時分數曲線
        hours = range(24)
        weekday_scores = [self.get_habit_score(datetime(2024, 1, 3, h))['score'] for h in hours]
        weekend_scores = [self.get_habit_score(datetime(2024, 1, 6, h))['score'] for h in hours]
        
        axes[0, 0].plot(hours, weekday_scores, 'o-', label='Weekday')
        axes[0, 0].plot(hours, weekend_scores, 's-', label='Weekend')
        axes[0, 0].set_title('24-Hour Score Pattern')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. 分數分布直方圖
        _, all_scores = self.evaluate_score_distribution()
        axes[0, 1].hist(all_scores, bins=20, alpha=0.7, color='skyblue')
        axes[0, 1].set_title('Score Distribution')
        axes[0, 1].set_xlabel('Habit Score')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. 置信度分析
        confidences = []
        for day in range(7):
            for hour in range(0, 24, 4):
                conf = self.get_habit_score(datetime(2024, 1, 1 + day, hour))['confidence']
                confidences.append(conf)
        
        axes[0, 2].hist(confidences, bins=15, alpha=0.7, color='lightgreen')
        axes[0, 2].set_title('Confidence Distribution')
        axes[0, 2].set_xlabel('Confidence Score')
        
        # 4. 週末vs平日箱線圖
        weekday_all = [self.get_habit_score(datetime(2024, 1, 3, h))['score'] for h in range(24)]
        weekend_all = [self.get_habit_score(datetime(2024, 1, 6, h))['score'] for h in range(24)]
        
        axes[1, 0].boxplot([weekday_all, weekend_all], labels=['Weekday', 'Weekend'])
        axes[1, 0].set_title('Weekday vs Weekend Comparison')
        axes[1, 0].set_ylabel('Habit Score')
        
        # 5. 節能決策模擬 - 修正這部分
        energy_results = self.simulate_energy_saving()
        decisions_list = energy_results['decisions']  # 獲取決策列表
        
        decision_counts = {}
        for decision_record in decisions_list:
            decision = decision_record['decision']
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
        
        if decision_counts:  # 確保有數據
            axes[1, 1].pie(decision_counts.values(), labels=decision_counts.keys(), autopct='%1.1f%%')
        else:
            axes[1, 1].text(0.5, 0.5, 'No decisions found', ha='center', va='center')
        axes[1, 1].set_title('Energy Saving Decisions')
        
        # 6. 評估指標雷達圖
        _, eval_details = self.comprehensive_evaluation()
        metrics = ['Accuracy', 'Stability', 'Logic', 'Distribution', 'Energy Saving']
        values = [
            eval_details['accuracy'],
            eval_details['stability'], 
            1.0 if eval_details['logic_correct'] else 0.0,
            1.0 if eval_details['distribution']['full_range_used'] else 0.5,
            eval_details['energy_saving']['energy_saving_rate']
        ]
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        values_plot = values + [values[0]]  # 閉合
        angles_plot = list(angles) + [angles[0]]
        
        axes[1, 2].plot(angles_plot, values_plot, 'o-', linewidth=2)
        axes[1, 2].fill(angles_plot, values_plot, alpha=0.3)
        axes[1, 2].set_xticks(angles)
        axes[1, 2].set_xticklabels(metrics)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].set_title('System Performance Radar')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self, usage_file_path):
        """運行完整的使用習慣分析"""
        print("="*80)
        print("USER HABIT SCORE MODULE - COMPLETE ANALYSIS")
        print("="*80)
        
        # 載入數據
        df = self.load_usage_data(usage_file_path)
        if df is None:
            return None
        
        # 提取狀態轉換
        self.extract_state_transitions(df)
        
        # 計算各項指標
        self.calculate_usage_probability()
        self.calculate_usage_stability(df)
        self.calculate_time_factor(df)
        
        # 定義模糊規則
        self.define_habit_rules()
        
        # 測試幾個時間點
        test_times = [
            datetime(2024, 1, 3, 8, 0),   # 週三早上8:00
            datetime(2024, 1, 3, 8, 15),  # 週三早上8:15
            datetime(2024, 1, 3, 14, 30), # 週三下午2:30
            datetime(2024, 1, 3, 20, 0),  # 週三晚上8:00
            datetime(2024, 1, 6, 10, 45), # 週六上午10:45
            datetime(2024, 1, 6, 22, 15), # 週六晚上10:15
        ]
        
        print("\n==== Sample Habit Score Results ====")
        for test_time in test_times:
            result = self.calculate_habit_score(test_time)
            day_name = test_time.strftime("%A")
            time_str = test_time.strftime("%H:%M")
            print(f"{day_name} {time_str} - "
                  f"Habit Score: {result['habit_score']:.3f}, "
                  f"Confidence: {result['confidence']:.3f}, "
                  f"Slot: {result['memberships']['time_slot']}")
        
        # 創建視覺化
        self.visualize_habit_analysis()
        
        print("="*80)
        print("USER HABIT SCORE ANALYSIS COMPLETE")
        print("="*80)
        
        return {
            'usage_probability_matrix': self.usage_probability_matrix,
            'stability_matrix': self.stability_matrix,
            'time_factor_matrix': self.time_factor_matrix,
            'transition_data': self.transition_data
        }

# 使用示例
if __name__ == "__main__":
    # 初始化使用習慣模組
    habit_module = UserHabitScoreModule()
    
    # 檔案路徑
    usage_file_path = "C:/Users/王俞文/Documents/glasgow/msc project/data/data_after_preprocessing.csv"
    
    # 運行完整分析
    result = habit_module.run_complete_analysis(usage_file_path)
    
    # 運行綜合評估
    if result is not None:
        overall_score, details = habit_module.comprehensive_evaluation()
        habit_module.create_evaluation_report()
        
        # 測試外部調用介面
        current_time = datetime.now()
        habit_result = habit_module.get_habit_score(current_time)
        print(f"\nCurrent Habit Score: {habit_result}")