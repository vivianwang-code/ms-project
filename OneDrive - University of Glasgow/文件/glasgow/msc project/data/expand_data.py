import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from collections import defaultdict

class PowerDataExpander:
    def __init__(self):
        """
        初始化數據擴展器
        """
        self.original_data = None
        self.hourly_patterns = {}
        self.weekday_patterns = {}
        self.transition_patterns = {}
        self.missing_patterns = {}
        
    def load_original_data(self, csv_path="C:/Users/王俞文/OneDrive - University of Glasgow/文件/glasgow/msc project/data/data_after_preprocessing.csv"):
        """
        載入預處理後的CSV數據
        """
        try:
            self.original_data = pd.read_csv(csv_path)
            self.original_data['timestamp'] = pd.to_datetime(self.original_data['timestamp'])
            print(f"✅ 成功載入 {len(self.original_data)} 筆原始數據")
            print(f"📅 時間範圍：{self.original_data['timestamp'].min()} 到 {self.original_data['timestamp'].max()}")
            return True
        except FileNotFoundError:
            print(f"❌ 找不到文件 {csv_path}")
            print("💡 請確保文件在當前目錄下，或提供正確的文件路徑")
            return False
        except Exception as e:
            print(f"❌ 載入數據時發生錯誤：{e}")
            return False
    
    def analyze_patterns(self):
        """
        分析原始數據的使用模式
        """
        if self.original_data is None:
            raise ValueError("請先載入原始數據")
        
        data = self.original_data.copy()
        data['hour'] = data['timestamp'].dt.hour
        data['weekday'] = data['timestamp'].dt.weekday
        data['is_weekend'] = data['weekday'].isin([5, 6])
        data['time_slot'] = data['hour'].apply(self._get_time_slot)
        
        print("🔍 開始分析數據模式...")
        
        # 1. 分析每小時的功率分佈
        self._analyze_hourly_patterns(data)
        
        # 2. 分析工作日vs週末模式
        self._analyze_weekday_patterns(data)
        
        # 3. 分析狀態轉換模式
        self._analyze_transition_patterns(data)
        
        # 4. 分析缺失數據模式
        self._analyze_missing_patterns(data)
        
        print("✅ 模式分析完成")
    
    def _analyze_hourly_patterns(self, data):
        """
        分析每小時的使用模式
        """
        for hour in range(24):
            hour_data = data[data['hour'] == hour]
            if len(hour_data) > 0:
                self.hourly_patterns[hour] = {
                    'power_mean': hour_data['power'].mean(),
                    'power_std': hour_data['power'].std(),
                    'power_min': hour_data['power'].min(),
                    'power_max': hour_data['power'].max(),
                    'phantom_prob': hour_data['is_phantom_load'].mean(),
                    'light_prob': hour_data['is_light_use'].mean(),
                    'regular_prob': hour_data['is_regular_use'].mean(),
                    'count': len(hour_data)
                }
        
        print(f"📊 分析了 {len(self.hourly_patterns)} 個小時的模式")
    
    def _analyze_weekday_patterns(self, data):
        """
        分析工作日vs週末模式
        """
        for is_weekend in [False, True]:
            weekend_data = data[data['is_weekend'] == is_weekend]
            day_type = "週末" if is_weekend else "工作日"
            
            if len(weekend_data) > 0:
                self.weekday_patterns[is_weekend] = {
                    'power_mean': weekend_data['power'].mean(),
                    'power_std': weekend_data['power'].std(),
                    'phantom_prob': weekend_data['is_phantom_load'].mean(),
                    'light_prob': weekend_data['is_light_use'].mean(),
                    'regular_prob': weekend_data['is_regular_use'].mean(),
                    'count': len(weekend_data)
                }
                print(f"📈 {day_type}：平均功率 {weekend_data['power'].mean():.1f}W")
    
    def _analyze_transition_patterns(self, data):
        """
        分析狀態轉換模式
        """
        data_sorted = data.sort_values('timestamp')
        data_sorted['prev_state'] = data_sorted['power_state'].shift(1)
        
        transitions = data_sorted.groupby(['prev_state', 'power_state']).size()
        total_transitions = len(data_sorted) - 1
        
        for (prev_state, curr_state), count in transitions.items():
            if pd.notna(prev_state):
                key = f"{prev_state}→{curr_state}"
                self.transition_patterns[key] = {
                    'probability': count / total_transitions,
                    'count': count
                }
        
        print(f"🔄 分析了 {len(self.transition_patterns)} 種狀態轉換")
    
    def _analyze_missing_patterns(self, data):
        """
        分析缺失數據模式（基於時間差異）
        """
        # 分析時間間隔，識別可能的缺失模式
        data_sorted = data.sort_values('timestamp')
        
        # 找出非標準間隔（不是900秒的）
        non_standard = data_sorted[data_sorted['time_diff_seconds'] > 900]
        
        for hour in range(24):
            hour_missing = non_standard[non_standard['hour'] == hour]
            missing_prob = len(hour_missing) / max(1, len(data_sorted[data_sorted['hour'] == hour]))
            self.missing_patterns[hour] = min(0.15, missing_prob)  # 限制最大缺失率15%
        
        print(f"📉 分析了各時段的數據缺失模式")
    
    def _get_time_slot(self, hour):
        """
        將小時轉換為時段
        """
        if 6 <= hour <= 9:
            return "morning"
        elif 10 <= hour <= 17:
            return "daytime"
        elif 18 <= hour <= 22:
            return "evening"
        else:
            return "night"
    
    def generate_extended_data(self, weeks=8):
        """
        生成擴展數據（2個月）
        """
        if not self.hourly_patterns:
            self.analyze_patterns()
        
        # 確定新的時間範圍
        original_start = self.original_data['timestamp'].min()
        new_start = original_start
        new_end = new_start + timedelta(weeks=weeks)
        
        # 生成15分鐘間隔的時間序列
        time_range = pd.date_range(start=new_start, end=new_end, freq='15min')
        
        print(f"🚀 開始生成 {weeks} 週的擴展數據...")
        print(f"📅 時間範圍：{new_start.date()} 到 {new_end.date()}")
        print(f"📝 預計生成：{len(time_range):,} 筆記錄")
        
        extended_records = []
        prev_power_state = "phantom load"  # 初始狀態
        
        for i, timestamp in enumerate(time_range):
            # 計算時間差
            if i == 0:
                time_diff = None
            else:
                time_diff = 900.0  # 15分鐘 = 900秒
            
            # 檢查是否應該模擬缺失數據
            if self._should_skip_record(timestamp):
                if i > 0:
                    # 更新上一筆記錄的時間差
                    extended_records[-1]['time_diff_seconds'] = extended_records[-1].get('time_diff_seconds', 0) + 900
                continue
            
            # 生成功率和狀態
            power, power_state = self._generate_power_and_state(timestamp, prev_power_state)
            prev_power_state = power_state
            
            # 創建記錄
            record = {
                'timestamp': timestamp,
                'power': power,
                'power_state': power_state,
                'is_phantom_load': power_state == 'phantom load',
                'is_off': False,  # 根據您的數據，沒有完全關機
                'is_on': True,
                'is_light_use': power_state == 'light use',
                'is_regular_use': power_state == 'regular use',
                'time_diff_seconds': time_diff
            }
            
            extended_records.append(record)
        
        # 轉換為DataFrame
        extended_df = pd.DataFrame(extended_records)
        
        print(f"✅ 生成完成：{len(extended_df):,} 筆記錄")
        print(f"📉 模擬缺失：{len(time_range) - len(extended_df):,} 筆")
        
        return extended_df
    
    def _should_skip_record(self, timestamp):
        """
        根據學習到的缺失模式決定是否跳過記錄
        """
        hour = timestamp.hour
        missing_prob = self.missing_patterns.get(hour, 0.02)
        
        # 添加一些隨機性
        return np.random.random() < missing_prob
    
    def _generate_power_and_state(self, timestamp, prev_state):
        """
        基於時間和前一狀態生成功率值和狀態
        """
        hour = timestamp.hour
        weekday = timestamp.weekday()
        is_weekend = weekday >= 5
        
        # 獲取該小時的基礎模式
        if hour in self.hourly_patterns:
            hour_pattern = self.hourly_patterns[hour]
        else:
            # 使用全天平均作為後備
            hour_pattern = {
                'power_mean': self.original_data['power'].mean(),
                'power_std': self.original_data['power'].std(),
                'phantom_prob': 0.76,  # 基於您的統計
                'light_prob': 0.20,
                'regular_prob': 0.04
            }
        
        # 週末調整因子
        weekend_factor = 1.0
        if is_weekend:
            weekend_factor = self.weekday_patterns.get(True, {}).get('power_mean', 1) / \
                           self.weekday_patterns.get(False, {}).get('power_mean', 1)
        
        # 狀態轉換考慮
        state_probs = self._get_state_probabilities(hour_pattern, prev_state)
        
        # 隨機選擇狀態
        rand = np.random.random()
        if rand < state_probs['phantom']:
            power_state = 'phantom load'
            power = np.random.normal(18, 2)  # 基於您的數據
            power = np.clip(power, 16, 35)
        elif rand < state_probs['phantom'] + state_probs['light']:
            power_state = 'light use'
            power = np.random.normal(55, 10)
            power = np.clip(power, 37, 75)
        else:
            power_state = 'regular use'
            power = np.random.normal(90, 20)
            power = np.clip(power, 80, 173)
        
        # 應用週末因子和添加變異
        power *= weekend_factor
        power *= np.random.normal(1.0, 0.05)  # 5%的隨機變異
        
        # 確保功率值在合理範圍內
        power = round(max(16, min(173, power)), 1)
        
        return power, power_state
    
    def _get_state_probabilities(self, hour_pattern, prev_state):
        """
        基於小時模式和前一狀態計算狀態概率
        """
        base_probs = {
            'phantom': hour_pattern.get('phantom_prob', 0.76),
            'light': hour_pattern.get('light_prob', 0.20),
            'regular': hour_pattern.get('regular_prob', 0.04)
        }
        
        # 狀態持續性調整（狀態有一定慣性）
        persistence_factor = 0.3
        if prev_state in ['phantom load', 'phantom']:
            base_probs['phantom'] += persistence_factor * (1 - base_probs['phantom'])
        elif prev_state in ['light use', 'light']:
            base_probs['light'] += persistence_factor * (1 - base_probs['light'])
        elif prev_state in ['regular use', 'regular']:
            base_probs['regular'] += persistence_factor * (1 - base_probs['regular'])
        
        # 歸一化
        total = sum(base_probs.values())
        return {k: v/total for k, v in base_probs.items()}
    
    def save_extended_data(self, extended_df, filename="extended_power_data_2months.csv"):
        """
        保存擴展數據並顯示統計信息
        """
        extended_df.to_csv(filename, index=False)
        print(f"\n💾 擴展數據已保存到：{filename}")
        
        # 顯示詳細統計
        self._print_comprehensive_statistics(extended_df)
        
        return filename
    
    def _print_comprehensive_statistics(self, df):
        """
        顯示comprehensive統計信息
        """
        print("\n" + "="*80)
        print("📊 擴展數據統計報告")
        print("="*80)
        
        # 基本信息
        print(f"📅 時間範圍：{df['timestamp'].min()} 到 {df['timestamp'].max()}")
        print(f"📝 總記錄數：{len(df):,}")
        print(f"⏱️  時間跨度：{(df['timestamp'].max() - df['timestamp'].min()).days} 天")
        
        # 功率統計
        print(f"\n⚡ 功率統計：")
        print(f"   平均功率：{df['power'].mean():.1f}W")
        print(f"   最小功率：{df['power'].min():.1f}W")
        print(f"   最大功率：{df['power'].max():.1f}W")
        print(f"   標準差：{df['power'].std():.1f}W")
        
        # 狀態分佈
        print(f"\n🔋 使用狀態分佈：")
        state_counts = df['power_state'].value_counts()
        for state, count in state_counts.items():
            percentage = count / len(df) * 100
            print(f"   {state}: {count:,} 筆 ({percentage:.1f}%)")
        
        # 時間分佈
        print(f"\n📅 時間分佈：")
        df_copy = df.copy()
        df_copy['weekday'] = df_copy['timestamp'].dt.weekday
        weekday_count = (df_copy['weekday'] < 5).sum()
        weekend_count = (df_copy['weekday'] >= 5).sum()
        print(f"   工作日：{weekday_count:,} 筆 ({weekday_count/len(df)*100:.1f}%)")
        print(f"   週末：{weekend_count:,} 筆 ({weekend_count/len(df)*100:.1f}%)")
        
        # 預估耗電量
        print(f"\n💡 預估耗電量：")
        total_kwh = df['power'].sum() * 0.25 / 1000  # 15分鐘間隔
        daily_kwh = total_kwh / ((df['timestamp'].max() - df['timestamp'].min()).days)
        annual_kwh = daily_kwh * 365
        print(f"   總耗電量：{total_kwh:.2f} kWh")
        print(f"   日均耗電：{daily_kwh:.2f} kWh")
        print(f"   年度預估：{annual_kwh:.0f} kWh")
        
        # 對比原始數據
        if hasattr(self, 'original_data') and self.original_data is not None:
            orig_mean = self.original_data['power'].mean()
            orig_phantom_rate = self.original_data['is_phantom_load'].mean()
            new_phantom_rate = df['is_phantom_load'].mean()
            
            print(f"\n🔄 與原始數據對比：")
            print(f"   原始平均功率：{orig_mean:.1f}W → 擴展平均功率：{df['power'].mean():.1f}W")
            print(f"   原始Phantom Load率：{orig_phantom_rate:.1%} → 擴展Phantom Load率：{new_phantom_rate:.1%}")
        
        print("="*80)

# 使用示例
def main():
    """
    主函數 - 展示如何使用數據擴展器
    """
    print("🚀 電力數據擴展器啟動")
    print("="*50)
    
    # 初始化擴展器
    expander = PowerDataExpander()
    
    # 載入數據
    if not expander.load_original_data("data_after_preprocessing.csv"):
        print("❌ 無法載入數據，請檢查文件路徑")
        return
    
    # 分析模式
    expander.analyze_patterns()
    
    # 生成8週（2個月）的擴展數據
    extended_data = expander.generate_extended_data(weeks=8)
    
    # 保存數據
    filename = expander.save_extended_data(extended_data)
    
    print(f"\n🎉 擴展完成！請查看 {filename}")
    print("💡 您現在可以使用這個擴展數據集來訓練您的智能電源管理系統")

if __name__ == "__main__":
    main()