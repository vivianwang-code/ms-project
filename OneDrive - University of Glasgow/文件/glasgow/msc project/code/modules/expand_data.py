import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from collections import defaultdict

class PowerDataExpander():
    def __init__(self, csv_path):
        """
        初始化數據擴展器（只支持往前新增）
        """
        self.original_data = None
        self.hourly_patterns = {}
        self.weekday_patterns = {}
        self.transition_patterns = {}
        self.missing_patterns = {}

        self.load_original_data(csv_path) 
        
    def load_original_data(self, csv_path):
        """
        載入預處理後的CSV數據
        """
        try:
            print(f"csv_path : {csv_path}")
            self.original_data = pd.read_csv(csv_path)
            self.original_data['timestamp'] = pd.to_datetime(self.original_data['timestamp'], format='ISO8601')
            
            # 排序數據
            self.original_data = self.original_data.sort_values('timestamp').reset_index(drop=True)
            
            print(f"✅ 成功載入 {len(self.original_data)} 筆原始數據")
            print(f"📅 時間範圍：{self.original_data['timestamp'].min()} 到 {self.original_data['timestamp'].max()}")
            
            # 顯示數據採樣頻率
            if len(self.original_data) > 1:
                time_diff = (self.original_data.iloc[1]['timestamp'] - self.original_data.iloc[0]['timestamp']).total_seconds()
                print(f"📊 數據採樣間隔：{time_diff:.0f} 秒")
            
            return True
        except FileNotFoundError:
            print(f"❌ 找不到文件 {csv_path}")
            print("💡 請確保文件在當前目錄下，或提供正確的文件路徑")
            return False
        except Exception as e:
            print(f"❌ 載入數據時發生錯誤：{e}")
            return False
    
    def _calculate_time_diff(self, data):
        """
        計算相鄰記錄之間的時間差（秒）
        """
        time_diffs = []
        
        for i in range(len(data)):
            if i == 0:
                time_diffs.append(None)  # 第一筆記錄沒有前一筆
            else:
                time_diff = (data.iloc[i]['timestamp'] - data.iloc[i-1]['timestamp']).total_seconds()
                time_diffs.append(time_diff)
        
        return time_diffs

    def _print_time_diff_stats(self, data):
        """
        顯示時間差統計資訊
        """
        valid_diffs = data['time_diff_seconds'].dropna()
        
        if len(valid_diffs) > 0:
            print(f"\n⏱️  時間間隔統計：")
            print(f"   標準間隔（900秒）：{(valid_diffs == 900).sum()} 筆")
            print(f"   非標準間隔：{(valid_diffs != 900).sum()} 筆")
            print(f"   平均間隔：{valid_diffs.mean():.1f} 秒")
            print(f"   最大間隔：{valid_diffs.max():.1f} 秒")
            print(f"   最小間隔：{valid_diffs.min():.1f} 秒")
            
            # 分析缺失數據模式
            gaps = valid_diffs[valid_diffs > 900]
            if len(gaps) > 0:
                print(f"   發現 {len(gaps)} 個數據缺失間隔")
                print(f"   最大缺失：{gaps.max():.0f} 秒 ({gaps.max()/3600:.1f} 小時)")

    def _generate_missing_columns(self, data):
        """
        根據 power 值生成缺失的分類欄位
        """
        # 使用簡單的閾值分類（您可以根據需要調整）
        data['is_phantom_load'] = data['power'] <= 35
        data['is_light_use'] = (data['power'] > 35) & (data['power'] <= 75)
        data['is_regular_use'] = data['power'] > 75
        
        # 生成 power_state 欄位
        conditions = [
            data['is_phantom_load'],
            data['is_light_use'],
            data['is_regular_use']
        ]
        choices = ['phantom load', 'light use', 'regular use']
        data['power_state'] = np.select(conditions, choices, default='unknown')
        
        print("✅ 已根據 power 值自動生成分類欄位")
        return data
    
    def analyze_patterns(self):
        """
        分析原始數據的使用模式
        """
        if self.original_data is None:
            raise ValueError("請先載入原始數據")
        
        data = self.original_data.copy()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # 計算時間差
        data['time_diff_seconds'] = self._calculate_time_diff(data)
        
        # 檢查並創建必要的欄位
        required_columns = ['is_phantom_load', 'is_light_use', 'is_regular_use', 'power_state']
        for col in required_columns:
            if col not in data.columns:
                print(f"⚠️  欄位 '{col}' 不存在，將根據 power 值自動生成")
                data = self._generate_missing_columns(data)
                break
        
        data['hour'] = data['timestamp'].dt.hour
        data['weekday'] = data['timestamp'].dt.weekday
        data['is_weekend'] = data['weekday'].isin([5, 6])
        data['time_slot'] = data['hour'].apply(self._get_time_slot)
        
        print("🔍 開始分析數據模式...")
        
        # 顯示時間差統計
        self._print_time_diff_stats(data)
        
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
        print(f"📉 分析數據缺失模式...")
        
        # 分析時間間隔，識別可能的缺失模式
        data_sorted = data.sort_values('timestamp')
        
        # 找出非標準間隔（不是900秒的）
        valid_time_diffs = data_sorted['time_diff_seconds'].dropna()
        non_standard = data_sorted[data_sorted['time_diff_seconds'] > 900]
        
        print(f"   總時間間隔：{len(valid_time_diffs)} 個")
        print(f"   非標準間隔：{len(non_standard)} 個")
        
        # 按小時分析缺失率
        for hour in range(24):
            hour_data = data_sorted[data_sorted['hour'] == hour]
            hour_missing = non_standard[non_standard['hour'] == hour]
            
            if len(hour_data) > 0:
                missing_prob = len(hour_missing) / len(hour_data)
                self.missing_patterns[hour] = min(0.15, missing_prob)  # 限制最大缺失率15%
            else:
                self.missing_patterns[hour] = 0.02  # 預設缺失率2%
        
        # 顯示缺失率最高的時段
        sorted_missing = sorted(self.missing_patterns.items(), key=lambda x: x[1], reverse=True)
        print(f"   缺失率最高的時段：")
        for hour, rate in sorted_missing[:5]:
            print(f"     {hour:02d}:00 - {rate:.1%}")
        
        print(f"   完成各時段的數據缺失模式分析")
    
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
        往前生成擴展數據（在原始數據之前增加歷史數據）
        """
        if not self.hourly_patterns:
            self.analyze_patterns()
        
        # 確定新的時間範圍（往前生成）
        original_start = self.original_data['timestamp'].min()
        new_start = original_start - timedelta(weeks=weeks)
        new_end = original_start
        
        print(f"🚀 開始往前生成 {weeks} 週的歷史數據...")
        print(f"📅 新時間範圍：{new_start.date()} 到 {new_end.date()}")
        print(f"🔗 將與原始數據（{original_start.date()} 開始）連接")
        
        # 生成15分鐘間隔的時間序列
        time_range = pd.date_range(start=new_start, end=new_end, freq='15min')
        print(f"📝 預計生成：{len(time_range):,} 筆記錄")
        
        extended_records = []
        
        # 獲取原始數據開始時的狀態作為結束狀態
        target_end_state = self._get_original_start_state()
        prev_power_state = "phantom load"  # 初始狀態
        
        for i, timestamp in enumerate(time_range):
            # 計算時間差
            time_diff = 900.0 if i > 0 else None  # 15分鐘 = 900秒
            
            # 檢查是否應該模擬缺失數據
            if self._should_skip_record(timestamp):
                if i > 0 and extended_records:
                    # 更新上一筆記錄的時間差
                    extended_records[-1]['time_diff_seconds'] = extended_records[-1].get('time_diff_seconds', 0) + 900
                continue
            
            # 生成功率和狀態（隨著時間接近原始數據，狀態會趨向於target_end_state）
            power, power_state = self._generate_power_and_state_with_trend(
                timestamp, prev_power_state, target_end_state, new_start, new_end
            )
            prev_power_state = power_state
            
            # 創建記錄
            record = {
                'timestamp': timestamp,
                'power': power,
                'power_state': power_state,
                'is_phantom_load': power_state == 'phantom load',
                'is_off': False,
                'is_on': True,
                'is_light_use': power_state == 'light use',
                'is_regular_use': power_state == 'regular use',
                'time_diff_seconds': time_diff
            }
            
            extended_records.append(record)
        
        # 轉換為DataFrame
        extended_df = pd.DataFrame(extended_records)
        
        print(f"✅ 生成完成：{len(extended_df):,} 筆記錄")
        
        return extended_df

    def _get_original_start_state(self):
        """
        獲取原始數據開始時的主要狀態
        """
        if len(self.original_data) > 0:
            # 使用原始數據前100筆記錄的主要狀態
            first_records = self.original_data.head(100)
            
            if 'power_state' in first_records.columns:
                most_common_state = first_records['power_state'].mode().iloc[0]
                return most_common_state
            else:
                # 根據功率值推斷
                avg_power = first_records['power'].mean()
                if avg_power <= 35:
                    return 'phantom load'
                elif avg_power <= 75:
                    return 'light use'
                else:
                    return 'regular use'
        
        return "phantom load"  # 預設值
    
    def _should_skip_record(self, timestamp):
        """
        根據學習到的缺失模式決定是否跳過記錄
        """
        hour = timestamp.hour
        missing_prob = self.missing_patterns.get(hour, 0.02)
        
        # 添加一些隨機性
        return np.random.random() < missing_prob
    
    def _generate_power_and_state_with_trend(self, timestamp, prev_state, target_state, start_time, end_time):
        """
        生成功率和狀態，隨時間趨向目標狀態
        """
        hour = timestamp.hour
        weekday = timestamp.weekday()
        is_weekend = weekday >= 5
        
        # 獲取該小時的基礎模式
        if hour in self.hourly_patterns:
            hour_pattern = self.hourly_patterns[hour]
        else:
            hour_pattern = {
                'power_mean': self.original_data['power'].mean(),
                'power_std': self.original_data['power'].std(),
                'phantom_prob': 0.76,
                'light_prob': 0.20,
                'regular_prob': 0.04
            }
        
        # 計算時間進度（0=開始，1=結束）
        progress = (timestamp - start_time).total_seconds() / (end_time - start_time).total_seconds()
        
        # 狀態概率調整（隨時間趨向目標狀態）
        state_probs = self._get_state_probabilities_with_trend(hour_pattern, prev_state, target_state, progress)
        
        # 週末調整因子
        weekend_factor = 1.0
        if is_weekend and self.weekday_patterns:
            weekend_factor = self.weekday_patterns.get(True, {}).get('power_mean', 1) / \
                           self.weekday_patterns.get(False, {}).get('power_mean', 1)
        
        # 隨機選擇狀態
        rand = np.random.random()
        if rand < state_probs['phantom']:
            power_state = 'phantom load'
            power = np.random.normal(18, 2)
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

    def _get_state_probabilities_with_trend(self, hour_pattern, prev_state, target_state, progress):
        """
        基於小時模式、前一狀態和目標狀態計算狀態概率
        progress: 0-1，表示時間進度
        """
        base_probs = {
            'phantom': hour_pattern.get('phantom_prob', 0.76),
            'light': hour_pattern.get('light_prob', 0.20),
            'regular': hour_pattern.get('regular_prob', 0.04)
        }
        
        # 狀態持續性調整
        persistence_factor = 0.3
        if prev_state in ['phantom load', 'phantom']:
            base_probs['phantom'] += persistence_factor * (1 - base_probs['phantom'])
        elif prev_state in ['light use', 'light']:
            base_probs['light'] += persistence_factor * (1 - base_probs['light'])
        elif prev_state in ['regular use', 'regular']:
            base_probs['regular'] += persistence_factor * (1 - base_probs['regular'])
        
        # 目標狀態趨勢調整（隨時間增強）
        trend_factor = 0.2 * progress  # 最大20%的調整
        if target_state in ['phantom load', 'phantom']:
            base_probs['phantom'] += trend_factor
        elif target_state in ['light use', 'light']:
            base_probs['light'] += trend_factor
        elif target_state in ['regular use', 'regular']:
            base_probs['regular'] += trend_factor
        
        # 歸一化
        total = sum(base_probs.values())
        return {k: v/total for k, v in base_probs.items()}
    
    def save_extended_data(self, extended_df, filename="data/complete_power_data_with_history.csv"):
        """
        保存擴展數據並與原始數據合併
        """
        print("🔗 合併擴展數據與原始數據...")
        
        # 確保原始數據有必要的欄位
        original_copy = self.original_data.copy()
        if 'power_state' not in original_copy.columns:
            original_copy = self._generate_missing_columns(original_copy)
        
        # 合併數據（擴展數據在前，原始數據在後）
        combined_df = pd.concat([extended_df, original_copy], ignore_index=True)
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        # 重新計算時間差（確保連接處正確）
        combined_df['time_diff_seconds'] = self._calculate_time_diff(combined_df)
        
        print(f"✅ 合併完成：擴展數據 {len(extended_df)} 筆 + 原始數據 {len(self.original_data)} 筆 = 總計 {len(combined_df)} 筆")
        
        # 確保data目錄存在
        import os
        os.makedirs('data', exist_ok=True)
        
        # 保存檔案
        combined_df.to_csv(filename, index=False)
        print(f"\n💾 完整數據已保存到：{filename}")
        
        # 顯示詳細統計
        self._print_comprehensive_statistics(combined_df)
        
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
            orig_phantom_rate = self.original_data.get('is_phantom_load', pd.Series()).mean()
            new_phantom_rate = df['is_phantom_load'].mean()
            
            print(f"\n🔄 與原始數據對比：")
            print(f"   原始平均功率：{orig_mean:.1f}W → 擴展平均功率：{df['power'].mean():.1f}W")
            if not pd.isna(orig_phantom_rate):
                print(f"   原始Phantom Load率：{orig_phantom_rate:.1%} → 擴展Phantom Load率：{new_phantom_rate:.1%}")
        
        print("="*80)


# 使用示例
def main(csv_path):
    """
    主函數 - 往前擴展數據到2個月
    """
    print("🚀 電力數據往前擴展器")
    print("="*50)
    
    # 初始化擴展器
    expander = PowerDataExpander(csv_path)
    
    # 分析模式
    expander.analyze_patterns()
    
    # 往前生成8週（2個月）的歷史數據
    extended_data = expander.generate_extended_data(weeks=8)
    
    # 保存數據（自動合併原始數據）
    filename = expander.save_extended_data(extended_data)
    
    print(f"\n🎉 擴展完成！請查看 {filename}")
    print("💡 您現在有完整的2個月歷史數據可以用於訓練系統")

# 最簡單的使用方式
if __name__ == "__main__":
    csv_path = "data/historical_power.csv"  # 替換為您的CSV檔案路徑
    main(csv_path)