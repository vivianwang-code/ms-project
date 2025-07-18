# # Confidence Score Module
# # Based on peak hour detection, sleep hour detection, and data completeness

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime, timedelta
# import warnings
# warnings.filterwarnings('ignore')

# # 設置中文字體
# import matplotlib
# matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
# matplotlib.rcParams['axes.unicode_minus'] = False

# class ConfidenceScoreModule:

#     def __init__(self):
#         self.time_slots = 96  # 96個時段，每15分鐘一個 (24*4)
#         self.time_outliers_data = {}
#         self.data_completeness_score = {}
#         self.peak_hours = []
#         self.sleep_hours = []
#         self.confidence_matrix = {}
#         self.data_quality_report = {}
        
#         # 載入您提供的time outliers統計數據
#         self._load_time_outliers_data()

#     def _load_time_outliers_data(self):
#         """載入time outliers統計數據"""
#         print("==== Loading Time Outliers Data ====")
        
#         # 您提供的統計數據
#         raw_data = """
#         00:00,21 00:15,21 00:30,21 00:45,21 01:00,23 01:15,23 01:30,21 01:45,22 02:00,27 02:15,31 02:30,30 02:45,36 
#         03:00,36 03:15,35 03:30,33 03:45,37 04:00,36 04:15,36 04:30,37 04:45,36 05:00,37 05:15,36 05:30,38 05:45,34 
#         06:00,36 06:15,38 06:30,37 06:45,38 07:00,40 07:15,39 07:30,37 07:45,40 08:00,37 08:15,40 08:30,39 08:45,39 
#         09:00,38 09:15,39 09:30,37 09:45,38 10:00,36 10:15,36 10:30,35 10:45,33 11:00,38 11:15,37 11:30,36 11:45,36 
#         12:00,36 12:15,37 12:30,37 12:45,36 13:00,36 13:15,35 13:30,39 13:45,41 14:00,43 14:15,42 14:30,44 14:45,45 
#         15:00,44 15:15,45 15:30,43 15:45,40 16:00,40 16:15,38 16:30,38 16:45,37 17:00,37 17:15,36 17:30,28 17:45,23 
#         18:00,24 18:15,21 18:30,20 18:45,20 19:00,19 19:15,20 19:30,18 19:45,18 20:00,18 20:15,18 20:30,18 20:45,21 
#         21:00,22 21:15,21 21:30,21 21:45,21 22:00,22 22:15,22 22:30,23 22:45,22 23:00,21 23:15,21 23:30,21 23:45,21
#         """
        
#         # 解析數據
#         entries = raw_data.replace('\n', ' ').split()
        
#         for entry in entries:
#             if ',' in entry:
#                 time_str, missing_count = entry.strip().split(',')
#                 try:
#                     # 轉換時間格式
#                     hour, minute = map(int, time_str.split(':'))
#                     time_slot = hour * 4 + minute // 15
#                     missing_count = int(missing_count)
                    
#                     self.time_outliers_data[time_slot] = {
#                         'time': time_str,
#                         'missing_count': missing_count,
#                         'hour': hour,
#                         'minute': minute
#                     }
#                 except:
#                     continue
        
#         print(f"✓ Loaded time outliers data for {len(self.time_outliers_data)} time slots")
        
#         # 計算統計信息
#         missing_counts = [data['missing_count'] for data in self.time_outliers_data.values()]
#         print(f"Missing data statistics:")
#         print(f"  Min: {min(missing_counts)} occurrences")
#         print(f"  Max: {max(missing_counts)} occurrences") 
#         print(f"  Mean: {np.mean(missing_counts):.1f} occurrences")
#         print(f"  Median: {np.median(missing_counts):.1f} occurrences")

#     def detect_peak_hours(self):
#         """檢測使用高峰時段"""
#         print("==== Detecting Peak Hours ====")
        
#         if not self.time_outliers_data:
#             print("❌ No time outliers data available")
#             return []
        
#         missing_counts = [data['missing_count'] for data in self.time_outliers_data.values()]
        
#         # 使用統計方法檢測peak hours
#         # 缺失數據少 = 使用頻繁 = peak hour
#         q25 = np.percentile(missing_counts, 25)
#         threshold = q25  # 使用第一四分位數作為閾值
        
#         peak_hours = []
#         for time_slot, data in self.time_outliers_data.items():
#             if data['missing_count'] <= threshold:
#                 peak_hours.append({
#                     'time_slot': time_slot,
#                     'time': data['time'],
#                     'hour': data['hour'],
#                     'missing_count': data['missing_count'],
#                     'usage_intensity': 1 - (data['missing_count'] - min(missing_counts)) / (max(missing_counts) - min(missing_counts))
#                 })
        
#         # 按時間排序
#         peak_hours.sort(key=lambda x: x['time_slot'])
#         self.peak_hours = peak_hours
        
#         print(f"✓ Detected {len(peak_hours)} peak hour slots")
#         print(f"Peak hours threshold: ≤ {threshold} missing occurrences")
        
#         # 顯示連續的peak hour時段
#         if peak_hours:
#             print("Peak hour periods:")
#             current_start = peak_hours[0]['hour']
#             current_end = peak_hours[0]['hour']
            
#             for i, slot in enumerate(peak_hours[1:], 1):
#                 if slot['hour'] == current_end or slot['hour'] == current_end + 1:
#                     current_end = slot['hour']
#                 else:
#                     print(f"  {current_start:02d}:00 - {current_end:02d}:45 (強度: {np.mean([p['usage_intensity'] for p in peak_hours if current_start <= p['hour'] <= current_end]):.2f})")
#                     current_start = slot['hour']
#                     current_end = slot['hour']
                
#                 if i == len(peak_hours) - 1:
#                     print(f"  {current_start:02d}:00 - {current_end:02d}:45 (強度: {np.mean([p['usage_intensity'] for p in peak_hours if current_start <= p['hour'] <= current_end]):.2f})")
        
#         return peak_hours

#     def detect_sleep_hours(self):
#         """檢測睡眠/低使用時段"""
#         print("==== Detecting Sleep Hours ====")
        
#         if not self.time_outliers_data:
#             print("❌ No time outliers data available")
#             return []
        
#         missing_counts = [data['missing_count'] for data in self.time_outliers_data.values()]
        
#         # 使用統計方法檢測sleep hours
#         # 缺失數據多 = 使用少 = sleep hour
#         q75 = np.percentile(missing_counts, 75)
#         threshold = q75  # 使用第三四分位數作為閾值
        
#         sleep_hours = []
#         for time_slot, data in self.time_outliers_data.items():
#             if data['missing_count'] >= threshold:
#                 sleep_hours.append({
#                     'time_slot': time_slot,
#                     'time': data['time'],
#                     'hour': data['hour'],
#                     'missing_count': data['missing_count'],
#                     'sleep_intensity': (data['missing_count'] - min(missing_counts)) / (max(missing_counts) - min(missing_counts))
#                 })
        
#         # 按時間排序
#         sleep_hours.sort(key=lambda x: x['time_slot'])
#         self.sleep_hours = sleep_hours
        
#         print(f"✓ Detected {len(sleep_hours)} sleep hour slots")
#         print(f"Sleep hours threshold: ≥ {threshold} missing occurrences")
        
#         # 顯示連續的sleep hour時段
#         if sleep_hours:
#             print("Sleep hour periods:")
#             current_start = sleep_hours[0]['hour']
#             current_end = sleep_hours[0]['hour']
            
#             for i, slot in enumerate(sleep_hours[1:], 1):
#                 if slot['hour'] == current_end or slot['hour'] == current_end + 1:
#                     current_end = slot['hour']
#                 else:
#                     print(f"  {current_start:02d}:00 - {current_end:02d}:45 (強度: {np.mean([s['sleep_intensity'] for s in sleep_hours if current_start <= s['hour'] <= current_end]):.2f})")
#                     current_start = slot['hour']
#                     current_end = slot['hour']
                
#                 if i == len(sleep_hours) - 1:
#                     print(f"  {current_start:02d}:00 - {current_end:02d}:45 (強度: {np.mean([s['sleep_intensity'] for s in sleep_hours if current_start <= s['hour'] <= current_end]):.2f})")
        
#         return sleep_hours

#     def calculate_data_completeness_score(self):
#         """計算數據完整性分數"""
#         print("==== Calculating Data Completeness Score ====")
        
#         if not self.time_outliers_data:
#             print("❌ No time outliers data available")
#             return {}
        
#         missing_counts = [data['missing_count'] for data in self.time_outliers_data.values()]
#         min_missing = min(missing_counts)
#         max_missing = max(missing_counts)
        
#         # 計算每個時段的數據完整性分數
#         for time_slot, data in self.time_outliers_data.items():
#             missing_count = data['missing_count']
            
#             # 完整性分數：缺失越少，分數越高
#             if max_missing > min_missing:
#                 completeness = 1.0 - (missing_count - min_missing) / (max_missing - min_missing)
#             else:
#                 completeness = 1.0
            
#             # 確保分數在[0,1]範圍內
#             completeness = max(0.0, min(1.0, completeness))
            
#             self.data_completeness_score[time_slot] = {
#                 'time': data['time'],
#                 'hour': data['hour'],
#                 'missing_count': missing_count,
#                 'completeness_score': completeness,
#                 'data_quality': 'high' if completeness >= 0.8 else 'medium' if completeness >= 0.5 else 'low'
#             }
        
#         # 統計各質量等級的時段數
#         quality_counts = {'high': 0, 'medium': 0, 'low': 0}
#         for score_data in self.data_completeness_score.values():
#             quality_counts[score_data['data_quality']] += 1
        
#         print(f"✓ Calculated completeness scores for {len(self.data_completeness_score)} time slots")
#         print(f"Data quality distribution:")
#         print(f"  High quality (≥0.8): {quality_counts['high']} slots")
#         print(f"  Medium quality (0.5-0.8): {quality_counts['medium']} slots")
#         print(f"  Low quality (<0.5): {quality_counts['low']} slots")
        
#         return self.data_completeness_score

#     def calculate_pattern_consistency_score(self):
#         """計算模式一致性分數"""
#         print("==== Calculating Pattern Consistency Score ====")
        
#         if not self.peak_hours or not self.sleep_hours:
#             print("⚠️  Peak hours or sleep hours not detected, using default consistency")
#             return 0.5
        
#         # 檢查peak hours和sleep hours的時間分佈是否合理
#         peak_hour_set = set(p['hour'] for p in self.peak_hours)
#         sleep_hour_set = set(s['hour'] for s in self.sleep_hours)
        
#         # 計算模式一致性
#         # 1. Peak hours應該主要在晚上 (18-23)
#         evening_peak_hours = sum(1 for hour in peak_hour_set if 18 <= hour <= 23)
#         evening_consistency = evening_peak_hours / len(peak_hour_set) if peak_hour_set else 0
        
#         # 2. Sleep hours應該主要在深夜/凌晨 (0-7, 14-16)
#         night_sleep_hours = sum(1 for hour in sleep_hour_set if hour <= 7 or 14 <= hour <= 16)
#         night_consistency = night_sleep_hours / len(sleep_hour_set) if sleep_hour_set else 0
        
#         # 3. Peak和sleep時段不應該重疊
#         overlap = len(peak_hour_set & sleep_hour_set)
#         overlap_penalty = overlap / max(len(peak_hour_set), len(sleep_hour_set)) if (peak_hour_set or sleep_hour_set) else 0
        
#         # 綜合一致性分數
#         pattern_consistency = (evening_consistency * 0.4 + night_consistency * 0.4 + (1 - overlap_penalty) * 0.2)
        
#         print(f"Pattern consistency analysis:")
#         print(f"  Evening peak consistency: {evening_consistency:.2f}")
#         print(f"  Night sleep consistency: {night_consistency:.2f}")
#         print(f"  Peak-sleep overlap penalty: {overlap_penalty:.2f}")
#         print(f"  Overall pattern consistency: {pattern_consistency:.2f}")
        
#         return pattern_consistency

#     def calculate_confidence_score(self, timestamp):
#         """計算指定時間點的置信度分數（不使用三角隸屬函數）"""
#         try:
#             # 提取時間特徵
#             hour = timestamp.hour
#             minute = timestamp.minute
#             time_slot = hour * 4 + minute // 15
            
#             result = {
#                 'hour': hour,
#                 'minute': minute,
#                 'time_slot': time_slot,
#                 'timestamp': timestamp
#             }
            
#             # 1. 數據完整性分數
#             if time_slot in self.data_completeness_score:
#                 completeness = self.data_completeness_score[time_slot]['completeness_score']
#                 data_quality = self.data_completeness_score[time_slot]['data_quality']
#             else:
#                 completeness = 0.5
#                 data_quality = 'medium'
            
#             # 2. 時段類型識別
#             is_peak_hour = any(p['hour'] == hour for p in self.peak_hours)
#             is_sleep_hour = any(s['hour'] == hour for s in self.sleep_hours)
            
#             if is_peak_hour:
#                 time_pattern = 'peak'
#                 pattern_confidence = 0.9
#             elif is_sleep_hour:
#                 time_pattern = 'sleep'
#                 pattern_confidence = 0.8
#             else:
#                 time_pattern = 'normal'
#                 pattern_confidence = 0.6
            
#             # 3. 時間穩定性評估（基於相鄰時段的一致性）
#             adjacent_consistency = self._calculate_adjacent_consistency(time_slot)
            
#             # 4. 綜合置信度計算（統計方法，非模糊邏輯）
#             weights = {
#                 'completeness': 0.4,      # 數據完整性權重最高
#                 'pattern': 0.35,          # 模式識別權重
#                 'consistency': 0.25       # 時間一致性權重
#             }
            
#             confidence_score = (
#                 completeness * weights['completeness'] +
#                 pattern_confidence * weights['pattern'] +
#                 adjacent_consistency * weights['consistency']
#             )
            
#             # 5. 添加基於使用模式的調整
#             pattern_adjustment = self._get_pattern_adjustment(hour, is_peak_hour, is_sleep_hour)
#             confidence_score += pattern_adjustment
            
#             # 確保分數在[0,1]範圍內
#             confidence_score = max(0.0, min(1.0, confidence_score))
            
#             # 計算置信等級
#             if confidence_score >= 0.8:
#                 confidence_level = 'very_high'
#             elif confidence_score >= 0.6:
#                 confidence_level = 'high'
#             elif confidence_score >= 0.4:
#                 confidence_level = 'medium'
#             elif confidence_score >= 0.2:
#                 confidence_level = 'low'
#             else:
#                 confidence_level = 'very_low'
            
#             result.update({
#                 'confidence_score': confidence_score,
#                 'confidence_level': confidence_level,
#                 'data_completeness': completeness,
#                 'data_quality': data_quality,
#                 'time_pattern': time_pattern,
#                 'pattern_confidence': pattern_confidence,
#                 'adjacent_consistency': adjacent_consistency,
#                 'is_peak_hour': is_peak_hour,
#                 'is_sleep_hour': is_sleep_hour
#             })
            
#             return result
            
#         except Exception as e:
#             print(f"⚠️  Error calculating confidence score: {e}")
#             return {
#                 'confidence_score': 0.5,
#                 'confidence_level': 'medium',
#                 'error': str(e)
#             }

#     def _calculate_adjacent_consistency(self, time_slot):
#         """計算與相鄰時段的一致性"""
#         try:
#             if time_slot not in self.data_completeness_score:
#                 return 0.5
            
#             current_score = self.data_completeness_score[time_slot]['completeness_score']
#             adjacent_scores = []
            
#             # 檢查前後各2個時段
#             for offset in [-2, -1, 1, 2]:
#                 adj_slot = (time_slot + offset) % self.time_slots
#                 if adj_slot in self.data_completeness_score:
#                     adjacent_scores.append(self.data_completeness_score[adj_slot]['completeness_score'])
            
#             if not adjacent_scores:
#                 return 0.5
            
#             # 計算與相鄰時段的相似性
#             differences = [abs(current_score - adj_score) for adj_score in adjacent_scores]
#             avg_difference = np.mean(differences)
            
#             # 一致性 = 1 - 平均差異
#             consistency = 1.0 - avg_difference
            
#             return max(0.0, min(1.0, consistency))
            
#         except:
#             return 0.5

#     def _get_pattern_adjustment(self, hour, is_peak_hour, is_sleep_hour):
#         """基於使用模式的置信度調整"""
#         # 基於時間的額外調整
#         if is_peak_hour and 18 <= hour <= 23:
#             return 0.1  # 晚間peak hour，增加置信度
#         elif is_sleep_hour and (0 <= hour <= 7 or 14 <= hour <= 16):
#             return 0.05  # 合理的sleep hour，小幅增加置信度
#         elif is_peak_hour and not (18 <= hour <= 23):
#             return -0.05  # 非晚間的peak hour，略降置信度
#         elif is_sleep_hour and not (0 <= hour <= 7 or 14 <= hour <= 16):
#             return -0.1  # 非合理時段的sleep hour，降低置信度
#         else:
#             return 0.0  # 正常時段，無調整

#     # def plot_confidence_analysis(self):
#     #     """繪製置信度分析圖表"""
#     #     if not self.time_outliers_data:
#     #         print("❌ No data available for plotting")
#     #         return
        
#     #     # 準備數據
#     #     hours = []
#     #     missing_counts = []
#     #     completeness_scores = []
#     #     time_patterns = []
        
#     #     for time_slot in range(0, self.time_slots, 4):  # 每小時取一個點
#     #         if time_slot in self.time_outliers_data:
#     #             hour = time_slot // 4
#     #             hours.append(hour)
#     #             missing_counts.append(self.time_outliers_data[time_slot]['missing_count'])
                
#     #             if time_slot in self.data_completeness_score:
#     #                 completeness_scores.append(self.data_completeness_score[time_slot]['completeness_score'])
#     #             else:
#     #                 completeness_scores.append(0.5)
                
#     #             # 判斷時段類型
#     #             is_peak = any(p['hour'] == hour for p in self.peak_hours)
#     #             is_sleep = any(s['hour'] == hour for s in self.sleep_hours)
                
#     #             if is_peak:
#     #                 time_patterns.append('Peak')
#     #             elif is_sleep:
#     #                 time_patterns.append('Sleep')
#     #             else:
#     #                 time_patterns.append('Normal')
        
#     #     # 創建圖表
#     #     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
#     #     fig.suptitle('Confidence Score Analysis Dashboard', fontsize=16, fontweight='bold')
        
#     #     # 1. 缺失數據分佈
#     #     ax1.plot(hours, missing_counts, 'b-', linewidth=2, marker='o', markersize=4)
#     #     ax1.set_xlabel('Hour of Day')
#     #     ax1.set_ylabel('Missing Data Count')
#     #     ax1.set_title('Missing Data Distribution (24 Hours)')
#     #     ax1.grid(True, alpha=0.3)
#     #     ax1.set_xlim(0, 23)
        
#     #     # 標記peak和sleep時段
#     #     for p in self.peak_hours:
#     #         if p['hour'] in hours:
#     #             ax1.axvspan(p['hour']-0.3, p['hour']+0.3, alpha=0.2, color='green', label='Peak Hour' if p == self.peak_hours[0] else "")
        
#     #     for s in self.sleep_hours:
#     #         if s['hour'] in hours:
#     #             ax1.axvspan(s['hour']-0.3, s['hour']+0.3, alpha=0.2, color='red', label='Sleep Hour' if s == self.sleep_hours[0] else "")
        
#     #     ax1.legend()
        
#     #     # 2. 數據完整性分數
#     #     ax2.plot(hours, completeness_scores, 'g-', linewidth=2, marker='s', markersize=4)
#     #     ax2.set_xlabel('Hour of Day')
#     #     ax2.set_ylabel('Completeness Score')
#     #     ax2.set_title('Data Completeness Score (24 Hours)')
#     #     ax2.grid(True, alpha=0.3)
#     #     ax2.set_xlim(0, 23)
#     #     ax2.set_ylim(0, 1)
        
#     #     # 添加閾值線
#     #     ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='High Quality')
#     #     ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium Quality')
#     #     ax2.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Low Quality')
#     #     ax2.legend()
        
#     #     # 3. 時段類型分佈
#     #     pattern_counts = {'Peak': 0, 'Sleep': 0, 'Normal': 0}
#     #     for pattern in time_patterns:
#     #         pattern_counts[pattern] += 1
        
#     #     colors = ['green', 'red', 'blue']
#     #     ax3.pie(pattern_counts.values(), labels=pattern_counts.keys(), colors=colors, autopct='%1.1f%%', startangle=90)
#     #     ax3.set_title('Time Pattern Distribution')
        
#     #     # 4. 置信度熱力圖（24小時）
#     #     confidence_scores_24h = []
#     #     for hour in range(24):
#     #         test_time = datetime(2024, 6, 15, hour, 0)
#     #         result = self.calculate_confidence_score(test_time)
#     #         confidence_scores_24h.append(result['confidence_score'])
        
#     #     # 創建熱力圖數據
#     #     heatmap_data = np.array(confidence_scores_24h).reshape(1, -1)
        
#     #     im = ax4.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
#     #     ax4.set_xlabel('Hour of Day')
#     #     ax4.set_title('Confidence Score Heatmap')
#     #     ax4.set_xticks(range(0, 24, 4))
#     #     ax4.set_xticklabels([f'{h:02d}' for h in range(0, 24, 4)])
#     #     ax4.set_yticks([])
        
#     #     # 添加顏色條
#     #     plt.colorbar(im, ax=ax4, label='Confidence Score')
        
#     #     plt.tight_layout()
#     #     plt.show()
        
#     #     # 打印統計信息
#     #     print("="*60)
#     #     print("Confidence Score Analysis Summary")
#     #     print("="*60)
#     #     print(f"📊 Peak Hours: {len(self.peak_hours)} time slots")
#     #     if self.peak_hours:
#     #         # 修正：正確處理去重複的邏輯
#     #         unique_peak_hours = []
#     #         seen_hours = set()
#     #         for p in self.peak_hours:
#     #             if p['hour'] not in seen_hours:
#     #                 unique_peak_hours.append(p)
#     #                 seen_hours.add(p['hour'])
            
#     #         # 按小時排序
#     #         unique_peak_hours.sort(key=lambda x: x['hour'])
#     #         peak_hours_str = ", ".join([f"{p['hour']:02d}:xx" for p in unique_peak_hours])
#     #         print(f"   Hours: {peak_hours_str}")
        
#     #     print(f"📊 Sleep Hours: {len(self.sleep_hours)} time slots")
#     #     if self.sleep_hours:
#     #         # 修正：正確處理去重複的邏輯
#     #         unique_sleep_hours = []
#     #         seen_hours = set()
#     #         for s in self.sleep_hours:
#     #             if s['hour'] not in seen_hours:
#     #                 unique_sleep_hours.append(s)
#     #                 seen_hours.add(s['hour'])
            
#     #         # 按小時排序
#     #         unique_sleep_hours.sort(key=lambda x: x['hour'])
#     #         sleep_hours_str = ", ".join([f"{s['hour']:02d}:xx" for s in unique_sleep_hours])
#     #         print(f"   Hours: {sleep_hours_str}")
        
#     #     print(f"📊 Data Quality Distribution:")
#     #     for quality, count in pattern_counts.items():
#     #         print(f"   {quality}: {count} hours ({count/24*100:.1f}%)")
        
#     #     avg_confidence = np.mean(confidence_scores_24h)
#     #     print(f"📊 Average Confidence Score: {avg_confidence:.3f}")

#     def test_confidence_score_calculation(self, num_tests=5):
#         """測試置信度分數計算功能"""
#         print("==== Testing Confidence Score Calculation ====")
        
#         test_times = [
#             datetime(2024, 1, 15, 9, 0),   # 週一早上9點
#             datetime(2024, 1, 15, 14, 30), # 週一下午2:30
#             datetime(2024, 1, 15, 21, 0),  # 週一晚上9點 (可能是peak)
#             datetime(2024, 1, 13, 3, 15),  # 週六凌晨3:15 (可能是sleep)
#             datetime(2024, 1, 13, 19, 45), # 週六晚上7:45 (可能是peak)
#         ]
        
#         test_results = []
        
#         for i, test_time in enumerate(test_times[:num_tests]):
#             try:
#                 result = self.calculate_confidence_score(test_time)
                
#                 day_type = "Weekend" if test_time.weekday() >= 5 else "Weekday"
#                 print(f"\nTest {i+1}: {test_time.strftime('%Y-%m-%d %H:%M')} ({day_type})")
#                 print(f"  Confidence Score: {result['confidence_score']:.3f}")
#                 print(f"  Confidence Level: {result['confidence_level']}")
#                 print(f"  Time Pattern: {result['time_pattern']}")
#                 print(f"  Data Quality: {result['data_quality']}")
#                 print(f"  Data Completeness: {result['data_completeness']:.3f}")
#                 print(f"  Is Peak Hour: {result['is_peak_hour']}")
#                 print(f"  Is Sleep Hour: {result['is_sleep_hour']}")
                
#                 test_results.append({
#                     'time': test_time,
#                     'confidence_score': result['confidence_score'],
#                     'confidence_level': result['confidence_level'],
#                     'time_pattern': result['time_pattern']
#                 })
                
#             except Exception as e:
#                 print(f"⚠️  Error in test {i+1}: {e}")
#                 test_results.append({
#                     'time': test_time,
#                     'confidence_score': 0.5,
#                     'confidence_level': 'medium',
#                     'time_pattern': 'unknown'
#                 })
        
#         return test_results

#     def comprehensive_evaluation(self):
#         """完整的系統評估"""
#         print("\n" + "="*60)
#         print("CONFIDENCE SCORE MODULE - COMPREHENSIVE EVALUATION")
#         print("="*60)
        
#         # 1. 數據完整性評估
#         if self.data_completeness_score:
#             completeness_scores = [data['completeness_score'] for data in self.data_completeness_score.values()]
#             avg_completeness = np.mean(completeness_scores)
#             min_completeness = np.min(completeness_scores)
#             max_completeness = np.max(completeness_scores)
            
#             print(f"\n1. Data Completeness Assessment:")
#             print(f"   Average Completeness: {avg_completeness:.3f}")
#             print(f"   Range: {min_completeness:.3f} - {max_completeness:.3f}")
#             print(f"   Data Quality Distribution:")
            
#             quality_counts = {'high': 0, 'medium': 0, 'low': 0}
#             for data in self.data_completeness_score.values():
#                 quality_counts[data['data_quality']] += 1
            
#             for quality, count in quality_counts.items():
#                 percentage = count / len(self.data_completeness_score) * 100
#                 print(f"     {quality.capitalize()}: {count} slots ({percentage:.1f}%)")
        
#         # 2. 模式檢測評估
#         print(f"\n2. Pattern Detection Assessment:")
#         print(f"   Peak Hours Detected: {len(self.peak_hours)} time slots")
#         print(f"   Sleep Hours Detected: {len(self.sleep_hours)} time slots")
        
#         # 計算模式覆蓋率
#         total_pattern_coverage = (len(self.peak_hours) + len(self.sleep_hours)) / self.time_slots * 100
#         print(f"   Pattern Coverage: {total_pattern_coverage:.1f}% of all time slots")
        
#         # 3. 模式一致性評估
#         pattern_consistency = self.calculate_pattern_consistency_score()
#         print(f"\n3. Pattern Consistency: {pattern_consistency:.3f}")
        
#         # 4. 置信度分數測試
#         print(f"\n4. Confidence Score Tests:")
#         test_scenarios = [
#             (datetime(2024, 1, 15, 9, 0), (0.4, 0.8), '工作日早上'),
#             (datetime(2024, 1, 15, 15, 0), (0.2, 0.6), '工作日下午'),  # 可能是sleep時段
#             (datetime(2024, 1, 15, 21, 0), (0.6, 1.0), '工作日晚上'),  # 可能是peak時段
#             (datetime(2024, 1, 13, 3, 0), (0.3, 0.7), '週末凌晨'),   # 可能是sleep時段
#             (datetime(2024, 1, 13, 19, 0), (0.5, 0.9), '週末晚上'),  # 可能是peak時段
#         ]
        
#         passed_tests = 0
#         for test_time, expected_range, desc in test_scenarios:
#             try:
#                 result = self.calculate_confidence_score(test_time)
#                 score = result['confidence_score']
                
#                 is_reasonable = expected_range[0] <= score <= expected_range[1]
                
#                 if is_reasonable:
#                     passed_tests += 1
#                     status = '✓'
#                 else:
#                     status = '❌'
                
#                 print(f"   {status} {desc}: {score:.3f} (期望: {expected_range}) - {result['time_pattern']}")
#             except Exception as e:
#                 print(f"   ❌ {desc}: Error - {e}")
        
#         # 5. 最終評分
#         print(f"\n=== FINAL ASSESSMENT ===")
        
#         # 計算各項分數
#         data_quality_score = avg_completeness if self.data_completeness_score else 0
#         pattern_detection_score = min(1.0, total_pattern_coverage / 50)  # 50%覆蓋率為滿分
#         test_pass_rate = passed_tests / len(test_scenarios)
        
#         overall_score = (data_quality_score * 0.4 + pattern_consistency * 0.3 + 
#                         pattern_detection_score * 0.2 + test_pass_rate * 0.1)
        
#         print(f"Data Quality Score: {data_quality_score:.2f}")
#         print(f"Pattern Consistency: {pattern_consistency:.2f}")
#         print(f"Pattern Detection Score: {pattern_detection_score:.2f}")
#         print(f"Test Pass Rate: {test_pass_rate:.2f}")
#         print(f"Overall System Quality: {overall_score:.2f}")
        
#         if overall_score >= 0.8:
#             print("🎉 System Quality: Excellent")
#         elif overall_score >= 0.6:
#             print("✅ System Quality: Good") 
#         elif overall_score >= 0.4:
#             print("⚠️  System Quality: Acceptable")
#         else:
#             print("❌ System Quality: Needs Improvement")

#     def run_complete_analysis(self):
#         """運行完整分析"""
#         print("="*80)
#         print("CONFIDENCE SCORE MODULE - COMPLETE ANALYSIS")
#         print("="*80)

#         # 1. 檢測高峰時段
#         print("\n" + "-"*50)
#         self.detect_peak_hours()
        
#         # 2. 檢測睡眠時段
#         print("\n" + "-"*50)
#         self.detect_sleep_hours()
        
#         # 3. 計算數據完整性分數
#         print("\n" + "-"*50)
#         self.calculate_data_completeness_score()

#         # 4. 測試置信度計算
#         print("\n" + "-"*50)
#         test_results = self.test_confidence_score_calculation()

#         # 5. 綜合評估
#         print("\n" + "-"*50)
#         self.comprehensive_evaluation()

#         # 6. 繪製分析圖表
#         print("\n" + "-"*50)
#         print("==== Plotting Confidence Analysis Dashboard ====")
#         # self.plot_confidence_analysis()

#         print("\n" + "="*80)
#         print("✅ ANALYSIS COMPLETE - Confidence Score system ready!")
#         print("="*80)

#         return {
#             'peak_hours': self.peak_hours,
#             'sleep_hours': self.sleep_hours,
#             'data_completeness_score': self.data_completeness_score,
#             'time_outliers_data': self.time_outliers_data,
#             'test_results': test_results
#         }

# # 使用示例
# if __name__ == "__main__":
#     # 初始化置信度分數模組
#     confidence_module = ConfidenceScoreModule()
    
#     # 運行完整分析
#     result = confidence_module.run_complete_analysis()
    
#     # 單獨測試置信度分數計算
#     if result:
#         print("\n" + "="*50)
#         print("TESTING INDIVIDUAL CONFIDENCE SCORE CALCULATIONS")
#         print("="*50)
        
#         # 測試幾個特定時間點
#         test_times = [
#             datetime(2024, 6, 15, 8, 30),   # 週六早上8:30
#             datetime(2024, 6, 17, 19, 45),  # 週一晚上7:45 (peak hour)
#             datetime(2024, 6, 20, 15, 15),  # 週四下午3:15 (sleep hour)
#             datetime(2024, 6, 18, 3, 0),    # 週二凌晨3點 (sleep hour)
#         ]
        
#         for test_time in test_times:
#             result = confidence_module.calculate_confidence_score(test_time)
#             day_type = "Weekend" if test_time.weekday() >= 5 else "Weekday"
            
#             print(f"\n時間: {test_time.strftime('%Y-%m-%d %H:%M')} ({day_type})")
#             print(f"置信度分數: {result['confidence_score']:.3f}")
#             print(f"置信度等級: {result['confidence_level']}")
#             print(f"時段模式: {result['time_pattern']}")
#             print(f"數據質量: {result['data_quality']}")
            
#             # 提供解釋
#             if result['confidence_score'] >= 0.8:
#                 explanation = "🟢 置信度很高，數據可靠性極佳"
#             elif result['confidence_score'] >= 0.6:
#                 explanation = "🟡 置信度高，數據可靠"
#             elif result['confidence_score'] >= 0.4:
#                 explanation = "🟠 置信度中等，數據基本可信"
#             else:
#                 explanation = "🔴 置信度較低，建議謹慎使用數據"
            
#             print(f"解釋: {explanation}")
        
#         print(f"\n💡 提示：如果想重新查看置信度分析圖表，可以運行：")
#         # print(f"confidence_module.plot_confidence_analysis()")


# Improved Confidence Score Module
# Based on time reasonableness rather than historical usage patterns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ConfidenceScoreModule:

    def __init__(self):
        self.time_slots = 96  # 96個時段，每15分鐘一個 (24*4)
        
        # 重新定義：基於時間合理性而非歷史數據
        self.time_reasonableness_profiles = {}
        self.confidence_matrix = {}
        self.weekday_patterns = {}
        self.weekend_patterns = {}
        
        # 初始化時間合理性配置
        self._initialize_time_reasonableness()

    def _initialize_time_reasonableness(self):
        """初始化基於生理和社會合理性的時間配置"""
        print("==== Initializing Time Reasonableness Profiles ====")
        
        # 定義不同時段的使用合理性（0.0-1.0）
        # 基於人類作息和社會規範，而非歷史數據
        
        self.time_reasonableness_profiles = {
            # 深夜到凌晨 (00:00-05:59) - 睡眠時間
            'deep_night': {
                'hours': list(range(0, 6)),
                'base_confidence': 0.2,  # 很低，應該睡覺
                'description': '深夜睡眠時間，不建議使用電子設備'
            },
            
            # 早晨 (06:00-08:59) - 起床準備時間
            'early_morning': {
                'hours': list(range(6, 9)),
                'base_confidence': 0.5,  # 中等，適度使用
                'description': '早晨時光，適度使用'
            },
            
            # 上午 (09:00-11:59) - 工作/學習時間
            'morning': {
                'hours': list(range(9, 12)),
                'base_confidence': 0.4,  # 中低，應該專注工作
                'description': '上午工作時間，娛樂設備使用適度'
            },
            
            # 下午 (12:00-14:59) - 午餐和休息
            'afternoon': {
                'hours': list(range(12, 15)),
                'base_confidence': 0.6,  # 中高，午休娛樂
                'description': '午餐時間，適合放鬆'
            },
            
            # 下午晚些 (15:00-17:59) - 工作時間
            'late_afternoon': {
                'hours': list(range(15, 18)),
                'base_confidence': 0.4,  # 中低，工作時間
                'description': '下午工作時間'
            },
            
            # 傍晚 (18:00-20:59) - 放鬆娛樂時間
            'evening': {
                'hours': list(range(18, 21)),
                'base_confidence': 0.8,  # 高，主要娛樂時間
                'description': '傍晚娛樂時間，合理使用'
            },
            
            # 晚上 (21:00-21:59) - 準備休息
            'night': {
                'hours': [21],
                'base_confidence': 0.6,  # 中高，但開始準備休息
                'description': '晚上時間，開始準備休息'
            },
            
            # 深夜前 (22:00-23:59) - 應該準備睡覺
            'late_night': {
                'hours': [22, 23],
                'base_confidence': 0.3,  # 低，應該準備睡覺
                'description': '晚間時光，建議準備休息'
            }
        }
        
        print("✓ Time reasonableness profiles initialized")
        
        # 為每個時段創建詳細配置
        self._create_detailed_time_matrix()

    def _create_detailed_time_matrix(self):
        """創建詳細的時間合理性矩陣"""
        
        for day_type in ['weekday', 'weekend']:
            for hour in range(24):
                # 找到對應的時間段
                time_period = self._get_time_period(hour)
                base_confidence = self.time_reasonableness_profiles[time_period]['base_confidence']
                
                # 週末調整
                if day_type == 'weekend':
                    weekend_confidence = self._apply_weekend_adjustment(hour, base_confidence)
                else:
                    weekend_confidence = base_confidence
                
                # 為該小時的4個15分鐘時段創建配置
                for quarter in range(4):
                    time_slot = hour * 4 + quarter
                    minute = quarter * 15
                    
                    # 小幅時間內變化（模擬現實中的微調）
                    micro_adjustment = np.random.normal(0, 0.05)  # 小幅隨機調整
                    final_confidence = max(0.1, min(0.9, weekend_confidence + micro_adjustment))
                    
                    self.confidence_matrix[(day_type, time_slot)] = {
                        'hour': hour,
                        'minute': minute,
                        'time_period': time_period,
                        'base_confidence': base_confidence,
                        'adjusted_confidence': weekend_confidence,
                        'final_confidence': final_confidence,
                        'reasonableness_level': self._get_reasonableness_level(final_confidence)
                    }
        
        print(f"✓ Created detailed time matrix for {len(self.confidence_matrix)} time slots")

    def _get_time_period(self, hour):
        """根據小時獲取時間段"""
        for period_name, period_data in self.time_reasonableness_profiles.items():
            if hour in period_data['hours']:
                return period_name
        return 'unknown'

    def _apply_weekend_adjustment(self, hour, base_confidence):
        """應用週末調整"""
        # 週末的調整邏輯
        if 6 <= hour <= 9:  # 週末早晨可以晚起
            return min(0.9, base_confidence + 0.1)
        elif 22 <= hour <= 23:  # 週末可以稍微晚睡
            return min(0.9, base_confidence + 0.15)
        elif 0 <= hour <= 1:  # 週末深夜可能更寬鬆
            return min(0.9, base_confidence + 0.1)
        else:
            return base_confidence

    def _get_reasonableness_level(self, confidence_score):
        """根據置信度分數獲取合理性等級"""
        if confidence_score >= 0.7:
            return 'very_reasonable'
        elif confidence_score >= 0.5:
            return 'reasonable'
        elif confidence_score >= 0.3:
            return 'somewhat_reasonable'
        else:
            return 'unreasonable'

    def calculate_confidence_score(self, timestamp):
        """計算基於時間合理性的置信度分數"""
        try:
            # 提取時間特徵
            hour = timestamp.hour
            minute = timestamp.minute
            weekday = timestamp.weekday()
            is_weekend = weekday >= 5
            day_type = 'weekend' if is_weekend else 'weekday'
            time_slot = hour * 4 + minute // 15
            
            result = {
                'hour': hour,
                'minute': minute,
                'weekday': weekday,
                'day_type': day_type,
                'time_slot': time_slot,
                'timestamp': timestamp
            }
            
            # 獲取基礎配置
            if (day_type, time_slot) in self.confidence_matrix:
                config = self.confidence_matrix[(day_type, time_slot)]
                
                base_confidence = config['final_confidence']
                time_period = config['time_period']
                reasonableness_level = config['reasonableness_level']
                
            else:
                # 後備計算
                time_period = self._get_time_period(hour)
                base_confidence = self.time_reasonableness_profiles[time_period]['base_confidence']
                if day_type == 'weekend':
                    base_confidence = self._apply_weekend_adjustment(hour, base_confidence)
                reasonableness_level = self._get_reasonableness_level(base_confidence)
            
            # 應用更細緻的時間調整
            minute_adjustment = self._get_minute_level_adjustment(hour, minute)
            social_adjustment = self._get_social_context_adjustment(timestamp)
            
            # 計算最終置信度
            confidence_score = base_confidence + minute_adjustment + social_adjustment
            confidence_score = max(0.1, min(0.9, confidence_score))
            
            # 確定置信度等級
            if confidence_score >= 0.7:
                confidence_level = 'high'
            elif confidence_score >= 0.5:
                confidence_level = 'medium'
            elif confidence_score >= 0.3:
                confidence_level = 'low'
            else:
                confidence_level = 'very_low'
            
            # 生成解釋
            explanation = self._generate_explanation(time_period, confidence_level, day_type)
            
            result.update({
                'confidence_score': confidence_score,
                'confidence_level': confidence_level,
                'time_period': time_period,
                'reasonableness_level': reasonableness_level,
                'base_confidence': base_confidence,
                'minute_adjustment': minute_adjustment,
                'social_adjustment': social_adjustment,
                'explanation': explanation,
                'recommendation': self._get_recommendation(confidence_level)
            })
            
            return result
            
        except Exception as e:
            print(f"⚠️ Error calculating confidence score: {e}")
            return {
                'confidence_score': 0.5,
                'confidence_level': 'medium',
                'error': str(e)
            }

    def _get_minute_level_adjustment(self, hour, minute):
        """分鐘級別的微調"""
        # 在某些關鍵時間點進行微調
        if hour == 22 and minute >= 30:  # 22:30後進一步降低
            return -0.1
        elif hour == 23:  # 23點後大幅降低
            return -0.15
        elif hour in [0, 1, 2, 3, 4, 5] and minute == 0:  # 整點深夜更不合理
            return -0.05
        elif hour == 21 and minute >= 45:  # 21:45後開始準備休息
            return -0.05
        elif hour in [18, 19, 20] and minute in [15, 30, 45]:  # 晚間娛樂黃金時間
            return 0.05
        else:
            return 0.0

    def _get_social_context_adjustment(self, timestamp):
        """社會情境調整"""
        weekday = timestamp.weekday()
        hour = timestamp.hour
        
        # 工作日vs週末的不同標準
        if weekday < 5:  # 工作日
            if 9 <= hour <= 17:  # 工作時間，娛樂設備使用不太合理
                return -0.1
            elif hour in [22, 23]:  # 工作日晚上，需要早睡
                return -0.1
        else:  # 週末
            if hour in [22, 23]:  # 週末可以稍微晚一點
                return 0.05
            elif 10 <= hour <= 22:  # 週末白天，相對寬鬆
                return 0.05
        
        return 0.0

    def _generate_explanation(self, time_period, confidence_level, day_type):
        """生成人性化解釋"""
        explanations = {
            'deep_night': {
                'high': f'{day_type}深夜時間，雖然系統判斷可以使用，但建議考慮睡眠健康',
                'medium': f'{day_type}深夜時間，建議適度使用並準備休息',
                'low': f'{day_type}深夜時間，強烈建議休息，避免影響睡眠',
                'very_low': f'{day_type}深夜時間，非常不建議使用，應該睡覺'
            },
            'early_morning': {
                'high': f'{day_type}早晨時光，適合輕度使用',
                'medium': f'{day_type}早晨時間，可以適度使用',
                'low': f'{day_type}早晨時間，建議專注於準備一天的開始',
                'very_low': f'{day_type}清晨時間，建議繼續休息'
            },
            'evening': {
                'high': f'{day_type}傍晚娛樂時間，可以放心使用',
                'medium': f'{day_type}傍晚時間，適合放鬆娛樂',
                'low': f'{day_type}傍晚時間，適度使用即可',
                'very_low': f'{day_type}傍晚時間，建議其他活動'
            },
            'late_night': {
                'high': f'{day_type}晚間時間，建議準備逐漸減少使用',
                'medium': f'{day_type}晚間時間，適度使用並準備休息',
                'low': f'{day_type}晚間時間，建議開始準備睡覺',
                'very_low': f'{day_type}晚間時間，應該準備休息了'
            }
        }
        
        if time_period in explanations and confidence_level in explanations[time_period]:
            return explanations[time_period][confidence_level]
        else:
            return f'{day_type}{time_period}時間，置信度為{confidence_level}'

    def _get_recommendation(self, confidence_level):
        """獲取使用建議"""
        recommendations = {
            'high': '可以正常使用，這是合理的使用時間',
            'medium': '適度使用，注意時間管理',
            'low': '建議減少使用，考慮其他活動',
            'very_low': '強烈建議停止使用，關注健康作息'
        }
        return recommendations.get(confidence_level, '需要根據具體情況判斷')

    def test_confidence_score_calculation(self, num_tests=5):
        """測試置信度分數計算功能"""
        print("==== Testing Improved Confidence Score Calculation ====")
        
        test_times = [
            (datetime(2024, 1, 15, 9, 0), (0.3, 0.5), '工作日早上'),   # 工作時間，中低
            (datetime(2024, 1, 15, 14, 30), (0.5, 0.7), '工作日下午'), # 午休時間，中高  
            (datetime(2024, 1, 15, 19, 0), (0.7, 0.9), '工作日晚上'),  # 娛樂時間，高
            (datetime(2024, 1, 15, 23, 0), (0.2, 0.4), '工作日深夜'), # 睡覺時間，低
            (datetime(2024, 1, 13, 20, 0), (0.7, 0.9), '週末晚上'),   # 週末娛樂，高
        ]
        
        test_results = []
        
        for i, (test_time, expected_range, desc) in enumerate(test_times[:num_tests]):
            try:
                result = self.calculate_confidence_score(test_time)
                
                day_type = "Weekend" if test_time.weekday() >= 5 else "Weekday"
                print(f"\nTest {i+1}: {test_time.strftime('%Y-%m-%d %H:%M')} ({day_type})")
                print(f"  Confidence Score: {result['confidence_score']:.3f}")
                print(f"  Confidence Level: {result['confidence_level']}")
                print(f"  Time Period: {result['time_period']}")
                print(f"  Explanation: {result['explanation']}")
                print(f"  Recommendation: {result['recommendation']}")
                
                # 檢查是否符合預期
                score = result['confidence_score']
                is_in_range = expected_range[0] <= score <= expected_range[1]
                status = "✓ PASS" if is_in_range else "❌ FAIL"
                print(f"  Expected: {expected_range}, Result: {status}")
                
                test_results.append({
                    'time': test_time,
                    'confidence_score': result['confidence_score'],
                    'confidence_level': result['confidence_level'],
                    'time_period': result['time_period'],
                    'pass': is_in_range
                })
                
            except Exception as e:
                print(f"⚠️ Error in test {i+1}: {e}")
                test_results.append({
                    'time': test_time,
                    'confidence_score': 0.5,
                    'confidence_level': 'medium',
                    'time_period': 'unknown',
                    'pass': False
                })
        
        # 統計測試結果
        passed_tests = sum(1 for result in test_results if result['pass'])
        print(f"\n📊 Test Results: {passed_tests}/{len(test_results)} passed")
        
        return test_results

    def comprehensive_evaluation(self):
        """完整的系統評估"""
        print("\n" + "="*60)
        print("IMPROVED CONFIDENCE SCORE MODULE - COMPREHENSIVE EVALUATION")
        print("="*60)
        
        # 1. 測試不同時間段的分數分布
        print(f"\n1. Time Period Score Distribution:")
        
        test_hours = [2, 7, 10, 14, 19, 22]  # 代表不同時段
        weekday_scores = []
        weekend_scores = []
        
        for hour in test_hours:
            # 工作日
            weekday_time = datetime(2024, 1, 15, hour, 0)  # Monday
            weekday_result = self.calculate_confidence_score(weekday_time)
            weekday_scores.append(weekday_result['confidence_score'])
            
            # 週末
            weekend_time = datetime(2024, 1, 13, hour, 0)  # Saturday  
            weekend_result = self.calculate_confidence_score(weekend_time)
            weekend_scores.append(weekend_result['confidence_score'])
            
            print(f"   {hour:02d}:00 - Weekday: {weekday_result['confidence_score']:.3f} ({weekday_result['confidence_level']}), "
                  f"Weekend: {weekend_result['confidence_score']:.3f} ({weekend_result['confidence_level']})")
        
        # 2. 檢查分數合理性
        print(f"\n2. Score Reasonableness Check:")
        
        # 檢查深夜分數是否較低
        night_scores = [weekday_scores[0], weekday_scores[5]]  # 2:00, 22:00
        day_scores = [weekday_scores[2], weekday_scores[4]]    # 10:00, 19:00
        
        night_avg = np.mean(night_scores)
        day_avg = np.mean(day_scores)
        
        print(f"   Night time average (2:00, 22:00): {night_avg:.3f}")
        print(f"   Day time average (10:00, 19:00): {day_avg:.3f}")
        print(f"   Night < Day: {'✓ PASS' if night_avg < day_avg else '❌ FAIL'}")
        
        # 檢查晚間娛樂時間分數是否較高
        evening_score = weekday_scores[4]  # 19:00
        work_score = weekday_scores[2]     # 10:00
        
        print(f"   Evening (19:00): {evening_score:.3f}")
        print(f"   Work time (10:00): {work_score:.3f}")
        print(f"   Evening > Work: {'✓ PASS' if evening_score > work_score else '❌ FAIL'}")
        
        # 3. 測試置信度計算
        test_results = self.test_confidence_score_calculation()
        
        # 4. 最終評分
        print(f"\n=== FINAL ASSESSMENT ===")
        
        reasonableness_score = 1.0 if night_avg < day_avg and evening_score > work_score else 0.5
        test_pass_rate = sum(1 for r in test_results if r['pass']) / len(test_results)
        score_distribution = 1.0 if 0.2 <= night_avg <= 0.4 and 0.6 <= day_avg <= 0.8 else 0.5
        
        overall_score = (reasonableness_score + test_pass_rate + score_distribution) / 3
        
        print(f"Time Reasonableness: {reasonableness_score:.2f}")
        print(f"Test Pass Rate: {test_pass_rate:.2f}")
        print(f"Score Distribution: {score_distribution:.2f}")
        print(f"Overall System Quality: {overall_score:.2f}")
        
        if overall_score >= 0.8:
            print("🎉 System Quality: Excellent - Logic is human-friendly")
        elif overall_score >= 0.6:
            print("✅ System Quality: Good - Reasonable time-based logic")
        else:
            print("⚠️ System Quality: Needs Improvement")

    def run_complete_analysis(self):
        """運行完整分析"""
        print("="*80)
        print("IMPROVED CONFIDENCE SCORE MODULE - COMPLETE ANALYSIS")
        print("="*80)
        
        print("✅ Time reasonableness profiles loaded")
        print("✅ Confidence matrix initialized")
        
        # 綜合評估
        self.comprehensive_evaluation()
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE - Improved Confidence Score system ready!")
        print("💡 Now considers time reasonableness rather than historical patterns")
        print("="*80)
        
        return {
            'confidence_matrix': self.confidence_matrix,
            'time_reasonableness_profiles': self.time_reasonableness_profiles
        }

# 使用示例
if __name__ == "__main__":
    # 初始化改進版置信度分數模組
    confidence_module = ConfidenceScoreModule()
    
    # 運行完整分析
    result = confidence_module.run_complete_analysis()
    
    # 測試特定時間點
    if result:
        print("\n" + "="*50)
        print("TESTING SPECIFIC TIME POINTS")
        print("="*50)
        
        test_times = [
            datetime(2024, 7, 16, 23, 1),   # 您的原始測試時間
            datetime(2024, 7, 16, 19, 0),   # 晚間娛樂時間
            datetime(2024, 7, 16, 10, 0),   # 工作時間
            datetime(2024, 7, 16, 14, 30),  # 午休時間
        ]
        
        for test_time in test_times:
            result = confidence_module.calculate_confidence_score(test_time)
            
            print(f"\n🕐 時間: {test_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"📊 置信度分數: {result['confidence_score']:.3f}")
            print(f"🎯 置信度等級: {result['confidence_level']}")
            print(f"⏰ 時間段: {result['time_period']}")
            print(f"💡 解釋: {result['explanation']}")
            print(f"📝 建議: {result['recommendation']}")