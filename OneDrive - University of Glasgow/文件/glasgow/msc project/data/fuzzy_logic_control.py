# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime, timedelta
# import warnings
# from collections import deque
# warnings.filterwarnings('ignore')

# ### 找三個model ###

# try:
#     from device_activity import DeviceActivityScoreModule
#     HAS_DEVICE_ACTIVITY = True
# except ImportError:
#     HAS_DEVICE_ACTIVITY = False
#     print("⚠️  device_activity 模組未找到")

# # try:
# #     from user_habit import ImprovedUserHabitScoreModule
# #     HAS_USER_HABIT = True
# # except ImportError:
# #     HAS_USER_HABIT = False
# #     print("⚠️  user_habit 模組未找到")


# try:
#     from user_habit_nooff import NoShutdownUserHabitScoreModule
#     HAS_USER_HABIT = True
# except ImportError:
#     HAS_USER_HABIT = False
#     print("⚠️  user_habit 模組未找到")

# try:
#     from confidence_score import ConfidenceScoreModule
#     HAS_CONFIDENCE_SCORE = True
# except ImportError:
#     HAS_CONFIDENCE_SCORE = False
#     print("⚠️  confidence_score 模組未找到")

# try:
#     from decision_evaluator import DecisionEvaluator
#     HAS_EVALUATOR = True
#     print("✅ DecisionEvaluator模組載入成功")
# except ImportError:
#     HAS_EVALUATOR = False
#     print("⚠️ 警告：DecisionEvaluator模組未找到，評估功能將被禁用")

# _decision_evaluator = None
# _oscillation_detector = None

# def init_decision_evaluator():
#     """初始化決策評估器"""
#     global _decision_evaluator
    
#     if not HAS_EVALUATOR:
#         return None
    
#     if _decision_evaluator is None:
#         _decision_evaluator = DecisionEvaluator(
#             window_size_minutes=45,
#             evaluation_interval_minutes=30
#         )
#         print("✅ DecisionEvaluator初始化完成")
    
#     return _decision_evaluator

# def estimate_predicted_power(actual_power, fuzzy_output):
#     """估算預測功率值（用於評估器）"""
#     if fuzzy_output > 0.7:
#         predicted_power = actual_power * 0.9
#     elif fuzzy_output < 0.3:
#         predicted_power = actual_power * 1.1
#     else:
#         predicted_power = actual_power
    
#     # 添加一些隨機噪聲來模擬預測不確定性
#     noise = np.random.normal(0, actual_power * 0.05)
#     predicted_power += noise
#     return max(0, predicted_power)

# def calculate_fuzzy_output(activity_score, habit_score, confidence_score, power_value):
#     """計算fuzzy控制器輸出（0-1）"""
#     # 基於三個分數計算fuzzy輸出
#     activity_weight = 0.4
#     habit_weight = 0.4
#     confidence_weight = 0.2
    
#     # 將活動和習慣分數反轉（分數越低越可能關閉）
#     fuzzy_output = (
#         activity_weight * (1 - activity_score) +
#         habit_weight * (1 - habit_score) +
#         confidence_weight * confidence_score
#     )
    
#     # 根據功率值調整
#     if power_value < 36:  # phantom load
#         fuzzy_output = min(1.0, fuzzy_output + 0.2)  # 增加關閉傾向
    
#     return np.clip(fuzzy_output, 0, 1)

# class AntiOscillationFilter:
#     def __init__(self, 
#                  hysteresis_enabled=True,
#                  phantom_threshold_low=17,
#                  phantom_threshold_high=21,
#                  decision_cooldown_seconds=30,
#                  min_state_duration_minutes=1,
#                  stability_check_enabled=False,
                 
#                  # 🆕 新增參數 - 針對休眠狀態檢測
#                  sleep_mode_detection_enabled=True,
#                  sleep_mode_threshold=25,
#                  sleep_mode_force_shutdown_minutes=8):
        
#         self.hysteresis_enabled = hysteresis_enabled
#         self.phantom_low = phantom_threshold_low
#         self.phantom_high = phantom_threshold_high
#         self.decision_cooldown = timedelta(seconds=decision_cooldown_seconds)
#         self.min_state_duration = timedelta(minutes=min_state_duration_minutes)
#         self.stability_check_enabled = stability_check_enabled
        
#         # 🆕 休眠模式檢測參數
#         self.sleep_mode_detection_enabled = sleep_mode_detection_enabled
#         self.sleep_mode_threshold = sleep_mode_threshold
#         self.sleep_mode_force_minutes = sleep_mode_force_shutdown_minutes
        
#         self.last_decision_time = None
#         self.last_decision = None
#         self.current_power_state = 'unknown'
#         self.state_start_time = None
#         self.recent_powers = deque(maxlen=10)
        
#         # 🆕 新增狀態追蹤
#         self.power_history = deque(maxlen=50)
#         self.timestamp_history = deque(maxlen=50)
#         self.sleep_mode_start_time = None
        
#         print(f"✅ 增強型防震盪濾波器初始化")
#         print(f"   - 遲滯閾值: {phantom_threshold_low}W ~ {phantom_threshold_high}W")
#         print(f"   - 休眠檢測: {'啟用' if sleep_mode_detection_enabled else '禁用'}")
#         if sleep_mode_detection_enabled:
#             print(f"     * 休眠閾值: <{sleep_mode_threshold}W")
#             print(f"     * 強制關機時間: {sleep_mode_force_shutdown_minutes}分鐘")
    
#     def filter_decision(self, original_decision, power_value, timestamp, scores=None):
#         # 更新歷史記錄
#         self.recent_powers.append(power_value)
#         self.power_history.append(power_value)
#         self.timestamp_history.append(timestamp)
        
#         # 🆕 休眠模式檢測
#         sleep_mode_result = self._detect_sleep_mode(timestamp, power_value)
        
#         # 🆕 如果檢測到需要強制關機的休眠狀態
#         if sleep_mode_result['force_shutdown']:
#             if self._is_likely_sleep_time(timestamp):
#                 suggested_decision = 'suggest_shutdown'
#             elif self._is_work_hours(timestamp):
#                 suggested_decision = 'send_notification'  # 工作時間比較保守
#             else:
#                 suggested_decision = 'suggest_shutdown'
            
#             return {
#                 'filtered_decision': suggested_decision,
#                 'original_decision': original_decision,
#                 'filter_reason': f'時間感知休眠檢測({sleep_mode_result["duration_minutes"]:.1f}分鐘)',
#                 'power_state': 'sleep_mode',
#                 'should_use_filtered': True,
#                 'sleep_mode_detected': True
#             }
        
#         # 🆕 如果是低功率且原決策是keep_on，需要修正
#         if (power_value < self.sleep_mode_threshold and 
#             original_decision == 'keep_on' and
#             sleep_mode_result['is_sleep_mode'] and
#             sleep_mode_result['duration_minutes'] > 10):  # 需要持續10分鐘以上
            
#             # 先改為通知，而不是直接關機
#             if power_value < 16:  # 只有極低功率才直接建議關機
#                 filtered_decision = 'suggest_shutdown'
#             else:
#                 filtered_decision = 'send_notification'  # 其他情況發通知
            
#             return {
#                 'filtered_decision': filtered_decision,
#                 'original_decision': original_decision,
#                 'filter_reason': f'長時間低功率修正(功率{power_value:.1f}W, {sleep_mode_result["duration_minutes"]:.1f}分鐘)',
#                 'power_state': 'sleep_mode_correction',
#                 'should_use_filtered': True,
#                 'sleep_mode_detected': True
#             }
        
#         # 檢查冷卻期
#         if self._in_cooldown_period(timestamp):
#             return {
#                 'filtered_decision': 'delay_decision',
#                 'original_decision': original_decision,
#                 'filter_reason': '決策冷卻期內',
#                 'power_state': self.current_power_state,
#                 'should_use_filtered': True,
#                 'sleep_mode_detected': sleep_mode_result['is_sleep_mode']
#             }
        
#         # 更新功率狀態
#         self._update_power_state(power_value, timestamp)
        
#         # 檢查持續時間
#         if not self._meets_minimum_duration(timestamp):
#             return {
#                 'filtered_decision': 'delay_decision',
#                 'original_decision': original_decision,
#                 'filter_reason': '狀態持續時間不足',
#                 'power_state': self.current_power_state,
#                 'should_use_filtered': True,
#                 'sleep_mode_detected': sleep_mode_result['is_sleep_mode']
#             }
        
#         # 檢查穩定性
#         if self.stability_check_enabled and not self._is_power_stable():
#             # 🆕 如果在休眠模式中震盪，直接建議關機
#             if sleep_mode_result['is_sleep_mode']:
#                 return {
#                     'filtered_decision': 'suggest_shutdown',
#                     'original_decision': original_decision,
#                     'filter_reason': '休眠模式中的功率震盪',
#                     'power_state': 'sleep_mode_unstable',
#                     'should_use_filtered': True,
#                     'sleep_mode_detected': True
#                 }
#             else:
#                 return {
#                     'filtered_decision': 'delay_decision',
#                     'original_decision': original_decision,
#                     'filter_reason': '功率不穩定',
#                     'power_state': self.current_power_state,
#                     'should_use_filtered': True,
#                     'sleep_mode_detected': False
#                 }
        
#         # 根據功率狀態調整決策
#         filtered_decision = self._adjust_decision_by_power_state(original_decision, sleep_mode_result)

#         valid_decisions = ['suggest_shutdown', 'send_notification', 'delay_decision', 'keep_on']
#         if filtered_decision not in valid_decisions:
#             print(f"⚠️ 警告：濾波器返回了無效決策 '{filtered_decision}', 改為 'delay_decision'")
#             filtered_decision = 'delay_decision'
        
#         # 更新決策歷史
#         if filtered_decision != 'delay_decision':
#             self.last_decision = filtered_decision
#             self.last_decision_time = timestamp
        
#         return {
#             'filtered_decision': filtered_decision,
#             'original_decision': original_decision,
#             'filter_reason': '濾波完成',
#             'power_state': self.current_power_state,
#             'should_use_filtered': filtered_decision != original_decision,
#             'sleep_mode_detected': sleep_mode_result['is_sleep_mode']
#         }
        
    
#     def _detect_sleep_mode(self, current_time, current_power):
#         """🆕 檢測休眠模式"""
#         if not self.sleep_mode_detection_enabled:
#             return {
#                 'is_sleep_mode': False,
#                 'duration_minutes': 0,
#                 'force_shutdown': False
#             }
        
#         # 檢查當前功率是否為休眠狀態
#         is_current_sleep = current_power < self.sleep_mode_threshold
        
#         # 更新休眠開始時間
#         if is_current_sleep:
#             if self.sleep_mode_start_time is None:
#                 self.sleep_mode_start_time = current_time
#         else:
#             self.sleep_mode_start_time = None
        
#         # 計算休眠持續時間
#         duration_minutes = 0
#         if self.sleep_mode_start_time:
#             duration = current_time - self.sleep_mode_start_time
#             duration_minutes = duration.total_seconds() / 60
        
#         # 判斷是否需要強制關機
#         force_shutdown = (duration_minutes >= self.sleep_mode_force_minutes and current_power < 18)
        
#         return {
#             'is_sleep_mode': is_current_sleep,
#             'duration_minutes': duration_minutes,
#             'force_shutdown': force_shutdown
#         }
    
#     def _adjust_decision_by_power_state(self, original_decision, sleep_mode_result):
#         """根據功率狀態調整決策 - 加入休眠模式考慮"""
        
#         # 🆕 如果檢測到休眠模式，優先處理
#         # 🔧 更漸進的休眠模式處理
#         if sleep_mode_result['is_sleep_mode']:
#             duration = sleep_mode_result['duration_minutes']
            
#             if duration > 12:  # 超過12分鐘才考慮修正
#                 if original_decision == 'keep_on':
#                     # 根據功率值決定修正強度
#                     if self.recent_powers and np.mean(list(self.recent_powers)[-3:]) < 16:
#                         return 'suggest_shutdown'  # 極低功率才直接關機
#                     else:
#                         return 'send_notification'  # 其他情況發通知
#                 elif original_decision == 'delay_decision':
#                     return 'send_notification'
#             elif duration > 6:  # 6-12分鐘之間，輕微修正
#                 if original_decision == 'keep_on' and self.recent_powers:
#                     recent_avg = np.mean(list(self.recent_powers)[-3:])
#                     if recent_avg < 16:  # 只修正極低功率的情況
#                         return 'send_notification'
        
#         # 原有邏輯
#         if self.current_power_state == 'uncertain':
#             if original_decision in ['suggest_shutdown', 'send_notification']:
#                 return 'delay_decision'
        
#         elif self.current_power_state == 'phantom':
#             if len(self.recent_powers) >= 3:
#                 recent_avg = np.mean(list(self.recent_powers)[-3:])
#                 # 🆕 使用休眠閾值進行更積極的判斷
#                 if recent_avg < self.sleep_mode_threshold:
#                     if original_decision == 'keep_on':
#                         return 'suggest_shutdown'  # 直接建議關機
#                 elif 18 <= recent_avg <= 22:
#                     if original_decision == 'keep_on':
#                         return 'send_notification'  # 改為通知
        
#         elif self.current_power_state == 'active':
#             if original_decision == 'suggest_shutdown':
#                 return 'send_notification'
        
#         return original_decision
    
#     def get_filter_status(self):
#         """獲取濾波器狀態 - 包含休眠檢測狀態"""
#         sleep_info = self._detect_sleep_mode(datetime.now(), 
#                                            self.recent_powers[-1] if self.recent_powers else 0)
        
#         return {
#             'current_power_state': self.current_power_state,
#             'state_duration_minutes': (
#                 (datetime.now() - self.state_start_time).total_seconds() / 60 
#                 if self.state_start_time else 0
#             ),
#             'last_decision': self.last_decision,
#             'recent_powers': list(self.recent_powers),
#             'is_in_cooldown': self._in_cooldown_period(datetime.now()) if self.last_decision_time else False,
#             'sleep_mode_detected': sleep_info['is_sleep_mode'],
#             'sleep_duration_minutes': sleep_info['duration_minutes']
#         }
    
#     # 保留原有方法
#     def _in_cooldown_period(self, timestamp):
#         if self.last_decision_time is None:
#             return False
#         return timestamp - self.last_decision_time < self.decision_cooldown
    
#     def _update_power_state(self, power_value, timestamp):
#         if not self.hysteresis_enabled:
#             new_state = 'phantom' if power_value < 19 else 'active'
#         else:
#             if power_value <= self.phantom_low:
#                 new_state = 'phantom'
#             elif power_value >= self.phantom_high:
#                 new_state = 'active'
#             else:
#                 new_state = self.current_power_state if self.current_power_state in ['phantom', 'active'] else 'uncertain'
        
#         if new_state != self.current_power_state:
#             self.current_power_state = new_state
#             self.state_start_time = timestamp
    
#     def _meets_minimum_duration(self, timestamp):
#         if self.state_start_time is None:
#             return False
#         duration = timestamp - self.state_start_time
#         return duration >= self.min_state_duration
    
#     def _is_power_stable(self):
#         if len(self.recent_powers) < 3:
#             return True
        
#         recent_list = list(self.recent_powers)[-5:]
#         if len(recent_list) < 3:
#             return True
        
#         std_dev = np.std(recent_list)
#         mean_power = np.mean(recent_list)
#         coefficient_of_variation = std_dev / mean_power if mean_power > 0 else 0
#         return coefficient_of_variation < 0.1 or std_dev < 2.0
    
#     def _is_likely_sleep_time(self, timestamp):
#         """判斷是否為可能的睡眠時間"""
#         hour = timestamp.hour
#         # 深夜到早晨 (23:00-07:00) 更容易接受關機建議
#         return hour >= 23 or hour <= 7

#     def _is_work_hours(self, timestamp):
#         """判斷是否為工作時間"""
#         hour = timestamp.hour
#         weekday = timestamp.weekday()
#         # 工作日的工作時間
#         return weekday < 5 and 9 <= hour <= 17


# class DecisionTreeSmartPowerAnalysis:
#     def __init__(self):
#         self.data_file = 'C:/Users/王俞文/OneDrive - University of Glasgow/文件/glasgow/msc project/data/complete_power_data_with_history.csv'
        
#         print("start decision tree smart power analysis...")
        
#         # 電費設定
#         self.uk_electricity_rate = 0.30  # £0.30/kWh
        
#         # 初始化並訓練模型
#         self.device_activity_model = None
#         self.user_habit_model = None
#         self.confidence_model = None

#         init_decision_evaluator()

#         self.anti_oscillation_filter = AntiOscillationFilter(
#             hysteresis_enabled=True,
#             phantom_threshold_low=17,
#             phantom_threshold_high=21,
#             decision_cooldown_seconds=30,
#             min_state_duration_minutes=2,
#             stability_check_enabled=True,

#             sleep_mode_detection_enabled=True,
#             sleep_mode_threshold=20,              # 休眠閾值
#             sleep_mode_force_shutdown_minutes=15
#         )
        
#         # 決策統計
#         self.decision_stats = {
#             'total_decisions': 0,
#             'decision_paths': {},  # 記錄每種決策路徑
#             'level_combinations': {},  # 記錄每種等級組合
#             'filtered_decisions': 0,        # 🆕 添加這行
#             'oscillation_prevented': 0, 
#             'sleep_mode_corrections': 0,      # 🆕 添加這行
#             'sleep_mode_detections': 0  
#         }
        
#         # 訓練設備活動模型
#         if HAS_DEVICE_ACTIVITY:
#             try:
#                 print("\n🔄 正在初始化並訓練設備活動模型...")
#                 self.device_activity_model = DeviceActivityScoreModule()
#                 self.device_activity_model.run_complete_analysis(self.data_file)
#                 print("✅ 設備活動模型訓練完成")
#             except Exception as e:
#                 print(f"❌ 設備活動模型訓練失敗: {e}")
#                 self.device_activity_model = None
        
#         # 訓練用戶習慣模型
#         if HAS_USER_HABIT:
#             try:
#                 print("\n🔄 正在初始化並訓練用戶習慣模型...")
#                 self.user_habit_model = NoShutdownUserHabitScoreModule()
#                 self.user_habit_model.run_complete_analysis(self.data_file)
#                 print("✅ 用戶習慣模型訓練完成")
#             except Exception as e:
#                 print(f"❌ 用戶習慣模型訓練失敗: {e}")
#                 self.user_habit_model = None
        
#         # 訓練置信度模型
#         if HAS_CONFIDENCE_SCORE:
#             try:
#                 print("\n🔄 正在初始化並訓練置信度模型...")
#                 self.confidence_model = ConfidenceScoreModule()
#                 self.confidence_model.run_complete_analysis()
#                 print("✅ 置信度模型訓練完成")
#             except Exception as e:
#                 print(f"❌ 置信度模型訓練失敗: {e}")
#                 self.confidence_model = None
        
#         print("\n🎉 決策樹版智能電源管理系統初始化完成！")
        
#         self.results = {
#             'phantom_load_detected': 0,
#             'suggest_shutdown': 0,
#             'keep_on': 0,
#             'send_notification': 0,
#             'delay_decision': 0,
#             'total_opportunities': 0,
#             'error': 0
#         }

        

#     def _generate_phantom_load_opportunities(self, df):

#         df['timestamp'] = pd.to_datetime(df['timestamp'])
#         df = df.sort_values('timestamp')

#         # df['is_phantom'] = df['power'] < 92
#         # df['is_phantom'] = df['power'] < 60
#         # df['is_phantom'] = df['power'] < 36
#         df['is_phantom'] = df['power'] < 36
#         print(f'phantom load (< 60W) : {len(df[df["is_phantom"]])} counts')

#         opportunities = []
#         in_session = False   # 判斷是否在phantom load
#         start_time = None
#         records = []

#         for i, row in df.iterrows():
#             if row['is_phantom']:
#                 if not in_session:
#                     in_session = True
#                     start_time = row['timestamp']
#                     records = []
#                 records.append(row)
#             else:
#                 if in_session:
#                     end_time = row['timestamp']
#                     power_list = [r['power'] for r in records if r['power'] > 0]
#                     avg_power = np.mean(power_list) if power_list else 75
#                     opportunities.append({
#                         'device_id': 'phantom_device',
#                         'start_time': start_time,
#                         'end_time': end_time,
#                         'power_watt': avg_power
#                     })
#                     in_session = False

#         if in_session:
#             end_time = df['timestamp'].iloc[-1]
#             power_list = [r['power'] for r in records if r['power'] > 0]
#             avg_power = np.mean(power_list) if power_list else 75
#             opportunities.append({
#                 'device_id': 'phantom_device',
#                 'start_time': start_time,
#                 'end_time': end_time,
#                 'power_watt': avg_power
#             })

#         return opportunities

#     def _make_intelligent_decision(self, activity_score, habit_score, confidence_score, features):

#         def to_level(score):
#             """將連續分數轉換為離散等級"""
#             if score < 0.33:
#                 return "low"
#             elif score < 0.66:
#                 return "medium"
#             else:
#                 return "high"
        
#         # 轉換分數為等級
#         user_habit = to_level(habit_score)
#         device_activity = to_level(activity_score)
#         confidence_score = to_level(confidence_score)
        
#         # 記錄等級組合統計
#         combination = f"{user_habit}-{device_activity}-{confidence_score}"
#         if combination not in self.decision_stats['level_combinations']:
#             self.decision_stats['level_combinations'][combination] = 0
#         self.decision_stats['level_combinations'][combination] += 1
        
#         # 合理的智能決策樹邏輯 - 基於實際使用場景
#         decision_path = []

        
#         decision = "delay_decision"  # 默認值
        
#         if user_habit == "low":  # 很少使用設備
#             decision_path.append("user habit=low")
            
#             if device_activity == "low":  # 長時間待機
#                 decision_path.append("device activity=low")
#                 if confidence_score == "low":
#                     decision_path.append("confidence score=low")
#                     decision = "suggest_shutdown"  # 很少用+長時間待機+不確定時段 -> 關機
#                 elif confidence_score == "medium":
#                     decision_path.append("confidence score=medium")
#                     decision = "suggest_shutdown"  # 很少用+長時間待機+中等確定 -> 關機
#                 elif confidence_score == "high":
#                     decision_path.append("confidence score=high")
#                     decision = "delay_decision"  # 很少用+長時間待機+確定時段，可能特殊情況 -> 等待
                    
#             elif device_activity == "medium":  # 中等活躍度
#                 decision_path.append("device activity=medium")
#                 if confidence_score == "low":
#                     decision_path.append("confidence score=low")
#                     decision = "delay_decision"  # 很少用但有些活躍+不確定 -> 等待
#                 elif confidence_score == "medium":
#                     decision_path.append("confidence score=medium")
#                     decision = "send_notification"  # 很少用但有些活躍+中等確定 -> 通知
#                 elif confidence_score == "high":
#                     decision_path.append("confidence score=high")
#                     decision = "send_notification"  # 很少用但有些活躍+確定時段 -> 通知
                    
#             elif device_activity == "high":  # 最近很活躍
#                 decision_path.append("device activity=high")
#                 if confidence_score == "low":
#                     decision_path.append("confidence score=low")
#                     decision = "keep_on"  # 很少用但剛剛活躍+不確定 -> 保持
#                 elif confidence_score == "medium":
#                     decision_path.append("confidence score=medium")
#                     decision = "keep_on"  # 很少用但剛剛活躍+中等確定 -> 保持
#                 elif confidence_score == "high":
#                     decision_path.append("confidence score=high")
#                     decision = "keep_on"  # 很少用但剛剛活躍+確定 -> 保持
                    
#         elif user_habit == "medium":  # 中等使用頻率
#             decision_path.append("user habit=medium")
            
#             if device_activity == "low":  # 長時間待機
#                 decision_path.append("device activity=low")
#                 if confidence_score == "low":
#                     decision_path.append("confidence score=low")
#                     decision = "suggest_shutdown"  # 中等使用+長時間待機+不確定 -> 關機
#                 elif confidence_score == "medium":
#                     decision_path.append("confidence score=medium")
#                     decision = "suggest_shutdown"  # 中等使用+長時間待機+中等確定 -> 關機
#                 elif confidence_score == "high":
#                     decision_path.append("confidence score=high")
#                     decision = "send_notification"  # 中等使用+長時間待機+確定時段 -> 通知
                    
#             elif device_activity == "medium":  # 中等活躍度
#                 decision_path.append("device activity=medium")
#                 if confidence_score == "low":
#                     decision_path.append("confidence score=low")
#                     decision = "delay_decision"  # 中等使用+中等活躍+不確定 -> 等待
#                 elif confidence_score == "medium":
#                     decision_path.append("confidence score=medium")
#                     decision = "send_notification"  # 中等使用+中等活躍+中等確定 -> 通知
#                 elif confidence_score == "high":
#                     decision_path.append("confidence score=high")
#                     decision = "keep_on"  # 中等使用+中等活躍+確定peak hour -> 保持
                    
#             elif device_activity == "high":  # 最近很活躍
#                 decision_path.append("device activity=high")
#                 if confidence_score == "low":
#                     decision_path.append("confidence score=low")
#                     decision = "delay_decision"  # 中等使用+剛剛活躍+不確定 -> 等待
#                 elif confidence_score == "medium":
#                     decision_path.append("confidence score=medium")
#                     decision = "keep_on"  # 中等使用+剛剛活躍+中等確定 -> 保持
#                 elif confidence_score == "high":
#                     decision_path.append("confidence score=high")
#                     decision = "keep_on"  # 中等使用+剛剛活躍+確定 -> 保持
                    
#         elif user_habit == "high":  # 經常使用設備
#             decision_path.append("user habit=high")
            
#             if device_activity == "low":  # 長時間待機
#                 decision_path.append("device activity=low")
#                 if confidence_score == "low":
#                     decision_path.append("confidence score=low")
#                     decision = "suggest_shutdown"  # 經常使用但長時間待機+不確定 -> 可能睡覺，關機
#                 elif confidence_score == "medium":
#                     decision_path.append("confidence score=medium")
#                     decision = "delay_decision"  # 經常使用但長時間待機+中等確定 -> 等待
#                 elif confidence_score == "high":
#                     decision_path.append("confidence score=high")
#                     decision = "delay_decision"  # 經常使用但長時間待機+確定睡眠 -> 等待再決定
                    
#             elif device_activity == "medium":  # 中等活躍度
#                 decision_path.append("device activity=medium")
#                 if confidence_score == "low":
#                     decision_path.append("confidence score=low")
#                     decision = "suggest_shutdown"  # 經常使用+中等活躍+不確定 -> 異常情況，關機
#                 elif confidence_score == "medium":
#                     decision_path.append("confidence score=medium")
#                     decision = "send_notification"  # 經常使用+中等活躍+中等確定 -> 通知
#                 elif confidence_score == "high":
#                     decision_path.append("confidence score=high")
#                     decision = "keep_on"  # 經常使用+中等活躍+確定peak hour -> 保持
                    
#             elif device_activity == "high":  # 最近很活躍
#                 decision_path.append("device activity=high")
#                 if confidence_score == "low":
#                     decision_path.append("confidence score=low")
#                     decision = "delay_decision"  # 經常使用+剛剛活躍+不確定 -> 等待
#                 elif confidence_score == "medium":
#                     decision_path.append("confidence score=medium")
#                     decision = "send_notification"  # 經常使用+剛剛活躍+中等確定 -> 通知確認
#                 elif confidence_score == "high":
#                     decision_path.append("confidence score=high")
#                     decision = "keep_on"  # 經常使用+剛剛活躍+確定 -> 保持
        
#         # 記錄決策路徑統計
#         path_key = " -> ".join(decision_path) + f" => {decision}"
#         if path_key not in self.decision_stats['decision_paths']:
#             self.decision_stats['decision_paths'][path_key] = 0
#         self.decision_stats['decision_paths'][path_key] += 1
        
#         self.decision_stats['total_decisions'] += 1
        
#         # 添加調試信息
#         # print(f"   決策詳情: {user_habit}-{device_activity}-{confidence_score} => {decision}")
        
#         # 創建詳細的debug信息
#         debug_info = {
#             'user_habit_level': user_habit,
#             'device_activity_level': device_activity,
#             'confidence_score_level': confidence_score,
#             'decision_path': decision_path,
#             'scores': {
#                 'activity_score': activity_score,
#                 'habit_score': habit_score,
#                 'confidence_score': confidence_score
#             },
#             'features': features
#         }
        
#         return decision, debug_info

#     def _fallback_activity_score(self, features, timestamp):
#         print("fallback activity score")
#         """改進的fallback活動分數 - 確保多樣化分布"""
#         hour = timestamp.hour
#         weekday = timestamp.weekday()
        
#         # 更明確的分數範圍，確保三個等級都會出現
#         if weekday < 5:  # 工作日
#             if 9 <= hour <= 17:  # 工作時間 - 偏向 medium/high
#                 base_score = np.random.choice([0.2, 0.5, 0.8], p=[0.2, 0.4, 0.4])
#             elif 18 <= hour <= 22:  # 晚間 - 偏向 high
#                 base_score = np.random.choice([0.1, 0.4, 0.8], p=[0.1, 0.3, 0.6])
#             else:  # 深夜早晨 - 偏向 low
#                 base_score = np.random.choice([0.1, 0.4, 0.7], p=[0.6, 0.3, 0.1])
#         else:  # 週末
#             if 8 <= hour <= 22:  # 白天 - 平均分布
#                 base_score = np.random.choice([0.2, 0.5, 0.8], p=[0.3, 0.4, 0.3])
#             else:  # 夜間 - 偏向 low
#                 base_score = np.random.choice([0.1, 0.4, 0.7], p=[0.7, 0.2, 0.1])
        
#         # 添加小幅隨機變動
#         variation = np.random.normal(0, 0.1)
#         final_score = max(0.05, min(0.95, base_score + variation))
        
#         # print(f'fallback activity: {final_score:.2f}')
#         return final_score

#     def _fallback_habit_score(self, features, timestamp):
#         """改進的fallback習慣分數 - 確保多樣化分布"""
#         print("fallback habit score")
#         hour = timestamp.hour
#         weekday = timestamp.weekday()
        
#         # 更明確的分數範圍
#         if weekday < 5:  # 工作日
#             if 7 <= hour <= 9 or 18 <= hour <= 23:  # 高使用時段 - 偏向 high
#                 base_score = np.random.choice([0.1, 0.4, 0.8], p=[0.1, 0.3, 0.6])
#             elif 10 <= hour <= 17:  # 工作時間 - 偏向 medium
#                 base_score = np.random.choice([0.2, 0.5, 0.8], p=[0.3, 0.5, 0.2])
#             else:  # 其他時間 - 偏向 low
#                 base_score = np.random.choice([0.1, 0.4, 0.7], p=[0.6, 0.3, 0.1])
#         else:  # 週末
#             if 9 <= hour <= 23:  # 週末活躍時間 - 平均分布
#                 base_score = np.random.choice([0.2, 0.5, 0.8], p=[0.3, 0.4, 0.3])
#             else:  # 週末休息時間 - 偏向 low
#                 base_score = np.random.choice([0.1, 0.4, 0.7], p=[0.7, 0.2, 0.1])
        
#         # 添加小幅隨機變動
#         variation = np.random.normal(0, 0.1)
#         final_score = max(0.05, min(0.95, base_score + variation))
        
#         # print(f'fallback habit: {final_score:.2f}')
#         return final_score

#     def _fallback_confidence_score(self, features, timestamp):
#         """改進的fallback置信度分數 - 確保多樣化分布"""
#         print("fallback confidence score")
#         hour = timestamp.hour
        
#         # 更明確的分數範圍
#         if 18 <= hour <= 23:  # 晚間高使用期 - 偏向 high
#             base_score = np.random.choice([0.1, 0.4, 0.8], p=[0.1, 0.3, 0.6])
#         elif 14 <= hour <= 16:  # 下午可能是低使用期 - 偏向 medium
#             base_score = np.random.choice([0.2, 0.5, 0.8], p=[0.3, 0.5, 0.2])
#         elif 2 <= hour <= 6:  # 深夜低使用期 - 偏向 low
#             base_score = np.random.choice([0.1, 0.4, 0.7], p=[0.6, 0.3, 0.1])
#         else:  # 其他時間 - 平均分布
#             base_score = np.random.choice([0.2, 0.5, 0.8], p=[0.3, 0.4, 0.3])
        
#         # 添加小幅隨機變動
#         variation = np.random.normal(0, 0.08)
#         final_score = max(0.1, min(0.9, base_score + variation))
        
#         # print(f'fallback confidence: {final_score:.2f}')
#         return final_score

#     def _extract_enhanced_features(self, opportunity, df):
#         return {
#             'device_id': opportunity.get('device_id', 'unknown'),
#             'duration_minutes': (opportunity['end_time'] - opportunity['start_time']).total_seconds() / 60,
#             'hour_of_day': opportunity['start_time'].hour,
#             'power_watt': opportunity.get('power_watt', 75),
#             'weekday': opportunity['start_time'].weekday()
#         }

#     def _apply_decision_tree_models(self, opportunities, df):
#         print("\n🌳 使用決策樹方法進行決策分析...")
#         decision_results = []
#         debug_logs = []

#         for i, opp in enumerate(opportunities):
#             try:
#                 features = self._extract_enhanced_features(opp, df)
#                 timestamp = opp['start_time']

#                 # 使用訓練好的模型或fallback
#                 if self.device_activity_model:
#                     try:
#                         activity_result = self.device_activity_model.calculate_activity_score(timestamp)
#                         activity_score = activity_result['activity_score']
#                     except Exception as e:
#                         activity_score = self._fallback_activity_score(features, timestamp)
#                 else:
#                     activity_score = self._fallback_activity_score(features, timestamp)

#                 if self.user_habit_model:
#                     try:
#                         habit_result = self.user_habit_model.calculate_habit_score(timestamp)
#                         habit_score = habit_result['habit_score']
#                     except Exception as e:
#                         habit_score = self._fallback_habit_score(features, timestamp)
#                 else:
#                     habit_score = self._fallback_habit_score(features, timestamp)

#                 if self.confidence_model:
#                     try:
#                         confidence_result = self.confidence_model.calculate_confidence_score(timestamp)
#                         confidence_score = confidence_result['confidence_score']
#                     except Exception as e:
#                         confidence_score = self._fallback_confidence_score(features, timestamp)
#                 else:
#                     confidence_score = self._fallback_confidence_score(features, timestamp)

#                 # 🌳 使用決策樹方法
#                 decision, debug_info = self._make_intelligent_decision(
#                     activity_score, habit_score, confidence_score, features
#                 )

#                 filter_result = self.anti_oscillation_filter.filter_decision(
#                 original_decision=decision,
#                 power_value=features['power_watt'],
#                 timestamp=timestamp,
#                 scores={
#                     'activity': activity_score,
#                     'habit': habit_score,
#                     'confidence': confidence_score
#                 }
#                 )

#                 # 使用濾波後的決策
#                 final_decision = filter_result['filtered_decision']

#                 # 統計濾波效果
#                 if filter_result['should_use_filtered']:
#                     self.decision_stats['filtered_decisions'] += 1
#                     if filter_result['filter_reason'] in ['決策冷卻期內', '功率不穩定']:
#                         self.decision_stats['oscillation_prevented'] += 1
                    
#                     # 🆕 統計休眠模式相關修正
#                     if 'sleep_mode_detected' in filter_result and filter_result['sleep_mode_detected']:
#                         self.decision_stats['sleep_mode_detections'] += 1
                    
#                     if '休眠' in filter_result['filter_reason']:
#                         self.decision_stats['sleep_mode_corrections'] += 1
#                         print(f"🛌 休眠模式修正: {timestamp.strftime('%H:%M')} - {filter_result['filter_reason']}")

#                 # 添加濾波信息到debug_info
#                 debug_info['filter_applied'] = filter_result['should_use_filtered']
#                 debug_info['filter_reason'] = filter_result['filter_reason']
#                 debug_info['power_state'] = filter_result['power_state']
#                 debug_info['original_decision'] = decision


#                 if HAS_EVALUATOR and _decision_evaluator is not None:
#                     try:
#                         # 計算fuzzy輸出
#                         fuzzy_output = calculate_fuzzy_output(
#                             activity_score, habit_score, confidence_score, 
#                             features['power_watt']
#                         )
                        
#                         # 估算預測功率
#                         predicted_power = estimate_predicted_power(
#                             features['power_watt'], 
#                             fuzzy_output
#                         )
                        
#                         # 添加決策記錄到評估器
#                         _decision_evaluator.add_decision_record(
#                             timestamp=timestamp,
#                             fuzzy_output=fuzzy_output,
#                             predicted_power=predicted_power,
#                             actual_power=features['power_watt'],
#                             decision=final_decision,
#                             confidence_scores={
#                                 'activity': activity_score,
#                                 'habit': habit_score,
#                                 'confidence': confidence_score
#                             }
#                         )
#                     except Exception as e:
#                         print(f"評估器記錄錯誤 (opportunity {i+1}): {e}")


#                 if final_decision in self.results:
#                     self.results[final_decision] += 1
#                 else:
#                     print(f"   ⚠️ Unknown decision result: {final_decision}")
#                     # 🔧 確保results字典有所有可能的決策類型
#                     if final_decision not in self.results:
#                         self.results[final_decision] = 0
#                     self.results[final_decision] += 1

#                 result = {
#                     'opportunity': opp,
#                     'features': features,
#                     'activity_score': activity_score,
#                     'user_habit_score': habit_score,
#                     'confidence_score': confidence_score,
#                     'decision': final_decision,
#                     'debug_info': debug_info
#                 }
#                 decision_results.append(result)
                
#                 # 記錄前10個的詳細debug資訊
#                 if i < 10:
#                     debug_logs.append({
#                         'index': i+1,
#                         'time': timestamp,
#                         'power': features['power_watt'],
#                         'duration': features['duration_minutes'],
#                         'scores': [activity_score, habit_score, confidence_score],
#                         'levels': [debug_info['device_activity_level'], 
#                                   debug_info['user_habit_level'], 
#                                   debug_info['confidence_score_level']],
#                         'decision_path': debug_info['decision_path'],
#                         'decision': decision
#                     })

#             except Exception as e:
#                 print(f"   ⚠️ Error processing opportunity {i+1}: {e}")
#                 self.results['delay_decision'] += 1

#         # 打印決策樹統計
#         self._print_decision_tree_stats()

#         # 打印前幾個決策的詳細路徑
#         print(f"\n🔍 決策樹分析 (前5個樣本):")
#         for log in debug_logs[:5]:
#             scores_str = f"{log['scores'][0]:.2f}/{log['scores'][1]:.2f}/{log['scores'][2]:.2f}"
#             levels_str = f"{log['levels'][0]}/{log['levels'][1]}/{log['levels'][2]}"
#             path_str = " -> ".join(log['decision_path'])
#             print(f"   #{log['index']}: {log['time'].strftime('%H:%M')} | "
#                   f"Power: {log['power']:.0f}W | Duration: {log['duration']:.0f}min")
#             print(f"      Scores: {scores_str} | Levels: {levels_str}")
#             print(f"      Path: {path_str} => {log['decision']}")
#             print()

#         return decision_results

#     def _print_decision_tree_stats(self):
#         """打印決策樹統計信息"""
#         print(f"\n🌳 決策樹統計分析:")
#         print(f"   總決策次數: {self.decision_stats['total_decisions']}")

#         print(f"\n🔧 防震盪濾波器統計:")
#         print(f"   被濾波的決策: {self.decision_stats['filtered_decisions']}")
#         print(f"   防止的震盪: {self.decision_stats['oscillation_prevented']}")
#         print(f"   休眠模式檢測: {self.decision_stats.get('sleep_mode_detections', 0)}")      # 🆕 添加
#         print(f"   休眠模式修正: {self.decision_stats.get('sleep_mode_corrections', 0)}")      # 🆕 添加
#         filter_rate = (self.decision_stats['filtered_decisions'] / 
#                     max(1, self.decision_stats['total_decisions']) * 100)
#         print(f"   濾波率: {filter_rate:.1f}%")
        
#         # 打印決策分布
#         total_decisions = sum(self.results.values()) - self.results['phantom_load_detected'] - self.results['total_opportunities']
#         print(f"\n📊 決策分布:")
#         for decision in ['suggest_shutdown', 'send_notification', 'delay_decision', 'keep_on']:
#             count = self.results[decision]
#             percentage = (count / total_decisions * 100) if total_decisions > 0 else 0
#             if count > 0:
#                 print(f"   {decision}: {count} 次 ({percentage:.1f}%)")
        
#         # 打印等級組合統計
#         print(f"\n🎯 等級組合分布 (用戶習慣-設備活動-置信度):")
#         sorted_combinations = sorted(self.decision_stats['level_combinations'].items(), 
#                                    key=lambda x: x[1], reverse=True)
#         for combination, count in sorted_combinations[:10]:  # 顯示前10個
#             percentage = (count / self.decision_stats['total_decisions'] * 100)
#             print(f"   {combination}: {count} 次 ({percentage:.1f}%)")
        
#         # 打印最常見的決策路徑
#         print(f"\n🛤️ 最常見決策路徑 (前5個):")
#         sorted_paths = sorted(self.decision_stats['decision_paths'].items(), 
#                             key=lambda x: x[1], reverse=True)
#         for path, count in sorted_paths[:5]:
#             percentage = (count / self.decision_stats['total_decisions'] * 100)
#             print(f"   {count} 次 ({percentage:.1f}%): {path}")

#     def _calculate_data_period_info(self, df):
#         """計算數據時間範圍資訊"""
#         df['timestamp'] = pd.to_datetime(df['timestamp'])
#         start_date = df['timestamp'].min()
#         end_date = df['timestamp'].max()
#         total_days = (end_date - start_date).days + 1
        
#         return {
#             'start_date': start_date,
#             'end_date': end_date,
#             'total_days': total_days
#         }

#     def _estimate_energy_saving(self, decision_results, df):
#         """修正版節能計算 - 解決節能比例過低問題"""
        
#         # 計算數據期間資訊
#         period_info = self._calculate_data_period_info(df)
#         total_days = period_info['total_days']
        
#         print(f"\n🔍 節能計算詳細分析（修正版）：")
#         print(f"   📅 分析期間: {total_days} 天")
#         print(f"   📊 總決策數量: {len(decision_results)}")
        
#         # 詳細分析每個決策的能耗
#         total_baseline_kwh = 0
#         shutdown_saved_kwh = 0      # 直接關機節省的
#         notification_involved_kwh = 0  # 通知涉及的電量
#         keep_on_kwh = 0            # 繼續使用的
#         delay_kwh = 0              # 延遲決策的
        
#         for i, result in enumerate(decision_results):
#             opp = result['opportunity']
#             decision = result['decision']
            
#             # 計算這個機會的基本信息
#             duration_hr = (opp['end_time'] - opp['start_time']).total_seconds() / 3600
#             power_watt = opp.get('power_watt', 15)  # 🔧 改為15W，更符合phantom load
#             energy_kwh = power_watt * duration_hr / 1000
            
#             total_baseline_kwh += energy_kwh
            
#             # 🔧 修正：根據決策類型正確分類能耗
#             if decision == 'suggest_shutdown':
#                 shutdown_saved_kwh += energy_kwh
#             elif decision == 'send_notification':
#                 notification_involved_kwh += energy_kwh
#             elif decision == 'keep_on':
#                 keep_on_kwh += energy_kwh
#             elif decision == 'delay_decision':
#                 delay_kwh += energy_kwh
#             else:
#                 keep_on_kwh += energy_kwh
        
#         # 計算各種場景的節能效果
#         print(f"\n📊 決策分類統計 (修正版):")
#         print(f"   🔴 總基線電量: {total_baseline_kwh:.5f} kWh")
#         print(f"   🟢 直接關機節省: {shutdown_saved_kwh:.5f} kWh")
#         print(f"   🟡 通知涉及電量: {notification_involved_kwh:.5f} kWh")
#         print(f"   ⚪ 繼續使用電量: {keep_on_kwh:.5f} kWh")
#         print(f"   ⚫ 延遲決策電量: {delay_kwh:.5f} kWh")
        
#         # 不同 notification 響應率的節能計算
#         notification_response_scenarios = {
#             '0% 用戶響應': 0.0,
#             '50% 用戶響應': 0.5,
#             '80% 用戶響應': 0.8,
#             '100% 用戶響應': 1.0
#         }
        
#         print(f"\n🎯 不同場景的節能效果:")
#         print("場景           | 總節省電量(kWh) | 剩餘消耗(kWh) | 節能率 | 節省電費(£)")
#         print("-" * 75)
        
#         scenario_results = {}
        
#         for scenario_name, response_rate in notification_response_scenarios.items():
#             # 計算這個場景下的總節省
#             notification_saved = notification_involved_kwh * response_rate
#             total_saved = shutdown_saved_kwh + notification_saved
#             remaining_consumption = total_baseline_kwh - total_saved
#             savings_rate = (total_saved / total_baseline_kwh * 100) if total_baseline_kwh > 0 else 0
#             cost_saved = total_saved * self.uk_electricity_rate
            
#             scenario_results[scenario_name] = {
#                 'total_saved_kwh': total_saved,
#                 'remaining_kwh': remaining_consumption,
#                 'savings_rate': savings_rate,
#                 'cost_saved': cost_saved
#             }
            
#             print(f"{scenario_name:15s} | {total_saved:15.5f} | {remaining_consumption:13.5f} | "
#                 f"{savings_rate:6.1f}% | £{cost_saved:.4f}")
        
#         # 🎉 最終對比報告
#         best_case = scenario_results['100% 用戶響應']
#         print(f"\n{'='*60}")
#         print(f"🎉 【修正版】原本 vs 智能系統後對比")
#         print(f"{'='*60}")
#         print(f"📊 期間總耗能對比 (最佳情況 - 100%用戶響應)：")
#         print(f"   🔴 原本總耗能：    {total_baseline_kwh:.5f} kWh (£{total_baseline_kwh * self.uk_electricity_rate:.5f})")
#         print(f"   🟢 智能系統後耗能：{best_case['remaining_kwh']:.5f} kWh (£{best_case['remaining_kwh'] * self.uk_electricity_rate:.5f})")
#         print(f"   💚 確定節省：      {best_case['total_saved_kwh']:.5f} kWh (£{best_case['cost_saved']:.5f})")
#         print(f"   📉 節能比例：      {best_case['savings_rate']:.1f}%")
        
#         # 保守情況
#         conservative_case = scenario_results['50% 用戶響應']
#         print(f"\n📊 期間總耗能對比 (保守情況 - 50%用戶響應)：")
#         print(f"   🔴 原本總耗能：    {total_baseline_kwh:.5f} kWh (£{total_baseline_kwh * self.uk_electricity_rate:.5f})")
#         print(f"   🟢 智能系統後耗能：{conservative_case['remaining_kwh']:.5f} kWh (£{conservative_case['remaining_kwh'] * self.uk_electricity_rate:.5f})")
#         print(f"   💚 確定節省：      {conservative_case['total_saved_kwh']:.5f} kWh (£{conservative_case['cost_saved']:.5f})")
#         print(f"   📉 節能比例：      {conservative_case['savings_rate']:.1f}%")
        
#         print(f"{'='*60}")
        
#         return scenario_results

#     def _create_energy_saving_visualization(self, decision_breakdown, notification_scenarios, 
#                                           total_baseline_kwh, fixed_saving_kwh, notification_kwh, total_days):
#         """創建詳細的節能視覺化分析（兩張圖兩張圖顯示）"""

#         # 第一組圖：決策分布和能耗分析
#         plt.style.use('default')
#         fig1 = plt.figure(figsize=(20, 10))
#         fig1.suptitle('Decision Tree Intelligent Power Management - Decision Analysis', 
#                      fontsize=16, fontweight='bold', y=0.95)
        
#         colors = {
#             'suggest_shutdown': '#FF6B6B',
#             'send_notification': '#4ECDC4', 
#             'delay_decision': '#45B7D1',
#             'keep_on': '#96CEB4',
#             'baseline': '#FFE66D',
#             'saved': '#66D9EF',
#             'remaining': '#F8F8F2'
#         }
        
#         decision_labels = {
#             'suggest_shutdown': 'Shutdown',
#             'send_notification': 'Send Notification', 
#             'delay_decision': 'Delay Decision',
#             'keep_on': 'Maintain State'
#         }
        
#         # 1. 決策分布圓餅圖
#         ax1 = fig1.add_subplot(1, 2, 1)
#         decisions = []
#         counts = []
#         decision_colors = []
        
#         for k in ['suggest_shutdown', 'send_notification', 'delay_decision', 'keep_on']:
#             if decision_breakdown[k]['count'] > 0:
#                 decisions.append(decision_labels[k])
#                 counts.append(decision_breakdown[k]['count'])
#                 decision_colors.append(colors[k])
        
#         if len(decisions) > 0:
#             wedges, texts, autotexts = ax1.pie(counts, labels=decisions, colors=decision_colors, 
#                                               autopct='%1.1f%%', startangle=90,     
#                                               textprops={'fontsize': 12})
#             for autotext in autotexts:
#                 autotext.set_color('white')
#                 autotext.set_fontweight('bold')
        
#         ax1.set_title('Daily Decision Distribution', fontweight='bold', fontsize=14, pad=20)
        
#         # 2. 能耗分布柱狀圖
#         ax2 = fig1.add_subplot(1, 2, 2)
#         decision_names = []
#         kwh_values = []
#         bar_colors = []
        
#         for k in ['suggest_shutdown', 'send_notification', 'delay_decision', 'keep_on']:
#             if decision_breakdown[k]['kwh'] > 0:
#                 decision_names.append(decision_labels[k])
#                 kwh_values.append(decision_breakdown[k]['kwh'] / total_days)  # 轉為日平均
#                 bar_colors.append(colors[k])
        
#         if len(decision_names) > 0:
#             x_pos = np.arange(len(decision_names))
#             bars = ax2.bar(x_pos, kwh_values, color=bar_colors, alpha=0.9, 
#                           edgecolor='white', linewidth=2, width=0.6)
            
#             ax2.set_xlabel('Decision Type', fontsize=12, fontweight='bold')
#             ax2.set_ylabel('Daily Average Power Consumption (kWh/day)', fontsize=12, fontweight='bold')
#             ax2.set_title('Daily Average Energy Consumption Distribution by Decision Type', fontweight='bold', fontsize=14, pad=20)
#             ax2.set_xticks(x_pos)
#             ax2.set_xticklabels(decision_names, rotation=0, ha='center', fontsize=11)
#             ax2.grid(True, alpha=0.3, axis='y')
            
#             # 添加數值標籤
#             for bar, value in zip(bars, kwh_values):
#                 height = bar.get_height()
#                 ax2.text(bar.get_x() + bar.get_width()/2., height + max(kwh_values)*0.02,
#                         f'{value:.3f}\nkWh/day', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
#         plt.tight_layout()
#         plt.show()

#         # 第二組圖：Phantom Load能源和電費分析
#         fig2 = plt.figure(figsize=(20, 10))
#         fig2.suptitle('Daily Phantom Load Energy & Cost Distribution', 
#                      fontsize=16, fontweight='bold', y=0.95)
        
#         # 1. 能源分配圓環圖
#         ax3 = fig2.add_subplot(1, 2, 1)
        
#         daily_baseline = total_baseline_kwh / total_days
#         daily_fixed_saving = fixed_saving_kwh / total_days
#         daily_notification = notification_kwh / total_days
#         daily_remaining = daily_baseline - daily_fixed_saving - daily_notification
        
#         energy_categories = []
#         energy_values = []
#         energy_colors = []
        
#         if daily_fixed_saving > 0:
#             energy_categories.append(f'Determine Savings\n({daily_fixed_saving:.3f} kWh/day, £{daily_fixed_saving * self.uk_electricity_rate:.3f}/day)')
#             energy_values.append(daily_fixed_saving)
#             energy_colors.append(colors['suggest_shutdown'])
        
#         if daily_notification > 0:
#             energy_categories.append(f'Possible Savings\n({daily_notification:.3f} kWh/day, £{daily_notification * self.uk_electricity_rate:.3f}/day)')
#             energy_values.append(daily_notification)
#             energy_colors.append(colors['send_notification'])
        
#         if daily_remaining > 0:
#             energy_categories.append(f'Remain Using\n({daily_remaining:.3f} kWh/day, £{daily_remaining * self.uk_electricity_rate:.3f}/day)')
#             energy_values.append(daily_remaining)
#             energy_colors.append(colors['keep_on'])
        
#         if len(energy_categories) > 0:
#             wedges, texts, autotexts = ax3.pie(energy_values, labels=energy_categories, 
#                                               colors=energy_colors, autopct='%1.0f%%', 
#                                               startangle=90, textprops={'fontsize': 10})
            
#             # 創建圓環效果
#             centre_circle = plt.Circle((0,0), 0.4, fc='white')
#             ax3.add_artist(centre_circle)
            
#             # 在中心添加總電量
#             ax3.text(0, 0, f'Total Phantom Load\n{daily_baseline:.3f} kWh/day\n£{daily_baseline * self.uk_electricity_rate:.3f}/day', 
#                     ha='center', va='center', fontsize=11, fontweight='bold')
        
#         ax3.set_title('Daily Phantom Load Energy Distribution', fontweight='bold', fontsize=14, pad=20)
        
#         # 2. Send Notification 用戶響應率影響分析
#         ax4 = fig2.add_subplot(1, 2, 2)
#         if notification_scenarios and len(notification_scenarios) > 0:
#             scenarios = list(notification_scenarios.keys())
#             scenario_labels = []
#             for s in scenarios:
#                 rate = notification_scenarios[s]['response_rate']
#                 scenario_labels.append(f'{int(rate*100)}% Agree')
            
#             daily_total_saved = [notification_scenarios[s]['daily_saved_kwh'] for s in scenarios]
#             daily_cost_saved = [s * self.uk_electricity_rate for s in daily_total_saved]
#             savings_percentage = [notification_scenarios[s]['savings_percentage'] for s in scenarios]
            
#             x = np.arange(len(scenarios))
#             width = 0.25
            
#             # 三個柱狀圖：日節省電量、日節省電費、節能百分比
#             bars1 = ax4.bar(x - width, daily_total_saved, width, label='Power Saving (kWh/day)', 
#                            color=colors['saved'], alpha=0.8, edgecolor='white', linewidth=2)
#             bars2 = ax4.bar(x, daily_cost_saved, width, label='Cost Saving (£/day)', 
#                            color=colors['send_notification'], alpha=0.8, edgecolor='white', linewidth=2)
            
#             # 右側Y軸顯示百分比
#             ax4_twin = ax4.twinx()
#             bars3 = ax4_twin.bar(x + width, savings_percentage, width, label='Energy Saving Rate (%)', 
#                                 color='orange', alpha=0.8, edgecolor='white', linewidth=2)
            
#             ax4.set_xlabel('User Response Rate', fontsize=12, fontweight='bold')
#             ax4.set_ylabel('Daily Savings', fontsize=12, fontweight='bold')
#             ax4_twin.set_ylabel('Energy Saving Rate (%)', color='orange', fontsize=12, fontweight='bold')
#             ax4.set_title('Send Notification: Energy & Cost Saving Effects with Different User Response Rates', 
#                          fontweight='bold', fontsize=14, pad=20)
#             ax4.set_xticks(x)
#             ax4.set_xticklabels(scenario_labels, fontsize=10)
#             ax4.grid(True, alpha=0.3, axis='y')
            
#             # 數值標籤
#             for i, (bar, value) in enumerate(zip(bars1, daily_total_saved)):
#                 height = bar.get_height()
#                 ax4.text(bar.get_x() + bar.get_width()/2., height + max(daily_total_saved)*0.02,
#                         f'{value:.3f}', ha='center', va='bottom', fontsize=9, 
#                         color=colors['saved'], fontweight='bold')
                             
#             for i, (bar, value) in enumerate(zip(bars2, daily_cost_saved)):
#                 height = bar.get_height()
#                 ax4.text(bar.get_x() + bar.get_width()/2., height + max(daily_cost_saved)*0.02,
#                         f'£{value:.3f}', ha='center', va='bottom', fontsize=9, 
#                         color=colors['send_notification'], fontweight='bold')
                        
#             for i, (bar, value) in enumerate(zip(bars3, savings_percentage)):
#                 height = bar.get_height()
#                 ax4_twin.text(bar.get_x() + bar.get_width()/2., height + max(savings_percentage)*0.02,
#                              f'{value:.1f}%', ha='center', va='bottom', fontsize=9, 
#                              color='orange', fontweight='bold')
            
#             ax4.legend(loc='upper left', fontsize=10)
#             ax4_twin.legend(loc='upper right', fontsize=10)
#         else:
#             # 如果沒有notification，顯示說明
#             ax4.text(0.5, 0.5, '本次分析無發送通知決策\n所有節能效果為確定值', 
#                     ha='center', va='center', transform=ax4.transAxes, fontsize=14,
#                     bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
#             ax4.set_title('🔔 發送通知分析', fontweight='bold', fontsize=14)
        
#         plt.tight_layout()
#         plt.show()

#     def _print_final_phantom_load_report(self, total_baseline_kwh, fixed_saving_kwh, 
#                                        notification_kwh, notification_scenarios, total_days):
#         """打印最終的Phantom Load節能報告（含英國電費）"""
        
#         # 計算日平均和年度數據
#         daily_baseline = total_baseline_kwh / total_days
#         daily_fixed_saving = fixed_saving_kwh / total_days
#         daily_notification = notification_kwh / total_days
        
#         annual_baseline = daily_baseline * 365
#         annual_fixed_saving = daily_fixed_saving * 365
        
#         daily_baseline_cost = daily_baseline * self.uk_electricity_rate
#         daily_fixed_saving_cost = daily_fixed_saving * self.uk_electricity_rate
#         annual_baseline_cost = annual_baseline * self.uk_electricity_rate
#         annual_fixed_saving_cost = annual_fixed_saving * self.uk_electricity_rate
        
#         # 基於數據推算的年度總電費（假設Phantom Load佔總用電的某個比例）
#         estimated_total_annual_cost = annual_baseline_cost / 0.23  # 假設phantom load佔總用電23%
        
#         print("\n" + "="*100)
#         print("🎉 決策樹版智能電源管理 - 最終Phantom Load節能報告（含英國電費 £0.30/kWh）")
#         print("📅 (基於 {:.1f} 天數據的日平均分析)".format(total_days))
#         print("="*100)
        
#         print(f"📊 系統分析結果摘要（僅針對Phantom Load部分）：")
#         print(f"   🔋 日平均phantom load耗電量: {daily_baseline:.2f} kWh")
#         print(f"   💰 日平均phantom load電費: £{daily_baseline_cost:.3f}")
#         print(f"   📅 年度phantom load電費: £{annual_baseline_cost:.0f}")
        
#         print(f"\n✅ 確定節能效果（suggest_shutdown）：")
#         print(f"   💡 確定節省電量: {daily_fixed_saving:.2f} kWh/日 ({annual_fixed_saving:.0f} kWh/年)")
#         print(f"   💰 確定節省電費: £{daily_fixed_saving_cost:.3f}/日 (£{annual_fixed_saving_cost:.0f}/年)")
#         phantom_saving_rate = (daily_fixed_saving / daily_baseline * 100) if daily_baseline > 0 else 0
#         print(f"   📈 Phantom load節能率: {phantom_saving_rate:.1f}%")
        
#         # 相對於不同家庭年度電費的節省百分比
#         print(f"\n📊 相對於年度總電費的節省百分比：")
#         household_types = {
#             '🏠 中型家庭': 1050,
#             '🇬🇧 英國平均': 1200, 
#             '🏢 大型家庭': 1500,
#             f'📊 基於您的數據推算': estimated_total_annual_cost
#         }
        
#         for household_type, annual_cost in household_types.items():
#             saving_percentage = (annual_fixed_saving_cost / annual_cost * 100)
#             print(f"   {household_type} (£{annual_cost:.0f}/年): {saving_percentage:.1f}%")
        
#         if notification_scenarios and len(notification_scenarios) > 0:
#             print(f"\n🔔 Send Notification 潛在節能效果：")
#             print(f"   📬 總通知次數: {len(notification_scenarios)} 種情境")
#             print(f"   ⚡ 涉及電量：{daily_notification:.2f} kWh/日")
#             print(f"   💰 涉及電費：£{daily_notification * self.uk_electricity_rate:.3f}/日")
            
#             # 最佳和最差情況
#             best_case = max(notification_scenarios.values(), key=lambda x: x['annual_saved_cost'])
#             worst_case = min(notification_scenarios.values(), key=lambda x: x['annual_saved_cost'])
            
#             print(f"\n🏆 最佳情況 (100%用戶同意):")
#             print(f"   🎯 年度最大節省: £{best_case['annual_saved_cost']:.0f}")
#             print(f"   📈 相對英國平均電費: {(best_case['annual_saved_cost']/1200*100):.1f}%")
            
#             print(f"\n🔻 最差情況 (0%用戶同意):")
#             print(f"   🎯 年度最小節省: £{worst_case['annual_saved_cost']:.0f}")
#             print(f"   📈 相對英國平均電費: {(worst_case['annual_saved_cost']/1200*100):.1f}%")
            
#             print(f"\n📊 潛在節能範圍:")
#             range_cost = best_case['annual_saved_cost'] - worst_case['annual_saved_cost']
#             print(f"   💰 年度電費範圍: £{worst_case['annual_saved_cost']:.0f} - £{best_case['annual_saved_cost']:.0f}")
#             print(f"   📈 相對電費百分比範圍: {(worst_case['annual_saved_cost']/1200*100):.1f}% - {(best_case['annual_saved_cost']/1200*100):.1f}%")
            
#             # 生活化效益比較
#             best_annual_saving = best_case['annual_saved_cost']
#             print(f"\n⚡ 生活化效益比較（基於最佳情況）：")
#             print(f"   📅 相當於 {(best_annual_saving/(1200/365)):.0f} 天的免費電力")
#             print(f"   📺 相當於 {(best_annual_saving/10.99):.1f} 個月的Netflix訂閱")
#             print(f"   ☕ 相當於 {(best_annual_saving/3.5):.0f} 杯咖啡")
            
#         else:
#             print(f"\n🔔 本次分析無 Send Notification 決策")
            
#             # 生活化效益比較
#             print(f"\n⚡ 生活化效益比較（確定節省）：")
#             print(f"   📅 相當於 {(annual_fixed_saving_cost/(1200/365)):.0f} 天的免費電力")
#             print(f"   📺 相當於 {(annual_fixed_saving_cost/10.99):.1f} 個月的Netflix訂閱")
#             print(f"   ☕ 相當於 {(annual_fixed_saving_cost/3.5):.0f} 杯咖啡")
        
#         # 環境效益
#         co2_factor = 0.233  # kg CO2 per kWh in UK
#         annual_co2_saving = annual_fixed_saving * co2_factor
#         cars_equivalent = annual_co2_saving / 4600  # 平均汽車年排放4.6噸CO2
        
#         print(f"\n🌱 環境效益：")
#         print(f"   🌍 每年減少 {annual_co2_saving:.0f} kg CO₂ 排放")
#         print(f"   🚗 相當於減少 {cars_equivalent:.2f} 輛汽車一年的排放")
        
#         print("="*100)

#     def test(self, samples):
#         print("\n🧪 測試決策樹模型決策系統...")
#         for i, sample in enumerate(samples, 1):
#             timestamp = sample["start_time"]
            
#             # 獲取分數
#             if self.device_activity_model:
#                 try:
#                     activity_result = self.device_activity_model.calculate_activity_score(timestamp)
#                     activity = activity_result['activity_score']
#                 except:
#                     activity = self._fallback_activity_score({}, timestamp)
#             else:
#                 activity = self._fallback_activity_score({}, timestamp)

#             if self.user_habit_model:
#                 try:
#                     habit_result = self.user_habit_model.calculate_habit_score(timestamp)
#                     habit = habit_result['habit_score']
#                 except:
#                     habit = self._fallback_habit_score({}, timestamp)
#             else:
#                 habit = self._fallback_habit_score({}, timestamp)

#             if self.confidence_model:
#                 try:
#                     confidence_result = self.confidence_model.calculate_confidence_score(timestamp)
#                     confidence = confidence_result['confidence_score']
#                 except:
#                     confidence = self._fallback_confidence_score({}, timestamp)
#             else:
#                 confidence = self._fallback_confidence_score({}, timestamp)

#             features = {
#                 "device_id": "test_device",
#                 "duration_minutes": 60,
#                 "hour_of_day": timestamp.hour,
#                 "power_watt": sample.get("avg_power", 100),
#                 "weekday": timestamp.weekday()
#             }

#             decision, debug_info = self._make_intelligent_decision(activity, habit, confidence, features)

#             print(f"--- 第 {i} 筆測試 ---")
#             print(f"🕒 時間：{timestamp}")
#             print(f"⚡ 功率：{sample.get('avg_power', 100)} W")
#             print(f"📈 分數: Activity:{activity:.2f} Habit:{habit:.2f} Confidence:{confidence:.2f}")
#             print(f"🎯 等級: {debug_info['device_activity_level']}-{debug_info['user_habit_level']}-{debug_info['confidence_score_level']}")
#             print(f"🛤️ 決策路徑: {' -> '.join(debug_info['decision_path'])}")
#             print(f"🧠 最終決策：{decision}")
#             print()

#     def debug_decision_flow(self, sample_opportunity):
#         """調試決策流程"""
#         print("🔍 調試決策流程:")
        
#         try:
#             features = self._extract_enhanced_features(sample_opportunity, None)
#             timestamp = sample_opportunity['start_time']
            
#             print(f"1. 提取特徵: {features}")
            
#             # 測試分數計算
#             activity_score = 0.3
#             habit_score = 0.4
#             confidence_score = 0.2
            
#             print(f"2. 分數: activity={activity_score}, habit={habit_score}, confidence={confidence_score}")
            
#             # 測試原始決策
#             decision, debug_info = self._make_intelligent_decision(
#                 activity_score, habit_score, confidence_score, features
#             )
#             print(f"3. 原始決策: {decision}")
            
#             # 測試濾波器
#             filter_result = self.anti_oscillation_filter.filter_decision(
#                 original_decision=decision,
#                 power_value=features['power_watt'],
#                 timestamp=timestamp
#             )
#             print(f"4. 濾波結果: {filter_result}")
            
#             final_decision = filter_result['filtered_decision']
#             print(f"5. 最終決策: {final_decision}")
            
#             # 檢查是否在results字典中
#             print(f"6. 是否在results中: {final_decision in self.results}")
#             print(f"7. results字典keys: {list(self.results.keys())}")
            
#         except Exception as e:
#             print(f"❌ 調試過程中發生錯誤: {e}")
#             import traceback
#             traceback.print_exc()

#     def run_analysis(self):
#         """運行決策樹版完整分析"""
#         print("\n" + "="*80)
#         print("開始運行決策樹版智能電源管理分析")
#         print("="*80)
        
#         try:
#             df = pd.read_csv(self.data_file)
#             print(f"✅ 成功載入數據：{len(df)} 筆記錄")
#         except Exception as e:
#             print(f"❌ 無法讀取 CSV: {e}")
#             return

#         # 生成機會點
#         opportunities = self._generate_phantom_load_opportunities(df)
#         print(f"✅ 建立 {len(opportunities)} 筆機會點")

#         if len(opportunities) > 0:
#             print("\n🔍 執行決策流程調試:")
#             self.debug_decision_flow(opportunities[0])

#         # 應用決策樹決策
#         decision_results = self._apply_decision_tree_models(opportunities, df)

#         # 顯示詳細結果
#         print("\n📋 前 5 筆決策結果詳情：")
#         for i, result in enumerate(decision_results[:5], start=1):
#             opp = result['opportunity']
#             debug = result['debug_info']
#             print(f"\n--- 第 {i} 筆 ---")
#             print(f"🕒 時間：{opp['start_time'].strftime('%H:%M')} ~ {opp['end_time'].strftime('%H:%M')}")
#             print(f"⚡ 平均功率：{opp['power_watt']:.1f} W")
#             print(f"⏱️ 持續時間：{result['features']['duration_minutes']:.0f} 分鐘")
#             print(f"📊 原始分數: A:{result['activity_score']:.2f} H:{result['user_habit_score']:.2f} C:{result['confidence_score']:.2f}")
#             print(f"🎯 轉換等級: {debug['device_activity_level']}-{debug['user_habit_level']}-{debug['confidence_score_level']}")
#             print(f"🛤️ 決策路徑: {' -> '.join(debug['decision_path'])}")
#             print(f"🧠 最終決策：{result['decision']}")

#         # 計算節能效果
#         self._estimate_energy_saving(decision_results, df)
#         from collections import Counter
#         decisions = [result['decision'] for result in decision_results]
#         decision_counts = Counter(decisions)
#         total_decisions = len(decisions)

#         print(f"\n🔍 決策分布調試:")
#         print(f"   總決策數: {total_decisions}")
#         for decision, count in decision_counts.items():
#             percentage = (count / total_decisions * 100) if total_decisions > 0 else 0
#             print(f"   {decision}: {count} 次 ({percentage:.1f}%)")

#         active_decisions = decision_counts.get('suggest_shutdown', 0) + decision_counts.get('send_notification', 0)
#         active_ratio = (active_decisions / total_decisions * 100) if total_decisions > 0 else 0
#         print(f"   📊 主動節能決策比例: {active_ratio:.1f}%")

#         if HAS_EVALUATOR and _decision_evaluator is not None:
#             try:
#                 print("\n" + "="*80)
#                 print("🔍 DecisionEvaluator 最終評估報告")
#                 print("="*80)
                
#                 # 匯出評估結果
#                 evaluation_file = _decision_evaluator.export_evaluation_results('decision_tree_evaluation_log.csv')
#                 if evaluation_file:
#                     print(f"✅ 決策評估結果已匯出: {evaluation_file}")
                
#                 # 獲取評估摘要
#                 evaluation_summary = _decision_evaluator.get_evaluation_summary()
                
#                 if 'average_scores' in evaluation_summary:
#                     print(f"\n📊 評估摘要:")
#                     print(f"   評估次數: {evaluation_summary['evaluation_count']}")
                    
#                     avg_scores = evaluation_summary['average_scores']
#                     # print(f"\n🎯 平均評估分數:")
#                     # print(f"   - 穩定性分數: {avg_scores['stability']:.3f}")
#                     # print(f"   - 一致性分數: {avg_scores['consistency']:.3f}")
#                     # print(f"   - 準確性分數: {avg_scores['accuracy']:.3f}")
#                     # print(f"   - 綜合評估分數: {avg_scores['overall']:.3f}")
                    
#                     # 根據分數給出建議
#                     overall_score = avg_scores['overall']
#                     # print(f"\n🏆 系統性能評級:")
#                     # if overall_score > 0.8:
#                     #     print("   ✅ 優秀 - 決策系統性能優秀，運行穩定")
#                     # elif overall_score > 0.6:
#                     #     print("   ⚠️ 良好 - 決策系統性能良好，但有改進空間")
#                     # else:
#                     #     print("   ❌ 需要改進 - 決策系統性能較差，需要重新檢查")
                
#             except Exception as e:
#                 print(f"❌ 評估結果處理錯誤: {e}")

#         # 運行測試
#         test_samples = [
#             {"avg_power": 150, "start_time": datetime(2024, 3, 26, 9, 0)},   # medium activity
#             {"avg_power": 80,  "start_time": datetime(2024, 5, 26, 13, 0)},  # low power, work time
#             {"avg_power": 50,  "start_time": datetime(2024, 7, 26, 20, 0)},  # very low power, evening
#             {"avg_power": 30,  "start_time": datetime(2024, 9, 26, 2, 30)},  # very low power, night
#             {"avg_power": 100, "start_time": datetime(2024, 11, 26, 18, 30)}, # medium power, evening
#         ]


#         # 🆕 顯示濾波器最終狀態
#         print(f"\n🔧 防震盪濾波器最終狀態:")
#         filter_status = self.anti_oscillation_filter.get_filter_status()
#         print(f"   當前功率狀態: {filter_status['current_power_state']}")
#         print(f"   狀態持續時間: {filter_status['state_duration_minutes']:.1f} 分鐘")
#         print(f"   最後決策: {filter_status['last_decision']}")
#         print(f"   是否在冷卻期: {filter_status['is_in_cooldown']}")
#         print(f"   休眠模式檢測: {'是' if filter_status.get('sleep_mode_detected', False) else '否'}")    # 🆕 添加
#         print(f"   休眠持續時間: {filter_status.get('sleep_duration_minutes', 0):.1f} 分鐘")               # 🆕 添加

#         self.test(test_samples)


# if __name__ == '__main__':
#     print("🚀 啟動決策樹版智能電源管理分析系統")
#     print("="*50)
    
#     # 創建決策樹版分析實例
#     analysis = DecisionTreeSmartPowerAnalysis() 
    
#     # 運行分析
#     analysis.run_analysis()
    
#     print("\n🎉 決策樹版分析完成！")

#     if HAS_EVALUATOR and _decision_evaluator is not None:
#         print(f"\n📋 DecisionEvaluator 最終狀態:")
#         print(f"   歷史記錄數量: {len(_decision_evaluator.decision_history)}")
#         print(f"   評估執行次數: {len(_decision_evaluator.evaluation_results)}")
        
#         if len(_decision_evaluator.evaluation_results) > 0:
#             last_evaluation = _decision_evaluator.evaluation_results[-1]
#             print(f"   最後評估時間: {last_evaluation['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
#             print(f"   最後綜合分數: {last_evaluation['overall_score']['overall_score']:.3f}")
    
#     print("\n🔄 如需重新運行，請重新執行此腳本")
#     print("📊 評估結果已保存，可用於後續分析和改進")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
from collections import deque
warnings.filterwarnings('ignore')

# from . import device_activity
# from . import user_habit
# from . import confidence

### 找三個model ###

try:
    from device_activity import DeviceActivityScoreModule
    HAS_DEVICE_ACTIVITY = True
except ImportError:
    HAS_DEVICE_ACTIVITY = False
    print("⚠️  device_activity 模組未找到")

try:
    from user_habit_nooff import NoShutdownUserHabitScoreModule
    HAS_USER_HABIT = True
except ImportError:
    HAS_USER_HABIT = False
    print("⚠️  user_habit 模組未找到")

try:
    from confidence_score import ConfidenceScoreModule
    HAS_CONFIDENCE_SCORE = True
except ImportError:
    HAS_CONFIDENCE_SCORE = False
    print("⚠️  confidence_score 模組未找到")

class AntiOscillationFilter:
    def __init__(self, 
                 hysteresis_enabled=True,
                 phantom_threshold_low=20,
                 phantom_threshold_high=30,
                 decision_cooldown_seconds=30,
                 min_state_duration_minutes=1,
                 stability_check_enabled=False,
                 
                 # 休眠狀態檢測
                 sleep_mode_detection_enabled=True,
                 sleep_mode_threshold=25,
                 sleep_mode_force_shutdown_minutes=8):
        
        self.hysteresis_enabled = hysteresis_enabled
        self.phantom_low = phantom_threshold_low
        self.phantom_high = phantom_threshold_high
        self.decision_cooldown = timedelta(seconds=decision_cooldown_seconds)
        self.min_state_duration = timedelta(minutes=min_state_duration_minutes)
        self.stability_check_enabled = stability_check_enabled
        
        # 休眠模式檢測參數
        self.sleep_mode_detection_enabled = sleep_mode_detection_enabled
        self.sleep_mode_threshold = sleep_mode_threshold
        self.sleep_mode_force_minutes = sleep_mode_force_shutdown_minutes
        
        self.last_decision_time = None
        self.last_decision = None
        self.current_power_state = 'unknown'
        self.state_start_time = None
        self.recent_powers = deque(maxlen=10)
        
        # 新增狀態追蹤
        self.power_history = deque(maxlen=50)
        self.timestamp_history = deque(maxlen=50)
        self.sleep_mode_start_time = None
        
        print(f"✅ 增強型防震盪濾波器初始化")
        print(f"   - 遲滯閾值: {phantom_threshold_low}W ~ {phantom_threshold_high}W")
        print(f"   - 休眠檢測: {'啟用' if sleep_mode_detection_enabled else '禁用'}")
        if sleep_mode_detection_enabled:
            print(f"     * 休眠閾值: <{sleep_mode_threshold}W")
            print(f"     * 強制關機時間: {sleep_mode_force_shutdown_minutes}分鐘")
    
    def filter_decision(self, original_decision, power_value, timestamp, scores=None):
        # 更新歷史記錄
        self.recent_powers.append(power_value)
        self.power_history.append(power_value)
        self.timestamp_history.append(timestamp)
        
        # 休眠模式檢測
        sleep_mode_result = self._detect_sleep_mode(timestamp, power_value)
        
        # 如果檢測到需要強制關機的休眠狀態
        if sleep_mode_result['force_shutdown']:
            if self._is_likely_sleep_time(timestamp):
                suggested_decision = 'suggest_shutdown'
            elif self._is_work_hours(timestamp):
                suggested_decision = 'send_notification'  # 工作時間比較保守
            else:
                suggested_decision = 'suggest_shutdown'
            
            return {
                'filtered_decision': suggested_decision,
                'original_decision': original_decision,
                'filter_reason': f'時間感知休眠檢測({sleep_mode_result["duration_minutes"]:.1f}分鐘)',
                'power_state': 'sleep_mode',
                'should_use_filtered': True,
                'sleep_mode_detected': True
            }
        
        # 如果是低功率且原決策是keep_on，需要修正
        if (power_value < self.sleep_mode_threshold and 
            original_decision == 'keep_on' and
            sleep_mode_result['is_sleep_mode'] and
            sleep_mode_result['duration_minutes'] > 10):  # 需要持續10分鐘以上
            
            # 先改為通知，而不是直接關機
            if power_value < 16:  # 只有極低功率才直接建議關機
                filtered_decision = 'suggest_shutdown'
            else:
                filtered_decision = 'send_notification'  # 其他情況發通知
            
            return {
                'filtered_decision': filtered_decision,
                'original_decision': original_decision,
                'filter_reason': f'長時間低功率修正(功率{power_value:.1f}W, {sleep_mode_result["duration_minutes"]:.1f}分鐘)',
                'power_state': 'sleep_mode_correction',
                'should_use_filtered': True,
                'sleep_mode_detected': True
            }
        
        # 檢查冷卻期
        if self._in_cooldown_period(timestamp):
            return {
                'filtered_decision': 'delay_decision',
                'original_decision': original_decision,
                'filter_reason': '決策冷卻期內',
                'power_state': self.current_power_state,
                'should_use_filtered': True,
                'sleep_mode_detected': sleep_mode_result['is_sleep_mode']
            }
        
        # 更新功率狀態
        self._update_power_state(power_value, timestamp)
        
        # 檢查持續時間
        if not self._meets_minimum_duration(timestamp):
            return {
                'filtered_decision': 'delay_decision',
                'original_decision': original_decision,
                'filter_reason': '狀態持續時間不足',
                'power_state': self.current_power_state,
                'should_use_filtered': True,
                'sleep_mode_detected': sleep_mode_result['is_sleep_mode']
            }
        
        # 檢查穩定性
        if self.stability_check_enabled and not self._is_power_stable():
            # 如果在休眠模式中震盪，直接建議關機
            if sleep_mode_result['is_sleep_mode']:
                return {
                    'filtered_decision': 'suggest_shutdown',
                    'original_decision': original_decision,
                    'filter_reason': '休眠模式中的功率震盪',
                    'power_state': 'sleep_mode_unstable',
                    'should_use_filtered': True,
                    'sleep_mode_detected': True
                }
            else:
                return {
                    'filtered_decision': 'delay_decision',
                    'original_decision': original_decision,
                    'filter_reason': '功率不穩定',
                    'power_state': self.current_power_state,
                    'should_use_filtered': True,
                    'sleep_mode_detected': False
                }
        
        # 根據功率狀態調整決策
        filtered_decision = self._adjust_decision_by_power_state(original_decision, sleep_mode_result)

        valid_decisions = ['suggest_shutdown', 'send_notification', 'delay_decision', 'keep_on']
        if filtered_decision not in valid_decisions:
            print(f"⚠️ 警告：濾波器返回了無效決策 '{filtered_decision}', 改為 'delay_decision'")
            filtered_decision = 'delay_decision'
        
        # 更新決策歷史
        if filtered_decision != 'delay_decision':
            self.last_decision = filtered_decision
            self.last_decision_time = timestamp
        
        return {
            'filtered_decision': filtered_decision,
            'original_decision': original_decision,
            'filter_reason': '濾波完成',
            'power_state': self.current_power_state,
            'should_use_filtered': filtered_decision != original_decision,
            'sleep_mode_detected': sleep_mode_result['is_sleep_mode']
        }
        
    
    def _detect_sleep_mode(self, current_time, current_power):
        """檢測休眠模式"""
        if not self.sleep_mode_detection_enabled:
            return {
                'is_sleep_mode': False,
                'duration_minutes': 0,
                'force_shutdown': False
            }
        
        # 檢查當前功率是否為休眠狀態
        is_current_sleep = current_power < self.sleep_mode_threshold
        
        # 更新休眠開始時間
        if is_current_sleep:
            if self.sleep_mode_start_time is None:
                self.sleep_mode_start_time = current_time
        else:
            self.sleep_mode_start_time = None
        
        # 計算休眠持續時間
        duration_minutes = 0
        if self.sleep_mode_start_time:
            duration = current_time - self.sleep_mode_start_time
            duration_minutes = duration.total_seconds() / 60
        
        # 判斷是否需要強制關機
        force_shutdown = (duration_minutes >= self.sleep_mode_force_minutes and current_power < 18)
        
        return {
            'is_sleep_mode': is_current_sleep,
            'duration_minutes': duration_minutes,
            'force_shutdown': force_shutdown
        }
    
    def _adjust_decision_by_power_state(self, original_decision, sleep_mode_result):
        """根據功率狀態調整決策 - 加入休眠模式考慮"""
        
        # 如果檢測到休眠模式，優先處理
        if sleep_mode_result['is_sleep_mode']:
            duration = sleep_mode_result['duration_minutes']
            
            if duration > 12:  # 超過12分鐘才考慮修正
                if original_decision == 'keep_on':
                    # 根據功率值決定修正強度
                    if self.recent_powers and np.mean(list(self.recent_powers)[-3:]) < 16:
                        return 'suggest_shutdown'  # 極低功率才直接關機
                    else:
                        return 'send_notification'  # 其他情況發通知
                elif original_decision == 'delay_decision':
                    return 'send_notification'
            elif duration > 6:  # 6-12分鐘之間，輕微修正
                if original_decision == 'keep_on' and self.recent_powers:
                    recent_avg = np.mean(list(self.recent_powers)[-3:])
                    if recent_avg < 16:  # 只修正極低功率的情況
                        return 'send_notification'
        
        # 原有邏輯
        if self.current_power_state == 'uncertain':
            if original_decision in ['suggest_shutdown', 'send_notification']:
                return 'delay_decision'
        
        elif self.current_power_state == 'phantom':
            if len(self.recent_powers) >= 3:
                recent_avg = np.mean(list(self.recent_powers)[-3:])
                # 使用休眠閾值進行更積極的判斷
                if recent_avg < self.sleep_mode_threshold:
                    if original_decision == 'keep_on':
                        return 'suggest_shutdown'  # 直接建議關機
                elif 18 <= recent_avg <= 22:
                    if original_decision == 'keep_on':
                        return 'send_notification'  # 改為通知
        
        elif self.current_power_state == 'active':
            if original_decision == 'suggest_shutdown':
                return 'send_notification'
        
        return original_decision
    
    def get_filter_status(self):
        """獲取濾波器狀態 - 包含休眠檢測狀態"""
        sleep_info = self._detect_sleep_mode(datetime.now(), 
                                           self.recent_powers[-1] if self.recent_powers else 0)
        
        return {
            'current_power_state': self.current_power_state,
            'state_duration_minutes': (
                (datetime.now() - self.state_start_time).total_seconds() / 60 
                if self.state_start_time else 0
            ),
            'last_decision': self.last_decision,
            'recent_powers': list(self.recent_powers),
            'is_in_cooldown': self._in_cooldown_period(datetime.now()) if self.last_decision_time else False,
            'sleep_mode_detected': sleep_info['is_sleep_mode'],
            'sleep_duration_minutes': sleep_info['duration_minutes']
        }
    
    # 保留原有方法
    def _in_cooldown_period(self, timestamp):
        if self.last_decision_time is None:
            return False
        return timestamp - self.last_decision_time < self.decision_cooldown
    
    def _update_power_state(self, power_value, timestamp):
        if not self.hysteresis_enabled:
            new_state = 'phantom' if power_value < 37 else 'active'
        else:
            if power_value <= self.phantom_low:
                new_state = 'phantom'
            elif power_value >= self.phantom_high:
                new_state = 'active'
            else:
                new_state = self.current_power_state if self.current_power_state in ['phantom', 'active'] else 'uncertain'
        
        if new_state != self.current_power_state:
            self.current_power_state = new_state
            self.state_start_time = timestamp
    
    def _meets_minimum_duration(self, timestamp):
        if self.state_start_time is None:
            return False
        duration = timestamp - self.state_start_time
        return duration >= self.min_state_duration
    
    def _is_power_stable(self):
        if len(self.recent_powers) < 3:
            return True
        
        recent_list = list(self.recent_powers)[-5:]
        if len(recent_list) < 3:
            return True
        
        std_dev = np.std(recent_list)
        mean_power = np.mean(recent_list)
        coefficient_of_variation = std_dev / mean_power if mean_power > 0 else 0
        return coefficient_of_variation < 0.1 or std_dev < 2.0
    
    def _is_likely_sleep_time(self, timestamp):
        """判斷是否為可能的睡眠時間"""
        hour = timestamp.hour
        # 深夜到早晨 (23:00-07:00) 更容易接受關機建議
        return hour >= 23 or hour <= 7

    def _is_work_hours(self, timestamp):
        """判斷是否為工作時間"""
        hour = timestamp.hour
        weekday = timestamp.weekday()
        # 工作日的工作時間
        return weekday < 5 and 9 <= hour <= 17


class DecisionTreeSmartPowerAnalysis:
    def __init__(self):
        self.data_file = 'C:/Users/王俞文/OneDrive - University of Glasgow/文件/glasgow/msc project/data/complete_power_data_with_history.csv'
        
        print("start decision tree smart power analysis...")
        
        # 電費設定
        self.uk_electricity_rate = 0.30  # £0.30/kWh
        
        # 初始化並訓練模型
        self.device_activity_model = None
        self.user_habit_model = None
        self.confidence_model = None

        self.anti_oscillation_filter = AntiOscillationFilter(
            hysteresis_enabled=True,
            phantom_threshold_low=20,
            phantom_threshold_high=30,
            decision_cooldown_seconds=30,
            min_state_duration_minutes=2,
            stability_check_enabled=True,
            sleep_mode_detection_enabled=True,
            sleep_mode_threshold=20,              # 休眠閾值
            sleep_mode_force_shutdown_minutes=15
        )
        
        # 決策統計
        self.decision_stats = {
            'total_decisions': 0,
            'decision_paths': {},  # 記錄每種決策路徑
            'level_combinations': {},  # 記錄每種等級組合
            'filtered_decisions': 0,        # 添加這行
            'oscillation_prevented': 0, 
            'sleep_mode_corrections': 0,      # 添加這行
            'sleep_mode_detections': 0  
        }
        
        # 訓練設備活動模型
        if HAS_DEVICE_ACTIVITY:
            try:
                print("\n🔄 正在初始化並訓練設備活動模型...")
                self.device_activity_model = DeviceActivityScoreModule()
                self.device_activity_model.run_complete_analysis(self.data_file)
                print("✅ 設備活動模型訓練完成")
            except Exception as e:
                print(f"❌ 設備活動模型訓練失敗: {e}")
                self.device_activity_model = None
        
        # 訓練用戶習慣模型
        if HAS_USER_HABIT:
            try:
                print("\n🔄 正在初始化並訓練用戶習慣模型...")
                self.user_habit_model = NoShutdownUserHabitScoreModule()
                self.user_habit_model.run_complete_analysis(self.data_file)
                print("✅ 用戶習慣模型訓練完成")
            except Exception as e:
                print(f"❌ 用戶習慣模型訓練失敗: {e}")
                self.user_habit_model = None
        
        # 訓練置信度模型
        if HAS_CONFIDENCE_SCORE:
            try:
                print("\n🔄 正在初始化並訓練置信度模型...")
                self.confidence_model = ConfidenceScoreModule()
                self.confidence_model.run_complete_analysis()
                print("✅ 置信度模型訓練完成")
            except Exception as e:
                print(f"❌ 置信度模型訓練失敗: {e}")
                self.confidence_model = None
        
        print("\n🎉 決策樹版智能電源管理系統初始化完成！")
        
        self.results = {
            'phantom_load_detected': 0,
            'suggest_shutdown': 0,
            'keep_on': 0,
            'send_notification': 0,
            'delay_decision': 0,
            'total_opportunities': 0,
            'error': 0
        }

    def _make_intelligent_decision(self, activity_score, habit_score, confidence_score, features):

        def to_level(score):
            """將連續分數轉換為離散等級"""
            if score < 0.33:
                return "low"
            elif score < 0.66:
                return "medium"
            else:
                return "high"
        
        # 轉換分數為等級
        user_habit = to_level(habit_score)
        device_activity = to_level(activity_score)
        confidence_score_level = to_level(confidence_score)
        
        # 記錄等級組合統計
        combination = f"{user_habit}-{device_activity}-{confidence_score_level}"
        if combination not in self.decision_stats['level_combinations']:
            self.decision_stats['level_combinations'][combination] = 0
        self.decision_stats['level_combinations'][combination] += 1
        
        # 合理的智能決策樹邏輯 - 基於實際使用場景
        decision_path = []
        decision = "delay_decision"  # 默認值
        
        if user_habit == "low":  # 很少使用設備
            decision_path.append("user habit=low")
            
            if device_activity == "low":  # 長時間待機
                decision_path.append("device activity=low")
                if confidence_score_level == "low":
                    decision_path.append("confidence score=low")
                    decision = "suggest_shutdown"  # 很少用+長時間待機+不確定時段 -> 關機
                elif confidence_score_level == "medium":
                    decision_path.append("confidence score=medium")
                    decision = "suggest_shutdown"  # 很少用+長時間待機+中等確定 -> 關機
                elif confidence_score_level == "high":
                    decision_path.append("confidence score=high")
                    decision = "delay_decision"  # 很少用+長時間待機+確定時段，可能特殊情況 -> 等待
                    
            elif device_activity == "medium":  # 中等活躍度
                decision_path.append("device activity=medium")
                if confidence_score_level == "low":
                    decision_path.append("confidence score=low")
                    decision = "delay_decision"  # 很少用但有些活躍+不確定 -> 等待
                elif confidence_score_level == "medium":
                    decision_path.append("confidence score=medium")
                    decision = "send_notification"  # 很少用但有些活躍+中等確定 -> 通知
                elif confidence_score_level == "high":
                    decision_path.append("confidence score=high")
                    decision = "send_notification"  # 很少用但有些活躍+確定時段 -> 通知
                    
            elif device_activity == "high":  # 最近很活躍
                decision_path.append("device activity=high")
                if confidence_score_level == "low":
                    decision_path.append("confidence score=low")
                    decision = "keep_on"  # 很少用但剛剛活躍+不確定 -> 保持
                elif confidence_score_level == "medium":
                    decision_path.append("confidence score=medium")
                    decision = "keep_on"  # 很少用但剛剛活躍+中等確定 -> 保持
                elif confidence_score_level == "high":
                    decision_path.append("confidence score=high")
                    decision = "keep_on"  # 很少用但剛剛活躍+確定 -> 保持
                    
        elif user_habit == "medium":  # 中等使用頻率
            decision_path.append("user habit=medium")
            
            if device_activity == "low":  # 長時間待機
                decision_path.append("device activity=low")
                if confidence_score_level == "low":
                    decision_path.append("confidence score=low")
                    decision = "suggest_shutdown"  # 中等使用+長時間待機+不確定 -> 關機
                elif confidence_score_level == "medium":
                    decision_path.append("confidence score=medium")
                    decision = "suggest_shutdown"  # 中等使用+長時間待機+中等確定 -> 關機
                elif confidence_score_level == "high":
                    decision_path.append("confidence score=high")
                    decision = "send_notification"  # 中等使用+長時間待機+確定時段 -> 通知
                    
            elif device_activity == "medium":  # 中等活躍度
                decision_path.append("device activity=medium")
                if confidence_score_level == "low":
                    decision_path.append("confidence score=low")
                    decision = "delay_decision"  # 中等使用+中等活躍+不確定 -> 等待
                elif confidence_score_level == "medium":
                    decision_path.append("confidence score=medium")
                    decision = "send_notification"  # 中等使用+中等活躍+中等確定 -> 通知
                elif confidence_score_level == "high":
                    decision_path.append("confidence score=high")
                    decision = "keep_on"  # 中等使用+中等活躍+確定peak hour -> 保持
                    
            elif device_activity == "high":  # 最近很活躍
                decision_path.append("device activity=high")
                if confidence_score_level == "low":
                    decision_path.append("confidence score=low")
                    decision = "delay_decision"  # 中等使用+剛剛活躍+不確定 -> 等待
                elif confidence_score_level == "medium":
                    decision_path.append("confidence score=medium")
                    decision = "keep_on"  # 中等使用+剛剛活躍+中等確定 -> 保持
                elif confidence_score_level == "high":
                    decision_path.append("confidence score=high")
                    decision = "keep_on"  # 中等使用+剛剛活躍+確定 -> 保持
                    
        elif user_habit == "high":  # 經常使用設備
            decision_path.append("user habit=high")
            
            if device_activity == "low":  # 長時間待機
                decision_path.append("device activity=low")
                if confidence_score_level == "low":
                    decision_path.append("confidence score=low")
                    decision = "suggest_shutdown"  # 經常使用但長時間待機+不確定 -> 可能睡覺，關機
                elif confidence_score_level == "medium":
                    decision_path.append("confidence score=medium")
                    decision = "delay_decision"  # 經常使用但長時間待機+中等確定 -> 等待
                elif confidence_score_level == "high":
                    decision_path.append("confidence score=high")
                    decision = "delay_decision"  # 經常使用但長時間待機+確定睡眠 -> 等待再決定
                    
            elif device_activity == "medium":  # 中等活躍度
                decision_path.append("device activity=medium")
                if confidence_score_level == "low":
                    decision_path.append("confidence score=low")
                    decision = "suggest_shutdown"  # 經常使用+中等活躍+不確定 -> 異常情況，關機
                elif confidence_score_level == "medium":
                    decision_path.append("confidence score=medium")
                    decision = "send_notification"  # 經常使用+中等活躍+中等確定 -> 通知
                elif confidence_score_level == "high":
                    decision_path.append("confidence score=high")
                    decision = "keep_on"  # 經常使用+中等活躍+確定peak hour -> 保持
                    
            elif device_activity == "high":  # 最近很活躍
                decision_path.append("device activity=high")
                if confidence_score_level == "low":
                    decision_path.append("confidence score=low")
                    decision = "delay_decision"  # 經常使用+剛剛活躍+不確定 -> 等待
                elif confidence_score_level == "medium":
                    decision_path.append("confidence score=medium")
                    decision = "send_notification"  # 經常使用+剛剛活躍+中等確定 -> 通知確認
                elif confidence_score_level == "high":
                    decision_path.append("confidence score=high")
                    decision = "keep_on"  # 經常使用+剛剛活躍+確定 -> 保持
        
        # 記錄決策路徑統計
        path_key = " -> ".join(decision_path) + f" => {decision}"
        if path_key not in self.decision_stats['decision_paths']:
            self.decision_stats['decision_paths'][path_key] = 0
        self.decision_stats['decision_paths'][path_key] += 1
        
        self.decision_stats['total_decisions'] += 1
        
        # 創建詳細的debug信息
        debug_info = {
            'user_habit_level': user_habit,
            'device_activity_level': device_activity,
            'confidence_score_level': confidence_score_level,
            'decision_path': decision_path,
            'scores': {
                'activity_score': activity_score,
                'habit_score': habit_score,
                'confidence_score': confidence_score
            },
            'features': features
        }
        
        return decision, debug_info

    def _fallback_activity_score(self, features, timestamp):
        """改進的fallback活動分數 - 確保多樣化分布"""
        hour = timestamp.hour
        weekday = timestamp.weekday()
        
        # 更明確的分數範圍，確保三個等級都會出現
        if weekday < 5:  # 工作日
            if 9 <= hour <= 17:  # 工作時間 - 偏向 medium/high
                base_score = np.random.choice([0.2, 0.5, 0.8], p=[0.2, 0.4, 0.4])
            elif 18 <= hour <= 22:  # 晚間 - 偏向 high
                base_score = np.random.choice([0.1, 0.4, 0.8], p=[0.1, 0.3, 0.6])
            else:  # 深夜早晨 - 偏向 low
                base_score = np.random.choice([0.1, 0.4, 0.7], p=[0.6, 0.3, 0.1])
        else:  # 週末
            if 8 <= hour <= 22:  # 白天 - 平均分布
                base_score = np.random.choice([0.2, 0.5, 0.8], p=[0.3, 0.4, 0.3])
            else:  # 夜間 - 偏向 low
                base_score = np.random.choice([0.1, 0.4, 0.7], p=[0.7, 0.2, 0.1])
        
        # 添加小幅隨機變動
        variation = np.random.normal(0, 0.1)
        final_score = max(0.05, min(0.95, base_score + variation))
        
        return final_score

    def _fallback_habit_score(self, features, timestamp):
        """改進的fallback習慣分數 - 確保多樣化分布"""
        hour = timestamp.hour
        weekday = timestamp.weekday()
        
        # 更明確的分數範圍
        if weekday < 5:  # 工作日
            if 7 <= hour <= 9 or 18 <= hour <= 23:  # 高使用時段 - 偏向 high
                base_score = np.random.choice([0.1, 0.4, 0.8], p=[0.1, 0.3, 0.6])
            elif 10 <= hour <= 17:  # 工作時間 - 偏向 medium
                base_score = np.random.choice([0.2, 0.5, 0.8], p=[0.3, 0.5, 0.2])
            else:  # 其他時間 - 偏向 low
                base_score = np.random.choice([0.1, 0.4, 0.7], p=[0.6, 0.3, 0.1])
        else:  # 週末
            if 9 <= hour <= 23:  # 週末活躍時間 - 平均分布
                base_score = np.random.choice([0.2, 0.5, 0.8], p=[0.3, 0.4, 0.3])
            else:  # 週末休息時間 - 偏向 low
                base_score = np.random.choice([0.1, 0.4, 0.7], p=[0.7, 0.2, 0.1])
        
        # 添加小幅隨機變動
        variation = np.random.normal(0, 0.1)
        final_score = max(0.05, min(0.95, base_score + variation))
        
        return final_score

    def _fallback_confidence_score(self, features, timestamp):
        """改進的fallback置信度分數 - 確保多樣化分布"""
        hour = timestamp.hour
        
        # 更明確的分數範圍
        if 18 <= hour <= 23:  # 晚間高使用期 - 偏向 high
            base_score = np.random.choice([0.1, 0.4, 0.8], p=[0.1, 0.3, 0.6])
        elif 14 <= hour <= 16:  # 下午可能是低使用期 - 偏向 medium
            base_score = np.random.choice([0.2, 0.5, 0.8], p=[0.3, 0.5, 0.2])
        elif 2 <= hour <= 6:  # 深夜低使用期 - 偏向 low
            base_score = np.random.choice([0.1, 0.4, 0.7], p=[0.6, 0.3, 0.1])
        else:  # 其他時間 - 平均分布
            base_score = np.random.choice([0.2, 0.5, 0.8], p=[0.3, 0.4, 0.3])
        
        # 添加小幅隨機變動
        variation = np.random.normal(0, 0.08)
        final_score = max(0.1, min(0.9, base_score + variation))
        
        return final_score

    def _extract_enhanced_features(self, opportunity, df):
        return {
            'device_id': opportunity.get('device_id', 'unknown'),
            'duration_minutes': (opportunity['end_time'] - opportunity['start_time']).total_seconds() / 60,
            'hour_of_day': opportunity['start_time'].hour,
            'power_watt': opportunity.get('power_watt', 75),
            'weekday': opportunity['start_time'].weekday()
        }


if __name__ == '__main__':
    print("🚀 啟動決策樹版智能電源管理分析系統")
    print("="*50)
    
    # 創建決策樹版分析實例
    analysis = DecisionTreeSmartPowerAnalysis()
    
    print("\n🎉 決策樹版分析初始化完成！")
    print("\n🔄 如需重新運行，請重新執行此腳本")
    print("📊 系統已準備好提供智能決策服務")