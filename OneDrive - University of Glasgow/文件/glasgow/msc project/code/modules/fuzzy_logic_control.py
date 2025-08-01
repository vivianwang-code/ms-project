import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
from collections import deque
warnings.filterwarnings('ignore')

from . import device_activity
from . import user_habit
from . import confidence

### 找三個model ###

try:
    from .device_activity import DeviceActivityScoreModule
    HAS_DEVICE_ACTIVITY = True
except ImportError:
    HAS_DEVICE_ACTIVITY = False
    print("⚠️  device_activity 模組未找到")

try:
    from .user_habit import NoShutdownUserHabitScoreModule
    HAS_USER_HABIT = True
except ImportError:
    HAS_USER_HABIT = False
    print("⚠️  user_habit 模組未找到")

try:
    from .confidence import ConfidenceScoreModule
    HAS_CONFIDENCE_SCORE = True
except ImportError:
    HAS_CONFIDENCE_SCORE = False
    print("⚠️  confidence_score 模組未找到")

class AntiOscillationFilter:
    def __init__(self, 
                 hysteresis_enabled=True,
                 phantom_threshold_low=17,
                 phantom_threshold_high=21,
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
            new_state = 'phantom' if power_value < 19 else 'active'
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
            phantom_threshold_low=17,
            phantom_threshold_high=21,
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