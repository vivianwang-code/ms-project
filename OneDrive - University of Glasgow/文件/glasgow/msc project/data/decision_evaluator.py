import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

class DecisionEvaluator:
    """
    增強版DecisionEvaluator - 包含震盪檢測和決策干預
    """
    
    def __init__(self, 
                 window_size_minutes=45,
                 evaluation_interval_minutes=30,
                 oscillation_detection_enabled=True,
                 oscillation_window_minutes=15,
                 min_oscillation_count=5,
                 oscillation_threshold_ratio=0.6,
                 auto_shutdown_enabled=True,
                 shutdown_delay_minutes=2,
                 # 🆕 新增決策干預參數
                 intervention_enabled=True,
                 force_shutdown_on_oscillation=True,
                 force_shutdown_on_long_phantom=True,
                 intervention_cooldown_minutes=30):
        """
        初始化增強版決策評估器
        """
        # 原有評估器參數
        self.window_size = timedelta(minutes=window_size_minutes)
        self.evaluation_interval = timedelta(minutes=evaluation_interval_minutes)
        
        # 震盪檢測參數
        self.oscillation_detection_enabled = oscillation_detection_enabled
        self.oscillation_window = timedelta(minutes=oscillation_window_minutes)
        self.min_oscillation_count = min_oscillation_count
        self.oscillation_threshold_ratio = oscillation_threshold_ratio
        self.auto_shutdown_enabled = auto_shutdown_enabled
        self.shutdown_delay = timedelta(minutes=shutdown_delay_minutes)
        
        # 🆕 決策干預參數
        self.intervention_enabled = intervention_enabled
        self.force_shutdown_on_oscillation = force_shutdown_on_oscillation
        self.force_shutdown_on_long_phantom = force_shutdown_on_long_phantom
        self.intervention_cooldown = timedelta(minutes=intervention_cooldown_minutes)
        self.last_intervention_time = None
        
        # 數據存儲
        self.decision_history = deque()
        self.power_history = deque()
        self.fuzzy_output_history = deque()
        self.prediction_history = deque()
        
        # 評估結果存儲
        self.evaluation_results = []
        self.last_evaluation_time = None
        
        # 震盪檢測存儲
        self.oscillation_events = []
        self.auto_shutdown_events = []
        self.intervention_events = []  # 🆕 干預事件記錄
        self.current_oscillation_start = None
        self.consecutive_phantom_time = None
        self.last_shutdown_time = None
        
        print(f"✅ 增強版DecisionEvaluator初始化完成")
        print(f"   - 評估窗口: {window_size_minutes}分鐘，評估間隔: {evaluation_interval_minutes}分鐘")
        if oscillation_detection_enabled:
            print(f"   - 震盪檢測: 啟用，窗口: {oscillation_window_minutes}分鐘")
        print(f"   - 決策干預: {'啟用' if intervention_enabled else '禁用'}")  # 🆕
    
    def evaluate_and_override_decision(self, original_decision, timestamp, power_value, scores=None):
        """
        🆕 評估並可能覆蓋原始決策 - 類似 AntiOscillationFilter.filter_decision
        
        Returns:
            dict: {
                'final_decision': str,          # 最終決策
                'intervention_applied': bool,   # 是否進行了干預
                'intervention_reason': str,     # 干預原因
                'original_decision': str,       # 原始決策
                'evaluator_warnings': list      # 評估器警告
            }
        """
        if not self.intervention_enabled:
            return {
                'final_decision': original_decision,
                'intervention_applied': False,
                'intervention_reason': '決策干預功能已禁用',
                'original_decision': original_decision,
                'evaluator_warnings': []
            }
        
        # 檢查冷卻期
        if (self.last_intervention_time and 
            timestamp - self.last_intervention_time < self.intervention_cooldown):
            return {
                'final_decision': original_decision,
                'intervention_applied': False,
                'intervention_reason': '干預冷卻期內',
                'original_decision': original_decision,
                'evaluator_warnings': []
            }
        
        warnings = []
        intervention_reasons = []
        
        # 檢查1：嚴重震盪干預
        if self.current_oscillation_start:
            oscillation_duration = (timestamp - self.current_oscillation_start).total_seconds() / 60
            
            if oscillation_duration > 10 and self.force_shutdown_on_oscillation:
                if original_decision not in ['suggest_shutdown']:
                    intervention_reasons.append(f'嚴重震盪({oscillation_duration:.1f}分鐘)，強制關機')
                    warnings.append('檢測到持續震盪，已強制切換為關機決策')
        
        # 檢查2：超長phantom load干預
        if self.consecutive_phantom_time:
            phantom_duration = (timestamp - self.consecutive_phantom_time).total_seconds() / 60
            
            if phantom_duration > 90 and self.force_shutdown_on_long_phantom:  # 90分鐘
                if original_decision not in ['suggest_shutdown']:
                    intervention_reasons.append(f'超長phantom load({phantom_duration:.1f}分鐘)，強制關機')
                    warnings.append('檢測到異常長時間待機，已強制切換為關機決策')
        
        # 檢查3：危險組合干預（震盪 + phantom）
        if (self.current_oscillation_start and self.consecutive_phantom_time and 
            power_value < 5):  # 極低功率
            combo_reasons = []
            
            if self.current_oscillation_start:
                osc_duration = (timestamp - self.current_oscillation_start).total_seconds() / 60
                combo_reasons.append(f'震盪{osc_duration:.1f}分鐘')
            
            if self.consecutive_phantom_time:
                phantom_duration = (timestamp - self.consecutive_phantom_time).total_seconds() / 60
                combo_reasons.append(f'phantom load{phantom_duration:.1f}分鐘')
            
            if combo_reasons and original_decision not in ['suggest_shutdown']:
                intervention_reasons.append(f'危險組合({"+".join(combo_reasons)})，強制關機')
                warnings.append('檢測到震盪與長時間待機組合，強制關機保護')
        
        # 決定最終決策
        if intervention_reasons:
            self.last_intervention_time = timestamp
            final_decision = 'suggest_shutdown'
            intervention_applied = True
            
            # 記錄干預事件
            self.intervention_events.append({
                'timestamp': timestamp,
                'original_decision': original_decision,
                'final_decision': final_decision,
                'reasons': intervention_reasons,
                'power_value': power_value
            })
            
            print(f"🚨 DecisionEvaluator 決策干預!")
            print(f"   原始決策: {original_decision} → 覆蓋為: {final_decision}")
            print(f"   干預原因: {'; '.join(intervention_reasons)}")
            
        else:
            final_decision = original_decision
            intervention_applied = False
        
        return {
            'final_decision': final_decision,
            'intervention_applied': intervention_applied,
            'intervention_reason': '; '.join(intervention_reasons) if intervention_reasons else '無需干預',
            'original_decision': original_decision,
            'evaluator_warnings': warnings
        }

    def add_decision_record(self, timestamp, fuzzy_output, predicted_power, 
                          actual_power, decision, confidence_scores=None):
        """添加決策記錄並進行評估"""

        # 創建記錄
        record = {
            'timestamp': timestamp,
            'fuzzy_output': fuzzy_output,
            'predicted_power': predicted_power,
            'actual_power': actual_power,
            'decision': decision,
            'confidence_scores': confidence_scores or {},
            'binary_decision': self._decision_to_binary(decision)
        }
        
        # 添加到歷史記錄
        self.decision_history.append(record)
        self.power_history.append(actual_power)
        self.fuzzy_output_history.append(fuzzy_output)
        self.prediction_history.append(predicted_power)
        
        # 清理過期記錄
        self._cleanup_old_records(timestamp)
        
        # 原有評估邏輯
        evaluation_triggered = False
        if (self.last_evaluation_time is None or 
            timestamp - self.last_evaluation_time >= self.evaluation_interval):
            self._perform_evaluation(timestamp)
            self.last_evaluation_time = timestamp
            evaluation_triggered = True
        
        # 震盪檢測
        oscillation_result = {}
        if self.oscillation_detection_enabled:
            oscillation_result = self._detect_and_handle_oscillation(timestamp, actual_power)
        
        return {
            'evaluation_triggered': evaluation_triggered,
            'oscillation_detected': oscillation_result.get('oscillation_detected', False),
            'oscillation_intensity': oscillation_result.get('oscillation_intensity', 0),
            'auto_shutdown_recommended': oscillation_result.get('auto_shutdown_recommended', False),
            'shutdown_reason': oscillation_result.get('shutdown_reason', ''),
            'continuous_phantom_minutes': oscillation_result.get('continuous_phantom_minutes', 0)
        }

    def _decision_to_binary(self, decision):
        """將決策轉換為二進制"""
        shutdown_decisions = ['suggest_shutdown', 'send_notification']
        return 1 if decision in shutdown_decisions else 0

    def _cleanup_old_records(self, current_time):
        """清理過期的歷史記錄"""
        max_window = max(self.window_size, self.oscillation_window)
        cutoff_time = current_time - max_window
        
        while self.decision_history and self.decision_history[0]['timestamp'] < cutoff_time:
            removed = self.decision_history.popleft()
            if self.power_history:
                self.power_history.popleft()
            if self.fuzzy_output_history:
                self.fuzzy_output_history.popleft()
            if self.prediction_history:
                self.prediction_history.popleft()

    def _detect_and_handle_oscillation(self, current_time, actual_power):
        """檢測震盪並處理自動關機邏輯"""
        oscillation_window_data = self._get_oscillation_window_data(current_time)
        
        if len(oscillation_window_data) < 3:
            return {
                'oscillation_detected': False,
                'oscillation_intensity': 0,
                'auto_shutdown_recommended': False,
                'shutdown_reason': '數據不足',
                'continuous_phantom_minutes': 0
            }
        
        # 檢測震盪
        oscillation_info = self._analyze_oscillation(oscillation_window_data, current_time)
        
        # 檢測連續phantom load
        phantom_info = self._detect_continuous_phantom_load(current_time, actual_power)
        
        # 評估是否需要自動關機
        shutdown_decision = self._evaluate_auto_shutdown(
            current_time, oscillation_info, phantom_info
        )
        
        return {
            'oscillation_detected': oscillation_info['is_oscillating'],
            'oscillation_intensity': oscillation_info['intensity'],
            'jump_count': oscillation_info['jump_count'],
            'continuous_phantom_minutes': phantom_info['duration_minutes'],
            'auto_shutdown_recommended': shutdown_decision['should_shutdown'],
            'shutdown_reason': shutdown_decision['reason']
        }

    def _get_oscillation_window_data(self, current_time):
        """獲取震盪檢測窗口內的數據"""
        window_start = current_time - self.oscillation_window
        window_data = []
        
        for record in self.decision_history:
            if window_start <= record['timestamp'] <= current_time:
                window_data.append(record)
        
        return window_data

    def _analyze_oscillation(self, window_data, current_time):
        """分析震盪情況"""
        binary_decisions = [record['binary_decision'] for record in window_data]
        
        # 計算跳動次數
        jump_count = 0
        for i in range(1, len(binary_decisions)):
            if binary_decisions[i] != binary_decisions[i-1]:
                jump_count += 1
        
        # 計算震盪強度
        total_decisions = len(binary_decisions) - 1
        jump_ratio = jump_count / total_decisions if total_decisions > 0 else 0
        
        # 判斷是否為震盪
        is_oscillating = (
            jump_count >= self.min_oscillation_count and 
            jump_ratio >= self.oscillation_threshold_ratio
        )
        
        # 記錄震盪事件
        if is_oscillating and not self.current_oscillation_start:
            self.current_oscillation_start = current_time
            self.oscillation_events.append({
                'start_time': current_time,
                'jump_count': jump_count,
                'intensity': jump_ratio
            })
            print(f"🔄 檢測到決策震盪！時間: {current_time.strftime('%H:%M:%S')}, "
                  f"強度: {jump_ratio:.1%}")
        
        elif not is_oscillating and self.current_oscillation_start:
            # 震盪結束
            if self.oscillation_events:
                duration = current_time - self.current_oscillation_start
                self.oscillation_events[-1]['end_time'] = current_time
                self.oscillation_events[-1]['duration_minutes'] = duration.total_seconds() / 60
                print(f"✅ 震盪結束，持續時間: {duration.total_seconds()/60:.1f} 分鐘")
            self.current_oscillation_start = None
        
        return {
            'is_oscillating': is_oscillating,
            'intensity': jump_ratio,
            'jump_count': jump_count,
            'oscillation_start': self.current_oscillation_start
        }

    def _detect_continuous_phantom_load(self, current_time, power_value, phantom_threshold=19):
        """檢測連續phantom load時間"""
        is_phantom = power_value < phantom_threshold
        
        if is_phantom:
            if self.consecutive_phantom_time is None:
                self.consecutive_phantom_time = current_time
        else:
            self.consecutive_phantom_time = None
        
        duration_minutes = 0
        if self.consecutive_phantom_time:
            duration = current_time - self.consecutive_phantom_time
            duration_minutes = duration.total_seconds() / 60
        
        return {
            'is_continuous_phantom': is_phantom and duration_minutes > 0,
            'duration_minutes': duration_minutes,
            'start_time': self.consecutive_phantom_time
        }

    def _evaluate_auto_shutdown(self, current_time, oscillation_info, phantom_info):
        """評估是否應該自動關機"""
        if not self.auto_shutdown_enabled:
            return {'should_shutdown': False, 'reason': '自動關機功能已禁用'}
        
        shutdown_reasons = []
        
        # 情況1：嚴重震盪 + 連續phantom load
        if (oscillation_info['is_oscillating'] and 
            oscillation_info['intensity'] > 0.8 and
            phantom_info['duration_minutes'] > 10):
            shutdown_reasons.append(
                f"嚴重震盪({oscillation_info['intensity']:.1%}) + "
                f"連續phantom load({phantom_info['duration_minutes']:.1f}分鐘)"
            )
        
        # 情況2：長時間震盪
        if (oscillation_info['is_oscillating'] and 
            oscillation_info['oscillation_start'] and
            current_time - oscillation_info['oscillation_start'] > timedelta(minutes=20)):
            duration = (current_time - oscillation_info['oscillation_start']).total_seconds() / 60
            shutdown_reasons.append(f"長時間震盪({duration:.1f}分鐘)")
        
        # 情況3：極長時間phantom load
        if phantom_info['duration_minutes'] > 60:
            shutdown_reasons.append(f"超長時間phantom load({phantom_info['duration_minutes']:.1f}分鐘)")
        
        should_shutdown = len(shutdown_reasons) > 0
        
        return {
            'should_shutdown': should_shutdown,
            'reason': '; '.join(shutdown_reasons) if shutdown_reasons else '未達到自動關機條件'
        }

    def _get_window_data(self, current_time):
        """獲取評估滑動窗口內的數據"""
        window_start = current_time - self.window_size
        window_data = []
        
        for record in self.decision_history:
            if window_start <= record['timestamp'] <= current_time:
                window_data.append(record)
        
        return window_data

    def _perform_evaluation(self, current_time):
        """執行評估"""
        window_data = self._get_window_data(current_time)
        
        if len(window_data) < 2:
            return
        
        # 簡化的評估邏輯
        evaluation_result = {
            'timestamp': current_time,
            'window_size': len(window_data),
            'evaluation_summary': 'Basic evaluation completed'
        }
        
        self.evaluation_results.append(evaluation_result)

    def get_intervention_summary(self):
        """🆕 獲取干預事件摘要"""
        return {
            'total_interventions': len(self.intervention_events),
            'recent_interventions': self.intervention_events[-5:],  # 最近5次
            'intervention_enabled': self.intervention_enabled,
            'last_intervention': self.last_intervention_time.strftime('%H:%M:%S') if self.last_intervention_time else 'N/A'
        }

    def get_oscillation_summary(self):
        """獲取震盪檢測摘要"""
        if not self.oscillation_detection_enabled:
            return {"message": "震盪檢測功能未啟用"}
        
        return {
            'total_oscillation_events': len(self.oscillation_events),
            'total_shutdown_events': len(self.auto_shutdown_events),
            'current_oscillating': self.current_oscillation_start is not None,
            'oscillation_events': self.oscillation_events[-5:],
            'shutdown_events': self.auto_shutdown_events[-3:]
        }
    
    def get_evaluation_summary(self):
        """獲取評估摘要"""
        if not self.evaluation_results:
            return {"message": "尚無評估結果"}
        
        return {
            'total_evaluations': len(self.evaluation_results),
            'intervention_summary': self.get_intervention_summary()  # 🆕 包含干預摘要
        }

    # def export_evaluation_results(self, filename=None):
    #     """匯出評估結果"""
    #     if filename is None:
    #         filename = f"decision_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
    #     print(f"✅ 評估結果將匯出至: {filename}")
    #     return filename