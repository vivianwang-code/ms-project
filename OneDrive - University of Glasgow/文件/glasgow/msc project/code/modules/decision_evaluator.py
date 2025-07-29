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
    增強版DecisionEvaluator - 包含震盪檢測和自動關機建議
    
    功能：
    - 使用30~60分鐘滑動窗口評估fuzzy controller決策穩定性
    - 計算fuzzy output vs 預測值 vs 實際功率的一致性
    - 檢測決策跳動和不穩定性
    - 提供多種評估指標
    - 🆕 震盪檢測和自動關機建議
    - 🆕 連續phantom load檢測
    """
    
    def __init__(self, 
                 window_size_minutes=45,    # 系統會分析過去 45 分鐘的能耗資料
                 evaluation_interval_minutes=30,     # 每 30 分鐘觸發一次評估/偵測邏輯
                 # 新增震盪檢測參數
                 oscillation_detection_enabled=True,
                 oscillation_window_minutes=15,    # 在最近 15 分鐘內偵測是否有震盪行為
                 min_oscillation_count=5,    # 如果在 15 分鐘內出現超過 5 次開關，就判定為震盪
                 oscillation_threshold_ratio=0.6,     # 如果有超過 60% 的時間內裝置處於震盪狀態，則會被標記為異常行為
                 auto_shutdown_enabled=True,     # 是否啟用自動關閉功能
                 shutdown_delay_minutes=2):    #從判定到真正關閉裝置的延遲時間，給使用者 2 分鐘的時間來取消或中止自動關閉
        """
        初始化增強版決策評估器
        
        Args:
            window_size_minutes (int): 評估滑動窗口大小（分鐘），預設45分鐘
            evaluation_interval_minutes (int): 評估間隔（分鐘），預設30分鐘
            oscillation_detection_enabled (bool): 是否啟用震盪檢測
            oscillation_window_minutes (int): 震盪檢測窗口（分鐘）
            min_oscillation_count (int): 最小震盪次數
            oscillation_threshold_ratio (float): 震盪比例閾值
            auto_shutdown_enabled (bool): 是否啟用自動關機建議
            shutdown_delay_minutes (int): 關機延遲時間
        """
        # 原有評估器參數
        self.window_size = timedelta(minutes=window_size_minutes)
        self.evaluation_interval = timedelta(minutes=evaluation_interval_minutes)
        
        # 新增震盪檢測參數
        self.oscillation_detection_enabled = oscillation_detection_enabled
        self.oscillation_window = timedelta(minutes=oscillation_window_minutes)
        self.min_oscillation_count = min_oscillation_count
        self.oscillation_threshold_ratio = oscillation_threshold_ratio
        self.auto_shutdown_enabled = auto_shutdown_enabled
        self.shutdown_delay = timedelta(minutes=shutdown_delay_minutes)
        
        # 原有數據存儲
        self.decision_history = deque()  # 決策歷史記錄
        self.power_history = deque()     # 功率歷史記錄
        self.fuzzy_output_history = deque()  # Fuzzy輸出歷史
        self.prediction_history = deque()    # 預測值歷史
        
        # 評估結果存儲
        self.evaluation_results = []
        self.last_evaluation_time = None
        
        # 新增震盪檢測存儲
        self.oscillation_events = []
        self.auto_shutdown_events = []
        self.current_oscillation_start = None
        self.consecutive_phantom_time = None
        self.last_shutdown_time = None
        
        # 閾值設定
        self.volatility_threshold = 0.3    # 決策波動性閾值
        self.consistency_threshold = 0.7   # 一致性閾值
        self.correlation_threshold = 0.6   # 相關性閾值
        
        print(f"✅ 增強版DecisionEvaluator初始化完成")
        print(f"   - 評估窗口: {window_size_minutes}分鐘，評估間隔: {evaluation_interval_minutes}分鐘")
        if oscillation_detection_enabled:
            print(f"   - 震盪檢測: 啟用，窗口: {oscillation_window_minutes}分鐘")
            print(f"   - 自動關機建議: {'啟用' if auto_shutdown_enabled else '禁用'}")
        
    def add_decision_record(self, timestamp, fuzzy_output, predicted_power, 
                          actual_power, decision, confidence_scores=None):

        # 創建記錄
        record = {
            'timestamp': timestamp,
            'fuzzy_output': fuzzy_output,
            'predicted_power': predicted_power,
            'actual_power': actual_power,
            'decision': decision,
            'confidence_scores': confidence_scores or {},
            'binary_decision': self._decision_to_binary(decision)  # 新增
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
        
        # 新增震盪檢測
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
        # 保持評估窗口和震盪檢測窗口中較大的一個
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
        # 獲取震盪檢測窗口內的數據
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
            'oscillation_pattern': oscillation_info['pattern'],
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
        # 提取二進制決策
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
        
        # 分析震盪模式
        pattern = self._analyze_oscillation_pattern(binary_decisions)
        
        # 記錄震盪事件
        if is_oscillating and not self.current_oscillation_start:
            self.current_oscillation_start = current_time
            self.oscillation_events.append({
                'start_time': current_time,
                'jump_count': jump_count,
                'intensity': jump_ratio,
                'pattern': pattern
            })
            print(f"🔄 檢測到決策震盪！時間: {current_time.strftime('%H:%M:%S')}, "
                  f"強度: {jump_ratio:.1%}, 模式: {pattern}")
        
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
            'pattern': pattern,
            'oscillation_start': self.current_oscillation_start
        }

    def _analyze_oscillation_pattern(self, binary_decisions):
        """分析震盪模式"""
        if len(binary_decisions) < 4:
            return 'unknown'
        
        # 檢測規律性震盪 (0101 或 1010)
        regular_pattern = True
        for i in range(2, len(binary_decisions)):
            if binary_decisions[i] != binary_decisions[i-2]:
                regular_pattern = False
                break
        
        if regular_pattern:
            return 'regular_alternating'
        
        # 檢測不規律震盪
        unique_sequences = set()
        for i in range(len(binary_decisions) - 2):
            sequence = tuple(binary_decisions[i:i+3])
            unique_sequences.add(sequence)
        
        if len(unique_sequences) > 3:
            return 'chaotic'
        else:
            return 'irregular'

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
        
        # 檢查是否太頻繁關機
        if (self.last_shutdown_time and 
            current_time - self.last_shutdown_time < timedelta(hours=1)):
            return {'should_shutdown': False, 'reason': '距離上次關機時間太短'}
        
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
        
        # 情況4：高頻震盪
        if (oscillation_info['is_oscillating'] and 
            oscillation_info['jump_count'] > 10 and
            phantom_info['duration_minutes'] > 5):
            shutdown_reasons.append(f"高頻震盪({oscillation_info['jump_count']}次跳動)")
        
        should_shutdown = len(shutdown_reasons) > 0
        
        if should_shutdown:
            # 記錄自動關機事件
            self.auto_shutdown_events.append({
                'timestamp': current_time,
                'reasons': shutdown_reasons,
                'oscillation_intensity': oscillation_info['intensity'],
                'phantom_duration': phantom_info['duration_minutes']
            })
            
            self.last_shutdown_time = current_time
            
            print(f"🔌 自動關機建議觸發！時間: {current_time.strftime('%H:%M:%S')}")
            print(f"   原因: {'; '.join(shutdown_reasons)}")
        
        return {
            'should_shutdown': should_shutdown,
            'reason': '; '.join(shutdown_reasons) if shutdown_reasons else '未達到自動關機條件'
        }

    # ========================================
    # 以下是原有的DecisionEvaluator方法（保持不變）
    # ========================================
    
    def _get_window_data(self, current_time):
        """獲取評估滑動窗口內的數據"""
        window_start = current_time - self.window_size
        window_data = []
        
        for record in self.decision_history:
            if window_start <= record['timestamp'] <= current_time:
                window_data.append(record)
        
        return window_data
    
    def _calculate_decision_volatility(self, window_data):
        """計算決策波動性（增強版 - 使用已有的binary_decision）"""
        if len(window_data) < 2:
            return {'volatility': 0, 'jump_count': 0, 'is_stable': True}
        
        # 提取fuzzy輸出和二進制決策
        fuzzy_outputs = [record['fuzzy_output'] for record in window_data]
        binary_decisions = [record['binary_decision'] for record in window_data]
        
        # 計算fuzzy輸出的標準差
        fuzzy_volatility = np.std(fuzzy_outputs) if len(fuzzy_outputs) > 1 else 0
        
        # 計算決策跳動次數
        jump_count = 0
        for i in range(1, len(binary_decisions)):
            if binary_decisions[i] != binary_decisions[i-1]:
                jump_count += 1
        
        # 計算跳動率
        jump_rate = jump_count / (len(binary_decisions) - 1) if len(binary_decisions) > 1 else 0
        
        # 判斷是否穩定
        is_stable = (fuzzy_volatility < self.volatility_threshold and 
                    jump_rate < self.volatility_threshold)
        
        return {
            'volatility': fuzzy_volatility,
            'jump_count': jump_count,
            'jump_rate': jump_rate,
            'is_stable': is_stable
        }

    def _calculate_consistency_metrics(self, window_data):
        """計算一致性指標"""
        if len(window_data) < 2:
            return {
                'cosine_similarity': 1.0,
                'correlation_coefficient': 1.0,
                'prediction_accuracy': 1.0,
                'consistency_score': 1.0
            }
        
        # 提取數據
        fuzzy_outputs = np.array([record['fuzzy_output'] for record in window_data])
        predicted_powers = np.array([record['predicted_power'] for record in window_data])
        actual_powers = np.array([record['actual_power'] for record in window_data])
        
        # 計算余弦相似度
        if np.max(actual_powers) > np.min(actual_powers):
            normalized_powers = (actual_powers - np.min(actual_powers)) / (np.max(actual_powers) - np.min(actual_powers))
        else:
            normalized_powers = np.ones_like(actual_powers) * 0.5
        
        try:
            cosine_sim = cosine_similarity(
                fuzzy_outputs.reshape(1, -1), 
                normalized_powers.reshape(1, -1)
            )[0][0]
        except:
            cosine_sim = 0.0
        
        # 計算相關係數
        try:
            correlation_coeff, _ = pearsonr(predicted_powers, actual_powers)
            if np.isnan(correlation_coeff):
                correlation_coeff = 0.0
        except:
            correlation_coeff = 0.0
        
        # 計算預測準確性
        try:
            mape = np.mean(np.abs((actual_powers - predicted_powers) / 
                                 np.maximum(actual_powers, 1e-8))) * 100
            prediction_accuracy = max(0, (100 - mape) / 100)
        except:
            prediction_accuracy = 0.0
        
        # 綜合一致性分數
        consistency_score = (abs(cosine_sim) + abs(correlation_coeff) + prediction_accuracy) / 3
        
        return {
            'cosine_similarity': cosine_sim,
            'correlation_coefficient': correlation_coeff,
            'prediction_accuracy': prediction_accuracy,
            'mape': mape if 'mape' in locals() else 0,
            'consistency_score': consistency_score
        }

    def _calculate_error_rates(self, window_data):
        """計算誤判率和延遲判斷率"""
        if len(window_data) < 2:
            return {
                'false_positive_rate': 0.0,
                'false_negative_rate': 0.0,
                'delayed_decision_rate': 0.0,
                'total_error_rate': 0.0
            }
        
        phantom_threshold = 19
        false_positives = 0
        false_negatives = 0
        delayed_decisions = 0
        
        for record in window_data:
            actual_power = record['actual_power']
            decision = record['decision']
            
            # 判斷實際是否為待機狀態
            is_phantom = actual_power < phantom_threshold
            
            # 判斷決策是否為關閉建議
            is_shutdown_suggested = decision in ['suggest_shutdown', 'send_notification']
            
            # 計算各種錯誤
            if not is_phantom and is_shutdown_suggested:
                false_positives += 1
            if is_phantom and not is_shutdown_suggested and decision != 'delay_decision':
                false_negatives += 1
            if decision == 'delay_decision':
                delayed_decisions += 1
        
        total_samples = len(window_data)
        
        return {
            'false_positive_rate': false_positives / total_samples,
            'false_negative_rate': false_negatives / total_samples,
            'delayed_decision_rate': delayed_decisions / total_samples,
            'total_error_rate': (false_positives + false_negatives) / total_samples
        }

    def _perform_evaluation(self, current_time):
        """執行評估"""
        window_data = self._get_window_data(current_time)
        
        if len(window_data) < 2:
            return
        
        # 計算各項指標
        volatility_metrics = self._calculate_decision_volatility(window_data)
        consistency_metrics = self._calculate_consistency_metrics(window_data)
        error_metrics = self._calculate_error_rates(window_data)
        
        # 計算綜合評估分數
        overall_score = self._calculate_overall_score(
            volatility_metrics, consistency_metrics, error_metrics
        )
        
        # 生成評估結果
        evaluation_result = {
            'timestamp': current_time,
            'window_size': len(window_data),
            'volatility_metrics': volatility_metrics,
            'consistency_metrics': consistency_metrics,
            'error_metrics': error_metrics,
            'overall_score': overall_score,
            'recommendation': self._generate_recommendation(
                volatility_metrics, consistency_metrics, error_metrics, overall_score
            )
        }
        
        self.evaluation_results.append(evaluation_result)
        

    def _calculate_overall_score(self, volatility_metrics, consistency_metrics, error_metrics):
        """計算綜合評估分數"""
        stability_score = 1.0 - min(1.0, volatility_metrics['volatility'])
        consistency_score = consistency_metrics['consistency_score']
        accuracy_score = 1.0 - error_metrics['total_error_rate']
        
        weights = {'stability': 0.3, 'consistency': 0.4, 'accuracy': 0.3}
        
        overall_score = (
            weights['stability'] * stability_score +
            weights['consistency'] * consistency_score +
            weights['accuracy'] * accuracy_score
        )
        
        return {
            'stability_score': stability_score,
            'consistency_score': consistency_score,
            'accuracy_score': accuracy_score,
            'overall_score': overall_score
        }

    def _generate_recommendation(self, volatility_metrics, consistency_metrics, 
                               error_metrics, overall_score):
        """生成改進建議（增強版 - 包含震盪相關建議）"""
        recommendations = []
        priority = 'LOW'
        
        # 檢查穩定性
        if not volatility_metrics['is_stable']:
            recommendations.append("決策不穩定，建議調整fuzzy控制器參數")
            priority = 'HIGH'
        
        # 檢查震盪情況
        if hasattr(self, 'current_oscillation_start') and self.current_oscillation_start:
            recommendations.append("檢測到持續震盪，建議啟用自動關機或調整決策閾值")
            priority = 'HIGH'
        
        # 檢查一致性
        if consistency_metrics['consistency_score'] < self.consistency_threshold:
            recommendations.append("預測與實際功率一致性較低，建議改進預測模型")
            if priority == 'LOW':
                priority = 'MEDIUM'
        
        # 檢查錯誤率
        if error_metrics['total_error_rate'] > 0.2:
            recommendations.append("錯誤率較高，建議重新訓練決策模型")
            priority = 'HIGH'
        
        # 檢查延遲判斷
        if error_metrics['delayed_decision_rate'] > 0.3:
            recommendations.append("延遲判斷比例過高，建議增強決策信心度")
            if priority == 'LOW':
                priority = 'MEDIUM'
        
        # 檢查震盪事件數量
        if len(self.oscillation_events) > 3:
            recommendations.append("頻繁震盪事件，建議檢查決策邏輯或添加遲滯機制")
            priority = 'HIGH'
        
        if not recommendations:
            recommendations.append("系統運行良好，無需特別調整")
        
        return {
            'recommendations': recommendations,
            'priority': priority,
            'overall_performance': 'EXCELLENT' if overall_score['overall_score'] > 0.8 else
                                 'GOOD' if overall_score['overall_score'] > 0.6 else
                                 'FAIR' if overall_score['overall_score'] > 0.4 else 'POOR'
        }

    def get_oscillation_summary(self):
        """獲取震盪檢測摘要"""
        if not self.oscillation_detection_enabled:
            return {"message": "震盪檢測功能未啟用"}
        
        total_events = len(self.oscillation_events)
        total_shutdown_events = len(self.auto_shutdown_events)
        
        # 計算平均震盪強度
        avg_intensity = 0
        if self.oscillation_events:
            avg_intensity = np.mean([event['intensity'] for event in self.oscillation_events])
        
        # 計算總震盪時間
        total_oscillation_minutes = 0
        for event in self.oscillation_events:
            if 'duration_minutes' in event:
                total_oscillation_minutes += event['duration_minutes']
        
        return {
            'total_oscillation_events': total_events,
            'total_shutdown_events': total_shutdown_events,
            'average_oscillation_intensity': avg_intensity,
            'total_oscillation_minutes': total_oscillation_minutes,
            'current_oscillating': self.current_oscillation_start is not None,
            'oscillation_events': self.oscillation_events[-5:],  # 最近5個事件
            'shutdown_events': self.auto_shutdown_events[-3:]    # 最近3個關機事件
        }
    
    def get_evaluation_summary(self):
        """獲取評估摘要"""
        if not self.evaluation_results:
            return {"message": "尚無評估結果"}
        
        latest_result = self.evaluation_results[-1]
        
        # 計算歷史平均分數
        historical_scores = []
        for result in self.evaluation_results:
            historical_scores.append(result['overall_score']['overall_score'])
        
        avg_score = np.mean(historical_scores)
        score_trend = "穩定"
        
        if len(historical_scores) >= 3:
            recent_avg = np.mean(historical_scores[-3:])
            earlier_avg = np.mean(historical_scores[:-3]) if len(historical_scores) > 3 else avg_score
            
            if recent_avg > earlier_avg + 0.1:
                score_trend = "改善"
            elif recent_avg < earlier_avg - 0.1:
                score_trend = "下降"
        
        return {
            'total_evaluations': len(self.evaluation_results),
            'latest_overall_score': latest_result['overall_score']['overall_score'],
            'historical_average_score': avg_score,
            'score_trend': score_trend,
            'latest_performance': latest_result['recommendation']['overall_performance'],
            'high_priority_recommendations': len([r for r in self.evaluation_results 
                                                if r['recommendation']['priority'] == 'HIGH']),
            'system_stability': latest_result['volatility_metrics']['is_stable']
        }
    
    
    def reset_statistics(self):
        """重置統計信息（保留配置）"""
        self.oscillation_events.clear()
        self.auto_shutdown_events.clear()
        self.evaluation_results.clear()
        self.current_oscillation_start = None
        self.consecutive_phantom_time = None
        self.last_shutdown_time = None
        self.last_evaluation_time = None
        
        print("✅ 統計信息已重置")
    
    def update_thresholds(self, **kwargs):
        """動態更新閾值"""
        updated = []
        
        if 'volatility_threshold' in kwargs:
            self.volatility_threshold = kwargs['volatility_threshold']
            updated.append(f"波動性閾值: {self.volatility_threshold}")
        
        if 'consistency_threshold' in kwargs:
            self.consistency_threshold = kwargs['consistency_threshold']
            updated.append(f"一致性閾值: {self.consistency_threshold}")
        
        if 'oscillation_threshold_ratio' in kwargs:
            self.oscillation_threshold_ratio = kwargs['oscillation_threshold_ratio']
            updated.append(f"震盪比例閾值: {self.oscillation_threshold_ratio}")
        
        if 'min_oscillation_count' in kwargs:
            self.min_oscillation_count = kwargs['min_oscillation_count']
            updated.append(f"最小震盪次數: {self.min_oscillation_count}")
        
        if updated:
            print(f"✅ 已更新閾值: {'; '.join(updated)}")
        else:
            print("❌ 未提供有效的閾值參數")
    
    def get_current_status(self):
        """獲取當前系統狀態"""
        return {
            'timestamp': datetime.now(),
            'total_records': len(self.decision_history),
            'window_size_minutes': self.window_size.total_seconds() / 60,
            'evaluation_interval_minutes': self.evaluation_interval.total_seconds() / 60,
            'oscillation_detection_enabled': self.oscillation_detection_enabled,
            'auto_shutdown_enabled': self.auto_shutdown_enabled,
            'current_oscillating': self.current_oscillation_start is not None,
            'consecutive_phantom_active': self.consecutive_phantom_time is not None,
            'total_oscillation_events': len(self.oscillation_events),
            'total_shutdown_events': len(self.auto_shutdown_events),
            'total_evaluations': len(self.evaluation_results),
            'last_evaluation': self.last_evaluation_time.strftime('%H:%M:%S') if self.last_evaluation_time else 'N/A'
        }


# ========================================
# 使用示例和測試代碼
# ========================================

if __name__ == "__main__":
    # 創建增強版評估器實例
    evaluator = DecisionEvaluator(
        window_size_minutes=30,
        evaluation_interval_minutes=15,
        oscillation_detection_enabled=True,
        oscillation_window_minutes=10,
        auto_shutdown_enabled=True
    )
    
    # 模擬測試數據
    import random
    
    print("\n🧪 開始模擬測試...")
    base_time = datetime.now()
    
    # 模擬30分鐘的決策數據
    for i in range(60):  # 每30秒一個決策
        current_time = base_time + timedelta(seconds=i * 30)
        
        # 模擬不同的功率值和決策
        if i < 20:
            # 正常使用期間
            actual_power = random.uniform(50, 200)
            decision = 'continue_monitoring'
        elif i < 40:
            # 開始進入待機，可能震盪
            actual_power = random.uniform(15, 25)
            decision = random.choice(['suggest_shutdown', 'delay_decision', 'continue_monitoring'])
        else:
            # 穩定的phantom load期間
            actual_power = random.uniform(10, 18)
            decision = 'suggest_shutdown'
        
        fuzzy_output = min(1.0, actual_power / 100)
        predicted_power = actual_power + random.uniform(-5, 5)
        
        # 添加決策記錄
        result = evaluator.add_decision_record(
            timestamp=current_time,
            fuzzy_output=fuzzy_output,
            predicted_power=predicted_power,
            actual_power=actual_power,
            decision=decision
        )
        
        # 如果觸發了特殊事件，打印信息
        if result['oscillation_detected']:
            print(f"⚠️  震盪檢測: 強度 {result['oscillation_intensity']:.2f}")
        
        if result['auto_shutdown_recommended']:
            print(f"🔌 自動關機建議: {result['shutdown_reason']}")
    
    # 打印最終摘要
    print("\n📊 測試完成，生成摘要報告...")
    
    # 獲取震盪摘要
    osc_summary = evaluator.get_oscillation_summary()
    print(f"\n震盪檢測摘要: {osc_summary}")
    
    # 獲取評估摘要
    eval_summary = evaluator.get_evaluation_summary()
    print(f"\n評估摘要: {eval_summary}")
    
    # 獲取當前狀態
    status = evaluator.get_current_status()
    print(f"\n當前狀態: {status}")
    
    # 導出詳細報告
    # report_path = evaluator.export_detailed_report()
    
    print(f"\n✅ 增強版DecisionEvaluator測試完成！")