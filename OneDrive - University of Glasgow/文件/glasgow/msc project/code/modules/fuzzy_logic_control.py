import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from . import device_activity
from . import user_habit
from . import confidence

### 找三個model ###

try:
    from .device_activity import DeviceActivityScoreModule
    print("found device activity module")
    HAS_DEVICE_ACTIVITY = True
except ImportError:
    HAS_DEVICE_ACTIVITY = False
    print("⚠️  device_activity 模組未找到")


try:
    from .user_habit import NoShutdownUserHabitScoreModule
    HAS_USER_HABIT = True
    print("found user habit module")
except ImportError:
    HAS_USER_HABIT = False
    print("⚠️  user_habit 模組未找到")

try:
    from .confidence import ConfidenceScoreModule
    print("found confidence module")
    HAS_CONFIDENCE_SCORE = True
except ImportError:
    HAS_CONFIDENCE_SCORE = False
    print("⚠️  confidence_score 模組未找到")


class DecisionTreeSmartPowerAnalysis:

    def load_phantom_threshold():
        """從 config/thresholds.json 載入 phantom load threshold"""
        try:
            with open('config/thresholds.json', 'r') as f:
                data = json.load(f)
            
            phantom_threshold = data['thresholds']['phantom_upper']
            print(f"✅ 成功載入 Phantom Load Threshold: {phantom_threshold}W")
            return phantom_threshold
            
        except FileNotFoundError:
            print("❌ 找不到 config/thresholds.json，使用預設值 36.9W")
            print("💡 請先執行 K-means 分析生成 threshold 配置")
            return 36.9  # 預設值
        except KeyError as e:
            print(f"❌ config/thresholds.json 格式錯誤，缺少欄位: {e}")
            print("💡 請重新執行 K-means 分析")
            return 36.9  # 預設值
        except Exception as e:
            print(f"❌ 讀取 threshold 時發生錯誤: {e}")
            return 36.9  # 預設值
        
    def __init__(self):
        self.data_file = 'C:/Users/王俞文/OneDrive - University of Glasgow/文件/glasgow/msc project/data/extended_power_data_2months.csv'
        
        print("start decision tree smart power analysis...")
        
        # 電費設定
        self.uk_electricity_rate = 0.30  # £0.30/kWh
        
        # 初始化並訓練模型
        self.device_activity_model = None
        self.user_habit_model = None
        self.confidence_model = None
        
        # 決策統計
        self.decision_stats = {
            'total_decisions': 0,
            'decision_paths': {},  # 記錄每種決策路徑
            'level_combinations': {}  # 記錄每種等級組合
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
            'total_opportunities': 0
        }

    def _generate_phantom_load_opportunities(self, df):

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # df['is_phantom'] = df['power'] < 92
        # df['is_phantom'] = df['power'] < 60
        phantom_load_threshold = self.load_phantom_threshold()
        df['is_phantom'] = df['power'] < phantom_load_threshold
        print(f'phantom load (< 60W) : {len(df[df["is_phantom"]])} counts')

        opportunities = []
        in_session = False   # 判斷是否在phantom load
        start_time = None
        records = []

        for i, row in df.iterrows():
            if row['is_phantom']:
                if not in_session:
                    in_session = True
                    start_time = row['timestamp']
                    records = []
                records.append(row)
            else:
                if in_session:
                    end_time = row['timestamp']
                    power_list = [r['power'] for r in records if r['power'] > 0]
                    avg_power = np.mean(power_list) if power_list else 75
                    opportunities.append({
                        'device_id': 'phantom_device',
                        'start_time': start_time,
                        'end_time': end_time,
                        'power_watt': avg_power
                    })
                    in_session = False

        if in_session:
            end_time = df['timestamp'].iloc[-1]
            power_list = [r['power'] for r in records if r['power'] > 0]
            avg_power = np.mean(power_list) if power_list else 75
            opportunities.append({
                'device_id': 'phantom_device',
                'start_time': start_time,
                'end_time': end_time,
                'power_watt': avg_power
            })

        return opportunities

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
        confidence_score = to_level(confidence_score)
        
        # 記錄等級組合統計
        combination = f"{user_habit}-{device_activity}-{confidence_score}"
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
                if confidence_score == "low":
                    decision_path.append("confidence score=low")
                    decision = "suggest_shutdown"  # 很少用+長時間待機+不確定時段 -> 關機
                elif confidence_score == "medium":
                    decision_path.append("confidence score=medium")
                    decision = "suggest_shutdown"  # 很少用+長時間待機+中等確定 -> 關機
                elif confidence_score == "high":
                    decision_path.append("confidence score=high")
                    decision = "delay_decision"  # 很少用+長時間待機+確定時段，可能特殊情況 -> 等待
                    
            elif device_activity == "medium":  # 中等活躍度
                decision_path.append("device activity=medium")
                if confidence_score == "low":
                    decision_path.append("confidence score=low")
                    decision = "delay_decision"  # 很少用但有些活躍+不確定 -> 等待
                elif confidence_score == "medium":
                    decision_path.append("confidence score=medium")
                    decision = "send_notification"  # 很少用但有些活躍+中等確定 -> 通知
                elif confidence_score == "high":
                    decision_path.append("confidence score=high")
                    decision = "send_notification"  # 很少用但有些活躍+確定時段 -> 通知
                    
            elif device_activity == "high":  # 最近很活躍
                decision_path.append("device activity=high")
                if confidence_score == "low":
                    decision_path.append("confidence score=low")
                    decision = "keep_on"  # 很少用但剛剛活躍+不確定 -> 保持
                elif confidence_score == "medium":
                    decision_path.append("confidence score=medium")
                    decision = "keep_on"  # 很少用但剛剛活躍+中等確定 -> 保持
                elif confidence_score == "high":
                    decision_path.append("confidence score=high")
                    decision = "keep_on"  # 很少用但剛剛活躍+確定 -> 保持
                    
        elif user_habit == "medium":  # 中等使用頻率
            decision_path.append("user habit=medium")
            
            if device_activity == "low":  # 長時間待機
                decision_path.append("device activity=low")
                if confidence_score == "low":
                    decision_path.append("confidence score=low")
                    decision = "suggest_shutdown"  # 中等使用+長時間待機+不確定 -> 關機
                elif confidence_score == "medium":
                    decision_path.append("confidence score=medium")
                    decision = "suggest_shutdown"  # 中等使用+長時間待機+中等確定 -> 關機
                elif confidence_score == "high":
                    decision_path.append("confidence score=high")
                    decision = "send_notification"  # 中等使用+長時間待機+確定時段 -> 通知
                    
            elif device_activity == "medium":  # 中等活躍度
                decision_path.append("device activity=medium")
                if confidence_score == "low":
                    decision_path.append("confidence score=low")
                    decision = "delay_decision"  # 中等使用+中等活躍+不確定 -> 等待
                elif confidence_score == "medium":
                    decision_path.append("confidence score=medium")
                    decision = "send_notification"  # 中等使用+中等活躍+中等確定 -> 通知
                elif confidence_score == "high":
                    decision_path.append("confidence score=high")
                    decision = "keep_on"  # 中等使用+中等活躍+確定peak hour -> 保持
                    
            elif device_activity == "high":  # 最近很活躍
                decision_path.append("device activity=high")
                if confidence_score == "low":
                    decision_path.append("confidence score=low")
                    decision = "delay_decision"  # 中等使用+剛剛活躍+不確定 -> 等待
                elif confidence_score == "medium":
                    decision_path.append("confidence score=medium")
                    decision = "keep_on"  # 中等使用+剛剛活躍+中等確定 -> 保持
                elif confidence_score == "high":
                    decision_path.append("confidence score=high")
                    decision = "keep_on"  # 中等使用+剛剛活躍+確定 -> 保持
                    
        elif user_habit == "high":  # 經常使用設備
            decision_path.append("user habit=high")
            
            if device_activity == "low":  # 長時間待機
                decision_path.append("device activity=low")
                if confidence_score == "low":
                    decision_path.append("confidence score=low")
                    decision = "suggest_shutdown"  # 經常使用但長時間待機+不確定 -> 可能睡覺，關機
                elif confidence_score == "medium":
                    decision_path.append("confidence score=medium")
                    decision = "delay_decision"  # 經常使用但長時間待機+中等確定 -> 等待
                elif confidence_score == "high":
                    decision_path.append("confidence score=high")
                    decision = "delay_decision"  # 經常使用但長時間待機+確定睡眠 -> 等待再決定
                    
            elif device_activity == "medium":  # 中等活躍度
                decision_path.append("device activity=medium")
                if confidence_score == "low":
                    decision_path.append("confidence score=low")
                    decision = "suggest_shutdown"  # 經常使用+中等活躍+不確定 -> 異常情況，關機
                elif confidence_score == "medium":
                    decision_path.append("confidence score=medium")
                    decision = "send_notification"  # 經常使用+中等活躍+中等確定 -> 通知
                elif confidence_score == "high":
                    decision_path.append("confidence score=high")
                    decision = "keep_on"  # 經常使用+中等活躍+確定peak hour -> 保持
                    
            elif device_activity == "high":  # 最近很活躍
                decision_path.append("device activity=high")
                if confidence_score == "low":
                    decision_path.append("confidence score=low")
                    decision = "delay_decision"  # 經常使用+剛剛活躍+不確定 -> 等待
                elif confidence_score == "medium":
                    decision_path.append("confidence score=medium")
                    decision = "send_notification"  # 經常使用+剛剛活躍+中等確定 -> 通知確認
                elif confidence_score == "high":
                    decision_path.append("confidence score=high")
                    decision = "keep_on"  # 經常使用+剛剛活躍+確定 -> 保持
        
        # 記錄決策路徑統計
        path_key = " -> ".join(decision_path) + f" => {decision}"
        if path_key not in self.decision_stats['decision_paths']:
            self.decision_stats['decision_paths'][path_key] = 0
        self.decision_stats['decision_paths'][path_key] += 1
        
        self.decision_stats['total_decisions'] += 1
        
        # 添加調試信息
        # print(f"   決策詳情: {user_habit}-{device_activity}-{confidence_score} => {decision}")
        
        # 創建詳細的debug信息
        debug_info = {
            'user_habit_level': user_habit,
            'device_activity_level': device_activity,
            'confidence_score_level': confidence_score,
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
        print("fallback activity score")
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
        
        # print(f'fallback activity: {final_score:.2f}')
        return final_score

    def _fallback_habit_score(self, features, timestamp):
        """改進的fallback習慣分數 - 確保多樣化分布"""
        print("fallback habit score")
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
        
        # print(f'fallback habit: {final_score:.2f}')
        return final_score

    def _fallback_confidence_score(self, features, timestamp):
        """改進的fallback置信度分數 - 確保多樣化分布"""
        print("fallback confidence score")
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
        
        # print(f'fallback confidence: {final_score:.2f}')
        return final_score

    def _extract_enhanced_features(self, opportunity, df):
        return {
            'device_id': opportunity.get('device_id', 'unknown'),
            'duration_minutes': (opportunity['end_time'] - opportunity['start_time']).total_seconds() / 60,
            'hour_of_day': opportunity['start_time'].hour,
            'power_watt': opportunity.get('power_watt', 75),
            'weekday': opportunity['start_time'].weekday()
        }

    def _apply_decision_tree_models(self, opportunities, df):
        print("\n🌳 使用決策樹方法進行決策分析...")
        decision_results = []
        debug_logs = []

        for i, opp in enumerate(opportunities):
            try:
                features = self._extract_enhanced_features(opp, df)
                timestamp = opp['start_time']

                # 使用訓練好的模型或fallback
                if self.device_activity_model:
                    try:
                        activity_result = self.device_activity_model.calculate_activity_score(timestamp)
                        activity_score = activity_result['activity_score']
                    except Exception as e:
                        activity_score = self._fallback_activity_score(features, timestamp)
                else:
                    activity_score = self._fallback_activity_score(features, timestamp)

                if self.user_habit_model:
                    try:
                        habit_result = self.user_habit_model.calculate_habit_score(timestamp)
                        habit_score = habit_result['habit_score']
                    except Exception as e:
                        habit_score = self._fallback_habit_score(features, timestamp)
                else:
                    habit_score = self._fallback_habit_score(features, timestamp)

                if self.confidence_model:
                    try:
                        confidence_result = self.confidence_model.calculate_confidence_score(timestamp)
                        confidence_score = confidence_result['confidence_score']
                    except Exception as e:
                        confidence_score = self._fallback_confidence_score(features, timestamp)
                else:
                    confidence_score = self._fallback_confidence_score(features, timestamp)

                # 🌳 使用決策樹方法
                decision, debug_info = self._make_intelligent_decision(
                    activity_score, habit_score, confidence_score, features
                )

                if decision in self.results:
                    self.results[decision] += 1
                else:
                    print(f"   ⚠️ Unknown decision result: {decision}")
                    self.results['delay_decision'] += 1

                result = {
                    'opportunity': opp,
                    'features': features,
                    'activity_score': activity_score,
                    'user_habit_score': habit_score,
                    'confidence_score': confidence_score,
                    'decision': decision,
                    'debug_info': debug_info
                }
                decision_results.append(result)
                
                # 記錄前10個的詳細debug資訊
                if i < 10:
                    debug_logs.append({
                        'index': i+1,
                        'time': timestamp,
                        'power': features['power_watt'],
                        'duration': features['duration_minutes'],
                        'scores': [activity_score, habit_score, confidence_score],
                        'levels': [debug_info['device_activity_level'], 
                                  debug_info['user_habit_level'], 
                                  debug_info['confidence_score_level']],
                        'decision_path': debug_info['decision_path'],
                        'decision': decision
                    })

            except Exception as e:
                print(f"   ⚠️ Error processing opportunity {i+1}: {e}")
                self.results['delay_decision'] += 1

        # 打印決策樹統計
        self._print_decision_tree_stats()

        # 打印前幾個決策的詳細路徑
        print(f"\n🔍 決策樹分析 (前5個樣本):")
        for log in debug_logs[:5]:
            scores_str = f"{log['scores'][0]:.2f}/{log['scores'][1]:.2f}/{log['scores'][2]:.2f}"
            levels_str = f"{log['levels'][0]}/{log['levels'][1]}/{log['levels'][2]}"
            path_str = " -> ".join(log['decision_path'])
            print(f"   #{log['index']}: {log['time'].strftime('%H:%M')} | "
                  f"Power: {log['power']:.0f}W | Duration: {log['duration']:.0f}min")
            print(f"      Scores: {scores_str} | Levels: {levels_str}")
            print(f"      Path: {path_str} => {log['decision']}")
            print()

        return decision_results

    def _print_decision_tree_stats(self):
        """打印決策樹統計信息"""
        print(f"\n🌳 決策樹統計分析:")
        print(f"   總決策次數: {self.decision_stats['total_decisions']}")
        
        # 打印決策分布
        total_decisions = sum(self.results.values()) - self.results['phantom_load_detected'] - self.results['total_opportunities']
        print(f"\n📊 決策分布:")
        for decision in ['suggest_shutdown', 'send_notification', 'delay_decision', 'keep_on']:
            count = self.results[decision]
            percentage = (count / total_decisions * 100) if total_decisions > 0 else 0
            if count > 0:
                print(f"   {decision}: {count} 次 ({percentage:.1f}%)")
        
        # 打印等級組合統計
        print(f"\n🎯 等級組合分布 (用戶習慣-設備活動-置信度):")
        sorted_combinations = sorted(self.decision_stats['level_combinations'].items(), 
                                   key=lambda x: x[1], reverse=True)
        for combination, count in sorted_combinations[:10]:  # 顯示前10個
            percentage = (count / self.decision_stats['total_decisions'] * 100)
            print(f"   {combination}: {count} 次 ({percentage:.1f}%)")
        
        # 打印最常見的決策路徑
        print(f"\n🛤️ 最常見決策路徑 (前5個):")
        sorted_paths = sorted(self.decision_stats['decision_paths'].items(), 
                            key=lambda x: x[1], reverse=True)
        for path, count in sorted_paths[:5]:
            percentage = (count / self.decision_stats['total_decisions'] * 100)
            print(f"   {count} 次 ({percentage:.1f}%): {path}")

    def _calculate_data_period_info(self, df):
        """計算數據時間範圍資訊"""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        start_date = df['timestamp'].min()
        end_date = df['timestamp'].max()
        total_days = (end_date - start_date).days + 1
        
        return {
            'start_date': start_date,
            'end_date': end_date,
            'total_days': total_days
        }

    def _estimate_energy_saving(self, decision_results, df):
        """計算詳細的節能效果並視覺化（含英國電費計算）"""
        # 計算數據期間資訊
        period_info = self._calculate_data_period_info(df)
        total_days = period_info['total_days']
        
        total_baseline_kwh = 0
        notification_kwh = 0

        decision_breakdown = {
            'suggest_shutdown': {'count': 0, 'kwh': 0},
            'send_notification': {'count': 0, 'kwh': 0},
            'keep_on': {'count': 0, 'kwh': 0},
            'delay_decision': {'count': 0, 'kwh': 0}
        }

        # 計算各決策的能耗
        for result in decision_results:
            opp = result['opportunity']
            decision = result['decision']

            duration_hr = (opp['end_time'] - opp['start_time']).total_seconds() / 3600
            power_watt = opp.get('power_watt', 100)
            energy_kwh = power_watt * duration_hr / 1000

            total_baseline_kwh += energy_kwh

            if decision in decision_breakdown:
                decision_breakdown[decision]['count'] += 1
                decision_breakdown[decision]['kwh'] += energy_kwh

            if decision == 'send_notification':
                notification_kwh += energy_kwh

        # 轉換為日平均值
        daily_baseline_kwh = total_baseline_kwh / total_days
        daily_notification_kwh = notification_kwh / total_days
        
        # 計算不同 send notification 響應率的節能效果
        notification_count = decision_breakdown['send_notification']['count']
        
        # 用戶響應場景
        user_response_scenarios = {
            '用戶100%同意關機': 1.0,
            '用戶80%同意關機': 0.8,
            '用戶60%同意關機': 0.6,
            '用戶40%同意關機': 0.4,
            '用戶20%同意關機': 0.2,
            '用戶0%同意關機': 0.0
        }

        print(f"\n💡 決策樹版詳細節能分析（基於 {total_days} 天數據）：")
        print(f"   🔋 系統原始預估耗電量：{total_baseline_kwh:.2f} kWh (日均: {daily_baseline_kwh:.2f} kWh)")
        print(f"   💰 系統原始預估電費：£{total_baseline_kwh * self.uk_electricity_rate:.3f} (日均: £{daily_baseline_kwh * self.uk_electricity_rate:.3f})")
        
        print(f"\n📊 決策分類統計：")
        for decision, data in decision_breakdown.items():
            if data['count'] > 0:
                percentage = (data['kwh'] / total_baseline_kwh * 100)
                daily_kwh = data['kwh'] / total_days
                daily_cost = daily_kwh * self.uk_electricity_rate
                print(f"   📌 {decision}: {data['count']} 次, {data['kwh']:.2f} kWh ({percentage:.1f}%) | 日均: {daily_kwh:.3f} kWh (£{daily_cost:.3f})")

        # 固定節能（suggest_shutdown）
        fixed_saving_kwh = decision_breakdown['suggest_shutdown']['kwh']
        daily_fixed_saving_kwh = fixed_saving_kwh / total_days
        
        print(f"\n✅ 確定節能效果（suggest_shutdown）：")
        print(f"   💡 確定節省電量：{fixed_saving_kwh:.2f} kWh (日均: {daily_fixed_saving_kwh:.2f} kWh)")
        print(f"   💰 確定節省電費：£{fixed_saving_kwh * self.uk_electricity_rate:.3f} (日均: £{daily_fixed_saving_kwh * self.uk_electricity_rate:.3f})")

        # Send notification 情況分析
        notification_scenarios = {}
        if notification_count > 0:
            print(f"\n🔔 Send Notification 情況分析：")
            print(f"   📬 總通知次數：{notification_count} 次")
            print(f"   ⚡ 涉及電量：{notification_kwh:.2f} kWh (日均: {daily_notification_kwh:.2f} kWh)")
            print(f"   💰 涉及電費：£{notification_kwh * self.uk_electricity_rate:.3f} (日均: £{daily_notification_kwh * self.uk_electricity_rate:.3f})")
            print(f"\n   📈 不同用戶響應率的總節能效果：")
            
            for scenario, response_rate in user_response_scenarios.items():
                notification_saving = notification_kwh * response_rate
                total_scenario_saving = fixed_saving_kwh + notification_saving
                remaining_consumption = total_baseline_kwh - total_scenario_saving
                savings_percentage = (total_scenario_saving / total_baseline_kwh * 100)
                
                # 日平均和年度預估
                daily_total_saving = total_scenario_saving / total_days
                annual_saving_kwh = daily_total_saving * 365
                annual_saving_cost = annual_saving_kwh * self.uk_electricity_rate
                
                notification_scenarios[scenario] = {
                    'response_rate': response_rate,
                    'notification_saved_kwh': notification_saving,
                    'total_saved_kwh': total_scenario_saving,
                    'remaining_kwh': remaining_consumption,
                    'savings_percentage': savings_percentage,
                    'daily_saved_kwh': daily_total_saving,
                    'annual_saved_kwh': annual_saving_kwh,
                    'annual_saved_cost': annual_saving_cost
                }
                
                print(f"     🎯 {scenario}:")
                print(f"        節省: {total_scenario_saving:.2f} kWh (節能率: {savings_percentage:.1f}%)")
                print(f"        日均節省: {daily_total_saving:.3f} kWh | 年度節省: {annual_saving_kwh:.0f} kWh (£{annual_saving_cost:.0f})")
                print(f"        剩餘耗電: {remaining_consumption:.2f} kWh")
        else:
            print(f"\n🔔 本次分析無 Send Notification 決策")

        actual_scenarios = notification_scenarios['用戶100%同意關機']  # 或你希望的響應率

        total_saving_kwh = fixed_saving_kwh + actual_scenarios['notification_saved_kwh']
        after_system_kwh = total_baseline_kwh - total_saving_kwh
        saving_percent = total_saving_kwh / total_baseline_kwh * 100

        print(f"\n=== 💡 加入系統前後的能耗對比 ===")
        print(f"⚡ 原始總耗電量: {total_baseline_kwh:.2f} kWh")
        print(f"✅ 加入系統後預估耗電量: {after_system_kwh:.2f} kWh")
        print(f"💸 節省電量: {total_saving_kwh:.2f} kWh")
        print(f"📉 節能比例: {saving_percent:.1f}%")

        # 假設你用的是 100% 用戶回應的情境
        response_scenario = notification_scenarios.get('用戶100%同意關機')
        if response_scenario:
            daily_total_saving = response_scenario['daily_saved_kwh']
            daily_after_system_kwh = daily_baseline_kwh - daily_total_saving
            saving_percent = daily_total_saving / daily_baseline_kwh * 100

            print(f"\n=== 📉 每日能耗對比分析 ===")
            print(f"🔋 每日原始耗能: {daily_baseline_kwh:.3f} kWh")
            print(f"✅ 每日系統介入後預估耗能: {daily_after_system_kwh:.3f} kWh")
            print(f"💡 每日節省: {daily_total_saving:.3f} kWh")
            print(f"📈 每日節能率: {saving_percent:.1f}%")



        # 生成視覺化圖表
        self._create_energy_saving_visualization(
            decision_breakdown, 
            notification_scenarios, 
            total_baseline_kwh,
            fixed_saving_kwh,
            notification_kwh,
            total_days
        )

        # 打印最終報告
        self._print_final_phantom_load_report(
            total_baseline_kwh, 
            fixed_saving_kwh, 
            notification_kwh, 
            notification_scenarios,
            total_days
        )

        return {
            'baseline_kwh': total_baseline_kwh,
            'fixed_saved_kwh': fixed_saving_kwh,
            'notification_kwh': notification_kwh,
            'decision_breakdown': decision_breakdown,
            'notification_scenarios': notification_scenarios,
            'total_days': total_days
        }

    def _create_energy_saving_visualization(self, decision_breakdown, notification_scenarios, 
                                          total_baseline_kwh, fixed_saving_kwh, notification_kwh, total_days):
        """創建詳細的節能視覺化分析（兩張圖兩張圖顯示）"""

        # 第一組圖：決策分布和能耗分析
        plt.style.use('default')
        fig1 = plt.figure(figsize=(20, 10))
        fig1.suptitle('Decision Tree Intelligent Power Management - Decision Analysis', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        colors = {
            'suggest_shutdown': '#FF6B6B',
            'send_notification': '#4ECDC4', 
            'delay_decision': '#45B7D1',
            'keep_on': '#96CEB4',
            'baseline': '#FFE66D',
            'saved': '#66D9EF',
            'remaining': '#F8F8F2'
        }
        
        decision_labels = {
            'suggest_shutdown': 'Shutdown',
            'send_notification': 'Send Notification', 
            'delay_decision': 'Delay Decision',
            'keep_on': 'Maintain State'
        }
        
        # 1. 決策分布圓餅圖
        ax1 = fig1.add_subplot(1, 2, 1)
        decisions = []
        counts = []
        decision_colors = []
        
        for k in ['suggest_shutdown', 'send_notification', 'delay_decision', 'keep_on']:
            if decision_breakdown[k]['count'] > 0:
                decisions.append(decision_labels[k])
                counts.append(decision_breakdown[k]['count'])
                decision_colors.append(colors[k])
        
        if len(decisions) > 0:
            wedges, texts, autotexts = ax1.pie(counts, labels=decisions, colors=decision_colors, 
                                              autopct='%1.1f%%', startangle=90,     
                                              textprops={'fontsize': 12})
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        ax1.set_title('Daily Decision Distribution', fontweight='bold', fontsize=14, pad=20)
        
        # 2. 能耗分布柱狀圖
        ax2 = fig1.add_subplot(1, 2, 2)
        decision_names = []
        kwh_values = []
        bar_colors = []
        
        for k in ['suggest_shutdown', 'send_notification', 'delay_decision', 'keep_on']:
            if decision_breakdown[k]['kwh'] > 0:
                decision_names.append(decision_labels[k])
                kwh_values.append(decision_breakdown[k]['kwh'] / total_days)  # 轉為日平均
                bar_colors.append(colors[k])
        
        if len(decision_names) > 0:
            x_pos = np.arange(len(decision_names))
            bars = ax2.bar(x_pos, kwh_values, color=bar_colors, alpha=0.9, 
                          edgecolor='white', linewidth=2, width=0.6)
            
            ax2.set_xlabel('Decision Type', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Daily Average Power Consumption (kWh/day)', fontsize=12, fontweight='bold')
            ax2.set_title('Daily Average Energy Consumption Distribution by Decision Type', fontweight='bold', fontsize=14, pad=20)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(decision_names, rotation=0, ha='center', fontsize=11)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # 添加數值標籤
            for bar, value in zip(bars, kwh_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(kwh_values)*0.02,
                        f'{value:.3f}\nkWh/day', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.show()

        # 第二組圖：Phantom Load能源和電費分析
        fig2 = plt.figure(figsize=(20, 10))
        fig2.suptitle('Daily Phantom Load Energy & Cost Distribution', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        # 1. 能源分配圓環圖
        ax3 = fig2.add_subplot(1, 2, 1)
        
        daily_baseline = total_baseline_kwh / total_days
        daily_fixed_saving = fixed_saving_kwh / total_days
        daily_notification = notification_kwh / total_days
        daily_remaining = daily_baseline - daily_fixed_saving - daily_notification
        
        energy_categories = []
        energy_values = []
        energy_colors = []
        
        if daily_fixed_saving > 0:
            energy_categories.append(f'Determine Savings\n({daily_fixed_saving:.3f} kWh/day, £{daily_fixed_saving * self.uk_electricity_rate:.3f}/day)')
            energy_values.append(daily_fixed_saving)
            energy_colors.append(colors['suggest_shutdown'])
        
        if daily_notification > 0:
            energy_categories.append(f'Possible Savings\n({daily_notification:.3f} kWh/day, £{daily_notification * self.uk_electricity_rate:.3f}/day)')
            energy_values.append(daily_notification)
            energy_colors.append(colors['send_notification'])
        
        if daily_remaining > 0:
            energy_categories.append(f'Remain Using\n({daily_remaining:.3f} kWh/day, £{daily_remaining * self.uk_electricity_rate:.3f}/day)')
            energy_values.append(daily_remaining)
            energy_colors.append(colors['keep_on'])
        
        if len(energy_categories) > 0:
            wedges, texts, autotexts = ax3.pie(energy_values, labels=energy_categories, 
                                              colors=energy_colors, autopct='%1.0f%%', 
                                              startangle=90, textprops={'fontsize': 10})
            
            # 創建圓環效果
            centre_circle = plt.Circle((0,0), 0.4, fc='white')
            ax3.add_artist(centre_circle)
            
            # 在中心添加總電量
            ax3.text(0, 0, f'Total Phantom Load\n{daily_baseline:.3f} kWh/day\n£{daily_baseline * self.uk_electricity_rate:.3f}/day', 
                    ha='center', va='center', fontsize=11, fontweight='bold')
        
        ax3.set_title('Daily Phantom Load Energy Distribution', fontweight='bold', fontsize=14, pad=20)
        
        # 2. Send Notification 用戶響應率影響分析
        ax4 = fig2.add_subplot(1, 2, 2)
        if notification_scenarios and len(notification_scenarios) > 0:
            scenarios = list(notification_scenarios.keys())
            scenario_labels = []
            for s in scenarios:
                rate = notification_scenarios[s]['response_rate']
                scenario_labels.append(f'{int(rate*100)}% Agree')
            
            daily_total_saved = [notification_scenarios[s]['daily_saved_kwh'] for s in scenarios]
            daily_cost_saved = [s * self.uk_electricity_rate for s in daily_total_saved]
            savings_percentage = [notification_scenarios[s]['savings_percentage'] for s in scenarios]
            
            x = np.arange(len(scenarios))
            width = 0.25
            
            # 三個柱狀圖：日節省電量、日節省電費、節能百分比
            bars1 = ax4.bar(x - width, daily_total_saved, width, label='Power Saving (kWh/day)', 
                           color=colors['saved'], alpha=0.8, edgecolor='white', linewidth=2)
            bars2 = ax4.bar(x, daily_cost_saved, width, label='Cost Saving (£/day)', 
                           color=colors['send_notification'], alpha=0.8, edgecolor='white', linewidth=2)
            
            # 右側Y軸顯示百分比
            ax4_twin = ax4.twinx()
            bars3 = ax4_twin.bar(x + width, savings_percentage, width, label='Energy Saving Rate (%)', 
                                color='orange', alpha=0.8, edgecolor='white', linewidth=2)
            
            ax4.set_xlabel('User Response Rate', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Daily Savings', fontsize=12, fontweight='bold')
            ax4_twin.set_ylabel('Energy Saving Rate (%)', color='orange', fontsize=12, fontweight='bold')
            ax4.set_title('Send Notification: Energy & Cost Saving Effects with Different User Response Rates', 
                         fontweight='bold', fontsize=14, pad=20)
            ax4.set_xticks(x)
            ax4.set_xticklabels(scenario_labels, fontsize=10)
            ax4.grid(True, alpha=0.3, axis='y')
            
            # 數值標籤
            for i, (bar, value) in enumerate(zip(bars1, daily_total_saved)):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + max(daily_total_saved)*0.02,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9, 
                        color=colors['saved'], fontweight='bold')
                             
            for i, (bar, value) in enumerate(zip(bars2, daily_cost_saved)):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + max(daily_cost_saved)*0.02,
                        f'£{value:.3f}', ha='center', va='bottom', fontsize=9, 
                        color=colors['send_notification'], fontweight='bold')
                        
            for i, (bar, value) in enumerate(zip(bars3, savings_percentage)):
                height = bar.get_height()
                ax4_twin.text(bar.get_x() + bar.get_width()/2., height + max(savings_percentage)*0.02,
                             f'{value:.1f}%', ha='center', va='bottom', fontsize=9, 
                             color='orange', fontweight='bold')
            
            ax4.legend(loc='upper left', fontsize=10)
            ax4_twin.legend(loc='upper right', fontsize=10)
        else:
            # 如果沒有notification，顯示說明
            ax4.text(0.5, 0.5, '本次分析無發送通知決策\n所有節能效果為確定值', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=14,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
            ax4.set_title('🔔 發送通知分析', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        plt.show()

    def _print_final_phantom_load_report(self, total_baseline_kwh, fixed_saving_kwh, 
                                       notification_kwh, notification_scenarios, total_days):
        """打印最終的Phantom Load節能報告（含英國電費）"""
        
        # 計算日平均和年度數據
        daily_baseline = total_baseline_kwh / total_days
        daily_fixed_saving = fixed_saving_kwh / total_days
        daily_notification = notification_kwh / total_days
        
        annual_baseline = daily_baseline * 365
        annual_fixed_saving = daily_fixed_saving * 365
        
        daily_baseline_cost = daily_baseline * self.uk_electricity_rate
        daily_fixed_saving_cost = daily_fixed_saving * self.uk_electricity_rate
        annual_baseline_cost = annual_baseline * self.uk_electricity_rate
        annual_fixed_saving_cost = annual_fixed_saving * self.uk_electricity_rate
        
        # 基於數據推算的年度總電費（假設Phantom Load佔總用電的某個比例）
        estimated_total_annual_cost = annual_baseline_cost / 0.23  # 假設phantom load佔總用電23%
        
        print("\n" + "="*100)
        print("🎉 決策樹版智能電源管理 - 最終Phantom Load節能報告（含英國電費 £0.30/kWh）")
        print("📅 (基於 {:.1f} 天數據的日平均分析)".format(total_days))
        print("="*100)
        
        print(f"📊 系統分析結果摘要（僅針對Phantom Load部分）：")
        print(f"   🔋 日平均phantom load耗電量: {daily_baseline:.2f} kWh")
        print(f"   💰 日平均phantom load電費: £{daily_baseline_cost:.3f}")
        print(f"   📅 年度phantom load電費: £{annual_baseline_cost:.0f}")
        
        print(f"\n✅ 確定節能效果（suggest_shutdown）：")
        print(f"   💡 確定節省電量: {daily_fixed_saving:.2f} kWh/日 ({annual_fixed_saving:.0f} kWh/年)")
        print(f"   💰 確定節省電費: £{daily_fixed_saving_cost:.3f}/日 (£{annual_fixed_saving_cost:.0f}/年)")
        phantom_saving_rate = (daily_fixed_saving / daily_baseline * 100) if daily_baseline > 0 else 0
        print(f"   📈 Phantom load節能率: {phantom_saving_rate:.1f}%")
        
        # 相對於不同家庭年度電費的節省百分比
        print(f"\n📊 相對於年度總電費的節省百分比：")
        household_types = {
            '🏠 中型家庭': 1050,
            '🇬🇧 英國平均': 1200, 
            '🏢 大型家庭': 1500,
            f'📊 基於您的數據推算': estimated_total_annual_cost
        }
        
        for household_type, annual_cost in household_types.items():
            saving_percentage = (annual_fixed_saving_cost / annual_cost * 100)
            print(f"   {household_type} (£{annual_cost:.0f}/年): {saving_percentage:.1f}%")
        
        if notification_scenarios and len(notification_scenarios) > 0:
            print(f"\n🔔 Send Notification 潛在節能效果：")
            print(f"   📬 總通知次數: {len(notification_scenarios)} 種情境")
            print(f"   ⚡ 涉及電量：{daily_notification:.2f} kWh/日")
            print(f"   💰 涉及電費：£{daily_notification * self.uk_electricity_rate:.3f}/日")
            
            # 最佳和最差情況
            best_case = max(notification_scenarios.values(), key=lambda x: x['annual_saved_cost'])
            worst_case = min(notification_scenarios.values(), key=lambda x: x['annual_saved_cost'])
            
            print(f"\n🏆 最佳情況 (100%用戶同意):")
            print(f"   🎯 年度最大節省: £{best_case['annual_saved_cost']:.0f}")
            print(f"   📈 相對英國平均電費: {(best_case['annual_saved_cost']/1200*100):.1f}%")
            
            print(f"\n🔻 最差情況 (0%用戶同意):")
            print(f"   🎯 年度最小節省: £{worst_case['annual_saved_cost']:.0f}")
            print(f"   📈 相對英國平均電費: {(worst_case['annual_saved_cost']/1200*100):.1f}%")
            
            print(f"\n📊 潛在節能範圍:")
            range_cost = best_case['annual_saved_cost'] - worst_case['annual_saved_cost']
            print(f"   💰 年度電費範圍: £{worst_case['annual_saved_cost']:.0f} - £{best_case['annual_saved_cost']:.0f}")
            print(f"   📈 相對電費百分比範圍: {(worst_case['annual_saved_cost']/1200*100):.1f}% - {(best_case['annual_saved_cost']/1200*100):.1f}%")
            
            # 生活化效益比較
            best_annual_saving = best_case['annual_saved_cost']
            print(f"\n⚡ 生活化效益比較（基於最佳情況）：")
            print(f"   📅 相當於 {(best_annual_saving/(1200/365)):.0f} 天的免費電力")
            print(f"   📺 相當於 {(best_annual_saving/10.99):.1f} 個月的Netflix訂閱")
            print(f"   ☕ 相當於 {(best_annual_saving/3.5):.0f} 杯咖啡")
            
        else:
            print(f"\n🔔 本次分析無 Send Notification 決策")
            
            # 生活化效益比較
            print(f"\n⚡ 生活化效益比較（確定節省）：")
            print(f"   📅 相當於 {(annual_fixed_saving_cost/(1200/365)):.0f} 天的免費電力")
            print(f"   📺 相當於 {(annual_fixed_saving_cost/10.99):.1f} 個月的Netflix訂閱")
            print(f"   ☕ 相當於 {(annual_fixed_saving_cost/3.5):.0f} 杯咖啡")
        
        # 環境效益
        co2_factor = 0.233  # kg CO2 per kWh in UK
        annual_co2_saving = annual_fixed_saving * co2_factor
        cars_equivalent = annual_co2_saving / 4600  # 平均汽車年排放4.6噸CO2
        
        print(f"\n🌱 環境效益：")
        print(f"   🌍 每年減少 {annual_co2_saving:.0f} kg CO₂ 排放")
        print(f"   🚗 相當於減少 {cars_equivalent:.2f} 輛汽車一年的排放")
        
        print("="*100)

    def test(self, samples):
        print("\n🧪 測試決策樹模型決策系統...")
        for i, sample in enumerate(samples, 1):
            timestamp = sample["start_time"]
            
            # 獲取分數
            if self.device_activity_model:
                try:
                    activity_result = self.device_activity_model.calculate_activity_score(timestamp)
                    activity = activity_result['activity_score']
                except:
                    activity = self._fallback_activity_score({}, timestamp)
            else:
                activity = self._fallback_activity_score({}, timestamp)

            if self.user_habit_model:
                try:
                    habit_result = self.user_habit_model.calculate_habit_score(timestamp)
                    habit = habit_result['habit_score']
                except:
                    habit = self._fallback_habit_score({}, timestamp)
            else:
                habit = self._fallback_habit_score({}, timestamp)

            if self.confidence_model:
                try:
                    confidence_result = self.confidence_model.calculate_confidence_score(timestamp)
                    confidence = confidence_result['confidence_score']
                except:
                    confidence = self._fallback_confidence_score({}, timestamp)
            else:
                confidence = self._fallback_confidence_score({}, timestamp)

            features = {
                "device_id": "test_device",
                "duration_minutes": 60,
                "hour_of_day": timestamp.hour,
                "power_watt": sample.get("avg_power", 100),
                "weekday": timestamp.weekday()
            }

            decision, debug_info = self._make_intelligent_decision(activity, habit, confidence, features)

            print(f"--- 第 {i} 筆測試 ---")
            print(f"🕒 時間：{timestamp}")
            print(f"⚡ 功率：{sample.get('avg_power', 100)} W")
            print(f"📈 分數: Activity:{activity:.2f} Habit:{habit:.2f} Confidence:{confidence:.2f}")
            print(f"🎯 等級: {debug_info['device_activity_level']}-{debug_info['user_habit_level']}-{debug_info['confidence_score_level']}")
            print(f"🛤️ 決策路徑: {' -> '.join(debug_info['decision_path'])}")
            print(f"🧠 最終決策：{decision}")
            print()

    def run_analysis(self):
        """運行決策樹版完整分析"""
        print("\n" + "="*80)
        print("開始運行決策樹版智能電源管理分析")
        print("="*80)
        
        try:
            df = pd.read_csv(self.data_file)
            print(f"✅ 成功載入數據：{len(df)} 筆記錄")
        except Exception as e:
            print(f"❌ 無法讀取 CSV: {e}")
            return

        # 生成機會點
        opportunities = self._generate_phantom_load_opportunities(df)
        print(f"✅ 建立 {len(opportunities)} 筆機會點")

        # 應用決策樹決策
        decision_results = self._apply_decision_tree_models(opportunities, df)

        # 顯示詳細結果
        print("\n📋 前 5 筆決策結果詳情：")
        for i, result in enumerate(decision_results[:5], start=1):
            opp = result['opportunity']
            debug = result['debug_info']
            print(f"\n--- 第 {i} 筆 ---")
            print(f"🕒 時間：{opp['start_time'].strftime('%H:%M')} ~ {opp['end_time'].strftime('%H:%M')}")
            print(f"⚡ 平均功率：{opp['power_watt']:.1f} W")
            print(f"⏱️ 持續時間：{result['features']['duration_minutes']:.0f} 分鐘")
            print(f"📊 原始分數: A:{result['activity_score']:.2f} H:{result['user_habit_score']:.2f} C:{result['confidence_score']:.2f}")
            print(f"🎯 轉換等級: {debug['device_activity_level']}-{debug['user_habit_level']}-{debug['confidence_score_level']}")
            print(f"🛤️ 決策路徑: {' -> '.join(debug['decision_path'])}")
            print(f"🧠 最終決策：{result['decision']}")

        # 計算節能效果
        self._estimate_energy_saving(decision_results, df)

        # 運行測試
        test_samples = [
            {"avg_power": 150, "start_time": datetime(2024, 3, 26, 9, 0)},   # medium activity
            {"avg_power": 80,  "start_time": datetime(2024, 5, 26, 13, 0)},  # low power, work time
            {"avg_power": 50,  "start_time": datetime(2024, 7, 26, 20, 0)},  # very low power, evening
            {"avg_power": 30,  "start_time": datetime(2024, 9, 26, 2, 30)},  # very low power, night
            {"avg_power": 100, "start_time": datetime(2024, 11, 26, 18, 30)}, # medium power, evening
        ]

        self.test(test_samples)


if __name__ == '__main__':
    print("🚀 啟動決策樹版智能電源管理分析系統")
    print("="*50)
    
    # 創建決策樹版分析實例
    analysis = DecisionTreeSmartPowerAnalysis()
    
    # 運行分析
    analysis.run_analysis()
    
    print("\n🎉 決策樹版分析完成！")