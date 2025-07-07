#!/usr/bin/env python3
"""
智能電源管理分析系統（含節能計算與測試資料）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from device_activity import DeviceActivityScoreModule
    HAS_DEVICE_ACTIVITY = True
except ImportError:
    HAS_DEVICE_ACTIVITY = False

try:
    from user_habit import ImprovedUserHabitScoreModule
    HAS_USER_HABIT = True
except ImportError:
    HAS_USER_HABIT = False

try:
    from confidence_score import ConfidenceScoreModule
    HAS_CONFIDENCE_SCORE = True
except ImportError:
    HAS_CONFIDENCE_SCORE = False

class RealSmartPowerAnalysis:
    def __init__(self):
        self.data_file = 'data_after_preprocessing.csv'
        self.electricity_rate = 0.15
        self.device_activity_model = DeviceActivityScoreModule() if HAS_DEVICE_ACTIVITY else None
        self.user_habit_model = ImprovedUserHabitScoreModule() if HAS_USER_HABIT else None
        self.confidence_model = ConfidenceScoreModule() if HAS_CONFIDENCE_SCORE else None
        self.results = {
            'phantom_load_detected': 0,
            'suggest_shutdown': 0,
            'keep_on': 0,
            'send_notification': 0,
            'delay_decision': 0,
            'total_opportunities': 0
        }

    def _generate_phantom_load_opportunities(self, df):
        """
        從連續 power < 91.4W 區段找出幽靈電機會點
        """
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # 加入 phantom load 標籤：power < 91.4 為 phantom load
        df['is_phantom'] = df['power'] < 91.4
        print(f'phantom load : {len(df['is_phantom'])} counts')

        opportunities = []
        in_session = False
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
                    avg_power = np.mean(power_list) if power_list else 50  # phantom load 常較小
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
            avg_power = np.mean(power_list) if power_list else 50
            opportunities.append({
                'device_id': 'phantom_device',
                'start_time': start_time,
                'end_time': end_time,
                'power_watt': avg_power
            })

        return opportunities

    def _fallback_activity_score(self, features):
        print('fallback activity')
        return 0.1  # 模擬 low 分數

    def _fallback_habit_score(self, features):
        print('fallback habit')
        return 0.1  # 模擬 low 分數

    def _fallback_confidence_score(self, features):
        print('fallback confidence')
        return 0.2  # 模擬 low 分數

    def _extract_enhanced_features(self, opportunity, df):
        return {
            'device_id': opportunity.get('device_id', 'unknown'),
            'duration_minutes': (opportunity['end_time'] - opportunity['start_time']).total_seconds() / 60,
            'hour_of_day': opportunity['start_time'].hour
        }

    def _make_intelligent_decision(self, activity_score, habit_score, confidence_score, features):
        def to_level(score):
            if score < 0.33:
                return "low"
            elif score < 0.66:
                return "medium"
            else:
                return "high"

        user_habit = to_level(habit_score)
        device_activity = to_level(activity_score)
        context_score = to_level(confidence_score)

        if user_habit == "low":
            if device_activity == "low":
                if context_score in ["low", "medium"]:
                    return "suggest_shutdown"
                elif context_score == "high":
                    return "delay_decision"
            elif device_activity == "medium":
                return "delay_decision"
            elif device_activity == "high":
                if context_score == "low":
                    return "delay_decision"
                elif context_score == "medium":
                    return "send_notification"
                elif context_score == "high":
                    return "keep_on"

        elif user_habit == "medium":
            if device_activity == "low":
                if context_score == "low":
                    return "suggest_shutdown"
                else:
                    return "delay_decision"
            elif device_activity == "medium":
                if context_score == "low":
                    return "delay_decision"
                elif context_score == "medium":
                    return "send_notification"
                elif context_score == "high":
                    return "keep_on"
            elif device_activity == "high":
                return "keep_on"

        elif user_habit == "high":
            if device_activity == "low":
                return "delay_decision"
            elif device_activity == "medium":
                return "send_notification"
            elif device_activity == "high":
                return "keep_on"

        return "delay_decision"

    def _apply_real_models(self, opportunities, df):
        print("\n🧠 Applying real intelligent model decisions...")
        decision_results = []

        for opp in opportunities:
            try:
                features = self._extract_enhanced_features(opp, df)

                activity_score = self.device_activity_model.calculate_activity_score(opp['start_time'])['activity_score'] \
                    if self.device_activity_model else self._fallback_activity_score(features)

                habit_score = self.user_habit_model.calculate_habit_score(opp['start_time'])['habit_score'] \
                    if self.user_habit_model else self._fallback_habit_score(features)

                confidence_score = self.confidence_model.calculate_confidence_score(opp['start_time'])['confidence_score'] \
                    if self.confidence_model else self._fallback_confidence_score(features)

                decision = self._make_intelligent_decision(activity_score, habit_score, confidence_score, features)

                # print(f"▶️ Decision: {decision} | Activity: {activity_score:.2f}, Habit: {habit_score:.2f}, Confidence: {confidence_score:.2f}")

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
                    'decision': decision
                }
                decision_results.append(result)

            except Exception as e:
                print(f"   ⚠️ Error processing opportunity: {e}")
                self.results['delay_decision'] += 1

        print(f"\n🎯 Decision distribution:")
        for decision in ['suggest_shutdown', 'keep_on', 'send_notification', 'delay_decision']:
            count = self.results[decision]
            if count > 0:
                print(f"   {decision}: {count} times")

        return decision_results

    def _estimate_energy_saving(self, decision_results):
        total_baseline_kwh = 0
        total_saved_kwh = 0

        for result in decision_results:
            opp = result['opportunity']
            decision = result['decision']

            duration_hr = (opp['end_time'] - opp['start_time']).total_seconds() / 3600
            power_watt = opp.get('power_watt', 100)
            energy_kwh = power_watt * duration_hr / 1000

            total_baseline_kwh += energy_kwh

            if decision == 'suggest_shutdown':
                total_saved_kwh += energy_kwh
            elif decision == 'send_notification':
                total_saved_kwh += energy_kwh * 0.5

        total_saved_cost = total_saved_kwh * self.electricity_rate

        print(f"\n💡 能源節省總結：")
        print(f"   🔋 原始預估耗電量：{total_baseline_kwh:.2f} kWh")
        print(f"   ✅ 實際節省電量：{total_saved_kwh:.2f} kWh")
        print(f"   💰 節省電費約：${total_saved_cost:.2f}")

        return {
            'baseline_kwh': total_baseline_kwh,
            'saved_kwh': total_saved_kwh,
            'saved_cost': total_saved_cost
        }
    
    def test(self, samples):
        for i, sample in enumerate(samples, 1):
            activity = sample.get("activity_score", 0.1)
            habit = sample.get("habit_score", 0.1)
            confidence = sample.get("confidence_score", 0.2)

            features = {
                "device_id": "test_device",
                "duration_minutes": 60,
                "hour_of_day": sample["start_time"].hour
            }

            decision = self._make_intelligent_decision(activity, habit, confidence, features)

            print(f"--- 第 {i} 筆 ---")
            print(f"🕒 時間：{sample['start_time']}")
            print(f"📈 Activity Score: {activity:.2f}")
            print(f"👤 Habit Score: {habit:.2f}")
            print(f"📊 Confidence Score: {confidence:.2f}")
            print(f"🧠 決策結果：{decision}")
            print()


    

if __name__ == '__main__':
    analysis = RealSmartPowerAnalysis()

    try:
        df = pd.read_csv(analysis.data_file)
    except:
        print("❌ 無法讀取 CSV")
        df = pd.DataFrame()

    opportunities = analysis._generate_phantom_load_opportunities(df)
    print(f"✅ 建立 {len(opportunities)} 筆機會點")

    decision_results = analysis._apply_real_models(opportunities, df)

    print("\n📋 前 20 筆決策結果：")
    for i, result in enumerate(decision_results[:20], start=1):
        opp = result['opportunity']
        print(f"\n--- 第 {i} 筆 ---")
        print(f"🕒 時間：{opp['start_time']} ~ {opp['end_time']}")
        print(f"⚡ 平均功率：{opp['power_watt']:.2f} W")
        print(f"⏱️ 持續時間：{result['features']['duration_minutes']:.1f} 分鐘")
        print(f"📈 Activity Score: {result['activity_score']:.2f}")
        print(f"👤 Habit Score: {result['user_habit_score']:.2f}")
        print(f"📊 Confidence Score: {result['confidence_score']:.2f}")
        print(f"🧠 決策結果：{result['decision']}")

    analysis._estimate_energy_saving(decision_results)

    test_samples = [
    {"avg_power": 150, "start_time": datetime(2024, 3, 26, 9, 0)},    # ☀️ 早上高功率
    {"avg_power": 80,  "start_time": datetime(2024, 5, 26, 13, 0)},   # 🌤️ 中午中功率
    {"avg_power": 50,  "start_time": datetime(2024, 7, 26, 20, 0)},   # 🌆 晚上低功率
    {"avg_power": 30,  "start_time": datetime(2024, 9, 26, 2, 30)},   # 🌙 凌晨非常低功率
    {"avg_power": 100, "start_time": datetime(2024, 11, 26, 18, 30)},  # 🌇 傍晚中高功率
    ]

    analysis.test(test_samples)
