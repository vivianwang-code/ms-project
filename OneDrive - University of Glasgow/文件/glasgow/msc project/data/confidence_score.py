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
                'base_confidence': 0.1,  # 很低，應該睡覺
                'description': '深夜睡眠時間，不建議使用電子設備'
            },
            
            # 早晨 (06:00-08:59) - 起床準備時間
            'early_morning': {
                'hours': list(range(6, 9)),
                'base_confidence': 0.4,  # 中等，適度使用
                'description': '早晨時光，適度使用'
            },
            
            # 上午 (09:00-11:59) - 工作/學習時間
            'morning': {
                'hours': list(range(9, 12)),
                'base_confidence': 1.0,  # 中低，應該專注工作
                'description': '上午工作時間'
            },
            
            # 下午 (12:00-14:59) - 午餐和休息
            'afternoon': {
                'hours': list(range(12, 15)),
                'base_confidence': 1.0,  # 中高
                'description': '午餐時間'
            },
            
            # 下午晚些 (15:00-17:59) - 工作時間
            'late_afternoon': {
                'hours': list(range(15, 18)),
                'base_confidence': 0.8,  # 中低
                'description': '下午工作時間'
            },
            
            # 傍晚 (18:00-20:59) - 放鬆娛樂時間
            'evening': {
                'hours': list(range(18, 21)),
                'base_confidence': 0.5,  # 高
                'description': '傍晚娛樂時間，合理使用'
            },
            
            # 晚上 (21:00-21:59) 
            'night': {
                'hours': [21],
                'base_confidence': 0.4,  # 中高
                'description': '晚上時間'
            },
            
            # 深夜前 (22:00-23:59) - 應該準備睡覺
            'late_night': {
                'hours': [22, 23],
                'base_confidence': 0.2,  # 低，應該準備睡覺
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
        if 6 <= hour <= 9:  # 週末早晨
            return max(0.1, base_confidence - 0.5)
        elif 9 <= hour <= 12:  # 週末上午工作時間
            return max(0.1, base_confidence - 0.2)  # 大懲罰
        elif 12 <= hour <= 15:  # 週末午餐時間（你的14:30在這裡）
            return max(0.1, base_confidence - 0.3)  # 新增！大懲罰
        elif 15 <= hour <= 18:  # 週末下午工作時間
            return max(0.1, base_confidence - 0.35)  # 大懲罰
        elif 18 <= hour <= 21:  # 週末晚間
            return max(0.1, base_confidence - 0.4)  # 相對寬鬆
        elif 22 <= hour <= 23:  # 週末晚上
            return max(0.1, base_confidence - 0.5)
        elif 0 <= hour <= 2:   # 週末深夜
            return max(0.1, base_confidence - 0.6)
        else:
            return max(0.1, base_confidence - 0.3)

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
                return -0.2
            elif 10 <= hour <= 22:  # 週末白天，相對寬鬆
                return -0.15
        
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
            (datetime(2025, 7, 17, 9, 0), (0.3, 0.5), '工作日早上'),   # 工作時間，中低
            (datetime(2025, 1, 17, 10, 0), (0.5, 0.7), '工作日下午'), # 午休時間，中高  
            (datetime(2025, 1, 17, 11, 0), (0.7, 0.9), '工作日晚上'),  # 娛樂時間，高
            (datetime(2025, 1, 17, 16, 0), (0.2, 0.4), '工作日深夜'), # 睡覺時間，低
            (datetime(2025, 1, 13, 20, 0), (0.7, 0.9), '週末晚上'),   # 週末娛樂，高
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
            datetime(2025, 7, 17, 9, 0),   # 您的原始測試時間
            datetime(2025, 7, 17, 10, 0),   # 晚間娛樂時間
            datetime(2025, 7, 17, 11, 0),   # 工作時間
            datetime(2025, 7, 17, 16, 30),  # 午休時間
            datetime(2025, 7, 17, 5, 30),
            datetime(2025, 7, 17, 23, 30),
            datetime(2025, 7, 19, 10, 30),
        ]
        
        for test_time in test_times:
            result = confidence_module.calculate_confidence_score(test_time)
            
            print(f"\n🕐 時間: {test_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"📊 置信度分數: {result['confidence_score']:.3f}")
            print(f"🎯 置信度等級: {result['confidence_level']}")
            print(f"⏰ 時間段: {result['time_period']}")
            print(f"💡 解釋: {result['explanation']}")
            print(f"📝 建議: {result['recommendation']}")