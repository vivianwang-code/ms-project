"""
調試版決策系統 - 顯示所有print輸出，檢查是否使用專業模型還是fallback
"""

import sys
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

try:
    from fuzzy_logic_control import DecisionTreeSmartPowerAnalysis
    HAS_MAIN_SYSTEM = True
except ImportError:
    HAS_MAIN_SYSTEM = False
    print("❌ 請確保 fuzzy_logic_control.py 在同一目錄下")

# 全局變量
_debug_system = None

def init_debug_system():
    """調試版初始化（只隱藏訓練過程，保留調試信息）"""
    global _debug_system
    
    if _debug_system is None:
        if not HAS_MAIN_SYSTEM:
            raise ImportError("無法導入 fuzzy_logic_control 模組")
        
        print("🚀 初始化調試版決策系統...")
        # 只隱藏訓練過程的輸出
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            _debug_system = DecisionTreeSmartPowerAnalysis()
        print("✅ 調試版系統初始化完成！")
    
    return _debug_system

def debug_decision(power_value, timestamp=None):
    """
    調試版決策 - 顯示詳細過程
    """
    
    print(f"\n🔍 調試決策過程 - 功率 {power_value}W")
    print("-" * 40)
    
    # 初始化系統
    system = init_debug_system()
    
    if timestamp is None:
        timestamp = datetime.now()
    
    print(f"📅 時間: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 檢查模型狀態
    print(f"\n📊 模型狀態檢查:")
    print(f"   device_activity_model: {'✅ 存在' if system.device_activity_model else '❌ 不存在'}")
    print(f"   user_habit_model: {'✅ 存在' if system.user_habit_model else '❌ 不存在'}")
    print(f"   confidence_model: {'✅ 存在' if system.confidence_model else '❌ 不存在'}")
    
    # 創建機會點
    opportunity = {
        'device_id': 'test_device',
        'start_time': timestamp,
        'end_time': timestamp + timedelta(minutes=15),
        'power_watt': power_value
    }
    
    # 提取特徵
    features = system._extract_enhanced_features(opportunity, None)
    print(f"\n🎯 提取的特徵: {features}")
    
    # 獲取分數（不隱藏輸出）
    print(f"\n🧮 計算分數:")
    
    # 1. Activity Score
    print(f"\n1️⃣ Activity Score:")
    if system.device_activity_model:
        try:
            activity_result = system.device_activity_model.calculate_activity_score(timestamp)
            activity_score = activity_result['activity_score']
            print(f"   ✅ 專業模型成功!")
            print(f"   📊 完整結果: {activity_result}")
            print(f"   🎯 Activity Score: {activity_score}")
            activity_source = "專業模型"
        except Exception as e:
            print(f"   ❌ 專業模型失敗: {e}")
            print(f"   🔄 使用 Fallback...")
            activity_score = system._fallback_activity_score(features, timestamp)
            print(f"   🎲 Fallback Activity Score: {activity_score}")
            activity_source = "Fallback"
    else:
        print(f"   ❌ 模型不存在，使用 Fallback")
        activity_score = system._fallback_activity_score(features, timestamp)
        print(f"   🎲 Fallback Activity Score: {activity_score}")
        activity_source = "Fallback"
    
    # 2. Habit Score
    print(f"\n2️⃣ Habit Score:")
    if system.user_habit_model:
        try:
            habit_result = system.user_habit_model.calculate_habit_score(timestamp)
            habit_score = habit_result['habit_score']
            print(f"   ✅ 專業模型成功!")
            print(f"   📊 完整結果: {habit_result}")
            print(f"   🎯 Habit Score: {habit_score}")
            habit_source = "專業模型"
        except Exception as e:
            print(f"   ❌ 專業模型失敗: {e}")
            print(f"   🔄 使用 Fallback...")
            habit_score = system._fallback_habit_score(features, timestamp)
            print(f"   🎲 Fallback Habit Score: {habit_score}")
            habit_source = "Fallback"
    else:
        print(f"   ❌ 模型不存在，使用 Fallback")
        habit_score = system._fallback_habit_score(features, timestamp)
        print(f"   🎲 Fallback Habit Score: {habit_score}")
        habit_source = "Fallback"
    
    # 3. Confidence Score
    print(f"\n3️⃣ Confidence Score:")
    if system.confidence_model:
        try:
            confidence_result = system.confidence_model.calculate_confidence_score(timestamp)
            confidence_score = confidence_result['confidence_score']
            print(f"   ✅ 專業模型成功!")
            print(f"   📊 完整結果: {confidence_result}")
            print(f"   🎯 Confidence Score: {confidence_score}")
            confidence_source = "專業模型"
        except Exception as e:
            print(f"   ❌ 專業模型失敗: {e}")
            print(f"   🔄 使用 Fallback...")
            confidence_score = system._fallback_confidence_score(features, timestamp)
            print(f"   🎲 Fallback Confidence Score: {confidence_score}")
            confidence_source = "Fallback"
    else:
        print(f"   ❌ 模型不存在，使用 Fallback")
        confidence_score = system._fallback_confidence_score(features, timestamp)
        print(f"   🎲 Fallback Confidence Score: {confidence_score}")
        confidence_source = "Fallback"
    
    # 決策
    print(f"\n🧠 進行決策:")
    decision, debug_info = system._make_intelligent_decision(
        activity_score, habit_score, confidence_score, features
    )
    
    print(f"   📊 最終分數: A:{activity_score:.3f} H:{habit_score:.3f} C:{confidence_score:.3f}")
    print(f"   🎯 等級轉換: {debug_info['device_activity_level']}-{debug_info['user_habit_level']}-{debug_info['confidence_score_level']}")
    print(f"   🛤️ 決策路徑: {' → '.join(debug_info['decision_path'])}")
    print(f"   🧠 最終決策: {decision}")
    
    # 總結數據源
    print(f"\n📋 數據源總結:")
    print(f"   Activity Score ({activity_score:.3f}): {activity_source}")
    print(f"   Habit Score ({habit_score:.3f}): {habit_source}")
    print(f"   Confidence Score ({confidence_score:.3f}): {confidence_source}")
    
    # 計算專業模型使用率
    professional_count = sum([
        1 if activity_source == "專業模型" else 0,
        1 if habit_source == "專業模型" else 0,
        1 if confidence_source == "專業模型" else 0
    ])
    
    professional_percentage = (professional_count / 3) * 100
    print(f"   🎯 專業模型使用率: {professional_percentage:.0f}% ({professional_count}/3)")
    
    return {
        'decision': decision,
        'scores': {
            'activity': activity_score,
            'habit': habit_score,
            'confidence': confidence_score
        },
        'sources': {
            'activity': activity_source,
            'habit': habit_source,
            'confidence': confidence_source
        },
        'professional_percentage': professional_percentage,
        'debug_info': debug_info
    }

def quick_debug_test(power_value):
    """快速調試測試"""
    result = debug_decision(power_value)
    
    # 決策對應的中文說明
    answers = {
        'suggest_shutdown': '🔴 建議關機',
        'send_notification': '🔔 發送通知', 
        'keep_on': '🟢 保持開啟',
        'delay_decision': '🟡 延遲決策'
    }
    
    answer = answers.get(result['decision'], result['decision'])
    
    print(f"\n🎯 最終結果: {power_value}W → {answer}")
    
    if result['professional_percentage'] == 100:
        print(f"   ✅ 完美！所有模型都在正常工作")
    elif result['professional_percentage'] >= 50:
        print(f"   ⚠️ 部分模型使用 Fallback")
    else:
        print(f"   ❌ 大部分模型使用 Fallback - 結果可能不穩定")
    
    return result

def test_multiple_powers():
    """測試多個功率值"""
    print("🧪 批量調試測試")
    print("=" * 50)
    
    test_powers = [15, 25, 50, 95]
    results = []
    
    for power in test_powers:
        print(f"\n{'='*60}")
        print(f"測試功率: {power}W")
        print(f"{'='*60}")
        
        result = quick_debug_test(power)
        results.append((power, result))
    
    # 總結
    print(f"\n🏁 總結報告:")
    print("-" * 30)
    
    for power, result in results:
        decision_cn = {
            'suggest_shutdown': '關機',
            'send_notification': '通知', 
            'keep_on': '保持',
            'delay_decision': '延遲'
        }.get(result['decision'], result['decision'])
        
        print(f"{power:3d}W → {decision_cn} | 專業模型使用率: {result['professional_percentage']:.0f}%")

if __name__ == "__main__":
    print("🔍 調試版功率決策系統")
    print("=" * 40)
    
    # 測試單個值
    quick_debug_test(15)
    
    # 測試多個值
    test_multiple_powers()