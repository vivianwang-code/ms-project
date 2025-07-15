# Integrated Power Management Decision System
# 整合信心分數、設備活躍度分數、用戶習慣分數三個模組
# 使用模糊邏輯進行最終決策

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from user_habit import ImprovedUserHabitScoreModule
from device_activity import DeviceActivityScoreModule
from confidence_score import ConfidenceScoreModule

# 假設您已經有了三個模組的類別（需要導入或複製到同一個文件中）
# from confidence_score import ConfidenceScoreModule
# from device_activity_score import DeviceActivityScoreModule  
# from user_habit_score import ImprovedUserHabitScoreModule
# from fuzzy_decision_system import FuzzyLogicPowerDecisionSystem

class IntegratedPowerManagementSystem:
    """
    整合電源管理決策系統
    結合三個評分模組和模糊邏輯決策系統
    """
    
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        
        # 初始化三個核心模組
        print("="*80)
        print("INITIALIZING INTEGRATED POWER MANAGEMENT SYSTEM")
        print("="*80)
        
        self.confidence_module = None
        self.activity_module = None  
        self.habit_module = None
        self.fuzzy_system = None
        
        # 系統狀態
        self.system_ready = False
        self.initialization_log = []
        
    def initialize_all_modules(self):
        """初始化所有模組"""
        try:
            print("\n🔧 STEP 1: Initializing Confidence Score Module...")
            self.confidence_module = ConfidenceScoreModule()
            # 運行信心分數分析
            self.confidence_module.detect_peak_hours()
            self.confidence_module.detect_sleep_hours()
            self.confidence_module.calculate_data_completeness_score()
            self.initialization_log.append("✅ Confidence Score Module: Ready")
            
        except Exception as e:
            self.initialization_log.append(f"❌ Confidence Score Module: Failed - {e}")
            print(f"⚠️  Confidence module initialization failed: {e}")
        
        try:
            print("\n🔧 STEP 2: Initializing Device Activity Score Module...")
            self.activity_module = DeviceActivityScoreModule()
            # 運行活躍度分析
            activity_result = self.activity_module.run_complete_analysis(self.data_file_path)
            if activity_result:
                self.initialization_log.append("✅ Device Activity Score Module: Ready")
            else:
                self.initialization_log.append("❌ Device Activity Score Module: Failed")
                
        except Exception as e:
            self.initialization_log.append(f"❌ Device Activity Score Module: Failed - {e}")
            print(f"⚠️  Activity module initialization failed: {e}")
        
        try:
            print("\n🔧 STEP 3: Initializing User Habit Score Module...")
            self.habit_module = ImprovedUserHabitScoreModule()
            # 運行習慣分析
            habit_result = self.habit_module.run_complete_analysis(self.data_file_path)
            if habit_result:
                self.initialization_log.append("✅ User Habit Score Module: Ready")
            else:
                self.initialization_log.append("❌ User Habit Score Module: Failed")
                
        except Exception as e:
            self.initialization_log.append(f"❌ User Habit Score Module: Failed - {e}")
            print(f"⚠️  Habit module initialization failed: {e}")
        
        try:
            print("\n🔧 STEP 4: Initializing Fuzzy Logic Decision System...")
            self.fuzzy_system = FuzzyLogicPowerDecisionSystem(
                confidence_module=self.confidence_module,
                activity_module=self.activity_module,
                habit_module=self.habit_module
            )
            self.initialization_log.append("✅ Fuzzy Logic Decision System: Ready")
            
        except Exception as e:
            self.initialization_log.append(f"❌ Fuzzy Logic Decision System: Failed - {e}")
            print(f"⚠️  Fuzzy system initialization failed: {e}")
        
        # 檢查系統就緒狀態
        successful_modules = sum(1 for log in self.initialization_log if "✅" in log)
        total_modules = len(self.initialization_log)
        
        self.system_ready = successful_modules >= 2  # 至少需要2個模組成功
        
        print(f"\n📊 INITIALIZATION SUMMARY:")
        for log in self.initialization_log:
            print(f"   {log}")
        
        print(f"\n🎯 SYSTEM STATUS: {'READY' if self.system_ready else 'PARTIAL/FAILED'}")
        print(f"   Successfully initialized: {successful_modules}/{total_modules} modules")
        
        if self.system_ready:
            print("✅ System ready for power management decisions!")
        else:
            print("⚠️  System partially ready - some features may be limited")
        
        return self.system_ready

    def make_smart_power_decision(self, timestamp=None, context=None):
        """智能電源決策"""
        if not self.system_ready:
            print("❌ System not ready. Please initialize modules first.")
            return None
        
        if timestamp is None:
            timestamp = datetime.now()
        
        print(f"\n{'='*60}")
        print(f"SMART POWER DECISION for {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        try:
            # 使用模糊邏輯系統進行決策
            decision_result = self.fuzzy_system.make_power_decision(timestamp, verbose=True)
            
            # 增強決策結果
            enhanced_result = self._enhance_decision_result(decision_result, context)
            
            # 生成行動建議
            action_plan = self._generate_action_plan(enhanced_result)
            
            enhanced_result['action_plan'] = action_plan
            enhanced_result['system_status'] = 'ready'
            
            return enhanced_result
            
        except Exception as e:
            print(f"❌ Error making power decision: {e}")
            return {
                'timestamp': timestamp,
                'decision': 'delay',
                'confidence': 0.1,
                'error': str(e),
                'system_status': 'error'
            }

    def _enhance_decision_result(self, decision_result, context=None):
        """增強決策結果"""
        enhanced = decision_result.copy()
        
        # 添加上下文信息
        if context:
            enhanced['context'] = context
            
            # 根據上下文調整決策
            if context.get('user_override', False):
                enhanced['decision'] = 'keep_on'
                enhanced['override_reason'] = 'User manually overridden'
            
            if context.get('system_critical', False):
                if enhanced['decision'] == 'shutdown':
                    enhanced['decision'] = 'notification'
                    enhanced['critical_reason'] = 'System critical processes detected'
        
        # 添加決策信心等級
        confidence = enhanced.get('confidence', 0.5)
        if confidence >= 0.8:
            enhanced['confidence_level'] = 'very_high'
        elif confidence >= 0.6:
            enhanced['confidence_level'] = 'high'
        elif confidence >= 0.4:
            enhanced['confidence_level'] = 'medium'
        elif confidence >= 0.2:
            enhanced['confidence_level'] = 'low'
        else:
            enhanced['confidence_level'] = 'very_low'
        
        # 添加風險評估
        enhanced['risk_assessment'] = self._assess_decision_risk(enhanced)
        
        return enhanced

    def _assess_decision_risk(self, decision_result):
        """評估決策風險"""
        decision = decision_result.get('decision', 'delay')
        confidence = decision_result.get('confidence', 0.5)
        
        risk_factors = []
        risk_level = 'low'
        
        # 基於決策類型的風險
        if decision == 'shutdown':
            if confidence < 0.7:
                risk_factors.append('低信心度的關機決策可能導致用戶不便')
                risk_level = 'medium'
            if decision_result.get('input_scores', {}).get('habit_score', 0) > 0.6:
                risk_factors.append('用戶習慣性使用強，關機可能不當')
                risk_level = 'high'
        
        elif decision == 'keep_on':
            if decision_result.get('input_scores', {}).get('activity_score', 0) < 0.3:
                risk_factors.append('低活躍度時保持開機可能浪費能源')
                risk_level = 'medium'
        
        # 基於信心度的風險
        if confidence < 0.3:
            risk_factors.append('決策信心度很低，結果可能不準確')
            risk_level = 'high'
        
        return {
            'level': risk_level,
            'factors': risk_factors
        }

    def _generate_action_plan(self, decision_result):
        """生成行動計劃"""
        decision = decision_result.get('decision', 'delay')
        confidence = decision_result.get('confidence', 0.5)
        
        action_plan = {
            'immediate_action': '',
            'follow_up_actions': [],
            'monitoring_required': False,
            'user_notification': False
        }
        
        if decision == 'shutdown':
            action_plan['immediate_action'] = '執行系統關機程序'
            action_plan['follow_up_actions'] = [
                '確認所有應用程序正常關閉',
                '記錄關機時間和原因',
                '下次開機時檢查系統狀態'
            ]
            action_plan['user_notification'] = True
            
        elif decision == 'delay':
            action_plan['immediate_action'] = '延遲決策，繼續監控'
            action_plan['follow_up_actions'] = [
                f'在{15}分鐘後重新評估',
                '收集更多使用數據',
                '監控用戶活動變化'
            ]
            action_plan['monitoring_required'] = True
            
        elif decision == 'notification':
            action_plan['immediate_action'] = '發送節能提醒給用戶'
            action_plan['follow_up_actions'] = [
                '等待用戶響應(最多30分鐘)',
                '記錄用戶響應行為',
                '根據響應調整未來決策'
            ]
            action_plan['user_notification'] = True
            action_plan['monitoring_required'] = True
            
        else:  # keep_on
            action_plan['immediate_action'] = '保持系統開機狀態'
            action_plan['follow_up_actions'] = [
                '繼續正常運行',
                '監控電源使用效率',
                '記錄使用模式變化'
            ]
        
        # 根據信心度調整行動計劃
        if confidence < 0.4:
            action_plan['follow_up_actions'].insert(0, '由於低信心度，建議人工審核決策')
        
        return action_plan

    def run_24hour_simulation(self, start_date=None, interval_minutes=60):
        """運行24小時決策模擬"""
        if not self.system_ready:
            print("❌ System not ready for simulation")
            return None
        
        if start_date is None:
            start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        print(f"\n{'='*80}")
        print(f"24-HOUR POWER DECISION SIMULATION")
        print(f"Start: {start_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Interval: {interval_minutes} minutes")
        print(f"{'='*80}")
        
        # 生成24小時的時間點
        timestamps = []
        current_time = start_date
        while current_time < start_date + timedelta(days=1):
            timestamps.append(current_time)
            current_time += timedelta(minutes=interval_minutes)
        
        # 運行批量決策分析
        simulation_results = self.fuzzy_system.batch_decision_analysis(timestamps, save_results=True)
        
        # 分析模擬結果
        self._analyze_simulation_results(simulation_results, start_date)
        
        # 繪製模擬圖表
        self._plot_simulation_results(simulation_results, start_date)
        
        return simulation_results

    def _analyze_simulation_results(self, results, start_date):
        """分析模擬結果"""
        print(f"\n📊 SIMULATION ANALYSIS:")
        
        # 決策分布統計
        decisions = [r.get('decision', 'unknown') for r in results]
        decision_counts = pd.Series(decisions).value_counts()
        
        print(f"\n🎯 Decision Distribution:")
        for decision, count in decision_counts.items():
            percentage = count / len(results) * 100
            print(f"   {decision.capitalize()}: {count} times ({percentage:.1f}%)")
        
        # 每個時段的平均分數
        hourly_stats = {}
        for result in results:
            if 'input_scores' in result:
                hour = result['timestamp'].hour
                if hour not in hourly_stats:
                    hourly_stats[hour] = {'confidence': [], 'activity': [], 'habit': []}
                
                hourly_stats[hour]['confidence'].append(result['input_scores']['confidence_score'])
                hourly_stats[hour]['activity'].append(result['input_scores']['activity_score'])
                hourly_stats[hour]['habit'].append(result['input_scores']['habit_score'])
        
        # 找出關機建議最多的時段
        shutdown_hours = []
        for result in results:
            if result.get('decision') == 'shutdown':
                shutdown_hours.append(result['timestamp'].hour)
        
        if shutdown_hours:
            most_shutdown_hour = max(set(shutdown_hours), key=shutdown_hours.count)
            print(f"\n⚡ Most frequent shutdown hour: {most_shutdown_hour:02d}:00")
        
        # 節能效果估算
        shutdown_count = decision_counts.get('shutdown', 0)
        delay_count = decision_counts.get('delay', 0)
        total_hours = 24
        
        # 假設每小時待機功率為50W，關機後功率為5W
        standby_power = 50  # 瓦
        shutdown_power = 5  # 瓦
        energy_saved = (shutdown_count * (standby_power - shutdown_power)) / 1000  # kWh
        
        print(f"\n💡 Energy Efficiency Estimation:")
        print(f"   Shutdown decisions: {shutdown_count}/{total_hours} hours")
        print(f"   Estimated energy saved: {energy_saved:.2f} kWh")
        print(f"   Energy saving percentage: {(energy_saved/(standby_power*24/1000))*100:.1f}%")

    def _plot_simulation_results(self, results, start_date):
        """繪製模擬結果"""
        if not results:
            return
        
        # 過濾有效結果
        valid_results = [r for r in results if 'input_scores' in r and 'decision' in r]
        
        if len(valid_results) < 2:
            print("❌ Insufficient data for plotting")
            return
        
        # 創建圖表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'24-Hour Power Management Simulation\n{start_date.strftime("%Y-%m-%d")}', 
                     fontsize=16, fontweight='bold')
        
        # 準備數據
        hours = [r['timestamp'].hour for r in valid_results]
        decisions = [r['decision'] for r in valid_results]
        confidence_scores = [r['input_scores']['confidence_score'] for r in valid_results]
        activity_scores = [r['input_scores']['activity_score'] for r in valid_results]
        habit_scores = [r['input_scores']['habit_score'] for r in valid_results]
        
        # 1. 24小時決策時間線
        ax1 = axes[0, 0]
        decision_colors = {'shutdown': 'red', 'delay': 'orange', 'notification': 'yellow', 'keep_on': 'green'}
        
        for i, (hour, decision) in enumerate(zip(hours, decisions)):
            color = decision_colors.get(decision, 'gray')
            ax1.scatter(hour, i, c=color, s=100, alpha=0.7)
        
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Decision Sequence')
        ax1.set_title('24-Hour Decision Timeline')
        ax1.set_xticks(range(0, 24, 4))
        ax1.grid(True, alpha=0.3)
        
        # 添加圖例
        for decision, color in decision_colors.items():
            ax1.scatter([], [], c=color, label=decision.capitalize())
        ax1.legend()
        
        # 2. 分數趨勢圖
        ax2 = axes[0, 1]
        ax2.plot(hours, confidence_scores, 'b-o', label='Confidence', linewidth=2, markersize=4)
        ax2.plot(hours, activity_scores, 'g-s', label='Activity', linewidth=2, markersize=4)
        ax2.plot(hours, habit_scores, 'r-^', label='Habit', linewidth=2, markersize=4)
        
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Score')
        ax2.set_title('Score Trends Throughout Day')
        ax2.set_xticks(range(0, 24, 4))
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # 3. 每小時決策分布
        ax3 = axes[1, 0]
        hourly_decisions = {}
        for hour, decision in zip(hours, decisions):
            if hour not in hourly_decisions:
                hourly_decisions[hour] = {}
            hourly_decisions[hour][decision] = hourly_decisions[hour].get(decision, 0) + 1
        
        # 創建堆疊柱狀圖
        decision_types = ['shutdown', 'delay', 'notification', 'keep_on']
        hour_range = sorted(set(hours))
        
        bottom = np.zeros(len(hour_range))
        for decision in decision_types:
            values = [hourly_decisions.get(h, {}).get(decision, 0) for h in hour_range]
            color = decision_colors.get(decision, 'gray')
            ax3.bar(hour_range, values, bottom=bottom, label=decision.capitalize(), color=color, alpha=0.8)
            bottom += values
        
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('Decision Count')
        ax3.set_title('Hourly Decision Distribution')
        ax3.legend()
        ax3.set_xticks(range(0, 24, 4))
        
        # 4. 決策信心度分布
        ax4 = axes[1, 1]
        confidence_levels = [r.get('confidence', 0.5) for r in valid_results]
        
        # 按決策類型分組顯示信心度
        for decision in decision_types:
            decision_confidences = [
                conf for conf, dec in zip(confidence_levels, decisions) if dec == decision
            ]
            if decision_confidences:
                ax4.hist(decision_confidences, bins=10, alpha=0.7, 
                        label=f'{decision.capitalize()} (n={len(decision_confidences)})',
                        color=decision_colors.get(decision, 'gray'))
        
        ax4.set_xlabel('Decision Confidence')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Decision Confidence Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def generate_system_report(self):
        """生成系統報告"""
        print(f"\n{'='*80}")
        print("INTEGRATED POWER MANAGEMENT SYSTEM REPORT")
        print(f"{'='*80}")
        
        print(f"\n📋 SYSTEM INITIALIZATION STATUS:")
        for log in self.initialization_log:
            print(f"   {log}")
        
        print(f"\n🎯 SYSTEM CAPABILITIES:")
        capabilities = []
        
        if self.confidence_module:
            capabilities.append("✅ 資料信心度評估")
        if self.activity_module:
            capabilities.append("✅ 設備活躍度分析")
        if self.habit_module:
            capabilities.append("✅ 用戶習慣模式識別")
        if self.fuzzy_system:
            capabilities.append("✅ 模糊邏輯智能決策")
        
        for capability in capabilities:
            print(f"   {capability}")
        
        print(f"\n⚙️  DECISION FRAMEWORK:")
        print("   Input Factors:")
        print("     • Data Confidence Score (0.0-1.0)")
        print("     • Device Activity Score (0.0-1.0)")  
        print("     • User Habit Score (0.0-1.0)")
        print("   Output Decisions:")
        print("     • SHUTDOWN: 立即關機節能")
        print("     • DELAY: 延遲決策，繼續觀察")
        print("     • NOTIFICATION: 發送節能提醒")
        print("     • KEEP_ON: 保持開機狀態")
        
        print(f"\n🔧 SYSTEM READY: {'YES' if self.system_ready else 'NO'}")
        
        if self.system_ready:
            print("\n✅ 系統已就緒，可以進行智能電源管理決策！")
            print("\n📝 建議使用方式:")
            print("   1. make_smart_power_decision() - 單次決策")
            print("   2. run_24hour_simulation() - 24小時模擬")
            print("   3. 根據決策結果執行相應的電源管理動作")
        else:
            print("\n⚠️  系統未完全就緒，請檢查模組初始化問題")

# 使用示例
if __name__ == "__main__":
    # 假設的數據文件路徑
    data_file_path = "C:/Users/王俞文/Documents/glasgow/msc project/data/data_after_preprocessing.csv"
    
    # 初始化整合系統
    power_management_system = IntegratedPowerManagementSystem(data_file_path)
    
    # 初始化所有模組
    system_ready = power_management_system.initialize_all_modules()
    
    if system_ready:
        # 生成系統報告
        power_management_system.generate_system_report()
        
        # 測試單次決策
        print(f"\n{'-'*60}")
        print("TESTING SINGLE DECISION")
        print(f"{'-'*60}")
        
        test_time = datetime(2024, 6, 15, 14, 30)
        decision_result = power_management_system.make_smart_power_decision(
            timestamp=test_time,
            context={'user_override': False, 'system_critical': False}
        )
        
        if decision_result:
            print(f"\n📋 DECISION SUMMARY:")
            print(f"   Decision: {decision_result['decision'].upper()}")
            print(f"   Confidence: {decision_result['confidence']:.3f}")
            print(f"   Action Plan: {decision_result['action_plan']['immediate_action']}")
        
        # 運行24小時模擬
        print(f"\n{'-'*60}")
        print("RUNNING 24-HOUR SIMULATION")
        print(f"{'-'*60}")
        
        simulation_start = datetime(2024, 6, 15, 0, 0)
        simulation_results = power_management_system.run_24hour_simulation(
            start_date=simulation_start,
            interval_minutes=120  # 每2小時一次決策
        )
        
        print("\n" + "="*80)
        print("✅ INTEGRATED SYSTEM TESTING COMPLETE!")
        print("="*80)
    
    else:
        print("\n❌ 系統初始化失敗，無法進行完整測試")
        power_management_system.generate_system_report()