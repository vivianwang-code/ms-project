import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class IntegratedPowerManagementSystem:
    """
    整合電源管理決策系統
    使用物件導向方式整合三個評分模組和模糊邏輯決策系統
    """
    
    def __init__(self, data_file_path=None):
        """
        初始化整合系統
        
        Parameters:
        -----------
        data_file_path : str, optional
            數據文件路徑，如果提供將自動初始化所有模組
        """
        self.data_file_path = data_file_path
        
        # 模組實例
        self.confidence_module = None
        self.activity_module = None
        self.habit_module = None
        self.fuzzy_system = None
        
        # 系統狀態
        self.system_ready = False
        self.initialization_log = []
        
        print("="*80)
        print("INTEGRATED POWER MANAGEMENT SYSTEM - OBJECT-ORIENTED")
        print("="*80)
        print("📋 系統支援三種初始化方式：")
        print("   1. 自動初始化：提供數據文件路徑")
        print("   2. 手動注入：使用 inject_modules() 方法")
        print("   3. 逐步初始化：使用個別的 set_xxx_module() 方法")

    def inject_modules(self, confidence_module=None, activity_module=None, 
                      habit_module=None, auto_create_fuzzy=True):
        """
        注入已初始化的模組實例
        
        Parameters:
        -----------
        confidence_module : ConfidenceScoreModule, optional
            信心分數模組實例
        activity_module : DeviceActivityScoreModule, optional
            設備活躍度模組實例
        habit_module : ImprovedUserHabitScoreModule, optional
            用戶習慣模組實例
        auto_create_fuzzy : bool, default=True
            是否自動創建模糊邏輯決策系統
        """
        print("\n🔧 INJECTING MODULES...")
        
        # 注入模組
        if confidence_module is not None:
            self.confidence_module = confidence_module
            self.initialization_log.append("✅ Confidence Module: Injected")
            print("   ✅ 信心分數模組已注入")
            
        if activity_module is not None:
            self.activity_module = activity_module
            self.initialization_log.append("✅ Activity Module: Injected")
            print("   ✅ 設備活躍度模組已注入")
            
        if habit_module is not None:
            self.habit_module = habit_module
            self.initialization_log.append("✅ Habit Module: Injected")
            print("   ✅ 用戶習慣模組已注入")
        
        # 自動創建模糊邏輯系統
        if auto_create_fuzzy:
            self._create_fuzzy_system()
        
        # 更新系統狀態
        self._update_system_status()
        
        return self.system_ready

    def set_confidence_module(self, confidence_module):
        """設置信心分數模組"""
        self.confidence_module = confidence_module
        self.initialization_log.append("✅ Confidence Module: Set")
        print("✅ 信心分數模組已設置")
        self._update_system_status()

    def set_activity_module(self, activity_module):
        """設置設備活躍度模組"""
        self.activity_module = activity_module
        self.initialization_log.append("✅ Activity Module: Set")
        print("✅ 設備活躍度模組已設置")
        self._update_system_status()

    def set_habit_module(self, habit_module):
        """設置用戶習慣模組"""
        self.habit_module = habit_module
        self.initialization_log.append("✅ Habit Module: Set")
        print("✅ 用戶習慣模組已設置")
        self._update_system_status()

    def _create_fuzzy_system(self):
        """創建模糊邏輯決策系統"""
        try:
            self.fuzzy_system = FuzzyLogicPowerDecisionSystem(
                confidence_module=self.confidence_module,
                activity_module=self.activity_module,
                habit_module=self.habit_module
            )
            self.initialization_log.append("✅ Fuzzy Logic System: Created")
            print("   ✅ 模糊邏輯決策系統已創建")
            return True
        except Exception as e:
            self.initialization_log.append(f"❌ Fuzzy Logic System: Failed - {e}")
            print(f"   ❌ 模糊邏輯系統創建失敗: {e}")
            return False

    def _update_system_status(self):
        """更新系統就緒狀態"""
        # 計算成功模組數量
        available_modules = sum([
            self.confidence_module is not None,
            self.activity_module is not None,
            self.habit_module is not None
        ])
        
        # 至少需要2個模組和模糊邏輯系統
        self.system_ready = (available_modules >= 2 and self.fuzzy_system is not None)
        
        if available_modules >= 2 and self.fuzzy_system is None:
            self._create_fuzzy_system()

    def auto_initialize_from_data(self, data_file_path=None):
        """
        從數據文件自動初始化所有模組
        
        Parameters:
        -----------
        data_file_path : str, optional
            數據文件路徑，如果不提供則使用初始化時的路徑
        """
        if data_file_path:
            self.data_file_path = data_file_path
        
        if not self.data_file_path:
            print("❌ 未提供數據文件路徑")
            return False
        
        print(f"\n🔧 AUTO-INITIALIZING FROM DATA: {self.data_file_path}")
        
        success_count = 0
        
        # 1. 初始化信心分數模組
        try:
            print("\n1️⃣  初始化信心分數模組...")
            from confidence_score import ConfidenceScoreModule  # 假設模組在獨立文件中
            
            self.confidence_module = ConfidenceScoreModule()
            self.confidence_module.detect_peak_hours()
            self.confidence_module.detect_sleep_hours()
            self.confidence_module.calculate_data_completeness_score()
            
            self.initialization_log.append("✅ Confidence Module: Auto-initialized")
            print("   ✅ 信心分數模組初始化成功")
            success_count += 1
            
        except Exception as e:
            self.initialization_log.append(f"❌ Confidence Module: Failed - {e}")
            print(f"   ❌ 信心分數模組初始化失敗: {e}")
        
        # 2. 初始化設備活躍度模組
        try:
            print("\n2️⃣  初始化設備活躍度模組...")
            from device_activity import DeviceActivityScoreModule  # 假設模組在獨立文件中
            
            self.activity_module = DeviceActivityScoreModule()
            result = self.activity_module.run_complete_analysis(self.data_file_path)
            
            if result:
                self.initialization_log.append("✅ Activity Module: Auto-initialized")
                print("   ✅ 設備活躍度模組初始化成功")
                success_count += 1
            else:
                raise Exception("Analysis returned None")
                
        except Exception as e:
            self.initialization_log.append(f"❌ Activity Module: Failed - {e}")
            print(f"   ❌ 設備活躍度模組初始化失敗: {e}")
        
        # 3. 初始化用戶習慣模組
        try:
            print("\n3️⃣  初始化用戶習慣模組...")
            from user_habit import ImprovedUserHabitScoreModule  # 假設模組在獨立文件中
            
            self.habit_module = ImprovedUserHabitScoreModule()
            result = self.habit_module.run_complete_analysis(self.data_file_path)
            
            if result:
                self.initialization_log.append("✅ Habit Module: Auto-initialized")
                print("   ✅ 用戶習慣模組初始化成功")
                success_count += 1
            else:
                raise Exception("Analysis returned None")
                
        except Exception as e:
            self.initialization_log.append(f"❌ Habit Module: Failed - {e}")
            print(f"   ❌ 用戶習慣模組初始化失敗: {e}")
        
        # 4. 創建模糊邏輯系統
        if success_count >= 2:
            print("\n4️⃣  創建模糊邏輯決策系統...")
            fuzzy_created = self._create_fuzzy_system()
            if fuzzy_created:
                success_count += 1
        
        # 更新系統狀態
        self._update_system_status()
        
        # 總結初始化結果
        print(f"\n📊 AUTO-INITIALIZATION SUMMARY:")
        print(f"   成功初始化: {success_count}/4 個組件")
        print(f"   系統狀態: {'就緒' if self.system_ready else '部分就緒/失敗'}")
        
        return self.system_ready

    def make_power_decision(self, timestamp=None, context=None, verbose=True):
        """
        進行智能電源決策
        
        Parameters:
        -----------
        timestamp : datetime, optional
            決策時間點，默認為當前時間
        context : dict, optional
            上下文信息，如 {'user_override': False, 'system_critical': False}
        verbose : bool, default=True
            是否顯示詳細信息
        
        Returns:
        --------
        dict : 決策結果
        """
        if not self.system_ready:
            print("❌ 系統未就緒，無法進行決策")
            return None
        
        if timestamp is None:
            timestamp = datetime.now()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🤖 MAKING POWER DECISION")
            print(f"⏰ Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
        
        try:
            # 使用模糊邏輯系統進行決策
            decision_result = self.fuzzy_system.make_power_decision(timestamp, verbose=verbose)
            
            # 增強決策結果
            enhanced_result = self._enhance_decision_result(decision_result, context)
            
            # 生成行動建議
            action_plan = self._generate_action_plan(enhanced_result)
            enhanced_result['action_plan'] = action_plan
            
            if verbose:
                self._print_enhanced_summary(enhanced_result)
            
            return enhanced_result
            
        except Exception as e:
            error_result = {
                'timestamp': timestamp,
                'decision': 'delay',
                'confidence': 0.1,
                'error': str(e),
                'system_status': 'error'
            }
            if verbose:
                print(f"❌ 決策過程發生錯誤: {e}")
            return error_result

    def _enhance_decision_result(self, decision_result, context=None):
        """增強決策結果"""
        enhanced = decision_result.copy()
        
        # 添加上下文調整
        if context:
            enhanced['context'] = context
            
            if context.get('user_override', False):
                enhanced['original_decision'] = enhanced['decision']
                enhanced['decision'] = 'keep_on'
                enhanced['override_reason'] = '用戶手動覆蓋決策'
            
            if context.get('system_critical', False):
                if enhanced['decision'] == 'shutdown':
                    enhanced['original_decision'] = enhanced['decision']
                    enhanced['decision'] = 'notification'
                    enhanced['critical_reason'] = '檢測到系統關鍵進程'
        
        # 添加風險評估
        enhanced['risk_assessment'] = self._assess_decision_risk(enhanced)
        
        # 添加節能預估
        enhanced['energy_impact'] = self._estimate_energy_impact(enhanced)
        
        return enhanced

    def _assess_decision_risk(self, decision_result):
        """評估決策風險"""
        decision = decision_result.get('decision', 'delay')
        confidence = decision_result.get('confidence', 0.5)
        
        risk_factors = []
        risk_level = 'low'
        
        if decision == 'shutdown':
            if confidence < 0.7:
                risk_factors.append('低信心度關機可能造成用戶不便')
                risk_level = 'medium'
            
            habit_score = decision_result.get('input_scores', {}).get('habit_score', 0)
            if habit_score > 0.6:
                risk_factors.append('用戶習慣性使用強，不當關機風險高')
                risk_level = 'high'
        
        elif decision == 'keep_on':
            activity_score = decision_result.get('input_scores', {}).get('activity_score', 0)
            if activity_score < 0.3:
                risk_factors.append('低活躍度時保持開機浪費能源')
                risk_level = 'medium'
        
        if confidence < 0.3:
            risk_factors.append('決策信心度很低，結果不確定性高')
            risk_level = 'high'
        
        return {
            'level': risk_level,
            'factors': risk_factors,
            'score': 0.8 if risk_level == 'low' else 0.5 if risk_level == 'medium' else 0.2
        }

    def _estimate_energy_impact(self, decision_result):
        """估算能源影響"""
        decision = decision_result.get('decision', 'delay')
        
        # 假設功率數據（瓦特）
        power_consumption = {
            'active': 200,      # 活躍使用時功率
            'standby': 50,      # 待機功率
            'shutdown': 5       # 關機後功率
        }
        
        # 根據決策估算節能效果
        if decision == 'shutdown':
            saved_power = power_consumption['standby'] - power_consumption['shutdown']
            energy_impact = {
                'action': '關機節能',
                'power_saved_watts': saved_power,
                'estimated_daily_savings_kwh': saved_power * 24 / 1000,
                'co2_reduction_kg': (saved_power * 24 / 1000) * 0.5  # 假設碳排放係數
            }
        elif decision == 'keep_on':
            energy_impact = {
                'action': '保持開機',
                'power_saved_watts': 0,
                'estimated_daily_savings_kwh': 0,
                'co2_reduction_kg': 0
            }
        else:  # delay or notification
            # 假設50%機率最終關機
            saved_power = (power_consumption['standby'] - power_consumption['shutdown']) * 0.5
            energy_impact = {
                'action': '待觀察',
                'power_saved_watts': saved_power,
                'estimated_daily_savings_kwh': saved_power * 24 / 1000,
                'co2_reduction_kg': (saved_power * 24 / 1000) * 0.5
            }
        
        return energy_impact

    def _generate_action_plan(self, decision_result):
        """生成詳細行動計劃"""
        decision = decision_result.get('decision', 'delay')
        confidence = decision_result.get('confidence', 0.5)
        risk_level = decision_result.get('risk_assessment', {}).get('level', 'medium')
        
        action_plan = {
            'immediate_action': '',
            'follow_up_actions': [],
            'monitoring_required': False,
            'user_notification': False,
            'estimated_duration_minutes': 0,
            'success_criteria': []
        }
        
        if decision == 'shutdown':
            action_plan.update({
                'immediate_action': '執行系統關機程序',
                'follow_up_actions': [
                    '確認所有應用程序正常關閉',
                    '記錄關機時間和原因到日誌',
                    '下次開機時檢查系統狀態',
                    '統計實際節能效果'
                ],
                'user_notification': True,
                'estimated_duration_minutes': 2,
                'success_criteria': ['系統正常關機', '無數據丟失', '用戶接受決策']
            })
            
            if risk_level == 'high':
                action_plan['follow_up_actions'].insert(0, '由於高風險，建議用戶確認後再關機')
                
        elif decision == 'delay':
            action_plan.update({
                'immediate_action': '延遲決策，繼續監控系統狀態',
                'follow_up_actions': [
                    '15分鐘後重新評估',
                    '收集更多使用數據',
                    '監控用戶活動變化',
                    '記錄延遲決策的原因'
                ],
                'monitoring_required': True,
                'estimated_duration_minutes': 15,
                'success_criteria': ['獲得更多數據', '提高決策信心度']
            })
            
        elif decision == 'notification':
            action_plan.update({
                'immediate_action': '發送節能提醒通知給用戶',
                'follow_up_actions': [
                    '等待用戶響應（最多30分鐘）',
                    '記錄用戶響應行為',
                    '根據響應調整未來決策策略',
                    '若無響應則重新評估'
                ],
                'user_notification': True,
                'monitoring_required': True,
                'estimated_duration_minutes': 30,
                'success_criteria': ['用戶收到通知', '獲得用戶反饋']
            })
            
        else:  # keep_on
            action_plan.update({
                'immediate_action': '保持系統開機狀態',
                'follow_up_actions': [
                    '繼續正常運行',
                    '監控電源使用效率',
                    '記錄使用模式變化',
                    '定期重新評估決策'
                ],
                'estimated_duration_minutes': 60,
                'success_criteria': ['系統穩定運行', '用戶需求得到滿足']
            })
        
        # 根據信心度調整
        if confidence < 0.4:
            action_plan['follow_up_actions'].insert(0, '⚠️  低信心度：建議人工審核決策')
        
        return action_plan

    def _print_enhanced_summary(self, result):
        """打印增強版決策摘要"""
        print(f"\n📊 DECISION SUMMARY:")
        print(f"   🎯 決策: {result['decision'].upper()}")
        print(f"   🔒 信心度: {result['confidence']:.3f}")
        
        if 'risk_assessment' in result:
            risk = result['risk_assessment']
            risk_emoji = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}[risk['level']]
            print(f"   {risk_emoji} 風險等級: {risk['level'].upper()}")
        
        if 'energy_impact' in result:
            energy = result['energy_impact']
            print(f"   ⚡ 節能效果: {energy['power_saved_watts']}W")
        
        print(f"   📋 立即行動: {result['action_plan']['immediate_action']}")
        
        if result['action_plan']['user_notification']:
            print(f"   📨 需要用戶通知: 是")
        
        if result['action_plan']['monitoring_required']:
            print(f"   👀 需要持續監控: 是")

    def run_batch_analysis(self, start_time=None, duration_hours=24, 
                          interval_minutes=60, save_results=True):
        """
        運行批量決策分析
        
        Parameters:
        -----------
        start_time : datetime, optional
            開始時間，默認為今天0點
        duration_hours : int, default=24
            分析持續時間（小時）
        interval_minutes : int, default=60
            決策間隔時間（分鐘）
        save_results : bool, default=True
            是否保存結果到CSV文件
        
        Returns:
        --------
        list : 批量決策結果
        """
        if not self.system_ready:
            print("❌ 系統未就緒，無法進行批量分析")
            return None
        
        if start_time is None:
            start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        print(f"\n{'='*80}")
        print(f"🔄 BATCH ANALYSIS")
        print(f"⏰ Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Duration: {duration_hours} hours, Interval: {interval_minutes} minutes")
        print(f"{'='*80}")
        
        # 生成時間點
        timestamps = []
        current_time = start_time
        end_time = start_time + timedelta(hours=duration_hours)
        
        while current_time < end_time:
            timestamps.append(current_time)
            current_time += timedelta(minutes=interval_minutes)
        
        # 批量決策
        results = []
        for i, timestamp in enumerate(timestamps):
            if i % 10 == 0:
                print(f"Processing {i+1}/{len(timestamps)}...")
            
            result = self.make_power_decision(timestamp, verbose=False)
            if result:
                results.append(result)
        
        # 分析結果
        self._analyze_batch_results(results, start_time, duration_hours)
        
        # 繪製圖表
        self._plot_batch_analysis(results, start_time)
        
        # 保存結果
        if save_results:
            self._save_batch_results(results)
        
        return results

    def _analyze_batch_results(self, results, start_time, duration_hours):
        """分析批量結果"""
        print(f"\n📊 BATCH ANALYSIS RESULTS:")
        
        # 決策分布
        decisions = [r.get('decision', 'unknown') for r in results]
        decision_counts = {}
        for decision in decisions:
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
        
        total = len(results)
        print(f"\n🎯 Decision Distribution (Total: {total}):")
        for decision, count in decision_counts.items():
            percentage = count / total * 100
            emoji = {'shutdown': '🔴', 'delay': '🟠', 'notification': '🟡', 'keep_on': '🟢'}.get(decision, '❓')
            print(f"   {emoji} {decision.capitalize()}: {count} ({percentage:.1f}%)")
        
        # 節能估算
        total_energy_saved = sum([
            r.get('energy_impact', {}).get('power_saved_watts', 0) * (1/60)  # 轉換為分鐘
            for r in results
        ]) / 1000  # 轉換為kWh
        
        daily_energy_saved = total_energy_saved * (24 * 60) / (duration_hours * 60)
        
        print(f"\n💡 Energy Impact Estimation:")
        print(f"   Total energy saved: {total_energy_saved:.2f} kWh")
        print(f"   Daily energy saved (estimated): {daily_energy_saved:.2f} kWh")
        print(f"   Monthly savings (estimated): ${daily_energy_saved * 30 * 0.15:.2f}")  # 假設電費0.15$/kWh
        
        # 風險分析
        risk_distribution = {}
        for result in results:
            risk_level = result.get('risk_assessment', {}).get('level', 'unknown')
            risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
        
        print(f"\n⚠️  Risk Assessment:")
        for risk_level, count in risk_distribution.items():
            percentage = count / total * 100
            emoji = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}.get(risk_level, '❓')
            print(f"   {emoji} {risk_level.capitalize()} risk: {count} ({percentage:.1f}%)")

    def _plot_batch_analysis(self, results, start_time):
        """繪製批量分析圖表"""
        if len(results) < 2:
            print("❌ 數據不足，無法繪製圖表")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Integrated Power Management Analysis\n{start_time.strftime("%Y-%m-%d")}', 
                     fontsize=16, fontweight='bold')
        
        # 準備數據
        timestamps = [r['timestamp'] for r in results if 'timestamp' in r]
        decisions = [r['decision'] for r in results if 'decision' in r]
        confidences = [r.get('confidence', 0.5) for r in results]
        
        # 1. 決策時間線
        ax1 = axes[0, 0]
        decision_colors = {'shutdown': 'red', 'delay': 'orange', 'notification': 'yellow', 'keep_on': 'green'}
        decision_nums = [['shutdown', 'delay', 'notification', 'keep_on'].index(d) for d in decisions]
        
        scatter = ax1.scatter(timestamps, decision_nums, c=[decision_colors.get(d, 'gray') for d in decisions], 
                            s=100, alpha=0.7)
        ax1.set_ylabel('Decision Type')
        ax1.set_title('Decision Timeline')
        ax1.set_yticks(range(4))
        ax1.set_yticklabels(['Shutdown', 'Delay', 'Notification', 'Keep On'])
        ax1.grid(True, alpha=0.3)
        
        # 2. 信心度趨勢
        ax2 = axes[0, 1]
        ax2.plot(timestamps, confidences, 'b-', linewidth=2, alpha=0.7)
        ax2.fill_between(timestamps, confidences, alpha=0.3)
        ax2.set_ylabel('Decision Confidence')
        ax2.set_title('Confidence Trend')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # 3. 決策分布餅圖
        ax3 = axes[1, 0]
        decision_counts = {}
        for decision in decisions:
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
        
        colors = [decision_colors.get(d, 'gray') for d in decision_counts.keys()]
        ax3.pie(decision_counts.values(), labels=[d.capitalize() for d in decision_counts.keys()], 
               colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Decision Distribution')
        
        # 4. 節能效果
        ax4 = axes[1, 1]
        energy_saved = [r.get('energy_impact', {}).get('power_saved_watts', 0) for r in results]
        cumulative_energy = np.cumsum(energy_saved) / 1000  # 轉換為kWh
        
        ax4.plot(timestamps, cumulative_energy, 'g-', linewidth=3)
        ax4.fill_between(timestamps, cumulative_energy, alpha=0.3, color='green')
        ax4.set_ylabel('Cumulative Energy Saved (kWh)')
        ax4.set_title('Energy Savings Over Time')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def _save_batch_results(self, results):
        """保存批量結果"""
        try:
            df_data = []
            for result in results:
                if 'decision' in result:
                    df_data.append({
                        'timestamp': result['timestamp'],
                        'decision': result['decision'],
                        'confidence': result.get('confidence', 0.5),
                        'confidence_score': result.get('input_scores', {}).get('confidence_score', 0.5),
                        'activity_score': result.get('input_scores', {}).get('activity_score', 0.5),
                        'habit_score': result.get('input_scores', {}).get('habit_score', 0.5),
                        'risk_level': result.get('risk_assessment', {}).get('level', 'unknown'),
                        'power_saved_watts': result.get('energy_impact', {}).get('power_saved_watts', 0),
                        'immediate_action': result.get('action_plan', {}).get('immediate_action', ''),
                        'reasoning': result.get('reasoning', '')
                    })
            
            if df_data:
                df = pd.DataFrame(df_data)
                filename = f"integrated_power_decisions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(filename, index=False, encoding='utf-8-sig')
                print(f"✅ 結果已保存至: {filename}")
                
        except Exception as e:
            print(f"⚠️  保存結果時發生錯誤: {e}")

    def get_system_status(self):
        """獲取系統狀態報告"""
        print(f"\n{'='*80}")
        print("INTEGRATED POWER MANAGEMENT SYSTEM STATUS")
        print(f"{'='*80}")
        
        print(f"\n📋 MODULE STATUS:")
        modules = [
            ('Confidence Score Module', self.confidence_module),
            ('Device Activity Module', self.activity_module),
            ('User Habit Module', self.habit_module),
            ('Fuzzy Logic System', self.fuzzy_system)
        ]
        
        for name, module in modules:
            status = "✅ Ready" if module is not None else "❌ Not Available"
            print(f"   {name}: {status}")
        
        print(f"\n🎯 SYSTEM CAPABILITIES:")
        if self.system_ready:
            print("   ✅ Smart Power Decision Making")
            print("   ✅ Batch Analysis & Simulation")
            print("   ✅ Risk Assessment")
            print("   ✅ Energy Impact Estimation")
            print("   ✅ Action Plan Generation")
        else:
            print("   ❌ System not ready - missing required modules")
        
        print(f"\n⚙️  INITIALIZATION LOG:")
        for log in self.initialization_log:
            print(f"   {log}")
        
        print(f"\n🔧 SYSTEM READY: {'YES' if self.system_ready else 'NO'}")
        
        return {
            'system_ready': self.system_ready,
            'modules_available': sum([m is not None for _, m in modules]),
            'total_modules': len(modules),
            'initialization_log': self.initialization_log
        }


class FuzzyLogicPowerDecisionSystem:
    """
    模糊邏輯電源管理決策系統
    整合三個評分模組，使用模糊邏輯規則輸出最終決策
    """
    
    def __init__(self, confidence_module=None, activity_module=None, habit_module=None):
        self.confidence_module = confidence_module
        self.activity_module = activity_module
        self.habit_module = habit_module
        
        # 決策輸出選項
        self.decision_options = ['shutdown', 'delay', 'notification', 'keep_on']
        
        # 模糊邏輯規則
        self.fuzzy_rules = []
        self.membership_functions = {}
        
        # 系統配置
        self.decision_thresholds = {
            'shutdown': 0.25,
            'delay': 0.45,
            'notification': 0.65,
            'keep_on': 0.65
        }
        
        self._initialize_fuzzy_system()

    def _initialize_fuzzy_system(self):
        """初始化模糊邏輯系統"""
        # 定義輸入變數的模糊集合
        self.membership_functions = {
            'confidence': {
                'very_low': (0.0, 0.0, 0.2),
                'low': (0.0, 0.2, 0.4),
                'medium': (0.2, 0.4, 0.6),
                'high': (0.4, 0.6, 0.8),
                'very_high': (0.6, 0.8, 1.0)
            },
            'activity': {
                'very_low': (0.0, 0.0, 0.2),
                'low': (0.0, 0.2, 0.4),
                'medium': (0.2, 0.4, 0.6),
                'high': (0.4, 0.6, 0.8),
                'very_high': (0.6, 0.8, 1.0)
            },
            'habit': {
                'very_low': (0.0, 0.0, 0.2),
                'low': (0.0, 0.2, 0.4),
                'medium': (0.2, 0.4, 0.6),
                'high': (0.4, 0.6, 0.8),
                'very_high': (0.6, 0.8, 1.0)
            }
        }
        
        # 定義模糊邏輯規則
        self._define_fuzzy_rules()

    def triangular_membership(self, x, a, b, c):
        """三角隸屬函數"""
        if isinstance(x, (list, np.ndarray)):
            return np.array([self.triangular_membership(xi, a, b, c) for xi in x])
        
        if np.isnan(x) or x <= a or x >= c:
            return 0.0
        elif x == b:
            return 1.0
        elif a < x < b:
            return (x - a) / (b - a) if (b - a) > 0 else 0.0
        else:
            return (c - x) / (c - b) if (c - b) > 0 else 0.0

    def _define_fuzzy_rules(self):
        """定義模糊邏輯規則"""
        # 規則格式: (confidence_level, activity_level, habit_level, output_decision, weight)
        self.fuzzy_rules = [
            # 立即關機規則
            ('very_low', 'very_low', 'very_low', 'shutdown', 1.0),
            ('very_low', 'very_low', 'low', 'shutdown', 0.9),
            ('very_low', 'low', 'very_low', 'shutdown', 0.9),
            ('low', 'very_low', 'very_low', 'shutdown', 0.9),
            
            # 延遲決策規則
            ('low', 'low', 'low', 'delay', 0.8),
            ('very_low', 'medium', 'low', 'delay', 0.7),
            ('low', 'very_low', 'medium', 'delay', 0.7),
            ('medium', 'very_low', 'low', 'delay', 0.7),
            
            # 發送提醒規則
            ('medium', 'medium', 'low', 'notification', 0.7),
            ('medium', 'low', 'medium', 'notification', 0.7),
            ('low', 'medium', 'medium', 'notification', 0.7),
            ('medium', 'medium', 'medium', 'notification', 0.6),
            ('high', 'low', 'low', 'notification', 0.6),
            
            # 保持開機規則
            ('very_high', 'very_high', 'very_high', 'keep_on', 1.0),
            ('very_high', 'very_high', 'high', 'keep_on', 0.95),
            ('very_high', 'high', 'very_high', 'keep_on', 0.95),
            ('high', 'very_high', 'very_high', 'keep_on', 0.95),
            ('high', 'high', 'high', 'keep_on', 0.9),
        ]

    def calculate_membership_degrees(self, confidence_score, activity_score, habit_score):
        """計算隸屬度"""
        try:
            memberships = {'confidence': {}, 'activity': {}, 'habit': {}}
            
            for level, (a, b, c) in self.membership_functions['confidence'].items():
                memberships['confidence'][level] = self.triangular_membership(confidence_score, a, b, c)
            
            for level, (a, b, c) in self.membership_functions['activity'].items():
                memberships['activity'][level] = self.triangular_membership(activity_score, a, b, c)
            
            for level, (a, b, c) in self.membership_functions['habit'].items():
                memberships['habit'][level] = self.triangular_membership(habit_score, a, b, c)
            
            return memberships
            
        except Exception as e:
            print(f"⚠️  Error calculating memberships: {e}")
            return {
                'confidence': {'very_low': 0.5, 'low': 0.5, 'medium': 0.0, 'high': 0.0, 'very_high': 0.0},
                'activity': {'very_low': 0.5, 'low': 0.5, 'medium': 0.0, 'high': 0.0, 'very_high': 0.0},
                'habit': {'very_low': 0.5, 'low': 0.5, 'medium': 0.0, 'high': 0.0, 'very_high': 0.0}
            }

    def apply_fuzzy_rules(self, memberships):
        """應用模糊邏輯規則"""
        try:
            decision_activations = {decision: 0.0 for decision in self.decision_options}
            total_weights = {decision: 0.0 for decision in self.decision_options}
            
            for rule in self.fuzzy_rules:
                confidence_level, activity_level, habit_level, output_decision, rule_weight = rule
                
                confidence_membership = memberships['confidence'].get(confidence_level, 0.0)
                activity_membership = memberships['activity'].get(activity_level, 0.0)
                habit_membership = memberships['habit'].get(habit_level, 0.0)
                
                rule_activation = min(confidence_membership, activity_membership, habit_membership)
                weighted_activation = rule_activation * rule_weight
                
                decision_activations[output_decision] += weighted_activation
                total_weights[output_decision] += rule_weight
            
            normalized_activations = {}
            for decision in self.decision_options:
                if total_weights[decision] > 0:
                    normalized_activations[decision] = decision_activations[decision] / total_weights[decision]
                else:
                    normalized_activations[decision] = 0.0
            
            return normalized_activations
            
        except Exception as e:
            print(f"⚠️  Error applying fuzzy rules: {e}")
            return {decision: 0.25 for decision in self.decision_options}

    def defuzzify_decision(self, activations):
        """去模糊化，得出最終決策"""
        try:
            decision_values = {
                'shutdown': 0.1,
                'delay': 0.35,
                'notification': 0.65,
                'keep_on': 0.9
            }
            
            numerator = sum(activation * decision_values[decision] 
                          for decision, activation in activations.items())
            denominator = sum(activations.values())
            
            crisp_output = numerator / denominator if denominator > 0 else 0.5
            max_activation = max(activations.values())
            
            if crisp_output <= self.decision_thresholds['shutdown']:
                final_decision = 'shutdown'
            elif crisp_output <= self.decision_thresholds['delay']:
                final_decision = 'delay'
            elif crisp_output <= self.decision_thresholds['notification']:
                final_decision = 'notification'
            else:
                final_decision = 'keep_on'
            
            # 保守調整：低信心度時避免關機
            if max_activation < 0.3 and final_decision == 'shutdown':
                final_decision = 'delay'
            
            return {
                'final_decision': final_decision,
                'crisp_output': crisp_output,
                'confidence_level': max_activation,
                'all_activations': activations
            }
            
        except Exception as e:
            print(f"⚠️  Error in defuzzification: {e}")
            return {
                'final_decision': 'delay',
                'crisp_output': 0.5,
                'confidence_level': 0.1,
                'all_activations': activations
            }

    def make_power_decision(self, timestamp, verbose=True):
        """做出電源管理決策"""
        try:
            # 獲取三個模組的分數
            scores = self._get_module_scores(timestamp, verbose)
            
            # 計算隸屬度
            memberships = self.calculate_membership_degrees(
                scores['confidence_score'],
                scores['activity_score'], 
                scores['habit_score']
            )
            
            # 應用模糊邏輯規則
            activations = self.apply_fuzzy_rules(memberships)
            
            # 去模糊化得出決策
            decision_result = self.defuzzify_decision(activations)
            
            # 組合完整結果
            result = {
                'timestamp': timestamp,
                'input_scores': scores,
                'memberships': memberships,
                'rule_activations': activations,
                'decision': decision_result['final_decision'],
                'crisp_output': decision_result['crisp_output'],
                'confidence': decision_result['confidence_level'],
                'reasoning': self._generate_reasoning(scores, decision_result)
            }
            
            return result
            
        except Exception as e:
            print(f"⚠️  Error making power decision: {e}")
            return {
                'timestamp': timestamp,
                'decision': 'delay',
                'confidence': 0.1,
                'error': str(e)
            }

    def _get_module_scores(self, timestamp, verbose=True):
        """獲取三個模組的分數"""
        scores = {
            'confidence_score': 0.5,
            'activity_score': 0.5,
            'habit_score': 0.5
        }
        
        try:
            if self.confidence_module:
                confidence_result = self.confidence_module.calculate_confidence_score(timestamp)
                scores['confidence_score'] = confidence_result.get('confidence_score', 0.5)
                if verbose:
                    print(f"Confidence Score: {scores['confidence_score']:.3f}")
            
            if self.activity_module:
                activity_result = self.activity_module.calculate_activity_score(timestamp)
                scores['activity_score'] = activity_result.get('activity_score', 0.5)
                if verbose:
                    print(f"Activity Score: {scores['activity_score']:.3f}")
            
            if self.habit_module:
                habit_result = self.habit_module.calculate_habit_score(timestamp)
                scores['habit_score'] = habit_result.get('habit_score', 0.5)
                if verbose:
                    print(f"Habit Score: {scores['habit_score']:.3f}")
                    
        except Exception as e:
            print(f"⚠️  Error getting module scores: {e}")
        
        return scores

    def _generate_reasoning(self, scores, decision_result):
        """生成決策推理說明"""
        decision = decision_result['final_decision']
        confidence = scores['confidence_score']
        activity = scores['activity_score']
        habit = scores['habit_score']
        
        reasoning = []
        
        if confidence >= 0.8:
            reasoning.append("資料可靠性很高")
        elif confidence >= 0.6:
            reasoning.append("資料可靠性良好")
        elif confidence >= 0.4:
            reasoning.append("資料可靠性中等")
        else:
            reasoning.append("資料可靠性較低")
            
        if activity >= 0.7:
            reasoning.append("設備活躍度高")
        elif activity >= 0.4:
            reasoning.append("設備活躍度中等")
        else:
            reasoning.append("設備活躍度低")
            
        if habit >= 0.7:
            reasoning.append("用戶習慣性使用強")
        elif habit >= 0.4:
            reasoning.append("用戶習慣性使用中等")
        else:
            reasoning.append("用戶習慣性使用弱")
        
        if decision == 'shutdown':
            reasoning.append("→ 建議立即關機節能")
        elif decision == 'delay':
            reasoning.append("→ 建議延遲決策，持續觀察")
        elif decision == 'notification':
            reasoning.append("→ 建議發送提醒給用戶")
        else:
            reasoning.append("→ 建議保持開機狀態")
        
        return "; ".join(reasoning)


# ===================================
# 使用示例
# ===================================
if __name__ == "__main__":
    print("="*80)
    print("完整整合電源管理決策系統 - 物件導向版本")
    print("="*80)
    
    # 示例1: 手動注入模組（假設您已經有了初始化好的模組）
    print("\n📋 使用方式示例:")
    print("1. 手動注入已初始化的模組")
    print("2. 自動從數據文件初始化")
    print("3. 逐步設置各個模組")
    
    # 創建整合系統
    power_system = IntegratedPowerManagementSystem()
    
    # 假設您已經有了初始化好的模組實例
    # confidence_module = ConfidenceScoreModule()  # 您的信心分數模組
    # activity_module = DeviceActivityScoreModule()  # 您的活躍度模組
    # habit_module = ImprovedUserHabitScoreModule()  # 您的習慣模組
    
    # 注入模組
    # power_system.inject_modules(
    #     confidence_module=confidence_module,
    #     activity_module=activity_module,
    #     habit_module=habit_module
    # )
    
    # 示例2: 自動初始化（如果您有數據文件）
    data_file_path = "C:/Users/王俞文/Documents/glasgow/msc project/data/fuzzy_logic_control.py"
    power_system.auto_initialize_from_data(data_file_path)
    
    # 示例3: 測試決策（模擬模式）
    print("\n🧪 SIMULATION MODE TESTING...")
    
    # 創建一個簡化的模糊系統進行測試
    fuzzy_system = FuzzyLogicPowerDecisionSystem()
    
    # 測試時間點
    test_timestamps = [
        datetime(2024, 6, 15, 9, 0),   # 早上
        datetime(2024, 6, 15, 14, 30), # 下午
        datetime(2024, 6, 15, 21, 0),  # 晚上
        datetime(2024, 6, 15, 3, 0),   # 凌晨
    ]
    
    print("\n📊 模擬決策測試:")
    for timestamp in test_timestamps:
        result = fuzzy_system.make_power_decision(timestamp, verbose=False)
        if result:
            time_str = timestamp.strftime('%H:%M')
            decision = result['decision'].upper()
            confidence = result['confidence']
            print(f"   {time_str} -> {decision} (信心度: {confidence:.3f})")
    
    # 顯示系統狀態
    power_system.get_system_status()
    
    
    print("\n" + "="*80)
    print("物件導向整合系統準備完成！")
    print("="*80)