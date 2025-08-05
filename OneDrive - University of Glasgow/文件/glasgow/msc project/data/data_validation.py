import time
from datetime import datetime, timedelta
import warnings
import csv
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.patches import Rectangle
import seaborn as sns
from collections import deque
import requests

warnings.filterwarnings('ignore')
import fuzzy_logic_control
from fuzzy_logic_control import AntiOscillationFilter

from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

# ========================================
# 導入DecisionEvaluator模組
# ========================================
try:
    from decision_evaluator import DecisionEvaluator
    HAS_EVALUATOR = True
except ImportError:
    HAS_EVALUATOR = False

_decision_evaluator = None
_anti_oscillation_filter = None

# 震盪追蹤器
_oscillation_tracker = {
    'recent_decisions': deque(maxlen=20),
    'last_force_shutdown': None,
    'oscillation_count': 0
}

def get_octopus_latest_unit_rate():

    product_code = "VAR-22-11-01"
    tariff_code = "E-1R-VAR-22-11-01-K"
    
    url = f"https://api.octopus.energy/v1/products/{product_code}/electricity-tariffs/{tariff_code}/standard-unit-rates/"

    try:
        response = requests.get(url)
        data = response.json()

        for result in data['results']:
            if result['valid_from'] <= datetime.now().isoformat():
                unit_rate_pence = result['value_inc_vat']  # 單位: pence
                unit_rate_gbp = unit_rate_pence / 100       # 換成英鎊
                return round(unit_rate_gbp, 4)

    except Exception as e:
        print(f"⚠️ 無法抓取 Octopus 電價: {e}")
        return 0.30  # fallback 預設值

# 使用範例
uk_electricity_rate = get_octopus_latest_unit_rate()
print(f"✅ 使用最新英國電價：£{uk_electricity_rate}/kWh")

def init_decision_evaluator():
    """初始化決策評估器"""
    global _decision_evaluator
    
    if not HAS_EVALUATOR:
        return None
    
    if _decision_evaluator is None:
        _decision_evaluator = DecisionEvaluator(
            window_size_minutes=30,
            evaluation_interval_minutes=15,
            oscillation_detection_enabled=True,
            oscillation_window_minutes=10,
            min_oscillation_count=3,
            oscillation_threshold_ratio=0.4,
            auto_shutdown_enabled=True,
            shutdown_delay_minutes=1
        )
    
    return _decision_evaluator

def init_anti_oscillation_filter():
    """初始化防震盪濾波器"""
    global _anti_oscillation_filter
    
    if _anti_oscillation_filter is None:
        _anti_oscillation_filter = AntiOscillationFilter(
            hysteresis_enabled=True,
            phantom_threshold_low=20,
            phantom_threshold_high=30,
            decision_cooldown_seconds=30,
            min_state_duration_minutes=1,
            stability_check_enabled=False
        )
    
    return _anti_oscillation_filter

def _check_continuous_oscillation(timestamp, power_value):
    """檢查連續震盪並決定是否強制關機"""
    global _oscillation_tracker
    
    is_phantom = power_value < 36
    # is_phantom = power_value < 20
    current_state = 'phantom' if is_phantom else 'active'
    
    _oscillation_tracker['recent_decisions'].append({
        'timestamp': timestamp,
        'power': power_value,
        'state': current_state
    })
    
    if len(_oscillation_tracker['recent_decisions']) < 15:
        return {'should_force_shutdown': False, 'reason': '數據不足'}
    
    if (_oscillation_tracker['last_force_shutdown'] and 
        timestamp - _oscillation_tracker['last_force_shutdown'] < timedelta(hours=3)):
        return {'should_force_shutdown': False, 'reason': '距離上次強制關機太近'}
    
    recent_states = [d['state'] for d in _oscillation_tracker['recent_decisions']]
    
    # 計算狀態切換次數
    state_changes = 0
    for i in range(1, len(recent_states)):
        if recent_states[i] != recent_states[i-1]:
            state_changes += 1
    
    oscillation_conditions = []
    
    # 高頻狀態切換
    if state_changes > 10:
            oscillation_conditions.append(f"極高頻切換({state_changes}次)")
        
    # 🔧 更嚴格：15個決策中超過12個在臨界區間
    recent_powers = [d['power'] for d in _oscillation_tracker['recent_decisions']]
    power_near_threshold = sum(1 for p in recent_powers if 18 <= p <= 20)
    if power_near_threshold > 12:
        oscillation_conditions.append(f"功率長期在極窄臨界區間({power_near_threshold}/15)")
    
    # 🔧 需要滿足3個條件才強制關機
    should_shutdown = len(oscillation_conditions) >= 2 and state_changes > 8
    
    if should_shutdown:
        _oscillation_tracker['last_force_shutdown'] = timestamp
        _oscillation_tracker['oscillation_count'] += 1
        reason = "; ".join(oscillation_conditions)
        print(f"🚨 強制關機：{reason}")
        return {'should_force_shutdown': True, 'reason': reason}
    
    return {'should_force_shutdown': False, 'reason': '未達到強制關機條件'}

def estimate_predicted_power(actual_power, fuzzy_output):
    """估算預測功率值"""
    if fuzzy_output > 0.7:
        predicted_power = actual_power * 0.9
    elif fuzzy_output < 0.3:
        predicted_power = actual_power * 1.1
    else:
        predicted_power = actual_power
    
    noise = np.random.normal(0, actual_power * 0.05)
    predicted_power += noise
    return max(0, predicted_power)

def calculate_fuzzy_output(scores, power_value):
    """計算fuzzy控制器輸出"""
    if not scores:
        if power_value < PHANTOM_LOAD_THRESHOLD:
            return 0.8
        else:
            return 0.2
    
    activity_score = scores.get('activity', 0.5)
    habit_score = scores.get('habit', 0.5)
    confidence_score = scores.get('confidence', 0.5)
    
    activity_weight = 0.4
    habit_weight = 0.4
    confidence_weight = 0.2
    
    fuzzy_output = (
        activity_weight * (1 - activity_score) +
        habit_weight * (1 - habit_score) +
        confidence_weight * confidence_score
    )
    
    if power_value < PHANTOM_LOAD_THRESHOLD:
        fuzzy_output = min(1.0, fuzzy_output + 0.2)
    
    return np.clip(fuzzy_output, 0, 1)

def load_phantom_threshold():
    """Load phantom load threshold from config"""
    try:
        with open('config/thresholds.json', 'r') as f:
            data = json.load(f)
        return data['thresholds']['phantom_upper']
    except:
        return 36

# Configuration
PHANTOM_LOAD_THRESHOLD = 36

LOG_CONFIG = {
    'csv_file': 'recent_3days_phantom_load_analysis_log.csv',
    'json_file': 'recent_3days_phantom_load_detailed_log.json',
    'summary_file': 'recent_3days_analysis_summary.txt',
    'evaluation_file': 'decision_evaluation_log.csv'
}

# Import decision system
try:
    from fuzzy_logic_control import DecisionTreeSmartPowerAnalysis
    HAS_DECISION_SYSTEM = True
except ImportError:
    HAS_DECISION_SYSTEM = False

_decision_system = None

def init_logging():
    """Initialize log files"""
    os.makedirs('data', exist_ok=True)
    
    if not os.path.exists(LOG_CONFIG['csv_file']):
        with open(LOG_CONFIG['csv_file'], 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'row_number', 'data_time', 'power_w', 'type', 'action', 'decision',
                'activity_score', 'habit_score', 'confidence_score',
                'activity_level', 'habit_level', 'confidence_level',
                'professional_percentage', 'reason'
            ])
    
    if not os.path.exists(LOG_CONFIG['json_file']):
        with open(LOG_CONFIG['json_file'], 'w', encoding='utf-8') as jsonfile:
            json.dump([], jsonfile)

def load_data_from_file(file_path):
    """Load data from file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.csv':
        df = pd.read_csv(file_path)
    elif file_extension == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    elif file_extension in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    return df

def parse_data_row(row, row_number):
    """Parse data row"""
    power_columns = ['power', 'power_w', 'watt', 'watts', 'Power', 'POWER', '_value', 'value']
    power_value = None
    
    for col in power_columns:
        if col in row:
            power_value = row[col]
            break
    
    if power_value is None:
        for col in row.index:
            if pd.api.types.is_numeric_dtype(type(row[col])) and not pd.isna(row[col]):
                power_value = row[col]
                break
    
    if power_value is None or pd.isna(power_value):
        raise ValueError(f"Row {row_number}: No valid power value found")
    
    time_columns = ['time', 'timestamp', 'datetime', '_time', 'Time', 'DateTime']
    data_time = None
    
    for col in time_columns:
        if col in row and not pd.isna(row[col]):
            try:
                data_time = pd.to_datetime(row[col])
                break
            except:
                continue
    
    if data_time is None:
        data_time = datetime.now()
    
    return {
        'time': data_time,
        'value': float(power_value)
    }

def filter_recent_3_days(df, time_column='time'):
    """Filter data to keep only recent 3 days"""
    df[time_column] = pd.to_datetime(df[time_column])
    latest_date = df[time_column].max()
    three_days_ago = latest_date - timedelta(days=1)
    recent_df = df[df[time_column] >= three_days_ago].copy()
    
    return recent_df, latest_date, three_days_ago

def init_decision_system():
    """Initialize decision system"""
    global _decision_system
    
    if not HAS_DECISION_SYSTEM:
        return None
    
    if _decision_system is None:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            _decision_system = DecisionTreeSmartPowerAnalysis()
    
    return _decision_system

def is_phantom_load(power_value, threshold=PHANTOM_LOAD_THRESHOLD):
    """Check if it's phantom load"""
    return power_value < threshold

def make_phantom_decision(power_value, timestamp=None):
    """Make phantom load decision"""
    
    if not HAS_DECISION_SYSTEM:
        return {
            'decision': 'no_system',
            'action': 'No Decision System',
            'reason': 'Decision system module not found',
            'scores': {},
            'fuzzy_output': 0.5
        }
    
    system = init_decision_system()
    filter_system = init_anti_oscillation_filter()
    
    if system is None:
        return {
            'decision': 'system_error',
            'action': 'System Error',
            'reason': 'Decision system initialization failed',
            'scores': {},
            'fuzzy_output': 0.5
        }
    
    if timestamp is None:
        timestamp = datetime.now()
    
    opportunity = {
        'device_id': 'batch_analysis',
        'start_time': timestamp,
        'end_time': timestamp + timedelta(minutes=15),
        'power_watt': power_value
    }
    
    try:
        features = system._extract_enhanced_features(opportunity, None)
        
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            # Get scores
            if system.device_activity_model:
                try:
                    activity_result = system.device_activity_model.calculate_activity_score(timestamp)
                    activity_score = activity_result['activity_score']
                    activity_source = "Professional Model"
                except:
                    activity_score = system._fallback_activity_score(features, timestamp)
                    activity_source = "Fallback Method"
            else:
                activity_score = system._fallback_activity_score(features, timestamp)
                activity_source = "Fallback Method"
            
            if system.user_habit_model:
                try:
                    habit_result = system.user_habit_model.calculate_habit_score(timestamp)
                    habit_score = habit_result['habit_score']
                    habit_source = "Professional Model"
                except:
                    habit_score = system._fallback_habit_score(features, timestamp)
                    habit_source = "Fallback Method"
            else:
                habit_score = system._fallback_habit_score(features, timestamp)
                habit_source = "Fallback Method"
            
            if system.confidence_model:
                try:
                    confidence_result = system.confidence_model.calculate_confidence_score(timestamp)
                    confidence_score = confidence_result['confidence_score']
                    confidence_source = "Professional Model"
                except:
                    confidence_score = system._fallback_confidence_score(features, timestamp)
                    confidence_source = "Fallback Method"
            else:
                confidence_score = system._fallback_confidence_score(features, timestamp)
                confidence_source = "Fallback Method"
            
            # Make decision
            decision, debug_info = system._make_intelligent_decision(
                activity_score, habit_score, confidence_score, features
            )

            # Apply anti-oscillation filter
            filter_result = _anti_oscillation_filter.filter_decision(
                original_decision=decision,
                power_value=power_value,
                timestamp=timestamp,
                scores={'activity': activity_score, 'habit': habit_score, 'confidence': confidence_score}
            )

            final_decision = filter_result['filtered_decision']

            # Check for continuous oscillation and force shutdown if needed
            oscillation_info = _check_continuous_oscillation(timestamp, power_value)
            
            if oscillation_info['should_force_shutdown']:
                final_decision = 'suggest_shutdown'

            debug_info['filter_applied'] = filter_result['should_use_filtered']
            debug_info['filter_reason'] = filter_result['filter_reason']
            debug_info['power_state'] = filter_result['power_state']
            debug_info['original_decision'] = decision
        
        english_actions = {
            'suggest_shutdown': 'Suggest Shutdown',
            'send_notification': 'Send Notification',
            'keep_on': 'Keep On',
            'delay_decision': 'Delay Decision'
        }
        
        professional_count = sum([
            1 if source == "Professional Model" else 0 
            for source in [activity_source, habit_source, confidence_source]
        ])
        professional_percentage = (professional_count / 3) * 100
        
        final_action = english_actions.get(final_decision, final_decision)
        
        scores_dict = {
            'activity': activity_score,
            'habit': habit_score,
            'confidence': confidence_score
        }
        fuzzy_output = calculate_fuzzy_output(scores_dict, power_value)
        
        return {
            'decision': final_decision,
            'action': final_action,
            'scores': scores_dict,
            'levels': {
                'activity': debug_info.get('device_activity_level', '?'),
                'habit': debug_info.get('user_habit_level', '?'),
                'confidence': debug_info.get('confidence_score_level', '?')
            },
            'decision_path': debug_info.get('decision_path', ''),
            'professional_percentage': professional_percentage,
            'model_sources': {
                'activity': activity_source,
                'habit': habit_source,
                'confidence': confidence_source
            },
            'fuzzy_output': fuzzy_output
        }
        
    except Exception as e:
        return {
            'decision': 'error',
            'action': 'Decision Error',
            'reason': str(e),
            'scores': {},
            'fuzzy_output': 0.5
        }

def process_power_data(data, row_number):
    """Process power data and make smart control decisions"""
    
    if not data:
        return None
    
    power_value = data['value']
    data_time = data['time']
    
    if not is_phantom_load(power_value, PHANTOM_LOAD_THRESHOLD):
        result = {
            'type': 'normal_usage',
            'power': power_value,
            'data_time': data_time,
            'action': 'No Action Needed',
            'decision': 'normal_usage',
            'reason': f'Power {power_value}W ≥ {PHANTOM_LOAD_THRESHOLD}W, normal usage range',
            'fuzzy_output': 0.1,
            'scores': {}
        }
    else:
        decision_result = make_phantom_decision(power_value, data_time)
        
        result = {
            'type': 'phantom_load',
            'power': power_value,
            'data_time': data_time,
            'action': decision_result['action'],
            'decision': decision_result['decision'],
            'scores': decision_result.get('scores', {}),
            'levels': decision_result.get('levels', {}),
            'professional_percentage': decision_result.get('professional_percentage', 0),
            'model_sources': decision_result.get('model_sources', {}),
            'reason': decision_result.get('reason', ''),
            'fuzzy_output': decision_result.get('fuzzy_output', 0.5)
        }
    
    if HAS_EVALUATOR and _decision_evaluator is not None:
        try:
            predicted_power = estimate_predicted_power(
                power_value, 
                result.get('fuzzy_output', 0.5)
            )
            
            _decision_evaluator.add_decision_record(
                timestamp=data_time,
                fuzzy_output=result.get('fuzzy_output', 0.5),
                predicted_power=predicted_power,
                actual_power=power_value,
                decision=result['decision'],
                confidence_scores=result.get('scores', {})
            )
        except:
            pass
    
    return result

def log_result_to_csv(result, row_number):
    """Log result to CSV file"""
    if not result:
        return
    
    row_data = [
        row_number,
        result['data_time'].strftime("%Y-%m-%d %H:%M:%S") if result['data_time'] else None,
        result['power'],
        result['type'],
        result['action'],
        result.get('decision', ''),
        result.get('scores', {}).get('activity', ''),
        result.get('scores', {}).get('habit', ''),
        result.get('scores', {}).get('confidence', ''),
        result.get('levels', {}).get('activity', ''),
        result.get('levels', {}).get('habit', ''),
        result.get('levels', {}).get('confidence', ''),
        result.get('professional_percentage', ''),
        result.get('reason', '')
    ]
    
    with open(LOG_CONFIG['csv_file'], 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row_data)

def plot_power_analysis_results(df_all, results_df):
    """Plot power analysis results with decision-based background colors"""
    
    plt.figure(figsize=(16, 10))
    
    decision_colors = {
        'suggest_shutdown': '#ff6b6b',
        'send_notification': '#ffd93d',
        'keep_on': '#6bcf7f',
        'delay_decision': '#4ecdc4',
        'normal_usage': '#95a5a6',
        'no_system': '#ff9f43',
        'system_error': '#e056fd',
        'error': '#e056fd'
    }
    
    decision_labels = {
        'suggest_shutdown': 'Shutdown',
        'send_notification': 'Send Notification',
        'keep_on': 'Keep On',
        'delay_decision': 'Delay Decision',
        'normal_usage': 'Normal Usage',
        'no_system': 'No System',
        'system_error': 'System Error',
        'error': 'Error'
    }
    
    results_df['data_time'] = pd.to_datetime(results_df['data_time'])
    results_df = results_df.sort_values('data_time')
    
    ax = plt.gca()
    
    # Group consecutive same decisions
    decision_groups = []
    current_decision = None
    current_start = None
    
    for i, row in results_df.iterrows():
        decision = row['decision']
        time_point = row['data_time']
        
        if decision != current_decision:
            if current_decision is not None:
                decision_groups.append({
                    'decision': current_decision,
                    'start': current_start,
                    'end': time_point
                })
            
            current_decision = decision
            current_start = time_point
    
    if current_decision is not None:
        decision_groups.append({
            'decision': current_decision,
            'start': current_start,
            'end': results_df['data_time'].iloc[-1]
        })
    
    # Plot background colors
    y_min, y_max = 0, results_df['power'].max() * 1.1
    
    for group in decision_groups:
        color = decision_colors.get(group['decision'], '#95a5a6')
        width = (group['end'] - group['start']).total_seconds() / 3600 / 24
        
        rect = Rectangle((mdates.date2num(group['start']), y_min), 
                        width, y_max - y_min, 
                        facecolor=color, alpha=0.6, edgecolor='none')
        ax.add_patch(rect)
    
    # 修改這裡：加上 marker 參數來顯示數據點
    plt.plot(results_df['data_time'], results_df['power'], 
             linewidth=2, color="#000000", alpha=0.8, 
             marker='o', markersize=4, markerfacecolor="#FFFFFF", 
             markeredgecolor='black', markeredgewidth=2,
             label='Power Consumption')
    
    # plt.plot(results_df['data_time'], results_df['power'], 
    #          linewidth=2, color="#000000", alpha=0.8, 
    #          marker='o', markersize=3, markerfacecolor="#737171", 
    #          label='Power Consumption')
    
    # plt.plot(results_df['data_time'], results_df['power'], 
    #          linewidth=2, color="#000000", alpha=0.8, 
    #          label='Power Consumption')
    
    plt.axhline(y=PHANTOM_LOAD_THRESHOLD, color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label=f'Phantom Load Threshold ({PHANTOM_LOAD_THRESHOLD}W)')
    
    plt.xlabel('Time', fontsize=24, fontweight='bold')
    plt.ylabel('Power (W)', fontsize=24, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=22)
    # plt.title('Power Consumption Analysis - Recent 1 Day\nwith Decision-based Background Colors', 
    #           fontsize=14, fontweight='bold', pad=20)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.xticks(rotation=0)
    
    # Add legend
    legend_elements = []
    used_decisions = results_df['decision'].unique()
    
    for decision in used_decisions:
        if decision in decision_colors:
            color = decision_colors[decision]
            label = decision_labels.get(decision, decision.replace('_', ' ').title())
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.6, label=label))
    
    legend_elements.append(plt.Line2D([0], [0], color="#000000", linewidth=2, 
                                    marker='o', markersize=4, label='Power Consumption'))
    legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Phantom Load Threshold'))
    
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
           fontsize=24, ncol=3, frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # plot_filename = f'power_analysis_recent_1day_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    # plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print statistics
    print("\n=== Decision Statistics ===")
    decision_counts = results_df['decision'].value_counts()
    total_decisions = len(results_df)
    
    for decision, count in decision_counts.items():
        percentage = (count / total_decisions) * 100
        display_label = decision_labels.get(decision, decision.replace('_', ' ').title())
        print(f"{display_label}: {count} ({percentage:.1f}%)")

def calculate_energy_saving_from_results(results_df, df_all):
    """計算節能效果"""
    
    # uk_electricity_rate = 0.30
    uk_electricity_rate = get_octopus_latest_unit_rate()
    
    # 使用實際處理的數據天數
    results_df['data_time'] = pd.to_datetime(results_df['data_time'])
    start_date = results_df['data_time'].min()
    end_date = results_df['data_time'].max()
    actual_days = (end_date - start_date).days + 1
    
    print(f"\n" + "="*60)
    print(f"🔋 原本總耗能 vs 智能系統節省對比分析")
    print(f"📅 分析期間：{actual_days} 天（{start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}）")
    print(f"⚡ 英國電價：£{uk_electricity_rate}/kWh")
    print(f"="*60)
    
    phantom_results = results_df[results_df['type'] == 'phantom_load'].copy()
    
    if len(phantom_results) == 0:
        print("❌ 沒有 phantom load 數據可分析")
        return
    
    # 計算數據間隔
    total_records = len(results_df)
    total_hours = actual_days * 24
    avg_interval_hours = total_hours / total_records if total_records > 0 else 0.25
    
    # ========================================
    # 1. 計算原本總耗能（假設沒有智能系統介入）
    # ========================================
    original_total_kwh = 0
    
    for _, row in phantom_results.iterrows():
        power_w = row['power']
        energy_kwh = power_w * avg_interval_hours / 1000
        original_total_kwh += energy_kwh
    
    original_daily_kwh = original_total_kwh / actual_days
    original_total_cost = original_total_kwh * uk_electricity_rate
    original_daily_cost = original_daily_kwh * uk_electricity_rate
    
    print(f"📊 原本總耗能（無智能系統介入）：")
    print(f"   🔋 總電量：{original_total_kwh:.3f} kWh")
    print(f"   💰 總電費：£{original_total_cost:.3f}")
    print(f"   📅 日均電量：{original_daily_kwh:.3f} kWh/天")
    print(f"   💷 日均電費：£{original_daily_cost:.3f}/天")
    
    # ========================================
    # 2. 計算各決策類型的實際耗能
    # ========================================
    decision_breakdown = {
        'suggest_shutdown': {'count': 0, 'kwh': 0},      # 確定節省
        'send_notification': {'count': 0, 'kwh': 0},     # 潛在節省
        'keep_on': {'count': 0, 'kwh': 0},              # 繼續耗能
        'delay_decision': {'count': 0, 'kwh': 0}         # 延遲決策（保守耗能）
    }
    
    for _, row in phantom_results.iterrows():
        power_w = row['power']
        decision = row['decision']
        energy_kwh = power_w * avg_interval_hours / 1000
        
        if decision in decision_breakdown:
            decision_breakdown[decision]['count'] += 1
            decision_breakdown[decision]['kwh'] += energy_kwh
    
    print(f"\n📊 智能系統決策分析：")
    for decision, data in decision_breakdown.items():
        if data['count'] > 0:
            percentage = (data['kwh'] / original_total_kwh * 100)
            daily_kwh = data['kwh'] / actual_days
            daily_cost = daily_kwh * uk_electricity_rate
            
            if decision == 'suggest_shutdown':
                print(f"   🔴 確定關機：{data['count']} 次, {data['kwh']:.3f} kWh ({percentage:.1f}%) → 節省電費")
            elif decision == 'send_notification':
                print(f"   🟡 發送通知：{data['count']} 次, {data['kwh']:.3f} kWh ({percentage:.1f}%) → 潛在節省")
            elif decision == 'keep_on':
                print(f"   🟢 保持開機：{data['count']} 次, {data['kwh']:.3f} kWh ({percentage:.1f}%) → 繼續耗能")
            elif decision == 'delay_decision':
                print(f"   🔵 延遲決策：{data['count']} 次, {data['kwh']:.3f} kWh ({percentage:.1f}%) → 保守耗能")
    
    # ========================================
    # 3. 計算確定節省效果
    # ========================================
    definite_saved_kwh = decision_breakdown['suggest_shutdown']['kwh']
    definite_saved_cost = definite_saved_kwh * uk_electricity_rate
    
    # 計算實際耗能（扣除確定節省）
    actual_consumed_kwh = original_total_kwh - definite_saved_kwh
    actual_consumed_cost = actual_consumed_kwh * uk_electricity_rate
    
    print(f"\n✅ 確定節省效果（suggest_shutdown）：")
    print(f"   💡 確定節省電量：{definite_saved_kwh:.3f} kWh")
    print(f"   💰 確定節省電費：£{definite_saved_cost:.3f}")
    print(f"   📈 節能率：{(definite_saved_kwh/original_total_kwh*100):.1f}%")
    
    # ========================================
    # 4. 原本 vs 節省後對比
    # ========================================
    print(f"\n" + "="*50)
    print(f"💡 【原本 vs 智能系統後】對比")
    print(f"="*50)
    print(f"📊 期間總耗能對比：")
    print(f"   🔴 原本總耗能：    {original_total_kwh:.3f} kWh (£{original_total_cost:.3f})")
    print(f"   🟢 智能系統後耗能：{actual_consumed_kwh:.3f} kWh (£{actual_consumed_cost:.3f})")
    print(f"   💚 確定節省：      {definite_saved_kwh:.3f} kWh (£{definite_saved_cost:.3f})")
    print(f"   📉 節能比例：      {(definite_saved_kwh/original_total_kwh*100):.1f}%")
    
    print(f"\n📅 每日平均對比：")
    daily_saved_kwh = definite_saved_kwh / actual_days
    daily_saved_cost = daily_saved_kwh * uk_electricity_rate
    daily_after_kwh = original_daily_kwh - daily_saved_kwh
    daily_after_cost = daily_after_kwh * uk_electricity_rate
    
    print(f"   🔴 原本日均耗能：  {original_daily_kwh:.3f} kWh/天 (£{original_daily_cost:.3f}/天)")
    print(f"   🟢 智能系統後日均：{daily_after_kwh:.3f} kWh/天 (£{daily_after_cost:.3f}/天)")
    print(f"   💚 每日節省：      {daily_saved_kwh:.3f} kWh/天 (£{daily_saved_cost:.3f}/天)")
    
    # ========================================
    # 5. 年度效益預估
    # ========================================
    annual_original_kwh = original_daily_kwh * 365
    annual_original_cost = annual_original_kwh * uk_electricity_rate
    annual_saved_kwh = daily_saved_kwh * 365
    annual_saved_cost = annual_saved_kwh * uk_electricity_rate
    annual_after_kwh = annual_original_kwh - annual_saved_kwh
    annual_after_cost = annual_after_kwh * uk_electricity_rate
    
    print(f"\n📅 年度效益預估：")
    print(f"   🔴 原本年度耗能：  {annual_original_kwh:.1f} kWh/年 (£{annual_original_cost:.0f}/年)")
    print(f"   🟢 智能系統後年度：{annual_after_kwh:.1f} kWh/年 (£{annual_after_cost:.0f}/年)")
    print(f"   💚 年度節省：      {annual_saved_kwh:.1f} kWh/年 (£{annual_saved_cost:.0f}/年)")
    print(f"   📈 年度節能率：    {(annual_saved_kwh/annual_original_kwh*100):.1f}%")
    
    # ========================================
    # 6. 潛在節省效果（Send Notification）
    # ========================================
    potential_saved_kwh = decision_breakdown['send_notification']['kwh']
    
    if potential_saved_kwh > 0:
        print(f"\n🔔 潛在節省效果（Send Notification）：")
        print(f"   📬 通知次數：{decision_breakdown['send_notification']['count']} 次")
        print(f"   ⚡ 涉及電量：{potential_saved_kwh:.3f} kWh")
        print(f"   💰 涉及電費：£{potential_saved_kwh * uk_electricity_rate:.3f}")
        
        # 不同用戶響應率的效果
        print(f"\n   📈 不同用戶響應率的總節省效果：")
        for response_rate in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]:
            additional_saved = potential_saved_kwh * response_rate
            total_saved = definite_saved_kwh + additional_saved
            total_saved_cost = total_saved * uk_electricity_rate
            total_after = original_total_kwh - total_saved
            total_after_cost = total_after * uk_electricity_rate
            total_saving_rate = (total_saved / original_total_kwh * 100)
            
            print(f"     🎯 {int(response_rate*100)}%響應率:")
            print(f"        總節省: {total_saved:.3f} kWh (£{total_saved_cost:.3f}) | 節能率: {total_saving_rate:.1f}%")
            print(f"        剩餘耗能: {total_after:.3f} kWh (£{total_after_cost:.3f})")
    
    # ========================================
    # 7. 相對於家庭總電費的節省比例
    # ========================================
    print(f"\n📊 相對於不同家庭年度電費的節省比例：")
    household_types = {
        '🏠 中型家庭': 1050,
        '🇬🇧 英國平均': 1200, 
        '🏢 大型家庭': 1500,
        '🏘️ 小型家庭': 800
    }
    
    for household_type, annual_total_cost in household_types.items():
        saving_percentage = (annual_saved_cost / annual_total_cost * 100)
        phantom_percentage = (annual_original_cost / annual_total_cost * 100)
        print(f"   {household_type} (£{annual_total_cost}/年):")
        print(f"      Phantom Load佔比: {phantom_percentage:.1f}% | 節省佔比: {saving_percentage:.2f}%")
    
    # ========================================
    # 8. 環境效益
    # ========================================
    co2_factor = 0.233  # kg CO2 per kWh in UK
    annual_co2_saved = annual_saved_kwh * co2_factor
    cars_equivalent = annual_co2_saved / 4600  # 平均汽車年排放4.6噸CO2
    
    print(f"\n🌱 環境效益：")
    print(f"   🌍 年度減少 CO₂ 排放：{annual_co2_saved:.1f} kg")
    print(f"   🚗 相當於減少汽車排放：{cars_equivalent:.3f} 輛/年")
    
    # ========================================
    # 9. 生活化效益比較
    # ========================================
    print(f"\n⚡ 年度節省生活化比較：")
    print(f"   📅 相當於 {(annual_saved_cost/(1200/365)):.0f} 天的免費電力")
    print(f"   ☕ 相當於 {(annual_saved_cost/3.5):.0f} 杯咖啡")
    print(f"   📺 相當於 {(annual_saved_cost/10.99):.1f} 個月的Netflix訂閱")
    print(f"   🍎 相當於 {(annual_saved_cost/2.0):.0f} 個蘋果")
    
    print(f"\n" + "="*60)
    
    return {
        'original_total_kwh': original_total_kwh,
        'definite_saved_kwh': definite_saved_kwh,
        'actual_consumed_kwh': actual_consumed_kwh,
        'annual_saved_cost': annual_saved_cost,
        'saving_rate': (definite_saved_kwh/original_total_kwh*100),
        'decision_breakdown': decision_breakdown
    }


def update_summary_file(total_count, phantom_count, normal_count, file_path, results_df):
    """Update summary file"""
    total_valid = phantom_count + normal_count
    phantom_rate = (phantom_count / total_valid * 100) if total_valid > 0 else 0
    
    decision_labels = {
        'suggest_shutdown': 'Shutdown',
        'send_notification': 'Send Notification',
        'keep_on': 'Keep On',
        'delay_decision': 'Delay Decision',
        'normal_usage': 'Normal Usage',
        'no_system': 'No System',
        'system_error': 'System Error',
        'error': 'Error'
    }
    
    decision_stats = results_df['decision'].value_counts().to_dict()
    
    summary_content = f"""Recent 3 Days Phantom Load Analysis System - Summary
=======================================================
Analysis Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Source File: {file_path}
Analysis Period: Recent 3 days

Analysis Statistics:
- Total Processed Records: {total_count}
- Valid Data Records: {total_valid}
- Phantom Load Detected: {phantom_count} records ({phantom_rate:.1f}%)
- Normal Usage Detected: {normal_count} records ({100-phantom_rate:.1f}%)

Decision Statistics:
"""
    
    for decision, count in decision_stats.items():
        percentage = (count / total_count) * 100
        display_label = decision_labels.get(decision, decision.replace('_', ' ').title())
        summary_content += f"- {display_label}: {count} records ({percentage:.1f}%)\n"
    
    if HAS_EVALUATOR and _decision_evaluator is not None:
        evaluation_summary = _decision_evaluator.get_evaluation_summary()
        if 'average_scores' in evaluation_summary:
            summary_content += f"""
Decision Evaluation Summary:
- Evaluation Count: {evaluation_summary['evaluation_count']}
- Average Stability Score: {evaluation_summary['average_scores']['stability']:.3f}
- Average Consistency Score: {evaluation_summary['average_scores']['consistency']:.3f}
- Average Accuracy Score: {evaluation_summary['average_scores']['accuracy']:.3f}
- Overall Score: {evaluation_summary['average_scores']['overall']:.3f}
- Performance Trend: {evaluation_summary['trend']['trend']}
"""
    
    summary_content += f"""
System Configuration:
- Phantom Load Threshold: {PHANTOM_LOAD_THRESHOLD}W
- Decision System Status: {'Available' if HAS_DECISION_SYSTEM else 'Unavailable'}
- Decision Evaluator Status: {'Available' if HAS_EVALUATOR else 'Unavailable'}

Output Files:
- CSV Record: {LOG_CONFIG['csv_file']}
- Detailed Record: {LOG_CONFIG['json_file']}
- Summary Record: {LOG_CONFIG['summary_file']}
"""
    
    if HAS_EVALUATOR:
        summary_content += f"- Evaluation Record: {LOG_CONFIG['evaluation_file']}\n"
    
    with open(LOG_CONFIG['summary_file'], 'w', encoding='utf-8') as f:
        f.write(summary_content)

def main(input_file_path):
    """Main program - Recent 3 days data analysis and smart control"""
    
    init_logging()
    init_decision_evaluator()
    init_anti_oscillation_filter()
    
    print("Loading dataset...")
    df_all = load_data_from_file(input_file_path)
    
    if HAS_DECISION_SYSTEM:
        print("Initializing decision system...")
        init_decision_system()
    
    try:
        time_columns = ['time', 'timestamp', 'datetime', '_time', 'Time', 'DateTime']
        time_col = None
        
        for col in time_columns:
            if col in df_all.columns:
                time_col = col
                break
        
        if time_col is None:
            raise ValueError("No time column found in the dataset")
        
        print("Filtering to recent 3 days...")
        df_recent, latest_date, three_days_ago = filter_recent_3_days(df_all, time_col)
        
        if len(df_recent) == 0:
            print("No data found in recent 3 days!")
            return
        
        total_rows = len(df_recent)
        phantom_count = 0
        normal_count = 0
        error_count = 0
        
        results_list = []
        
        print(f"Processing {total_rows} records...")
        for index, row in df_recent.iterrows():
            row_number = index + 1
            
            try:
                data = parse_data_row(row, row_number)
                result = process_power_data(data, row_number)
                
                if result:
                    log_result_to_csv(result, row_number)
                    
                    results_list.append({
                        'data_time': result['data_time'],
                        'power': result['power'],
                        'type': result['type'],
                        'action': result['action'],
                        'decision': result['decision']
                    })
                    
                    if result['type'] == 'phantom_load':
                        phantom_count += 1
                    elif result['type'] == 'normal_usage':
                        normal_count += 1
                    
            except Exception as e:
                error_count += 1
        
        results_df = pd.DataFrame(results_list)
        
        if HAS_EVALUATOR and _decision_evaluator is not None:
            try:
                evaluation_file = _decision_evaluator.export_detailed_report()
                if evaluation_file:
                    print(f"✅ 評估結果已匯出: {evaluation_file}")
            except:
                pass
        
        update_summary_file(total_rows, phantom_count, normal_count, input_file_path, results_df)
        
        if len(results_df) > 0:
            print("Generating visualization...")
            plot_power_analysis_results(df_all, results_df)
            
            # 節能分析
            energy_analysis = calculate_energy_saving_from_results(results_df, df_all)
            # create_energy_comparison_visualization(results_df, energy_analysis)
        
        print(f"\n✅ Analysis completed!")
        print(f"Period: {three_days_ago.strftime('%Y-%m-%d %H:%M:%S')} to {latest_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Processed: {total_rows} records")
        print(f"Phantom Load: {phantom_count} records")
        print(f"Normal Usage: {normal_count} records")
        print(f"Errors: {error_count} records")
        
        # 顯示震盪統計
        print(f"\n🔧 Anti-Oscillation Statistics:")
        print(f"   Forced shutdowns: {_oscillation_tracker['oscillation_count']}")
        
        if HAS_EVALUATOR and _decision_evaluator is not None:
            print(f"   Decision evaluations: {len(_decision_evaluator.evaluation_results)}")
        
    except Exception as e:
        print(f"❌ System Error: {e}")
        raise

if __name__ == "__main__":
    input_file = "C:/Users/王俞文/OneDrive - University of Glasgow/文件/glasgow/msc project/data/complete_power_data_with_history.csv"
    main(input_file)