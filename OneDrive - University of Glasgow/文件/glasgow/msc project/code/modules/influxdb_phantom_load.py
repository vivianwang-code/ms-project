from influxdb_client import InfluxDBClient
import time
from datetime import datetime, timedelta
import warnings
import csv
import os
import json
warnings.filterwarnings('ignore')
from . import fuzzy_logic_control
from .fuzzy_logic_control import AntiOscillationFilter

try:
    from .fuzzy_logic_control import init_decision_evaluator, _decision_evaluator
    HAS_EVALUATOR = True
except ImportError:
    HAS_EVALUATOR = False
    print("⚠️ DecisionEvaluator module not found")

from utils.logger_config import get_phantom_logger, get_error_logger


from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

# InfluxDB Configuration
INFLUX_CONFIG = {
    'url': "http://aties-digital-twin.ddns.net:1003",
    'token': "VTPT42ftuyixhrddaurUEnBWeVA3vBiZONqf5eDCADAUc-8LZSEoLKJdt98oshQp6ZM7l0HQFsdrzIOnI6-11A==",
    'org': "myorg",
    'bucket': "iotproject",
    'measurement': "influxdb-JWN-D8",
    'field': "power"
}

phantom_logger = get_phantom_logger()
error_logger = get_error_logger()


def load_phantom_threshold():
    """Load phantom load threshold from config/thresholds.json"""
    try:
        with open('config/thresholds.json', 'r') as f:
            data = json.load(f)
        
        phantom_threshold = data['thresholds']['phantom_upper']
        phantom_logger.info(f"Successfully loaded Phantom Load Threshold: {phantom_threshold}W")
        print(f"✅ Successfully loaded Phantom Load Threshold: {phantom_threshold}W")
        return phantom_threshold
        
    except FileNotFoundError:
        error_logger.warning("config/thresholds.json not found, using default value")
        print("❌ config/thresholds.json not found, using default value 36.9W")
        print("💡 Please run K-means analysis first to generate threshold configuration")
        return 36.9  # Default value
    except KeyError as e:
        error_logger.error(f"config/thresholds.json format error: {e}")
        print(f"❌ config/thresholds.json format error, missing field: {e}")
        print("💡 Please re-run K-means analysis")
        return 36.9  # Default value
    except Exception as e:
        error_logger.error(f"Error occurred while reading threshold: {e}")
        print(f"❌ Error occurred while reading threshold: {e}")
        return 36.9  # Default value
    

# Phantom Load Configuration
PHANTOM_LOAD_THRESHOLD = load_phantom_threshold()  # Below this value considered as phantom load
MONITOR_INTERVAL = 900  # Monitoring interval (seconds), 15 minutes = 900 seconds

# Logging Configuration
LOG_CONFIG = {
    'csv_file': 'data/phantom_load_monitoring_log.csv',
    'json_file': 'data/phantom_load_detailed_log.json',
    'summary_file': 'data/monitoring_summary.txt'
}

# Import decision system
try:
    from .fuzzy_logic_control import DecisionTreeSmartPowerAnalysis
    HAS_DECISION_SYSTEM = True
except ImportError:
    HAS_DECISION_SYSTEM = False
    print("⚠️ Decision system module not found, only data monitoring function available")

# Global decision system
_decision_system = None



def init_logging():
    """Initialize logging files"""

    os.makedirs('data', exist_ok=True)
    # Create CSV header if file doesn't exist
    if not os.path.exists(LOG_CONFIG['csv_file']):
        with open(LOG_CONFIG['csv_file'], 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'monitor_time', 'data_time', 'power_w', 'type', 'action', 'decision',
                'activity_score', 'habit_score', 'confidence_score',
                'activity_level', 'habit_level', 'confidence_level',
                'professional_percentage', 'data_age_minutes', 'reason'
            ])
    
    # Create JSON log file if doesn't exist
    if not os.path.exists(LOG_CONFIG['json_file']):
        with open(LOG_CONFIG['json_file'], 'w', encoding='utf-8') as jsonfile:
            json.dump([], jsonfile)
    
    print(f"📝 Logging files initialized successfully:")
    print(f"    CSV records: {LOG_CONFIG['csv_file']}")
    print(f"    Detailed records: {LOG_CONFIG['json_file']}")
    print(f"    Summary records: {LOG_CONFIG['summary_file']}")

def log_result_to_csv(result, monitor_time):
    """Record result to CSV file"""
    if not result:
        return
    
    # Calculate data freshness
    data_time = result['data_time']
    if data_time:
        now = datetime.now(data_time.tzinfo) if data_time.tzinfo else datetime.now()
        time_diff = now - data_time
        data_age_minutes = time_diff.total_seconds() / 60
    else:
        data_age_minutes = None
    
    # Prepare CSV row data
    row_data = [
        monitor_time,
        data_time.strftime("%Y-%m-%d %H:%M:%S") if data_time else None,
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
        f"{data_age_minutes:.1f}" if data_age_minutes is not None else '',
        result.get('reason', '')
    ]
    
    # Write to CSV
    with open(LOG_CONFIG['csv_file'], 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row_data)

def log_result_to_json(result, monitor_time, monitor_count):
    """Record detailed result to JSON file"""
    if not result:
        return
    
    # Calculate data freshness
    data_time = result['data_time']
    if data_time:
        now = datetime.now(data_time.tzinfo) if data_time.tzinfo else datetime.now()
        time_diff = now - data_time
        data_age_minutes = time_diff.total_seconds() / 60
    else:
        data_age_minutes = None
    
    # Prepare detailed record
    log_entry = {
        'monitor_count': monitor_count,
        'monitor_time': monitor_time,
        'data_time': data_time.isoformat() if data_time else None,
        'power_w': result['power'],
        'type': result['type'],
        'action': result['action'],
        'decision': result.get('decision', ''),
        'scores': result.get('scores', {}),
        'levels': result.get('levels', {}),
        'professional_percentage': result.get('professional_percentage', 0),
        'model_sources': result.get('model_sources', {}),
        'data_age_minutes': data_age_minutes,
        'reason': result.get('reason', ''),
        'threshold_used': PHANTOM_LOAD_THRESHOLD
    }
    
    # Read existing data
    try:
        with open(LOG_CONFIG['json_file'], 'r', encoding='utf-8') as jsonfile:
            log_data = json.load(jsonfile)
    except (FileNotFoundError, json.JSONDecodeError):
        log_data = []
    
    # Add new record
    log_data.append(log_entry)
    
    # Write back to file
    with open(LOG_CONFIG['json_file'], 'w', encoding='utf-8') as jsonfile:
        json.dump(log_data, jsonfile, indent=2, ensure_ascii=False)

def update_summary_file(monitor_count, phantom_count, normal_count):
    """Update summary file"""
    total_valid = phantom_count + normal_count
    phantom_rate = (phantom_count / total_valid * 100) if total_valid > 0 else 0
    
    summary_content = f"""Phantom Load Monitoring System - Runtime Summary
=====================================
Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Monitoring Statistics:
- Total monitoring cycles: {monitor_count}
- Valid data cycles: {total_valid}
- Phantom load detections: {phantom_count} times ({phantom_rate:.1f}%)
- Normal usage detections: {normal_count} times ({100-phantom_rate:.1f}%)

System Configuration:
- Phantom load threshold: {PHANTOM_LOAD_THRESHOLD}W
- Monitor interval: {MONITOR_INTERVAL} seconds
- Decision system status: {'Available' if HAS_DECISION_SYSTEM else 'Unavailable'}

Log Files:
- CSV records: {LOG_CONFIG['csv_file']}
- Detailed records: {LOG_CONFIG['json_file']}
- Summary records: {LOG_CONFIG['summary_file']}
"""
    
    with open(LOG_CONFIG['summary_file'], 'w', encoding='utf-8') as f:
        f.write(summary_content)

def init_decision_system():
    """Initialize decision system"""
    global _decision_system
    
    if not HAS_DECISION_SYSTEM:
        return None
    
    if _decision_system is None:
        print("🚀 Initializing Phantom Load decision system...")
        # Hide training process output
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            _decision_system = DecisionTreeSmartPowerAnalysis()
        print("✅ Decision system initialization completed!")

        # 🆕 新增：初始化防震盪濾波器
        _decision_system.anti_oscillation_filter = AntiOscillationFilter(
            hysteresis_enabled=True,
            phantom_threshold_low=17,
            phantom_threshold_high=21,
            decision_cooldown_seconds=30,
            min_state_duration_minutes=2,
            stability_check_enabled=True,
            sleep_mode_detection_enabled=True,
            sleep_mode_threshold=20,
            sleep_mode_force_shutdown_minutes=15
        )
        print("✅ Anti-oscillation filter initialized!")

        if HAS_EVALUATOR:
            try:
                init_decision_evaluator()
                print("✅ Decision evaluator initialized!")
            except Exception as e:
                print(f"⚠️ Decision evaluator initialization failed: {e}")
    
    return _decision_system

def get_latest_power_data():
    """Get latest power data"""
    phantom_logger.debug("Starting to retrieve latest power data...")
    
    client = InfluxDBClient(
        url=INFLUX_CONFIG['url'], 
        token=INFLUX_CONFIG['token'], 
        org=INFLUX_CONFIG['org']
    )
    query_api = client.query_api()
    
    # Query latest data
    query = f'''
    from(bucket: "{INFLUX_CONFIG['bucket']}")
      |> range(start: -30m)
      |> filter(fn: (r) => r._measurement == "{INFLUX_CONFIG['measurement']}" and r._field == "{INFLUX_CONFIG['field']}")
      |> sort(columns: ["_time"], desc: true)
      |> limit(n: 1)
    '''
    
    try:
        tables = query_api.query(query)
        for table in tables:
            for record in table.records:
                client.close()
                power_value = record.get_value()
                data_time = record.get_time()
                
                phantom_logger.info(f"Successfully retrieved data: {power_value}W @ {data_time}")
                
                return {
                    'time': data_time,
                    'value': power_value,
                    'measurement': record.get_measurement(),
                    'field': record.get_field()
                }
    except Exception as e:
        error_logger.error(f"InfluxDB query error: {e}")
        phantom_logger.warning(f"Unable to retrieve InfluxDB data, error: {e}")
        print(f"❌ InfluxDB query error: {e}")  # Keep original print as backup
    finally:
        client.close()
    
    phantom_logger.warning("No data retrieved")
    return None

def is_phantom_load(power_value, threshold=PHANTOM_LOAD_THRESHOLD):
    """Check if it's Phantom Load"""
    return power_value < threshold

def make_phantom_decision(power_value, timestamp=None):
    """Make Phantom Load decision"""
    
    phantom_logger.debug(f"Starting intelligent decision making for {power_value}W...")
    
    if not HAS_DECISION_SYSTEM:
        phantom_logger.warning("Decision system module not found, using default response")
        return {
            'decision': 'no_system',
            'action': 'No Decision System',
            'reason': 'Decision system module not found'
        }
    
    system = init_decision_system()
    if system is None:
        error_logger.error("Decision system initialization failed")
        return {
            'decision': 'system_error',
            'action': 'System Error',
            'reason': 'Decision system initialization failed'
        }
    
    if timestamp is None:
        timestamp = datetime.now()
    
    phantom_logger.debug(f"Using timestamp: {timestamp}")
    
    # Create opportunity
    opportunity = {
        'device_id': 'D8_realtime',
        'start_time': timestamp,
        'end_time': timestamp + timedelta(minutes=15),
        'power_watt': power_value
    }
    
    phantom_logger.debug(f"Created opportunity object: device_id=D8_realtime, power={power_value}W")
    
    try:
        phantom_logger.debug("Starting feature extraction...")
        # Extract features
        features = system._extract_enhanced_features(opportunity, None)
        phantom_logger.debug("Feature extraction completed")
        
        # Get scores (hide output)
        phantom_logger.debug("Starting score calculations...")
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            # Activity Score
            phantom_logger.debug("Calculating Activity Score...")
            if system.device_activity_model:
                try:
                    activity_result = system.device_activity_model.calculate_activity_score(timestamp)
                    activity_score = activity_result['activity_score']
                    activity_source = "Professional Model"
                    phantom_logger.debug(f"Activity Score (Professional Model): {activity_score:.3f}")
                except Exception as activity_error:
                    phantom_logger.warning(f"Professional Activity model failed, using fallback method: {activity_error}")
                    activity_score = system._fallback_activity_score(features, timestamp)
                    activity_source = "Fallback"
                    phantom_logger.debug(f"Activity Score (Fallback): {activity_score:.3f}")
            else:
                phantom_logger.debug("No professional Activity model, using fallback method")
                activity_score = system._fallback_activity_score(features, timestamp)
                activity_source = "Fallback"
                phantom_logger.debug(f"Activity Score (Fallback): {activity_score:.3f}")
            
            # Habit Score
            phantom_logger.debug("Calculating Habit Score...")
            if system.user_habit_model:
                try:
                    habit_result = system.user_habit_model.calculate_habit_score(timestamp)
                    habit_score = habit_result['habit_score']
                    habit_source = "Professional Model"
                    phantom_logger.debug(f"Habit Score (Professional Model): {habit_score:.3f}")
                except Exception as habit_error:
                    phantom_logger.warning(f"Professional Habit model failed, using fallback method: {habit_error}")
                    habit_score = system._fallback_habit_score(features, timestamp)
                    habit_source = "Fallback"
                    phantom_logger.debug(f"Habit Score (Fallback): {habit_score:.3f}")
            else:
                phantom_logger.debug("No professional Habit model, using fallback method")
                habit_score = system._fallback_habit_score(features, timestamp)
                habit_source = "Fallback"
                phantom_logger.debug(f"Habit Score (Fallback): {habit_score:.3f}")
            
            # Confidence Score
            phantom_logger.debug("Calculating Confidence Score...")
            if system.confidence_model:
                try:
                    confidence_result = system.confidence_model.calculate_confidence_score(timestamp)
                    confidence_score = confidence_result['confidence_score']
                    confidence_source = "Professional Model"
                    phantom_logger.debug(f"Confidence Score (Professional Model): {confidence_score:.3f}")
                except Exception as confidence_error:
                    phantom_logger.warning(f"Professional Confidence model failed, using fallback method: {confidence_error}")
                    confidence_score = system._fallback_confidence_score(features, timestamp)
                    confidence_source = "Fallback"
                    phantom_logger.debug(f"Confidence Score (Fallback): {confidence_score:.3f}")
            else:
                phantom_logger.debug("No professional Confidence model, using fallback method")
                confidence_score = system._fallback_confidence_score(features, timestamp)
                confidence_source = "Fallback"
                phantom_logger.debug(f"Confidence Score (Fallback): {confidence_score:.3f}")
            
            # Make decision
            phantom_logger.debug("Starting intelligent decision making...")
            decision, debug_info = system._make_intelligent_decision(
                activity_score, habit_score, confidence_score, features
            )
            phantom_logger.debug(f"Decision completed: {decision}")
        
        # Record score summary
        phantom_logger.info(f"Score calculation completed - Activity: {activity_score:.2f}, Habit: {habit_score:.2f}, Confidence: {confidence_score:.2f}")
        
        # English decision mapping
        english_actions = {
            'suggest_shutdown': 'Suggest Shutdown',
            'send_notification': 'Send Notification',
            'keep_on': 'Keep On',
            'delay_decision': 'Delay Decision'
        }
        
        # Calculate professional model usage rate
        professional_count = sum([
            1 if source == "Professional Model" else 0 
            for source in [activity_source, habit_source, confidence_source]
        ])
        professional_percentage = (professional_count / 3) * 100
        
        phantom_logger.info(f"Professional model usage rate: {professional_percentage:.0f}%")
        phantom_logger.info(f"Model sources - Activity: {activity_source}, Habit: {habit_source}, Confidence: {confidence_source}")
        
        # Final decision
        final_action = english_actions.get(decision, decision)
        phantom_logger.info(f"Final decision: {final_action}")
        
        # Record detailed level information
        phantom_logger.debug(f"Decision levels - Activity: {debug_info.get('device_activity_level', '?')}, "
                           f"Habit: {debug_info.get('user_habit_level', '?')}, "
                           f"Confidence: {debug_info.get('confidence_score_level', '?')}")
        
        if hasattr(system, 'anti_oscillation_filter'):
            filter_result = system.anti_oscillation_filter.filter_decision(
                original_decision=decision,
                power_value=power_value,
                timestamp=timestamp,
                scores={
                    'activity': activity_score,
                    'habit': habit_score,
                    'confidence': confidence_score
                }
            )
            
            # 使用濾波後的決策
            final_decision = filter_result['filtered_decision']
            final_action = english_actions.get(final_decision, final_decision)
            
            # 如果決策被濾波器修改了，記錄原因
            if filter_result['should_use_filtered']:
                phantom_logger.info(f"Decision filtered: {decision} -> {final_decision} ({filter_result['filter_reason']})")

            if HAS_EVALUATOR and _decision_evaluator is not None:
                try:
                    # 計算fuzzy輸出（簡化版）
                    fuzzy_output = (activity_score + habit_score + confidence_score) / 3
                    
                    # 估算預測功率
                    predicted_power = power_value * (1 - fuzzy_output * 0.2)
                    
                    # 添加記錄
                    _decision_evaluator.add_decision_record(
                        timestamp=timestamp,
                        fuzzy_output=fuzzy_output,
                        predicted_power=predicted_power,
                        actual_power=power_value,
                        decision=final_decision,
                        confidence_scores={
                            'activity': activity_score,
                            'habit': habit_score,
                            'confidence': confidence_score
                        }
                    )
                except Exception as e:
                    phantom_logger.warning(f"Decision evaluator recording failed: {e}")
            
            return {
                'decision': final_decision,
                'action': final_action,
                'scores': {
                    'activity': activity_score,
                    'habit': habit_score,
                    'confidence': confidence_score
                },
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
                # 🆕 添加濾波器信息
                'filter_applied': filter_result['should_use_filtered'],
                'filter_reason': filter_result['filter_reason'],
                'original_decision': decision,
                'power_state': filter_result['power_state']
            }
        else:

            if HAS_EVALUATOR and _decision_evaluator is not None:
                try:
                    # 計算fuzzy輸出（簡化版）
                    fuzzy_output = (activity_score + habit_score + confidence_score) / 3
                    
                    # 估算預測功率
                    predicted_power = power_value * (1 - fuzzy_output * 0.2)
                    
                    # 添加記錄
                    _decision_evaluator.add_decision_record(
                        timestamp=timestamp,
                        fuzzy_output=fuzzy_output,
                        predicted_power=predicted_power,
                        actual_power=power_value,
                        decision=decision,
                        confidence_scores={
                            'activity': activity_score,
                            'habit': habit_score,
                            'confidence': confidence_score
                        }
                    )
                except Exception as e:
                    phantom_logger.warning(f"Decision evaluator recording failed: {e}")
            
            # 原有邏輯
            return {
                'decision': decision,
                'action': final_action,
                'scores': {
                    'activity': activity_score,
                    'habit': habit_score,
                    'confidence': confidence_score
                },
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
                }
            }
        
    except Exception as e:
        error_logger.error(f"Error occurred during decision process: {e}")
        phantom_logger.error(f"Decision failed, power value: {power_value}W, error: {e}")
        
        # Record detailed error information
        import traceback
        error_logger.error(f"Error stack trace: {traceback.format_exc()}")
        
        return {
            'decision': 'error',
            'action': 'Decision Error',
            'reason': str(e)
        }

def process_power_data(data):
    """Process power data and perform intelligent control"""
    
    if not data:
        phantom_logger.warning("Received empty data, skipping processing")
        return None
    
    power_value = data['value']
    data_time = data['time']
    
    phantom_logger.debug(f"Processing data: {power_value}W, using threshold: {PHANTOM_LOAD_THRESHOLD}W")
    
    # Check if it's Phantom Load
    if not is_phantom_load(power_value, PHANTOM_LOAD_THRESHOLD):
        phantom_logger.info(f"Normal usage: {power_value}W (>= {PHANTOM_LOAD_THRESHOLD}W)")
        return {
            'type': 'normal_usage',
            'power': power_value,
            'data_time': data_time,
            'action': 'No Action',
            'reason': f'Power {power_value}W ≥ {PHANTOM_LOAD_THRESHOLD}W, within normal usage range'
        }
    
    # Phantom Load detected
    phantom_logger.warning(f"🚨 Phantom Load detected: {power_value}W (< {PHANTOM_LOAD_THRESHOLD}W)")
    
    # Make Phantom Load decision
    phantom_logger.debug("Starting intelligent decision analysis...")
    decision_result = make_phantom_decision(power_value, data_time)
    
    phantom_logger.info(f"Intelligent decision result: {decision_result['action']}")
    
    return {
        'type': 'phantom_load',
        'power': power_value,
        'data_time': data_time,
        'action': decision_result['action'],
        'decision': decision_result['decision'],
        'scores': decision_result.get('scores', {}),
        'levels': decision_result.get('levels', {}),
        'professional_percentage': decision_result.get('professional_percentage', 0),
        'model_sources': decision_result.get('model_sources', {}),
        'reason': decision_result.get('reason', '')
    }

def print_monitoring_result(result, current_time):
    """Format and print monitoring results"""
    
    if not result:
        print(f"[{current_time}] ❌ No data received")
        return
    
    power = result['power']
    data_time = result['data_time']
    action = result['action']
    
    # Calculate data freshness
    now = datetime.now(data_time.tzinfo) if data_time.tzinfo else datetime.now()
    time_diff = now - data_time
    minutes_old = time_diff.total_seconds() / 60
    
    if result['type'] == 'normal_usage':
        print(f"[{current_time}] ⚪ Normal Usage")
        print(f"    Power: {power}W | Data Time: {data_time}")
        print(f"    Status: {action} | Data Age: {minutes_old:.1f} minutes ago")
        
    elif result['type'] == 'phantom_load':
        # Emoji mapping
        emoji_map = {
            'Suggest Shutdown': '🔴',
            'Send Notification': '🔔',
            'Keep On': '🟢',
            'Delay Decision': '🟡',
            'No Decision System': '⚫',
            'System Error': '❌',
            'Decision Error': '❌'
        }
        emoji = emoji_map.get(action, '❓')
        
        print(f"[{current_time}] 🚨 Phantom Load Detected")
        print(f"    Power: {power}W | Data Time: {data_time}")
        print(f"    {emoji} Smart Decision: {action}")
        
        if 'scores' in result and result['scores']:
            scores = result['scores']
            levels = result.get('levels', {})
            print(f"    Scores: A:{scores.get('activity', 0):.2f} H:{scores.get('habit', 0):.2f} C:{scores.get('confidence', 0):.2f}")
            print(f"    Levels: {levels.get('activity', '?')}-{levels.get('habit', '?')}-{levels.get('confidence', '?')}")
            
            professional_percentage = result.get('professional_percentage', 0)
            if professional_percentage == 100:
                print(f"    ✅ Professional Model: {professional_percentage:.0f}% (System working normally)")
            elif professional_percentage >= 50:
                print(f"    ⚠️ Professional Model: {professional_percentage:.0f}% (Partial fallback usage)")
            else:
                print(f"    ❌ Professional Model: {professional_percentage:.0f}% (Heavy fallback usage)")
        
        if result.get('reason'):
            print(f"    Reason: {result['reason']}")
        
        print(f"    Data Age: {minutes_old:.1f} minutes ago")

        # 🆕 添加濾波器信息顯示
        if result.get('filter_applied', False):
            print(f"    🔧 Filter Applied: {result.get('filter_reason', 'Unknown')}")
            if result.get('original_decision') != result.get('decision'):
                print(f"    🔄 Original -> Filtered: {result.get('original_decision')} -> {result.get('decision')}")
    
    # Log to files
    print(f"    📝 Results logged to files")

def main():
    """Main program - Real-time monitoring and intelligent control"""
    
    phantom_logger.info("🚀 Phantom Load monitoring system startup")
    phantom_logger.info(f"Monitoring device: {INFLUX_CONFIG['measurement']}")
    phantom_logger.info(f"Phantom Load threshold: {PHANTOM_LOAD_THRESHOLD}W")
    phantom_logger.info(f"Monitor interval: {MONITOR_INTERVAL} seconds")
    
    print("⚡ Real-time InfluxDB Phantom Load Intelligent Control System")
    print("=" * 70)
    print(f"📊 Monitoring Device: {INFLUX_CONFIG['measurement']}")
    print(f"🎯 Phantom Load Threshold: {PHANTOM_LOAD_THRESHOLD}W")
    print(f"⏰ Monitor Interval: {MONITOR_INTERVAL} seconds")
    print(f"🛑 Press Ctrl+C to stop monitoring")
    print("-" * 70)
    
    # Initialize logging
    phantom_logger.info("Initializing logging system...")
    init_logging()
    print("-" * 70)
    
    # Initialize decision system
    if HAS_DECISION_SYSTEM:
        phantom_logger.info("Initializing decision system...")
        init_decision_system()
    else:
        phantom_logger.warning("Decision system unavailable, providing data monitoring only")
        print("⚠️ Data monitoring only, no decision functionality")
        print("-" * 70)
    
    monitoring_count = 0
    phantom_count = 0
    normal_count = 0
    
    phantom_logger.info("Starting monitoring loop...")
    
    try:
        while True:
            monitoring_count += 1
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            phantom_logger.debug(f"Starting monitoring cycle #{monitoring_count}...")
            print(f"\n🔍 Monitor #{monitoring_count}:")
            
            # Get latest data
            data = get_latest_power_data()
            
            # Process data and make control decisions
            result = process_power_data(data)
            
            # Print results
            print_monitoring_result(result, current_time)
            
            # Record to files
            if result:
                log_result_to_csv(result, current_time)
                log_result_to_json(result, current_time, monitoring_count)
                
                # Update statistics
                if result['type'] == 'phantom_load':
                    phantom_count += 1
                elif result['type'] == 'normal_usage':
                    normal_count += 1
            
            # Update summary file
            update_summary_file(monitoring_count, phantom_count, normal_count)
            
            # Show statistics every 5 monitoring cycles
            if monitoring_count % 5 == 0:
                total_valid = phantom_count + normal_count
                if total_valid > 0:
                    phantom_rate = (phantom_count / total_valid) * 100
                    phantom_logger.info(f"Statistics - Phantom: {phantom_count} times ({phantom_rate:.1f}%), Normal: {normal_count} times")
                    
                    print(f"\n📈 Monitoring Statistics (Last {total_valid} valid data):")
                    print(f"    Phantom Load: {phantom_count} times ({phantom_rate:.1f}%)")
                    print(f"    Normal Usage: {normal_count} times ({100-phantom_rate:.1f}%)")
                    print(f"    📁 Records saved to: {LOG_CONFIG['csv_file']}")

                    # 🆕 顯示濾波器狀態
                    if _decision_system and hasattr(_decision_system, 'anti_oscillation_filter'):
                        filter_status = _decision_system.anti_oscillation_filter.get_filter_status()
                        print(f"\n🔧 Anti-oscillation Filter Status:")
                        print(f"    Power State: {filter_status['current_power_state']}")
                        print(f"    Sleep Mode: {'Detected' if filter_status.get('sleep_mode_detected', False) else 'Not Detected'}")
            
            phantom_logger.debug(f"Waiting {MONITOR_INTERVAL} seconds for next monitoring cycle...")
            print(f"\n⏳ Waiting {MONITOR_INTERVAL} seconds for next monitoring...")
            time.sleep(MONITOR_INTERVAL)
            
    except KeyboardInterrupt:
        phantom_logger.info("🛑 Received interrupt signal, shutting down monitoring system normally")
        phantom_logger.info(f"Monitoring statistics - Total: {monitoring_count} times, Phantom: {phantom_count} times, Normal: {normal_count} times")
        
        print(f"\n\n🛑 Monitoring stopped")
        print(f"📊 Total monitoring cycles: {monitoring_count}")
        print(f"🚨 Phantom Load detections: {phantom_count} times")
        print(f"⚪ Normal usage detections: {normal_count} times")
        
        # Final summary update
        update_summary_file(monitoring_count, phantom_count, normal_count)
        
        print(f"\n📁 Monitoring records saved to:")
        print(f"    📊 CSV data: {LOG_CONFIG['csv_file']}")
        print(f"    📋 Detailed records: {LOG_CONFIG['json_file']}")
        print(f"    📝 Runtime summary: {LOG_CONFIG['summary_file']}")
        print("👋 Thank you for using!")

        if HAS_EVALUATOR and _decision_evaluator is not None:
            try:
                print(f"\n📊 Decision Evaluation Summary:")
                evaluation_summary = _decision_evaluator.get_evaluation_summary()
                if 'average_scores' in evaluation_summary:
                    avg_scores = evaluation_summary['average_scores']
                    print(f"    Overall Performance: {avg_scores['overall']:.3f}")
                    print(f"    Stability Score: {avg_scores['stability']:.3f}")
                    print(f"    Consistency Score: {avg_scores['consistency']:.3f}")
                
                # 匯出評估結果
                eval_file = _decision_evaluator.export_evaluation_results('realtime_evaluation_log.csv')
                if eval_file:
                    print(f"    📋 Evaluation results exported: {eval_file}")
            except Exception as e:
                print(f"⚠️ Evaluation summary failed: {e}")
        
    except Exception as e:
        error_logger.critical(f"Monitoring system terminated abnormally: {e}")
        phantom_logger.error(f"System error, program terminating: {e}")
        raise

if __name__ == "__main__":
    main()