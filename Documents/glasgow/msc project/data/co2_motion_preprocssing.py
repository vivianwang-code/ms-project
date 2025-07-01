import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class ContextScoreModule:
    
    def __init__(self):
        self.co2_thresholds = {}
        self.time_thresholds = {}
        self.context_thresholds = {}
        self.rules = []
        
    def load_co2_data(self, co2_file_path):
        print("==== Loading CO2 Data for Context Score ====")
        
        if co2_file_path is None:
            print(f"Cannot find {co2_file_path}")
            return None
            
        with open(co2_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        start_data = 0
        for i, line in enumerate(lines):
            if not line.startswith('#') and ',result' in line:
                start_data = i
                
        df = pd.read_csv(co2_file_path, skiprows=start_data)
        df['co2_value'] = pd.to_numeric(df['_value'])
        
        # Calculate CO2 thresholds
        self.co2_thresholds = {
            'low': df['co2_value'].quantile(0.25),
            'medium': df['co2_value'].quantile(0.5),
            'high': df['co2_value'].quantile(0.75)
        }
        
        print(f"CO2 thresholds: Low={self.co2_thresholds['low']:.1f}, "
              f"Medium={self.co2_thresholds['medium']:.1f}, High={self.co2_thresholds['high']:.1f}")
        
        return df
    
    def load_motion_data(self, motion_file_path):

        print("==== Loading Motion Data for Context Score ====")
        
        if motion_file_path is None:
            print(f"Cannot find {motion_file_path}")
            return None
            
        with open(motion_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        start_data = 0
        for i, line in enumerate(lines):
            if not line.startswith('#') and ',result' in line:
                start_data = i
                
        df = pd.read_csv(motion_file_path, skiprows=start_data)
        df['motion_value'] = pd.to_numeric(df['_value'])
        
        print(f"Motion data: {len(df)} records, "
              f"{(df['motion_value'] > 0).sum()} motion detected, "
              f"{(df['motion_value'] == 0).sum()} no motion")
        
        # Display motion value distribution for better understanding
        non_zero_motion = df[df['motion_value'] > 0]['motion_value']
        if len(non_zero_motion) > 0:
            print(f"Non-zero motion statistics:")
            print(f"  Min: {non_zero_motion.min():.2f}")
            print(f"  Max: {non_zero_motion.max():.2f}")
            print(f"  Mean: {non_zero_motion.mean():.2f}")
            print(f"  50%: {non_zero_motion.quantile(0.5):.2f}")
            print(f"  75%: {non_zero_motion.quantile(0.75):.2f}")
            print(f"  90%: {non_zero_motion.quantile(0.9):.2f}")
        
        return df
    
    def calculate_time_since_last_motion(self, motion_df):
        """Calculate time since last motion"""

        print("==== Calculating Time Since Last Motion ====")
        
        # Find time column
        time_col = '_time' if '_time' in motion_df.columns else 'timestamp'
        
        # Convert time and sort
        motion_df[time_col] = pd.to_datetime(motion_df[time_col], format='ISO8601')
        motion_df = motion_df.sort_values(by=time_col).reset_index(drop=True)
        
        # Calculate time intervals
        motion_df['time_since_last_motion'] = np.nan   #nan : initialize the column 設置為空欄
        last_motion_time = None
        
        for i in range(len(motion_df)):
            current_time = motion_df.iloc[i][time_col]
            motion = motion_df.iloc[i]['motion_value']
            
            if motion > 0:  # Motion detected
                motion_df.iloc[i, motion_df.columns.get_loc('time_since_last_motion')] = 0   #setting time_since_last_motion = 0
                last_motion_time = current_time
            else:  # if motion = 0
                if last_motion_time is not None:
                    delta = current_time - last_motion_time
                    motion_df.iloc[i, motion_df.columns.get_loc('time_since_last_motion')] = delta.total_seconds() / 60.0  #convert to minute
        
        # Calculate time thresholds
        valid_times = motion_df['time_since_last_motion'].dropna()
        if len(valid_times) > 0:
            self.time_thresholds = {
                'recent': valid_times.quantile(0.25),
                'medium': valid_times.quantile(0.5),
                'long': valid_times.quantile(0.75)
            }
            print(f"Time thresholds: Recent={self.time_thresholds['recent']:.1f}min, "
                  f"Medium={self.time_thresholds['medium']:.1f}min, Long={self.time_thresholds['long']:.1f}min")
        
        return motion_df
    
    def triangular_membership(self, x, a, b, c):
        """Triangular membership function"""
        return np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))
    
    def calculate_co2_membership(self, co2_df):
        """Calculate CO2 membership functions"""

        print("==== Calculating CO2 Membership Functions ====")
        
        min_co2 = co2_df['co2_value'].min()
        max_co2 = co2_df['co2_value'].max()
        
        co2_df['co2_low_membership'] = self.triangular_membership(
            co2_df['co2_value'], 
            min_co2, 
            self.co2_thresholds['low'], 
            self.co2_thresholds['medium']
            )
        
        co2_df['co2_medium_membership'] = self.triangular_membership(
            co2_df['co2_value'], 
            self.co2_thresholds['low'], 
            self.co2_thresholds['medium'], 
            self.co2_thresholds['high'])
        
        co2_df['co2_high_membership'] = self.triangular_membership(
            co2_df['co2_value'], 
            self.co2_thresholds['medium'], 
            self.co2_thresholds['high'], 
            max_co2)
        
        return co2_df
    
    def calculate_motion_membership(self, motion_df):
        """Calculate properly differentiated Motion membership functions"""
        #將motion分為 : no / light / strong 

        print("==== Calculating Properly Differentiated Motion Membership Functions ====")
        
        # Analyze motion data distribution for optimal thresholds
        non_zero_motion = motion_df[motion_df['motion_value'] > 0]['motion_value']
        if len(non_zero_motion) > 0:
            # Use more aggressive thresholds for better differentiation
            light_threshold = non_zero_motion.quantile(0.6)   # 60th percentile
            strong_threshold = non_zero_motion.quantile(0.85) # 85th percentile  
            max_motion = non_zero_motion.max()
        else:
            print('non zero motion is empty')
        
        print(f"Motion differentiation thresholds: Light={light_threshold:.2f}, Strong={strong_threshold:.2f}")
        
        # No Motion Membership - Binary: 1 for no motion, 0 for any motion
        motion_df['no_motion_membership'] = (motion_df['motion_value'] == 0).astype(float)   
        
        # Light Motion Membership - Peaks at light_threshold, decreases towards strong_threshold
        motion_df['light_motion_membership'] = np.where(
            motion_df['motion_value'] == 0, 0.0,  # No membership for zero motion
            np.where(
                motion_df['motion_value'] <= light_threshold,
                0.8 * motion_df['motion_value'] / light_threshold,  # Rise to 0.8 max
                np.maximum(0, 0.8 - 0.8 * (motion_df['motion_value'] - light_threshold) / 
                          (strong_threshold - light_threshold))  # Decrease to 0
            )
        )
        
        # Strong Motion Membership - Starts at light_threshold, peaks above strong_threshold
        motion_df['strong_motion_membership'] = np.where(
            motion_df['motion_value'] <= light_threshold, 0.0,  # No strong motion below light threshold
            np.where(
                motion_df['motion_value'] <= strong_threshold,
                0.2 + 0.8 * (motion_df['motion_value'] - light_threshold) / 
                (strong_threshold - light_threshold),  # Rise from 0.2 to 1.0
                1.0  # Full membership above strong threshold
            )
        )
        
        # Ensure all memberships are between 0 and 1
        for col in ['no_motion_membership', 'light_motion_membership', 'strong_motion_membership']:
            motion_df[col] = np.clip(motion_df[col], 0, 1)
        
        # Display statistics and check differentiation
        print(f"Motion membership statistics:")
        print(f"  No motion - Mean: {motion_df['no_motion_membership'].mean():.3f}, "
              f"Max: {motion_df['no_motion_membership'].max():.3f}")
        print(f"  Light motion - Mean: {motion_df['light_motion_membership'].mean():.3f}, "
              f"Max: {motion_df['light_motion_membership'].max():.3f}")
        print(f"  Strong motion - Mean: {motion_df['strong_motion_membership'].mean():.3f}, "
              f"Max: {motion_df['strong_motion_membership'].max():.3f}")
        
        # Check class distribution
        motion_classes = ['no_motion_membership', 'light_motion_membership', 'strong_motion_membership']
        dominant_motion = motion_df[motion_classes].idxmax(axis=1)
        class_counts = dominant_motion.value_counts()
        print(f"Motion class distribution:")
        for class_name, count in class_counts.items():
            percentage = count / len(motion_df) * 100
            print(f"  {class_name.replace('_membership', '')}: {count} ({percentage:.1f}%)")
        
        # Visualize the membership functions
        self.visualize_motion_memberships(motion_df, light_threshold, strong_threshold)
        
        return motion_df
    
    def visualize_motion_memberships(self, motion_df, light_threshold, strong_threshold):
        """Visualize the differentiated motion membership functions"""
        plt.figure(figsize=(13, 12))
        
        # Plot 1: Membership functions
        # 結果顯示
        # no motion : 脈衝，只有motion = 0 時，才會有反應
        # light motion : max. membership function 設為0.8，0~2.5上升至0.8，2.5~4.0下降至0
        # strong motion : max. membership fuction 設為1.0，2.5~4.0上升至1.0，4.0以上保持0
        # 允許同時具有多種運動狀態的隸屬度
        plt.subplot(2, 2, 1)
        motion_range = np.linspace(0, motion_df['motion_value'].max(), 1000)
        
        # Calculate membership values for plotting
        no_motion_plot = (motion_range == 0).astype(float)
        
        light_motion_plot = np.where(motion_range == 0, 0.0, 
                                   np.where(motion_range <= light_threshold,
                                           0.8 * motion_range / light_threshold,
                                           np.maximum(0, 0.8 - 0.8 * (motion_range - light_threshold) / 
                                                     (strong_threshold - light_threshold))))
        
        strong_motion_plot = np.where(motion_range <= light_threshold, 0.0, 
                                    np.where(motion_range <= strong_threshold,
                                           0.2 + 0.8 * (motion_range - light_threshold) / 
                                           (strong_threshold - light_threshold),
                                           1.0))
        
        plt.plot(motion_range, no_motion_plot, 'r-', label='No Motion', linewidth=3)
        plt.plot(motion_range, light_motion_plot, 'orange', label='Light Motion', linewidth=3)
        plt.plot(motion_range, strong_motion_plot, 'g-', label='Strong Motion', linewidth=3)
        
        plt.axvline(light_threshold, color='gray', linestyle='--', alpha=0.8, linewidth=2, 
                   label=f'Light Threshold: {light_threshold:.1f}')
        plt.axvline(strong_threshold, color='black', linestyle=':', alpha=0.8, linewidth=2, 
                   label=f'Strong Threshold: {strong_threshold:.1f}')
        
        plt.xlabel('Motion Value')
        plt.ylabel('Membership Degree')
        plt.title('Differentiated Motion Membership Functions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.1)
        
        # Plot 2: Motion value vs memberships scatter
        # 真實數據分布
        # 與前一個圖(理論圖)符合，表membership function 之設計有效性
        plt.subplot(2, 2, 2)
        non_zero_data = motion_df[motion_df['motion_value'] > 0].sample(min(500, len(motion_df[motion_df['motion_value'] > 0])))
        if len(non_zero_data) > 0:
            plt.scatter(non_zero_data['motion_value'], non_zero_data['light_motion_membership'], 
                       c='orange', alpha=0.6, s=20, label='Light Motion')
            plt.scatter(non_zero_data['motion_value'], non_zero_data['strong_motion_membership'], 
                       c='green', alpha=0.6, s=20, label='Strong Motion')
            plt.xlabel('Motion Value')
            plt.ylabel('Membership Degree')
            plt.title('Motion Memberships vs Values')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 3: Class distribution
        plt.subplot(2, 2, 3)
        motion_classes = ['no_motion_membership', 'light_motion_membership', 'strong_motion_membership']
        dominant_motion = motion_df[motion_classes].idxmax(axis=1)
        class_counts = dominant_motion.value_counts()
        
        labels = [name.replace('_membership', '').replace('_', ' ').title() for name in class_counts.index]
        colors = ['lightcoral', 'orange', 'lightgreen']
        
        plt.pie(class_counts.values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Motion Class Distribution')
        
        # # Plot 4: Membership distributions
        # plt.subplot(2, 4, 4)
        # colors = ['red', 'orange', 'green']
        # for i, (mem, color) in enumerate(zip(motion_classes, colors)):
        #     plt.hist(motion_df[mem], bins=20, alpha=0.6, color=color, 
        #             label=mem.replace('_membership', '').replace('_', ' ').title())
        # plt.xlabel('Membership Degree')
        # plt.ylabel('Frequency')
        # plt.title('Membership Distributions')
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        
        # # Plot 5: Sample time series
        # plt.subplot(2, 4, 5)
        # sample_size = min(200, len(motion_df))
        # sample_indices = range(0, sample_size)
        # for mem, color in zip(motion_classes, colors):
        #     plt.plot(sample_indices, motion_df[mem].iloc[sample_indices], 
        #             color=color, alpha=0.7, linewidth=1,
        #             label=mem.replace('_membership', '').replace('_', ' ').title())
        # plt.xlabel('Data Point')
        # plt.ylabel('Membership Degree')
        # plt.title('Memberships Over Time (Sample)')
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        
        # Plot 6: Motion value distribution
        plt.subplot(2, 2, 4)
        non_zero_motion = motion_df[motion_df['motion_value'] > 0]['motion_value']
        if len(non_zero_motion) > 0:
            plt.hist(non_zero_motion, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(light_threshold, color='orange', linestyle='--', linewidth=2, 
                       label=f'Light: {light_threshold:.1f}')
            plt.axvline(strong_threshold, color='green', linestyle='--', linewidth=2, 
                       label=f'Strong: {strong_threshold:.1f}')
            plt.xlabel('Motion Value (Non-zero)')
            plt.ylabel('Frequency')
            plt.title('Motion Value Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # # Plot 7: Correlation matrix
        # plt.subplot(2, 4, 7)
        # correlation_data = motion_df[motion_classes].corr()
        # sns.heatmap(correlation_data, annot=True, cmap='RdBu_r', center=0, 
        #            xticklabels=['No Motion', 'Light Motion', 'Strong Motion'],
        #            yticklabels=['No Motion', 'Light Motion', 'Strong Motion'])
        # plt.title('Motion Membership Correlations')
        
        # # Plot 8: Motion effectiveness check
        # plt.subplot(2, 4, 8)
        # # Group by motion value ranges and show membership patterns
        # motion_ranges = pd.cut(motion_df['motion_value'], bins=[0, 1, light_threshold, strong_threshold, motion_df['motion_value'].max()], 
        #                       labels=['Zero', 'Low', 'Medium', 'High'], include_lowest=True)
        
        # range_stats = motion_df.groupby(motion_ranges)[motion_classes].mean()
        
        # x_pos = np.arange(len(range_stats))
        # width = 0.25
        
        # for i, (mem, color) in enumerate(zip(motion_classes, colors)):
        #     plt.bar(x_pos + i*width, range_stats[mem], width, 
        #            color=color, alpha=0.7, 
        #            label=mem.replace('_membership', '').replace('_', ' ').title())
        
        # plt.xlabel('Motion Value Range')
        # plt.ylabel('Average Membership')
        # plt.title('Membership by Motion Range')
        # plt.xticks(x_pos + width, range_stats.index)
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        
        # plt.tight_layout()
        plt.subplots_adjust(
            left=0.05,      # 左邊距
            bottom=0.08,    # 下邊距  
            right=0.95,     # 右邊距
            top=0.92,       # 上邊距
            wspace=0.3,     # 子圖間水平間距 (增加這個值讓左右圖分開)
            hspace=0.4      # 子圖間垂直間距 (增加這個值讓上下圖分開)
        )
        plt.show()
    
    def calculate_time_membership(self, motion_df):
        """Calculate Time since last motion membership functions"""

        print("==== Calculating Time Membership Functions ====")
        
        if len(self.time_thresholds) == 0:
            print("No time thresholds available")
            return motion_df
            
        valid_times = motion_df['time_since_last_motion'].dropna()
        min_time = valid_times.min()
        max_time = valid_times.max()
        
        motion_df['time_recent_membership'] = self.triangular_membership(
            motion_df['time_since_last_motion'], 
            min_time, 
            self.time_thresholds['recent'], 
            self.time_thresholds['medium'])
        
        motion_df['time_medium_membership'] = self.triangular_membership(
            motion_df['time_since_last_motion'], 
            self.time_thresholds['recent'], 
            self.time_thresholds['medium'], 
            self.time_thresholds['long'])
        
        motion_df['time_long_membership'] = self.triangular_membership(
            motion_df['time_since_last_motion'], 
            self.time_thresholds['medium'], 
            self.time_thresholds['long'], 
            max_time)
        
        return motion_df
    
    def define_context_rules(self):
        """Define motion-focused fuzzy rules that properly reward strong motion"""

        print("==== Defining Motion-Focused Context Score Rules ====")
        
        # Rules designed to clearly differentiate motion classes
        self.rules = [
        # CO2, motion, time since last motion, 系統評分, weight

        # === HIGH評分：確定有人使用電視，絕對保持電源 ===
        # 強烈活動 = 正在積極使用電視（換台、調音量等）
        ('low', 'strong_motion', 'recent', 'high', 1.0),      # 1人積極使用 - 最高權重
        ('low', 'strong_motion', 'medium', 'high', 0.95),     # 1人積極，短暫停頓
        ('medium', 'strong_motion', 'recent', 'high', 1.0),   # 2-3人積極使用 - 最高權重  
        ('medium', 'strong_motion', 'medium', 'high', 0.95),  # 2-3人積極，短暫停頓
        ('high', 'strong_motion', 'recent', 'high', 1.0),     # 多人積極使用 - 最高權重
        ('high', 'strong_motion', 'medium', 'high', 0.95),    # 多人積極，短暫停頓
        
        # === MEDIUM評分：可能在看電視，需要謹慎判斷 ===
        # 長時間強烈活動後突然停止 = 可能暫時離開但會回來
        ('low', 'strong_motion', 'long', 'medium', 0.7),      # 長時間沒人後重新啟用 - 降低權重
        ('medium', 'strong_motion', 'long', 'medium', 0.75),  # 多人長時間離開後回來
        ('high', 'strong_motion', 'long', 'medium', 0.8),     # 多人長時間離開後回來
        
        # 輕微活動 = 安靜看電視（最重要的電視使用場景）
        ('low', 'light_motion', 'recent', 'medium', 0.85),    # 1人安靜看電視 - 重要場景，高權重
        ('low', 'light_motion', 'medium', 'medium', 0.8),     # 1人安靜，短暫停頓 - 高權重
        ('medium', 'light_motion', 'recent', 'medium', 0.9),  # 2-3人安靜看電視 - 重要場景
        ('medium', 'light_motion', 'medium', 'medium', 0.85), # 2-3人安靜，短暫停頓
        ('high', 'light_motion', 'recent', 'medium', 0.9),    # 多人安靜看電視 - 重要場景
        
        # === 過渡期評分：剛停止活動，謹慎處理 ===
        # 剛停止活動 = 可能暫時離開（接電話、上廁所等）
        ('low', 'no_motion', 'recent', 'medium', 0.6),        # 1人剛停止 - 中等權重，避免誤切
        ('medium', 'no_motion', 'recent', 'medium', 0.7),     # 多人剛停止 - 稍高權重
        
        # === LOW評分：可以考慮切斷電源，但需要不同信心程度 ===
        # 長時間輕微活動 = 可能忘記關電視
        ('low', 'light_motion', 'long', 'low', 0.6),          # 輕微活動但時間久 - 中等權重
        ('medium', 'light_motion', 'long', 'low', 0.65),      # 多人輕微活動但時間久
        ('high', 'light_motion', 'medium', 'low', 0.7),       # 多人安靜且空氣差 - 可能睡著
        ('high', 'light_motion', 'long', 'low', 0.8),         # 多人輕微活動但時間久 - 較高信心切斷
        
        # 中期無活動 = 中等信心可以切斷
        ('low', 'no_motion', 'medium', 'low', 0.8),           # 1人中期無活動 - 較高信心
        ('medium', 'no_motion', 'medium', 'low', 0.85),       # 多人中期無活動 - 高信心
        
        # 長期無活動 = 高信心可以切斷電源（低CO2情況）
        ('low', 'no_motion', 'long', 'low', 1.0),             # 1人長時間離開 - 最高信心切斷
        ('medium', 'no_motion', 'long', 'low', 1.0),          # 多人長時間離開 - 最高信心切斷
        
        # 高CO2但無活動 = 複雜情況，需要謹慎判斷
        ('high', 'no_motion', 'recent', 'medium', 0.5),       # 高CO2剛無活動 - 可能專心看電視，低權重
        ('high', 'no_motion', 'medium', 'low', 0.6),          # 高CO2中期無活動 - 可能睡著，中低權重  
        ('high', 'no_motion', 'long', 'low', 0.7), 
    ]
        
        print(f"Defined {len(self.rules)} motion-focused context rules")
        print("Rule priorities: Strong Motion > Light Motion > No Motion")
        return self.rules
    
    def calculate_rule_activation(self, merged_df):
        """Calculate rule activation strengths"""

        print("==== Calculating Rule Activation Strengths ====")
        
        # Initialize context membership columns
        merged_df['context_low_activation'] = 0.0
        merged_df['context_medium_activation'] = 0.0
        merged_df['context_high_activation'] = 0.0
        
        for i, row in merged_df.iterrows():
            low_sum = 0.0
            medium_sum = 0.0  
            high_sum = 0.0
            
            for rule in self.rules:
                co2_level, motion_status, time_level, context_output, weight = rule
                
                # Get membership values for this row
                co2_membership = row[f'co2_{co2_level}_membership']   # 計算co2濃度的模糊歸屬度
                motion_membership = row[f'{motion_status}_membership'] 
                time_membership = row[f'time_{time_level}_membership']
                
                # Calculate rule activation (minimum of antecedents)
                activation = min(co2_membership, motion_membership, time_membership) * weight
                
                # Accumulate activations for each context level
                if context_output == 'low':
                    low_sum += activation
                elif context_output == 'medium':
                    medium_sum += activation
                elif context_output == 'high':
                    high_sum += activation
            
            # Store activation strengths (cap at 1.0)
            merged_df.at[i, 'context_low_activation'] = min(low_sum, 1.0)
            merged_df.at[i, 'context_medium_activation'] = min(medium_sum, 1.0)
            merged_df.at[i, 'context_high_activation'] = min(high_sum, 1.0)
        
        return merged_df
    
    def calculate_context_score(self, merged_df):
        """Calculate motion-weighted context score with clear differentiation"""

        print("==== Calculating Motion-Weighted Context Score ====")
        
        context_scores = []
        
        for i, row in merged_df.iterrows():
            low_activation = row['context_low_activation']
            medium_activation = row['context_medium_activation']
            high_activation = row['context_high_activation']
            
            # Motion-based bonuses to ensure differentiation
            strong_motion_bonus = row['strong_motion_membership'] * 0.15  # Up to +0.15
            light_motion_bonus = row['light_motion_membership'] * 0.08    # Up to +0.08
            no_motion_penalty = row['no_motion_membership'] * 0.05        # Up to -0.05
            
            # Base score calculation with clear thresholds
            if high_activation > 0.7:
                base_score = 0.75
            elif high_activation > 0.4:
                base_score = 0.65
            elif high_activation > 0.2:
                base_score = 0.55
            elif medium_activation > 0.7:
                base_score = 0.50
            elif medium_activation > 0.4:
                base_score = 0.40
            elif medium_activation > 0.2:
                base_score = 0.35
            elif low_activation > 0.5:
                base_score = 0.25
            else:
                base_score = 0.20
            
            # Apply motion adjustments
            adjusted_score = base_score + strong_motion_bonus + light_motion_bonus - no_motion_penalty
            
            # Traditional weighted average as backup
            total_activation = low_activation + medium_activation + high_activation
            if total_activation > 0:
                weighted_score = (low_activation * 0.2 + 
                                medium_activation * 0.5 + 
                                high_activation * 0.8) / total_activation
                
                # Combine methods (favor adjusted score for motion differentiation)
                final_score = 0.8 * adjusted_score + 0.2 * weighted_score
            else:
                final_score = adjusted_score
            
            # Ensure proper bounds
            final_score = max(0.15, min(0.85, final_score))
            context_scores.append(final_score)
        
        merged_df['context_score'] = context_scores
        
        # Enhanced statistics
        unique_scores = len(np.unique(np.round(context_scores, 3)))
        print(f"Motion-Weighted Context Score Statistics:")
        print(f"  Mean: {np.mean(context_scores):.3f}")
        print(f"  Min: {np.min(context_scores):.3f}")
        print(f"  Max: {np.max(context_scores):.3f}")
        print(f"  Std: {np.std(context_scores):.3f}")
        print(f"  Unique values: {unique_scores}")
        print(f"  25th percentile: {np.percentile(context_scores, 25):.3f}")
        print(f"  50th percentile: {np.percentile(context_scores, 50):.3f}")
        print(f"  75th percentile: {np.percentile(context_scores, 75):.3f}")
        
        return merged_df
    
    def analyze_motion_effectiveness(self, merged_df):
        """Comprehensive analysis of motion classification effectiveness"""
        print("==== Analyzing Motion Classification Effectiveness ====")

        # 系統效果檢查
        
        # Calculate motion class dominance
        motion_classes = ['no_motion_membership', 'light_motion_membership', 'strong_motion_membership']
        merged_df['dominant_motion_class'] = merged_df[motion_classes].idxmax(axis=1)
        merged_df['dominant_motion_strength'] = merged_df[motion_classes].max(axis=1)
        
        # Detailed analysis by motion class
        motion_analysis = merged_df.groupby('dominant_motion_class').agg({
            'context_score': ['mean', 'std', 'min', 'max', 'count'],
            'co2_value': 'mean',
            'time_since_last_motion': 'mean',
            'motion_value': 'mean'
        }).round(3)
        
        print("Detailed Motion Class Analysis:")
        print("="*50)
        print(motion_analysis)

        #檢查差異夠大嗎
        
        # Check for proper differentiation
        class_means = merged_df.groupby('dominant_motion_class')['context_score'].mean().round(3)  #檢查分類效果
        print(f"\nContext Score by Motion Class:")
        print("="*30)
        for class_name, mean_score in class_means.items():
            class_label = class_name.replace('_membership', '').replace('_', ' ').title()
            print(f"  {class_label}: {mean_score:.3f}")
        
        # Differentiation check
        motion_means = class_means.reindex(['no_motion_membership', 'light_motion_membership', 'strong_motion_membership'])
        motion_means = motion_means.dropna()
        
        if len(motion_means) >= 2:
            print(f"\nDifferentiation Analysis:")
            print("="*25)
            if 'strong_motion_membership' in motion_means.index and 'light_motion_membership' in motion_means.index:
                strong_vs_light = motion_means['strong_motion_membership'] - motion_means['light_motion_membership']  #差異夠大嗎
                print(f"  Strong vs Light Motion: {strong_vs_light:.3f}")
            
            if 'light_motion_membership' in motion_means.index and 'no_motion_membership' in motion_means.index:
                light_vs_no = motion_means['light_motion_membership'] - motion_means['no_motion_membership']
                print(f"  Light vs No Motion: {light_vs_no:.3f}")
            
            # Check if properly ordered
            if len(motion_means) == 3:   
                is_ordered = (motion_means['strong_motion_membership'] > motion_means['light_motion_membership'] > 
                             motion_means['no_motion_membership'])   #順序正確
                print(f"  Proper ordering (Strong > Light > No): {'✅ YES' if is_ordered else '❌ NO'}")
                
                total_range = motion_means.max() - motion_means.min()   #最大跟最小有相差大於0.1
                print(f"  Total score range: {total_range:.3f}")
                print(f"  Classification effective: {'✅ YES' if total_range > 0.1 else '❌ NO'}")
        
        # Score ranges by class
        score_ranges = merged_df.groupby('dominant_motion_class')['context_score'].apply(
            lambda x: f"{x.min():.3f} - {x.max():.3f}"
        )
        print(f"\nContext Score Ranges by Motion Class:")
        print("="*35)
        for class_name, score_range in score_ranges.items():
            class_label = class_name.replace('_membership', '').replace('_', ' ').title()
            print(f"  {class_label}: {score_range}")
        
        return merged_df
    
    def debug_rule_activations(self, merged_df, sample_size=5):
        """Debug rule activations for sample data points"""
        print("==== Debugging Rule Activations (Sample) ====")
        
        # Sample from different motion classes
        motion_classes = ['no_motion_membership', 'light_motion_membership', 'strong_motion_membership']
        merged_df['dominant_motion'] = merged_df[motion_classes].idxmax(axis=1)
        
        samples = []
        for motion_class in merged_df['dominant_motion'].unique():
            class_data = merged_df[merged_df['dominant_motion'] == motion_class]
            if len(class_data) > 0:
                sample_idx = np.random.choice(class_data.index, size=min(2, len(class_data)), replace=False)
                samples.extend(sample_idx)
        
        for idx in samples[:sample_size]:
            row = merged_df.iloc[idx]
            motion_class = row['dominant_motion'].replace('_membership', '').replace('_', ' ').title()
            
            print(f"\nData Point {idx} - Dominant: {motion_class}")
            print(f"  CO2: {row['co2_value']:.1f}")
            print(f"  Motion: {row['motion_value']:.1f}")
            print(f"  Time since motion: {row.get('time_since_last_motion', 'N/A')}")
            print(f"  Motion memberships:")
            print(f"    No={row['no_motion_membership']:.3f}, Light={row['light_motion_membership']:.3f}, Strong={row['strong_motion_membership']:.3f}")
            print(f"  Context activations:")
            print(f"    Low={row['context_low_activation']:.3f}, Med={row['context_medium_activation']:.3f}, High={row['context_high_activation']:.3f}")
            print(f"  Final Context Score: {row['context_score']:.3f}")
    
    def merge_datasets(self, co2_df, motion_df):
        """Merge CO2 and motion datasets by timestamp"""
        print("==== Merging CO2 and Motion Datasets ====")
        
        # Ensure both have time columns
        co2_time_col = '_time' if '_time' in co2_df.columns else 'timestamp'
        motion_time_col = '_time' if '_time' in motion_df.columns else 'timestamp'
        
        # Convert to datetime if not already
        co2_df[co2_time_col] = pd.to_datetime(co2_df[co2_time_col], format='ISO8601')
        motion_df[motion_time_col] = pd.to_datetime(motion_df[motion_time_col], format='ISO8601')
        
        # Round timestamps to nearest 15 minutes for matching
        co2_df['time_rounded'] = co2_df[co2_time_col].dt.round('15min')
        motion_df['time_rounded'] = motion_df[motion_time_col].dt.round('15min')
        
        # Merge on rounded time
        merged_df = pd.merge(co2_df, motion_df, on='time_rounded', how='inner', suffixes=('_co2', '_motion'))
        
        print(f"Merged dataset: {len(merged_df)} records")
        print(f"Original CO2: {len(co2_df)}, Motion: {len(motion_df)}")
        
        return merged_df
    
    def visualize_context_analysis(self, merged_df):
        """Create comprehensive visualizations for motion-differentiated analysis"""
        print("==== Creating Motion-Differentiated Context Score Visualizations ====")
        
        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        
        # 1. Enhanced Context Score Distribution with statistics
        axes[0, 0].hist(merged_df['context_score'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        mean_score = merged_df['context_score'].mean()
        std_score = merged_df['context_score'].std()
        axes[0, 0].axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.3f}')
        axes[0, 0].axvline(mean_score + std_score, color='orange', linestyle=':', linewidth=2, label=f'+1 STD: {mean_score + std_score:.3f}')
        axes[0, 0].axvline(mean_score - std_score, color='orange', linestyle=':', linewidth=2, label=f'-1 STD: {mean_score - std_score:.3f}')
        axes[0, 0].set_xlabel('Context Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Motion-Differentiated Context Score Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Context Score Over Time
        axes[0, 1].plot(merged_df.index, merged_df['context_score'], alpha=0.7, color='purple', linewidth=0.5)
        axes[0, 1].set_xlabel('Data Point')
        axes[0, 1].set_ylabel('Context Score')
        axes[0, 1].set_title('Context Score Over Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Rule Activation Strengths
        activation_cols = ['context_low_activation', 'context_medium_activation', 'context_high_activation']
        colors = ['red', 'orange', 'green']
        for i, (col, color) in enumerate(zip(activation_cols, colors)):
            axes[0, 2].plot(merged_df.index, merged_df[col], alpha=0.7, color=color, linewidth=0.5,
                           label=col.replace('context_', '').replace('_activation', ''))
        axes[0, 2].set_xlabel('Data Point')
        axes[0, 2].set_ylabel('Activation Strength')
        axes[0, 2].set_title('Rule Activation Strengths')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. CO2 vs Context Score
        scatter = axes[1, 0].scatter(merged_df['co2_value'], merged_df['context_score'], 
                                   c=merged_df['context_score'], cmap='RdYlGn', alpha=0.6, s=15)
        axes[1, 0].set_xlabel('CO2 Value')
        axes[1, 0].set_ylabel('Context Score')
        axes[1, 0].set_title('CO2 vs Context Score')
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0])
        
        # 5. ENHANCED Motion Class Impact - The key improvement
        motion_classes = ['no_motion_membership', 'light_motion_membership', 'strong_motion_membership']
        motion_labels = ['No Motion', 'Light Motion', 'Strong Motion']
        motion_colors = ['lightcoral', 'orange', 'lightgreen']
        
        # Find dominant motion class for each row
        merged_df['dominant_motion'] = merged_df[motion_classes].idxmax(axis=1)
        motion_stats = merged_df.groupby('dominant_motion')['context_score'].agg(['mean', 'std', 'count'])
        
        means = [motion_stats.loc[f'{cls}', 'mean'] if f'{cls}' in motion_stats.index else 0 for cls in motion_classes]
        stds = [motion_stats.loc[f'{cls}', 'std'] if f'{cls}' in motion_stats.index else 0 for cls in motion_classes]
        counts = [motion_stats.loc[f'{cls}', 'count'] if f'{cls}' in motion_stats.index else 0 for cls in motion_classes]
        
        bars = axes[1, 1].bar(motion_labels, means, color=motion_colors, alpha=0.7, 
                             edgecolor='black', yerr=stds, capsize=5)
        axes[1, 1].set_ylabel('Average Context Score')
        axes[1, 1].set_title('Motion Class Impact on Context Score\n(With Standard Deviation)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels and counts on bars
        for bar, mean_val, count in zip(bars, means, counts):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{mean_val:.3f}\n(n={count})', ha='center', va='bottom', fontweight='bold')
        
        # 6. Time Since Motion vs Context Score
        valid_mask = merged_df['time_since_last_motion'].notna()
        if valid_mask.sum() > 0:
            scatter2 = axes[1, 2].scatter(merged_df[valid_mask]['time_since_last_motion'], 
                                        merged_df[valid_mask]['context_score'],
                                        c=merged_df[valid_mask]['context_score'], 
                                        cmap='RdYlGn', alpha=0.6, s=15)
            axes[1, 2].set_xlabel('Time Since Last Motion (minutes)')
            axes[1, 2].set_ylabel('Context Score')
            axes[1, 2].set_title('Time Since Motion vs Context Score')
            axes[1, 2].grid(True, alpha=0.3)
            plt.colorbar(scatter2, ax=axes[1, 2])
        
        # 7. Motion Value vs Context Score with motion class coloring
        for i, (motion_class, color, label) in enumerate(zip(motion_classes, motion_colors, motion_labels)):
            mask = merged_df['dominant_motion'] == motion_class
            if mask.sum() > 0:
                axes[2, 0].scatter(merged_df[mask]['motion_value'], merged_df[mask]['context_score'], 
                                  c=color, alpha=0.6, s=15, label=label)
        axes[2, 0].set_xlabel('Motion Value')
        axes[2, 0].set_ylabel('Context Score')
        axes[2, 0].set_title('Motion Value vs Context Score\n(Colored by Dominant Motion Class)')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 8. Motion Membership Distribution
        for i, (mem, color, label) in enumerate(zip(motion_classes, motion_colors, motion_labels)):
            axes[2, 1].hist(merged_df[mem], bins=30, alpha=0.6, color=color, label=label)
        axes[2, 1].set_xlabel('Membership Degree')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].set_title('Motion Membership Distribution')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        # 9. Context Score by Time of Day with motion class breakdown
        if 'time_rounded' in merged_df.columns:
            merged_df['hour'] = merged_df['time_rounded'].dt.hour
            
            # Overall hourly pattern
            hourly_context = merged_df.groupby('hour')['context_score'].mean()
            hourly_std = merged_df.groupby('hour')['context_score'].std()
            
            axes[2, 2].errorbar(hourly_context.index, hourly_context.values, 
                              yerr=hourly_std.values, fmt='o-', color='purple', 
                              linewidth=2, capsize=3, capthick=1, alpha=0.8, label='Overall')
            
            # Add motion class breakdown
            for motion_class, color, label in zip(motion_classes, motion_colors, motion_labels):
                mask = merged_df['dominant_motion'] == motion_class
                if mask.sum() > 10:  # Only plot if enough data
                    hourly_motion = merged_df[mask].groupby('hour')['context_score'].mean()
                    axes[2, 2].plot(hourly_motion.index, hourly_motion.values, 
                                   '--', color=color, alpha=0.7, linewidth=1, label=label)
            
            axes[2, 2].set_xlabel('Hour of Day')
            axes[2, 2].set_ylabel('Average Context Score')
            axes[2, 2].set_title('Context Score by Time of Day\n(Overall + Motion Classes)')
            axes[2, 2].legend()
            axes[2, 2].grid(True, alpha=0.3)
            axes[2, 2].set_xlim(0, 23)
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self, co2_file_path, motion_file_path):
        """Run complete motion-differentiated context score analysis"""
        print("="*80)
        print("FINAL MOTION-DIFFERENTIATED CONTEXT SCORE MODULE")
        print("="*80)
        
        # Load data
        co2_df = self.load_co2_data(co2_file_path)
        motion_df = self.load_motion_data(motion_file_path)
        
        if co2_df is None or motion_df is None:
            print("Failed to load data")
            return None
        
        # Process motion data
        motion_df = self.calculate_time_since_last_motion(motion_df)
        
        # Calculate membership functions
        co2_df = self.calculate_co2_membership(co2_df)
        motion_df = self.calculate_motion_membership(motion_df)  # Properly differentiated motion
        motion_df = self.calculate_time_membership(motion_df)
        
        # Merge datasets
        merged_df = self.merge_datasets(co2_df, motion_df)
        
        # Define rules and calculate context score
        self.define_context_rules()
        merged_df = self.calculate_rule_activation(merged_df)
        merged_df = self.calculate_context_score(merged_df)  # Motion-weighted scoring
        
        # Analyze motion effectiveness
        merged_df = self.analyze_motion_effectiveness(merged_df)
        
        # Debug rule activations
        self.debug_rule_activations(merged_df, sample_size=5)
        
        # Visualize results
        self.visualize_context_analysis(merged_df)
        
        print("="*80)
        print("MOTION-DIFFERENTIATED CONTEXT SCORE ANALYSIS COMPLETE")
        print("="*80)
        
        return merged_df

# Usage example
if __name__ == "__main__":
    # Initialize Final Context Score Module
    context_module = ContextScoreModule()
    
    # File paths
    co2_file_path = 'C:/Users/王俞文/Documents/glasgow/msc project/data/6CL-IE2 living room co2 30days_15min.csv'
    motion_file_path = 'C:/Users/王俞文/Documents/glasgow/msc project/data/6CL-IE2 living room motion 30days_15min.csv'
    
    # Run complete motion-differentiated analysis
    result_df = context_module.run_complete_analysis(co2_file_path, motion_file_path)
    
    # Display comprehensive results
    if result_df is not None:
        print("\nFinal Motion-Differentiated Context Score Results:")
        print("="*55)
        
        # Sample data
        context_cols = ['co2_value', 'motion_value', 'time_since_last_motion', 
                       'no_motion_membership', 'light_motion_membership', 'strong_motion_membership',
                       'context_low_activation', 'context_medium_activation', 
                       'context_high_activation', 'context_score']
        print("\nSample Data:")
        print(result_df[context_cols].head(15).round(3))
        
        # Motion differentiation summary
        motion_classes = ['no_motion_membership', 'light_motion_membership', 'strong_motion_membership']
        result_df['dominant_motion'] = result_df[motion_classes].idxmax(axis=1)
        motion_summary = result_df.groupby('dominant_motion')['context_score'].agg(['mean', 'count'])
        
        print(f"\nMotion Class Performance Summary:")
        print("="*35)
        for motion_type, stats in motion_summary.iterrows():
            motion_name = motion_type.replace('_membership', '').replace('_', ' ').title()
            print(f"{motion_name:15}: Score={stats['mean']:.3f}, Count={int(stats['count'])}")
        
        # Overall statistics
        print(f"\nOverall Context Score Statistics:")
        print("="*32)
        print(f"  Range: {result_df['context_score'].min():.3f} - {result_df['context_score'].max():.3f}")
        print(f"  Mean: {result_df['context_score'].mean():.3f}")
        print(f"  Std Dev: {result_df['context_score'].std():.3f}")
        print(f"  Unique values: {len(result_df['context_score'].unique())}")
        
        # Motion differentiation success check
        motion_means = motion_summary['mean']
        if len(motion_means) >= 3:
            strong_mean = motion_means.get('strong_motion_membership', 0)
            light_mean = motion_means.get('light_motion_membership', 0)
            no_mean = motion_means.get('no_motion_membership', 0)
            
            print(f"\nMotion Differentiation Check:")
            print("="*28)
            print(f"  Strong > Light > No Motion: {strong_mean:.3f} > {light_mean:.3f} > {no_mean:.3f}")
            success = strong_mean > light_mean > no_mean
            print(f"  Differentiation Success: {'✅ YES' if success else '❌ NO'}")
            if success:
                total_range = strong_mean - no_mean
                print(f"  Total Range: {total_range:.3f}")
                print(f"  Quality: {'Excellent' if total_range > 0.2 else 'Good' if total_range > 0.1 else 'Needs Improvement'}")
        
        print("\n" + "="*80)