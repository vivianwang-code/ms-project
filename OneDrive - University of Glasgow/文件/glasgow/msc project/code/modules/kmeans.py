import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import json
import os
from datetime import datetime

def save_thresholds_to_config(thresholds):
    """將 threshold 保存到 config 檔案的函數"""
    
    # 確保 config 資料夾存在
    os.makedirs('config', exist_ok=True)
    
    # 準備要保存的數據
    config_data = {
        'generated_at': datetime.now().isoformat(),
        'method': 'kmeans_3_clusters',
        'thresholds': thresholds,
        'classification_rules': {
            'non_use': 'power == 0',
            'phantom_load': f'0 < power <= {thresholds.get("phantom_upper", "N/A")}',
            'light_use': f'{thresholds.get("light_lower", "N/A")} < power <= {thresholds.get("light_upper", "N/A")}', 
            'regular_use': f'power > {thresholds.get("regular_upper", "N/A")}'
        }
    }
    
    # 保存到檔案
    filename = 'config/thresholds.json'
    with open(filename, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"\n✅ Thresholds 已保存到: {filename}")
    
    # 顯示保存的內容
    print("\n📊 保存的 Threshold 值:")
    for key, value in thresholds.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value:.3f}")

def get_power_classification(df):
    """您的原始函數 + 保存功能"""
    
    print("kmeans power classification : ")

    zero_power = df[df['power'] == 0]
    nonzero_power = df[df['power'] != 0]

    print(f'zero_power : {len(zero_power)}')
    print(f'nonzero_power : {len(nonzero_power)}')

    # Create a Series with the same index as the original DataFrame to store results
    power_categories = pd.Series(index=df.index, dtype='object')
    
    # 新增：用於計算和保存 threshold 的變數
    thresholds = {}

    if len(nonzero_power) > 0:
        # Classification: 3 classes (phantom load / light use / regular use)
        k = 3
        power_kmeans_data = nonzero_power['power'].values.reshape(-1,1)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(power_kmeans_data)

        # Create temporary DataFrame to handle clustering
        temp_df = nonzero_power.copy()
        temp_df['cluster'] = clusters

        # Cluster statistics
        cluster_statistics = temp_df.groupby('cluster')['power'].agg(['count', 'min', 'max', 'mean']).round(2)
        print(f"cluster statistics : \n{cluster_statistics}")

        # Sort cluster by mean power
        cluster_sort = temp_df.groupby('cluster')['power'].mean().sort_values()

        # 新增：計算 threshold 邊界值
        sorted_means = cluster_sort.values
        
        if len(sorted_means) >= 2:
            # phantom 和 light 的邊界
            phantom_threshold = (sorted_means[0] + sorted_means[1]) / 2
            thresholds['phantom_upper'] = float(phantom_threshold)
            thresholds['light_lower'] = float(phantom_threshold)
        
        if len(sorted_means) >= 3:
            # light 和 regular 的邊界
            light_threshold = (sorted_means[1] + sorted_means[2]) / 2
            thresholds['light_upper'] = float(light_threshold)
            thresholds['regular_lower'] = float(phantom_threshold)
            thresholds['regular_upper'] = float(light_threshold)
            thresholds['heavy_lower'] = float(light_threshold)
        
        # 新增：保存聚類中心資訊
        thresholds['cluster_centers'] = {
            'phantom_center': float(sorted_means[0]),
            'light_center': float(sorted_means[1]) if len(sorted_means) > 1 else None,
            'regular_center': float(sorted_means[2]) if len(sorted_means) > 2 else None
        }

        # Create label mapping
        label_name = ['phantom load', 'light use', 'regular use']
        label_map = {}
        
        for i, cluster_id in enumerate(cluster_sort.index):
            label_map[cluster_id] = label_name[i]

        # Assign categories for non-zero power data
        for idx, cluster in zip(nonzero_power.index, clusters):
            power_categories.loc[idx] = label_map[cluster]
    
    # Assign category for zero power data
    power_categories.loc[zero_power.index] = 'non use'

    # Print result statistics
    print('result category')
    for category in power_categories.unique():
        if pd.notna(category):  # Exclude NaN values
            subset_mask = power_categories == category
            subset_power = df.loc[subset_mask, 'power']
            count_subset = subset_mask.sum()
            percentage = count_subset / len(df) * 100

            if len(subset_power) > 0 and subset_power.max() > 0:
                print(f"{category}: {count_subset} ({percentage:.1f}%) - {subset_power.min():.1f}W ~ {subset_power.max():.1f}W")
            else:
                print(f"{category}: {count_subset} ({percentage:.1f}%)")

    # 新增：調用保存函數
    if thresholds:  # 確保有計算出 threshold
        save_thresholds_to_config(thresholds)
    else:
        print("⚠️  沒有非零數據，無法計算 threshold")

    return power_categories

# 新增：載入 threshold 的輔助函數
def load_thresholds_from_config():
    """從 config 檔案載入 threshold"""
    try:
        with open('config/thresholds.json', 'r') as f:
            data = json.load(f)
        print("✅ 成功載入 threshold 配置")
        return data['thresholds']
    except FileNotFoundError:
        print("❌ 找不到 thresholds.json 檔案")
        return None
    except Exception as e:
        print(f"❌ 載入 threshold 時發生錯誤: {e}")
        return None

# 新增：使用 threshold 進行分類的函數
def classify_power_with_thresholds(power_value, thresholds=None):
    """使用保存的 threshold 來分類功率值"""
    
    if thresholds is None:
        thresholds = load_thresholds_from_config()
        if thresholds is None:
            raise Exception("無法載入 threshold，請先執行 K-means 分析")
    
    if power_value == 0:
        return 'non_use'
    elif power_value <= thresholds['phantom_upper']:
        return 'phantom_load'
    elif power_value <= thresholds['light_upper']:
        return 'light_use'
    else:
        return 'regular_use'