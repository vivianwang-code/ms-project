import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score

def read_file(file_path):
    """Read data file"""
    df = pd.read_csv(file_path)
    print('Original data information')
    print(f"Total data points: {len(df)}")
    print(f"Power range: {df['power'].min():.2f}W ~ {df['power'].max():.2f}W")
    return df

def step1_remove_low_power(df, threshold=1.5):
    """
    Step 1: Remove low power values (e.g., 0~1.5W), treated as standby values
    """
    print(f"\n=== Step 1: Remove low power values (≤{threshold}W) ===")
    
    # Separate low power and valid power data
    low_power_mask = df['power'] <= threshold
    valid_power_mask = df['power'] > threshold
    
    low_power_data = df[low_power_mask].copy()
    valid_power_data = df[valid_power_mask].copy()
    
    print(f"Low power data (≤{threshold}W): {len(low_power_data)} records")
    print(f"Valid power data (>{threshold}W): {len(valid_power_data)} records")
    
    if len(valid_power_data) > 0:
        print(f"Valid power range: {valid_power_data['power'].min():.2f}W ~ {valid_power_data['power'].max():.2f}W")
    
    return low_power_data, valid_power_data, threshold

def step2_kmeans_clustering(valid_power_data, k=3):
    """
    Step 2: Perform K-Means clustering on remaining data (K=3)
    """
    print(f"\n=== Step 2: K-Means clustering analysis (K={k}) ===")
    
    if len(valid_power_data) == 0:
        print("Warning: No valid power data for clustering")
        return valid_power_data, None
    
    # Prepare clustering data
    power_values = valid_power_data['power'].values.reshape(-1, 1)
    
    # Perform K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(power_values)
    
    # Add clustering results
    valid_power_data = valid_power_data.copy()
    valid_power_data['cluster'] = clusters
    
    # Display clustering statistics
    cluster_stats = valid_power_data.groupby('cluster')['power'].agg(['count', 'min', 'max', 'mean']).round(2)
    print("Clustering statistics:")
    print(cluster_stats)
    
    # Evaluate clustering performance
    if len(valid_power_data) > k:
        silhouette_avg = silhouette_score(power_values, clusters)
        print(f"Silhouette Score: {silhouette_avg:.3f}")
    
    return valid_power_data, kmeans

def step3_label_clusters(valid_power_data):
    """
    Step 3: Sort clusters by center and label as phantom / light / regular
    """
    print(f"\n=== Step 3: Cluster labeling assignment ===")
    
    if len(valid_power_data) == 0 or 'cluster' not in valid_power_data.columns:
        print("Warning: No clustering results to label")
        return valid_power_data
    
    # Calculate average power for each cluster and sort
    cluster_means = valid_power_data.groupby('cluster')['power'].mean().sort_values()
    
    # Define labels
    labels = ['phantom', 'light', 'regular']
    
    # Create label mapping
    label_mapping = {}
    for i, cluster_id in enumerate(cluster_means.index):
        label_mapping[cluster_id] = labels[i] if i < len(labels) else f'cluster_{i}'
    
    # Apply labels
    valid_power_data = valid_power_data.copy()
    valid_power_data['power_category'] = valid_power_data['cluster'].map(label_mapping)
    
    # Display labeling results
    print("Cluster labeling results:")
    for cluster_id, label in label_mapping.items():
        cluster_data = valid_power_data[valid_power_data['cluster'] == cluster_id]
        count = len(cluster_data)
        mean_power = cluster_data['power'].mean()
        min_power = cluster_data['power'].min()
        max_power = cluster_data['power'].max()
        print(f"  {label}: {count} records, mean power {mean_power:.2f}W ({min_power:.2f}W ~ {max_power:.2f}W)")
    
    return valid_power_data

def step4_combine_results(low_power_data, valid_power_data, threshold):

    print(f"\n=== Step 4: Combine results and final classification ===")
    
    # 🔧 修正：檢查 low_power_data 是否為空
    if not low_power_data.empty:
        # Assign labels to low power data
        low_power_data = low_power_data.copy()
        
        # 確保先添加必要的列
        low_power_data['cluster'] = -1  # Mark low power cluster as -1
        
        # 創建 mask
        zero_power_mask = low_power_data['power'] == 0
        micro_power_mask = (low_power_data['power'] > 0) & (low_power_data['power'] <= threshold)

        # 初始化 power_category 列
        low_power_data['power_category'] = ''
        
        # 分配標籤
        low_power_data.loc[zero_power_mask, 'power_category'] = 'no-use'
        low_power_data.loc[micro_power_mask, 'power_category'] = 'phantom'
    else:
        # 🔧 如果 low_power_data 為空，創建空的結構
        print("No low power data found.")
    
    # Combine all data
    if len(valid_power_data) > 0 and not low_power_data.empty:
        final_result = pd.concat([valid_power_data, low_power_data]).sort_index()
    elif len(valid_power_data) > 0:
        final_result = valid_power_data.sort_index()
    elif not low_power_data.empty:
        final_result = low_power_data.sort_index()
    else:
        # 🔧 如果兩個都為空，創建空的 DataFrame
        final_result = pd.DataFrame(columns=['power', 'cluster', 'power_category'])
    
    
    # Display final statistics
    print("Final classification results:")
    category_stats = []
    
    for category in final_result['power_category'].unique():
        if pd.notna(category):
            subset = final_result[final_result['power_category'] == category]
            count = len(subset)
            percentage = count / len(final_result) * 100
            
            if subset['power'].max() > 0:
                min_power = subset['power'].min()
                max_power = subset['power'].max()
                mean_power = subset['power'].mean()
                print(f"  {category}: {count} records ({percentage:.1f}%) - {min_power:.2f}W ~ {max_power:.2f}W (mean: {mean_power:.2f}W)")
                category_stats.append([category, count, f'{percentage:.1f}%', 
                                     f'{min_power:.2f}', f'{max_power:.2f}', f'{mean_power:.2f}'])
            else:
                print(f"  {category}: {count} records ({percentage:.1f}%) - 0.00W")
                category_stats.append([category, count, f'{percentage:.1f}%', '0.00', '0.00', '0.00'])
    
    return final_result, category_stats

def get_power_classification(df, threshold=1.5, k=3):
    """
    Return power classification Series, compatible with existing preprocessing workflow
    This function is specifically for compatibility with preprocessing.py
    """
    print("=== Starting new power classification workflow ===")
    
    # Step 1: Remove low power values
    low_power_data, valid_power_data, used_threshold = step1_remove_low_power(df, threshold)
    
    # Step 2: K-Means clustering
    valid_power_data, kmeans = step2_kmeans_clustering(valid_power_data, k)
    
    # Step 3: Label clusters
    valid_power_data = step3_label_clusters(valid_power_data)
    
    # Step 4: Combine results
    final_result, category_stats = step4_combine_results(low_power_data, valid_power_data, used_threshold)
    
    # Return only the classification Series to maintain compatibility with original code
    return final_result['power_category']

def get_power_classification_full(df, threshold=1.5, k=3):
    """
    Complete power classification function, returning full results and statistics
    Used for detailed analysis and plotting
    """
    print("=== Starting complete power classification workflow ===")
    
    # Step 1: Remove low power values
    low_power_data, valid_power_data, used_threshold = step1_remove_low_power(df, threshold)
    
    # Step 2: K-Means clustering
    valid_power_data, kmeans = step2_kmeans_clustering(valid_power_data, k)
    
    # Step 3: Label clusters
    valid_power_data = step3_label_clusters(valid_power_data)
    
    # Step 4: Combine results
    final_result, category_stats = step4_combine_results(low_power_data, valid_power_data, used_threshold)
    
    return final_result, category_stats

def kmeans_power(df):
    """
    Compatible kmeans_power function for existing code
    This function maintains the original interface but uses new classification logic
    """
    print("Using new four-step classification method...")
    
    # Call new complete classification function
    final_result, category_stats = get_power_classification_full(df)
    
    # Add necessary columns for compatibility
    if 'cluster' not in final_result.columns:
        final_result['cluster'] = -1
    
    final_result['power_state'] = final_result['power_category']  # Map to old column name
    
    return final_result

def evalute_kmeans_silhouette(power_kmeans_data, kmeans):
    """Silhouette score evaluation compatible with existing code"""
    silhouette_avg = silhouette_score(power_kmeans_data, kmeans.labels_)
    print(f"silhouette score: {silhouette_avg:.3f}")

def plot_new_classification_results(df, category_stats):
    """
    Plot charts for new classification results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Set color mapping
    color_map = {
        'no-use': 'red',
        'phantom': 'orange', 
        'light': 'lightblue',
        'regular': 'lightgreen'
    }
    
    # Chart 1: Power distribution histogram
    ax1 = axes[0, 0]
    for category in df['power_category'].unique():
        if category != 'no-use' and pd.notna(category):
            data = df[df['power_category'] == category]['power']
            if len(data) > 0:
                ax1.hist(data, alpha=0.7, label=category, bins=20, 
                        color=color_map.get(category, 'gray'))
    
    ax1.set_xlabel('Power (W)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Power Distribution by Category')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Chart 2: Pie chart
    ax2 = axes[0, 1]
    counts = df['power_category'].value_counts()
    colors = [color_map.get(cat, 'gray') for cat in counts.index]
    
    wedges, texts, autotexts = ax2.pie(counts.values, labels=counts.index, 
                                      autopct='%1.1f%%', colors=colors, 
                                      startangle=90)
    ax2.set_title('Category Proportion Distribution')
    
    # Chart 3: Box plot
    ax3 = axes[1, 0]
    box_data = []
    box_labels = []
    box_colors = []
    
    for category in ['phantom', 'light', 'regular']:
        if category in df['power_category'].values:
            data = df[df['power_category'] == category]['power']
            if len(data) > 0:
                box_data.append(data.tolist())
                box_labels.append(f'{category}\n(n={len(data)})')
                box_colors.append(color_map.get(category, 'gray'))
    
    if box_data:
        bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
    
    ax3.set_ylabel('Power (W)')
    ax3.set_title('Power Range Comparison by Category')
    ax3.grid(True, alpha=0.3)
    
    # Chart 4: Statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    if category_stats:
        table = ax4.table(cellText=category_stats,
                         colLabels=['Category', 'Count', 'Proportion', 'Min Power(W)', 'Max Power(W)', 'Mean Power(W)'],
                         cellLoc='center',
                         loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Set table style
        for i in range(len(category_stats) + 1):
            for j in range(6):
                if i == 0:  # Header row
                    table[(i, j)].set_facecolor('#40466e')
                    table[(i, j)].set_text_props(weight='bold', color='white')
                else:
                    table[(i, j)].set_facecolor('#f1f1f2')
    
    ax4.set_title('Detailed Statistics by Category', pad=20)
    
    plt.tight_layout()
    plt.show()

def plot_step_by_step_process(df, final_result, threshold=1.5):
    """
    Visualize step-by-step processing workflow
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Step 1: Original data vs filtered data
    ax1 = axes[0, 0]
    ax1.hist(df['power'], bins=30, alpha=0.7, color='lightgray', label='Original data')
    valid_data = df[df['power'] > threshold]['power']
    if len(valid_data) > 0:
        ax1.hist(valid_data, bins=30, alpha=0.7, color='blue', label=f'Valid data (>{threshold}W)')
    ax1.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold {threshold}W')
    ax1.set_xlabel('Power (W)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Step 1: Data Filtering')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Step 2-3: Clustering results
    ax2 = axes[0, 1]
    color_map = {'phantom': 'orange', 'light': 'lightblue', 'regular': 'lightgreen'}
    
    for category in ['phantom', 'light', 'regular']:
        if category in final_result['power_category'].values:
            data = final_result[final_result['power_category'] == category]['power']
            if len(data) > 0:
                ax2.scatter(range(len(data)), sorted(data), 
                           alpha=0.6, label=category, color=color_map[category])
    
    ax2.set_xlabel('Data Point Index')
    ax2.set_ylabel('Power (W)')
    ax2.set_title('Step 2-3: Clustering Results')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Step 4: Final results time series
    ax3 = axes[1, 0]
    category_colors = {
        'no-use': 'red', 'phantom': 'orange', 
        'light': 'lightblue', 'regular': 'lightgreen'
    }
    
    x_values = range(len(final_result))
    for category in final_result['power_category'].unique():
        if pd.notna(category):
            mask = final_result['power_category'] == category
            x_cat = np.array(x_values)[mask]
            y_cat = final_result[mask]['power']
            
            ax3.scatter(x_cat, y_cat, 
                       c=category_colors.get(category, 'gray'), 
                       alpha=0.6, label=category, s=20)
    
    ax3.set_xlabel('Data Point Index')
    ax3.set_ylabel('Power (W)')
    ax3.set_title('Step 4: Final Classification Results')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Processing workflow statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create workflow statistics text
    process_stats = [
        f"Original data total: {len(df)}",
        f"Low power data (≤{threshold}W): {len(df[df['power'] <= threshold])}",
        f"  - Zero power (no-use): {len(df[df['power'] == 0])}",
        f"  - Micro power (phantom): {len(df[(df['power'] > 0) & (df['power'] <= threshold)])}",
        f"Valid power data (>{threshold}W): {len(df[df['power'] > threshold])}",
        "",
        "Final classification statistics:",
    ]
    
    for category in final_result['power_category'].unique():
        if pd.notna(category):
            count = len(final_result[final_result['power_category'] == category])
            percentage = count / len(final_result) * 100
            process_stats.append(f"  {category}: {count} ({percentage:.1f}%)")
    
    ax4.text(0.05, 0.95, '\n'.join(process_stats), 
             transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    ax4.set_title('Processing Workflow Statistics', pad=20)
    
    plt.tight_layout()
    plt.show()

# Compatible plotting functions for legacy code
def plot_kmeans_power(df):
    """Compatible plotting function for existing code"""
    # Get statistics
    final_result, category_stats = get_power_classification_full(df)
    
    # Call new plotting function
    plot_new_classification_results(final_result, category_stats)

def plot_power_classification(df):
    """Compatible classification plotting function for existing code"""
    # Get statistics
    final_result, category_stats = get_power_classification_full(df)
    
    # Call new step plotting function
    plot_step_by_step_process(df, final_result, threshold=1.5)

if __name__ == "__main__":
    # Test file path (modify as needed)
    file_path = "C:/Users/王俞文/OneDrive - University of Glasgow/文件/glasgow/msc project/data/20250707_20250728_D8.csv"
    
    # Read data
    df = read_file(file_path)
    
    # Execute new classification workflow
    final_result, category_stats = get_power_classification_full(df, threshold=1.5, k=3)
    
    # Plot result charts
    plot_new_classification_results(final_result, category_stats)
    
    # Plot processing workflow visualization
    plot_step_by_step_process(df, final_result, threshold=1.5)
    
    print("\n=== Classification Complete ===")
    print(f"Final results contain {len(final_result)} data points")
    print("Data has been reclassified using the new four-step workflow!")