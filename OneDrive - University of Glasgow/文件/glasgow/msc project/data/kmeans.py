import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score

def read_file(file_path):
    df = pd.read_csv(file_path)
    print('data information')
    print(f"total data : {len(df)}")
    
    return df

def get_power_classification(df):

    print("kmeans power classification : ")

    zero_power = df[df['power'] == 0]
    nonzero_power = df[df['power'] != 0]

    print(f'zero_power : {len(zero_power)}')
    print(f'nonzero_power : {len(nonzero_power)}')

    # Create a Series with the same index as the original DataFrame to store results
    power_categories = pd.Series(index=df.index, dtype='object')

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

        # Create label mapping
        label_name = ['phantom load', 'light use', 'regular use']
        label_map = {}
        
        for i, cluster_id in enumerate(cluster_sort.index):
            label_map[cluster_id] = label_name[i]

        # Assign categories for non-zero power data
        for idx, cluster in zip(nonzero_power.index, clusters):
            power_categories.loc[idx] = label_map[cluster]

        # Evaluate clustering performance
        evalute_kmeans_silhouette(power_kmeans_data, kmeans)
    
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

    return power_categories

def kmeans_power(df):
    """
    Original kmeans_power function that returns complete DataFrame
    """
    print("kmeans power classification : ")

    zero_power = df[df['power'] == 0]
    nonzero_power = df[df['power'] != 0]

    print(f'zero_power : {len(zero_power)}')
    print(f'nonzero_power : {len(nonzero_power)}')

    #classification : 3 classes (phantom load / light use / regular use)
    k = 3
    power_kmeans_data = nonzero_power['power'].values.reshape(-1,1)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    nonzero_power = nonzero_power.copy() 
    nonzero_power['cluster'] = kmeans.fit_predict(power_kmeans_data)

    #label zero power as -1 cluster
    zero_power = zero_power.copy() 
    zero_power['cluster'] = -1   

    #combine zero_power + nonzero_power
    result = pd.concat([nonzero_power,zero_power]).sort_index()

    #cluster statistics
    cluster_statistics = nonzero_power.groupby('cluster')['power'].agg(['count', 'min', 'max', 'mean']).round(2)
    print(f"cluster statistics : \n{cluster_statistics}")

    #sort cluster
    cluster_sort = nonzero_power.groupby('cluster')['power'].mean().sort_values()

    label = {-1: 'non use'}
    label_name = ['phantom load', 'light use', 'regular use']
    
    for i, cluster_id in enumerate(cluster_sort.index):
        label[cluster_id] = label_name[i]

    result['power_category'] = result['cluster'].map(label)

    print('result category')
    for category in result['power_category'].unique():
        subset = result[result['power_category'] == category]
        count_subset = len(subset)
        percentage = count_subset / len(result) * 100

        if subset['power'].max() > 0:
            print(f"{category}: {count_subset} ({percentage:.1f}%) - {subset['power'].min():.1f}W ~ {subset['power'].max():.1f}W")
        else:
            print(f"{category}: {count_subset} ({percentage:.1f}%)")

    evalute_kmeans_silhouette(power_kmeans_data, kmeans)
        
    return result

def evalute_kmeans_silhouette(power_kmeans_data, kmeans):
    silhouette_avg = silhouette_score(power_kmeans_data, kmeans.labels_)
    print(f"silhouette score: {silhouette_avg:.3f}")

def plot_kmeans_power(df):
    fig, axes = plt.subplots(2, 2, figsize=(12,10))

    #figure 1
    ax1 = axes[0, 0]

    categories = df['power_category'].unique()
    colors = ['orange', 'lightblue', 'lightgreen']

    color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(categories)}

    for category in df['power_category'].unique():
        if category != 'non use':
            data = df[df['power_category'] == category]['power']
            if len(data) > 0:
                ax1.hist(data, alpha=0.6, label=category, bins=15, color=color_map[category])
    ax1.set_xlabel('power (W)')
    ax1.set_ylabel('times')
    ax1.set_title('power distribution')
    ax1.legend()

    #figure 2
    ax2 = axes[0,1]
    counts = df['power_category'].value_counts()
    colors_pie = ['red', 'lightgreen', 'lightblue', 'orange']
    
    wedges, texts, autotexts = ax2.pie(counts.values, labels=counts.index, autopct='%1.1f%%', 
                                      colors=colors_pie, startangle=90)
    ax2.set_title('Proportion of each category')

    #figure 3
    ax3 = axes[1, 0]
    nonzero_categories = []
    nonzero_data = []
    
    for category in categories:
        data = df[df['power_category'] == category]['power']
        if len(data) > 0 and data.max() > 0:  # Exclude 'non use'
            nonzero_categories.append(f'{category}\n(n={len(data)})')   
            nonzero_data.append(data.tolist())
    
    if nonzero_data:
        bp = ax3.boxplot(nonzero_data, labels=nonzero_categories, patch_artist=True)
        
        # Set colors
        box_colors = ['orange', 'lightblue', 'lightgreen' ]
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
    
    ax3.set_ylabel('power (W)')
    ax3.set_title('Comparison of power ranges by category')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    #figure 4
    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_data = []
    for category in categories:
        subset = df[df['power_category'] == category]
        count = len(subset)
        percentage = count / len(df) * 100
        
        if subset['power'].max() > 0:
            min_power = subset['power'].min()
            max_power = subset['power'].max()
            mean_power = subset['power'].mean()
            stats_data.append([category, count, f'{percentage:.1f}%', 
                             f'{min_power:.1f}', f'{max_power:.1f}', f'{mean_power:.1f}'])
        else:
            stats_data.append([category, count, f'{percentage:.1f}%', '0.0', '0.0', '0.0'])
    
    # Create table
    table = ax4.table(cellText=stats_data,
                     colLabels=['category', 'times', 'proportion', 'min power(W)', 'max power(W)', 'average power(W)'],
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Set table style
    for i in range(len(stats_data) + 1):
        for j in range(6):
            if i == 0:  # Header row
                table[(i, j)].set_facecolor('#40466e')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                table[(i, j)].set_facecolor('#f1f1f2')
    
    ax4.set_title('Detailed statistics by category', pad=20)

    plt.tight_layout()
    plt.show()

def plot_power_classification(df):
    plt.figure(figsize=(12,6))
    # Set category colors and markers
    category_colors = {
        'non use': 'red', 
        'phantom load': 'green', 
        'light use': 'blue', 
        'regular use': 'orange'
    }
    category_markers = {
        'non use': 'o', 
        'phantom load': 's', 
        'light use': '^', 
        'regular use': 'D'
    }
    
    # Create x-axis index
    x_values = range(len(df))
    
    # Plot scatter points for each category
    for category in df['power_category'].unique():
        mask = df['power_category'] == category
        x_cat = np.array(x_values)[mask]
        y_cat = df[mask]['power']
        
        plt.scatter(x_cat, y_cat, 
                   c=category_colors.get(category, 'gray'), 
                   marker=category_markers.get(category, 'o'),
                   alpha=0.7, 
                   label=category, 
                   s=30)
    
    # Calculate and draw threshold lines
    nonzero_categories = ['phantom load', 'light use', 'regular use']
    category_means = {}
    
    # Calculate mean value for each non-zero category
    for category in nonzero_categories:
        if category in df['power_category'].values:
            category_data = df[df['power_category'] == category]['power']
            if len(category_data) > 0:
                category_means[category] = category_data.mean()
            else : 
                print(f'{category} has no data')
    
    # Sort category means
    sorted_means = sorted(category_means.items(), key=lambda x: x[1])
    
    # Calculate thresholds and draw lines
    thresholds = []
    for i in range(len(sorted_means) - 1):
        threshold = (sorted_means[i][1] + sorted_means[i+1][1]) / 2
        thresholds.append(threshold)
        
        # Draw threshold line
        plt.axhline(y=threshold, color='black', linestyle='--', linewidth=2, alpha=0.8)
        plt.text(len(df)*0.02, threshold + plt.ylim()[1]*0.02, 
                f'Threshold: {threshold:.1f}W\n({sorted_means[i][0]} | {sorted_means[i+1][0]})', 
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Draw zero power threshold line
    plt.axhline(y=0, color='red', linestyle='-', linewidth=1, alpha=0.6)
    plt.text(len(df)*0.02, plt.ylim()[1]*0.05, 
            'Zero Power Line', 
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.xlabel('Data Point Index')
    plt.ylabel('Power (W)')
    plt.title('Power Classification Scatter Plot with Thresholds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print threshold information
    print("\n=== Classification Thresholds ===")
    print("Zero power threshold: 0W")
    for i, threshold in enumerate(thresholds):
        if i < len(sorted_means) - 1:
            print(f"Threshold {i+1}: {threshold:.2f}W (between {sorted_means[i][0]} and {sorted_means[i+1][0]})")
    print(f"Total thresholds: {len(thresholds) + 1}")  # +1 for zero line

if __name__ == "__main__":

    # file_path = "C:/Users/王俞文/OneDrive - University of Glasgow/文件/glasgow/msc project/data/6CL-S8 television 15min.csv"
    # file_path = "C:/Users/王俞文/OneDrive - University of Glasgow/文件/glasgow/msc project/data/6CL-S2_washing_machine_15min.csv"
    file_path = "C:/Users/王俞文/OneDrive - University of Glasgow/文件/glasgow/msc project/data/iotproject_D8.csv"

    df = read_file(file_path)

    df = kmeans_power(df)

    plot_kmeans_power(df)

    plot_power_classification(df)