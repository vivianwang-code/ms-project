import pandas as pd

# 讀取CSV
df = pd.read_csv('C:/Users/王俞文/OneDrive - University of Glasgow/文件/glasgow/msc project/data/batch_phantom_load_analysis_log.csv')

# 存成Excel
df.to_excel('phantom_load_monitoring_log_20250723.xlsx', index=False)