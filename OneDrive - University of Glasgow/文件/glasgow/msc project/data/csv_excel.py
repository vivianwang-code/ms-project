import pandas as pd

# 讀取CSV
df = pd.read_csv('phantom_load_monitoring_log.csv')

# 存成Excel
df.to_excel('phantom_load_monitoring_log.xlsx', index=False)