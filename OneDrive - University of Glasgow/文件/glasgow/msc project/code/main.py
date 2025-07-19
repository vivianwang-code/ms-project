import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from modules import kmeans  
from modules import influxdb_history
from modules import preprocessing
from modules import expand_data

if __name__ == "__main__":

    
    file_path = influxdb_history.export_to_csv()

    total_days, output_path = preprocessing.data_preprocessing(file_path)
    print(f"total_days:{total_days}")
    print(f"output_path:{output_path}")
    
    if total_days < 60:
        expand_data.main(output_path)   # 往前新增2個月的data