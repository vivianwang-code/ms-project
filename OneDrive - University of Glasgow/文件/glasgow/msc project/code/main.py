import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from modules import kmeans   
from modules import influxdb_history
from modules import preprocessing
from modules import expand_data
from modules import confidence
from modules import device_activity
from modules import user_habit
from modules import influxdb_phantom_load

# Add logger
from utils.logger_config import get_system_logger

def main():
    # Create system logger
    system_logger = get_system_logger()
    
    system_logger.info("🚀 Smart Power Management System Startup")
    system_logger.info("=" * 70)
    
    try:
        # Step 1: Export historical data
        system_logger.info("Step 1: Starting to export historical data from InfluxDB...")
        file_path = influxdb_history.export_to_csv()
        system_logger.info(f"✅ Step 1 Completed: Historical data exported to {file_path}")
        
        # Step 2: Data preprocessing
        system_logger.info("Step 2: Starting data preprocessing...")
        total_days, output_path = preprocessing.data_preprocessing(file_path)
        system_logger.info(f"✅ Step 2 Completed: Data preprocessing finished")
        system_logger.info(f"   Original data days: {total_days} days")
        system_logger.info(f"   Preprocessed file: {output_path}")
        
        print(f"total_days: {total_days}")
        print(f"output_path: {output_path}")
        
        # Step 3: Check data length and expand if needed
        if total_days < 60:
            system_logger.warning(f"Insufficient data length: {total_days} days < 60 days")
            system_logger.info("Step 3: Starting data expansion, extending data to 2 months...")
            print("\n ============== length of data is less than 2 months ===================")
            
            expand_data_file = expand_data.main(output_path)   # Add 2 months of data backwards
            
            system_logger.info(f"✅ Step 3 Completed: Data expanded to 2 months")
            system_logger.info(f"   Expanded file: {expand_data_file}")
            print(f"expand data file: {expand_data_file}")
            
            # Update file path to expanded file
            final_data_file = expand_data_file
        else:   
            system_logger.info(f"Sufficient data length: {total_days} days >= 60 days, skipping data expansion")
            final_data_file = output_path
        
        # Step 4: Start real-time monitoring
        system_logger.info("Step 4: Starting Phantom Load real-time monitoring...")
        system_logger.info("System initialization completed, entering monitoring mode")
        system_logger.info("-" * 70)
        
        influxdb_phantom_load.main()
        
    except KeyboardInterrupt:
        system_logger.info("🛑 Received interrupt signal, system shutting down normally")
        print("\nSystem safely closed")
        
    except Exception as e:
        system_logger.error(f"❌ System execution error: {e}")
        system_logger.error("System will terminate, please check error logs")
        print(f"\n❌ System error: {e}")
        raise

if __name__ == "__main__":
    main()