from influxdb_client import InfluxDBClient
import time
from datetime import datetime

url = "http://aties-digital-twin.ddns.net:1003"
token = "VTPT42ftuyixhrddaurUEnBWeVA3vBiZONqf5eDCADAUc-8LZSEoLKJdt98oshQp6ZM7l0HQFsdrzIOnI6-11A=="
org = "myorg"
bucket = "iotproject"

def get_latest_power_data():
    """獲取最新的power數據"""
    print("==== get latest power data ====")
    client = InfluxDBClient(url=url, token=token, org=org)
    query_api = client.query_api()
    
    # 查詢過去5分鐘的最新數據
    query = f'''
    from(bucket: "{bucket}")
      |> range(start: -5m)
      |> filter(fn: (r) => r._measurement == "influxdb-JWN-D6" and r._field == "power")
      |> sort(columns: ["_time"], desc: true)
      |> limit(n: 1)
    '''
    
    try:
        tables = query_api.query(query)
        for table in tables:
            for record in table.records:
                client.close()
                return {
                    'time': record.get_time(),
                    'value': record.get_value()
                }
    except Exception as e:
        print(f"查詢錯誤：{e}")
    
    return None

def main():
    print("Start real-time monitoring of D6 power data...")
    print("time interval：15 minutes")
    print("press Ctrl+C to stop monitor")
    print("-" * 50)
    
    try:
        while True:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data = get_latest_power_data()
            
            if data:
                print(f"[{current_time}] Power: {data['value']} (time: {data['time']})")
            else:
                print(f"[{current_time}] didn't get the data")
            
            # 等待15分鐘後再次查詢
            time.sleep(20)  # 15分鐘 = 900秒
            
    except KeyboardInterrupt:
        print("\nmonitor stop")

if __name__ == "__main__":
    main()