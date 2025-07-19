from influxdb_client import InfluxDBClient
import pandas as pd
import time
from datetime import datetime
import json

url = "http://aties-digital-twin.ddns.net:1003"
token = "VTPT42ftuyixhrddaurUEnBWeVA3vBiZONqf5eDCADAUc-8LZSEoLKJdt98oshQp6ZM7l0HQFsdrzIOnI6-11A=="
org = "myorg"
bucket = "iotproject"

def get_historical_data(months=3):
    print(f"Start getting data for the last {months} months...")
    
    client = InfluxDBClient(url=url, token=token, org=org)
    query_api = client.query_api()

    query = f'''
    from(bucket: "{bucket}")
      |> range(start: -{months}mo)
      |> filter(fn: (r) => r._measurement == "influxdb-JWN-D8" and r._field == "power")
      |> sort(columns: ["_time"], desc: false)
      |> yield(name: "power_data")
    '''
    
    result = query_api.query(query)
    
    data = []
    for table in result:
        for record in table.records:
            data.append({
                'timestamp': record.get_time(),
                'power_value': record.get_value()
            })
    
    print(f"get {len(data)} counts of data")
    return data

def export_to_csv(months=6):
    # 獲取數據
    historical_data = get_historical_data(months=months)
    
    if not historical_data:
        print("no data")
        return
    
    # 轉換成DataFrame
    df = pd.DataFrame(historical_data)
    
    # 匯出CSV
    filename = f'data/historical_power_{months}months.csv'
    df.to_csv(filename, index=False)
    print(f"Data exported to: {filename}")
    
    # 生成資訊檔案
    info = {
        'export_date': datetime.now().isoformat(),
        'months_requested': months,
        'total_records': len(historical_data),
        'date_range': {
            'start': historical_data[0]['timestamp'].isoformat(),
            'end': historical_data[-1]['timestamp'].isoformat()
        },
        'filename': filename
    }
    
    info_filename = f'data/export_D8_{months}months.json'
    with open(info_filename, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Export information has been saved to: {info_filename}")
    print(f"time range: {info['date_range']['start']} to {info['date_range']['end']}")

    return filename

# # 使用方式
# if __name__ == "__main__":
#     export_to_csv(months=6)  # 匯出最近6個月