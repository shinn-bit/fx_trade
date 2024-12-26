import requests
import csv
from datetime import datetime

# Alpha Vantage APIのキーを入力
API_KEY = 'GCQBFJ7AMYQTK8VP.'  # ここにあなたのAPIキーを入力

# 為替データを取得する通貨ペアを設定（例: USD/JPY）
from_currency = 'USD'
to_currency = 'JPY'

# 取得するデータの期間を指定（例: '2024-01-01'から'2024-11-01'まで）
start_date = '2024-01-01'
end_date = '2024-11-01'

# テクニカル指標を取得する関数（SMA, RSI）
def get_technical_indicators(from_currency, to_currency, API_KEY):
    BASE_URL = 'https://www.alphavantage.co/query'
    
    # 1日ごとのデータを取得 (FX_DAILY)
    parameters = {
        "function": "FX_DAILY",             # 1日ごとの為替データ
        "from_symbol": from_currency,
        "to_symbol": to_currency,
        "apikey": API_KEY,
        "datatype": "json"
    }
    response = requests.get(BASE_URL, params=parameters)
    data = response.json()

    # 移動平均線 (SMA) を取得
    sma_parameters = {
        "function": "SMA",                  # 移動平均線
        "symbol": f"{from_currency}{to_currency}",
        "interval": "daily",                # 1日ごとの移動平均線
        "time_period": "30",                # 期間: 30日
        "series_type": "close",
        "apikey": API_KEY,
        "datatype": "json"
    }
    sma_response = requests.get(BASE_URL, params=sma_parameters)
    sma_data = sma_response.json()

    # 相対力指数 (RSI) を取得
    rsi_parameters = {
        "function": "RSI",                  # 相対力指数
        "symbol": f"{from_currency}{to_currency}",
        "interval": "daily",                # 1日ごとのRSI
        "time_period": "14",                # 期間: 14日
        "series_type": "close",
        "apikey": API_KEY,
        "datatype": "json"
    }
    rsi_response = requests.get(BASE_URL, params=rsi_parameters)
    rsi_data = rsi_response.json()

    return data, sma_data, rsi_data

# テクニカル指標を含む為替データを取得しCSVに保存
def save_to_csv(data, sma_data, rsi_data):
    # CSVファイルに書き込む
    with open('exchange_rate_with_indicators_daily.csv', 'w', newline='') as csvfile:
        fieldnames = ['Date', 'Open', 'High', 'Low', 'Close', 'SMA', 'RSI', 'Volume']  # 'Volume'を追加
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()

        # 為替データとテクニカル指標データを一緒に書き込む
        time_series = data.get("Time Series FX (Daily)", {})
        sma_values = sma_data.get("Technical Analysis: SMA", {})
        rsi_values = rsi_data.get("Technical Analysis: RSI", {})

        for date, values in time_series.items():
            sma_value = sma_values.get(date, {}).get("SMA", "N/A")  # SMAがない場合は"N/A"
            rsi_value = rsi_values.get(date, {}).get("RSI", "N/A")  # RSIがない場合は"N/A"
            
            # Volumeが存在しない場合は'N/A'を代入
            volume = values.get('5. volume', 'N/A')
            
            # 各行をCSVに書き込む
            writer.writerow({
                'Date': date,
                'Open': values['1. open'],
                'High': values['2. high'],
                'Low': values['3. low'],
                'Close': values['4. close'],
                'SMA': sma_value,
                'RSI': rsi_value,
                'Volume': volume  # 'Volume'の項目を追加
            })

    print("為替データとテクニカル指標を 'exchange_rate_with_indicators_daily.csv' に保存しました。")

# 実行
data, sma_data, rsi_data = get_technical_indicators(from_currency, to_currency, API_KEY)
save_to_csv(data, sma_data, rsi_data)

