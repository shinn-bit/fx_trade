import requests
import csv
from datetime import datetime

# Alpha Vantage APIのキーを入力
API_KEY = 'GCQBFJ7AMYQTK8VP.'  # ここにあなたのAPIキーを入力

# 為替データを取得する通貨ペアを設定（例: USD/JPY）
from_currency = 'USD'
to_currency = 'JPY'

# 取得するデータの期間を指定（例: '2024-01-01'から'2024-11-01'まで）
start_date = '2004-01-01'
end_date = '2024-11-01'

# APIのURLとパラメータを設定
BASE_URL = 'https://www.alphavantage.co/query'
parameters = {
    "function": "FX_DAILY",          # 日足データ
    "from_symbol": from_currency,    # 変換元通貨（USD）
    "to_symbol": to_currency,        # 変換先通貨（JPY）
    "apikey": API_KEY,
    "datatype": "json",               # データ形式
    "outputsize": "full"              # 可能な限り全データを取得
}

# APIにリクエストを送信してデータを取得
response = requests.get(BASE_URL, params=parameters)
data = response.json()

# 取得したデータから為替レートの情報を抽出
if "Time Series FX (Daily)" in data:
    time_series = data["Time Series FX (Daily)"]

    # CSVファイルに保存
    with open('exchange_rate_data_filtered.csv', 'w', newline='') as csvfile:
        fieldnames = ['Date', 'Open', 'High', 'Low', 'Close']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        # 時系列データをフィルタリングしてCSVに書き込む
        for date, values in time_series.items():
            # 日付をフィルタリング
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')

            # 開始日と終了日の範囲内であれば書き込む
            if start_date_obj <= date_obj <= end_date_obj:
                writer.writerow({
                    'Date': date,
                    'Open': values['1. open'],
                    'High': values['2. high'],
                    'Low': values['3. low'],
                    'Close': values['4. close']
                })

    print(f"指定した期間の為替データを 'exchange_rate_data_filtered.csv' に保存しました。")
else:
    print("データ取得に失敗しました。APIのレスポンス:", data)
