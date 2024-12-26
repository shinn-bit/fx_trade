import requests

# Alpha VantageのAPIキーを設定
api_key = 'GCQBFJ7AMYQTK8VP.'

# 通貨ペアを指定（USD/JPY）
symbol = 'USDJPY'

# 取得するテクニカル指標の種類（例：SMA）
function = 'SMA'
interval = 'daily'
time_period = 14

# APIリクエストを送信
url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&interval={interval}&time_period={time_period}&apikey={api_key}'
response = requests.get(url)

# レスポンスをJSON形式で取得
data = response.json()

# テクニカル指標を表示
print(data)
