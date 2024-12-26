import os
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime
from dateutil.relativedelta import relativedelta

# fetch_fx_data関数は、指定された期間の為替レートデータを取得する関数です。
def fetch_fx_data(start_date, end_date):
    # Alpha Vantage APIキーを設定
    api_key = 'GCQBFJ7AMYQTK8VP.'
    ts = TimeSeries(key=api_key, output_format='json')

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # 5分足データをAlpha Vantage APIから取得
    data, _ = ts.get_intraday(symbol="USDJPY", interval="60min", outputsize="full")
    
    # 期間内のデータのみを抽出
    fx_data = {datetime.strptime(date, "%Y-%m-%d %H:%M:%S"): float(info["4. close"]) 
               for date, info in data.items() 
               if start_date <= datetime.strptime(date, "%Y-%m-%d %H:%M:%S") <= end_date}
    
    # データフレームに変換してソート
    fx_dataframe = pd.DataFrame.from_dict(fx_data, orient="index", columns=["Exchange Rate"])
    fx_dataframe.index.name = "Date"
    fx_dataframe.sort_index(ascending=True, inplace=True)

    return fx_dataframe

# write_fx_data_to_csv関数は、指定された期間の為替レートデータをCSVファイルに書き出す関数です。
def write_fx_data_to_csv(start_year_month, end_year_month):
    # 期間の開始日と終了日を計算
    start_date = datetime.strptime(start_year_month, "%Y-%m") + relativedelta(day=1)
    end_date = datetime.strptime(end_year_month, "%Y-%m") + relativedelta(day=31)

    fx_data = fetch_fx_data(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    print("Processed DataFrame:")
    print(fx_data.head())

    # 取得した為替レートデータをCSVファイルに書き出す
    fx_data.to_csv("usd_jpy_fx_data.csv", index_label="Date", header=["Exchange Rate"])

    # CSVファイル内容を表示
    with open("usd_jpy_fx_data.csv", "r") as f:
        print("CSV file content:")
        print(f.read())

if __name__ == "__main__":
    # ここで3年間分のデータを取得する
    start_year_month = "2021-10"  # 3年前の開始日
    end_year_month = "2024-10"    # 3年前から現在までの終了日
    write_fx_data_to_csv(start_year_month, end_year_month)
