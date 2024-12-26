import os
import pandas as pd
from alpha_vantage.foreignexchange import ForeignExchange
from datetime import datetime
from dateutil.relativedelta import relativedelta

#fetch_fx_data関数は、指定された期間の為替レートデータを取得する関数です。Alpha VantageのAPIキーを設定し、ForeignExchangeオブジェクトを作成します。
#また、引数として与えられた日付文字列をdatetimeオブジェクトに変換します。
def fetch_fx_data(start_date, end_date):
    # Alpha Vantage APIキーを設定
    api_key = 'GCQBFJ7AMYQTK8VP.'
    fx = ForeignExchange(key=api_key)
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    #USD/JPYの為替レートデータをAlpha Vantage APIから取得し、指定された期間のデータだけを抽出しています。
    # Get daily exchange rate data
    fx_data, _ = fx.get_currency_exchange_daily("USD", "JPY", outputsize="full")
    fx_data = {datetime.strptime(date, "%Y-%m-%d"): float(data["4. close"]) for date, data in fx_data.items() if start_date <= datetime.strptime(date, "%Y-%m-%d") <= end_date}
    #取得した為替レートデータをpandasのDataFrameに変換し、日付でソートして返しています。
    # Convert the dictionary to a DataFrame
    fx_dataframe = pd.DataFrame.from_dict(fx_data, orient="index", columns=["Exchange Rate"])
    fx_dataframe.index.name = "Date"
    fx_dataframe.sort_index(ascending=True, inplace=True)

    return fx_dataframe

#write_fx_data_to_csv関数は、指定された期間の為替レートデータをCSVファイルに書き出す関数です。まず、引数で指定された年月を元に、期間の開始日と終了日を計算します。
#次に、fetch_fx_data関数を使って為替レートデータを取得し、データフレームの先頭部分を表示します。
def write_fx_data_to_csv(start_year_month, end_year_month):
    start_date = datetime.strptime(start_year_month, "%Y-%m") + relativedelta(day=1)
    end_date = datetime.strptime(end_year_month, "%Y-%m") + relativedelta(day=31)

    fx_data = fetch_fx_data(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    print("Processed DataFrame:")
    print(fx_data.head())
    #取得した為替レートデータを含むDataFrameをCSVファイルに書き出します。ファイル名は "usd_jpy_fx_data.csv" で、
    #インデックス（日付）のラベルは "Date"、ヘッダーは "Exchange Rate" としています。
    fx_data.to_csv("usd_jpy_fx_data.csv", index_label="Date", header=["Exchange Rate"])

    with open("usd_jpy_fx_data.csv", "r") as f:
        print("CSV file content:")
        print(f.read())


if __name__ == "__main__":
    start_year_month = "2004-10"
    end_year_month = "2024-10"
    write_fx_data_to_csv(start_year_month, end_year_month)