def fetch_fx_data(start_date, end_date):
    api_key = 'GCQBFJ7AMYQTK8VP.'
    ts = TimeSeries(key=api_key, output_format='json')

    # APIからデータを取得
    data, _ = ts.get_intraday(symbol="USDJPY", interval="5min", outputsize="full")

    # レスポンスデータの確認
    if "Time Series (5min)" not in data:
        print("Error: API data retrieval failed.")
        return None

    # 日付でフィルタリング
    fx_data = {datetime.strptime(date, "%Y-%m-%d %H:%M:%S"): float(info["4. close"]) 
               for date, info in data["Time Series (5min)"].items() 
               if start_date <= datetime.strptime(date, "%Y-%m-%d %H:%M:%S") <= end_date}

    if not fx_data:
        print("Error: No data found for the specified date range.")
        return None
    
    # DataFrameに変換
    fx_dataframe = pd.DataFrame.from_dict(fx_data, orient="index", columns=["Exchange Rate"])
    fx_dataframe.index.name = "Date"
    fx_dataframe.sort_index(ascending=True, inplace=True)

    return fx_dataframe
