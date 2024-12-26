import pandas as pd
import numpy as np

class FXData:
    def __init__(self, csv_file_path):
        """
        CSVファイルを読み込み、価格データと日付を設定する。
        """
        # CSVファイルの読み込み
        df = pd.read_csv("./exchange_rate_data_filtered.csv")
        
        # 必要なデータを抽出（価格データと日付）
        self.date = pd.to_datetime(df['Date'])  # 日付
        self.price_data = df[['Open', 'High', 'Low', 'Close']].values  # 価格データ（Open, High, Low, Close）
        
        self.history = []  # 過去の履歴（取引履歴）
        self.to_play = 1  # 現在の取引ターン（通常はエージェントがプレイ中）

    def make_image(self, index):
        """
        価格データを元に状態画像（特徴量）を作成。
        ここでは、終値（Close）のデータを返すと仮定。
        """
        # 価格データを状態として使用（Close価格）
        return np.array(self.price_data[index])

    def make_target(self, index, num_unroll_steps, td_steps):
        """
        次の価格予測や報酬をターゲットとして生成。
        """
        target_price = self.price_data[index + num_unroll_steps, 3]  # 予測する価格（Close）
        target_reward = self.calculate_reward(index, num_unroll_steps)  # 報酬の計算
        
        return {
            'target_price': target_price,  # 予測する価格（Close）
            'target_reward': target_reward,  # 報酬
        }

    def calculate_reward(self, index, num_unroll_steps):
        """
        次の価格との差を報酬として計算。
        仮に、次の価格との差を報酬として設定。
        """
        reward = self.price_data[index + num_unroll_steps, 3] - self.price_data[index, 3]  # Closeの価格差
        return reward

# 使用例
# FXデータが入ったCSVファイルを読み込み
fx_data = FXData('fx_data.csv')

# インデックス0のデータを状態画像として取得
state_image = fx_data.make_image(0)

# ターゲットデータ（予測価格と報酬）を取得
target_data = fx_data.make_target(0, num_unroll_steps=5, td_steps=1)

# 結果を表示
print("State Image (Close Price):", state_image)
print("Target Data:", target_data)
