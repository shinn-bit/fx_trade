import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from collections import Counter
import math
class FXReplayBuffer:
    def __init__(self, config):
        """
        リプレイバッファの初期化。
        """
        self.window_size = config.window_size  # バッファの最大サイズ
        self.batch_size = config.batch_size    # サンプリング時のバッチサイズ
        self.buffer = []                       # バッファ本体

    def save_game(self, game):
        """
        ゲームの履歴全体をバッファに保存。
        """
        for step in game.history:
            if len(self.buffer) >= self.window_size:
                self.buffer.pop(0)  # 古いデータを削除
            self.buffer.append(step)

    def save_fx_data(self, fx_data, config):
        """
        FXデータ全体をリプレイバッファに保存。
        """
        cumulative_reward = 0.0  # 累積報酬の初期化

        for i in range(len(fx_data.scaled_data)):
            # 初期ステップの場合の累積報酬の設定
            if i == 0:
                cumulative_reward = 0.0
            else:
                # 累積報酬を更新（価格の差分を加算）
                cumulative_reward += (
                    fx_data.scaled_data.iloc[i]['Close'] - fx_data.scaled_data.iloc[i - 1]['Close']
                )

             # 古いデータを削除（バッファの制限サイズを超えた場合）
            if len(self.buffer) >= self.window_size:
                self.buffer.pop(0)

            # アクションを選択
            action = fx_data.select_action(i)

            # make_target を呼び出してターゲットデータを生成
            target_data = fx_data.make_target(
                index=i,
                num_unroll_steps=config.num_unroll_steps,
                td_steps=config.td_steps,
                config=config,
                cumulative_reward=cumulative_reward,
                action=fx_data.select_action(i)
            )

            # ステップデータをリプレイバッファに保存
            index = {
                'state': fx_data.make_image(i),
                'reward': target_data['target_reward'],
                'next_state': fx_data.make_image(i + 1) if i < len(fx_data.scaled_data) - 1 else None,
                'target_price': target_data['target_price'],  # 将来の価格
                'cumulative_reward': cumulative_reward,  # 累積報酬を記録
            }
            self.buffer.append(index)


    def sample_batch(self, num_unroll_steps, td_steps):
        """
        バッチサイズ分のデータをランダムにサンプリング。
        """
        if len(self.buffer) < self.batch_size:
            raise ValueError("バッファに十分なデータがありません。")

        # バッファからランダムにデータをサンプリング
        sampled_steps = np.random.choice(self.buffer, self.batch_size, replace=False)

        # サンプルデータに基づき、状態、履歴、ターゲットを返す
        return [
            (
                step['state'],  # 現在の状態
                step.get('actions', []),  # アクション履歴（デフォルトで空リスト）
                step.get('targets', []),  # ターゲット（デフォルトで空リスト）
            )
            for step in sampled_steps
        ]
def augment_data(df, columns=None, noise_level=0.001):
    # データフレームのコピーを作成
    augmented_df = df.copy()

    # 適用対象の列を特定
    if columns is None:
        # 数値列をデフォルトで選択
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # 指定された列にランダムノイズを追加
    for column in columns:
        if column in augmented_df.columns:
            augmented_df[column] += np.random.normal(0, noise_level, size=len(df))
        else:
            print(f"Warning: '{column}' 列がデータフレームに存在しません。")

    return augmented_df
# RSIの計算関数(windowは変更可能14⇒長期7⇒短期)
def calculate_rsi(prices, window=14):
    """
    RSI (Relative Strength Index) を計算する。
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
# ボリンジャーバンドの計算関数
def calculate_bollinger_bands(prices, window=20):
    """
    ボリンジャーバンドを計算する。
    """
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = sma + (2 * std)
    lower_band = sma - (2 * std)
    return sma, upper_band, lower_band
class MuZeroConfig:
    def __init__(self, initial_balance, stop_loss_percentage=5, take_profit_percentage=15):
        self.initial_balance = initial_balance
        self.stop_loss_limit = initial_balance * (1 - stop_loss_percentage / 100)
        self.take_profit_limit = initial_balance * (1 + take_profit_percentage / 100)
        self.window_size = 500
        self.batch_size = 64
        self.lr_init = 0.0005
        self.lr_decay_rate = 0.95
        self.lr_decay_steps = 100
        self.training_steps = 1000
        self.checkpoint_interval = 100
        self.num_unroll_steps = 5
        self.td_steps = 1
        self.momentum = 0.9
        self.weight_decay = 0.0005
        self.transaction_cost = 0.0005
        self.price_change_weight = 0.035
        self.risk_penalty_weight = 0.15
        self.success_bonus = 0.05
        self.buy_sell_bonus_base = 0.08
        self.max_buy_sell_bonus = 0.2
        self.hold_penalty = 0.01
        self.hold_penalty_weight = 0.05
        self.technical_adjustment_weight = 0.12
        self.reward_scale_factor = 1.0
        self.max_reward = 2.0
        self.max_episodes = 20
        self.max_steps = 100
        self.min_steps = 50
        self.min_cumulative_reward = -1.0
        self.log_interval = 50
        self.training_interval = 5
        self.validation_interval = 10
        self.min_buy_sell_bonus = -0.1
        self.buy_sell_bonus_scale = 0.1
        self.reward_target = 2.0
        self.loss_threshold = -1.5
        self.max_success_bonus = 0.1
        self.success_bonus_scale = 0.01
        self.transaction_cost_weight = 0.3
        self.success_bonus_weight = 0.6
        self.buy_sell_bonus_weight = 0.25
        self.max_risk_penalty = 0.3
        self.volatility_threshold = 0.3
        self.max_cost_ratio = 0.5
        self.min_success_streak = -3
        self.min_volatility = 0.02
        self.max_volatility = 1.5
        self.recent_reward_window = 10
        self.recent_loss_threshold = -0.5
        self.initial_steps = 10  # 初期ステップ数
        self.success_bonus_boost = 1.5  # 成功ボーナスの強化倍率
        self.buy_sell_bonus_boost = 1.5  # 売買ボーナスの強化倍率
        self.episode_priority_weight = 0.8
        self.min_balance = 1000
        self.max_volatility_limit = 0.18
        self.input_size = 10
        self.volatility_window = 20

class ActionTracker:
    def __init__(self):
        self.action_history = []  # アクションの履歴を記録

    def record_action(self, action):
        self.action_history.append(action)
class FXData:
    def __init__(self, data, client, account_id, noise_level=0.01):
        """
        データフレームを読み込み、価格データと日付を設定し、テクニカル指標を計算、
        ランダムノイズを加えたデータ拡張とスケーリングを行う。
        """
        df = data.copy()
        # OANDA APIから初期バランスを取得
        self.initial_balance = FXData.get_account_balance(client, account_id)
        self.balance = self.initial_balance  # バランスの初期化
        # 日付列をフォーマット
        df['Date'] = pd.to_datetime(df['Date'])

        # ランダムノイズの追加 (augment_data 関数を使用)
        df = augment_data(df, noise_level=noise_level)

        # 必要なデータを抽出
        self.date = df['Date']
        self.price_data = df[['Open', 'High', 'Low', 'Close']]

        # テクニカル指標を計算
        df['RSI'] = calculate_rsi(df['Close'])  # RSI
        df['BB_Middle'], df['BB_Upper'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
        df['SMA50'] = df['Close'].rolling(window=50).mean()  # 50日SMA
        df['SMA200'] = df['Close'].rolling(window=200).mean()  # 200日SMA
        df.ffill(inplace=True)  # 欠損値の前方補完
        df.bfill(inplace=True)  # 欠損値の後方補完

        # ボリンジャーバンドのシグナルを計算
        df['BB_Signal'] = self.calculate_bb_signal(df['Close'], df['BB_Upper'], df['BB_Lower'])
        # RSIに基づくシグナルを計算
        df['RSI_Signal'] = self.calculate_rsi_signal(df['RSI'])
        # ボリンジャーバンド距離を計算
        df['BB_Distance'] = df.apply(
            lambda row: min(abs(row['Close'] - row['BB_Upper']), abs(row['Close'] - row['BB_Lower'])), axis=1
        )
        # 価格変化を計算して列を追加
        df['price_change'] = df['Close'].diff()

        # スケーリング
        features_to_scale = [
            'Open', 'High', 'Low', 'Close',
            'SMA50', 'SMA200', 'RSI',
            'BB_Middle', 'BB_Upper', 'BB_Lower'
        ]
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(df[features_to_scale])

        # スケール済みデータを保持
        scaled_df = pd.DataFrame(scaled_features, columns=features_to_scale)
        scaled_df['Date'] = df['Date'].values

        self.scaled_data = scaled_df
        self.data_with_indicators = df
        self.history = []
        self.success_streak = 0
        self.reward_history = []
        # 初期バランスと取引関連の設定
        self.position = 0  # 保有ポジション (例: 1は買い, -1は売り, 0はポジションなし)
        self.cumulative_reward = 0  # 累積報酬の初期化
        self.current_price = df['Close'].iloc[0]
        self.current_volatility = 0
        self.risk_penalty = 0 
        self.current_volatility = None
        self.last_volatility_step = -1
        self.default_volatility = 0.01
    def update_current_price(self, step):
        """
        現在の価格を更新するメソッド。
        """
        if 0 <= step < len(self.price_data):
            self.current_price = self.price_data.iloc[step]['Close']
            print(f"[DEBUG] Updated Current Price: {self.current_price}")
        else:
            raise IndexError(f"Step {step} is out of bounds for price data.")
    @staticmethod
    def get_account_balance(client, account_id):
        """
        OANDA APIを使用してアカウントの残高を取得します。
        """
        r = accounts.AccountDetails(accountID=account_id)
        try:
            response = client.request(r)
            balance = response['account']['balance']
            print(f"Current Account Balance: {balance}")
            return float(balance)
        except Exception as e:
            print(f"Failed to fetch account balance: {e}")
            return 300000
    def select_action(self, index):

         """
         テクニカル指標に基づきアクションを選択。
         """
         row = self.data_with_indicators.iloc[index]

         if row['RSI_Signal'] == 1 or row['BB_Signal'] == 1:
             return 0  # Buy
         elif row['RSI_Signal'] == -1 or row['BB_Signal'] == -1:
             return 1  # Sell
         else:
             return 2  # Hold
    def execute_action(self, state, action, step, config, cumulative_reward):
        """
        アクションを実行し、報酬と終了条件を計算。
        - action: 0 (Buy), 1 (Sell), 2 (Hold)
        - step: 現在のステップインデックス
        """
        current_price = self.price_data['Close'].iloc[step]
        next_price = (
            self.price_data['Close'].iloc[step + 1]
            if step + 1 < len(self.price_data) else current_price
        )

        # 動的取引コストとボラティリティを計算
        rolling_std = self.price_data['Close'].rolling(window=10).std()
        # ボラティリティを計算またはキャッシュされた値を使用
        if self.current_volatility is None or self.last_volatility_step != step:
            self.current_volatility = self.calculate_current_volatility(step)
            self.last_volatility_step = step

        current_volatility = self.current_volatility  # キャッシュされた値を使用
        if current_volatility is None:
            current_volatility = 0  # デフォルト値
        transaction_cost= calculate_transaction_cost(config, step=step, fx_data=fx_data)

        # ボラティリティ正規化とスケール
        min_volatility = config.min_volatility
        max_volatility = config.max_volatility
        normalized_volatility = np.clip(
            (current_volatility - min_volatility) / (max_volatility - min_volatility + 1e-6),
            0,
            1
        )
        volatility_range = max_volatility - min_volatility
        dynamic_scale = 1 + (normalized_volatility * volatility_range)

        # デフォルトのペナルティ（Hold の場合）
        hold_penalty = -config.hold_penalty * dynamic_scale * normalized_volatility

        # 報酬の初期化
        reward = 0
        current_price = self.price_data['Close'].iloc[step]
        # アクションの実行
        if action == 0:  # Buy
            if self.position <= 0:  # ショートまたは未保有の場合
                self.balance -= current_price
                self.position = 1
            reward = (next_price - current_price) / current_price - transaction_cost

        elif action == 1:  # Sell
            if self.position >= 0:  # ロングまたは未保有の場合
                self.balance += current_price
                self.position = -1
            reward = (current_price - next_price) / current_price - transaction_cost

        elif action == 2:  # Hold
            reward = hold_penalty

        # 報酬の型変換とクリッピング
        reward = np.clip(reward, -1.0, 1.0)

        # 累積報酬を更新
        self.cumulative_reward += reward
        self.cumulative_reward = np.clip(self.cumulative_reward, -2.0, 2.0)

        # 成功ストリークの更新
        if reward > 0:
            self.success_streak += 1
        else:
            self.success_streak = 0
        # 次の状態を作成
        next_state = {
            'Close': next_price,
            'RSI': self.data_with_indicators['RSI'].iloc[step],
            'BB_Middle': self.data_with_indicators['BB_Middle'].iloc[step],
            'BB_Upper': self.data_with_indicators['BB_Upper'].iloc[step],
            'BB_Lower': self.data_with_indicators['BB_Lower'].iloc[step],
        }
        # 終了条件を判定
        done = check_done_condition(
            index=step,
            price_data=self.price_data,
            balance=self.balance,
            position=self.position,
            cumulative_reward=self.cumulative_reward,
            success_streak=self.success_streak,
            calculate_volatility=self.current_volatility,
            config=config,
            reward_components={
                "recent_rewards": self.reward_history[-config.recent_reward_window:],
                "risk_penalty": self.risk_penalty,
            }
        )
        # デバッグ情報を出力
        print(f"[DEBUG] Step: {step}")
        print(f"  Action: {'Buy' if action == 0 else 'Sell' if action == 1 else 'Hold'}")
        print(f"  Current Price: {current_price}, Next Price: {next_price}")
        print(f"  Balance: {self.balance}, Position: {self.position}")
        print(f"  Reward: {reward}, Cumulative Reward: {cumulative_reward}")
        print(f"  Volatility: {current_volatility}, Done: {done}")

        return next_state, reward, done, self.cumulative_reward, current_volatility
    def get_current_volatility(self, step):
        if self.current_volatility is None or self.last_volatility_step != step:
            self.current_volatility = self.calculate_current_volatility(step)
            self.last_volatility_step = step
        return self.current_volatility
    def calculate_bb_signal(self, close_prices, upper_band, lower_band, margin=0.2):
      """
      ボリンジャーバンドに基づく売買シグナルを計算。

      """
      # 上位バンドと下位バンドのしきい値を調整
      adjusted_upper_band = upper_band * (1 - margin)  # 少し下げる
      adjusted_lower_band = lower_band * (1 + margin)  # 少し上げる

      # シグナルの初期化
      signal = pd.Series(0, index=close_prices.index)

      # シグナルを計算
      signal[close_prices > adjusted_upper_band] = -1  # 売りシグナル
      signal[close_prices < adjusted_lower_band] = 1   # 買いシグナル

      return signal

    def calculate_current_volatility(self, index: int, window: int = 20) -> float:
        """
        現在のボラティリティを計算する関数。ステップごとに更新。

        Parameters:
        - index (int): 現在のステップ位置。
        - window (int): ボラティリティ計算の移動ウィンドウサイズ (デフォルトは10)。

        Returns:
        - float: 計算されたボラティリティ値。
        """
        # インデックス範囲の確認
        if index < 0 or index >= len(self.price_data):
            raise ValueError(f"[ERROR] step is out of range. Must be 0 <= step < {len(self.price_data)}. Got index={index}")

        # ウィンドウ期間のデータ不足の場合
        if index < window:
            print(f"[DEBUG] Not enough data for volatility calculation at step {index}. Returning volatility=0.")
            return 0.0  # データ不足時は0を返す

        try:

            # リターンの計算: パーセンテージ変化
            returns = self.price_data['Close'].pct_change().dropna()

            # 移動標準偏差を計算
            rolling_std = returns.rolling(window=window).std()

            # 計算された標準偏差からボラティリティを取得
            if index < len(rolling_std):
                current_volatility = rolling_std.iloc[index]
            else:
                # 範囲外の場合は平均を使用
                current_volatility = rolling_std.mean()

            # ボラティリティの最小値を設定してゼロを回避
            current_volatility = max(current_volatility, 1e-6)
            return current_volatility

        except KeyError as e:
             raise KeyError(f"[ERROR] Missing 'Close' column in price_data: {e}")
        except Exception as e:
             raise RuntimeError(f"[ERROR] Failed to calculate current_volatility at step {index}: {e}")

    def calculate_rsi_signal(self, rsi_values, upper_threshold=52, lower_threshold=49):
        """
        RSIに基づく売買シグナルを計算。
        """
        signal = pd.Series(0, index=rsi_values.index)
        signal[rsi_values > upper_threshold] = -1  # 売りシグナル
        signal[rsi_values < lower_threshold] = 1   # 買いシグナル
        return signal

    def make_image(self, index, columns=None):
        """
        特徴量データを生成する。
        """
        if columns is None:
            columns = ['Open', 'High', 'Low', 'Close', 'SMA50', 'SMA200', 'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower']
        row = self.scaled_data.iloc[index]
        image = np.array([row[col] for col in columns])
        return image

    def make_target(self, index, num_unroll_steps, td_steps, config, cumulative_reward, action):
        """
        目標データを生成する。
        """
         # current_volatility を更新
        current_volatility = fx_data.get_current_volatility(index)
        if index + num_unroll_steps < len(self.price_data):
            target_price = self.price_data.iloc[index + num_unroll_steps]['Close']
            target_reward = self.calculate_reward(
                index=index,
                num_unroll_steps=num_unroll_steps,
                config=config,
                cumulative_reward=cumulative_reward,
                action=action,
                current_volatility=current_volatility,
                fx_data=self
            )
        else:
            # 補間処理
            target_price = self.interpolate_price(index, num_unroll_steps)
            target_reward = self.interpolate_reward(index, num_unroll_steps, cumulative_reward)

        technical_action = self.select_action(index)

        return {
            'target_price': target_price,
            'target_reward': target_reward,
            'technical_action': technical_action
        }
    def calculate_reward(self, index, num_unroll_steps, config, action, current_volatility=None, cumulative_reward=None, fx_data=None):
        """
        改良版報酬関数:
        - ステップごとの報酬を計算。
        - エピソード終了時の純利益を目標として優先。
        """
        # データソースを決定
        data_source = fx_data if fx_data is not None else self
        # 現在の価格変化を計算
        if index + num_unroll_steps < len(self.price_data):
            price_change = (
                self.price_data.iloc[index + num_unroll_steps]['Close'] -
                self.price_data.iloc[index]['Close']
            )
        else:
            price_change = 0

        #  ボラティリティの計算
        volatility_window = config.volatility_window
        if index >= volatility_window:
            recent_prices = self.price_data['Close'].iloc[index - volatility_window:index]
            volatility = recent_prices.std()
        else:
            if current_volatility is None:
                volatility = self.calculate_current_volatility(index)
            else:
                volatility = current_volatility
            volatility = max(volatility, 1e-6) 

        #  正規化された価格変化
        price_change_scaled = np.tanh(price_change / (volatility + 1e-6))
        price_change_scaled *= config.price_change_weight

        # 成功ボーナス
        success_bonus = 0
        if price_change_scaled > 0:
            self.success_streak += 1
            success_bonus = min(config.max_success_bonus, self.success_streak * config.success_bonus_scale)
        else:
            self.success_streak = 0

        # リスクペナルティ
        self.risk_penalty = 0
        if price_change_scaled < 0:
            self.risk_penalty = config.risk_penalty_weight * abs(price_change_scaled)

        # トレードボーナス
        buy_sell_bonus = config.buy_sell_bonus_base * abs(price_change_scaled)
        buy_sell_bonus = np.clip(buy_sell_bonus, config.min_buy_sell_bonus, config.max_buy_sell_bonus)

        # 取引コスト
        transaction_cost = config.transaction_cost
        # ホールドペナルティ
        hold_penalty = 0
        if action == 2:  # Hold
            normalized_volatility = current_volatility / max(volatility, 1e-6)
            hold_penalty = -config.hold_penalty_weight * normalized_volatility

            # テクニカル調整
        technical_adjustment = 0
        rsi_signal = self.data_with_indicators.iloc[index]['RSI_Signal']
        bb_signal = self.data_with_indicators.iloc[index]['BB_Signal']
        adjustment_factor = abs(price_change_scaled) / max(1, current_volatility)
        if rsi_signal == -1 and bb_signal == -1:
            technical_adjustment = -config.technical_adjustment_weight * adjustment_factor
        elif rsi_signal == 1 and bb_signal == 1:
            technical_adjustment = config.technical_adjustment_weight * adjustment_factor
        elif rsi_signal != 0 or bb_signal != 0:
            technical_adjustment = (config.technical_adjustment_weight / 2) * adjustment_factor
        # ステップごとの報酬
        step_reward = (
            price_change_scaled * config.price_change_weight +
            success_bonus * config.success_bonus_weight -
            self.risk_penalty -
            transaction_cost +
            buy_sell_bonus * config.buy_sell_bonus_weight-
            hold_penalty * config.hold_penalty_weight+
            technical_adjustment * config.technical_adjustment_weight
        )
        combined_reward = 0
        # エピソード終了時の利益を計算
        episode_profit = self.calculate_episode_profit()
        if cumulative_reward is not None:
            step_reward += cumulative_reward
        # エピソード終了時の利益を重視するスムージング係数
        smoothing_factor = config.episode_priority_weight
        combined_reward = (
            (1 - smoothing_factor) * step_reward +
            smoothing_factor * (episode_profit / len(self.price_data))
        )
        # 報酬コンポーネントを定義
        reward_components = {
            "price_change": price_change_scaled,
            "transaction_cost": -transaction_cost,
            "success_bonus": success_bonus,
            "risk_penalty": -self.risk_penalty,
            "buy_sell_bonus": buy_sell_bonus,
            "hold_penalty": hold_penalty,
            "technical_adjustment": technical_adjustment,
            "step_reward": step_reward,
            "episode_profit": episode_profit,
            "total": combined_reward
        }
        # 辞書形式で返す
        return reward_components
    def calculate_episode_profit(self):
        """
        エピソード終了時の純利益を計算するメソッド。
        """
        # エピソード終了時の価格（最終の価格データ）
        final_price = self.price_data.iloc[-1]['Close']

        # 資産価値の計算 (現在のバランス + 保有ポジション * 終了時の価格)
        final_asset_value = self.balance + (self.position * final_price)

        # 純利益 (終了時の資産価値 - 初期バランス)
        episode_profit = final_asset_value - self.initial_balance
        return episode_profit

    def interpolate_price(self, index, num_unroll_steps):
        """
        データ範囲外の場合に価格を線形補間で推測。
        """
        if index >= len(self.price_data) - 1:
            return self.price_data.iloc[-1]['Close']

        known_price = self.price_data.iloc[index]['Close']
        next_price_trend = (self.price_data.iloc[-1]['Close'] - known_price) / (len(self.price_data) - index)
        return known_price + next_price_trend * num_unroll_steps

    def interpolate_reward(self, index, num_unroll_steps, cumulative_reward):
        """
        データ範囲外の場合に報酬を線形補間で推測。
        """
        if index >= len(self.price_data) - 1:
            return cumulative_reward / num_unroll_steps

        price_diff = self.interpolate_price(index, num_unroll_steps) - self.price_data.iloc[index]['Close']
        return price_diff - config.transaction_cost

    def plot_price_with_indicators(self):
        """
        価格データと全ての指標をプロットする。
        """
        self.plot_bollinger_bands()
        self.plot_moving_averages()
        self.plot_rsi()
    def plot_bollinger_bands(self):
        """
        ボリンジャーバンドと価格データをプロットする。
        """
        df = self.data_with_indicators
        plt.figure(figsize=(12, 6))
        plt.plot(df['Date'], df['Close'], label='Close', color='blue')
        plt.plot(df['Date'], df['BB_Middle'], label='BB Middle', color='orange')
        plt.plot(df['Date'], df['BB_Upper'], label='BB Upper', color='green')
        plt.plot(df['Date'], df['BB_Lower'], label='BB Lower', color='red')
        plt.fill_between(df['Date'], df['BB_Upper'], df['BB_Lower'], color='gray', alpha=0.1)
        plt.title('Bollinger Bands')
        plt.legend()
        plt.show()

    def plot_rsi(self):
        """
        RSIをプロットする。
        """
        df = self.data_with_indicators
        plt.figure(figsize=(12, 6))
        plt.plot(df['Date'], df['RSI'], label='RSI', color='purple')
        plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
        plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
        plt.title('Relative Strength Index (RSI)')
        plt.legend()
        plt.show()

    def plot_moving_averages(self):
        """
        移動平均（SMA50, SMA200）と価格データをプロットする。
        """
        df = self.data_with_indicators
        plt.figure(figsize=(12, 6))
        plt.plot(df['Date'], df['Close'], label='Close', color='blue')
        plt.plot(df['Date'], df['SMA50'], label='SMA50', color='orange')
        plt.plot(df['Date'], df['SMA200'], label='SMA200', color='green')
        plt.title('Moving Averages (SMA50 & SMA200)')
        plt.legend()
        plt.show()

    def plot_price_with_indicators(self):
        """
        価格データと全ての指標をプロットする。
        """
        self.plot_bollinger_bands()
        self.plot_moving_averages()
        self.plot_rsi()
def plot_predictions(y_true, y_pred, title="Model Predictions"):
    """
    予測結果と実際の値をプロット。
    """
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label="Actual", color='blue')
    plt.plot(y_pred, label="Predicted", color='red', linestyle='--')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()
def evaluate_model(y_true, y_pred):
    """
    モデルの性能を評価。
    """
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred)**2)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    #mae,mseは小さいほど良い
    return mae, mse
class Game:
    def __init__(self):
        """
        ゲームの状態や取引履歴を管理するクラス。
        """
        self.history = []  # 各ステップの履歴（状態、アクション、報酬など）

    def add_step(self, state, action, reward, next_state):
        """
        ゲームの各ステップ（状態、アクション、報酬、次の状態）を追加。
        """
        self.history.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state
        })
class NetworkOutput:
    def __init__(self, value, reward, policy_logits, hidden_state):
        """
        モデルの推論結果を格納するクラス。
        """
        self.value = value  # 推定される状態の価値
        self.reward = reward  # 得られる報酬
        self.policy_logits = policy_logits  # 各アクションに対するポリシー分布
        self.hidden_state = hidden_state  #https://openai.com/ja-JP/chatgpt/overview/ 隠れ状態

    def __str__(self):
        return (f"Value: {self.value}, Reward: {self.reward}, "
                f"Policy: {self.policy_logits}, Hidden State Shape: {self.hidden_state.shape}")
class Network:
    def __init__(self, input_shape=(10,), action_space_size=3):
        """
        MuZeroネットワーク構造（Value HeadとReward Headを追加）。
        """
        # Representationモデル: 状態表現を生成
        self.representation = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),  # 入力形状 (10,)
            tf.keras.layers.Dense(256, activation='relu'),  # 隠し層1
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation='relu'),  # 隠し層2
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation='relu'),  # 隠し層3
            tf.keras.layers.BatchNormalization(),
        ])

        # Dynamicsモデル: 次の状態と報酬を予測
        self.dynamics = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(64 + action_space_size,)),  # 状態 + アクション (64 + action_space_size)
            tf.keras.layers.Dense(256, activation='relu'),  # 隠し層1
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation='relu'),  # 隠し層2
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(65)  # 出力層: 次の状態 (64) と報酬 (1)
        ])

        # Predictionモデル: 価値とポリシーを予測
        self.prediction_policy = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(64,)),  # 状態の次元数
            tf.keras.layers.Dense(64, activation='relu'),  # 隠し層1
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32, activation='relu'),  # 隠し層2
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(action_space_size, activation='softmax')  # 出力層: ポリシー分布
        ])

        # Value Head: 状態から価値を予測
        self.value_head = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(64,)),  # 状態の次元数
            tf.keras.layers.Dense(64, activation='relu'),  # 隠し層1
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32, activation='relu'),  # 隠し層2
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1, activation='linear')  # 出力層: スカラー値
        ])

        # Reward Head: 状態から報酬を予測
        self.reward_head = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(64,)),  # 状態の次元数
            tf.keras.layers.Dense(64, activation='relu'),  # 隠し層1
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32, activation='relu'),  # 隠し層2
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1, activation='linear')  # 出力層: スカラー値
        ])
    def get_weights(self):
        """
        ネットワーク内の全ての学習可能な重みを取得。
        """
        weights = []
        for model in [
            self.representation,
            self.dynamics,
            self.prediction_policy,
            self.value_head,
            self.reward_head
        ]:
            weights.extend(model.trainable_variables)
        return weights
    def get_trainable_variables(self):
        """
        ネットワーク内のすべての学習可能な変数を取得。
        """
        trainable_vars = []
        for model in [
            self.representation,
            self.dynamics,
            self.prediction_policy,
            self.value_head,
            self.reward_head
        ]:
            trainable_vars.extend(model.trainable_variables)
        return trainable_vars

    def initial_inference(self, observation):
        """
        初期の状態をエンコードし、価値、報酬、ポリシーを出力。
        """
        # 観測データの形状を確認し、バッチ次元を追加
        if len(observation.shape) == 1:
            observation = tf.expand_dims(observation, axis=0)  # shape: (1, input_shape)

        # Representationモデルで状態を生成
        state = self.representation(observation)  # shape: (1, 64)

        # Value Headで価値を予測
        value = self.value_head(state).numpy().flatten()[0]  # スカラー値に変換

        # Reward Headで報酬を予測
        reward = self.reward_head(state).numpy().flatten()[0]  # スカラー値に変換

        # Predictionモデルでポリシーロジットを計算
        policy_logits = self.prediction_policy(state).numpy()  # NumPy形式に変換

        # NetworkOutput形式で返却
        return NetworkOutput(value, reward, policy_logits, state.numpy())
    def recurrent_inference(self, state, action):
      """
      再帰的な推論: 次の状態、価値、報酬、ポリシーを計算。
      """
      # アクションをone-hotエンコーディングし、バッチ次元を追加
      action_one_hot = tf.one_hot(action, depth=3)  # shape: (3,)
      action_one_hot = tf.expand_dims(action_one_hot, axis=0)  # shape: (1, 3)

      # 状態とアクションを結合
      state_action = tf.concat([state, action_one_hot], axis=-1)  # shape: (1, 64 + 3)

      # Dynamicsモデルで次の状態と報酬を計算
      dynamics_output = self.dynamics(state_action)
      next_state = dynamics_output[:, :-1]  # 次の状態 (最後のユニットを除く)
      reward = dynamics_output[:, -1]  # 報酬 (最後のユニット)

      # Value Headで次の状態の価値を計算
      value = self.value_head(next_state).numpy().flatten()[0]

      # Predictionモデルでポリシーロジットを計算
      policy_logits = self.prediction_policy(next_state).numpy()

      # NetworkOutput形式で返却
      return NetworkOutput(value, reward.numpy().flatten()[0], policy_logits, next_state.numpy())
class SharedStorage:
    def __init__(self):
        self._networks = {}  # ステップごとにネットワークを保存
        self._latest_step = -1  # 最新のステップを追跡

    def latest_network(self) -> Network:
        """
        最新のネットワークを返します。
        保存されているネットワークがなければ、ユニフォームネットワークを返します。
        """
        if self._networks:
            return self._networks[self._latest_step]
        else:
            # 保存されたネットワークがなければ、デフォルトのネットワークを返す
            return self.make_uniform_network()

    def save_network(self, step: int, network: Network):
        """
        指定されたステップでネットワークを保存します。
        """
        self._networks[step] = network
        self._latest_step = step  # 最新のステップを更新

    def make_uniform_network(self) -> Network:
        """
        デフォルトのユニフォームネットワークを作成し、返します。
        """
        action_space_size = 3  # アクションスペースサイズ（例: 買い、売り、何もしない）
        policy = uniform_policy(action_space_size)  # 均等なポリシー
        return Network(input_shape=(10,), action_space_size=action_space_size)
def uniform_policy(action_space_size):
    """
    アクションスペース全体に均等な確率を割り当てるポリシー。
    """
    return [1.0 / action_space_size] * action_space_size
def run_selfplay_with_action_tracking(config, storage, replay_buffer, fx_data):
    """
    Self-playを実行し、累積報酬、利益、およびアクション分布を追跡。
    """
    cumulative_rewards = []
    episode_returns = []

    for episode in range(config.max_episodes):
        network = storage.latest_network()
        results = play_fx_game(config, network, fx_data)

        cumulative_rewards.append(results["cumulative_reward"])
        episode_returns.append(results["total_return"])

        print(f"Episode {episode + 1}/{config.max_episodes} - "
              f"Cumulative Reward: {results['cumulative_reward']:.2f}, "
              f"Return: {results['total_return']:.2f}")

    return cumulative_rewards, episode_returns
def visualize_action_distribution(actions):
    """
    全エピソードを通じたアクション分布を可視化。
    """
    action_counts = Counter(actions)  # アクションの頻度をカウント
    action_labels = {0: "Buy", 1: "Sell", 2: "Hold"}  # アクションラベルの定義

    # ラベルの名前を使ったカウントの準備
    labeled_counts = {action_labels[action]: count for action, count in action_counts.items()}

    plt.bar(labeled_counts.keys(), labeled_counts.values(), color=["green", "red", "blue"])
    plt.title("Action Distribution (All Episodes)")
    plt.xlabel("Actions")
    plt.ylabel("Counts")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()



def visualize_cumulative_rewards(rewards):
    """
    累積報酬をプロット。
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Cumulative Reward", color="green", marker="o")
    plt.axhline(0, color="red", linestyle="--", label="Zero Reward")
    plt.title("Cumulative Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid(True)
    plt.show()
def initialize_fx_environment(fx_data: FXData):
    """
    FX取引シミュレーションの初期環境を設定。
    fx_data: FXDataクラスのインスタンス。
    """
    # データが空でないことを確認
    if len(fx_data.scaled_data) == 0:
        raise ValueError("FXデータが空です。環境を初期化できません。")

    # 初期状態をスケール済みデータの最初の行から取得
    initial_state = fx_data.make_image(0)  # インデックス0のデータ
    return initial_state
def network_policy(network: Network, state: np.ndarray, step: int, config: MuZeroConfig) -> int:
    """
    Epsilon-Greedy を使用してネットワークから次のアクションを選択する。
    """
    # 1. 状態をテンソル形式に変換
    if isinstance(state, dict):
        # 辞書の場合、値をリストに変換
        state_values = list(state.values())
        expected_state_size = config.input_size  # 例: ネットワークが期待する入力サイズ
        if len(state_values) < expected_state_size:
            # サイズ不足の場合ゼロ埋め
            state_values += [0] * (expected_state_size - len(state_values))
        elif len(state_values) > expected_state_size:
            # サイズ超過の場合トリミング
            state_values = state_values[:expected_state_size]
        state_tensor = tf.convert_to_tensor([state_values], dtype=tf.float32)
    elif len(state.shape) == 1:
        # 配列の場合
        state_tensor = tf.expand_dims(state, axis=0)  # [1, state_size]
    else:
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)

    # 2. 状態を表現モデルに通す
    state_representation = network.representation(state_tensor)  # [1, hidden_size]

    # 3. ポリシーロジットを取得
    policy_logits = network.prediction_policy(state_representation).numpy()  # [1, action_space_size]
    policy_probs = tf.nn.softmax(policy_logits[0]).numpy()  # [action_space_size]

    # 4. Epsilon-Greedy の epsilon 値を計算
    if hasattr(config, "epsilon_schedule"):
        epsilon = config.epsilon_schedule(step)  # 設定ファイルに関数として定義されている場合
    else:
        epsilon = max(0.05, 1.0 - step / config.training_steps)  # デフォルトの減少スケジュール

    # 5. 探索 vs 活用の選択
    if np.random.rand() < epsilon:
        action = np.random.choice(len(policy_probs))  # ランダムアクション（探索）
    else:
        action = np.argmax(policy_probs)  # 最良のアクション（活用）


    return action

def play_fx_game(config: MuZeroConfig, network: Network, fx_data: FXData, reward_history: list) -> dict:
    """
    MuZero を用いてFX取引のシミュレーションを行い、累積報酬と利益を記録。
    """
    # 初期化
    local_fx_data = fx_data
    state = initialize_fx_environment(local_fx_data)  # 初期状態
    game = Game()  # ゲームの記録
    action_tracker = ActionTracker()  # アクションの記録用
    done = False
    step = 0
    cumulative_reward = 0.0  # 累積報酬
    total_return = 0.0  # 利益の記録
    trade_history = []  # トレード履歴

    reward_components = {
        "price_change": 0.0,
        "transaction_cost": 0.0,
        "success_bonus": 0.0,
        "risk_penalty": 0.0,
        "buy_sell_bonus": 0.0,
        "hold_penalty": 0.0,
        "technical_adjustment": 0.0,
        "total": 0.0,
    }
    # ボラティリティの計算
    rolling_std = fx_data.price_data['Close'].rolling(window=20).std()

    while not done and step < config.max_steps:
        # アクション選択
        current_volatility = fx_data.get_current_volatility(step)

        action = network_policy(network, state, step, config)

        # アクションの実行と結果取得
        next_state, reward, done, cumulative_reward, current_volatility = fx_data.execute_action(
            state, action, step, config, cumulative_reward
        )

        # 取引コストとボラティリティを計算
        transaction_cost= calculate_transaction_cost(config, step=step, fx_data=fx_data)
        # 報酬から取引コストを差し引く
        adjusted_reward = reward - transaction_cost

        # トレード履歴の更新
        trade_history.append(adjusted_reward)
        total_return += adjusted_reward

        # 報酬の計算と記録
        reward_components = fx_data.calculate_reward(
            index=step, num_unroll_steps=1, config=config,
            action=action, current_volatility=current_volatility
        )

        reward_history.append(reward_components)

        # アクション履歴の記録
        action_tracker.record_action(action)

        # ゲーム履歴の更新
        game.add_step(state, action, adjusted_reward, next_state)

        # 状態の更新
        state = next_state
        step += 1

        if step < config.min_steps:  # 最低ステップ数未満の場合
            done = False
        else:
            done = check_done_condition(
                index=step,
                price_data=fx_data.price_data,
                balance=fx_data.balance,
                position=fx_data.position,
                cumulative_reward=fx_data.cumulative_reward,
                success_streak=fx_data.success_streak,
                calculate_volatility=current_volatility,
                config=config,
                reward_components={
                    "recent_rewards": fx_data.reward_history[-config.recent_reward_window:],
                    "risk_penalty": fx_data.risk_penalty,
                }
            )

    # 結果を返却
    return {
        "game": game,
        "cumulative_reward": cumulative_reward,
        "total_return": total_return,
        "trade_history": trade_history,
        "reward_history": reward_history,
    }

def calculate_transaction_cost(config, step, fx_data):
    """
    動的な取引コストを計算する関数。
    """
    # current_volatility を FXData クラスから取得
    current_volatility = fx_data.get_current_volatility(step)

    mean_volatility = fx_data.price_data['Close'].pct_change().rolling(window=20).std().mean()
    # 取引コストの基本計算
    transaction_cost = config.transaction_cost * (1 + current_volatility)

    # 上限と下限を設定
    min_cost = config.transaction_cost * 0.5  # コストを基準値の50%まで引き下げる
    max_cost = config.transaction_cost * (1 + 2 * mean_volatility)  # ボラティリティによる上限
    upper_limit = config.transaction_cost * (1 + mean_volatility)  #動的な最大値
    transaction_cost = min(transaction_cost, max_cost, upper_limit)


    # 成功トレードによる割引（任意）
    if fx_data and fx_data.success_streak > 0:
        discount = fx_data.success_streak * 0.00005  # 成功回数に応じて割引
        transaction_cost = max(config.transaction_cost * 0.5, transaction_cost - discount)

    return transaction_cost

def check_done_condition(
    index: int, 
    price_data: pd.DataFrame, 
    balance: float, 
    position: int, 
    cumulative_reward: float, 
    success_streak: int, 
    calculate_volatility, 
    config, 
    reward_components: dict
) -> bool:
    """
    エピソード終了条件を判断。
    """
    # ボラティリティを計算
    volatility = fx_data.get_current_volatility(index)
    print(f"[DEBUG] Volatility at index {index}: {volatility}")

    # インデックスの範囲を確認
    if index < 0 or index >= len(price_data):
        raise IndexError(f"[ERROR] Index {index} is out of bounds for price_data with length {len(price_data)}")
    if price_data['Close'].isnull().any():
        raise ValueError("[ERROR] price_data contains NaN values.")

    # 現在の資産価値を計算
    current_price = price_data['Close'].iloc[index]
    final_asset_value = balance + (fx_data.position * current_price)

    # 各終了条件を評価
    if index < config.min_steps:
        return False
    if final_asset_value < config.stop_loss_limit:
        print(f"[DEBUG] Stop-loss Condition Triggered: Final Asset Value = {final_asset_value}, Limit = {config.stop_loss_limit}")
        return True
    if final_asset_value > config.take_profit_limit:
        print(f"[DEBUG] Take-Profit Condition Triggered: Final Asset Value = {final_asset_value}, Limit = {config.take_profit_limit}")
        return True
    if volatility > config.max_volatility_limit:
        print(f"[DEBUG] Done due to high volatility: {volatility} > {config.max_volatility_limit}")
        return True
    if cumulative_reward < config.loss_threshold and success_streak < config.min_success_streak:
        print(f"[DEBUG] Done due to cumulative loss threshold: {cumulative_reward} < {config.loss_threshold}")
        return True
    recent_rewards = reward_components.get("recent_rewards", [])
    if len(recent_rewards) >= config.recent_reward_window and sum(recent_rewards) < config.recent_loss_threshold:
        print(f"[DEBUG] Done due to recent loss trend: Sum of recent rewards = {sum(recent_rewards)}")
        return True
    current_risk_penalty = reward_components.get("risk_penalty", 0)
    if current_risk_penalty < -config.max_risk_penalty:
        print(f"[DEBUG] Done due to excessive risk penalty: {current_risk_penalty}")
        return True
    if index + 1 >= config.max_steps:
        print(f"[DEBUG] Done due to max steps: {index + 1}/{config.max_steps}")
        return True
    if cumulative_reward >= config.reward_target:
        print(f"[DEBUG] Done due to reward target: {cumulative_reward} >= {config.reward_target}")
        return True

    return False

def visualize_reward_components(reward_history):
    """
    報酬内訳を可視化する関数。
    """
    steps = reward_history.index  # インデックスをステップとする
    plt.figure(figsize=(12, 8))

    # 各報酬要素をプロット
    plt.plot(steps, reward_history["price_change"], label="price_change", color="blue")
    plt.plot(steps, reward_history["transaction_cost"], label="transaction_cost", color="orange")
    plt.plot(steps, reward_history["success_bonus"], label="success_bonus", color="green")
    plt.plot(steps, reward_history["risk_penalty"], label="risk_penalty", color="red")
    plt.plot(steps, reward_history["buy_sell_bonus"], label="buy_sell_bonus", color="purple")
    plt.plot(steps, reward_history["hold_penalty"], label="hold_penalty", color="brown")
    plt.plot(steps, reward_history["technical_adjustment"], label="technical_adjustment", color="pink")

    # 合計のプロット
    plt.plot(steps, reward_history["total"], label="Total Reward", color="black", linestyle="--")

    plt.title("Reward Component Contributions (Self-Play)")
    plt.xlabel("Steps")
    plt.ylabel("Reward Contribution")
    plt.legend()
    plt.grid()
    plt.show()
def visualize_cumulative_rewards_and_returns(cumulative_rewards, returns):
    """
    累積報酬と各エピソードの利益をプロット。
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 累積報酬
    ax1.plot(cumulative_rewards, label="Cumulative Rewards", color="blue", marker="o")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulative Rewards", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    # エピソードごとの利益（リターン）
    ax2 = ax1.twinx()
    ax2.plot(returns, label="Returns", color="green", linestyle="--", marker="x")
    ax2.set_ylabel("Returns (Profit)", color="green")
    ax2.tick_params(axis='y', labelcolor="green")

    # タイトル
    fig.suptitle("Cumulative Rewards and Episode Returns")

    # 凡例
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # レイアウト調整
    fig.tight_layout()
    plt.show()
def visualize_model_performance(y_true, y_pred):
    """
    モデルの性能を視覚的に評価するためのプロットを作成。
    - 実際の値と予測値の比較プロット
    - 残差（予測誤差）の分布
    """
    # 1. 実際の値と予測値の比較
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="Actual", color='blue')
    plt.plot(y_pred, label="Predicted", color='red', linestyle='--')
    plt.title("Model Predictions vs Actual")
    plt.xlabel("Samples")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2. 残差（予測誤差）の分布
    residuals = y_true - y_pred
    plt.figure(figsize=(12, 6))
    plt.hist(residuals, bins=30, color='purple', alpha=0.7)
    plt.title("Distribution of Residuals (Prediction Errors)")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
def train_network_with_validation(config: MuZeroConfig, storage: SharedStorage, replay_buffer: FXReplayBuffer, validation_replay_buffer: FXReplayBuffer):
    """
    トレーニング損失と検証損失を記録しながらネットワークを訓練します。
    """
    training_loss = []  # トレーニング損失の記録
    validation_loss = []  # 検証損失の記録
    network = storage.latest_network()

    # 学習率スケジュールを設定
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=config.lr_init,
        decay_steps=config.training_steps,
        alpha=0.01
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    for step in range(config.training_steps):
        # トレーニングデータと検証データのバッチを取得
        try:
            train_batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
            validation_batch = validation_replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
        except ValueError as e:
            print(f"Batch sampling error at step {step}: {e}")
            break

        # トレーニング損失を計算して最適化
        with tf.GradientTape() as tape:
            train_loss = compute_loss(network, train_batch, config.weight_decay, step, config)

        gradients = tape.gradient(train_loss, network.get_trainable_variables())
        optimizer.apply_gradients(zip(gradients, network.get_trainable_variables()))

        # 検証損失を計算
        val_loss = compute_loss(network, validation_batch, config.weight_decay, step, config)

        # 損失を記録
        training_loss.append(train_loss.numpy())
        validation_loss.append(val_loss.numpy())

        # ログ出力
        if (step + 1) % config.log_interval == 0 or step == 0:
            print(
                f"Step {step + 1}/{config.training_steps} | "
                f"Train Loss: {train_loss.numpy():.4f} | "
                f"Val Loss: {val_loss.numpy():.4f} | "
                f"Learning Rate: {optimizer.learning_rate.numpy():.8f}"
            )

    # 損失曲線のプロット
    plot_loss_curve(training_loss, validation_loss)
    return training_loss, validation_loss
def compute_loss(network, batch, weight_decay, step, config):
    """
    バッチデータを使用してネットワークの損失を計算する。
    """
    total_loss = 0.0

    # バッチ内のデータをループ
    for image, actions, targets in batch:
        # 初期推論
        output = network.initial_inference(image)
        predictions = [(1.0, output.value, output.reward, output.policy_logits)]

        # 再帰的な推論
        for action in actions:
            output = network.recurrent_inference(output.hidden_state, action)
            predictions.append((1.0 / len(actions), output.value, output.reward, output.policy_logits))

        # 各ターゲットに基づき損失を計算
        for prediction, target in zip(predictions, targets):
            gradient_scale, value, reward, policy_logits = prediction
            target_value, target_reward, target_policy = target

            # 損失計算
            value_loss = scalar_loss(value, target_value)
            reward_loss = scalar_loss(reward, target_reward)
            policy_loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=policy_logits, labels=target_policy
            )

            # 合計損失
            total_loss += gradient_scale * (value_loss + reward_loss + policy_loss)

    # 重み減衰 (L2正則化)
    for weights in network.get_weights():
        total_loss += weight_decay * tf.nn.l2_loss(weights)

    return total_loss
def scalar_loss(prediction, target):
    """
    スカラー値の損失を計算する（平均二乗誤差）。
    """
    return tf.reduce_mean(tf.square(prediction - target))
def plot_loss_curve(training_loss, validation_loss=None):
    """
    トレーニング損失と検証損失をプロット。
    """
    plt.figure(figsize=(10, 6))
    plt.plot(training_loss, label="Training Loss", color='blue')
    if validation_loss is not None:
        plt.plot(validation_loss, label="Validation Loss", color='orange', linestyle='--')
    plt.title("Training and Validation Loss Curve")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
def plot_reward_components(reward_history):
    df = pd.DataFrame(reward_history)  # リワード履歴をDataFrameに変換

    # 各リワードコンポーネントのヒストグラムをプロット
    reward_components = ['price_change', 'transaction_cost', 'success_bonus',
                         'risk_penalty', 'buy_sell_bonus', 'hold_penalty', 'technical_adjustment', 'total']

    plt.figure(figsize=(16, 12))
    for i, component in enumerate(reward_components, 1):
        plt.subplot(3, 3, i)
        plt.hist(df[component], bins=50, alpha=0.7, label=component, color='blue')
        plt.axvline(df[component].mean(), color='red', linestyle='--', label='Mean')
        plt.title(f'Distribution of {component}')
        plt.legend()
    plt.tight_layout()
    plt.show()


def analyze_action_distribution(action_tracker):
    """
    アクションの分布を確認し、ヒストグラムをプロット。
    """
    if not action_tracker.action_history:
        print("Action history is empty. No data to analyze.")
        return

    # アクションのリスト
    actions = action_tracker.action_history

    # アクションの分布を計算
    unique_actions, counts = np.unique(actions, return_counts=True)
    print("Action Distribution:")
    for action, count in zip(unique_actions, counts):
        print(f"Action {action}: {count} occurrences")

    # ヒストグラムをプロット
    plt.figure(figsize=(10, 5))
    plt.bar(unique_actions, counts, color='skyblue', alpha=0.7)
    plt.title("Action Distribution")
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    plt.xticks(unique_actions)  # アクション値をX軸に表示
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
def plot_reward_timeseries(reward_history):
    df = pd.DataFrame(reward_history)

    plt.figure(figsize=(12, 6))
    plt.plot(df['total'], label='Total Reward', color='black')
    plt.title('Total Reward Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Total Reward')
    plt.axhline(0, color='red', linestyle='--')
    plt.legend()
    plt.show()

def debug_fx_data(fx_data):
    """
    fx_data オブジェクトの内容をデバッグ表示する関数。
    """
    if fx_data is None:
        print("[ERROR] fx_data is None. データが正しく渡されていません。")
        return

    try:
        # `data_with_indicators` の列名を確認
        print("[DEBUG] fx_data.data_with_indicators Columns:")
        print(fx_data.data_with_indicators.columns)

        # `price_data` の概要を表示
        print("[DEBUG] fx_data.price_data Head:")
        print(fx_data.price_data.head())

        # `success_streak` の状態を確認
        if hasattr(fx_data, 'success_streak'):
            print(f"[DEBUG] success_streak: {fx_data.success_streak}")
        else:
            print("[WARN] fx_data に success_streak 属性が見つかりません。")

        # `balance` や `position` を確認
        if hasattr(fx_data, 'balance') and hasattr(fx_data, 'position'):
            print(f"[DEBUG] Balance: {fx_data.balance}, Position: {fx_data.position}")
        else:
            print("[WARN] fx_data に balance または position 属性が見つかりません。")

    except AttributeError as e:
        print(f"[ERROR] fx_data に問題があります: {e}")

def update_weights(optimizer: Optimizer, network: Network, batch,
                   weight_decay: float, step: int, total_steps: int):
    """
    ネットワークの重みを更新する関数です。
    学習率スケジュールと動的損失重みを導入。
    """
    # 損失関数の動的重み計算
    value_loss_weight = 1.0 + (step / total_steps) * 0.5  # valueの重みを徐々に増加
    reward_loss_weight = 1.0
    policy_loss_weight = 2.0 - (step / total_steps) * 0.5  # policyの重みを徐々に減少

    # 学習率スケジュールに基づく学習率の調整
    initial_lr = optimizer.learning_rate.numpy()
    new_lr = initial_lr * (0.5 ** (step / total_steps))  # 学習率を指数関数的に減少
    optimizer.learning_rate.assign(new_lr)

    # 損失初期化
    loss = 0

    for image, actions, targets in batch:
        # 初期ステップ、実際の観察から推論
        output = network.initial_inference(image)
        predictions = [(1.0, output.value, output.reward, output.policy_logits)]

        # 再帰的なステップ、アクションと前の隠れ状態から推論
        for action in actions:
            output = network.recurrent_inference(output.hidden_state, action)
            predictions.append((1.0 / len(actions), output.value, output.reward, output.policy_logits))

        # 予測とターゲットを比較して損失を計算
        for prediction, target in zip(predictions, targets):
            gradient_scale, value, reward, policy_logits = prediction
            target_value, target_reward, target_policy = target

            # 動的損失重みを使用して損失を計算
            l = (
                value_loss_weight * scalar_loss(value, target_value) +
                reward_loss_weight * scalar_loss(reward, target_reward) +
                policy_loss_weight * tf.nn.softmax_cross_entropy_with_logits(
                    logits=policy_logits, labels=target_policy)
            )

            loss += tf.scale_gradient(l, gradient_scale)

    # 重み減衰を追加
    for weights in network.get_weights():
        loss += weight_decay * tf.nn.l2_loss(weights)

    # 勾配を計算し、最適化
    with tf.GradientTape() as tape:
        gradients = tape.gradient(loss, network.get_trainable_variables())
    optimizer.apply_gradients(zip(gradients, network.get_trainable_variables()))
import pandas as pd
import time
import oandapyV20
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts
from collections import Counter
import keyboard
API_KEY = "390c33dc1f6c743647ed7f49a1f3ca3a-feb4cd1718d954483e44696ef101b018"
ACCOUNT_ID = "101-009-30601503-001"
client = oandapyV20.API(access_token=API_KEY)
class RealTimeMonitor:
    def __init__(self):
        self.steps = []
        self.cumulative_rewards = []
        self.balances = []

        # グラフのセットアップ
        self.fig, self.ax1 = plt.subplots(figsize=(10, 6))
        self.line1, = self.ax1.plot([], [], label="Cumulative Reward", color="green")
        self.ax1.set_xlabel("Step")
        self.ax1.set_ylabel("Cumulative Reward", color="green")
        self.ax1.tick_params(axis='y', labelcolor="green")


        plt.title("Real-Time Trading Performance")
        self.fig.tight_layout()

    def update_graph(self, step, cumulative_reward, balance):
        self.steps.append(step)
        self.cumulative_rewards.append(cumulative_reward)
        self.balances.append(balance)
        self.line1.set_data(self.steps, self.cumulative_rewards)
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.fig.canvas.draw()
        plt.pause(0.01)

    def start_monitoring(self):
        plt.ion()
        plt.show()


def get_real_time_price(client, instrument="USD_JPY"):
    params = {"instruments": instrument}
    r = pricing.PricingInfo(accountID=ACCOUNT_ID, params=params)
    client.request(r)
    prices = r.response['prices']
    for price in prices:
        if price['instrument'] == instrument:
            bid = float(price['bids'][0]['price'])
            ask = float(price['asks'][0]['price'])
            return bid, ask

def check_real_time_done_condition(index, price_data, balance, position, cumulative_reward, config):
    current_price = price_data['Close'].iloc[index]
    volatility = fx_data.get_current_volatility(index)
    final_asset_value = balance + (position * current_price)

    # 終了条件
    if final_asset_value < config.stop_loss_limit:
        print(f"[DEBUG] Stop-loss Triggered: Balance = {final_asset_value}")
        return True
    if final_asset_value > config.take_profit_limit:
        print(f"[DEBUG] Take-Profit Triggered: Balance = {final_asset_value}")
        return True
    if volatility > config.max_volatility:
        print(f"[DEBUG] High Volatility Triggered: Volatility = {volatility}")
        return True
    return False
def calculate_units(balance, risk_percentage, pip_value=0.01, stop_loss_pips=50, leverage=30):
    risk_amount = balance * risk_percentage  # 許容リスク金額
    notional_value_per_pip = pip_value * stop_loss_pips  # 1ユニットのリスク
    units = risk_amount / notional_value_per_pip  # ユニット数を計算

    # 証拠金制約の計算
    available_margin = get_available_margin(client, ACCOUNT_ID)
    max_units_based_on_margin = available_margin / (pip_value * 100000)

    # レバレッジ制約
    max_units_based_on_leverage = balance * leverage / (pip_value * 100000)

    # 制約を満たす最大ユニット数を計算
    return int(min(units, max_units_based_on_margin, max_units_based_on_leverage))

def get_available_margin(client, account_id):
    """
    使用可能なマージンを取得する。
    """
    r = accounts.AccountDetails(accountID=account_id)
    try:
        response = client.request(r)
        return float(response['account']['marginAvailable'])
    except Exception as e:
        print(f"Failed to fetch available margin: {e}")
        return 0  # デフォルト値

def place_order(client, account_id, instrument, units, retries=3):
    for attempt in range(retries):
        try:
            order = {
                "order": {
                    "instrument": instrument,
                    "units": str(units),
                    "type": "MARKET",
                    "positionFill": "DEFAULT"
                }
            }
            r = orders.OrderCreate(account_id, data=order)
            response = client.request(r)
            if 'orderCancelTransaction' in response:
                reason = response['orderCancelTransaction']['reason']
                if reason == 'MARKET_HALTED':
                    print(f"[DEBUG] Market halted. Retrying... (Attempt {attempt+1}/{retries})")
                    time.sleep(1)
                    continue
            return response
        except Exception as e:
            print(f"Order failed: {e}")
            time.sleep(1)
# 日本時間に基づく
import datetime
def is_market_open():
    """
    市場が開いているかどうかを確認する関数。
    - 市場は土曜日と日曜日はクローズ。
    - 平日はUTC時間で22:00（日付切り替え）〜翌日21:00までオープン。
    """
    now = datetime.datetime.utcnow() + datetime.timedelta(hours=9)  # 日本時間に変換

    if now.weekday() >= 5:  # 土日クローズ
        return False

    # オープン時間 7:00(月曜) 〜 6:00(土曜)
    if now.weekday() == 0 and now.time() < datetime.time(7, 0):
        return False
    if now.weekday() == 5 and now.time() >= datetime.time(6, 0):
        return False
    return True


def execute_real_time_trading(config, network, fx_data, client, monitor, instrument="USD_JPY", risk_percentage=0.01):
    monitor.start_monitoring()
    step = 0  # ステップの初期化

    while True:  # 無限ループで再開可能
        state = initialize_fx_environment(fx_data)
        cumulative_reward = 0.0

        while True:
            try:
                # 市場オープン確認
                if not is_market_open():
                    print("[DEBUG] Market is closed. Pausing trading...")
                    time.sleep(60)
                    continue

                # データ不足チェック
                if len(fx_data.price_data) < 20:  # 最低20ステップ必要
                    print(f"[DEBUG] Waiting for more data: step {len(fx_data.price_data)}")
                    time.sleep(1)
                    continue

                # 価格データを取得
                bid, ask = get_real_time_price(client, instrument)
                current_price = (bid + ask) / 2
                fx_data.current_price = current_price

                # データ更新と状態取得
                state = fx_data.make_image(step)
                balance = FXData.get_account_balance(client, ACCOUNT_ID)

                # ボラティリティチェック
                volatility = fx_data.get_current_volatility(step)
                if volatility == 0:
                    print(f"[DEBUG] Volatility insufficient at step {step}. Skipping...")
                    step += 1
                    time.sleep(1)
                    continue

                # アクションと注文処理
                action = network_policy(network, state, step, config)
                units = calculate_units(balance, risk_percentage)

                if action == 0:  # Buy
                    if fx_data.position < 0:  # 売りポジションの決済
                        print("Closing Sell Position")
                        response = place_order(client, ACCOUNT_ID, instrument, abs(fx_data.position))
                        if 'orderFillTransaction' in response:
                            fx_data.position = 0

                    print("Executing Buy Order")
                    response = place_order(client, ACCOUNT_ID, instrument, units)
                    if 'orderFillTransaction' in response:
                        fx_data.position += units

                elif action == 1:  # Sell
                    if fx_data.position > 0:  # 買いポジションの決済
                        print("Closing Buy Position")
                        response = place_order(client, ACCOUNT_ID, instrument, -fx_data.position)
                        if 'orderFillTransaction' in response:
                            fx_data.position = 0

                    print("Executing Sell Order")
                    response = place_order(client, ACCOUNT_ID, instrument, -units)
                    if 'orderFillTransaction' in response:
                        fx_data.position -= units

                else:  # Hold
                    print("Hold action taken. No order placed.")

                # 終了条件
                if check_real_time_done_condition(step, fx_data.price_data, fx_data.balance, fx_data.position, cumulative_reward, config):
                    print("[DEBUG] Trading session ended due to a safety condition.")
                    break

                step += 1
                time.sleep(1)

            except Exception as e:
                print(f"Error fetching price data: {e}")
                time.sleep(1)
                continue




if __name__ == "__main__":
    # 初期残高を取得
    initial_balance = FXData.get_account_balance(client, ACCOUNT_ID)
    config = MuZeroConfig(initial_balance, stop_loss_percentage=10, take_profit_percentage=15)
    data = pd.read_csv('./exchange_rate_data_filtered.csv')
    fx_data = FXData(data, client=client, account_id=ACCOUNT_ID)
    storage = SharedStorage()
    if len(storage._networks) == 0:
        initial_network = Network(input_shape=(10,), action_space_size=3)
        storage.save_network(0, initial_network)

    network = storage.latest_network()
    monitor = RealTimeMonitor()
    execute_real_time_trading(config, network, fx_data, client, monitor)



