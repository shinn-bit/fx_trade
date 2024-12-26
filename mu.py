import numpy as np
import pandas as pd

class FXReplayBuffer:
    def __init__(self, config):
        self.window_size = config.window_size  # バッファに保存する最大数
        self.batch_size = config.batch_size    # サンプリングするバッチサイズ
        self.buffer = []                       # バッファの初期化

    def save_data(self, fx_data):
        """
        FXデータをバッファに保存する。バッファが満杯なら古いデータを削除。
        """
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(fx_data)

    def sample_batch(self, num_unroll_steps, td_steps):
        """
        バッチサイズ分だけFXデータをサンプリングし、モデルに学習させるデータを返す。
        """
        sampled_data = [self.sample_data() for _ in range(self.batch_size)]
        data_positions = [(data, self.sample_position(data)) for data in sampled_data]
        
        # サンプルデータに基づき、状態（画像）、履歴、ターゲットを返す
        return [(data.make_image(i), data.history[i:i + num_unroll_steps], 
                 data.make_target(i, num_unroll_steps, td_steps))
                for data, i in data_positions]

    def sample_data(self):
        """
        バッファからランダムに1つのFXデータをサンプリング。
        """
        return self.buffer[np.random.randint(len(self.buffer))]

    def sample_position(self, data):
        """
        FXデータの中からランダムに取引の位置（インデックス）を選ぶ。
        """
        # 例えば過去のデータの一部を選ぶ
        return np.random.randint(0, len(data.history) - 1)
    
# RSIの計算関数
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
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
csv_file_path ='/content/exchange_rate_data_filtered.csv'
class FXData:
    def __init__(self, csv_file_path):
        """
        CSVファイルを読み込み、価格データと日付を設定し、
        テクニカル指標を計算し、スケーリングを行う。
        """
        # CSVファイルの読み込み
        df = pd.read_csv(csv_file_path)
        df['Date'] = pd.to_datetime(df['Date'])

        # 必要なデータを抽出（価格データと日付）
        self.date = df['Date']
        self.price_data = df[['Open', 'High', 'Low', 'Close']]

        # テクニカル指標を計算
        df['SMA50'] = df['Close'].rolling(window=50).mean()  # 50日単純移動平均
        df['SMA200'] = df['Close'].rolling(window=200).mean()  # 200日単純移動平均
        df['RSI'] = calculate_rsi(df['Close'])  # RSI
        df['BB_Middle'], df['BB_Upper'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])

        # 欠損データを削除
        df = df.dropna()

        # スケーリングする列を選択
        features_to_scale = ['Open', 'High', 'Low', 'Close', 'SMA50', 'SMA200', 
                             'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower']
        
        # スケーリングを適用
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(df[features_to_scale])

        # スケール済みデータフレームを作成
        scaled_df = pd.DataFrame(scaled_features, columns=features_to_scale)
        scaled_df['Date'] = df['Date'].values  # 日付はそのまま保持

        # スケール済みデータを保持
        self.scaled_data = scaled_df

        # テクニカル指標を含む元データも保持
        self.data_with_indicators = df

        # 過去の履歴（取引履歴）
        self.history = []
        self.to_play = 1  # 現在の取引ターン（通常はエージェントがプレイ中）

    def make_image(self, index):
        """
        特徴量としてスケール済みの価格データとテクニカル指標を含む配列を生成。
        """
        row = self.scaled_data.iloc[index]
        return np.array([
            row['Open'],
            row['High'],
            row['Low'],
            row['Close'],
            row['SMA50'],
            row['SMA200'],
            row['RSI'],
            row['BB_Middle'],
            row['BB_Upper'],
            row['BB_Lower']
        ])

    def make_target(self, index, num_unroll_steps, td_steps):
        """
        次の価格予測や報酬をターゲットとして生成。
        """
        target_price = self.price_data.iloc[index + num_unroll_steps]['Close']  # 予測する価格（Close）
        target_reward = self.calculate_reward(index, num_unroll_steps)  # 報酬の計算
        
        return {
            'target_price': target_price,  # 予測する価格（Close）
            'target_reward': target_reward,  # 報酬
        }

    def calculate_reward(self, index, num_unroll_steps):
        """
        次の価格との差を報酬として計算。
        """
        reward = self.price_data.iloc[index + num_unroll_steps]['Close'] - self.price_data.iloc[index]['Close']
        return reward


    # 可視化機能
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


# 使用例
# FXデータが入ったCSVファイルを読み込み
fx_data = FXData(csv_file_path)

# インデックス0のデータを状態画像として取得
state_image = fx_data.make_image(0)

# ターゲットデータ（予測価格と報酬）を取得
target_data = fx_data.make_target(0, num_unroll_steps=5, td_steps=1)
# 結果を表示
print("State Image (Close Price):", state_image)
print("Target Data:", target_data)
# 可視化機能を実行
fx_data.plot_bollinger_bands()  # ボリンジャーバンドをプロット
fx_data.plot_rsi()              # RSI をプロット
fx_data.plot_moving_averages()  # 移動平均をプロット
fx_data.plot_price_with_indicators()  # すべての指標をまとめてプロット

import tensorflow as tf


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


# 訓練と評価用の補助関数
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

def plot_loss_curve(training_loss, validation_loss=None):
    """
    訓練中の損失曲線をプロット。
    """
    plt.figure(figsize=(10, 6))
    plt.plot(training_loss, label="Training Loss", color='blue')
    if validation_loss:
        plt.plot(validation_loss, label="Validation Loss", color='orange')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
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
    return mae, mse       
        
# MuZeroConfig クラスの定義（これを run_selfplay より前に置く必要があります）
class MuZeroConfig:
    def __init__(self):
        self.window_size = 100
        self.batch_size = 32
        self.lr_init = 0.01
        self.lr_decay_rate = 0.9
        self.lr_decay_steps = 100
        self.training_steps = 1000
        self.checkpoint_interval = 100
        self.num_unroll_steps = 5
        self.td_steps = 1
        self.momentum = 0.9
        self.weight_decay = 0.0001

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
import numpy as np
import tensorflow as tf
import tensorflow as tf
import numpy as np

class NetworkOutput:
    def __init__(self, value, reward, policy_logits, hidden_state):
        """
        モデルの推論結果を格納するクラス。
        """
        self.value = value  # 推定される状態の価値
        self.reward = reward  # 得られる報酬
        self.policy_logits = policy_logits  # 各アクションに対するポリシー分布
        self.hidden_state = hidden_state  # 隠れ状態

    def __str__(self):
        return (f"Value: {self.value}, Reward: {self.reward}, "
                f"Policy: {self.policy_logits}, Hidden State Shape: {self.hidden_state.shape}")


class Network:
    def __init__(self, input_shape=(10,), action_space_size=3):
        """
        改良されたMuZeroのネットワーク構造。
        """
        # Representationモデル: 状態表現を生成
        self.representation = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2)
        ])

        # Dynamicsモデル: 次の状態と報酬を予測
        self.dynamics = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(128 + action_space_size,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(129)  # 次の状態(128)と報酬(1)を同時に出力
        ])

        # Predictionモデル: 価値とポリシーを予測
        self.prediction = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(128,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(action_space_size, activation='softmax')  # ポリシー出力
        ])

    def initial_inference(self, observation):
        """
        初期の状態をエンコードし、価値、報酬、ポリシーを出力。
        """
        state = self.representation(observation)
        value = tf.reduce_mean(state).numpy()  # 仮の価値計算（平均値を利用）
        policy_logits = self.prediction(state).numpy()  # NumPy形式に変換
        return NetworkOutput(value, 0.0, policy_logits, state.numpy())  # 報酬は初期値として0を設定

    def recurrent_inference(self, state, action):
        """
        現在の状態とアクションを入力し、次の状態、価値、ポリシーを出力。
        """
        # アクションをワンホットエンコード
        action_one_hot = tf.one_hot(action, depth=3)  # [action_space_size]

        # アクションの形状を [batch_size, action_space_size] に変換
        action_one_hot = tf.expand_dims(action_one_hot, axis=0)  # 2D 化: [1, action_space_size]

        # 状態とアクションを結合
        combined_input = tf.concat([state, action_one_hot], axis=1)

        # Dynamicsモデルで次の状態と報酬を予測
        dynamics_output = self.dynamics(combined_input)
        next_state, reward = tf.split(dynamics_output, [128, 1], axis=1)  # 次の状態と報酬を分離

        # Predictionモデルで価値とポリシーを予測
        policy_logits = self.prediction(next_state).numpy()
        value = tf.reduce_mean(next_state).numpy()  # 仮の価値計算

        # 報酬を NumPy スカラーに変換
        reward = reward.numpy().squeeze()

        return NetworkOutput(value, reward, policy_logits, next_state.numpy())

# ダミーデータでテスト
if __name__ == "__main__":
    # ネットワークの初期化
    network = Network()

    # ダミー観測データ
    observation = tf.random.uniform((1, 10))  # バッチサイズ1、入力次元10

    # 初期推論
    initial_output = network.initial_inference(observation)
    print("Initial Inference Output:")
    print(initial_output)

    # 再帰推論
    action = 1  # ダミーアクション
    recurrent_output = network.recurrent_inference(initial_output.hidden_state, action)
    print("\nRecurrent Inference Output:")
    print(recurrent_output)


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
        # Network クラスの定義に合わせて調整してください
        # ここではポリシーが均等に初期化され、値と報酬が0であるネットワークを作成
        return Network(policy=uniform_policy(), value=0, reward=0)
    


def run_selfplay(config: MuZeroConfig,
                 storage: SharedStorage,
                 replay_buffer: FXReplayBuffer):
    """
    MuZero を用いてFXシステムトレーディングを行い、その結果をリプレイバッファに保存する関数です。
    """
    while True:
        # 最新のネットワークを取得
        network = storage.latest_network()

        # FX取引を行うゲームをプレイ
        game = play_fx_game(config, network)

        # ゲーム（取引の結果）をリプレイバッファに保存
        replay_buffer.save_data(game)

        # 定期的に訓練を行ったり、ネットワークを保存する処理も追加する
        # 例: ここでネットワークを保存したりする処理を入れることができます


def play_fx_game(config: MuZeroConfig, network: Network) -> Game:
    """
    MuZero を用いてFX取引のシミュレーションを行います。
    この関数では、為替レートデータに基づいてエージェントが取引を行う。
    """
    state = initialize_fx_environment(config)  # 初期状態（為替レートデータなど）
    game = Game()  # ゲームの結果を格納するためのクラス
    done = False
    
    while not done:
        # ネットワークからのアクションを選択
        action = network_policy(network, state)  # 売買のアクション（買い、売り、何もしない）
        
        # アクションを実行し、次の状態と報酬を計算
        next_state, reward, done = execute_action(state, action)
        
        # 状態、アクション、報酬、次の状態をゲームに保存
        game.add_step(state, action, reward, next_state)
        
        # 次の状態に移行
        state = next_state
    
    return game

import tensorflow as tf

def train_network(config: MuZeroConfig, storage: SharedStorage, replay_buffer: FXReplayBuffer):
    """
    MuZeroのネットワークを訓練する関数。
    損失の記録や視覚化を追加。
    """
    network = Network()
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=config.lr_init,
        momentum=config.momentum
    )

    training_loss = []  # 訓練損失を記録
    validation_loss = []  # 検証損失を記録（必要に応じて追加）

    for step in range(config.training_steps):
        # リプレイバッファからバッチをサンプリング
        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)

        # 損失を計算
        with tf.GradientTape() as tape:
            loss = compute_loss(network, batch, config.weight_decay)

        # 勾配の計算と適用
        gradients = tape.gradient(loss, network.trainable_variables)
        optimizer.apply_gradients(zip(gradients, network.trainable_variables))

        # 損失を記録
        training_loss.append(loss.numpy())

        # 定期的にチェックポイントを保存
        if step % config.checkpoint_interval == 0:
            storage.save_network(step, network)
            print(f"Step {step}: Loss = {loss.numpy()}")

    # 損失曲線をプロット
    plot_loss_curve(training_loss)

    # 訓練後のネットワークを保存
    storage.save_network(config.training_steps, network)


def compute_loss(network: Network, batch, weight_decay):
    """
    MuZeroの損失関数を計算する。
    """
    loss = 0
    for image, actions, targets in batch:
        value, reward, policy_logits, hidden_state = network.initial_inference(image)
        predictions = [(1.0, value, reward, policy_logits)]

        for action in actions:
            value, reward, policy_logits, hidden_state = network.recurrent_inference(hidden_state, action)
            predictions.append((1.0 / len(actions), value, reward, policy_logits))

        # 各ターゲットとの損失を計算
        for prediction, target in zip(predictions, targets):
            gradient_scale, value, reward, policy_logits = prediction
            target_value, target_reward, target_policy = target

            l = (
                scalar_loss(value, target_value) +
                scalar_loss(reward, target_reward) +
                tf.nn.softmax_cross_entropy_with_logits(labels=target_policy, logits=policy_logits)
            )

            loss += gradient_scale * l

    # 重み減衰（L2正則化）を追加
    for weights in network.trainable_variables:
        loss += weight_decay * tf.nn.l2_loss(weights)

    return loss



from tensorflow.keras.optimizers import Optimizer  # 正しい型ヒントを追加

def update_weights(optimizer: Optimizer, network: Network, batch,
                   weight_decay: float):

    """
    ネットワークの重みを更新する関数です。
    各バッチからサンプルを取得し、MuZero の損失関数を計算してネットワークの重みを更新します。
    """
    loss = 0
    for image, actions, targets in batch:
        # 初期ステップ、実際の観察から推論
        value, reward, policy_logits, hidden_state = network.initial_inference(image)
        predictions = [(1.0, value, reward, policy_logits)]

        # 再帰的なステップ、アクションと前の隠れ状態から推論
        for action in actions:
            value, reward, policy_logits, hidden_state = network.recurrent_inference(hidden_state, action)
            predictions.append((1.0 / len(actions), value, reward, policy_logits))
            hidden_state = tf.scale_gradient(hidden_state, 0.5)

        # 予測とターゲットを比較して損失を計算
        for prediction, target in zip(predictions, targets):
            gradient_scale, value, reward, policy_logits = prediction
            target_value, target_reward, target_policy = target

            # スカラー損失を計算
            l = (
                scalar_loss(value, target_value) +
                scalar_loss(reward, target_reward) +
                tf.nn.softmax_cross_entropy_with_logits(logits=policy_logits, labels=target_policy)
            )

            loss += tf.scale_gradient(l, gradient_scale)

    # 重み減衰を追加
    for weights in network.get_weights():
        loss += weight_decay * tf.nn.l2_loss(weights)

    # 勾配を計算し、最適化
    optimizer.minimize(loss)
  
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

if __name__ == "__main__":
    # ダミーデータ生成
    np.random.seed(42)
    y_true = np.linspace(0, 10, 100)  # 実際の値
    y_pred = y_true + np.random.normal(0, 0.5, size=y_true.shape)  # 予測値（ノイズ追加）

    # モデル性能の評価と可視化
    print("Evaluating model performance...")
    mae, mse = evaluate_model(y_true, y_pred)  # 評価指標を表示
    visualize_model_performance(y_true, y_pred)  # 性能を可視化

    # 訓練中の損失曲線をプロット
    training_loss = np.random.normal(1, 0.1, size=50)  # ダミーの損失データ
    plot_loss_curve(training_loss)



