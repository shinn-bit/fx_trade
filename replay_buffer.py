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

