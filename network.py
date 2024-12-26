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
