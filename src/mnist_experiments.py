#!/usr/bin/env python3
"""
MNIST データセットでのELM性能テスト
従来ELMと活性化関数逆転ELMの比較
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

class TraditionalELM:
    """従来のELM（分類用）"""
    
    def __init__(self, input_size, hidden_size, output_size, activation='tanh'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 隠れ層の重みとバイアスは固定（ランダム）
        self.hidden_weights = np.random.randn(input_size, hidden_size) * 0.5
        self.hidden_bias = np.random.randn(hidden_size) * 0.5
        
        # 出力層の重みのみ学習
        self.output_weights = None
        
        # 活性化関数
        if activation == 'tanh':
            self.activation = np.tanh
        elif activation == 'relu':
            self.activation = lambda x: np.maximum(0, x)
        else:
            self.activation = lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _compute_hidden_output(self, X):
        """隠れ層の出力を計算"""
        hidden_input = np.dot(X, self.hidden_weights) + self.hidden_bias
        return self.activation(hidden_input)
    
    def fit(self, X, y):
        """学習（出力層の重みを最小二乗法で計算）"""
        # 隠れ層の出力を計算
        H = self._compute_hidden_output(X)
        
        # 出力層の重みを最小二乗法で計算
        # H * output_weights = y を解く
        try:
            self.output_weights = np.linalg.pinv(H) @ y
        except np.linalg.LinAlgError:
            # 特異行列の場合は正則化項を追加
            reg = 1e-6
            self.output_weights = np.linalg.inv(H.T @ H + reg * np.eye(H.shape[1])) @ H.T @ y
    
    def predict(self, X):
        """予測"""
        H = self._compute_hidden_output(X)
        output = H @ self.output_weights
        return np.argmax(output, axis=1)
    
    def predict_proba(self, X):
        """予測確率"""
        H = self._compute_hidden_output(X)
        output = H @ self.output_weights
        # ソフトマックス
        exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
        return exp_output / np.sum(exp_output, axis=1, keepdims=True)

class ReversedActivationELM:
    """活性化関数位置逆転ELM（分類用）"""
    
    def __init__(self, input_size, hidden_size, output_size, activation='tanh'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 全ての重みを学習可能に初期化
        self.input_weights = np.random.randn(input_size, hidden_size) * 0.1
        self.output_weights = np.random.randn(hidden_size, output_size) * 0.1
        
        # 活性化関数
        if activation == 'tanh':
            self.activation = np.tanh
        elif activation == 'relu':
            self.activation = lambda x: np.maximum(0, x)
        else:
            self.activation = lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _forward(self, X):
        """順伝播（活性化関数位置逆転）"""
        # 1. 入力に活性化関数を先に適用
        activated_input = self.activation(X)
        
        # 2. 活性化済み入力で隠れ層計算
        hidden_output = np.dot(activated_input, self.input_weights)
        
        # 3. 隠れ層出力に再度活性化関数
        activated_hidden = self.activation(hidden_output)
        
        return activated_input, activated_hidden
    
    def fit(self, X, y, epochs=100, learning_rate=0.01, batch_size=100):
        """学習（全層を勾配降下法で学習）"""
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # バッチ学習
            indices = np.random.permutation(n_samples)
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                # 順伝播
                activated_input, activated_hidden = self._forward(X_batch)
                output = np.dot(activated_hidden, self.output_weights)
                
                # 誤差計算
                error = output - y_batch
                
                # 逆伝播（簡易版）
                # 出力層の勾配
                d_output_weights = activated_hidden.T @ error / batch_size
                
                # 隠れ層の勾配（活性化関数の微分を近似）
                d_hidden = error @ self.output_weights.T
                d_hidden_activated = d_hidden * (1 - activated_hidden**2)  # tanh微分の近似
                
                # 入力層の勾配
                d_input_weights = activated_input.T @ d_hidden_activated / batch_size
                
                # 重み更新
                self.output_weights -= learning_rate * d_output_weights
                self.input_weights -= learning_rate * d_input_weights
            
            # 定期的に損失を表示
            if epoch % 20 == 0:
                _, activated_hidden = self._forward(X)
                output = np.dot(activated_hidden, self.output_weights)
                loss = np.mean((output - y)**2)
                print(f"  Epoch {epoch:3d}: Loss = {loss:.6f}")
    
    def predict(self, X):
        """予測"""
        _, activated_hidden = self._forward(X)
        output = np.dot(activated_hidden, self.output_weights)
        return np.argmax(output, axis=1)
    
    def predict_proba(self, X):
        """予測確率"""
        _, activated_hidden = self._forward(X)
        output = np.dot(activated_hidden, self.output_weights)
        # ソフトマックス
        exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
        return exp_output / np.sum(exp_output, axis=1, keepdims=True)

def load_mnist_subset(n_samples=5000):
    """MNIST データセットの一部を読み込み"""
    print("MNISTデータセットを読み込み中...")
    
    # OpenMLからMNISTを取得（時間短縮のため一部のみ）
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)
    
    # データを正規化
    X = X / 255.0
    
    # サブセットを作成
    indices = np.random.choice(len(X), n_samples, replace=False)
    X_subset = X[indices]
    y_subset = y[indices]
    
    print(f"データセットサイズ: {X_subset.shape}")
    print(f"クラス分布: {np.bincount(y_subset)}")
    
    return X_subset, y_subset

def prepare_data(X, y):
    """データの前処理"""
    # 訓練・テストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ワンホットエンコーディング
    n_classes = len(np.unique(y))
    y_train_onehot = np.eye(n_classes)[y_train]
    y_test_onehot = np.eye(n_classes)[y_test]
    
    return X_train_scaled, X_test_scaled, y_train, y_test, y_train_onehot, y_test_onehot

def test_elm_performance():
    """ELM性能テスト"""
    print("=" * 60)
    print("MNIST ELM 性能テスト")
    print("=" * 60)
    
    # データ読み込み
    X, y = load_mnist_subset(n_samples=3000)  # 計算時間短縮のため3000サンプル
    X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot = prepare_data(X, y)
    
    input_size = X_train.shape[1]  # 784
    hidden_size = 200  # 隠れ層のサイズ
    output_size = 10   # クラス数
    
    results = {}
    
    # 1. 従来ELMのテスト
    print("\\n1. 従来ELM テスト")
    print("-" * 30)
    
    start_time = time.time()
    traditional_elm = TraditionalELM(input_size, hidden_size, output_size)
    traditional_elm.fit(X_train, y_train_onehot)
    train_time_traditional = time.time() - start_time
    
    # 予測
    start_time = time.time()
    y_pred_traditional = traditional_elm.predict(X_test)
    predict_time_traditional = time.time() - start_time
    
    accuracy_traditional = accuracy_score(y_test, y_pred_traditional)
    
    print(f"学習時間: {train_time_traditional:.3f}秒")
    print(f"予測時間: {predict_time_traditional:.3f}秒")
    print(f"精度: {accuracy_traditional:.4f} ({accuracy_traditional*100:.2f}%)")
    
    results['traditional'] = {
        'accuracy': accuracy_traditional,
        'train_time': train_time_traditional,
        'predict_time': predict_time_traditional
    }
    
    # 2. 活性化関数逆転ELMのテスト
    print("\\n2. 活性化関数逆転ELM テスト")
    print("-" * 30)
    
    start_time = time.time()
    reversed_elm = ReversedActivationELM(input_size, hidden_size, output_size)
    reversed_elm.fit(X_train, y_train_onehot, epochs=50, learning_rate=0.001)
    train_time_reversed = time.time() - start_time
    
    # 予測
    start_time = time.time()
    y_pred_reversed = reversed_elm.predict(X_test)
    predict_time_reversed = time.time() - start_time
    
    accuracy_reversed = accuracy_score(y_test, y_pred_reversed)
    
    print(f"学習時間: {train_time_reversed:.3f}秒")
    print(f"予測時間: {predict_time_reversed:.3f}秒")
    print(f"精度: {accuracy_reversed:.4f} ({accuracy_reversed*100:.2f}%)")
    
    results['reversed'] = {
        'accuracy': accuracy_reversed,
        'train_time': train_time_reversed,
        'predict_time': predict_time_reversed
    }
    
    # 3. 結果比較
    print("\\n" + "=" * 60)
    print("結果比較")
    print("=" * 60)
    
    print(f"{'手法':<20} {'精度':<10} {'学習時間':<10} {'予測時間':<10}")
    print("-" * 50)
    print(f"{'従来ELM':<20} {results['traditional']['accuracy']*100:>6.2f}% {results['traditional']['train_time']:>8.3f}s {results['traditional']['predict_time']:>8.3f}s")
    print(f"{'活性化関数逆転ELM':<20} {results['reversed']['accuracy']*100:>6.2f}% {results['reversed']['train_time']:>8.3f}s {results['reversed']['predict_time']:>8.3f}s")
    
    # 詳細レポート
    print("\\n詳細分析:")
    print(f"精度差: {(results['reversed']['accuracy'] - results['traditional']['accuracy'])*100:+.2f}%")
    print(f"学習時間比: {results['reversed']['train_time'] / results['traditional']['train_time']:.1f}倍")
    
    # 混同行列
    print("\\n従来ELM 混同行列:")
    print(confusion_matrix(y_test, y_pred_traditional))
    
    print("\\n活性化関数逆転ELM 混同行列:")
    print(confusion_matrix(y_test, y_pred_reversed))
    
    return results

def test_different_hidden_sizes():
    """異なる隠れ層サイズでの性能比較"""
    print("\\n" + "=" * 60)
    print("隠れ層サイズ別性能テスト")
    print("=" * 60)
    
    # データ読み込み
    X, y = load_mnist_subset(n_samples=2000)  # さらに小さなデータセット
    X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot = prepare_data(X, y)
    
    hidden_sizes = [50, 100, 200, 300]
    results = {'traditional': [], 'reversed': []}
    
    for hidden_size in hidden_sizes:
        print(f"\\n隠れ層サイズ: {hidden_size}")
        print("-" * 20)
        
        # 従来ELM
        traditional_elm = TraditionalELM(784, hidden_size, 10)
        start_time = time.time()
        traditional_elm.fit(X_train, y_train_onehot)
        train_time = time.time() - start_time
        
        y_pred = traditional_elm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results['traditional'].append({
            'hidden_size': hidden_size,
            'accuracy': accuracy,
            'train_time': train_time
        })
        
        print(f"従来ELM: {accuracy*100:.2f}% ({train_time:.3f}s)")
        
        # 活性化関数逆転ELM
        reversed_elm = ReversedActivationELM(784, hidden_size, 10)
        start_time = time.time()
        reversed_elm.fit(X_train, y_train_onehot, epochs=30, learning_rate=0.001)
        train_time = time.time() - start_time
        
        y_pred = reversed_elm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results['reversed'].append({
            'hidden_size': hidden_size,
            'accuracy': accuracy,
            'train_time': train_time
        })
        
        print(f"逆転ELM: {accuracy*100:.2f}% ({train_time:.3f}s)")
    
    # 結果をプロット
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    traditional_acc = [r['accuracy']*100 for r in results['traditional']]
    reversed_acc = [r['accuracy']*100 for r in results['reversed']]
    
    plt.plot(hidden_sizes, traditional_acc, 'ro-', label='Traditional ELM', linewidth=2)
    plt.plot(hidden_sizes, reversed_acc, 'bo-', label='Reversed Activation ELM', linewidth=2)
    plt.xlabel('Hidden Layer Size')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Hidden Layer Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    traditional_time = [r['train_time'] for r in results['traditional']]
    reversed_time = [r['train_time'] for r in results['reversed']]
    
    plt.plot(hidden_sizes, traditional_time, 'ro-', label='Traditional ELM', linewidth=2)
    plt.plot(hidden_sizes, reversed_time, 'bo-', label='Reversed Activation ELM', linewidth=2)
    plt.xlabel('Hidden Layer Size')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time vs Hidden Layer Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/mnist_elm_comparison.png', dpi=150, bbox_inches='tight')
    print("\\n比較グラフを保存: /home/ubuntu/mnist_elm_comparison.png")
    
    return results

if __name__ == "__main__":
    # メイン性能テスト
    main_results = test_elm_performance()
    
    # 隠れ層サイズ別テスト
    size_results = test_different_hidden_sizes()
    
    print("\\n" + "=" * 60)
    print("MNIST ELM テスト完了")
    print("=" * 60)
