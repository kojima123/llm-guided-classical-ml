#!/usr/bin/env python3
"""
LLM-ELMハイブリッドシステム実装
LLMが教師となってELMを指導するリアルタイム学習システム
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from typing import Dict, List, Tuple, Optional
import os

# OpenAI APIが利用できない場合のダミー実装
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not available, using dummy LLM evaluator")

class RobotSimulation:
    """2Dロボットシミュレーション環境"""
    
    def __init__(self, width: float = 10, height: float = 10):
        self.width = width
        self.height = height
        self.reset()
    
    def reset(self) -> np.ndarray:
        """環境をリセット"""
        # ロボット初期位置
        self.robot_pos = np.array([2.0, 2.0])
        self.robot_angle = 0.0
        self.robot_vel = np.array([0.0, 0.0])
        
        # ボール初期位置（ランダム）
        self.ball_pos = np.array([
            np.random.uniform(6.0, 9.0),
            np.random.uniform(6.0, 9.0)
        ])
        self.ball_vel = np.array([
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(-0.2, 0.2)
        ])
        
        # 障害物
        self.obstacles = [
            np.array([5.0, 5.0]),  # 中央の障害物
            np.array([3.0, 7.0]),  # 追加の障害物
            np.array([7.0, 3.0]),  # 追加の障害物
        ]
        
        self.time = 0
        self.collision_count = 0
        return self.get_sensor_data()
    
    def get_sensor_data(self) -> np.ndarray:
        """センサーデータを取得"""
        # ボールとの相対位置
        ball_diff = self.ball_pos - self.robot_pos
        ball_distance = np.linalg.norm(ball_diff)
        ball_angle = np.arctan2(ball_diff[1], ball_diff[0]) - self.robot_angle
        
        # 正規化
        ball_angle = np.arctan2(np.sin(ball_angle), np.cos(ball_angle))  # -π to π
        
        # 障害物との距離（8方向）
        obstacle_distances = []
        directions = np.array([
            [1, 0], [1, 1], [0, 1], [-1, 1], 
            [-1, 0], [-1, -1], [0, -1], [1, -1]
        ])
        
        for direction in directions:
            direction = direction / np.linalg.norm(direction)
            min_dist = float('inf')
            
            # レイキャスティング
            for step in np.arange(0.1, 4.0, 0.1):
                check_pos = self.robot_pos + direction * step
                
                # 境界チェック
                if (check_pos[0] <= 0 or check_pos[0] >= self.width or 
                    check_pos[1] <= 0 or check_pos[1] >= self.height):
                    min_dist = min(min_dist, step)
                    break
                
                # 障害物チェック
                for obstacle in self.obstacles:
                    if np.linalg.norm(check_pos - obstacle) < 0.6:
                        min_dist = min(min_dist, step)
                        break
                
                if min_dist < 4.0:
                    break
            
            obstacle_distances.append(min(min_dist, 4.0))
        
        # センサーデータ統合
        sensor_data = np.array([
            ball_distance / 10.0,           # ボール距離 (正規化)
            ball_angle / np.pi,             # ボール角度 (-1 to 1)
            np.linalg.norm(self.robot_vel), # 現在の速度
            self.robot_angle / np.pi,       # ロボット角度
        ] + [d / 4.0 for d in obstacle_distances])  # 障害物距離 (正規化)
        
        return sensor_data
    
    def step(self, action: np.ndarray) -> np.ndarray:
        """アクションを実行して環境を更新"""
        forward_speed = np.clip(action[0], 0, 1)
        angular_speed = np.clip(action[1], -1, 1)
        
        # ロボット移動
        dt = 0.1
        self.robot_angle += angular_speed * dt
        
        # 角度を-π to πに正規化
        self.robot_angle = np.arctan2(np.sin(self.robot_angle), np.cos(self.robot_angle))
        
        move_dir = np.array([np.cos(self.robot_angle), np.sin(self.robot_angle)])
        self.robot_vel = move_dir * forward_speed
        new_pos = self.robot_pos + self.robot_vel * dt
        
        # 境界チェック
        new_pos = np.clip(new_pos, 0.5, [self.width-0.5, self.height-0.5])
        self.robot_pos = new_pos
        
        # ボール移動
        self.ball_pos += self.ball_vel * dt
        
        # ボールの境界反射
        if self.ball_pos[0] <= 0.5 or self.ball_pos[0] >= self.width - 0.5:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = np.clip(self.ball_pos[0], 0.5, self.width - 0.5)
        if self.ball_pos[1] <= 0.5 or self.ball_pos[1] >= self.height - 0.5:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = np.clip(self.ball_pos[1], 0.5, self.height - 0.5)
        
        self.time += dt
        
        return self.get_sensor_data()
    
    def get_state_info(self) -> Dict:
        """現在の状態情報を取得（LLM評価用）"""
        ball_distance = np.linalg.norm(self.ball_pos - self.robot_pos)
        
        # 障害物との衝突チェック
        collision = False
        for obstacle in self.obstacles:
            if np.linalg.norm(self.robot_pos - obstacle) < 0.8:
                collision = True
                self.collision_count += 1
                break
        
        return {
            'ball_distance': float(ball_distance),
            'robot_pos': self.robot_pos.tolist(),
            'ball_pos': self.ball_pos.tolist(),
            'robot_angle': float(self.robot_angle),
            'robot_speed': float(np.linalg.norm(self.robot_vel)),
            'collision': collision,
            'collision_count': self.collision_count,
            'time': float(self.time)
        }

class ReversedActivationELM:
    """活性化関数位置逆転ELM（ブログアルゴリズム風）"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 重みの初期化
        self.input_weights = np.random.randn(input_size, hidden_size) * 0.1
        self.output_weights = np.random.randn(hidden_size, output_size) * 0.1
        
        # 学習用の履歴
        self.prev_activated_input = None
        self.prev_activated_hidden = None
        self.learning_history = []
    
    def activation(self, x: np.ndarray) -> np.ndarray:
        """活性化関数（tanh）"""
        return np.tanh(x)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """活性化関数位置逆転での順伝播"""
        # 1. 入力に活性化関数を先に適用
        activated_input = self.activation(x)
        
        # 2. 活性化済み入力で隠れ層計算
        hidden_output = np.dot(activated_input, self.input_weights)
        
        # 3. 隠れ層出力に再度活性化関数
        activated_hidden = self.activation(hidden_output)
        
        # 4. 出力層
        output = np.dot(activated_hidden, self.output_weights)
        
        # アクション制限
        output[0] = np.clip(output[0], 0, 1)      # 前進速度 0-1
        output[1] = np.clip(output[1], -1, 1)    # 角速度 -1 to 1
        
        # 次回の学習用に保存
        self.prev_activated_input = activated_input.copy()
        self.prev_activated_hidden = activated_hidden.copy()
        
        return output
    
    def update_weights(self, success_score: float, sensor_data: np.ndarray, 
                      learning_rate: float = 0.01):
        """ブログアルゴリズム風の重み更新"""
        if self.prev_activated_input is None or self.prev_activated_hidden is None:
            return
        
        # エラー率の計算
        error_rate = 1.0 - success_score
        
        # 学習率の適応的調整
        adaptive_lr = learning_rate * (0.5 + 0.5 * success_score)
        
        # 入力層重み更新
        if success_score > 0.5:
            # 成功時：現在の方向を強化
            weight_delta = adaptive_lr * self.prev_activated_input.reshape(-1, 1)
            self.input_weights += weight_delta * np.random.randn(1, self.hidden_size) * 0.1
        else:
            # 失敗時：探索的な調整
            weight_delta = adaptive_lr * error_rate * self.prev_activated_input.reshape(-1, 1)
            self.input_weights -= weight_delta * np.random.randn(1, self.hidden_size) * 0.2
        
        # 出力層重み更新
        if success_score > 0.5:
            weight_delta = adaptive_lr * self.prev_activated_hidden.reshape(-1, 1)
            self.output_weights += weight_delta * np.random.randn(1, self.output_size) * 0.1
        else:
            weight_delta = adaptive_lr * error_rate * self.prev_activated_hidden.reshape(-1, 1)
            self.output_weights -= weight_delta * np.random.randn(1, self.output_size) * 0.2
        
        # 重みの正規化（発散防止）
        self.input_weights = np.clip(self.input_weights, -2.0, 2.0)
        self.output_weights = np.clip(self.output_weights, -2.0, 2.0)
        
        # 学習履歴記録
        self.learning_history.append({
            'success_score': success_score,
            'error_rate': error_rate,
            'learning_rate': adaptive_lr
        })

class TraditionalELM:
    """従来のELM（比較用）"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 隠れ層の重みは固定
        self.hidden_weights = np.random.randn(input_size, hidden_size)
        self.hidden_bias = np.random.randn(hidden_size)
        
        # 出力層の重みのみ学習
        self.output_weights = np.random.randn(hidden_size, output_size) * 0.1
        
        self.learning_history = []
    
    def activation(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """従来ELMの順伝播"""
        # 隠れ層
        hidden_input = np.dot(x, self.hidden_weights) + self.hidden_bias
        hidden_output = self.activation(hidden_input)
        
        # 出力層
        output = np.dot(hidden_output, self.output_weights)
        
        # アクション制限
        output[0] = np.clip(output[0], 0, 1)
        output[1] = np.clip(output[1], -1, 1)
        
        return output, hidden_output
    
    def update_weights(self, success_score: float, sensor_data: np.ndarray, 
                      learning_rate: float = 0.01):
        """出力層重みのみ更新"""
        _, hidden_output = self.forward(sensor_data)
        
        # 成功度に基づく重み更新
        error_rate = 1.0 - success_score
        adaptive_lr = learning_rate * (0.5 + 0.5 * success_score)
        
        if success_score > 0.5:
            weight_delta = adaptive_lr * hidden_output.reshape(-1, 1)
            self.output_weights += weight_delta * np.random.randn(1, self.output_size) * 0.1
        else:
            weight_delta = adaptive_lr * error_rate * hidden_output.reshape(-1, 1)
            self.output_weights -= weight_delta * np.random.randn(1, self.output_size) * 0.2
        
        # 重みの正規化
        self.output_weights = np.clip(self.output_weights, -2.0, 2.0)
        
        self.learning_history.append({
            'success_score': success_score,
            'error_rate': error_rate,
            'learning_rate': adaptive_lr
        })

class LLMEvaluator:
    """LLM評価器（OpenAI API使用）"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        if OPENAI_AVAILABLE and api_key:
            self.client = openai.OpenAI(api_key=api_key)
            self.use_llm = True
        else:
            self.use_llm = False
            print("LLM評価器：ダミーモードで動作")
    
    def evaluate_performance(self, state_info: Dict, action: np.ndarray, 
                           task_description: str) -> Dict:
        """性能評価"""
        if self.use_llm:
            return self._llm_evaluation(state_info, action, task_description)
        else:
            return self._dummy_evaluation(state_info, action)
    
    def _llm_evaluation(self, state_info: Dict, action: np.ndarray, 
                       task_description: str) -> Dict:
        """実際のLLM評価"""
        prompt = f"""
あなたはロボット制御の専門家です。以下の状況でロボットの行動を評価してください。

タスク: {task_description}

現在の状況:
- ロボット位置: {state_info['robot_pos']}
- ボール位置: {state_info['ball_pos']}
- ボールとの距離: {state_info['ball_distance']:.2f}
- ロボット角度: {state_info['robot_angle']:.2f}
- ロボット速度: {state_info['robot_speed']:.2f}
- 障害物との衝突: {state_info['collision']}
- 累積衝突回数: {state_info['collision_count']}
- 経過時間: {state_info['time']:.1f}秒

ロボットの行動:
- 前進速度: {action[0]:.2f}
- 回転速度: {action[1]:.2f}

以下の基準で0.0-1.0の成功度を評価してください:
1. ボールに近づいているか（距離2.0以下で高評価）
2. 障害物を避けているか（衝突は大幅減点）
3. 効率的な動きをしているか（適度な速度と回転）
4. 時間効率（短時間での達成を評価）

JSON形式で回答してください:
{{
    "success_score": 0.0-1.0の数値,
    "reasoning": "評価理由",
    "suggestions": "改善提案"
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"LLM評価エラー: {e}")
            return self._dummy_evaluation(state_info, action)
    
    def _dummy_evaluation(self, state_info: Dict, action: np.ndarray) -> Dict:
        """ダミー評価（LLM不使用時）"""
        score = 0.5  # 基準点
        reasoning_parts = []
        
        # 距離による評価
        distance = state_info['ball_distance']
        if distance < 1.0:
            score += 0.4
            reasoning_parts.append("ボールに非常に近い")
        elif distance < 2.0:
            score += 0.3
            reasoning_parts.append("ボールに近い")
        elif distance < 4.0:
            score += 0.1
            reasoning_parts.append("ボールにやや近い")
        else:
            score -= 0.1
            reasoning_parts.append("ボールから遠い")
        
        # 衝突ペナルティ
        if state_info['collision']:
            score -= 0.5
            reasoning_parts.append("障害物に衝突")
        
        # 累積衝突ペナルティ
        if state_info['collision_count'] > 0:
            score -= 0.1 * state_info['collision_count']
            reasoning_parts.append(f"累積衝突{state_info['collision_count']}回")
        
        # 動作の妥当性
        if 0.1 <= action[0] <= 0.8 and -0.5 <= action[1] <= 0.5:
            score += 0.1
            reasoning_parts.append("適切な動作")
        
        # 時間効率
        if state_info['time'] > 20.0:
            score -= 0.1
            reasoning_parts.append("時間がかかりすぎ")
        
        score = np.clip(score, 0.0, 1.0)
        
        return {
            "success_score": score,
            "reasoning": "; ".join(reasoning_parts),
            "suggestions": "ボールに向かって効率的に移動し、障害物を避けてください"
        }

class LLMELMHybridSystem:
    """LLM-ELMハイブリッド学習システム"""
    
    def __init__(self, api_key: Optional[str] = None, elm_type: str = 'reversed'):
        self.sim = RobotSimulation()
        self.llm_evaluator = LLMEvaluator(api_key)
        
        # ELMモデル選択
        sensor_size = 12  # センサーデータサイズ
        if elm_type == 'reversed':
            self.elm = ReversedActivationELM(sensor_size, 20, 2)
            self.elm_type = 'Reversed Activation ELM'
        else:
            self.elm = TraditionalELM(sensor_size, 20, 2)
            self.elm_type = 'Traditional ELM'
        
        self.task_description = "ボールに近づきながら障害物を避ける"
        self.performance_history = []
        self.episode_details = []
    
    def run_episode(self, max_steps: int = 200, evaluate_interval: int = 10, 
                   verbose: bool = True) -> Dict:
        """1エピソードの実行"""
        sensor_data = self.sim.reset()
        episode_scores = []
        episode_actions = []
        episode_states = []
        
        start_time = time.time()
        
        for step in range(max_steps):
            # ELMで行動決定
            if hasattr(self.elm, 'forward'):
                if isinstance(self.elm, TraditionalELM):
                    action, _ = self.elm.forward(sensor_data)
                else:
                    action = self.elm.forward(sensor_data)
            else:
                action = np.array([0.5, 0.0])  # デフォルト行動
            
            episode_actions.append(action.copy())
            
            # 環境更新
            next_sensor_data = self.sim.step(action)
            state_info = self.sim.get_state_info()
            episode_states.append(state_info.copy())
            
            # 定期的にLLM評価
            if step % evaluate_interval == 0:
                evaluation = self.llm_evaluator.evaluate_performance(
                    state_info, action, self.task_description
                )
                
                success_score = evaluation['success_score']
                episode_scores.append(success_score)
                
                # ELM学習
                self.elm.update_weights(success_score, sensor_data)
                
                if verbose:
                    print(f"  Step {step:3d}: Score={success_score:.3f}, "
                          f"Distance={state_info['ball_distance']:.2f}, "
                          f"Collision={state_info['collision']}")
            
            sensor_data = next_sensor_data
            
            # 終了条件
            if state_info['ball_distance'] < 0.8:
                if verbose:
                    print("  ボールに到達！")
                break
            if state_info['collision_count'] > 5:
                if verbose:
                    print("  衝突回数が多すぎます...")
                break
        
        execution_time = time.time() - start_time
        avg_score = np.mean(episode_scores) if episode_scores else 0.0
        final_distance = state_info['ball_distance']
        
        episode_result = {
            'avg_score': avg_score,
            'final_distance': final_distance,
            'collision_count': state_info['collision_count'],
            'steps': step + 1,
            'execution_time': execution_time,
            'scores': episode_scores,
            'actions': episode_actions,
            'states': episode_states
        }
        
        self.performance_history.append(avg_score)
        self.episode_details.append(episode_result)
        
        return episode_result
    
    def train(self, episodes: int = 30, verbose: bool = True) -> None:
        """複数エピソードでの学習"""
        if verbose:
            print(f"=== {self.elm_type} ハイブリッド学習開始 ===")
            print(f"エピソード数: {episodes}")
            print(f"タスク: {self.task_description}")
        
        for episode in range(episodes):
            if verbose:
                print(f"\n--- エピソード {episode+1:2d}/{episodes} ---")
            
            result = self.run_episode(verbose=verbose)
            
            if verbose:
                print(f"  結果: Score={result['avg_score']:.3f}, "
                      f"Distance={result['final_distance']:.2f}, "
                      f"Collisions={result['collision_count']}, "
                      f"Steps={result['steps']}")
        
        if verbose:
            print(f"\n=== 学習完了 ===")
            print(f"最終平均スコア: {np.mean(self.performance_history[-5:]):.3f}")
            print(f"最高スコア: {max(self.performance_history):.3f}")
    
    def plot_learning_curve(self, save_path: Optional[str] = None) -> None:
        """学習曲線の表示"""
        plt.figure(figsize=(12, 8))
        
        # メインプロット
        plt.subplot(2, 2, 1)
        plt.plot(self.performance_history, 'b-', linewidth=2, alpha=0.7)
        
        # 移動平均
        if len(self.performance_history) > 5:
            window = min(5, len(self.performance_history) // 4)
            moving_avg = np.convolve(self.performance_history, 
                                   np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(self.performance_history)), 
                    moving_avg, 'r-', linewidth=2, label=f'Moving Avg (window={window})')
            plt.legend()
        
        plt.title(f'{self.elm_type} Learning Progress')
        plt.xlabel('Episode')
        plt.ylabel('Average Success Score')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # 最終距離の推移
        plt.subplot(2, 2, 2)
        final_distances = [ep['final_distance'] for ep in self.episode_details]
        plt.plot(final_distances, 'g-', linewidth=2)
        plt.title('Final Distance to Ball')
        plt.xlabel('Episode')
        plt.ylabel('Distance')
        plt.grid(True, alpha=0.3)
        
        # 衝突回数の推移
        plt.subplot(2, 2, 3)
        collision_counts = [ep['collision_count'] for ep in self.episode_details]
        plt.plot(collision_counts, 'r-', linewidth=2)
        plt.title('Collision Count per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Collisions')
        plt.grid(True, alpha=0.3)
        
        # ステップ数の推移
        plt.subplot(2, 2, 4)
        step_counts = [ep['steps'] for ep in self.episode_details]
        plt.plot(step_counts, 'm-', linewidth=2)
        plt.title('Steps per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"学習曲線を保存: {save_path}")
        
        plt.show()
    
    def demonstrate(self, steps: int = 150, visualize: bool = True) -> Dict:
        """学習後のデモンストレーション"""
        print(f"\n=== {self.elm_type} デモンストレーション ===")
        
        sensor_data = self.sim.reset()
        
        # 軌跡記録用
        robot_positions = []
        ball_positions = []
        actions_taken = []
        
        for step in range(steps):
            # 行動決定
            if isinstance(self.elm, TraditionalELM):
                action, _ = self.elm.forward(sensor_data)
            else:
                action = self.elm.forward(sensor_data)
            
            actions_taken.append(action.copy())
            
            # 環境更新
            sensor_data = self.sim.step(action)
            state_info = self.sim.get_state_info()
            
            # 位置記録
            robot_positions.append(state_info['robot_pos'].copy())
            ball_positions.append(state_info['ball_pos'].copy())
            
            if step % 20 == 0:
                print(f"  Step {step:3d}: Distance={state_info['ball_distance']:.2f}, "
                      f"Collisions={state_info['collision_count']}")
            
            if state_info['ball_distance'] < 0.8:
                print(f"  ボールに到達！ (Step {step})")
                break
        
        final_state = self.sim.get_state_info()
        demo_result = {
            'final_distance': final_state['ball_distance'],
            'collision_count': final_state['collision_count'],
            'steps': step + 1,
            'robot_positions': robot_positions,
            'ball_positions': ball_positions,
            'actions': actions_taken
        }
        
        print(f"デモ結果: Distance={demo_result['final_distance']:.2f}, "
              f"Collisions={demo_result['collision_count']}, "
              f"Steps={demo_result['steps']}")
        
        if visualize:
            self.plot_trajectory(demo_result)
        
        return demo_result
    
    def plot_trajectory(self, demo_result: Dict, save_path: Optional[str] = None) -> None:
        """軌跡の可視化"""
        robot_positions = np.array(demo_result['robot_positions'])
        ball_positions = np.array(demo_result['ball_positions'])
        
        plt.figure(figsize=(12, 10))
        
        # ロボットの軌跡
        plt.plot(robot_positions[:, 0], robot_positions[:, 1], 
                'b-', linewidth=3, alpha=0.7, label='Robot Path')
        plt.plot(robot_positions[0, 0], robot_positions[0, 1], 
                'go', markersize=12, label='Robot Start')
        plt.plot(robot_positions[-1, 0], robot_positions[-1, 1], 
                'ro', markersize=12, label='Robot End')
        
        # ボールの軌跡
        plt.plot(ball_positions[:, 0], ball_positions[:, 1], 
                'orange', linestyle='--', linewidth=2, alpha=0.8, label='Ball Path')
        plt.plot(ball_positions[0, 0], ball_positions[0, 1], 
                'y^', markersize=10, label='Ball Start')
        plt.plot(ball_positions[-1, 0], ball_positions[-1, 1], 
                'r^', markersize=10, label='Ball End')
        
        # 障害物
        for i, obstacle in enumerate(self.sim.obstacles):
            circle = plt.Circle(obstacle, 0.6, color='gray', alpha=0.7)
            plt.gca().add_patch(circle)
            if i == 0:
                plt.plot([], [], 'o', color='gray', alpha=0.7, 
                        markersize=10, label='Obstacles')
        
        plt.xlim(0, self.sim.width)
        plt.ylim(0, self.sim.height)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f'{self.elm_type} - Robot Trajectory After Training\\n'
                 f'Final Distance: {demo_result["final_distance"]:.2f}, '
                 f'Collisions: {demo_result["collision_count"]}, '
                 f'Steps: {demo_result["steps"]}')
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"軌跡図を保存: {save_path}")
        
        plt.show()

def compare_elm_types(api_key: Optional[str] = None, episodes: int = 20) -> None:
    """従来ELMと活性化関数逆転ELMの比較実験"""
    print("=" * 60)
    print("ELM手法比較実験")
    print("=" * 60)
    
    # 従来ELM
    print("\\n1. 従来ELM学習")
    system1 = LLMELMHybridSystem(api_key, elm_type='traditional')
    system1.train(episodes=episodes, verbose=True)
    
    # 活性化関数逆転ELM
    print("\\n2. 活性化関数逆転ELM学習")
    system2 = LLMELMHybridSystem(api_key, elm_type='reversed')
    system2.train(episodes=episodes, verbose=True)
    
    # 結果比較
    plt.figure(figsize=(15, 10))
    
    # 学習曲線比較
    plt.subplot(2, 3, 1)
    plt.plot(system1.performance_history, 'r-', label='Traditional ELM', linewidth=2)
    plt.plot(system2.performance_history, 'b-', label='Reversed Activation ELM', linewidth=2)
    plt.title('Learning Performance Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Average Success Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 最終距離比較
    plt.subplot(2, 3, 2)
    distances1 = [ep['final_distance'] for ep in system1.episode_details]
    distances2 = [ep['final_distance'] for ep in system2.episode_details]
    plt.plot(distances1, 'r-', label='Traditional ELM', linewidth=2)
    plt.plot(distances2, 'b-', label='Reversed Activation ELM', linewidth=2)
    plt.title('Final Distance Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Distance to Ball')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 衝突回数比較
    plt.subplot(2, 3, 3)
    collisions1 = [ep['collision_count'] for ep in system1.episode_details]
    collisions2 = [ep['collision_count'] for ep in system2.episode_details]
    plt.plot(collisions1, 'r-', label='Traditional ELM', linewidth=2)
    plt.plot(collisions2, 'b-', label='Reversed Activation ELM', linewidth=2)
    plt.title('Collision Count Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Collisions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 統計比較
    plt.subplot(2, 3, 4)
    methods = ['Traditional\\nELM', 'Reversed\\nActivation ELM']
    avg_scores = [np.mean(system1.performance_history[-5:]), 
                  np.mean(system2.performance_history[-5:])]
    colors = ['red', 'blue']
    bars = plt.bar(methods, avg_scores, color=colors, alpha=0.7)
    plt.title('Average Final Performance')
    plt.ylabel('Success Score')
    plt.ylim(0, 1)
    
    # 数値表示
    for bar, score in zip(bars, avg_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 最終距離統計
    plt.subplot(2, 3, 5)
    avg_distances = [np.mean(distances1[-5:]), np.mean(distances2[-5:])]
    bars = plt.bar(methods, avg_distances, color=colors, alpha=0.7)
    plt.title('Average Final Distance')
    plt.ylabel('Distance to Ball')
    
    for bar, dist in zip(bars, avg_distances):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{dist:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 衝突統計
    plt.subplot(2, 3, 6)
    avg_collisions = [np.mean(collisions1[-5:]), np.mean(collisions2[-5:])]
    bars = plt.bar(methods, avg_collisions, color=colors, alpha=0.7)
    plt.title('Average Collision Count')
    plt.ylabel('Collisions')
    
    for bar, coll in zip(bars, avg_collisions):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{coll:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # デモンストレーション
    print("\\n3. デモンストレーション比較")
    demo1 = system1.demonstrate(visualize=False)
    demo2 = system2.demonstrate(visualize=False)
    
    # 結果サマリー
    print("\\n" + "=" * 60)
    print("実験結果サマリー")
    print("=" * 60)
    print(f"従来ELM:")
    print(f"  平均スコア: {np.mean(system1.performance_history[-5:]):.3f}")
    print(f"  最終距離: {demo1['final_distance']:.2f}")
    print(f"  衝突回数: {demo1['collision_count']}")
    print(f"  完了ステップ: {demo1['steps']}")
    
    print(f"\\n活性化関数逆転ELM:")
    print(f"  平均スコア: {np.mean(system2.performance_history[-5:]):.3f}")
    print(f"  最終距離: {demo2['final_distance']:.2f}")
    print(f"  衝突回数: {demo2['collision_count']}")
    print(f"  完了ステップ: {demo2['steps']}")
    
    return system1, system2

def main():
    """メイン実行関数"""
    print("LLM-ELMハイブリッドシステム実装")
    print("=" * 50)
    
    # OpenAI APIキーの設定（環境変数から取得）
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("注意: OPENAI_API_KEYが設定されていません。ダミー評価器を使用します。")
    
    # 基本的な使用例
    print("\\n1. 基本的な学習例（活性化関数逆転ELM）")
    system = LLMELMHybridSystem(api_key, elm_type='reversed')
    system.train(episodes=15)
    system.plot_learning_curve()
    system.demonstrate()
    
    # 比較実験
    print("\\n2. ELM手法比較実験")
    compare_elm_types(api_key, episodes=15)

if __name__ == "__main__":
    main()
