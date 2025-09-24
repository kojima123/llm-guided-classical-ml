#!/usr/bin/env python3
"""
ロボットシミュレーションでのLLM教師付き従来ELMテスト
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import time

# 既存の実装をインポート
sys.path.append('/home/ubuntu')
from llm_elm_hybrid_implementation import RobotEnvironment

class TraditionalELMRobot:
    """ロボット制御用の従来ELM"""
    
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 従来ELMの構造：隠れ層は固定
        self.hidden_weights = np.random.randn(input_size, hidden_size) * 0.5
        self.hidden_bias = np.random.randn(hidden_size) * 0.5
        
        # 出力層の重みは学習可能
        self.output_weights = np.random.randn(hidden_size, output_size) * 0.1
        
        # 学習履歴
        self.learning_history = []
        self.performance_history = []
        
    def activation(self, x):
        return np.tanh(x)
    
    def _compute_hidden_output(self, x):
        """隠れ層の出力を計算（従来ELM）"""
        hidden_input = np.dot(x, self.hidden_weights) + self.hidden_bias
        return self.activation(hidden_input)
    
    def forward(self, x):
        """順伝播"""
        hidden_output = self._compute_hidden_output(x)
        
        # 出力層
        output = np.dot(hidden_output, self.output_weights)
        
        # アクション制限
        output[0] = np.clip(output[0], 0, 1)      # 前進速度 0-1
        output[1] = np.clip(output[1], -1, 1)    # 角速度 -1 to 1
        
        return output, hidden_output
    
    def update_weights(self, success_score, sensor_data, learning_rate=0.01):
        """重み更新（従来ELMでは出力層のみ）"""
        
        # 隠れ層出力を取得
        hidden_output = self._compute_hidden_output(sensor_data)
        
        # エラー率の計算
        error_rate = 1.0 - success_score
        
        # 適応的学習率
        adaptive_lr = learning_rate * (1.0 + error_rate)
        
        # 出力重みの更新
        if len(self.learning_history) > 0:
            prev_score = self.learning_history[-1]
            score_change = success_score - prev_score
            
            # スコア改善に基づく重み調整
            if score_change > 0:
                # 改善した場合は現在の方向を強化
                weight_adjustment = adaptive_lr * score_change
            else:
                # 悪化した場合は逆方向に調整
                weight_adjustment = -adaptive_lr * abs(score_change)
            
            # ランダムな重み調整（探索的学習）
            noise = np.random.randn(*self.output_weights.shape) * 0.01
            self.output_weights += weight_adjustment * noise
        
        self.learning_history.append(success_score)

class RobotEvaluator:
    """ロボット制御用の評価器"""
    
    def __init__(self, use_llm=False):
        self.use_llm = use_llm
        self.previous_distance = None
        self.episode_count = 0
        
    def evaluate_performance(self, state_info, action, task_description):
        """性能評価"""
        
        score = 0.5  # 基準点
        reasoning_parts = []
        
        distance = state_info['ball_distance']
        
        # 基本的な距離評価
        if distance < 2.0:
            score += 0.3
            reasoning_parts.append('ボールに近い')
        elif distance < 4.0:
            score += 0.1
            reasoning_parts.append('ボールにやや近い')
        elif distance > 6.0:
            score -= 0.2
            reasoning_parts.append('ボールから遠い')
        
        # 改善度の評価
        if self.previous_distance is not None:
            improvement = self.previous_distance - distance
            if improvement > 0.5:
                score += 0.15
                reasoning_parts.append('大幅に改善')
            elif improvement > 0.1:
                score += 0.1
                reasoning_parts.append('改善あり')
            elif improvement < -0.5:
                score -= 0.15
                reasoning_parts.append('大幅に悪化')
            elif improvement < -0.1:
                score -= 0.1
                reasoning_parts.append('やや悪化')
        
        self.previous_distance = distance
        
        # LLM風の詳細評価
        if self.use_llm:
            # より詳細な行動評価
            forward_speed, angular_speed = action[0], action[1]
            
            if 0.2 <= forward_speed <= 0.8 and abs(angular_speed) <= 0.5:
                score += 0.1
                reasoning_parts.append('適切な動作')
            elif forward_speed < 0.1:
                score -= 0.1
                reasoning_parts.append('動作が消極的')
            elif forward_speed > 0.9:
                score -= 0.05
                reasoning_parts.append('動作が急激')
            
            # 時間効率の評価
            if state_info['time'] > 15.0 and distance > 4.0:
                score -= 0.1
                reasoning_parts.append('時間効率が悪い')
            
            # 距離に応じた詳細評価
            if distance < 1.0:
                score += 0.2
                reasoning_parts.append('ボールに非常に近い（優秀）')
            elif distance < 1.5:
                score += 0.1
                reasoning_parts.append('ボールに非常に近い')
        
        # 衝突の評価
        if state_info['collision']:
            score -= 0.5
            reasoning_parts.append('衝突発生')
        
        score = np.clip(score, 0.0, 1.0)
        
        return {
            'success_score': score,
            'reasoning': '; '.join(reasoning_parts) if reasoning_parts else '標準的な動作',
            'suggestions': 'より効率的にボールに近づき、衝突を避けてください'
        }

class RobotELMSystem:
    """ロボット制御システム"""
    
    def __init__(self, use_llm_teacher=True):
        self.env = RobotEnvironment()
        self.elm_model = TraditionalELMRobot(12, 8, 2)  # 隠れ層を少し大きく
        self.evaluator = RobotEvaluator(use_llm=use_llm_teacher)
        
        # 履歴
        self.performance_history = []
        self.episode_details = []
        
    def run_episode(self, max_steps=100, evaluate_interval=20, verbose=True):
        """エピソード実行"""
        
        self.env.reset()
        
        total_score = 0
        evaluation_count = 0
        collision_count = 0
        
        for step in range(max_steps):
            # センサーデータ取得
            sensor_data = self.env.get_sensor_data()
            
            # ELMで行動決定
            action, _ = self.elm_model.forward(sensor_data)
            
            # 環境更新
            collision = self.env.step(action)
            if collision:
                collision_count += 1
            
            # 定期的な評価
            if step % evaluate_interval == 0:
                state_info = {
                    'ball_distance': self.env.get_distance_to_ball(),
                    'robot_pos': self.env.robot_pos.copy(),
                    'ball_pos': self.env.ball_pos.copy(),
                    'robot_angle': self.env.robot_angle,
                    'robot_speed': np.linalg.norm(action),
                    'collision': collision,
                    'collision_count': collision_count,
                    'time': step * 0.1
                }
                
                # 評価
                evaluation = self.evaluator.evaluate_performance(
                    state_info, action, "ボールに近づく"
                )
                
                total_score += evaluation['success_score']
                evaluation_count += 1
                
                # ELM重み更新
                self.elm_model.update_weights(
                    evaluation['success_score'], 
                    sensor_data
                )
        
        # エピソード結果
        avg_score = total_score / evaluation_count if evaluation_count > 0 else 0
        final_distance = self.env.get_distance_to_ball()
        
        # 履歴に追加
        self.performance_history.append(avg_score)
        self.episode_details.append({
            'avg_score': avg_score,
            'final_distance': final_distance,
            'collision_count': collision_count,
            'steps': max_steps
        })
        
        if verbose:
            print(f"エピソード完了: 平均スコア={avg_score:.3f}, 最終距離={final_distance:.2f}, 衝突={collision_count}")
        
        return {
            'avg_score': avg_score,
            'final_distance': final_distance,
            'collision_count': collision_count
        }

def test_robot_traditional_elm():
    """ロボット制御でのLLM教師付き従来ELMテスト"""
    
    print("=" * 60)
    print("ロボット制御でのLLM教師付き従来ELMテスト")
    print("=" * 60)
    
    results = {}
    
    # 1. 従来ELM（教師なし）
    print("\\n1. 従来ELM（教師なし）")
    print("-" * 30)
    
    system_no_teacher = RobotELMSystem(use_llm_teacher=False)
    
    for i in range(12):
        result = system_no_teacher.run_episode(max_steps=100, evaluate_interval=20, verbose=False)
        if i % 2 == 0:
            print(f"  エピソード {i+1:2d}: スコア={result['avg_score']:.3f}, 距離={result['final_distance']:.2f}, 衝突={result['collision_count']}")
    
    results['traditional_no_teacher'] = {
        'performance_history': system_no_teacher.performance_history.copy(),
        'final_performance': system_no_teacher.performance_history[-1],
        'improvement': system_no_teacher.performance_history[-1] - system_no_teacher.performance_history[0],
        'avg_final_distance': np.mean([ep['final_distance'] for ep in system_no_teacher.episode_details[-3:]]),
        'total_collisions': sum([ep['collision_count'] for ep in system_no_teacher.episode_details])
    }
    
    # 2. LLM教師付き従来ELM
    print("\\n2. LLM教師付き従来ELM")
    print("-" * 30)
    
    system_with_teacher = RobotELMSystem(use_llm_teacher=True)
    
    for i in range(12):
        result = system_with_teacher.run_episode(max_steps=100, evaluate_interval=20, verbose=False)
        if i % 2 == 0:
            print(f"  エピソード {i+1:2d}: スコア={result['avg_score']:.3f}, 距離={result['final_distance']:.2f}, 衝突={result['collision_count']}")
    
    results['traditional_with_teacher'] = {
        'performance_history': system_with_teacher.performance_history.copy(),
        'final_performance': system_with_teacher.performance_history[-1],
        'improvement': system_with_teacher.performance_history[-1] - system_with_teacher.performance_history[0],
        'avg_final_distance': np.mean([ep['final_distance'] for ep in system_with_teacher.episode_details[-3:]]),
        'total_collisions': sum([ep['collision_count'] for ep in system_with_teacher.episode_details])
    }
    
    # 結果比較
    print("\\n" + "=" * 60)
    print("結果比較")
    print("=" * 60)
    
    print(f"{'手法':<20} {'最終性能':<10} {'改善度':<10} {'平均距離':<10} {'総衝突':<8}")
    print("-" * 65)
    print(f"{'従来ELM':<20} {results['traditional_no_teacher']['final_performance']:>8.3f} {results['traditional_no_teacher']['improvement']:>+8.3f} {results['traditional_no_teacher']['avg_final_distance']:>8.2f} {results['traditional_no_teacher']['total_collisions']:>6d}")
    print(f"{'LLM教師付き従来ELM':<20} {results['traditional_with_teacher']['final_performance']:>8.3f} {results['traditional_with_teacher']['improvement']:>+8.3f} {results['traditional_with_teacher']['avg_final_distance']:>8.2f} {results['traditional_with_teacher']['total_collisions']:>6d}")
    
    # 性能差の分析
    performance_diff = results['traditional_with_teacher']['final_performance'] - results['traditional_no_teacher']['final_performance']
    improvement_diff = results['traditional_with_teacher']['improvement'] - results['traditional_no_teacher']['improvement']
    distance_diff = results['traditional_with_teacher']['avg_final_distance'] - results['traditional_no_teacher']['avg_final_distance']
    
    print("\\n詳細分析:")
    print(f"性能差: {performance_diff:+.3f} ({'改善' if performance_diff > 0 else '悪化'})")
    print(f"改善度差: {improvement_diff:+.3f} ({'LLM教師の方が学習効果大' if improvement_diff > 0 else 'LLM教師の学習効果小'})")
    print(f"距離差: {distance_diff:+.2f} ({'LLM教師の方が遠い' if distance_diff > 0 else 'LLM教師の方が近い'})")
    
    return results

def create_robot_comparison_graph(results):
    """ロボット制御結果の比較グラフを作成"""
    
    plt.figure(figsize=(15, 5))
    
    # 性能推移
    plt.subplot(1, 3, 1)
    plt.plot(results['traditional_no_teacher']['performance_history'], 'r-', label='従来ELM', linewidth=2, marker='o', markersize=4)
    plt.plot(results['traditional_with_teacher']['performance_history'], 'b-', label='LLM教師付き従来ELM', linewidth=2, marker='s', markersize=4)
    plt.title('性能の推移')
    plt.xlabel('Episode')
    plt.ylabel('Performance Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 最終性能比較
    plt.subplot(1, 3, 2)
    methods = ['従来ELM', 'LLM教師付き\\n従来ELM']
    performances = [results['traditional_no_teacher']['final_performance'],
                   results['traditional_with_teacher']['final_performance']]
    colors = ['red', 'blue']
    
    bars = plt.bar(methods, performances, color=colors, alpha=0.7)
    plt.title('最終性能比較')
    plt.ylabel('Final Performance')
    
    for bar, perf in zip(bars, performances):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{perf:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 学習改善度比較
    plt.subplot(1, 3, 3)
    improvements = [results['traditional_no_teacher']['improvement'],
                   results['traditional_with_teacher']['improvement']]
    
    bars = plt.bar(methods, improvements, color=colors, alpha=0.7)
    plt.title('学習改善度比較')
    plt.ylabel('Improvement')
    
    for bar, imp in zip(bars, improvements):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{imp:+.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/robot_traditional_elm_comparison.png', dpi=150, bbox_inches='tight')
    print("\\n比較グラフを保存: /home/ubuntu/robot_traditional_elm_comparison.png")

if __name__ == "__main__":
    # テスト実行
    results = test_robot_traditional_elm()
    
    # グラフ作成
    create_robot_comparison_graph(results)
    
    print("\\n" + "=" * 60)
    print("ロボット制御でのLLM教師付き従来ELMテスト完了")
    print("=" * 60)
