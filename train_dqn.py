#!/usr/bin/env python3
"""
Einfaches DQN-Training mit RLlib.
- 1 Worker (CPU only)
- 50 Episoden
- Nach jeder Episode Reward ausgeben
"""

import ray
from ray.rllib.algorithms.dqn import DQN
from src.trading_env import TradingEnv

def main():
    # Ray starten
    ray.init(ignore_reinit_error=True)

    # Alte Config als Dictionary
    config = {
        "env": TradingEnv,
        "framework": "torch",
        "num_workers": 0,
        "train_batch_size": 32,
        "buffer_size": 50000,
        "learning_starts": 1000,
        "target_network_update_freq": 500,
        "model": {
            "fcnet_hiddens": [256, 256],
        },
    }

    # Algorithmus bauen
    algo = DQN(config=config)

    # 50 Episoden trainieren
    for episode in range(50):
        result = algo.train()
        reward_mean = result.get('episode_reward_mean', 0.0)
        print(f"Episode {episode+1}/50 - Reward: {reward_mean:.2f}")

    # Checkpoint speichern
    checkpoint = algo.save('./checkpoints')
    print(f"Fertig. Checkpoint: {checkpoint}")

    ray.shutdown()

if __name__ == '__main__':
    main()