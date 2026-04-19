import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnv(gym.Env):
    """
    Einfache Trading-Umgebung.
    - State: 5 zufällige Werte (0-1)
    - Actions: 0=Buy, 1=Sell, 2=Hold
    - Reward: +1 wenn Gewinn, -1 wenn Verlust
    - Position bleibt geöffnet bis Sell
    - Maximale 100 Schritte pro Episode
    """

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.max_steps = 100
        self.steps = 0
        self.position = None
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.position = None
        self.state = self.np_random.uniform(0.0, 1.0, size=(5,)).astype(np.float32)
        return self.state, {}

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        # Erzeuge nächsten zufälligen State
        next_state = self.np_random.uniform(0.0, 1.0, size=(5,)).astype(np.float32)

        # Preis placeholder: wir nehmen einfach den ersten Wert des States als "Preis"
        current_price = self.state[0]

        if action == 0:  # Buy
            if self.position is None:
                self.position = current_price
                # Kein Reward beim Kauf
        elif action == 1:  # Sell
            if self.position is not None:
                # Simulierter Gewinn/Verlust: Differenz zu neuem Preis
                pnl = next_state[0] - self.position
                reward = 1.0 if pnl > 0 else -1.0
                self.position = None
        # action == 2 (Hold) macht nichts

        # Schritt zählen
        self.steps += 1
        if self.steps >= self.max_steps:
            terminated = True
            # Falls noch Position offen, zum Schluss schließen und kleinen Penalty geben
            if self.position is not None:
                final_pnl = next_state[0] - self.position
                reward += 1.0 if final_pnl > 0 else -1.0
                self.position = None

        self.state = next_state
        info = {'steps': self.steps, 'position_open': self.position is not None}

        return self.state, reward, terminated, truncated, info