## Agent and Policy
Try PPO+[Mlp|Categorical]Policy

| Action Space | Observation Space              | RL Agent    | Policy            |
|--------------|--------------------------------|-------------|-------------------|
| Continuous   | Structured (Numerical)         | PPO         | MlpPolicy         |
| Discrete     | Structured (Numerical)         | PPO         | CategoricalPolicy |
| Discrete     | Raw Pixel Data (Images)        | DQN         | CnnPolicy         |
| Continuous   | Structured (Numerical) or Raw Pixel Data (Images) | SAC         | MlpPolicy or CnnPolicy |




