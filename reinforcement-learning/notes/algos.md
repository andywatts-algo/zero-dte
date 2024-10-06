### Model-free or Model-based?
Model-free.  

### What to learn?
* Policy Optimization
    * Deterministic or stochastic
    * Stable and reliable as directly optimizing for target
    * A2C/A3C and PPO
* Action-value functions (Q-functions)
    * Less stable
    * Substantially more sample efficient
    * DQN and C51
* Combos!
    * DDPG concurrently learns deterministic policy and a Q-function by using each to improve the other,
    * TD3
    * SAC uses stochastic policies, entropy regularization, and other tricks

