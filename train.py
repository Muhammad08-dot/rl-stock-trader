import ray
from ray.rllib.algorithms.ppo import PPOConfig
from env import AdvancedStockTradingEnv
from ray.tune.registry import register_env
import os

def env_creator(env_config):
    return AdvancedStockTradingEnv(env_config)

def train():
    print("🚀 Initializing Ray Cluster for Distributed RL Training 🚀")
    ray.init(ignore_reinit_error=True)
    
    register_env("AdvancedStockTradingEnv-v0", env_creator)
    
    # Configure the Proximal Policy Optimization (PPO) algorithm
    # Utilizing PyTorch under the hood
    config = (
        PPOConfig()
        .environment("AdvancedStockTradingEnv-v0")
        .framework("torch")
        .rollouts(num_rollout_workers=2) # Distributed rollout workers
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .training(
            gamma=0.99,
            lr=5e-5,
            clip_param=0.2,
            vf_loss_coeff=1.0,
            entropy_coeff=0.01,
        )
    )
    
    print("\n[Ray RLlib] Building PPO Algorithm...")
    algo = config.build()
    
    print("[Ray RLlib] Starting Training Loop...")
    for i in range(5):
        result = algo.train()
        print(f"Iteration {i+1}: Mean Reward = {result['env_runners']['episode_reward_mean']:.2f}")
        
    print("\n[Ray RLlib] Training complete! Saving checkpoint...")
    checkpoint_dir = algo.save("./checkpoints")
    print(f"Checkpoint saved at: {checkpoint_dir}")
    
    ray.shutdown()

if __name__ == "__main__":
    train()
