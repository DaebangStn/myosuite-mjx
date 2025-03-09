#!/usr/bin/env python3
"""
Benchmark script comparing MJX and MUJOCO backends with parallel environment execution.
MJX should show better scaling with multiple environments due to GPU parallelization.
"""

import os
import time
import argparse
import numpy as np
import gymnasium as gym
import myosuite
from myosuite.physics.sim_scene import SimBackend

class EnvBatch:
    """Handles a batch of environments with the same backend"""
    
    def __init__(self, env_name, backend, num_envs, steps):
        self.env_name = env_name
        self.backend = backend
        self.num_envs = num_envs
        self.steps = steps
        
        # Set the backend
        os.environ["sim_backend"] = backend
        
        # Create environments
        start_time = time.time()
        self.envs = [gym.make(env_name) for _ in range(num_envs)]
        self.create_time = time.time() - start_time
        
        # Reset environments
        start_time = time.time()
        self.obs = [env.reset(seed=i)[0] for i, env in enumerate(self.envs)]
        self.reset_time = time.time() - start_time
        
        self.actions = [None] * num_envs
        self.total_rewards = [0] * num_envs
    
    def step_sequential(self):
        """Run one step in all environments sequentially"""
        start_time = time.time()
        
        for i, env in enumerate(self.envs):
            # Sample random action
            action = env.action_space.sample()
            self.actions[i] = action
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            self.obs[i] = obs
            self.total_rewards[i] += reward
            
            # Reset if needed
            if terminated or truncated:
                obs, _ = env.reset(seed=i)
                self.obs[i] = obs
        
        step_time = time.time() - start_time
        return step_time
    
    def run_benchmark(self):
        """Run the benchmark for all steps"""
        step_times = []
        
        for i in range(self.steps):
            step_time = self.step_sequential()
            step_times.append(step_time)
            
            if (i+1) % 20 == 0:
                print(f"{self.backend} - Step {i+1}/{self.steps}, time: {step_time:.5f}s, avg: {np.mean(step_times):.5f}s")
        
        # Calculate stats
        total_time = sum(step_times)
        avg_step_time = np.mean(step_times)
        steps_per_second = self.steps * self.num_envs / total_time
        
        result = {
            "backend": self.backend,
            "num_envs": self.num_envs,
            "total_time": total_time,
            "create_time": self.create_time,
            "reset_time": self.reset_time,
            "avg_step_time": avg_step_time,
            "steps_per_second": steps_per_second,
            "avg_reward": np.mean(self.total_rewards)
        }
        
        # Close environments
        for env in self.envs:
            env.close()
            
        return result

def run_comparison(env_name, backends, num_envs_list, steps=100):
    """Run comparison across backends and different numbers of environments"""
    results = []
    
    for backend in backends:
        for num_envs in num_envs_list:
            print(f"\n===== Running {backend} with {num_envs} environments =====")
            batch = EnvBatch(env_name, backend, num_envs, steps)
            result = batch.run_benchmark()
            results.append(result)
            
    return results

def print_comparison_table(results):
    """Print a comparison table of the results"""
    print("\n===== Performance Comparison =====")
    
    # Group results by number of environments
    num_envs_groups = {}
    for result in results:
        num_envs = result["num_envs"]
        if num_envs not in num_envs_groups:
            num_envs_groups[num_envs] = []
        num_envs_groups[num_envs].append(result)
    
    # Sort by number of environments
    num_envs_list = sorted(num_envs_groups.keys())
    
    # Print header
    print(f"{'Backend':<10} | {'Num Envs':<10} | {'Total Time (s)':<15} | {'Steps/Second':<15} | {'Avg Step (s)':<15}")
    print("-" * 75)
    
    # Print results by number of environments
    for num_envs in num_envs_list:
        group = num_envs_groups[num_envs]
        group.sort(key=lambda x: x["backend"])
        
        for result in group:
            print(f"{result['backend']:<10} | {result['num_envs']:<10} | {result['total_time']:<15.4f} | {result['steps_per_second']:<15.2f} | {result['avg_step_time']:<15.6f}")
        
        # Add separator between groups
        if num_envs != num_envs_list[-1]:
            print("-" * 75)
    
    # Calculate and print speedup for each group
    print("\n===== MJX Speedup vs MUJOCO =====")
    print(f"{'Num Envs':<10} | {'MJX Steps/Sec':<15} | {'MUJOCO Steps/Sec':<20} | {'Speedup':<10}")
    print("-" * 65)
    
    for num_envs in num_envs_list:
        group = {result["backend"]: result for result in num_envs_groups[num_envs]}
        
        if "MJX" in group and "MUJOCO" in group:
            mjx_speed = group["MJX"]["steps_per_second"]
            mujoco_speed = group["MUJOCO"]["steps_per_second"]
            speedup = mjx_speed / mujoco_speed
            
            print(f"{num_envs:<10} | {mjx_speed:<15.2f} | {mujoco_speed:<20.2f} | {speedup:<10.2f}x")

def main():
    parser = argparse.ArgumentParser(description="Benchmark MJX vs MUJOCO with parallel environments")
    parser.add_argument("--env", default="myoHandPoseFixed-v0", help="Environment to test")
    parser.add_argument("--steps", type=int, default=50, help="Number of steps per environment")
    parser.add_argument("--max-envs", type=int, default=128, help="Maximum number of parallel environments")
    args = parser.parse_args()
    
    # Check if MJX is available
    mjx_available = SimBackend.check_mjx_available()
    if not mjx_available:
        print("WARNING: MJX backend is not available!")
        backends = ["MUJOCO"]
    else:
        backends = ["MUJOCO", "MJX"]
    
    # Generate exponential list of environment counts (1, 2, 4, 8, 16, etc.)
    num_envs_list = [2**i for i in range(8) if 2**i <= args.max_envs]
    
    # Run comparison
    results = run_comparison(args.env, backends, num_envs_list, args.steps)
    
    # Print comparison table
    print_comparison_table(results)

if __name__ == "__main__":
    main()