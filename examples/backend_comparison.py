#!/usr/bin/env python3
"""
Example showing how to switch between MuJoCo backends (MJX vs MujocoPy vs standard MuJoCo)
and comparing their performance.
"""

import time
import argparse
import os
import numpy as np
import gymnasium as gym
import myosuite


def benchmark_env(env_name, backend, num_steps=1000, render=False):
    """Benchmark environment performance with specified backend."""
    # Set the backend environment variable
    os.environ["sim_backend"] = backend

    print(f"\nRunning benchmark with {backend} backend:")

    # Create environment
    start_time = time.time()
    env = gym.make(env_name)
    env_creation_time = time.time() - start_time
    print(f"Environment creation time: {env_creation_time:.4f} seconds")

    # Reset environment
    start_time = time.time()
    obs, info = env.reset(seed=42)
    reset_time = time.time() - start_time
    print(f"Environment reset time: {reset_time:.4f} seconds")

    # Run simulation steps
    total_reward = 0
    start_time = time.time()

    step_times = []
    for i in range(num_steps):
        step_start = time.time()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        step_time = time.time() - step_start
        step_times.append(step_time)

        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished at step {i}")
            break

    sim_time = time.time() - start_time
    avg_step_time = np.mean(step_times)
    fps = num_steps / sim_time if sim_time > 0 else 0

    print(f"Simulation time for {num_steps} steps: {sim_time:.4f} seconds")
    print(f"Average step time: {avg_step_time:.6f} seconds")
    print(f"Frames per second: {fps:.2f}")
    print(f"Total reward: {total_reward:.4f}")

    env.close()
    return {
        "env_creation_time": env_creation_time,
        "reset_time": reset_time,
        "sim_time": sim_time,
        "avg_step_time": avg_step_time,
        "fps": fps,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare MuJoCo backend performance")
    parser.add_argument(
        "--env", default="myoHandPoseFixed-v0", help="Environment to benchmark"
    )
    parser.add_argument(
        "--steps", type=int, default=1000, help="Number of simulation steps"
    )
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["MUJOCO", "MJX"],
        help="Backends to compare (MUJOCO, MJX)",
    )
    args = parser.parse_args()

    # List available environments
    print("\nAvailable MyoSuite environments:")
    myo_envs = sorted([env for env in gym.envs.registry.keys() if "myo" in env])
    print("\n".join(myo_envs))

    print(f"\nSelected environment: {args.env}")

    results = {}

    # Run benchmarks for each backend
    for backend in args.backends:
        try:
            results[backend] = benchmark_env(args.env, backend, args.steps, args.render)
        except Exception as e:
            print(f"Error with {backend} backend: {e}")
            results[backend] = None

    # Print comparison summary
    print("\n===== Performance Comparison =====")
    print(f"Environment: {args.env}")
    print(f"Steps: {args.steps}")
    print(f"Rendering: {'Enabled' if args.render else 'Disabled'}")
    print("-" * 40)

    # Format as a table
    headers = [
        "Backend",
        "Creation Time (s)",
        "Reset Time (s)",
        "Simulation Time (s)",
        "Avg Step Time (s)",
        "FPS",
    ]
    print(" | ".join(headers))
    print("-" * 100)

    for backend, data in results.items():
        if data:
            row = [
                f"{backend:>10}",
                f"{data['env_creation_time']:.4f}",
                f"{data['reset_time']:.4f}",
                f"{data['sim_time']:.4f}",
                f"{data['avg_step_time']:.6f}",
                f"{data['fps']:.2f}",
            ]
            print(" | ".join(row))

    # Find the fastest backend
    fastest = None
    max_fps = 0
    for backend, data in results.items():
        if data and data["fps"] > max_fps:
            max_fps = data["fps"]
            fastest = backend

    if fastest:
        print(f"\nFastest backend: {fastest} with {max_fps:.2f} FPS")


if __name__ == "__main__":
    main()

