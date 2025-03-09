#!/usr/bin/env python3
"""
Example demonstrating MyoSuite working with MuJoCo 3.3.0 and different backends
(standard MuJoCo, MJX, and MuJoCo-py)
"""

import os
import time
import argparse
import gymnasium as gym
import myosuite
from myosuite.physics.sim_scene import SimBackend

def run_example(backend="MUJOCO", device=None, render=True):
    """Run the example with the specified backend and device."""
    # Set backend environment variable
    os.environ["sim_backend"] = backend
    
    # Set device environment variable for MJX if specified
    if device and backend == "MJX":
        os.environ["JAX_PLATFORM_NAME"] = device
        print(f"Setting JAX to use {device.upper()}")
    
    print(f"\nRunning with {backend} backend")
    
    # Check if MJX is available when requested
    if backend == "MJX":
        if not SimBackend.check_mjx_available():
            print("WARNING: MJX backend requested but not available!")
            print("Will fall back to standard MuJoCo.")
        else:
            # Check JAX device configuration
            try:
                import jax
                devices = jax.devices()
                print(f"JAX is using: {jax.default_backend()} with {len(devices)} device(s)")
                for i, dev in enumerate(devices):
                    print(f"  Device {i}: {dev}")
            except ImportError:
                print("Could not check JAX configuration.")
    
    # List available environments
    print("\nAvailable MyoSuite environments:")
    myo_envs = sorted([env for env in gym.envs.registry.keys() if "myo" in env])[:5]
    print("\n".join(myo_envs) + "\n...")
    
    # Create and run a simple MyoSuite environment
    env_name = "myoHandPoseFixed-v0"  # Simple hand environment
    print(f"\nCreating environment: {env_name}")
    
    try:
        start_time = time.time()
        env = gym.make(env_name)
        creation_time = time.time() - start_time
        print(f"Environment created in {creation_time:.4f} seconds")
        
        # Environment information
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Run a simple simulation
        print("\nRunning simulation...")
        reset_start = time.time()
        obs, info = env.reset(seed=42)
        reset_time = time.time() - reset_start
        print(f"Reset time: {reset_time:.4f} seconds")
        
        sim_start = time.time()
        total_reward = 0
        
        step_times = []
        for i in range(100):  # Run for 100 timesteps
            step_start = time.time()
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            step_time = time.time() - step_start
            step_times.append(step_time)
            
            total_reward += reward
            
            if i % 20 == 0:
                print(f"Step {i}, reward: {reward:.4f}, time: {step_time:.6f}s")
            
            if terminated or truncated:
                print("Episode finished early")
                break
            
            if render:
                time.sleep(0.01)  # Slow down for visualization
        
        sim_time = time.time() - sim_start
        avg_step_time = sum(step_times) / len(step_times)
        fps = len(step_times) / sim_time
        
        print(f"\nSimulation completed:")
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Simulation time: {sim_time:.4f}s")
        print(f"  Average step time: {avg_step_time:.6f}s")
        print(f"  Frames per second: {fps:.1f}")
        
        env.close()
        print(f"\nSuccess! MyoSuite is working with {backend} backend.")
        
        return {
            "backend": backend,
            "creation_time": creation_time,
            "reset_time": reset_time,
            "sim_time": sim_time,
            "avg_step_time": avg_step_time,
            "fps": fps
        }
    
    except Exception as e:
        print(f"\nError running with {backend} backend: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run MyoSuite with different backends")
    parser.add_argument("--backend", choices=["MUJOCO", "MJX", "MUJOCO_PY"], default="MUJOCO",
                        help="Backend to use")
    parser.add_argument("--device", choices=["cpu", "gpu"], help="Device to use with MJX")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--compare-all", action="store_true", help="Compare all available backends")
    args = parser.parse_args()
    
    if args.compare_all:
        # Try all available backends
        results = {}
        
        # Check which backends are available
        backends = ["MUJOCO"]  # MuJoCo is always available
        
        # Check MJX
        if SimBackend.check_mjx_available():
            backends.append("MJX")
        
        # Check MuJoCo-py
        try:
            import mujoco_py
            backends.append("MUJOCO_PY")
        except ImportError:
            pass
        
        print(f"Testing with backends: {backends}")
        
        for backend in backends:
            result = run_example(backend, args.device, not args.no_render)
            if result:
                results[backend] = result
        
        # Print comparison
        if results:
            print("\n===== Backend Comparison =====")
            print(f"{'Backend':<10} | {'Creation (s)':<12} | {'Reset (s)':<10} | {'Step (s)':<10} | {'FPS':<10}")
            print("-" * 60)
            
            for backend, data in results.items():
                print(f"{backend:<10} | {data['creation_time']:<12.4f} | {data['reset_time']:<10.4f} | {data['avg_step_time']:<10.6f} | {data['fps']:<10.1f}")
        
    else:
        # Run with selected backend
        run_example(args.backend, args.device, not args.no_render)
    
    print("\nTo control backends in your code:")
    print("  os.environ['sim_backend'] = 'MUJOCO'  # or 'MJX' or 'MUJOCO_PY'")
    print("\nTo control MJX device:")
    print("  os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # or 'gpu'")

if __name__ == "__main__":
    main()
