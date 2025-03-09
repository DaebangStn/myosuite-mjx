#!/usr/bin/env python3
"""
Example demonstrating how to select a specific MuJoCo backend in MyoSuite
with additional device control for MJX
"""

import os
import time
import argparse
import gymnasium as gym
import myosuite
from myosuite.physics.sim_scene import SimBackend

def run_with_backend(env_name, backend, device=None, steps=100, render=True):
    os.environ["sim_backend"] = backend
    original_platform = os.environ.get('JAX_PLATFORM_NAME', None)
    if device and backend == "MJX":
        os.environ["JAX_PLATFORM_NAME"] = device
        print(f"Setting JAX to use {device.upper()}")
    
    print(f"\nRunning {env_name} with {backend} backend")
    
    if backend == "MJX":
        if not SimBackend.check_mjx_available():
            print("WARNING: MJX backend requested but not available!")
            print("Will fall back to standard MuJoCo.")
        else:
            try:
                import jax
                devices = jax.devices()
                print(f"JAX is using: {jax.default_backend()} with {len(devices)} device(s)")
                for i, dev in enumerate(devices):
                    print(f"  Device {i}: {dev}")
            except ImportError:
                print("Could not check JAX configuration.")
    
    # Create and run environment
    start_create = time.time()
    env = gym.make(env_name)
    create_time = time.time() - start_create
    print(f"Environment creation time: {create_time:.2f} seconds")
    
    start_reset = time.time()
    obs, info = env.reset(seed=42)
    reset_time = time.time() - start_reset
    print(f"Environment reset time: {reset_time:.2f} seconds")
    
    start_time = time.time()
    total_reward = 0
    
    step_times = []
    for i in range(steps):
        step_start = time.time()
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        step_end = time.time()
        step_times.append(step_end - step_start)
        
        total_reward += reward
        
        if i % 20 == 0:
            print(f"Step {i}, reward: {reward:.4f}, total: {total_reward:.4f}, time: {step_times[-1]:.5f}s")
        
        if terminated or truncated:
            print("Episode finished early")
            break
        
        if render:
            time.sleep(0.01)  # Slow down for visualization
    
    elapsed = time.time() - start_time
    avg_step_time = sum(step_times) / len(step_times)
    fps = steps / elapsed
    
    print(f"\nSimulation completed with total reward: {total_reward:.4f}")
    print(f"Time elapsed: {elapsed:.2f} seconds ({fps:.1f} FPS)")
    print(f"Average step time: {avg_step_time:.5f}s")
    
    env.close()
    
    # Restore original JAX platform setting
    if device and backend == "MJX":
        if original_platform:
            os.environ["JAX_PLATFORM_NAME"] = original_platform
        else:
            os.environ.pop("JAX_PLATFORM_NAME", None)
    
    return {
        "total_reward": total_reward,
        "fps": fps,
        "avg_step_time": avg_step_time,
        "create_time": create_time,
        "reset_time": reset_time
    }

def check_available_devices():
    """Check what devices JAX can use"""
    has_cpu = has_gpu = has_tpu = False
    devices = []
    
    try:
        import jax
        devices = jax.devices()
        has_cpu = True  # JAX always has CPU support
        has_gpu = any('cuda' in str(d).lower() for d in devices)  # Check for 'cuda' in device name
        has_tpu = any('tpu' in str(d).lower() for d in devices)
    except ImportError:
        pass
    
    return {
        "cpu": has_cpu,
        "gpu": has_gpu,
        "tpu": has_tpu,
        "devices": devices
    }

def main():
    parser = argparse.ArgumentParser(description="Select a MuJoCo backend for MyoSuite")
    parser.add_argument("--env", default="myoHandPoseFixed-v0", help="Environment to run")
    parser.add_argument("--backend", choices=["MUJOCO", "MJX", "MUJOCO_PY"], default="MUJOCO",
                        help="MuJoCo backend to use")
    parser.add_argument("--device", choices=["cpu", "gpu", "tpu"], help="Device to use with MJX")
    parser.add_argument("--steps", type=int, default=100, help="Number of simulation steps")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--compare-devices", action="store_true", help="Compare CPU vs GPU with MJX")
    args = parser.parse_args()
    
    # Print available backends
    print("Available backends:")
    print("  MUJOCO - Standard MuJoCo (CPU-based)")
    print("  MJX - MuJoCo with GPU acceleration through JAX")
    print("  MUJOCO_PY - Legacy MuJoCo Python bindings")
    
    # Check if MJX is available
    mjx_available = SimBackend.check_mjx_available()
    mujoco_version, is_mjx_compatible = SimBackend.check_mujoco_version()
    
    print(f"\nMuJoCo version: {mujoco_version}")
    print(f"MJX compatibility: {'Yes' if is_mjx_compatible else 'No'}")
    print(f"MJX backend is {'available' if mjx_available else 'not available'}")
    
    # Check JAX devices
    device_info = check_available_devices()
    print("\nJAX device availability:")
    print(f"  CPU: {'Yes' if device_info['cpu'] else 'No'}")
    print(f"  GPU: {'Yes' if device_info['gpu'] else 'No'}")
    print(f"  TPU: {'Yes' if device_info['tpu'] else 'No'}")
    
    if device_info['devices']:
        print("\nDetailed device information:")
        for i, device in enumerate(device_info['devices']):
            print(f"  Device {i}: {device}")
    
    # List available environments
    print("\nAvailable MyoSuite environments:")
    myo_envs = sorted([env for env in gym.envs.registry.keys() if "myo" in env])[:10]  # Show first 10
    print("\n".join(myo_envs) + ("\n..." if len(myo_envs) > 10 else ""))
    
    if args.compare_devices and args.backend == "MJX" and mjx_available:
        # Run with both CPU and GPU for comparison
        if device_info['gpu']:
            print("\n===== Comparing MJX performance: CPU vs GPU =====")
            cpu_results = run_with_backend(args.env, args.backend, "cpu", args.steps, not args.no_render)
            gpu_results = run_with_backend(args.env, args.backend, "gpu", args.steps, not args.no_render)
            
            # Show comparison
            print("\n===== Performance Comparison =====")
            print(f"Environment: {args.env}")
            print(f"Steps: {args.steps}")
            print(f"Rendering: {'Disabled' if args.no_render else 'Enabled'}")
            print("-" * 40)
            
            print(f"{'Device':<10} | {'FPS':<10} | {'Avg Step':<10} | {'Create':<10} | {'Reset':<10}")
            print("-" * 65)
            print(f"{'CPU':<10} | {cpu_results['fps']:<10.1f} | {cpu_results['avg_step_time']:<10.5f} | {cpu_results['create_time']:<10.2f} | {cpu_results['reset_time']:<10.2f}")
            print(f"{'GPU':<10} | {gpu_results['fps']:<10.1f} | {gpu_results['avg_step_time']:<10.5f} | {gpu_results['create_time']:<10.2f} | {gpu_results['reset_time']:<10.2f}")
            
            speedup = gpu_results['fps'] / cpu_results['fps'] if cpu_results['fps'] > 0 else float('inf')
            print(f"\nGPU speedup: {speedup:.2f}x")
        else:
            print("\nGPU not available, can't compare CPU vs GPU performance.")
            run_with_backend(args.env, args.backend, args.device, args.steps, not args.no_render)
    else:
        # Run with selected backend and device
        run_with_backend(args.env, args.backend, args.device, args.steps, not args.no_render)
    
    print("\nUseful commands:")
    print("1. To compare performance between backends:")
    print("   python backend_comparison.py --steps 1000 --env", args.env)
    print("2. To check MJX device configuration:")
    print("   python check_mjx_device.py")
    print("3. To force MJX to use a specific device:")
    print("   export JAX_PLATFORM_NAME=cpu|gpu|tpu  # Before running python")

if __name__ == "__main__":
    main()