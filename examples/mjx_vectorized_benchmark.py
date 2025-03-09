#!/usr/bin/env python3
"""
Benchmark demonstrating MJX's true parallelism advantage using vectorized operations.
This shows how MJX can leverage JAX's vectorization for truly parallel simulation on GPU.
"""

import os
import time
import argparse
import numpy as np
import gymnasium as gym
import myosuite
from myosuite.physics.sim_scene import SimBackend
import jax
import jax.numpy as jnp
from tqdm import tqdm

def benchmark_mjx_vectorized(env_name, n_envs, n_steps):
    """
    Run a vectorized benchmark with MJX leveraging JAX for true parallelism.
    
    This approach uses JAX's vectorization to run multiple environments
    in parallel on the GPU, which should show significant speedup over
    the sequential execution in standard MUJOCO.
    """
    # Set backend to MJX
    os.environ["sim_backend"] = "MJX"
    
    # Create one environment to extract model information
    print("Creating base environment...")
    env = gym.make(env_name)
    
    # Extract model and create basic MJX objects
    from myosuite.physics.mjx_sim_scene import MjxSimScene
    import mujoco
    import mujoco.mjx
    
    # For MyoSuite environments, try to access the sim object
    try:
        # Try to access the simulation scene from MyoSuite environment
        if hasattr(env.unwrapped, 'sim'):
            # Get the model from the sim object
            base_model = env.unwrapped.sim.model
            
            # Create a temporary XML file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as f:
                model_path = f.name
                mujoco.mj_saveLastXML(model_path, base_model)
            print(f"Created temporary model XML from sim.model at: {model_path}")
        else:
            # Try other common attributes
            for attr in ['model_path', 'model', 'physics_model']:
                if hasattr(env.unwrapped, attr):
                    if attr == 'model_path':
                        model_path = getattr(env.unwrapped, attr)
                        print(f"Found model_path: {model_path}")
                        break
                    else:
                        base_model = getattr(env.unwrapped, attr)
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as f:
                            model_path = f.name
                            mujoco.mj_saveLastXML(model_path, base_model)
                        print(f"Created temporary model XML from {attr} at: {model_path}")
                        break
            else:
                raise AttributeError("Could not find model or model_path in environment")
    except Exception as e:
        print(f"Error accessing model: {e}")
        print("Trying to inspect environment structure...")
        print(f"Environment type: {type(env.unwrapped)}")
        print(f"Available attributes: {dir(env.unwrapped)}")
        return None
    
    print(f"Using model: {model_path}")
    
    # Get action space dimension
    action_dim = env.action_space.shape[0]
    
    # Create the base model
    base_model = mujoco.MjModel.from_xml_path(model_path)
    base_data = mujoco.MjData(base_model)
    
    # Make MJX versions of model and data
    print("Setting up MJX models...")
    try:
        # Try different API patterns based on MJX version
        if hasattr(mujoco.mjx, 'device_put'):
            mjx_model = mujoco.mjx.device_put(base_model)
            mjx_data = mujoco.mjx.device_put(base_data)
        elif hasattr(mujoco.mjx, 'put_model'):
            mjx_model = mujoco.mjx.put_model(base_model)
            mjx_data = mujoco.mjx.put_data(base_model, base_data)
    except Exception as e:
        print(f"Error setting up MJX: {e}")
        return None
    
    # Check JAX device
    print(f"JAX is using: {jax.default_backend()} with {len(jax.devices())} devices")
    for i, dev in enumerate(jax.devices()):
        print(f"  Device {i}: {dev}")
    
    # Create vectorized version of step function
    print("Creating vectorized step function...")
    
    @jax.jit
    def step_n_envs(models, datas, actions):
        """Step multiple environments in parallel using JAX vectorization"""
        if hasattr(mujoco.mjx, 'step'):
            # Get the step function 
            step_fn = mujoco.mjx.step
            
            # Use vmap to vectorize over states and actions
            batch_step = jax.vmap(step_fn, in_axes=(None, 0, 0))
            
            # Run one step
            new_datas = batch_step(models, datas, actions)
            return new_datas
        else:
            # Fallback using manual vectorization if needed
            return jax.vmap(lambda d, a: mujoco.mjx.step(models, d, a))(datas, actions)
    
    # Create batch of environments
    print(f"Creating batch of {n_envs} environments...")
    start_time = time.time()
    
    # Vectorize the initialization - create n identical MJX datas
    batch_data = jax.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), n_envs, axis=0), mjx_data)
    
    create_time = time.time() - start_time
    print(f"Batch creation time: {create_time:.4f} seconds")
    
    # Run benchmark
    print(f"Running {n_steps} steps with {n_envs} environments...")
    step_times = []
    
    # Run benchmark for n_steps
    for step in tqdm(range(n_steps)):
        # Generate random actions 
        actions = np.random.uniform(-1, 1, size=(n_envs, action_dim))
        
        # Convert to JAX arrays
        jax_actions = jnp.array(actions)
        
        # Time the step
        start_time = time.time()
        
        # Step all environments at once using JAX vectorization
        batch_data = step_n_envs(mjx_model, batch_data, jax_actions)
        
        # Explicitly synchronize to measure true execution time
        _ = jax.device_get(batch_data)
        
        step_time = time.time() - start_time
        step_times.append(step_time)
    
    # Calculate results
    total_time = sum(step_times)
    avg_step_time = np.mean(step_times)
    steps_per_second = n_steps * n_envs / total_time
    
    result = {
        "backend": "MJX-Vectorized",
        "num_envs": n_envs,
        "total_time": total_time,
        "create_time": create_time,
        "avg_step_time": avg_step_time,
        "steps_per_second": steps_per_second,
    }
    
    # Close the base environment
    env.close()
    
    return result

def benchmark_mujoco_sequential(env_name, n_envs, n_steps):
    """Run a sequential benchmark with standard MUJOCO for comparison"""
    # Set backend to MUJOCO
    os.environ["sim_backend"] = "MUJOCO"
    
    # Create environments
    print(f"Creating {n_envs} sequential MUJOCO environments...")
    start_time = time.time()
    envs = [gym.make(env_name) for _ in range(n_envs)]
    create_time = time.time() - start_time
    
    # Reset environments
    obs = [env.reset()[0] for env in envs]
    
    # Run benchmark
    print(f"Running {n_steps} steps with {n_envs} environments sequentially...")
    step_times = []
    
    # Run benchmark for n_steps
    for step in tqdm(range(n_steps)):
        start_time = time.time()
        
        # Step each environment sequentially
        for i, env in enumerate(envs):
            action = env.action_space.sample()
            obs[i], _, _, _, _ = env.step(action)
        
        step_time = time.time() - start_time
        step_times.append(step_time)
    
    # Calculate results
    total_time = sum(step_times)
    avg_step_time = np.mean(step_times)
    steps_per_second = n_steps * n_envs / total_time
    
    result = {
        "backend": "MUJOCO-Sequential",
        "num_envs": n_envs,
        "total_time": total_time,
        "create_time": create_time,
        "avg_step_time": avg_step_time,
        "steps_per_second": steps_per_second,
    }
    
    # Close environments
    for env in envs:
        env.close()
    
    return result

def print_results(mjx_result, mujoco_result):
    """Print benchmark results and comparison"""
    print("\n===== Performance Comparison: Vectorized MJX vs Sequential MUJOCO =====")
    print(f"Environment count: {mjx_result['num_envs']}")
    print(f"Steps per environment: {int(mjx_result['total_time'] / mjx_result['avg_step_time'] / mjx_result['num_envs'])}")
    print("-" * 70)
    
    print(f"{'Metric':<20} | {'MJX-Vectorized':<20} | {'MUJOCO-Sequential':<20} | {'Ratio':<10}")
    print("-" * 80)
    
    # Compare creation time
    mjx_create = mjx_result["create_time"]
    mujoco_create = mujoco_result["create_time"]
    create_ratio = mujoco_create / mjx_create if mjx_create > 0 else float('inf')
    print(f"{'Creation time (s)':<20} | {mjx_create:<20.4f} | {mujoco_create:<20.4f} | {create_ratio:<10.2f}x")
    
    # Compare step time
    mjx_step = mjx_result["avg_step_time"]
    mujoco_step = mujoco_result["avg_step_time"]
    step_ratio = mujoco_step / mjx_step if mjx_step > 0 else float('inf')
    print(f"{'Average step time (s)':<20} | {mjx_step:<20.6f} | {mujoco_step:<20.6f} | {step_ratio:<10.2f}x")
    
    # Compare steps per second
    mjx_sps = mjx_result["steps_per_second"]
    mujoco_sps = mujoco_result["steps_per_second"]
    sps_ratio = mjx_sps / mujoco_sps if mujoco_sps > 0 else float('inf')
    print(f"{'Steps per second':<20} | {mjx_sps:<20.2f} | {mujoco_sps:<20.2f} | {sps_ratio:<10.2f}x")
    
    # Compare total time
    mjx_total = mjx_result["total_time"]
    mujoco_total = mujoco_result["total_time"]
    total_ratio = mujoco_total / mjx_total if mjx_total > 0 else float('inf')
    print(f"{'Total time (s)':<20} | {mjx_total:<20.4f} | {mujoco_total:<20.4f} | {total_ratio:<10.2f}x")
    
    print("\n===== Summary =====")
    if sps_ratio > 1:
        print(f"MJX with vectorization is {sps_ratio:.2f}x faster than sequential MUJOCO")
        print(f"This shows the benefit of GPU parallelism for batched environment execution")
    else:
        print(f"MJX with vectorization is {1/sps_ratio:.2f}x slower than sequential MUJOCO")
        print("This could be due to overhead or the environment being too simple to benefit from parallelism")
    
    theoretical_max = mjx_result["num_envs"]
    efficiency = sps_ratio / theoretical_max * 100
    print(f"Parallel efficiency: {efficiency:.2f}% of theoretical maximum ({theoretical_max}x)")

def main():
    parser = argparse.ArgumentParser(description="Benchmark MJX vectorized vs MUJOCO sequential")
    parser.add_argument("--env", default="myoHandPoseFixed-v0", help="Environment to test")
    parser.add_argument("--num-envs", type=int, default=32, help="Number of environments")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps per environment")
    args = parser.parse_args()
    
    # Check if MJX is available
    mjx_available = SimBackend.check_mjx_available()
    if not mjx_available:
        print("ERROR: MJX backend is not available! This benchmark requires MJX.")
        return
    
    # Check for JAX and GPU
    try:
        import jax
        print(f"JAX devices: {jax.devices()}")
        if len(jax.devices()) == 0:
            print("WARNING: No GPU detected for JAX. Vectorized MJX may not show significant speedup.")
    except ImportError:
        print("ERROR: JAX is not available! This benchmark requires JAX.")
        return
    
    # Run MJX vectorized benchmark
    print("\n===== Running vectorized MJX benchmark =====")
    mjx_result = benchmark_mjx_vectorized(args.env, args.num_envs, args.steps)
    
    # Run MUJOCO sequential benchmark
    print("\n===== Running sequential MUJOCO benchmark =====")
    mujoco_result = benchmark_mujoco_sequential(args.env, args.num_envs, args.steps)
    
    # Print comparison
    if mjx_result and mujoco_result:
        print_results(mjx_result, mujoco_result)

if __name__ == "__main__":
    main()