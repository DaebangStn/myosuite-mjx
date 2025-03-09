#!/usr/bin/env python3
"""
Utility to check MJX device configuration (CPU vs GPU)
"""

import os
import time
import sys
import numpy as np
import jax
import gymnasium as gym
import myosuite
from myosuite.physics.sim_scene import SimBackend

def print_device_info():
    """Print detailed information about available JAX devices"""
    print("\n===== JAX Device Information =====")
    
    # Check JAX version
    print(f"JAX version: {jax.__version__}")
    
    # Get available devices
    devices = jax.devices()
    print(f"Available devices: {len(devices)}")
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device}")
    
    # Check default device
    print(f"Default device: {jax.default_backend()}")
    
    # Check if GPU is available
    has_gpu = any(d.device_kind == 'gpu' for d in devices)
    print(f"GPU available: {has_gpu}")
    
    # Check CUDA configuration if available
    try:
        import jaxlib.xla_extension
        print(f"CUDA enabled: {hasattr(jaxlib.xla_extension, 'GpuAllocatorConfig')}")
    except ImportError:
        print("CUDA status: jaxlib.xla_extension not available")
    
    # Check if MJX is available
    try:
        import mujoco.mjx
        print(f"MJX available: True")
        
        # Check MJX version if available
        mjx_version = getattr(mujoco.mjx, '__version__', 'Unknown')
        print(f"MJX version: {mjx_version}")
        
        # Get all available MJX functions to diagnose API
        mjx_functions = [func for func in dir(mujoco.mjx) if not func.startswith('_')]
        print(f"Available MJX functions: {', '.join(sorted(mjx_functions))}")
        
        # Identify API pattern
        if hasattr(mujoco.mjx, 'device_put'):
            print("MJX API: Using device_put/device_get")
        elif hasattr(mujoco.mjx, 'put_model'):
            print("MJX API: Using put_model/put_data/get_data")
            
            # Check for step function signature
            if hasattr(mujoco.mjx, 'step'):
                import inspect
                try:
                    step_sig = inspect.signature(mujoco.mjx.step)
                    print(f"MJX step signature: {step_sig}")
                    
                    # Check if step can be JIT-compiled
                    print("Testing JIT compilation of step function...")
                    try:
                        jit_step = jax.jit(mujoco.mjx.step)
                        print("  ✓ Standard JIT compilation succeeded")
                    except Exception as e:
                        print(f"  ✗ Standard JIT compilation failed: {e}")
                        
                    try:
                        jit_step_static = jax.jit(mujoco.mjx.step, static_argnums=0)
                        print("  ✓ JIT with static_argnums=0 succeeded")
                    except Exception as e:
                        print(f"  ✗ JIT with static_argnums=0 failed: {e}")
                except:
                    print("Could not inspect step function signature")
        else:
            print("MJX API: Unknown version - couldn't identify standard functions")
            
    except ImportError:
        print("MJX available: False")
    
    # Return if GPU is available
    return has_gpu

def benchmark_mjx_vs_cpu(env_name, steps=1000):
    """Run a simple benchmark comparing MJX with and without GPU"""
    print("\n===== MJX Performance Benchmark =====")
    
    # Store original backend
    original_backend = os.environ.get('sim_backend', '')
    
    # Force JAX to run on CPU for the CPU benchmark
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    os.environ['sim_backend'] = 'MJX'
    print("Running benchmark with MJX on CPU...")
    
    try:
        # CPU benchmark
        start_time = time.time()
        env = gym.make(env_name)
        env.reset()
        for _ in range(steps):
            env.step(env.action_space.sample())
        env.close()
        cpu_time = time.time() - start_time
        print(f"MJX on CPU: {cpu_time:.3f}s for {steps} steps ({steps/cpu_time:.1f} FPS)")
    except Exception as e:
        print(f"CPU benchmark failed: {e}")
        cpu_time = float('inf')
    
    # Now try with GPU if available
    if any(d.device_kind == 'gpu' for d in jax.devices()):
        os.environ.pop('JAX_PLATFORM_NAME', None)  # Remove CPU restriction
        print("Running benchmark with MJX on GPU...")
        
        try:
            # GPU benchmark
            start_time = time.time()
            env = gym.make(env_name)
            env.reset()
            for _ in range(steps):
                env.step(env.action_space.sample())
            env.close()
            gpu_time = time.time() - start_time
            print(f"MJX on GPU: {gpu_time:.3f}s for {steps} steps ({steps/gpu_time:.1f} FPS)")
            
            if cpu_time < float('inf'):
                speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
                print(f"GPU speedup: {speedup:.2f}x")
        except Exception as e:
            print(f"GPU benchmark failed: {e}")
    else:
        print("GPU not available, skipping GPU benchmark")
    
    # Restore original environment
    if original_backend:
        os.environ['sim_backend'] = original_backend
    else:
        os.environ.pop('sim_backend', None)
    os.environ.pop('JAX_PLATFORM_NAME', None)

def check_with_minimal_model():
    """Try MJX with a minimal model to check device usage"""
    print("\n===== Testing MJX with Minimal Model =====")
    
    # Minimal MuJoCo XML model
    minimal_xml = """
    <mujoco>
      <worldbody>
        <light pos="0 0 1"/>
        <body pos="0 0 0.1">
          <joint type="free"/>
          <geom type="sphere" size="0.1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    
    try:
        import mujoco
        import mujoco.mjx
        import jax.numpy as jnp
        
        # Save XML to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.xml', mode='w') as f:
            f.write(minimal_xml)
            f.flush()
            xml_path = f.name
            
            # Load model with standard MuJoCo
            model = mujoco.MjModel.from_xml_path(xml_path)
            data = mujoco.MjData(model)
            
            # Try using MJX with better approach for the specific API version
            try:
                if hasattr(mujoco.mjx, 'device_put'):
                    print("Using MJX device_put API")
                    try:
                        mjx_model = mujoco.mjx.device_put(model)
                        mjx_data = mujoco.mjx.device_put(data)
                        print("Successfully created MJX model and data with device_put")
                    except Exception as e:
                        print(f"Error with device_put: {e}")
                        
                elif hasattr(mujoco.mjx, 'put_model'):
                    print("Using MJX put_model/put_data API")
                    
                    # For your specific MJX version, we need a direct approach without JIT first
                    try:
                        print("Trying direct put_model call without JIT")
                        mjx_model = mujoco.mjx.put_model(model)
                        mjx_data = mujoco.mjx.put_data(model, data)
                        print("Successfully created MJX model and data with direct put_model/put_data")
                    except Exception as e:
                        print(f"Error with direct put_model: {e}")
                        
                        # If direct call fails, try more approaches
                        try:
                            print("Trying put_model with explicit static_argnums=0")
                            # Use static_argnums to tell JAX the model is not a JAX array
                            put_model_static = jax.jit(mujoco.mjx.put_model, static_argnums=0)
                            mjx_model = put_model_static(model)
                            mjx_data = mujoco.mjx.put_data(model, data)
                            print("Successfully created MJX model and data with static put_model/put_data")
                        except Exception as e2:
                            print(f"Error with static put_model: {e2}")
                            
                            # Last resort - try non-JIT completely
                            try:
                                print("Trying to create a simplified version...")
                                # Create a JAX array directly to bypass the API
                                import numpy as np
                                dummy_array = jax.numpy.zeros((10, 10))
                                print(f"JAX device: {dummy_array.device()}")
                                print("Successfully created a JAX array, but couldn't create MJX model")
                                return
                            except Exception as e3:
                                print(f"All approaches failed: {e3}")
                                return
                else:
                    print("Could not find known MJX API functions")
                    return
                
                # Try stepping with MJX
                try:
                    if hasattr(mujoco.mjx, 'step'):
                        # Get device info for the model
                        if hasattr(mjx_model, 'device'):
                            print(f"MJX model device: {mjx_model.device}")
                        else:
                            print("Device info not directly available from MJX model")
                        
                        # Time MJX step
                        step_fn = jax.jit(mujoco.mjx.step)
                        
                        # Warm-up JIT
                        for _ in range(5):
                            mjx_data = step_fn(mjx_model, mjx_data)
                        
                        # Benchmark
                        start = time.time()
                        n_steps = 1000
                        for _ in range(n_steps):
                            mjx_data = step_fn(mjx_model, mjx_data)
                        jax.block_until_ready(mjx_data)
                        elapsed = time.time() - start
                        print(f"MJX step: {n_steps} steps in {elapsed:.3f}s ({n_steps/elapsed:.1f} FPS)")
                    else:
                        print("mujoco.mjx.step not available")
                except Exception as e:
                    print(f"Error during MJX step: {e}")
            except Exception as e:
                print(f"Error initializing MJX: {e}")
    except Exception as e:
        print(f"General error: {e}")
        
if __name__ == "__main__":
    print("MJX Device Configuration Utility")
    print("================================")
    
    has_gpu = print_device_info()
    
    # If no GPU is detected but the user is trying to force GPU usage
    if not has_gpu and os.environ.get('JAX_PLATFORM_NAME') == 'gpu':
        print("\nWARNING: You're trying to use GPU, but no GPU was detected by JAX!")
        print("Make sure your CUDA/ROCm installation is correct and visible to JAX.")
    
    # Try to run MJX test
    check_with_minimal_model()
    
    # Run benchmark if requested
    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        env_name = "myoHandPoseFixed-v0" if len(sys.argv) <= 2 else sys.argv[2]
        steps = 1000 if len(sys.argv) <= 3 else int(sys.argv[3])
        benchmark_mjx_vs_cpu(env_name, steps)
    else:
        print("\nTo run the MJX vs CPU benchmark, use:")
        print(f"python {sys.argv[0]} --benchmark [env_name] [steps]")
    
    print("\nTo force JAX to use a specific device, set one of these before running Python:")
    print("  export JAX_PLATFORM_NAME=cpu    # Force CPU")
    print("  export JAX_PLATFORM_NAME=gpu    # Force GPU")
    print("  export JAX_PLATFORM_NAME=tpu    # Force TPU (if available)")