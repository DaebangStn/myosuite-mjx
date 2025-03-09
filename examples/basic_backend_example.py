#!/usr/bin/env python3
"""
Basic example showing backend selection with fallback
"""

import os
import time
import gymnasium as gym
import myosuite
from myosuite.physics.sim_scene import SimBackend

# Try to load MJX and check compatibility
try:
    import mujoco.mjx
    import jax
    
    # Check JAX device availability
    devices = jax.devices()
    has_gpu = any(d.device_kind == 'gpu' for d in devices)
    
    # Get available MJX functions to confirm API presence
    mjx_functions = [func for func in dir(mujoco.mjx) if not func.startswith('_')]
    mjx_version = getattr(mujoco.mjx, '__version__', 'Unknown')
    
    print(f"Found MJX version: {mjx_version}")
    print(f"Available MJX functions: {', '.join(sorted(mjx_functions)[:5])}...")
    print(f"JAX devices: {devices}")
    
    # Check if critical functions exist
    if 'step' in mjx_functions and ('put_model' in mjx_functions or 'device_put' in mjx_functions):
        HAS_MJX = True
    else:
        print("MJX found but missing required functions")
        HAS_MJX = False
        
except (ImportError, AttributeError) as e:
    print(f"MJX not available: {e}")
    HAS_MJX = False

# Print available backends
print("\nChecking available backends:")
print(f"MJX available: {SimBackend.check_mjx_available()}")
mujoco_version, is_mjx_compatible = SimBackend.check_mujoco_version()
print(f"MuJoCo version: {mujoco_version}, MJX compatible: {is_mjx_compatible}")

# List of backends to try in priority order
available_backends = []

# Add MJX if available, but always prefer CPU with MJX for better reliability
if SimBackend.check_mjx_available():
    # Set platform to CPU for MJX by default unless GPU specifically needed
    if 'JAX_PLATFORM_NAME' not in os.environ:
        print("Setting JAX_PLATFORM_NAME=cpu for more reliable MJX operation")
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    available_backends.append("MJX")

# Add standard MuJoCo (always available if you're running this)
available_backends.append("MUJOCO")

# Try to add MUJOCO_PY (might not be available)
try:
    import mujoco_py
    available_backends.append("MUJOCO_PY")
except ImportError:
    pass

print(f"Available backends (in order of preference): {available_backends}")

# Use the best available backend
if available_backends:
    selected_backend = available_backends[0]
    print(f"Selected backend: {selected_backend}")
    os.environ["sim_backend"] = selected_backend
    
    # For MJX, prefer CPU by default for more reliable operation
    if selected_backend == "MJX" and "JAX_PLATFORM_NAME" not in os.environ:
        print("Setting JAX_PLATFORM_NAME=cpu for more reliable MJX operation")
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
else:
    print("No backends available!")
    exit(1)

# Run a simple environment
env_name = "myoHandPoseFixed-v0"
print(f"\nRunning {env_name} with {selected_backend} backend")

# Add debug function to check model structure
def debug_model_loading(env_name):
    """Print model information to debug loading issues"""
    from myosuite.envs.env_variants import get_env_class
    import inspect
    import os
    
    # Get environment class
    env_class, _ = get_env_class(env_name)
    
    # Inspect initialization parameters
    print(f"Environment class: {env_class}")
    print(f"Init parameters: {inspect.signature(env_class.__init__)}")
    
    # Get model path
    try:
        # Try to find model_path in init defaults
        init_defaults = {
            k: v.default
            for k, v in inspect.signature(env_class.__init__).parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        model_path = init_defaults.get("model_path", None)
        
        if model_path:
            print(f"Model path from defaults: {model_path}")
            if not os.path.isabs(model_path):
                print("Path is relative, needs to be resolved")
            if os.path.exists(model_path):
                print(f"Model file exists: {os.path.abspath(model_path)}")
            else:
                print(f"Model file does not exist at path: {os.path.abspath(model_path)}")
    except Exception as e:
        print(f"Error getting model path: {e}")
    
    # Try loading with standard MuJoCo to verify
    try:
        print("\nTesting model loading with standard MuJoCo:")
        import mujoco
        
        # Create test environment to get model path
        tmp_env = env_class()
        model_path = tmp_env.model_path
        tmp_env.close()
        
        # Load with standard MuJoCo
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        
        # Check for the site
        site_found = False
        if hasattr(model, 'site_names'):
            sites = model.site_names
            print(f"Sites in model: {sites}")
            if 'THtip' in sites:
                site_found = True
        
        # Fallback approach
        if not site_found:
            print("Checking site names with mj_name2id:")
            for possible_site in ["THtip", "TH_tip", "TH", "tip", "IFtip", "MFtip", "RFtip", "LFtip"]:
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.SITE, possible_site)
                print(f"  {possible_site}: {site_id}")
        
        print("Standard MuJoCo model loaded successfully")
    except Exception as e:
        print(f"Error loading with standard MuJoCo: {e}")

# Debug model loading first
debug_model_loading(env_name)

try:
    # Create and run environment
    print("\nCreating environment with gym.make()...")
    env = gym.make(env_name)
    obs, info = env.reset(seed=42)
except Exception as e:
    print(f"Error creating environment: {e}")
    
    # Try again with simplified approach
    print("\nTrying again with no render mode and with CPU:")
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    try:
        env = gym.make(env_name)
        obs, info = env.reset(seed=42)
    except Exception as e2:
        print(f"Still failed: {e2}")
        exit(1)

num_steps = 100
start_time = time.time()

for i in range(num_steps):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if i % 10 == 0:
        print(f"Step {i}")
    
    if terminated or truncated:
        print("Episode finished early")
        break
    
    time.sleep(0.01)  # Slow down for visualization

elapsed = time.time() - start_time
fps = num_steps / elapsed

print(f"\nSimulation completed")
print(f"Time elapsed: {elapsed:.2f} seconds ({fps:.1f} FPS)")
env.close()

print("\nTo try a different backend specifically:")
for backend in available_backends:
    print(f"  export sim_backend={backend} && python {__file__}")