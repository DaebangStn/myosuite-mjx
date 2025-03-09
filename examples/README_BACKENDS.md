# MyoSuite MuJoCo Backends

MyoSuite now supports multiple MuJoCo backends:

1. **MUJOCO** - Standard MuJoCo (CPU-based, default)
2. **MJX** - MuJoCo with GPU acceleration through JAX
3. **MUJOCO_PY** - Legacy MuJoCo Python bindings

## Requirements

- For MUJOCO: `mujoco>=3.0.0`
- For MJX: `mujoco>=3.2.0` with JAX support (`jax`, `jaxlib`)
- For MUJOCO_PY: `mujoco-py` (compatible with older MuJoCo versions)

## Selecting a Backend

You can select a backend in two ways:

### 1. Environment Variable

Set the `sim_backend` environment variable to your desired backend:

```bash
# For MJX
export sim_backend=MJX
python -m myosuite.examples.mjx_myosuite_example.py

# For MUJOCO_PY
export sim_backend=MUJOCO_PY
python -m myosuite.examples.mjx_myosuite_example.py

# For standard MUJOCO (default)
export sim_backend=MUJOCO  # or leave unset
python -m myosuite.examples.mjx_myosuite_example.py
```

### 2. Setting in Code

```python
import os
import gymnasium as gym
import myosuite

# Set backend before creating environment
os.environ["sim_backend"] = "MJX"  # or "MUJOCO" or "MUJOCO_PY"

# Create environment with selected backend
env = gym.make("myoHandPoseFixed-v0")
```

## Controlling MJX Device (CPU vs GPU)

When using the MJX backend, you can control whether JAX uses CPU or GPU:

### 1. Environment Variable

```bash
# Force MJX to use CPU
export JAX_PLATFORM_NAME=cpu
export sim_backend=MJX
python your_script.py

# Force MJX to use GPU
export JAX_PLATFORM_NAME=gpu
export sim_backend=MJX
python your_script.py
```

### 2. Setting in Code

```python
import os
import gymnasium as gym
import myosuite

# Set backend to MJX
os.environ["sim_backend"] = "MJX"

# Force JAX to use CPU
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Create environment with MJX on CPU
env = gym.make("myoHandPoseFixed-v0")
```

### 3. Command Line Arguments in Example Scripts

Our example scripts support device selection via command line arguments:

```bash
# Run with MJX on CPU
python select_backend_example.py --backend MJX --device cpu

# Run with MJX on GPU
python select_backend_example.py --backend MJX --device gpu

# Compare CPU vs GPU performance
python select_backend_example.py --backend MJX --compare-devices
```

## Checking MJX Status

We provide a utility to check if MJX is working correctly and which device it's using:

```bash
python check_mjx_device.py
```

This will show:
- Available JAX devices
- Whether MJX is properly initialized 
- Which device JAX is using
- Performance benchmarks with a minimal test model

## Example Scripts

We provide example scripts to demonstrate the backend selection:

1. **select_backend_example.py** - Select a backend and device
   ```
   python select_backend_example.py --backend MJX --device gpu --env myoHandPoseFixed-v0
   ```

2. **backend_comparison.py** - Benchmark comparing performance between backends
   ```
   python backend_comparison.py --steps 1000 --env myoHandPoseFixed-v0 --backends MUJOCO MJX
   ```

3. **check_mjx_device.py** - Check MJX configuration and hardware usage
   ```
   python check_mjx_device.py --benchmark
   ```

4. **basic_backend_example.py** - Simple example with automatic fallback
   ```
   python basic_backend_example.py
   ```

## Backend Feature Comparison

| Feature | MUJOCO | MJX | MUJOCO_PY |
|---------|--------|-----|-----------|
| Performance | Good | Best (with GPU) | Good |
| GPU Support | No | Yes | No |
| Compatibility | Modern | Modern | Legacy |
| Rendering | Yes | Yes | Yes |
| Stability | High | Medium | High |

## Performance Tips

1. **MJX Backend**:
   - Best for training where many simulation steps are needed
   - Requires a CUDA-compatible GPU for maximum performance
   - First environment creation may be slow due to JAX compilation
   - GPU acceleration works best with larger batch sizes or longer simulations
   - For small environments, CPU can sometimes be faster due to GPU transfer overhead

2. **MUJOCO Backend**:
   - Good balance of performance and stability
   - Default choice for most use cases
   - Recommended for visualization and testing

3. **MUJOCO_PY Backend**:
   - Legacy support for older code
   - May be needed for compatibility with certain projects

## MJX API Compatibility

The MJX API has evolved between MuJoCo versions. Our implementation supports multiple API patterns:

1. **Newer versions** use: `device_put` / `device_get`
2. **Some versions** use: `put_model` / `put_data` / `get_data`

Our implementation tries both patterns and falls back gracefully if needed.

## Troubleshooting

If you encounter issues with a specific backend:

1. **MJX Issues**:
   - Verify MuJoCo version is 3.2.0 or higher
   - Check JAX and CUDA installation with `python check_mjx_device.py`
   - Try forcing CPU with `export JAX_PLATFORM_NAME=cpu` if GPU causes issues
   - The system will fall back to MUJOCO if MJX is not available

2. **MUJOCO_PY Issues**:
   - May need older MuJoCo binaries
   - Check compatibility between mujoco-py and your MuJoCo version

3. **General Issues**:
   - Try a different backend to isolate if the issue is backend-specific
   - Check for environment-specific compatibility issues
   - Run with `--no-render` to eliminate rendering-related issues