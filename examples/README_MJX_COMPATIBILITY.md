# MyoSuite Compatibility with MuJoCo 3.3.0 and MJX

This README explains the changes needed to make MyoSuite compatible with MuJoCo 3.3.0 and MJX.

## The Problem

MyoSuite was originally designed to work with MuJoCo 2.x and earlier versions of MuJoCo 3.x (up to 3.1.x). When using it with MuJoCo 3.3.0, you encounter an error:

```
AttributeError: 'MjModel' object has no attribute 'flex_xvert0'
```

This is because MuJoCo 3.3.0 removed or renamed certain attributes like 'flex_xvert0' that were present in earlier versions.

## The Solution

We've created a compatibility layer that makes MyoSuite work with newer MuJoCo versions by gracefully handling missing attributes. The key changes are:

1. Create a patched version of the `dm_control.mujoco.index` module's `struct_indexer` function that ignores missing attributes
2. Monkey patch this function to be used by dm_control at runtime
3. Maintain a list of known attributes that might be missing in MuJoCo 3.3.0

## Implementation Details

### 1. Create a patched version of the index.py file

Copy dm_control's index.py to a custom location:
```bash
cp /path/to/envs/site-packages/dm_control/mujoco/index.py /path/to/myosuite/myosuite/physics/patched_index.py
```

### 2. Modify the struct_indexer function to handle missing attributes

Edit the patched_index.py file to add a list of known missing attributes and skip them when encountered:

```python
# Define a list of known attributes that might be missing in MuJoCo 3.3.0
missing_attrs_in_new_mujoco = [
    # MjModel missing attributes
    'flex_xvert0', 'flex_xvert', 'flex_vert', 'flex_edge', 'flex_edgeadr', 
    'flex_edgenum', 'flex_elemadr', 'flex_elemedge', 'flex_elemnum', 
    'flex_elem', 'flex_face', 'flex_faceadr', 'flex_facenum', 'flex_faceedge', 
    'flex_tendon', 'flex_tendonadr', 'tex_rgb', 'tex_rgba',
    # MjData missing attributes  
    'qLDiagSqrtInv', 'qLDiagInv', 'flexedge_velocity', 'flexedge_moment',
    'flexvert_xmat', 'flexvert_xpos', 'flexvert_xvel', 'flexvert_velp',
    'flexvert_velr', 'flexface_area', 'flexface_normal', 'flexface_velocity',
    'flexedge_length', 'flex_velm', 'flex_veln', 'flex_tensor', 'flex_state',
    'flex_velocity', 'flex_acceleration', 'flex_force',
    # Add any other attributes found to be missing
    'act_dot', 'efc_vel', 'efc_aref', 'alpha', 'obj', 'sdf_gradient', 'sdf_normal',
    'sdf_result', 'sdf_scratch'
]

# Modify the struct_indexer function to skip missing attributes
try:
  attr = getattr(struct, field_name)
  if not isinstance(attr, np.ndarray) or attr.dtype.fields:
    continue
except AttributeError:
  # Handle missing attributes in MuJoCo 3.3.0+
  if field_name in missing_attrs_in_new_mujoco:
    continue
  else:
    raise
```

### 3. Apply the monkey patch in mj_sim_scene.py

Import and apply the patched function:

```python
# Import our patched version of struct_indexer
from myosuite.physics.patched_index import struct_indexer as patched_struct_indexer

# Apply the monkey patch
dm_control.mujoco.index.struct_indexer = patched_struct_indexer
```

## Usage

After applying these changes, you can use MyoSuite with MuJoCo 3.3.0 and MJX:

```python
import gymnasium as gym
import myosuite

# Create and use a MyoSuite environment as usual
env = gym.make('myoHandPoseFixed-v0')
```

## Example

See the `mjx_myosuite_example.py` file for a complete example of using MyoSuite with MuJoCo 3.3.0 and MJX.

## Additional Notes

- If you encounter more missing attributes, add them to the `missing_attrs_in_new_mujoco` list in `patched_index.py`
- This approach maintains backward compatibility with older MuJoCo versions
- To use MJX, you need to install both mujoco==3.3.0 and mujoco-mjx==3.3.0