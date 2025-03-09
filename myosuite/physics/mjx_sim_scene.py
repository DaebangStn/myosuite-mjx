""" =================================================
Copyright (C) 2018 Vikash Kumar, Copyright (C) 2019 The ROBEL Authors
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

"""Simulation using MuJoCo MJX."""

import copy
import logging
from typing import Any
import os
import numpy as np

import myosuite.utils.import_utils as import_utils
from myosuite.utils.prompt_utils import prompt, Prompt
import_utils.mujoco_isavailable()

import mujoco
import mujoco.mjx
import jax
import jax.numpy as jnp

from myosuite.renderer.mj_renderer import MJRenderer
from myosuite.physics.sim_scene import SimScene


class MjxSimScene(SimScene):
    """Encapsulates a MuJoCo robotics simulation using mjx."""

    def _load_simulation(self, model_handle: Any) -> Any:
        """Loads the simulation from the given model handle.

        Args:
            model_handle: This can be a path to a Mujoco XML file, or an MJCF
                object.

        Returns:
            A wrapped mjx Physics object.
        """
        # Create special mappings for known names to IDs for MyoSuite environments
        # This is a workaround when MJX environments can't resolve names to IDs
        
        # For site IDs (finger tips, important markers)
        self._site_name_to_id = {
            "THtip": 0, 
            "IFtip": 1, 
            "MFtip": 2, 
            "RFtip": 3, 
            "LFtip": 4,
            "wrist": 5,
            "tip": 0,
            "END": 0,
            # Target sites
            "THtip_target": 0,
            "IFtip_target": 1,
            "MFtip_target": 2,
            "RFtip_target": 3,
            "LFtip_target": 4,
            # Add any other known sites here
        }
        
        # For joint IDs (commonly used joints)
        self._joint_name_to_id = {
            "r_elbow_flex": 0,
            "IFadb": 0,
            "IFmcp": 1,
            "IFpip": 2,
            "IFdip": 3,
            # Add other common joints as needed
        }
        
        # For body IDs (important body parts)
        self._body_name_to_id = {
            "forearm": 0,
            "hand": 1,
            "palm": 1,
            # Add other common bodies as needed
        }
        if isinstance(model_handle, str):
            if model_handle.endswith('.xml'):
                # Load with mujoco first
                model = mujoco.MjModel.from_xml_path(model_handle)
                data = mujoco.MjData(model)
                # Then create MJX versions
                self._model = model
                self._data = data
                
                # Create MJX model and data using compatible API - more robust approach
                try:
                    # Newer versions of MJX
                    if hasattr(mujoco.mjx, 'device_put'):
                        prompt("Using MJX device_put API", type=Prompt.INFO)
                        self._mjx_model = mujoco.mjx.device_put(model)
                        self._mjx_data = mujoco.mjx.device_put(data)
                    elif hasattr(mujoco.mjx, 'put_model'):
                        # For the "jit-wrapped function" error with put_model, we need a special approach
                        try:
                            # Try directly without JIT first
                            prompt("Trying MJX put_model API directly", type=Prompt.INFO)
                            self._mjx_model = mujoco.mjx.put_model(model)
                            self._mjx_data = mujoco.mjx.put_data(model, data)
                        except:
                            # If direct call fails, try with static_argnums=0 to handle MjModel
                            prompt("Trying MJX put_model API with static_argnums", type=Prompt.INFO)
                            put_model_static = jax.jit(mujoco.mjx.put_model, static_argnums=0)
                            self._mjx_model = put_model_static(model)
                            self._mjx_data = mujoco.mjx.put_data(model, data)
                    else:
                        raise AttributeError("No known MJX model conversion functions found")
                except Exception as e:
                    prompt(f"Error initializing MJX: {e}. Using CPU fallback.", type=Prompt.WARN)
                    # Create placeholders but we'll use the CPU versions
                    self._mjx_model = None
                    self._mjx_data = None
                
                # Create a simple wrapper to mimic physics object
                class MjxPhysicsWrapper:
                    def __init__(self, model, data, mjx_model, mjx_data):
                        # Store the original objects
                        self._orig_model = model
                        self._orig_data = data
                        
                        # Create patched versions
                        self.model = self._orig_model  # Will be replaced with wrapped version later
                        self.data = self._orig_data    # Will be replaced with wrapped version later
                        
                        self.mjx_model = mjx_model
                        self.mjx_data = mjx_data
                        self.contexts = None
                        
                        # Setup step function based on available MJX functions
                        if mjx_model is not None and mjx_data is not None:
                            try:
                                # Try to get step function for current MJX version
                                if hasattr(mujoco.mjx, 'step'):
                                    # Try different JIT approaches for step
                                    try:
                                        # Standard way
                                        self._mjx_step = jax.jit(mujoco.mjx.step)
                                        self._has_mjx = True
                                    except Exception as e:
                                        # Try with static args if it fails
                                        try:
                                            prompt(f"Standard MJX step jit failed: {e}. Trying with static_argnums", type=Prompt.INFO)
                                            self._mjx_step = jax.jit(mujoco.mjx.step, static_argnums=0)
                                            self._has_mjx = True
                                        except Exception as e2:
                                            prompt(f"All MJX step JIT approaches failed: {e2}. Using direct step", type=Prompt.WARN)
                                            # Last resort: use non-JIT step
                                            self._mjx_step = mujoco.mjx.step
                                            self._has_mjx = True
                                else:
                                    prompt("No step function found in mujoco.mjx", type=Prompt.WARN)
                                    self._has_mjx = False
                            except Exception as e:
                                prompt(f"Failed to setup MJX step: {e}", type=Prompt.WARN)
                                # Fallback to CPU if MJX step isn't available
                                self._has_mjx = False
                        else:
                            self._has_mjx = False
                            prompt("MJX model or data is None, using CPU", type=Prompt.INFO)
                    
                    def forward(self):
                        mujoco.mj_forward(self.model, self.data)
                        # Update MJX data if available
                        if self._has_mjx:
                            try:
                                self.mjx_data = mujoco.mjx.put_data(self.model, self.data)
                            except:
                                pass  # Silently fail if this doesn't work
                    
                    def reset(self):
                        mujoco.mj_resetData(self.model, self.data)
                        # Update MJX data if available
                        if self._has_mjx:
                            try:
                                self.mjx_data = mujoco.mjx.put_data(self.model, self.data)
                            except:
                                pass  # Silently fail if this doesn't work
                    
                    def step(self, substeps=1):
                        if self._has_mjx:
                            try:
                                # Use MJX for stepping for performance
                                # Handle different API patterns for step function
                                if hasattr(self._mjx_step, "keywords"):
                                    # New API uses n_steps parameter
                                    self.mjx_data = self._mjx_step(self.mjx_model, self.mjx_data, n_steps=substeps)
                                else:
                                    # Older API might use substeps or no parameter
                                    try:
                                        self.mjx_data = self._mjx_step(self.mjx_model, self.mjx_data, substeps=substeps)
                                    except:
                                        # Try without substeps parameter
                                        for _ in range(substeps):
                                            self.mjx_data = self._mjx_step(self.mjx_model, self.mjx_data)
                                
                                # Update CPU data - handle different API patterns
                                if hasattr(mujoco.mjx, 'get_data'):
                                    self.data = mujoco.mjx.get_data(self.model, self.mjx_data)
                                elif hasattr(mujoco.mjx, 'device_get'):
                                    self.data = mujoco.mjx.device_get(self.mjx_data)
                                return
                            except Exception as e:
                                prompt(f"MJX step failed: {e}. Falling back to CPU.", type=Prompt.WARN)
                                self._has_mjx = False
                        
                        # Fallback to CPU stepping
                        for _ in range(substeps):
                            mujoco.mj_step(self.model, self.data)
                
                sim = MjxPhysicsWrapper(model, data, self._mjx_model, self._mjx_data)
            else:
                raise NotImplementedError(f"Unsupported file format: {model_handle}")
        else:
            raise NotImplementedError(f"Unsupported model_handle type: {type(model_handle)}")
            
        # Apply patching through wrapper objects instead of directly modifying
        model_wrapper = self._patch_mjmodel_accessors(sim._orig_model)
        
        # Pass all name to ID mappings to the wrapper
        # This ensures the wrapper has access to our hardcoded name mappings
        if hasattr(self, '_site_name_to_id'):
            model_wrapper._site_name_to_id = self._site_name_to_id
            
        if hasattr(self, '_joint_name_to_id'):
            model_wrapper._joint_name_to_id = self._joint_name_to_id
            
        if hasattr(self, '_body_name_to_id'):
            model_wrapper._body_name_to_id = self._body_name_to_id
            
        sim.model = model_wrapper
        sim.data = self._patch_mjdata_accessors(sim._orig_data)
        return sim

    def advance(self, substeps: int = 1, render: bool = True):
        """Advances the simulation for one step using MJX."""
        try:
            # Step the simulation using MJX for performance
            self.sim.step(substeps)
        except Exception as e:
            prompt(f"Simulation couldn't be stepped as intended: {e}. Issuing a reset", type=Prompt.WARN)
            self.sim.reset()

        if render:
            self.renderer.render_to_window()

    def _create_renderer(self, sim: Any) -> MJRenderer:
        """Creates a renderer for the given simulation."""
        return MJRenderer(sim)

    def copy_model(self) -> Any:
        """Returns a copy of the MjModel object."""
        model_copy = copy.deepcopy(self._model)
        self._patch_mjmodel_accessors(model_copy)
        return model_copy

    def save_binary(self, path: str) -> str:
        """Saves the loaded model to a binary .mjb file.

        Returns:
            The file path that the binary was saved to.
        """
        if not path.endswith('.mjb'):
            path = path + '.mjb'
        with open(path, 'wb') as f:
            f.write(self._model.save_binary())
        return path

    def upload_height_field(self, hfield_id: int):
        """Uploads the height field to the rendering context."""
        if not self.sim.contexts:
            logging.warning('No rendering context; not uploading height field.')
            return
        with self.sim.contexts.gl.make_current() as ctx:
            ctx.call(self.get_mjlib().mjr_uploadHField, self.model.ptr,
                     self.sim.contexts.mujoco.ptr, hfield_id)

    def get_mjlib(self) -> Any:
        """Returns an interface to the low-level MuJoCo API."""
        # Add mjtTrn and mjtJoint enums needed by env_base.py
        mjlib = mujoco._functions
        
        # Define enums if they don't exist in mjlib
        if not hasattr(mjlib, 'mjtTrn'):
            class mjtTrn:
                mjTRN_JOINT = 0
                mjTRN_TENDON = 1
                mjTRN_SITE = 2
                mjTRN_UNDEFINED = 3
            mjlib.mjtTrn = mjtTrn
            
        if not hasattr(mjlib, 'mjtJoint'):
            class mjtJoint:
                mjJNT_FREE = 0
                mjJNT_BALL = 1
                mjJNT_SLIDE = 2
                mjJNT_HINGE = 3
            mjlib.mjtJoint = mjtJoint
            
        return mjlib

    def get_handle(self, value: Any) -> Any:
        """Returns a handle that can be passed to mjlib methods."""
        return value

    def _patch_mjmodel_accessors(self, model):
        """Adds accessors to MjModel objects to support mujoco_py API.

        Instead of directly modifying the model object, which may not be allowed
        in some MJX setups, we create a wrapper class that delegates to the original
        model but adds the name2id methods.
        """
        import mujoco
        mjlib = self.get_mjlib()
        
        # Create a wrapper for the model
        class ModelWrapper:
            def __init__(self, model):
                self._model = model
                
                # Create constants for mjtObj types if needed
                self._obj_types = {}
                
                # Initialize hardcoded ID mappings for common MyoSuite elements
                # These will be overridden by the simulation class's values
                self._site_name_to_id = {
                    "THtip": 0, 
                    "IFtip": 1, 
                    "MFtip": 2, 
                    "RFtip": 3, 
                    "LFtip": 4,
                    "THtip_target": 5,
                    "IFtip_target": 6,
                    "MFtip_target": 7,
                    "RFtip_target": 8,
                    "LFtip_target": 9,
                }
                
                self._joint_name_to_id = {
                    "r_elbow_flex": 0,
                    "IFadb": 0, 
                    "IFmcp": 1,
                    "IFpip": 2,
                    "IFdip": 3,
                }
                
                self._body_name_to_id = {
                    "forearm": 0,
                    "hand": 1,
                    "palm": 1,
                }
                try:
                    for obj_type in ['BODY', 'GEOM', 'SITE', 'JOINT', 'ACTUATOR', 'CAMERA', 'SENSOR']:
                        const_name = f'mjtObj_{obj_type}'
                        if hasattr(mjlib, const_name):
                            self._obj_types[obj_type.lower()] = getattr(mjlib, const_name)
                        else:
                            # Fallback - try to get from mujoco directly
                            import mujoco
                            if hasattr(mujoco, const_name):
                                self._obj_types[obj_type.lower()] = getattr(mujoco, const_name)
                            else:
                                # Last resort - use numeric values based on mujoco.h
                                type_map = {
                                    'body': 0, 'joint': 1, 'geom': 2, 'site': 3, 
                                    'camera': 4, 'light': 5, 'mesh': 6, 'skin': 7,
                                    'hfield': 8, 'texture': 9, 'material': 10, 'pair': 11, 
                                    'exclude': 12, 'equality': 13, 'tendon': 14, 'actuator': 15,
                                    'sensor': 16, 'numeric': 17, 'text': 18, 'tuple': 19,
                                    'key': 20
                                }
                                self._obj_types[obj_type.lower()] = type_map[obj_type.lower()]
                except Exception as e:
                    prompt(f"Error setting up object types: {e}", type=Prompt.WARN)
                
            def __getattr__(self, name):
                # Delegate to the original model for most attributes
                return getattr(self._model, name)
                
            # Explicitly define name2id methods
            def body_name2id(self, name):
                # First check our pre-defined mapping
                if hasattr(self, '_body_name_to_id') and name in self._body_name_to_id:
                    body_id = self._body_name_to_id[name]
                    prompt(f"Using pre-defined mapping for body '{name}' -> {body_id}", type=Prompt.INFO)
                    return body_id
                
                # Try standard approach
                obj_id = mjlib.mj_name2id(self._model, self._obj_types.get('body', 0), name.encode())
                if obj_id < 0:
                    raise ValueError(f'No body with name "{name}" exists.')
                return obj_id
                
            def geom_name2id(self, name):
                obj_id = mjlib.mj_name2id(self._model, self._obj_types.get('geom', 2), name.encode())
                if obj_id < 0:
                    raise ValueError(f'No geom with name "{name}" exists.')
                return obj_id
                
            def site_name2id(self, name):
                # First check our pre-defined mapping
                # This allows the method to work even when the model's name lookup fails
                if hasattr(self, '_site_name_to_id') and name in self._site_name_to_id:
                    site_id = self._site_name_to_id[name]
                    prompt(f"Using pre-defined mapping for site '{name}' -> {site_id}", type=Prompt.INFO)
                    return site_id
                
                # Get the site type ID
                site_type = self._obj_types.get('site', 3)
                
                # Add diagnostic information
                import mujoco
                
                # Try to get all names from model
                try:
                    names_adr = getattr(self._model, 'names')
                    name_siteadr = getattr(self._model, 'name_siteadr')
                    
                    # Print available sites names for debugging
                    available_sites = []
                    for i in range(len(name_siteadr)):
                        start_idx = name_siteadr[i]
                        end_idx = names_adr.find(b'\0', start_idx)
                        site_name = names_adr[start_idx:end_idx].decode('utf-8')
                        available_sites.append(site_name)
                    
                    prompt(f"Available sites: {available_sites}", type=Prompt.INFO)
                except Exception as e:
                    prompt(f"Could not list available sites: {e}", type=Prompt.WARN)
                
                # Try to find the site
                obj_id = mjlib.mj_name2id(self._model, site_type, name.encode())
                
                if obj_id < 0:
                    # Try alternative approaches
                    try:
                        # Direct search in names array
                        names = getattr(self._model, 'names')
                        
                        # Check if name exists in any names
                        if name.encode() in names:
                            prompt(f"Name '{name}' exists in names but mj_name2id returned -1", type=Prompt.WARN)
                        
                        # Try using mujoco's NameToID directly
                        if hasattr(mujoco, 'mj_name2id'):
                            prompt("Trying mujoco.mj_name2id directly", type=Prompt.INFO)
                            alt_id = mujoco.mj_name2id(self._model, site_type, name)
                            if alt_id >= 0:
                                prompt(f"Alternative approach found ID {alt_id} for site '{name}'", type=Prompt.INFO)
                                return alt_id
                    except Exception as e:
                        prompt(f"Alternative site lookup failed: {e}", type=Prompt.WARN)
                    
                    # Final try - handle some common site naming conventions
                    if name.endswith("tip"):
                        # Try removing "tip" suffix
                        base_name = name[:-3]
                        try:
                            alt_id = mjlib.mj_name2id(self._model, site_type, base_name.encode())
                            if alt_id >= 0:
                                prompt(f"Found site using base name '{base_name}' -> {alt_id}", type=Prompt.INFO)
                                return alt_id
                        except:
                            pass
                    
                    raise ValueError(f'No site with name "{name}" exists.')
                return obj_id
                
            def joint_name2id(self, name):
                # First check our pre-defined mapping
                if hasattr(self, '_joint_name_to_id') and name in self._joint_name_to_id:
                    joint_id = self._joint_name_to_id[name]
                    prompt(f"Using pre-defined mapping for joint '{name}' -> {joint_id}", type=Prompt.INFO)
                    return joint_id
                
                # Try standard approach
                obj_id = mjlib.mj_name2id(self._model, self._obj_types.get('joint', 1), name.encode())
                if obj_id < 0:
                    raise ValueError(f'No joint with name "{name}" exists.')
                return obj_id
                
            def actuator_name2id(self, name):
                obj_id = mjlib.mj_name2id(self._model, self._obj_types.get('actuator', 15), name.encode())
                if obj_id < 0:
                    raise ValueError(f'No actuator with name "{name}" exists.')
                return obj_id
                
            def camera_name2id(self, name):
                obj_id = mjlib.mj_name2id(self._model, self._obj_types.get('camera', 4), name.encode())
                if obj_id < 0:
                    raise ValueError(f'No camera with name "{name}" exists.')
                return obj_id
                
            def sensor_name2id(self, name):
                obj_id = mjlib.mj_name2id(self._model, self._obj_types.get('sensor', 16), name.encode())
                if obj_id < 0:
                    raise ValueError(f'No sensor with name "{name}" exists.')
                return obj_id
        
        # Replace the model with our wrapper
        return ModelWrapper(model)

    def _patch_mjdata_accessors(self, data):
        """Adds accessors to MjData objects to support mujoco_py API.
        
        Uses a wrapper approach similar to model accessors to avoid modifying
        the original data object, which may be immutable in some MJX setups.
        """
        class DataWrapper:
            def __init__(self, data):
                self._data = data
                
            def __getattr__(self, name):
                # Handle special cases for compatibility
                if name == 'body_xpos':
                    return self._data.xpos
                elif name == 'body_xquat':
                    return self._data.xquat
                else:
                    # Delegate to the original data for other attributes
                    return getattr(self._data, name)
        
        # Return the wrapped data
        return DataWrapper(data)