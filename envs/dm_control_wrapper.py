import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dm_control import suite
from dm_control.suite.wrappers import pixels
import cv2
from collections import deque

class DMControlWrapper(gym.Env):
    """Wrapper for DeepMind Control Suite environments."""
    
    def __init__(self, domain_name, task_name, size=(64, 64), camera_id=0, channels_first=False, seed=None):
        """Initialize the DM Control environment wrapper.
        
        Args:
            domain_name: The domain name for the DM Control environment (e.g., 'cartpole', 'walker')
            task_name: The task name for the DM Control environment (e.g., 'swingup', 'run')
            size: The size to render the image, as a tuple (height, width)
            camera_id: The camera ID to use for rendering
            channels_first: Whether to return images with shape (C, H, W) or (H, W, C)
            seed: Random seed for the environment
        """
        self.domain_name = domain_name
        self.task_name = task_name
        self.size = size
        self.camera_id = camera_id
        self.channels_first = channels_first
        
        # Create DM Control environment
        self.env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs={'random': seed})
        
        # Wrap environment to provide pixel observations
        self.env = pixels.Wrapper(self.env, render_kwargs={
            'height': size[0],
            'width': size[1],
            'camera_id': camera_id
        })
        
        # Define action space - Make sure this is correctly set up for continuous actions
        self.action_spec = self.env.action_spec()
        self.action_space = spaces.Box(
            low=self.action_spec.minimum.astype(np.float32),
            high=self.action_spec.maximum.astype(np.float32),
            dtype=np.float32
        )
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(3, size[0], size[1]) if channels_first else (size[0], size[1], 3),
            dtype=np.uint8
        )
        
        # For tracking episode information
        self.episode_frame_number = 0
        
    def reset(self, seed=None, options=None):
        """Reset the environment and return the initial observation."""
        # Handle seed parameter as per newer Gymnasium API
        if seed is not None:
            self.env = suite.load(domain_name=self.domain_name, task_name=self.task_name, 
                                  task_kwargs={'random': seed})
            self.env = pixels.Wrapper(self.env, render_kwargs={
                'height': self.size[0],
                'width': self.size[1],
                'camera_id': self.camera_id
            })
            
        time_step = self.env.reset()
        self.episode_frame_number = 0
        obs = self._get_observation(time_step)
        
        info = {
            'is_terminal': False,
            'episode_frame_number': self.episode_frame_number
        }
        return obs, info
    
    def step(self, action):
        """Take a step in the environment."""
        # Make sure action is a numpy array with the correct shape
        if isinstance(action, (int, float)):
            action = np.array([action], dtype=np.float32)
        elif isinstance(action, list):
            action = np.array(action, dtype=np.float32)
        
        time_step = self.env.step(action)
        self.episode_frame_number += 1
        
        obs = self._get_observation(time_step)
        reward = time_step.reward or 0.0
        done = time_step.last()
        
        # Gymnasium standard requires both terminated and truncated flags
        # For DM Control, we'll consider all done states as terminated (not truncated)
        terminated = done
        truncated = False
        
        info = {
            'is_terminal': done,
            'episode_frame_number': self.episode_frame_number
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self, time_step):
        """Extract pixel observation from the environment time step."""
        pixels = time_step.observation['pixels']
        
        # Convert to the expected format
        if self.channels_first:
            # Convert from (H, W, C) to (C, H, W)
            pixels = pixels.transpose(2, 0, 1)
            
        return pixels.astype(np.uint8)
    
    def close(self):
        """Close the environment."""
        self.env.close()

class FrameSkipWrapper(gym.Wrapper):
    """Wrapper for frame skipping in any environment."""
    
    def __init__(self, env, skip=4):
        """Initialize the frame skip wrapper.
        
        Args:
            env: The environment to wrap
            skip: Number of frames to skip
        """
        super().__init__(env)
        self.skip = skip
        self.obs_buffer = deque(maxlen=2)
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        obs, info = self.env.reset(seed=seed, options=options)
        self.obs_buffer.clear()
        self.obs_buffer.append(obs)
        return obs, info
    
    def step(self, action):
        """Take a step in the environment, repeat action for 'skip' frames."""
        total_reward = 0.0
        terminated = False
        truncated = False
        info = None
        
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.obs_buffer.append(obs)
            total_reward += reward
            if terminated or truncated:
                break
                
        if len(self.obs_buffer) == 1:
            obs = self.obs_buffer[0]
        else:
            # For visual observations, taking the max can help with flickering
            obs = np.max(np.stack(self.obs_buffer), axis=0)
            
        return obs, total_reward, terminated, truncated, info

def build_dm_control_env(domain_name, task_name, size=(64, 64), camera_id=0, channels_first=False, frame_skip=4, seed=None):
    """Build a DM Control environment with appropriate wrappers."""
    env = DMControlWrapper(domain_name, task_name, size, camera_id, channels_first, seed)
    if frame_skip > 1:
        env = FrameSkipWrapper(env, skip=frame_skip)
    return env