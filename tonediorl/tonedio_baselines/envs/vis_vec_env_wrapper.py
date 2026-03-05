import os
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


class VisFlightEnvVec(VecEnv):
    """
    SB3-compatible VecEnv wrapper for Flightmare's C++ VecEnv binding (flightgym.QuadrotorEnv_v1).

    The underlying C++ API is assumed to be:
      - getObsDim(), getActDim(), getNumOfEnvs()
      - getExtraInfoNames()
      - reset(obs_out)
      - step(action_in, obs_out, rew_out, done_out, extra_out)
      - setSeed(seed)
      - close()
      - connectUnity(), disconnectUnity()  (optional)
      - curriculumUpdate()                (optional)
    """

    IMG_HEIGHT = 84
    IMG_WIDTH = 84
    IMG_CHANNELS = 3

    def __init__(self, impl, use_obs_norm: bool = True):
        """
        :param impl: C++ VecEnv implementation (flightgym.QuadrotorEnv_v1)
        :param use_obs_norm: (bool) Whether to use observation normalization. 
                             If False, observations are returned without normalization.
        """
        self.wrapper = impl
        self.use_obs_norm = use_obs_norm

        self.num_obs = int(self.wrapper.getObsDim())
        self.num_acts = int(self.wrapper.getActDim())
        self._num_envs = int(self.wrapper.getNumOfEnvs())
        self._is_image_obs = self.num_obs == (
            self.IMG_HEIGHT * self.IMG_WIDTH * self.IMG_CHANNELS
        )

        if self._is_image_obs and use_obs_norm:
            print("[FlightEnvVecSB3] image observation detected, disabling observation normalization.")
            self.use_obs_norm = False

        # SB3 expects per-env spaces (not batched)
        if self._is_image_obs:
            self._observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS),
                dtype=np.uint8,
            )
        else:
            self._observation_space = spaces.Box(
                low=-np.inf * np.ones(self.num_obs, dtype=np.float32),
                high=np.inf * np.ones(self.num_obs, dtype=np.float32),
                dtype=np.float32,
            )
        self._action_space = spaces.Box(
            low=-1.0 * np.ones(self.num_acts, dtype=np.float32),
            high=1.0 * np.ones(self.num_acts, dtype=np.float32),
            dtype=np.float32,
        )

        self._actions = None
        


        # Buffers (batched)
        self._observation = np.zeros((self._num_envs, self.num_obs), dtype=np.float32)
        self._reward = np.zeros((self._num_envs,), dtype=np.float32)
        self._done = np.zeros((self._num_envs,), dtype=bool)

        self._extraInfoNames = list(self.wrapper.getExtraInfoNames())
        self._extraInfo = np.zeros((self._num_envs, len(self._extraInfoNames)), dtype=np.float32)

        # Episode bookkeeping (SB3 uses info["episode"] convention)
        self._ep_rewards = [[] for _ in range(self._num_envs)]

        self.max_episode_steps = 300

        # Observation normalization
        if self.use_obs_norm:
            # shape is (obs_dim,) - RunningMeanStd handles batched updates automatically
            self.obs_rms = RunningMeanStd(shape=(self.num_obs,))
            self.obs_rms_new = RunningMeanStd(shape=(self.num_obs,))
        else:
            self.obs_rms = None
            self.obs_rms_new = None

        print(f"[FlightEnvVecSB3] num_envs={self._num_envs}, obs_dim={self.num_obs}, act_dim={self.num_acts}, use_obs_norm={self.use_obs_norm}")

    def seed(self, seed=0):
        self.wrapper.setSeed(seed)

    # def step(self, action):
    #     self.wrapper.step(action, self._observation,
    #                       self._reward, self._done, self._extraInfo)

    #     if len(self._extraInfoNames) is not 0:
    #         info = [{'extra_info': {
    #             self._extraInfoNames[j]: self._extraInfo[i, j] for j in range(0, len(self._extraInfoNames))
    #         }} for i in range(self.num_envs)]
    #     else:
    #         info = [{} for i in range(self.num_envs)]

    #     for i in range(self.num_envs):
    #         self.rewards[i].append(self._reward[i])
    #         if self._done[i]:
    #             eprew = sum(self.rewards[i])
    #             eplen = len(self.rewards[i])
    #             epinfo = {"r": eprew, "l": eplen}
    #             info[i]['episode'] = epinfo
    #             self.rewards[i].clear()

    #     return self._observation.copy(), self._reward.copy(), \
    #         self._done.copy(), info.copy()

    def stepUnity(self, action, send_id):
        receive_id = self.wrapper.stepUnity(action, self._observation,
                                            self._reward, self._done, self._extraInfo, send_id)

        return receive_id

    def sample_actions(self):
        actions = []
        for i in range(self.num_envs):
            action = self.action_space.sample().tolist()
            actions.append(action)
        return np.asarray(actions, dtype=np.float32)

    def reset(self):
        self._reward[:] = 0.0
        self._done[:] = False
        # Flightmare fills the provided obs buffer
        self.wrapper.reset(self._observation)
        # Update normalization statistics (if enabled)
        if self.use_obs_norm:
            self.obs_rms_new.update(self._observation)
        # Return normalized observation (or raw if normalization disabled)
        return self._format_obs(self.normalize_obs(self._observation))

    # def reset_and_update_info(self):
    #     return self.reset(), self._update_epi_info()

    # def _update_epi_info(self):
    #     info = [{} for _ in range(self.num_envs)]

    #     for i in range(self.num_envs):
    #         eprew = sum(self.rewards[i])
    #         eplen = len(self.rewards[i])
    #         epinfo = {"r": eprew, "l": eplen}
    #         info[i]['episode'] = epinfo
    #         self.rewards[i].clear()
    #     return info

    def render(self, mode='human'):
        raise RuntimeError('This method is not implemented')

    def close(self):
        self.wrapper.close()

    def connectUnity(self):
        return self.wrapper.connectUnity()

    def disconnectUnity(self):
        try:
            return self.wrapper.disconnectUnity()
        except RuntimeError as e:
            print(f"[FlightEnvVecSB3] disconnectUnity warning: {e}")
            return False

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def extra_info_names(self):
        return self._extraInfoNames

    def start_recording_video(self, file_name):
        raise RuntimeError('This method is not implemented')

    def stop_recording_video(self):
        raise RuntimeError('This method is not implemented')

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    def step_async(self, actions):
        # SB3 may provide list, np.ndarray, or torch tensor converted to np by policy
        actions = np.asarray(actions, dtype=np.float32)

        # Expected shape: (num_envs, act_dim)
        if actions.ndim == 1:
            # If user accidentally provides (act_dim,), broadcast to all envs
            if actions.shape[0] != self.num_acts:
                raise ValueError(f"Invalid action shape {actions.shape}, expected ({self._num_envs}, {self.num_acts}) or ({self.num_acts},)")
            actions = np.tile(actions[None, :], (self._num_envs, 1))

        if actions.shape != (self._num_envs, self.num_acts):
            raise ValueError(f"Invalid action shape {actions.shape}, expected ({self._num_envs}, {self.num_acts})")

        self._actions = actions

    def step_wait(self):
        if self._actions is None:
            raise RuntimeError("step_wait() called before step_async().")

        # C++ fills buffers in-place
        self.wrapper.step(self._actions, self._observation, self._reward, self._done, self._extraInfo)

        # Update normalization statistics (if enabled)
        if self.use_obs_norm:
            self.obs_rms_new.update(self._observation)

        # infos: extra_info만 넣어줌 (episode는 VecMonitor가 처리)
        if len(self._extraInfoNames) > 0:
            infos = [
                {"extra_info": {self._extraInfoNames[j]: float(self._extraInfo[i, j])
                                for j in range(len(self._extraInfoNames))}}
                for i in range(self._num_envs)
            ]
        else:
            infos = [{} for _ in range(self._num_envs)]

        # Return normalized observation
        obs = self._format_obs(self.normalize_obs(self._observation))
        rews = self._reward.copy()
        dones = self._done.copy()

        self._actions = None

        return obs, rews, dones, infos
    
    def get_attr(self, attr_name, indices=None):
        """
        Return attribute from vectorized environment.
        :param attr_name: (str) The name of the attribute whose value to return
        :param indices: (list,int) Indices of envs to get attribute from
        :return: (list) List of values of 'attr_name' in all environments
        """
        indices = self._get_indices(indices)
        if not hasattr(self, attr_name):
            # SB3 expects a list with length=len(indices)
            return [None for _ in indices]
        value = getattr(self, attr_name)
        return [value for _ in indices]

    def set_attr(self, attr_name, value, indices=None):
        """
        Set attribute inside vectorized environments.
        :param attr_name: (str) The name of attribute to assign new value
        :param value: (obj) Value to assign to `attr_name`
        :param indices: (list,int) Indices of envs to assign value
        :return: (NoneType)
        """
        _ = self._get_indices(indices)
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """
        Call instance methods of vectorized environments.
        :param method_name: (str) The name of the environment method to invoke.
        :param indices: (list,int) Indices of envs whose method to call
        :param method_args: (tuple) Any positional arguments to provide in the call
        :param method_kwargs: (dict) Any keyword arguments to provide in the call
        :return: (list) List of items returned by the environment's method call
        """
        indices = self._get_indices(indices)

        # Prefer wrapper methods first
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            result = method(*method_args, **method_kwargs)
            return [result for _ in indices]

        # Fallback to underlying C++ env
        if hasattr(self.wrapper, method_name):
            method = getattr(self.wrapper, method_name)
            result = method(*method_args, **method_kwargs)
            return [result for _ in indices]

        raise AttributeError(f"Method '{method_name}' not found in wrapper or underlying env.")
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        """
        SB3 expects this to exist on VecEnv.
        We do not use Gymnasium wrappers here, so always return False.
        """
        indices = self._get_indices(indices) if hasattr(self, "_get_indices") else (
            list(range(self.num_envs)) if indices is None else ([indices] if isinstance(indices, int) else list(indices))
        )
        return [False for _ in indices]

    def get_images(self):
        """
        Optional method for SB3 VecEnv API.
        Flightmare rendering is handled by Unity; return empty list to satisfy interface.
        """
        return [None for _ in range(self.num_envs)]

    def _get_indices(self, indices):
        """
        Convert indices to a list format.
        :param indices: (int, list, None) Indices to convert
        :return: (list) List of indices
        """
        if indices is None:
            return list(range(self.num_envs))
        elif isinstance(indices, int):
            return [indices]
        else:
            return list(indices)

    def _normalize_obs(self, obs: np.ndarray, obs_rms: RunningMeanStd) -> np.ndarray:
        """
        Helper to normalize observation.
        :param obs: (np.ndarray) Observation to normalize
        :param obs_rms: (RunningMeanStd) Associated statistics
        :return: (np.ndarray) Normalized observation
        """
        return (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)
    
    def _format_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Convert flat C++ observations into the shape expected by SB3.
        """
        if self._is_image_obs:
            obs = np.clip(obs, 0.0, 255.0)
            return obs.reshape(
                self._num_envs, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS
            ).astype(np.uint8)
        return obs.astype(np.float32)


    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize observations using this VecEnv's observation statistics.
        If normalization is disabled, returns observations as-is.
        Calling this method does not update statistics.
        :param obs: (np.ndarray) Observation to normalize
        :return: (np.ndarray) Normalized observation (or raw if normalization disabled)
        """
        if self._is_image_obs:
            return obs.astype(np.float32)
        if not self.use_obs_norm:
            return obs.astype(np.float32)
        return self._normalize_obs(obs, self.obs_rms).astype(np.float32)

    def update_rms(self):
        """
        Update the running mean/std statistics by copying obs_rms_new to obs_rms.
        This should be called periodically (e.g., at the end of each rollout).
        Does nothing if normalization is disabled.
        """
        if not self.use_obs_norm:
            return
        # Copy stats including count
        self.obs_rms.mean = self.obs_rms_new.mean.copy()
        self.obs_rms.var = self.obs_rms_new.var.copy()
        self.obs_rms.count = float(self.obs_rms_new.count)

    def get_obs_norm(self):
        """
        Get current normalization statistics (mean and variance).
        :return: (tuple) (mean, var) tuple of current normalization statistics
        """
        if self.obs_rms is None:
            raise RuntimeError("Observation normalization is disabled.")
        return self.obs_rms.mean, self.obs_rms.var

    def save_rms(self, save_dir: str, n_iter: int) -> None:
        """
        Save normalization statistics to disk.
        :param save_dir: (str) Directory to save statistics to
        :param n_iter: (int) Iteration number for filename
        """
        if self.obs_rms is None:
            return
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"iter_{n_iter:05d}.npz")
        np.savez(
            path,
            mean=self.obs_rms.mean,
            var=self.obs_rms.var,
            count=np.array([self.obs_rms.count], dtype=np.float64),
        )

    def load_rms(self, data_path: str) -> None:
        """
        Load normalization statistics from disk.
        :param data_path: (str) Path to .npz file containing mean, var, and optionally count
        """
        if self.obs_rms is None:
            return
        np_file = np.load(data_path)
        self.obs_rms.mean = np_file["mean"].copy()
        self.obs_rms.var = np_file["var"].copy()
        self.obs_rms.count = float(np_file["count"][0]) if "count" in np_file else 1.0

        # new도 동일하게 맞춰두기
        self.obs_rms_new.mean = self.obs_rms.mean.copy()
        self.obs_rms_new.var = self.obs_rms.var.copy()
        self.obs_rms_new.count = float(self.obs_rms.count)


class ObsNormUpdateCallback(BaseCallback):
    """
    Callback to automatically update observation normalization statistics
    at the end of each rollout during SB3 training.
    
    This matches the pattern used in rpg_baselines where update_rms() is called
    periodically during training. In SB3, this happens at the end of each rollout.
    
    Usage:
        from tonedio_baselines.envs.vec_env_wrapper import ObsNormUpdateCallback
        
        callback = ObsNormUpdateCallback()
        model.learn(total_timesteps=1e6, callback=callback)
    """
    
    def _on_step(self) -> bool:
        """
        Called at each step. We don't need to do anything here,
        just return True to continue training.
        """
        return True
    
    def _on_rollout_end(self) -> None:
        """
        Called at the end of each rollout (after collecting n_steps).
        Updates normalization statistics for all FlightEnvVec instances in the wrapper chain.
        """
        # Find FlightEnvVec in the environment wrapper chain
        env = self.training_env
        while env is not None:
            if isinstance(env, FlightEnvVec):
                env.update_rms()
                break  # Only need to update once
            # Traverse wrapper chain
            if hasattr(env, 'venv'):
                env = env.venv
            elif hasattr(env, 'envs') and len(env.envs) > 0:
                # DummyVecEnv or similar
                env = env.envs[0]
            else:
                break


class CheckpointCallbackWithRMS(CheckpointCallback):
    """
    Extended CheckpointCallback that also saves observation normalization statistics
    whenever a model checkpoint is saved.
    
    Usage:
        from tonedio_baselines.envs.vec_env_wrapper import CheckpointCallbackWithRMS
        
        callback = CheckpointCallbackWithRMS(
            save_freq=10000,
            save_path="./checkpoints",
            name_prefix="ppo_model"
        )
        model.learn(total_timesteps=1e6, callback=callback)
    """
    
    def __init__(self, *args, **kwargs):
        """
        Same parameters as CheckpointCallback.
        """
        super().__init__(*args, **kwargs)
        self._rms_save_counter = 0
    
    def _on_step(self) -> bool:
        """
        Override to save RMS statistics when checkpoint is saved.
        """
        # Check if checkpoint will be saved this step
        will_save = self.n_calls > 0 and self.n_calls % self.save_freq == 0
        
        # Call parent to save checkpoint
        result = super()._on_step()
        
        # Save RMS statistics if checkpoint was saved
        if will_save:
            self._save_rms()
        
        return result
    
    def _save_rms(self) -> None:
        """
        Save normalization statistics to the same directory as checkpoints.
        """
        # Find FlightEnvVec in the environment wrapper chain
        env = self.training_env
        while env is not None:
            if isinstance(env, FlightEnvVec):
                # Save RMS in a subdirectory
                rms_dir = os.path.join(self.save_path, "RMS")
                self._rms_save_counter += 1
                env.save_rms(rms_dir, self._rms_save_counter)
                if self.verbose > 0:
                    print(f"Saved normalization statistics to {rms_dir}/iter_{self._rms_save_counter:05d}.npz")
                break
            # Traverse wrapper chain
            if hasattr(env, 'venv'):
                env = env.venv
            elif hasattr(env, 'envs') and len(env.envs) > 0:
                env = env.envs[0]
            else:
                break
