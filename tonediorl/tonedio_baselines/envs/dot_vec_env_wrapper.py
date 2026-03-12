import os
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


class DotFlightEnvVec(VecEnv):
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
    DOT_UV_DIM = 11
    DOT_POLICY_FEAT_DIM = 10

    def __init__(
        self,
        impl,
        use_obs_norm: bool = True,
        include_prev_action: bool = False,
        stage_switch_enabled: bool = True,
        include_area_obs: bool = True,
        include_shape_obs: bool = True,
    ):
        """
        :param impl: C++ VecEnv implementation (flightgym.QuadrotorEnv_v1)
        :param use_obs_norm: (bool) Whether to use observation normalization. 
                             If False, observations are returned without normalization.
        """
        self.wrapper = impl
        self.use_obs_norm = use_obs_norm
        self.include_prev_action = bool(include_prev_action)
        self.stage_switch_enabled = bool(stage_switch_enabled)
        self.include_area_obs = bool(include_area_obs)
        self.include_shape_obs = bool(include_shape_obs)

        self.num_obs = int(self.wrapper.getObsDim())
        self.num_acts = int(self.wrapper.getActDim())
        self._num_envs = int(self.wrapper.getNumOfEnvs())
        self._extraInfoNames = list(self.wrapper.getExtraInfoNames())
        self._extraInfoNameToIdx = {name: i for i, name in enumerate(self._extraInfoNames)}
        self._reward_obs_indices = []
        area_key = "metric_area" if "metric_area" in self._extraInfoNameToIdx else "reward_area"
        shape_key = "metric_shape2" if "metric_shape2" in self._extraInfoNameToIdx else "reward_shape2"
        if self.include_area_obs and area_key in self._extraInfoNameToIdx:
            self._reward_obs_indices.append(self._extraInfoNameToIdx[area_key])
        if self.include_shape_obs and shape_key in self._extraInfoNameToIdx:
            self._reward_obs_indices.append(self._extraInfoNameToIdx[shape_key])
        self._reward_obs_dim = len(self._reward_obs_indices)
        self._image_dim = self.IMG_HEIGHT * self.IMG_WIDTH * self.IMG_CHANNELS
        self._dot_uv_dim = max(0, self.num_obs - self._image_dim)
        self._is_image_obs = self.num_obs == (
            self._image_dim
        )
        self._is_dot_image_obs = (
            self.num_obs > self._image_dim
            and self._dot_uv_dim >= self.DOT_UV_DIM
            and self._dot_uv_dim % self.DOT_UV_DIM == 0
        )
        if self._is_dot_image_obs:
            self._num_dot_tags = self._dot_uv_dim // self.DOT_UV_DIM
            # stage_switch_enabled=True  -> use all tag features
            # stage_switch_enabled=False -> use first tag features only
            self._policy_num_dot_tags = self._num_dot_tags if self.stage_switch_enabled else 1
            # PPO input excludes tag_id from each tag block (11 -> 10).
            self._policy_dot_uv_dim = self._policy_num_dot_tags * self.DOT_POLICY_FEAT_DIM
        else:
            self._num_dot_tags = 0
            self._policy_num_dot_tags = 0
            self._policy_dot_uv_dim = self._dot_uv_dim
        self._append_prev_action = self.include_prev_action and (not self._is_image_obs)

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
        elif self._is_dot_image_obs:
            # Policy sees all tag UV features when stage switching is enabled.
            policy_dim = self._policy_dot_uv_dim + self._reward_obs_dim + (self.num_acts if self._append_prev_action else 0)
            self._observation_space = spaces.Box(
                low=-np.inf * np.ones(policy_dim, dtype=np.float32),
                high=np.inf * np.ones(policy_dim, dtype=np.float32),
                dtype=np.float32,
            )
        else:
            policy_dim = self.num_obs + (self.num_acts if self._append_prev_action else 0)
            self._observation_space = spaces.Box(
                low=-np.inf * np.ones(policy_dim, dtype=np.float32),
                high=np.inf * np.ones(policy_dim, dtype=np.float32),
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
        self._prev_actions = np.zeros((self._num_envs, self.num_acts), dtype=np.float32)

        self._extraInfo = np.zeros((self._num_envs, len(self._extraInfoNames)), dtype=np.float32)

        # Episode bookkeeping (SB3 uses info["episode"] convention)
        self._ep_rewards = [[] for _ in range(self._num_envs)]

        self.max_episode_steps = 300

        # Observation normalization
        if self.use_obs_norm:
            # Normalize only the observation features that are fed to policy.
            if self._is_dot_image_obs:
                rms_shape = (self._policy_dot_uv_dim,)
            else:
                rms_shape = (self.num_obs,)
            self.obs_rms = RunningMeanStd(shape=rms_shape)
            self.obs_rms_new = RunningMeanStd(shape=rms_shape)
        else:
            self.obs_rms = None
            self.obs_rms_new = None

        print(
            f"[FlightEnvVecSB3] num_envs={self._num_envs}, "
            f"raw_obs_dim={self.num_obs}, policy_obs_shape={self._observation_space.shape}, "
            f"dot_uv_dim={self._dot_uv_dim if self._is_dot_image_obs else 0}, "
            f"policy_dot_uv_dim={self._policy_dot_uv_dim if self._is_dot_image_obs else 0}, "
            f"reward_obs_dim={self._reward_obs_dim if self._is_dot_image_obs else 0}, "
            f"include_area_obs={self.include_area_obs}, "
            f"include_shape_obs={self.include_shape_obs}, "
            f"stage_switch_enabled={self.stage_switch_enabled}, "
            f"act_dim={self.num_acts}, use_obs_norm={self.use_obs_norm}, "
            f"include_prev_action={self._append_prev_action}"
        )

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
        self._prev_actions[:] = 0.0
        self._extraInfo[:] = 0.0
        # Flightmare fills the provided obs buffer
        self.wrapper.reset(self._observation)
        # Update normalization statistics (if enabled)
        if self.use_obs_norm:
            self.obs_rms_new.update(self._policy_base_obs(self._observation))
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
            self.obs_rms_new.update(self._policy_base_obs(self._observation))

        # infos: extra_info만 넣어줌 (episode는 VecMonitor가 처리)
        if len(self._extraInfoNames) > 0:
            infos = [
                {"extra_info": {self._extraInfoNames[j]: float(self._extraInfo[i, j])
                                for j in range(len(self._extraInfoNames))}}
                for i in range(self._num_envs)
            ]
        else:
            infos = [{} for _ in range(self._num_envs)]

        # Returned observation can include previous action (action from the just-finished step).
        if self._append_prev_action:
            self._prev_actions = self._actions.copy().astype(np.float32)
            if np.any(self._done):
                self._prev_actions[self._done] = 0.0

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
        if self._is_dot_image_obs:
            # `obs` may be raw C++ observation or already-extracted policy features.
            if obs.ndim == 2 and obs.shape[1] == self._policy_dot_uv_dim:
                policy_obs = obs.astype(np.float32)
            else:
                policy_obs = self._extract_policy_dot_obs(obs)
            if self._reward_obs_dim > 0:
                reward_obs = self._extraInfo[:, self._reward_obs_indices].astype(np.float32)
                policy_obs = np.concatenate([policy_obs, reward_obs], axis=1).astype(np.float32)
            if self._append_prev_action:
                policy_obs = np.concatenate([policy_obs, self._prev_actions], axis=1).astype(np.float32)
            return policy_obs
        policy_obs = obs.astype(np.float32)
        if self._append_prev_action:
            policy_obs = np.concatenate([policy_obs, self._prev_actions], axis=1).astype(np.float32)
        return policy_obs

    def _policy_base_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Extract policy observation features before optional previous-action append.
        """
        if self._is_image_obs:
            return obs.astype(np.float32)
        if self._is_dot_image_obs:
            return self._extract_policy_dot_obs(obs)
        return obs.astype(np.float32)

    def _extract_policy_dot_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Extract dot observations for policy and remove tag_id from each tag block.
        Raw tag block format is [center_x, center_y, c0x, c0y, c1x, c1y, c2x, c2y, c3x, c3y, tag_id].
        """
        if obs.ndim != 2:
            raise ValueError(f"Expected batched obs with shape (n_envs, dim), got {obs.shape}")
        raw_dot_dim = self._policy_num_dot_tags * self.DOT_UV_DIM
        if obs.shape[1] < raw_dot_dim:
            raise ValueError(
                f"Obs dim too small for dot extraction: got {obs.shape[1]}, need at least {raw_dot_dim}"
            )
        dot_obs = obs[:, :raw_dot_dim].astype(np.float32)
        dot_obs = dot_obs.reshape(obs.shape[0], self._policy_num_dot_tags, self.DOT_UV_DIM)
        # Keep first 10 values per tag (drop tag_id at index 10).
        dot_obs = dot_obs[:, :, :self.DOT_POLICY_FEAT_DIM]
        return dot_obs.reshape(obs.shape[0], self._policy_dot_uv_dim).astype(np.float32)


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
        if self._is_dot_image_obs:
            policy_obs = self._extract_policy_dot_obs(obs)
            if not self.use_obs_norm:
                return policy_obs
            # Dot UV observations are image-plane pixel coordinates in [0, 83].
            # Use fixed scaling for policy input normalization.
            return (policy_obs / 83.0).astype(np.float32)
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
        from tonedio_baselines.envs.dot_vec_env_wrapper import ObsNormUpdateCallback
        
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
        Updates normalization statistics for all DotFlightEnvVec instances in the wrapper chain.
        """
        # Find DotFlightEnvVec in the environment wrapper chain
        env = self.training_env
        while env is not None:
            if isinstance(env, DotFlightEnvVec):
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
        from tonedio_baselines.envs.dot_vec_env_wrapper import CheckpointCallbackWithRMS
        
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
        # Find DotFlightEnvVec in the environment wrapper chain
        env = self.training_env
        while env is not None:
            if isinstance(env, DotFlightEnvVec):
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


# Backward-compatible alias (same naming as other wrappers)
FlightEnvVec = DotFlightEnvVec
