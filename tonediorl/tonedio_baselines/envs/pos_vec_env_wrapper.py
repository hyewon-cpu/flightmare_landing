import os
import numpy as np
from typing import List
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


class PosFlightEnvVec(VecEnv):
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
    TAG_UV_DIM = 11
    TAG_POLICY_FEAT_DIM = 10
    FIXED_TAG_NORM_DIM = 10
    TAG_DIFF_FIXED_SCALE = 84.0
    GOAL_POS_NORM_MEAN = np.array([5.5, 6.5, 40.0], dtype=np.float32)

    @staticmethod
    def _parse_nstep_lags(include_nstep_pos_obs) -> List[int]:
        """
        Parse n-step lag configuration.
        Supported formats:
          - int: 5
          - str: "5,10" or "5 10"
          - list/tuple/set: [5, 10]
        Returns unique positive lags preserving input order.
        """
        lags = []
        if include_nstep_pos_obs is None:
            return lags
        if isinstance(include_nstep_pos_obs, (int, np.integer)):
            val = int(include_nstep_pos_obs)
            return [val] if val > 0 else []
        if isinstance(include_nstep_pos_obs, str):
            tokens = include_nstep_pos_obs.replace(",", " ").split()
            for tok in tokens:
                try:
                    val = int(tok)
                except ValueError:
                    continue
                if val > 0:
                    lags.append(val)
        elif isinstance(include_nstep_pos_obs, (list, tuple, set)):
            for item in include_nstep_pos_obs:
                try:
                    val = int(item)
                except (TypeError, ValueError):
                    continue
                if val > 0:
                    lags.append(val)
        else:
            try:
                val = int(include_nstep_pos_obs)
            except (TypeError, ValueError):
                val = 0
            if val > 0:
                lags.append(val)
        unique_lags = []
        seen = set()
        for lag in lags:
            if lag not in seen:
                seen.add(lag)
                unique_lags.append(lag)
        return unique_lags

    def __init__(
        self,
        impl,
        use_obs_norm: bool = True,
        include_tag_obs: bool = True,
        include_tag_vel_obs: bool = False,
        include_tag_diff_obs: bool = False,
        include_prev_action: int = 0,
        include_prev_tag_obs: int = 0,
        include_prev_pos_obs: int = 0,
        include_nstep_pos_obs: int = 0,
        include_area_obs: bool = True,
        include_shape_obs: bool = True,
        include_tag_id_obs: bool = False,
        include_real_p_c_obs: bool = True,
        include_imu_obs: bool = False,
        include_drone_pos_obs: bool = False,
        include_drone_z_obs: bool = False,
        include_drone_pos_diff_obs: bool = False,
        include_drone_ori_obs: bool = False,
        include_drone_ori_diff_obs: bool = False,
        include_drone_vel_obs: bool = False,
        include_drone_ang_vel_obs: bool = False,
        goal_center_pos_nstep_norm: bool = False,
    ):
        """
        :param impl: C++ VecEnv implementation (flightgym.QuadrotorEnv_v1)
        :param use_obs_norm: (bool) Whether to use observation normalization. 
                             If False, observations are returned without normalization.
        """
        self.wrapper = impl
        self.use_obs_norm = use_obs_norm
        self.include_tag_obs = bool(include_tag_obs)
        self.include_tag_vel_obs = bool(include_tag_vel_obs)
        self.include_tag_diff_obs = bool(include_tag_diff_obs)
        self.prev_action_history_len = max(0, int(include_prev_action))
        self.prev_tag_obs_history_len = max(0, int(include_prev_tag_obs))
        self.prev_pos_obs_history_len = max(0, int(include_prev_pos_obs))
        self.nstep_pos_lags = self._parse_nstep_lags(include_nstep_pos_obs)
        self.curr_nstep_pos_lag = max(self.nstep_pos_lags) if len(self.nstep_pos_lags) > 0 else 0
        self.include_prev_tag_obs = self.prev_tag_obs_history_len > 0
        self.include_prev_pos_obs = self.prev_pos_obs_history_len > 0
        self.include_nstep_pos_obs = len(self.nstep_pos_lags) > 0
        self.include_area_obs = bool(include_area_obs)
        self.include_shape_obs = bool(include_shape_obs)
        self.include_tag_id_obs = bool(include_tag_id_obs)
        self.include_real_p_c_obs = bool(include_real_p_c_obs)
        self.include_drone_pos_obs = bool(include_drone_pos_obs)
        self.include_drone_z_obs = bool(include_drone_z_obs)
        self.include_drone_pos_diff_obs = bool(include_drone_pos_diff_obs)
        # Backward compatibility: include_imu_obs acts as orientation obs toggle alias.
        self.include_drone_ori_obs = bool(include_drone_ori_obs) or bool(include_imu_obs)
        self.include_drone_ori_diff_obs = bool(include_drone_ori_diff_obs)
        self.include_drone_vel_obs = bool(include_drone_vel_obs)
        self.include_drone_ang_vel_obs = bool(include_drone_ang_vel_obs)
        self.goal_center_pos_nstep_norm = bool(goal_center_pos_nstep_norm)

        self.num_obs = int(self.wrapper.getObsDim())
        self.num_acts = int(self.wrapper.getActDim())
        self._num_envs = int(self.wrapper.getNumOfEnvs())
        self._extraInfoNames = list(self.wrapper.getExtraInfoNames())
        self._extraInfoNameToIdx = {name: i for i, name in enumerate(self._extraInfoNames)}
        self._reward_obs_indices = []
        self._pc_obs_indices = []
        self._prev_pos_obs_indices = []
        self._drone_pos_obs_indices = []
        self._drone_z_obs_indices = []
        self._drone_ori_obs_indices = []
        self._drone_vel_obs_indices = []
        self._drone_ang_vel_obs_indices = []
        self._curr_nstep_pos_indices = []
        area_key = "metric_area" if "metric_area" in self._extraInfoNameToIdx else "reward_area"
        shape_key = "metric_shape2" if "metric_shape2" in self._extraInfoNameToIdx else "reward_shape2"
        if self.include_real_p_c_obs:
            for key in ("real_p_c_x", "real_p_c_y", "real_p_c_z"):
                if key in self._extraInfoNameToIdx:
                    self._pc_obs_indices.append(self._extraInfoNameToIdx[key])
        if self.include_prev_pos_obs:
            for key in ("drone_pos_x", "drone_pos_y", "drone_pos_z"):
                if key in self._extraInfoNameToIdx:
                    self._prev_pos_obs_indices.append(self._extraInfoNameToIdx[key])
        if self.include_drone_pos_obs or self.include_drone_pos_diff_obs:
            for key in ("drone_pos_x", "drone_pos_y", "drone_pos_z"):
                if key in self._extraInfoNameToIdx:
                    self._drone_pos_obs_indices.append(self._extraInfoNameToIdx[key])
        if self.include_nstep_pos_obs:
            for key in ("drone_pos_x", "drone_pos_y", "drone_pos_z"):
                if key in self._extraInfoNameToIdx:
                    self._curr_nstep_pos_indices.append(self._extraInfoNameToIdx[key])
        if self.include_drone_z_obs:
            if "drone_pos_z" in self._extraInfoNameToIdx:
                self._drone_z_obs_indices.append(self._extraInfoNameToIdx["drone_pos_z"])
        if self.include_drone_ori_obs or self.include_drone_ori_diff_obs:
            for key in ("imu_roll", "imu_pitch", "imu_yaw"):
                if key in self._extraInfoNameToIdx:
                    self._drone_ori_obs_indices.append(self._extraInfoNameToIdx[key])
        if self.include_drone_vel_obs:
            for key in ("drone_vel_x", "drone_vel_y", "drone_vel_z"):
                if key in self._extraInfoNameToIdx:
                    self._drone_vel_obs_indices.append(self._extraInfoNameToIdx[key])
        if self.include_drone_ang_vel_obs:
            for key in ("drone_ang_vel_x", "drone_ang_vel_y", "drone_ang_vel_z"):
                if key in self._extraInfoNameToIdx:
                    self._drone_ang_vel_obs_indices.append(self._extraInfoNameToIdx[key])
        if self.include_area_obs and area_key in self._extraInfoNameToIdx:
            self._reward_obs_indices.append(self._extraInfoNameToIdx[area_key])
        if self.include_shape_obs and shape_key in self._extraInfoNameToIdx:
            self._reward_obs_indices.append(self._extraInfoNameToIdx[shape_key])
        self._reward_obs_dim = len(self._reward_obs_indices)
        self._pc_obs_dim = len(self._pc_obs_indices)
        self._prev_pos_obs_step_dim = len(self._prev_pos_obs_indices)
        self._drone_pos_obs_dim = len(self._drone_pos_obs_indices)
        self._drone_z_obs_dim = len(self._drone_z_obs_indices)
        self._drone_pos_diff_obs_dim = (
            len(self._drone_pos_obs_indices) if self.include_drone_pos_diff_obs else 0
        )
        self._drone_ori_obs_dim = len(self._drone_ori_obs_indices)
        self._drone_ori_diff_obs_dim = (
            len(self._drone_ori_obs_indices) if self.include_drone_ori_diff_obs else 0
        )
        self._drone_vel_obs_dim = len(self._drone_vel_obs_indices)
        self._drone_ang_vel_obs_dim = len(self._drone_ang_vel_obs_indices)
        self._curr_nstep_pos_single_dim = (
            len(self._curr_nstep_pos_indices)
            if (self.include_nstep_pos_obs and len(self._curr_nstep_pos_indices) == 3)
            else 0
        )
        self._curr_nstep_pos_obs_dim = self._curr_nstep_pos_single_dim * len(self.nstep_pos_lags)
        self._curr_nstep_pos_hist_len = (
            (max(self.nstep_pos_lags) + 1) if self._curr_nstep_pos_obs_dim > 0 else 0
        )
        self._image_dim = self.IMG_HEIGHT * self.IMG_WIDTH * self.IMG_CHANNELS
        self._tag_uv_dim = max(0, self.num_obs - self._image_dim)
        self._is_image_obs = self.num_obs == (
            self._image_dim
        )
        self._tag_aux_dim = 0
        self._is_tag_image_obs = (
            self.num_obs > self._image_dim
            and self._tag_uv_dim >= self.TAG_UV_DIM
        )
        if self._is_tag_image_obs:
            self._num_tags = self._tag_uv_dim // self.TAG_UV_DIM
            self._tag_aux_dim = self._tag_uv_dim - (self._num_tags * self.TAG_UV_DIM)
            if self._num_tags < 1:
                self._is_tag_image_obs = False
                self._tag_aux_dim = 0
        if self._is_tag_image_obs:
            # Policy uses only the first QR tag block.
            self._policy_num_tags = 1
            self._policy_tag_feat_dim = self.TAG_UV_DIM if self.include_tag_id_obs else self.TAG_POLICY_FEAT_DIM
            self._policy_tag_uv_dim = self._policy_num_tags * self._policy_tag_feat_dim
        else:
            self._num_tags = 0
            self._policy_num_tags = 0
            self._policy_tag_feat_dim = self._tag_uv_dim
            self._policy_tag_uv_dim = self._tag_uv_dim
        self._policy_tag_block_dim = (
            (self._policy_tag_uv_dim + self._tag_aux_dim)
            if (self._is_tag_image_obs and self.include_tag_obs)
            else 0
        )
        self._tag_vel_obs_dim = (
            self._policy_tag_block_dim
            if (self._is_tag_image_obs and self.include_tag_obs and self.include_tag_vel_obs)
            else 0
        )
        self._tag_diff_obs_dim = (
            self._policy_tag_block_dim
            if (self._is_tag_image_obs and self.include_tag_obs and self.include_tag_diff_obs)
            else 0
        )
        self._prev_tag_obs_step_dim = (
            self._policy_tag_block_dim
            if (self._is_tag_image_obs and self.include_tag_obs)
            else 0
        )
        self._prev_tag_obs_dim = self._prev_tag_obs_step_dim * self.prev_tag_obs_history_len
        self._tag_diff_obs_start = -1
        self._tag_diff_obs_end = -1
        if self._tag_diff_obs_dim > 0:
            # policy_obs layout (tag mode):
            # [tag_block, prev_tag_hist, tag_vel, tag_diff, ...]
            self._tag_diff_obs_start = (
                self._policy_tag_block_dim
                + self._prev_tag_obs_dim
                + self._tag_vel_obs_dim
            )
            self._tag_diff_obs_end = self._tag_diff_obs_start + self._tag_diff_obs_dim
        self._prev_pos_obs_dim = self._prev_pos_obs_step_dim * self.prev_pos_obs_history_len
        # Legacy compatibility:
        # Some older C++ builds may expose raw obs as pure 3D/12D drone state.
        # For those modes, skip legacy fixed /83 scaling and use RMS only (if enabled).
        self._is_drone_xyz_only_obs = (
            (not self._is_image_obs)
            and (not self._is_tag_image_obs)
            and (self.num_obs == 3)
        )
        self._is_drone_state12_only_obs = (
            (not self._is_image_obs)
            and (not self._is_tag_image_obs)
            and (self.num_obs == 12)
        )
        self._skip_fixed_tag_scaling = (
            (not self.include_tag_obs)
            or
            self._is_drone_xyz_only_obs or self._is_drone_state12_only_obs
        )
        self._append_prev_action = self.prev_action_history_len > 0 and (not self._is_image_obs)
        self._prev_action_obs_dim = self.prev_action_history_len * self.num_acts if self._append_prev_action else 0
        self._goal_center_pos_nstep_norm = self.goal_center_pos_nstep_norm
        self._goal_center_pos_indices = []
        self._goal_center_nstep_index_groups = []

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
        elif self._is_tag_image_obs:
            policy_dim = (
                self._policy_tag_block_dim
                + self._tag_vel_obs_dim
                + self._tag_diff_obs_dim
                + self._prev_tag_obs_dim
                + self._reward_obs_dim
                + self._prev_action_obs_dim
                + self._pc_obs_dim
                + self._drone_pos_obs_dim
                + self._drone_z_obs_dim
                + self._drone_pos_diff_obs_dim
                + self._prev_pos_obs_dim
                + self._drone_ori_obs_dim
                + self._drone_ori_diff_obs_dim
                + self._drone_vel_obs_dim
                + self._drone_ang_vel_obs_dim
                + self._curr_nstep_pos_obs_dim
            )
            drone_pos_start = (
                self._policy_tag_block_dim
                + self._prev_tag_obs_dim
                + self._tag_vel_obs_dim
                + self._tag_diff_obs_dim
                + self._reward_obs_dim
                + self._prev_action_obs_dim
                + self._pc_obs_dim
            )
            if self._drone_pos_obs_dim == 3:
                self._goal_center_pos_indices = [drone_pos_start + 0, drone_pos_start + 1, drone_pos_start + 2]
            if self._curr_nstep_pos_single_dim == 3 and len(self.nstep_pos_lags) > 0:
                nstep_start = (
                    drone_pos_start
                    + self._drone_pos_obs_dim
                    + self._drone_z_obs_dim
                    + self._drone_pos_diff_obs_dim
                    + self._prev_pos_obs_dim
                    + self._drone_ori_obs_dim
                    + self._drone_ori_diff_obs_dim
                    + self._drone_vel_obs_dim
                    + self._drone_ang_vel_obs_dim
                )
                self._goal_center_nstep_index_groups = [
                    [nstep_start + i * 3 + 0, nstep_start + i * 3 + 1, nstep_start + i * 3 + 2]
                    for i in range(len(self.nstep_pos_lags))
                ]
            self._observation_space = spaces.Box(
                low=-np.inf * np.ones(policy_dim, dtype=np.float32),
                high=np.inf * np.ones(policy_dim, dtype=np.float32),
                dtype=np.float32,
            )
        else:
            policy_dim = self.num_obs + self._prev_action_obs_dim + self._curr_nstep_pos_obs_dim
            if self.num_obs >= 3:
                self._goal_center_pos_indices = [0, 1, 2]
            if self._curr_nstep_pos_single_dim == 3 and len(self.nstep_pos_lags) > 0:
                nstep_start = self.num_obs + self._prev_action_obs_dim
                self._goal_center_nstep_index_groups = [
                    [nstep_start + i * 3 + 0, nstep_start + i * 3 + 1, nstep_start + i * 3 + 2]
                    for i in range(len(self.nstep_pos_lags))
                ]
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
        self._prev_actions = np.zeros(
            (self._num_envs, self.prev_action_history_len, self.num_acts), dtype=np.float32
        )
        self._prev_tag_obs = np.zeros(
            (self._num_envs, self.prev_tag_obs_history_len, self._prev_tag_obs_step_dim), dtype=np.float32
        )
        self._last_tag_block = np.zeros((self._num_envs, self._policy_tag_block_dim), dtype=np.float32)
        self._has_last_tag_block = np.zeros((self._num_envs,), dtype=bool)
        self._prev_pos_obs = np.zeros(
            (self._num_envs, self.prev_pos_obs_history_len, self._prev_pos_obs_step_dim), dtype=np.float32
        )
        self._curr_nstep_pos_hist = np.zeros(
            (self._num_envs, self._curr_nstep_pos_hist_len, len(self._curr_nstep_pos_indices)), dtype=np.float32
        )
        self._last_drone_pos_obs = np.zeros((self._num_envs, self._drone_pos_obs_dim), dtype=np.float32)
        self._has_last_drone_pos_obs = np.zeros((self._num_envs,), dtype=bool)
        self._last_drone_ori_obs = np.zeros((self._num_envs, self._drone_ori_obs_dim), dtype=np.float32)
        self._has_last_drone_ori_obs = np.zeros((self._num_envs,), dtype=bool)

        self._extraInfo = np.zeros((self._num_envs, len(self._extraInfoNames)), dtype=np.float32)
        # Use simulator dt when available; otherwise use Flightmare's default sim_dt.
        self._tag_vel_dt = 0.02
        get_sim_dt_fn = getattr(self.wrapper, "getSimTimeStep", None)
        if callable(get_sim_dt_fn):
            try:
                sim_dt = float(get_sim_dt_fn())
                if sim_dt > 0.0:
                    self._tag_vel_dt = sim_dt
            except Exception:
                pass

        # Episode bookkeeping (SB3 uses info["episode"] convention)
        self._ep_rewards = [[] for _ in range(self._num_envs)]

        self.max_episode_steps = 300

        # Observation normalization
        policy_obs_dim = int(self._observation_space.shape[0]) if len(self._observation_space.shape) > 0 else 0
        if self.use_obs_norm:
            self._fixed_tag_norm_dim = (
                0
                if self._skip_fixed_tag_scaling
                else min(self.FIXED_TAG_NORM_DIM, policy_obs_dim)
            )
            rms_mask = np.ones(policy_obs_dim, dtype=bool)
            if self._fixed_tag_norm_dim > 0:
                rms_mask[:self._fixed_tag_norm_dim] = False
            if self._tag_diff_obs_dim > 0 and self._tag_diff_obs_start >= 0:
                rms_mask[self._tag_diff_obs_start:self._tag_diff_obs_end] = False
            self._rms_indices = np.flatnonzero(rms_mask).astype(np.int64)
            self._policy_to_rms = np.full((policy_obs_dim,), -1, dtype=np.int64)
            self._policy_to_rms[self._rms_indices] = np.arange(self._rms_indices.shape[0], dtype=np.int64)
            rms_shape = (int(self._rms_indices.shape[0]),)
            self.obs_rms = RunningMeanStd(shape=rms_shape)
            self.obs_rms_new = RunningMeanStd(shape=rms_shape)
        else:
            self._fixed_tag_norm_dim = (
                0
                if self._skip_fixed_tag_scaling
                else min(
                    self.FIXED_TAG_NORM_DIM,
                    policy_obs_dim,
                )
            )
            self._rms_indices = np.zeros((0,), dtype=np.int64)
            self._policy_to_rms = np.zeros((policy_obs_dim,), dtype=np.int64)
            self.obs_rms = None
            self.obs_rms_new = None

        print(
            f"[FlightEnvVecSB3] num_envs={self._num_envs}, "
            f"raw_obs_dim={self.num_obs}, policy_obs_shape={self._observation_space.shape}, "
            f"tag_uv_dim={self._tag_uv_dim if self._is_tag_image_obs else 0}, "
            f"tag_aux_dim={self._tag_aux_dim if self._is_tag_image_obs else 0}, "
            f"policy_tag_uv_dim={self._policy_tag_uv_dim if self._is_tag_image_obs else 0}, "
            f"include_tag_obs={self.include_tag_obs}, "
            f"include_tag_vel_obs={self.include_tag_vel_obs}, "
            f"tag_vel_obs_dim={self._tag_vel_obs_dim if self._is_tag_image_obs else 0}, "
            f"include_tag_diff_obs={self.include_tag_diff_obs}, "
            f"tag_diff_obs_dim={self._tag_diff_obs_dim if self._is_tag_image_obs else 0}, "
            f"prev_tag_obs_dim={self._prev_tag_obs_dim if self._is_tag_image_obs else 0}, "
            f"prev_tag_obs_history_len={self.prev_tag_obs_history_len}, "
            f"reward_obs_dim={self._reward_obs_dim if self._is_tag_image_obs else 0}, "
            f"prev_pos_obs_dim={self._prev_pos_obs_dim if self._is_tag_image_obs else 0}, "
            f"prev_pos_obs_history_len={self.prev_pos_obs_history_len}, "
            f"pc_obs_dim={self._pc_obs_dim if self._is_tag_image_obs else 0}, "
            f"drone_pos_obs_dim={self._drone_pos_obs_dim if self._is_tag_image_obs else 0}, "
            f"drone_z_obs_dim={self._drone_z_obs_dim if self._is_tag_image_obs else 0}, "
            f"drone_pos_diff_obs_dim={self._drone_pos_diff_obs_dim if self._is_tag_image_obs else 0}, "
            f"drone_ori_obs_dim={self._drone_ori_obs_dim if self._is_tag_image_obs else 0}, "
            f"drone_ori_diff_obs_dim={self._drone_ori_diff_obs_dim if self._is_tag_image_obs else 0}, "
            f"drone_vel_obs_dim={self._drone_vel_obs_dim if self._is_tag_image_obs else 0}, "
            f"drone_ang_vel_obs_dim={self._drone_ang_vel_obs_dim if self._is_tag_image_obs else 0}, "
            f"curr_nstep_pos_obs_dim={self._curr_nstep_pos_obs_dim if (not self._is_image_obs) else 0}, "
            f"include_area_obs={self.include_area_obs}, "
            f"include_shape_obs={self.include_shape_obs}, "
            f"include_tag_id_obs={self.include_tag_id_obs}, "
            f"include_real_p_c_obs={self.include_real_p_c_obs}, "
            f"include_prev_tag_obs={self.include_prev_tag_obs}, "
            f"include_prev_pos_obs={self.include_prev_pos_obs}, "
            f"include_drone_pos_obs={self.include_drone_pos_obs}, "
            f"include_drone_z_obs={self.include_drone_z_obs}, "
            f"include_drone_pos_diff_obs={self.include_drone_pos_diff_obs}, "
            f"include_drone_ori_obs={self.include_drone_ori_obs}, "
            f"include_drone_ori_diff_obs={self.include_drone_ori_diff_obs}, "
            f"include_drone_vel_obs={self.include_drone_vel_obs}, "
            f"include_drone_ang_vel_obs={self.include_drone_ang_vel_obs}, "
            f"include_nstep_pos_obs={self.include_nstep_pos_obs}, "
            f"nstep_pos_lags={self.nstep_pos_lags}, "
            f"drone_xyz_only_obs={self._is_drone_xyz_only_obs}, "
            f"drone_state12_only_obs={self._is_drone_state12_only_obs}, "
            f"act_dim={self.num_acts}, use_obs_norm={self.use_obs_norm}, "
            f"prev_action_history_len={self.prev_action_history_len}"
        )
        if self.include_prev_tag_obs:
            current_tag_mode = "fixed_/83_then_no_rms_for_first10" if self.include_tag_obs else "disabled"
            prev_tag_mode = (
                "fixed_/83_then_no_rms_for_first10" if (self.include_tag_obs and self._prev_tag_obs_dim > 0) else "disabled"
            )
            print(
                f"[NormDebug] current_tag_mode={current_tag_mode}, "
                f"prev_tag_mode={prev_tag_mode}, "
                f"shared_scheme={str(current_tag_mode == prev_tag_mode)}"
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
        if self._prev_tag_obs_dim > 0:
            self._prev_tag_obs[:] = 0.0
        if self._policy_tag_block_dim > 0:
            self._last_tag_block[:] = 0.0
            self._has_last_tag_block[:] = False
        if self._prev_pos_obs_dim > 0:
            self._prev_pos_obs[:] = 0.0
        if self._curr_nstep_pos_obs_dim > 0:
            self._curr_nstep_pos_hist[:] = 0.0
        self._last_drone_pos_obs[:] = 0.0
        self._has_last_drone_pos_obs[:] = False
        self._last_drone_ori_obs[:] = 0.0
        self._has_last_drone_ori_obs[:] = False
        self._extraInfo[:] = 0.0
        # Flightmare fills the provided obs buffer
        self.wrapper.reset(self._observation)
        if self._policy_tag_block_dim > 0:
            current_tag_block = self._extract_current_tag_block(self._observation)
            self._last_tag_block[:, :] = current_tag_block
            self._has_last_tag_block[:] = True
        if self._curr_nstep_pos_obs_dim > 0:
            curr_pos = self._extraInfo[:, self._curr_nstep_pos_indices].astype(np.float32)
            self._curr_nstep_pos_hist[:, :, :] = curr_pos[:, None, :]
        policy_obs = self._format_obs(self._observation)
        if self.use_obs_norm:
            rms_tail = self._policy_rms_tail(policy_obs)
            if rms_tail.shape[1] > 0:
                self.obs_rms_new.update(rms_tail)
        return self.normalize_obs(policy_obs)

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

        # infos: extra_info만 넣어줌 (episode는 VecMonitor가 처리)
        if len(self._extraInfoNames) > 0:
            infos = [
                {"extra_info": {self._extraInfoNames[j]: float(self._extraInfo[i, j])
                                for j in range(len(self._extraInfoNames))}}
                for i in range(self._num_envs)
            ]
        else:
            infos = [{} for _ in range(self._num_envs)]

        # Returned observation can include a configurable history of previous actions.
        if self._append_prev_action:
            self._prev_actions[:, 1:, :] = self._prev_actions[:, :-1, :]
            self._prev_actions[:, 0, :] = self._actions.astype(np.float32)
            if np.any(self._done):
                self._prev_actions[self._done] = 0.0

        policy_obs = self._format_obs(self._observation)
        if self.use_obs_norm:
            rms_tail = self._policy_rms_tail(policy_obs)
            if rms_tail.shape[1] > 0:
                self.obs_rms_new.update(rms_tail)

        obs = self.normalize_obs(policy_obs)
        rews = self._reward.copy()
        dones = self._done.copy()

        if self._prev_tag_obs_dim > 0:
            current_tag_block = self._extract_current_tag_block(self._observation)
            self._prev_tag_obs[:, 1:, :] = self._prev_tag_obs[:, :-1, :]
            self._prev_tag_obs[:, 0, :] = current_tag_block
            if np.any(self._done):
                self._prev_tag_obs[self._done] = 0.0
        if self._policy_tag_block_dim > 0:
            current_tag_block = self._extract_current_tag_block(self._observation)
            self._last_tag_block[:, :] = current_tag_block
            self._has_last_tag_block[:] = True
            if np.any(self._done):
                self._last_tag_block[self._done] = 0.0
                self._has_last_tag_block[self._done] = False
        if self._prev_pos_obs_dim > 0:
            current_pos_block = self._extraInfo[:, self._prev_pos_obs_indices].astype(np.float32)
            self._prev_pos_obs[:, 1:, :] = self._prev_pos_obs[:, :-1, :]
            self._prev_pos_obs[:, 0, :] = current_pos_block
            if np.any(self._done):
                self._prev_pos_obs[self._done] = 0.0
        if self._drone_pos_diff_obs_dim > 0:
            current_drone_pos_obs = self._extraInfo[:, self._drone_pos_obs_indices].astype(np.float32)
            self._last_drone_pos_obs[:, :] = current_drone_pos_obs
            self._has_last_drone_pos_obs[:] = True
            if np.any(self._done):
                self._last_drone_pos_obs[self._done] = 0.0
                self._has_last_drone_pos_obs[self._done] = False
        if self._drone_ori_diff_obs_dim > 0:
            current_drone_ori_obs = self._extraInfo[:, self._drone_ori_obs_indices].astype(np.float32)
            self._last_drone_ori_obs[:, :] = current_drone_ori_obs
            self._has_last_drone_ori_obs[:] = True
            if np.any(self._done):
                self._last_drone_ori_obs[self._done] = 0.0
                self._has_last_drone_ori_obs[self._done] = False
        if self._curr_nstep_pos_obs_dim > 0:
            curr_pos = self._extraInfo[:, self._curr_nstep_pos_indices].astype(np.float32)
            self._curr_nstep_pos_hist[:, 1:, :] = self._curr_nstep_pos_hist[:, :-1, :]
            self._curr_nstep_pos_hist[:, 0, :] = curr_pos
            if np.any(self._done):
                self._curr_nstep_pos_hist[self._done] = curr_pos[self._done][:, None, :]

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
        if self._is_tag_image_obs:
            # `obs` may be raw C++ observation or already-extracted policy features.
            base_policy_dim = self._policy_tag_block_dim
            if obs.ndim == 2 and obs.shape[1] == base_policy_dim:
                policy_obs = obs.astype(np.float32)
            else:
                if self.include_tag_obs:
                    policy_obs = self._extract_policy_tag_obs(obs) #태그 부분만 분리
                    if self._tag_aux_dim > 0:
                        tag_aux_obs = self._extract_policy_tag_aux_obs(obs)
                        policy_obs = np.concatenate([policy_obs, tag_aux_obs], axis=1).astype(np.float32)
                else:
                    policy_obs = np.zeros((obs.shape[0], 0), dtype=np.float32)
            if self._prev_tag_obs_dim > 0:
                policy_obs = np.concatenate([policy_obs, self._flatten_prev_tag_obs()], axis=1).astype(np.float32)
            if self._tag_vel_obs_dim > 0:
                policy_obs = np.concatenate([policy_obs, self._extract_tag_vel_obs(obs)], axis=1).astype(np.float32)
            if self._tag_diff_obs_dim > 0:
                policy_obs = np.concatenate([policy_obs, self._extract_tag_diff_obs(obs)], axis=1).astype(np.float32)
            if self._reward_obs_dim > 0: #reward_obs_dim : 태그 말고 더 붙일 obs 개수 
                reward_obs = self._extraInfo[:, self._reward_obs_indices].astype(np.float32)
                policy_obs = np.concatenate([policy_obs, reward_obs], axis=1).astype(np.float32)
            if self._append_prev_action:
                policy_obs = np.concatenate([policy_obs, self._flatten_prev_actions()], axis=1).astype(np.float32)
            if self._pc_obs_dim > 0: #p_C 붙일거면 
                pc_obs = self._extraInfo[:, self._pc_obs_indices].astype(np.float32)
                policy_obs = np.concatenate([policy_obs, pc_obs], axis=1).astype(np.float32)
            if self._drone_pos_obs_dim > 0:
                drone_pos_obs = self._extraInfo[:, self._drone_pos_obs_indices].astype(np.float32)
                policy_obs = np.concatenate([policy_obs, drone_pos_obs], axis=1).astype(np.float32)
            if self._drone_z_obs_dim > 0:
                drone_z_obs = self._extraInfo[:, self._drone_z_obs_indices].astype(np.float32)
                policy_obs = np.concatenate([policy_obs, drone_z_obs], axis=1).astype(np.float32)
            if self._drone_pos_diff_obs_dim > 0:
                drone_pos_diff_obs = self._extract_drone_pos_diff_obs()
                policy_obs = np.concatenate([policy_obs, drone_pos_diff_obs], axis=1).astype(np.float32)
            if self._prev_pos_obs_dim > 0:
                policy_obs = np.concatenate([policy_obs, self._flatten_prev_pos_obs()], axis=1).astype(np.float32)
            if self._drone_ori_obs_dim > 0:
                drone_ori_obs = self._extraInfo[:, self._drone_ori_obs_indices].astype(np.float32)
                policy_obs = np.concatenate([policy_obs, drone_ori_obs], axis=1).astype(np.float32)
            if self._drone_ori_diff_obs_dim > 0:
                drone_ori_diff_obs = self._extract_drone_ori_diff_obs()
                policy_obs = np.concatenate([policy_obs, drone_ori_diff_obs], axis=1).astype(np.float32)
            if self._drone_vel_obs_dim > 0:
                drone_vel_obs = self._extraInfo[:, self._drone_vel_obs_indices].astype(np.float32)
                policy_obs = np.concatenate([policy_obs, drone_vel_obs], axis=1).astype(np.float32)
            if self._drone_ang_vel_obs_dim > 0:
                drone_ang_vel_obs = self._extraInfo[:, self._drone_ang_vel_obs_indices].astype(np.float32)
                policy_obs = np.concatenate([policy_obs, drone_ang_vel_obs], axis=1).astype(np.float32)
            if self._curr_nstep_pos_obs_dim > 0:
                curr_nstep_pos_obs = self._extract_curr_nstep_pos_obs()
                policy_obs = np.concatenate([policy_obs, curr_nstep_pos_obs], axis=1).astype(np.float32)
            return policy_obs
        policy_obs = obs.astype(np.float32)
        if self._append_prev_action:
            policy_obs = np.concatenate([policy_obs, self._flatten_prev_actions()], axis=1).astype(np.float32)
        if self._curr_nstep_pos_obs_dim > 0:
            curr_nstep_pos_obs = self._extract_curr_nstep_pos_obs()
            policy_obs = np.concatenate([policy_obs, curr_nstep_pos_obs], axis=1).astype(np.float32)
        return policy_obs

    def _flatten_prev_actions(self) -> np.ndarray:
        if not self._append_prev_action:
            return np.zeros((self._num_envs, 0), dtype=np.float32)
        return self._prev_actions.reshape(self._num_envs, self._prev_action_obs_dim).astype(np.float32)

    def _flatten_prev_tag_obs(self) -> np.ndarray:
        if self._prev_tag_obs_dim <= 0:
            return np.zeros((self._num_envs, 0), dtype=np.float32)
        return self._prev_tag_obs.reshape(self._num_envs, self._prev_tag_obs_dim).astype(np.float32)

    def _flatten_prev_pos_obs(self) -> np.ndarray:
        if self._prev_pos_obs_dim <= 0:
            return np.zeros((self._num_envs, 0), dtype=np.float32)
        return self._prev_pos_obs.reshape(self._num_envs, self._prev_pos_obs_dim).astype(np.float32)

    def _extract_curr_nstep_pos_obs(self) -> np.ndarray:
        if self._curr_nstep_pos_obs_dim <= 0:
            return np.zeros((self._num_envs, 0), dtype=np.float32)
        # History buffer is updated after obs extraction in step(), so before update:
        # hist[0] is t-1, hist[1] is t-2, ...
        lag_blocks = []
        for lag in self.nstep_pos_lags:
            hist_idx = max(0, int(lag) - 1)
            lag_blocks.append(self._curr_nstep_pos_hist[:, hist_idx, :].astype(np.float32))
        if len(lag_blocks) == 0:
            return np.zeros((self._num_envs, 0), dtype=np.float32)
        return np.concatenate(lag_blocks, axis=1).astype(np.float32)

    def _extract_tag_vel_obs(self, obs: np.ndarray) -> np.ndarray:
        if self._tag_vel_obs_dim <= 0:
            return np.zeros((obs.shape[0], 0), dtype=np.float32)
        if obs.ndim != 2:
            raise ValueError(f"Expected batched obs with shape (n_envs, dim), got {obs.shape}")
        current_tag_block = self._extract_current_tag_block(obs)
        if self._policy_tag_block_dim <= 0:
            return np.zeros((obs.shape[0], self._tag_vel_obs_dim), dtype=np.float32)
        prev_tag_block = self._last_tag_block.astype(np.float32)
        dt = float(self._tag_vel_dt) if self._tag_vel_dt > 0.0 else 0.02
        tag_vel = ((current_tag_block - prev_tag_block) / dt).astype(np.float32)
        if self._has_last_tag_block.shape[0] == obs.shape[0]:
            tag_vel[~self._has_last_tag_block] = 0.0
        return tag_vel

    def _extract_tag_diff_obs(self, obs: np.ndarray) -> np.ndarray:
        if self._tag_diff_obs_dim <= 0:
            return np.zeros((obs.shape[0], 0), dtype=np.float32)
        if obs.ndim != 2:
            raise ValueError(f"Expected batched obs with shape (n_envs, dim), got {obs.shape}")
        current_tag_block = self._extract_current_tag_block(obs)
        if self._policy_tag_block_dim <= 0:
            return np.zeros((obs.shape[0], self._tag_diff_obs_dim), dtype=np.float32)
        prev_tag_block = self._last_tag_block.astype(np.float32)
        tag_diff = (current_tag_block - prev_tag_block).astype(np.float32)
        if self._has_last_tag_block.shape[0] == obs.shape[0]:
            tag_diff[~self._has_last_tag_block] = 0.0
        return tag_diff

    def _extract_drone_pos_diff_obs(self) -> np.ndarray:
        if self._drone_pos_diff_obs_dim <= 0:
            return np.zeros((self._num_envs, 0), dtype=np.float32)
        if self._drone_pos_obs_dim <= 0:
            return np.zeros((self._num_envs, self._drone_pos_diff_obs_dim), dtype=np.float32)
        current_drone_pos_obs = self._extraInfo[:, self._drone_pos_obs_indices].astype(np.float32)
        drone_pos_diff_obs = current_drone_pos_obs - self._last_drone_pos_obs
        if self._has_last_drone_pos_obs.shape[0] == current_drone_pos_obs.shape[0]:
            drone_pos_diff_obs[~self._has_last_drone_pos_obs] = 0.0
        return drone_pos_diff_obs.astype(np.float32)

    def _extract_drone_ori_diff_obs(self) -> np.ndarray:
        if self._drone_ori_diff_obs_dim <= 0:
            return np.zeros((self._num_envs, 0), dtype=np.float32)
        if self._drone_ori_obs_dim <= 0:
            return np.zeros((self._num_envs, self._drone_ori_diff_obs_dim), dtype=np.float32)
        current_drone_ori_obs = self._extraInfo[:, self._drone_ori_obs_indices].astype(np.float32)
        drone_ori_diff_obs = current_drone_ori_obs - self._last_drone_ori_obs
        if self._has_last_drone_ori_obs.shape[0] == current_drone_ori_obs.shape[0]:
            drone_ori_diff_obs[~self._has_last_drone_ori_obs] = 0.0
        return drone_ori_diff_obs.astype(np.float32)

    def _policy_rms_tail(self, obs: np.ndarray) -> np.ndarray:
        if obs.ndim != 2:
            raise ValueError(f"Expected batched policy obs with shape (n_envs, dim), got {obs.shape}")
        if self._rms_indices.shape[0] == 0:
            return np.zeros((obs.shape[0], 0), dtype=np.float32)
        if obs.shape[1] <= int(self._rms_indices.max()):
            raise ValueError(
                f"Policy obs dim too small for RMS indices: got {obs.shape[1]}, need > {int(self._rms_indices.max())}"
            )
        return obs[:, self._rms_indices].astype(np.float32)

    def _extract_policy_tag_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Extract tag observations for policy and optionally keep tag_id in each tag block.
        Raw tag block format is [center_x, center_y, c0x, c0y, c1x, c1y, c2x, c2y, c3x, c3y, tag_id].
        """
        if obs.ndim != 2:
            raise ValueError(f"Expected batched obs with shape (n_envs, dim), got {obs.shape}")
        raw_tag_dim = self._policy_num_tags * self.TAG_UV_DIM
        if obs.shape[1] < raw_tag_dim:
            raise ValueError(
                f"Obs dim too small for tag extraction: got {obs.shape[1]}, need at least {raw_tag_dim}"
            )
        tag_obs = obs[:, :raw_tag_dim].astype(np.float32)
        tag_obs = tag_obs.reshape(obs.shape[0], self._policy_num_tags, self.TAG_UV_DIM)
        if not self.include_tag_id_obs:
            tag_obs = tag_obs[:, :, :self.TAG_POLICY_FEAT_DIM]
        return tag_obs.reshape(obs.shape[0], self._policy_tag_uv_dim).astype(np.float32)


    def _extract_policy_tag_aux_obs(self, obs: np.ndarray) -> np.ndarray:
        if obs.ndim != 2:
            raise ValueError(f"Expected batched obs with shape (n_envs, dim), got {obs.shape}")
        if self._tag_aux_dim <= 0:
            return np.zeros((obs.shape[0], 0), dtype=np.float32)
        raw_tag_dim = self._num_tags * self.TAG_UV_DIM
        aux_end = raw_tag_dim + self._tag_aux_dim
        if obs.shape[1] < aux_end:
            raise ValueError(
                f"Obs dim too small for tag aux extraction: got {obs.shape[1]}, need at least {aux_end}"
            )
        return obs[:, raw_tag_dim:aux_end].astype(np.float32)

    def _extract_current_tag_block(self, obs: np.ndarray) -> np.ndarray:
        """
        Extract the current-step tag block in the same representation used by policy.
        Shape: (n_envs, _policy_tag_block_dim)
        """
        if self._policy_tag_block_dim <= 0:
            return np.zeros((obs.shape[0], 0), dtype=np.float32)
        if obs.ndim != 2:
            raise ValueError(f"Expected batched obs with shape (n_envs, dim), got {obs.shape}")
        tag_obs = self._extract_policy_tag_obs(obs)
        if self._tag_aux_dim > 0:
            tag_aux_obs = self._extract_policy_tag_aux_obs(obs)
            return np.concatenate([tag_obs, tag_aux_obs], axis=1).astype(np.float32)
        return tag_obs.astype(np.float32)


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
        policy_obs = obs.astype(np.float32)
        if policy_obs.ndim != 2:
            raise ValueError(f"Expected batched obs with shape (n_envs, dim), got {policy_obs.shape}")
        normalized = policy_obs.copy()
        fixed_dim = 0 if self._skip_fixed_tag_scaling else min(self._fixed_tag_norm_dim, policy_obs.shape[1])
        if fixed_dim > 0:
            normalized[:, :fixed_dim] = normalized[:, :fixed_dim] / 83.0
        # Keep tag-diff features on a simple fixed scale instead of RMS.
        if self._tag_diff_obs_dim > 0 and self._tag_diff_obs_start >= 0:
            diff_start = min(self._tag_diff_obs_start, policy_obs.shape[1])
            diff_end = min(self._tag_diff_obs_end, policy_obs.shape[1])
            if diff_end > diff_start:
                normalized[:, diff_start:diff_end] = (
                    policy_obs[:, diff_start:diff_end] / self.TAG_DIFF_FIXED_SCALE
                ).astype(np.float32)
        if not self.use_obs_norm:
            return normalized.astype(np.float32)
        if self.obs_rms is not None and self._rms_indices.shape[0] > 0:
            normalized[:, self._rms_indices] = self._normalize_obs(
                policy_obs[:, self._rms_indices], self.obs_rms
            ).astype(np.float32)
            # Optionally override position / n-step position normalization to use
            # goal-centered mean with RMS std (shared from current position axes).
            if self._goal_center_pos_nstep_norm and len(self._goal_center_pos_indices) == 3:
                pos_std = np.ones((3,), dtype=np.float32)
                for axis in range(3):
                    src_idx = self._goal_center_pos_indices[axis]
                    if 0 <= src_idx < self._policy_to_rms.shape[0]:
                        rms_idx = int(self._policy_to_rms[src_idx])
                        if rms_idx >= 0:
                            pos_std[axis] = float(np.sqrt(self.obs_rms.var[rms_idx] + 1e-8))
                goal = self.GOAL_POS_NORM_MEAN
                for axis in range(3):
                    pos_idx = self._goal_center_pos_indices[axis]
                    if 0 <= pos_idx < policy_obs.shape[1]:
                        normalized[:, pos_idx] = (
                            (policy_obs[:, pos_idx] - goal[axis]) / pos_std[axis]
                        ).astype(np.float32)
                for nstep_group in self._goal_center_nstep_index_groups:
                    if len(nstep_group) != 3:
                        continue
                    for axis in range(3):
                        nstep_idx = nstep_group[axis]
                        if 0 <= nstep_idx < policy_obs.shape[1]:
                            normalized[:, nstep_idx] = (
                                (policy_obs[:, nstep_idx] - goal[axis]) / pos_std[axis]
                            ).astype(np.float32)

        # Apply the same fixed /83 scaling rule to previous-tag block as current tag:
        # first 10 tag coords are fixed-scaled and excluded from effective RMS output.
        if self._prev_tag_obs_dim > 0:
            prev_tag_start = self._policy_tag_block_dim
            prev_tag_fixed_dim = min(self.FIXED_TAG_NORM_DIM, self._prev_tag_obs_step_dim)
            for hist_idx in range(self.prev_tag_obs_history_len):
                block_start = prev_tag_start + hist_idx * self._prev_tag_obs_step_dim
                if prev_tag_fixed_dim > 0:
                    normalized[:, block_start:block_start + prev_tag_fixed_dim] = (
                        policy_obs[:, block_start:block_start + prev_tag_fixed_dim] / 83.0
                    ).astype(np.float32)
        return normalized.astype(np.float32)

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
        from tonedio_baselines.envs.pos_vec_env_wrapper import ObsNormUpdateCallback
        
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
        Updates normalization statistics for all PosFlightEnvVec instances in the wrapper chain.
        """
        # Find PosFlightEnvVec in the environment wrapper chain
        env = self.training_env
        while env is not None:
            if isinstance(env, PosFlightEnvVec):
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
        from tonedio_baselines.envs.pos_vec_env_wrapper import CheckpointCallbackWithRMS
        
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
        # Find PosFlightEnvVec in the environment wrapper chain
        env = self.training_env
        while env is not None:
            if isinstance(env, PosFlightEnvVec):
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
FlightEnvVec = PosFlightEnvVec
