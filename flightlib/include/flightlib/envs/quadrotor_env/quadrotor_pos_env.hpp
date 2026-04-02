#pragma once

// std lib
#include <array>
#include <stdlib.h>
#include <cmath>
#include <iostream>

// yaml cpp
#include <yaml-cpp/yaml.h>

// flightlib
#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/common/command.hpp"
#include "flightlib/common/logger.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/envs/env_base.hpp"
#include "flightlib/objects/quadrotor.hpp"
#include "flightlib/sensors/rgb_camera.hpp"

namespace flightlib {

namespace quadposenv {

enum Ctl : int {
  // observations:
  // For each QR tag:
  // [center_x, center_y, c0x, c0y, c1x, c1y, c2x, c2y, c3x, c3y, tag_id]
  // Then flattened_rgb is appended.
  // Observation layout:
  // [tag0(11), tag1(11), tag2(11), flattened_rgb]
  kObs = 0,
  kNumTags = 3,
  kTagFeat = 11,
  kTagObs = kNumTags * kTagFeat,
  // Backward-compatible aliases for tag0 indices.
  kCenterX = 0,
  kCenterY = 1,
  kCorner0X = 2,
  kCorner0Y = 3,
  kCorner1X = 4,
  kCorner1Y = 5,
  kCorner2X = 6,
  kCorner2Y = 7,
  kCorner3X = 8,
  kCorner3Y = 9,
  kTagId = 10,
  kImg = kTagObs,
  kImgWidth = 84,
  kImgHeight = 84,
  kImgChannels = 3,
  kNImg = kImgWidth * kImgHeight * kImgChannels,
  kNObs = kImg + kNImg,
  kNDroneStateObs = 12,
  // control actions
  kAct = 0,
  kNAct = 4,
};

};

class QuadrotorPosEnv final : public EnvBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  QuadrotorPosEnv();
  QuadrotorPosEnv(const std::string &cfg_path);
  ~QuadrotorPosEnv();

  // - public OpenAI-gym-style functions
  bool reset(Ref<Vector<>> obs, const bool random = true) override;
  Scalar step(const Ref<Vector<>> act, Ref<Vector<>> obs) override;

  // - public set functions
  bool loadParam(const YAML::Node &cfg);
  inline void setSpawnOffset(const Ref<Vector<3>> offset) {
    spawn_offset_ = offset;
  }

  // - public get functions
  bool getObs(Ref<Vector<>> obs) override;
  bool getAct(Ref<Vector<>> act) const;
  bool getAct(Command *const cmd) const;

  // - auxiliar functions
  void updateExtraInfo() override;
  bool isTerminalState(Scalar &reward) override;
  void addObjectsToUnity(std::shared_ptr<UnityBridge> bridge);

  friend std::ostream &operator<<(std::ostream &os,
                                  const QuadrotorPosEnv &quad_env);

 private:
  bool worldPointToCamera(const Ref<const Vector<3>> p_W,
                          Ref<Vector<3>> p_C) const;
  bool projectWorldPointToImage(const Ref<const Vector<3>> p_W,
                                Ref<Vector<2>> pixel_uv,
                                bool *in_front = nullptr,
                                bool *in_image = nullptr) const;

  // quadrotor
  std::shared_ptr<Quadrotor> quadrotor_ptr_;
  QuadState quad_state_;
  Command cmd_;
  Logger logger_{"QuadrotorPosEnv"};

  // Define reward for training
  Scalar pos_coeff_{0.0};
  Scalar ori_coeff_{0.0};
  Scalar lin_vel_coeff_{0.0};
  Scalar tag_pos_coeff_{0.0};
  Scalar tag_lin_vel_coeff_{0.0};
  Scalar ang_vel_coeff_{0.0};
  Scalar act_coeff_{0.0};
  Scalar survival_reward_{0.01};

  // observations and actions (for RL)
  Vector<quadposenv::kNObs> quad_obs_;
  Vector<quadposenv::kNAct> quad_act_;
  Vector<quadposenv::kNAct> prev_quad_act_ = Vector<quadposenv::kNAct>::Zero();
  bool prev_quad_act_valid_{false};
  std::shared_ptr<RGBCamera> rgb_camera_;

  // reward function design (for model-free reinforcement learning)
  Vector<3> goal_pos_{(Vector<3>() << 0.0, 0.0, 20.0).finished()};
  Vector<3> goal_ori_{Vector<3>::Zero()};
  Vector<3> goal_lin_vel_{Vector<3>::Zero()};
  Vector<3> goal_tag_pos_{Vector<3>::Zero()};
  Vector<3> goal_ang_vel_{Vector<3>::Zero()};
  Vector<10> goal_tag_lin_vel_{Vector<10>::Zero()};
  Scalar landing_w_vel_xy_near_{0.5};
  Scalar landing_w_vel_xy_far_{0.2};
  Scalar landing_w_vel_z_near_{0.5};
  Scalar landing_w_vel_z_far_{0.2};
  Scalar landing_near_ground_z_{5.0};
  Scalar landing_w_tilt_{0.2};
  Scalar landing_w_yaw_{0.0};
  Scalar landing_tilt_soft_{0.35};
  Scalar landing_tilt_hard_{0.7};
  Scalar landing_w_tilt_excess_{1.0};
  Scalar landing_tilt_hard_penalty_{2.0};
  Scalar landing_time_penalty_{0.01};
  Scalar landing_w_body_rate_{0.0};
  Scalar landing_w_rate_cmd_xy_{0.0};
  Scalar landing_w_duv_{0.0};
  Scalar landing_z_safe_margin_{2.0};
  Scalar landing_speed_soft_limit_{2.0};
  Scalar landing_speed_hard_limit_{4.0};
  Scalar landing_w_speed_excess_{1.0};
  Scalar landing_hard_speed_penalty_{3.0};
  Scalar landing_w_img_center_{0.0};
  Scalar landing_tag_not_visible_penalty_{0.0};
  Scalar landing_center_u_gate_{0.1};
  Scalar landing_center_v_gate_{0.1};
  Scalar landing_center_hold_bonus_{0.0};
  Scalar landing_xy_gate_{1.0};
  Scalar landing_w_early_descend_{0.5};
  Scalar landing_terminal_z_{0.02};
  Scalar landing_success_xy_error_{0.5};
  Scalar landing_success_vz_{1.0};
  Scalar landing_success_tilt_{0.35};
  Scalar landing_success_vxy_{0.25};
  Scalar landing_success_body_rate_{0.25};
  Scalar landing_success_reward_{10.0};
  Scalar landing_failure_reward_{-10.0};
  Scalar log_sum_{0.0};
  int log_counter_{0};
  int log_interval_steps_{200};
  bool enable_step_log_{true};

  // action and observation normalization (for learning)
  Vector<quadposenv::kNAct> act_mean_;
  Vector<quadposenv::kNAct> act_std_;
  bool use_ctbr_{false};
  Vector<3> init_pos_{(Vector<3>() << 0.0, 0.0, 20.0).finished()};
  bool randomize_position_on_reset_{true};
  Scalar randomize_position_scale_{1.0};
  bool randomize_velocity_on_reset_{true};
  Scalar randomize_velocity_scale_{1.0};
  bool randomize_attitude_on_reset_{true};
  Scalar randomize_attitude_scale_{1.0};
  Vector<quadposenv::kNObs> obs_mean_ = Vector<quadposenv::kNObs>::Zero();
  Vector<quadposenv::kNObs> obs_std_ = Vector<quadposenv::kNObs>::Ones();

  // tag target and camera projection parameters
  std::array<Vector<3>, quadposenv::kNumTags> tag_center_world_;
  std::array<Matrix<3, 4>, quadposenv::kNumTags> tag_corner_world_;
  std::array<int, quadposenv::kNumTags> tag_order_{{0, 1, 2}};
  std::array<bool, quadposenv::kNumTags> curr_tag_visible_{{false, false, false}};
  int stage_{0};
  bool area_reward_mode_active_{false};
  int miss_count_{0};
  bool hold_last_tag_obs_{false};
  bool use_projected_uv_out_of_view_{false};
  int stage_miss_threshold_{4};
  bool stage_require_next_visible_{true};
  Scalar stage_switch_bonus_{2.0};
  std::array<Scalar, quadposenv::kNumTags> stage_target_area_{{500.0, 350.0, 220.0}};
  Scalar invisible_base_penalty_{0.5};
  Scalar invisible_base_penalty_extra_below_threshold_{0.0};
  Scalar invisible_miss_penalty_{0.2};
  Scalar invisible_stage_penalty_{0.2};
  Scalar last_visible_area_{-1.0};
  Scalar miss_start_prev_area_{-1.0};
  Vector<3> B_r_BC_{(Vector<3>() << 0.0, 0.0, 0.3).finished()};
  Matrix<3, 3> R_BC_{Matrix<3, 3>::Identity()};
  Scalar fx_{0.0};
  Scalar fy_{0.0};
  Scalar cx_{0.0};
  Scalar cy_{0.0};
  int cam_width_{84};
  int cam_height_{84};
  bool log_missing_rgb_{false};
  Matrix<4, 2> prev_corner_uv_ = Matrix<4, 2>::Zero();
  bool prev_corner_uv_valid_{false};
  bool log_world_pose_{false};
  int log_world_pose_interval_steps_{50};
  int world_pose_log_counter_{0};
  Vector<3> estimated_p_C_{Vector<3>::Zero()};
  bool estimated_p_C_valid_{false};
  Vector<10> prev_tag_coord_{Vector<10>::Zero()};
  Vector<10> last_tag_lin_vel_{Vector<10>::Zero()};
  bool prev_tag_coord_valid_{false};

  Scalar last_total_reward_{0.0};
  Scalar last_metric_area_{0.0};
  Scalar last_signed_shape_score_{0.0};
  Scalar last_r_track_pos_{0.0};
  Scalar last_r_track_ori_{0.0};
  Scalar last_r_track_lin_vel_{0.0};
  Scalar last_r_track_tag_pos_{0.0};
  Scalar last_r_track_tag_lin_vel_{0.0};
  Scalar last_r_track_ang_vel_{0.0};
  Scalar last_r_track_act_{0.0};
  Scalar last_r_track_survival_{0.0};
  Scalar last_observed_area_{-1.0};
  bool last_tag_visible_{false};
  bool last_corners_visible_{false};

  YAML::Node cfg_;
  Matrix<3, 2> world_box_;
  Vector<3> spawn_offset_ = Vector<3>::Zero();
};

}  // namespace flightlib
