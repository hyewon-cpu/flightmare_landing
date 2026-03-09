#pragma once

// std lib
#include <stdlib.h>
#include <algorithm>
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

namespace flightlib {

namespace quadvisenv {

enum Ctl : int {
  // observations
  kObs = 0,
  kImgWidth = 84,
  kImgHeight = 84,
  kImgChannels = 3,
  kNObs = kImgWidth * kImgHeight * kImgChannels,
  // control actions
  kAct = 0,
  kNAct = 4,
};
};
class QuadrotorVisEnv final : public EnvBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  QuadrotorVisEnv();
  QuadrotorVisEnv(const std::string &cfg_path);
  ~QuadrotorVisEnv();

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
  bool isTerminalState(Scalar &reward) override;
  void addObjectsToUnity(std::shared_ptr<UnityBridge> bridge);

  friend std::ostream &operator<<(std::ostream &os,
                                  const QuadrotorVisEnv &quad_env);

 private:
  bool projectWorldPointToImage(const Ref<const Vector<3>> p_W,
                                Ref<Vector<2>> pixel_uv,
                                bool *in_front = nullptr,
                                bool *in_image = nullptr) const;

  // quadrotor
  std::shared_ptr<Quadrotor> quadrotor_ptr_;
  QuadState quad_state_;
  Command cmd_;
  Logger logger_{"QaudrotorEnv"};

  // Define reward for training
  Scalar pos_coeff_, ori_coeff_, lin_vel_coeff_, ang_vel_coeff_, act_coeff_;

  // observations and actions (for RL)
  Vector<quadvisenv::kNObs> quad_obs_;
  Vector<quadvisenv::kNAct> quad_act_;
  std::shared_ptr<RGBCamera> rgb_camera_;

  // reward function design (for model-free reinforcement learning)
  Vector<3> goal_pos_;
  Vector<3> goal_ori_;
  Vector<3> goal_lin_vel_;
  Vector<3> goal_ang_vel_;

  // action and observation normalization (for learning)
  Vector<quadvisenv::kNAct> act_mean_;
  Vector<quadvisenv::kNAct> act_std_;
  bool use_ctbr_{false};
  Vector<3> init_pos_{(Vector<3>() << 0.0, 0.0, 20.0).finished()};
  bool randomize_position_on_reset_{true};
  Scalar randomize_position_scale_{1.0};
  bool randomize_velocity_on_reset_{true};
  Scalar randomize_velocity_scale_{1.0};
  bool randomize_attitude_on_reset_{true};
  Scalar randomize_attitude_scale_{1.0};
  Vector<quadvisenv::kNObs> obs_mean_ = Vector<quadvisenv::kNObs>::Zero();
  Vector<quadvisenv::kNObs> obs_std_ = Vector<quadvisenv::kNObs>::Ones();
  Vector<3> tag_center_world_{(Vector<3>() << 0.0, 0.0, 0.0).finished()};
  Matrix<3, 4> tag_corner_world_{
    (Matrix<3, 4>() << -0.5, 0.5, 0.5, -0.5,
     -0.5, -0.5, 0.5, 0.5,
      0.0, 0.0, 0.0, 0.0).finished()};
  Vector<3> B_r_BC_{(Vector<3>() << 0.0, 0.0, 0.3).finished()};
  Matrix<3, 3> R_BC_{Matrix<3, 3>::Identity()};
  Scalar fx_{0.0};
  Scalar fy_{0.0};
  Scalar cx_{0.0};
  Scalar cy_{0.0};
  int cam_width_{84};
  int cam_height_{84};

  YAML::Node cfg_;
  Matrix<3, 2> world_box_;
  Vector<3> spawn_offset_ = Vector<3>::Zero();
};

}  // namespace flightlib
