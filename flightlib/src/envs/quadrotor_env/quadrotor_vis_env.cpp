#include "flightlib/envs/quadrotor_env/quadrotor_vis_env.hpp"
#include <opencv2/imgproc.hpp>

namespace flightlib {

QuadrotorVisEnv::QuadrotorVisEnv()
  : QuadrotorVisEnv(getenv("FLIGHTMARE_PATH") +
                 std::string("/flightlib/configs/quadrotor_env.yaml")) {}

QuadrotorVisEnv::QuadrotorVisEnv(const std::string &cfg_path)
  : EnvBase(),
    pos_coeff_(0.0),
    ori_coeff_(0.0),
    lin_vel_coeff_(0.0),
    ang_vel_coeff_(0.0),
    act_coeff_(0.0),
    goal_pos_((Vector<3>() << 0.0, 0.0, 5.0).finished()),
    goal_ori_(Vector<3>::Zero()),
    goal_lin_vel_(Vector<3>::Zero()),
    goal_ang_vel_(Vector<3>::Zero()) {

  // load configuration file
  YAML::Node cfg_ = YAML::LoadFile(cfg_path);

  quadrotor_ptr_ = std::make_shared<Quadrotor>();
  // update dynamics
  QuadrotorDynamics dynamics;
  dynamics.updateParams(cfg_);
  quadrotor_ptr_->updateDynamics(dynamics);

  // add RGB camera for image observations
  rgb_camera_ = std::make_shared<RGBCamera>();

  // Camera defaults (body -> camera). Can be overridden from YAML:
  // quadrotor_env.camera.{rel_pos, rel_quat_wxyz, width, height, fov}
  Vector<3> B_r_BC(0.0, 0.0, 0.3);
  Quaternion q_BC(1.0, 0.0, 0.0, 0.0);  // w, x, y, z
  int cam_width = quadvisenv::kImgWidth;
  int cam_height = quadvisenv::kImgHeight;
  Scalar cam_fov = 70.0;

  if (cfg_["quadrotor_env"] && cfg_["quadrotor_env"]["camera"]) {
    const YAML::Node cam_cfg = cfg_["quadrotor_env"]["camera"];

    if (cam_cfg["rel_pos"] && cam_cfg["rel_pos"].IsSequence() &&
        cam_cfg["rel_pos"].size() == 3) {
      B_r_BC << cam_cfg["rel_pos"][0].as<Scalar>(),
        cam_cfg["rel_pos"][1].as<Scalar>(), cam_cfg["rel_pos"][2].as<Scalar>();
    }

    if (cam_cfg["rel_quat_wxyz"] && cam_cfg["rel_quat_wxyz"].IsSequence() &&
        cam_cfg["rel_quat_wxyz"].size() == 4) {
      q_BC = Quaternion(cam_cfg["rel_quat_wxyz"][0].as<Scalar>(),
                        cam_cfg["rel_quat_wxyz"][1].as<Scalar>(),
                        cam_cfg["rel_quat_wxyz"][2].as<Scalar>(),
                        cam_cfg["rel_quat_wxyz"][3].as<Scalar>());
      if (q_BC.norm() > 1e-9) {
        q_BC.normalize();
      } else {
        logger_.warn("Invalid camera quaternion norm in YAML. Using identity.");
        q_BC = Quaternion(1.0, 0.0, 0.0, 0.0);
      }
    }

    if (cam_cfg["width"]) cam_width = cam_cfg["width"].as<int>();
    if (cam_cfg["height"]) cam_height = cam_cfg["height"].as<int>();
    if (cam_cfg["fov"]) cam_fov = cam_cfg["fov"].as<Scalar>();
  }

  Matrix<3, 3> R_BC = q_BC.toRotationMatrix();

  rgb_camera_->setWidth(cam_width);
  rgb_camera_->setHeight(cam_height);
  rgb_camera_->setFOV(cam_fov);
  rgb_camera_->setRelPose(B_r_BC, R_BC);

  rgb_camera_->setPostProcesscing(std::vector<bool>{false, false, false});
  quadrotor_ptr_->addRGBCamera(rgb_camera_);

  // define a bounding box
  // Keep horizontal bounds, but allow higher-altitude hovering tasks.
  world_box_ << -30, 30, -30, 30, 0, 30;
  if (!quadrotor_ptr_->setWorldBox(world_box_)) {
    logger_.error("cannot set wolrd box");
  };

  // define input and output dimension for the environment
  obs_dim_ = quadvisenv::kNObs;
  act_dim_ = quadvisenv::kNAct;

  Scalar mass = quadrotor_ptr_->getMass();
  act_mean_ = Vector<quadvisenv::kNAct>::Ones() * (-mass * Gz) / 4;
  act_std_ = Vector<quadvisenv::kNAct>::Ones() * (-mass * 2 * Gz) / 4;

  // load parameters
  loadParam(cfg_);
}

QuadrotorVisEnv::~QuadrotorVisEnv() {}

bool QuadrotorVisEnv::reset(Ref<Vector<>> obs, const bool random) {
  quad_state_.setZero();
  quad_act_.setZero();

  if (random) {
    // randomly reset the quadrotor state
    // reset position
    quad_state_.x(QS::POSX) = uniform_dist_(random_gen_) + spawn_offset_(0);
    quad_state_.x(QS::POSY) = uniform_dist_(random_gen_) + spawn_offset_(1);
    quad_state_.x(QS::POSZ) = uniform_dist_(random_gen_) + 20 + spawn_offset_(2);
    if (quad_state_.x(QS::POSX) < world_box_(0, 0) + 0.5)
      quad_state_.x(QS::POSX) = world_box_(0, 0) + 0.5;
    if (quad_state_.x(QS::POSX) > world_box_(0, 1) - 0.5)
      quad_state_.x(QS::POSX) = world_box_(0, 1) - 0.5;
    if (quad_state_.x(QS::POSY) < world_box_(1, 0) + 0.5)
      quad_state_.x(QS::POSY) = world_box_(1, 0) + 0.5;
    if (quad_state_.x(QS::POSY) > world_box_(1, 1) - 0.5)
      quad_state_.x(QS::POSY) = world_box_(1, 1) - 0.5;
    if (quad_state_.x(QS::POSZ) < -0.0)
      quad_state_.x(QS::POSZ) = -quad_state_.x(QS::POSZ);
    // reset linear velocity
    quad_state_.x(QS::VELX) = uniform_dist_(random_gen_);
    quad_state_.x(QS::VELY) = uniform_dist_(random_gen_);
    quad_state_.x(QS::VELZ) = uniform_dist_(random_gen_);
    // reset orientation
    quad_state_.x(QS::ATTW) = uniform_dist_(random_gen_);
    quad_state_.x(QS::ATTX) = uniform_dist_(random_gen_);
    quad_state_.x(QS::ATTY) = uniform_dist_(random_gen_);
    quad_state_.x(QS::ATTZ) = uniform_dist_(random_gen_);
    quad_state_.qx /= quad_state_.qx.norm();
  }
  // reset quadrotor with random states
  quadrotor_ptr_->reset(quad_state_);

  // reset control command
  cmd_.t = 0.0;
  cmd_.thrusts.setZero();

  // obtain observations
  getObs(obs);
  return true;
}

bool QuadrotorVisEnv::getObs(Ref<Vector<>> obs) {
  quadrotor_ptr_->getState(&quad_state_);

  // Keep previous frame if a new camera frame is not available this step.
  static int missing_rgb_warn_count = 0;

  cv::Mat rgb_image;
  if (rgb_camera_ != nullptr && rgb_camera_->getRGBImage(rgb_image) &&
      !rgb_image.empty()) {
    cv::Mat resized = rgb_image;
    if (rgb_image.cols != quadvisenv::kImgWidth ||
        rgb_image.rows != quadvisenv::kImgHeight) {
      cv::resize(rgb_image, resized,
                 cv::Size(quadvisenv::kImgWidth, quadvisenv::kImgHeight), 0.0, 0.0,
                 cv::INTER_AREA);
    }
    if (!resized.isContinuous()) resized = resized.clone();

    int flat_idx = 0;
    for (int r = 0; r < resized.rows; r++) {
      const cv::Vec3b *row = resized.ptr<cv::Vec3b>(r);
      for (int c = 0; c < resized.cols; c++) {
        quad_obs_(flat_idx++) = static_cast<Scalar>(row[c][0]);
        quad_obs_(flat_idx++) = static_cast<Scalar>(row[c][1]);
        quad_obs_(flat_idx++) = static_cast<Scalar>(row[c][2]);
      }
    }
  } else {
    if (missing_rgb_warn_count < 20) {
      logger_.warn(
        "getObs: RGB frame is empty (camera queue not updated yet). "
        "Check Unity connection/frame sync.");
      missing_rgb_warn_count++;
    }
  }

  obs.segment<quadvisenv::kNObs>(quadvisenv::kObs) = quad_obs_;
  return true;
}

Scalar QuadrotorVisEnv::step(const Ref<Vector<>> act, Ref<Vector<>> obs) {
  quad_act_ = act.cwiseProduct(act_std_) + act_mean_;
  cmd_.t += sim_dt_;
  cmd_.thrusts = quad_act_;

  // simulate quadrotor
  quadrotor_ptr_->run(cmd_, sim_dt_);

  // update observations
  getObs(obs);

  Vector<3> euler_zyx = quad_state_.q().toRotationMatrix().eulerAngles(2, 1, 0);

  // ---------------------- reward function design
  // - position tracking
  Scalar pos_reward = pos_coeff_ * (quad_state_.p - goal_pos_).squaredNorm();

  // - orientation tracking
  Scalar ori_reward = ori_coeff_ * (euler_zyx - goal_ori_).squaredNorm();

  // - linear velocity tracking
  Scalar lin_vel_reward =
    lin_vel_coeff_ * (quad_state_.v - goal_lin_vel_).squaredNorm();

  // - angular velocity tracking
  Scalar ang_vel_reward =
     ang_vel_coeff_ * (quad_state_.w - goal_ang_vel_).squaredNorm();

  // - control action penalty
  Scalar act_reward = act_coeff_ * act.cast<Scalar>().norm();

  Scalar total_reward =
    pos_reward + ori_reward + lin_vel_reward + ang_vel_reward + act_reward;

  // survival reward
  total_reward += 0.1;

  return total_reward;
}

bool QuadrotorVisEnv::isTerminalState(Scalar &reward) {
  if (quad_state_.x(QS::POSZ) <= 0.02) {
    reward = -0.02;
    return true;
  }
  reward = 0.0;
  return false;
}

bool QuadrotorVisEnv::loadParam(const YAML::Node &cfg) {
  if (cfg["quadrotor_env"]) {
    sim_dt_ = cfg["quadrotor_env"]["sim_dt"].as<Scalar>();
    max_t_ = cfg["quadrotor_env"]["max_t"].as<Scalar>();
  } else {
    return false;
  }

  if (cfg["rl"]) {
    // load reinforcement learning related parameters
    pos_coeff_ = cfg["rl"]["pos_coeff"].as<Scalar>();
    ori_coeff_ = cfg["rl"]["ori_coeff"].as<Scalar>();
    lin_vel_coeff_ = cfg["rl"]["lin_vel_coeff"].as<Scalar>();
    ang_vel_coeff_ = cfg["rl"]["ang_vel_coeff"].as<Scalar>();
    act_coeff_ = cfg["rl"]["act_coeff"].as<Scalar>();
  } else {
    return false;
  }
  return true;
}

bool QuadrotorVisEnv::getAct(Ref<Vector<>> act) const {
  if (cmd_.t >= 0.0 && quad_act_.allFinite()) {
    act = quad_act_;
    return true;
  }
  return false;
}

bool QuadrotorVisEnv::getAct(Command *const cmd) const {
  if (!cmd_.valid()) return false;
  *cmd = cmd_;
  return true;
}

void QuadrotorVisEnv::addObjectsToUnity(std::shared_ptr<UnityBridge> bridge) {
  bridge->addQuadrotor(quadrotor_ptr_);
}

std::ostream &operator<<(std::ostream &os, const QuadrotorVisEnv &quad_env) {
  os.precision(3);
  os << "Quadrotor Environment:\n"
     << "obs dim =            [" << quad_env.obs_dim_ << "]\n"
     << "act dim =            [" << quad_env.act_dim_ << "]\n"
     << "sim dt =             [" << quad_env.sim_dt_ << "]\n"
     << "max_t =              [" << quad_env.max_t_ << "]\n"
     << "act_mean =           [" << quad_env.act_mean_.transpose() << "]\n"
     << "act_std =            [" << quad_env.act_std_.transpose() << "]\n"
     << "obs_mean =           [" << quad_env.obs_mean_.transpose() << "]\n"
     << "obs_std =            [" << quad_env.obs_std_.transpose() << std::endl;
  os.precision();
  return os;
}

}  // namespace flightlib
