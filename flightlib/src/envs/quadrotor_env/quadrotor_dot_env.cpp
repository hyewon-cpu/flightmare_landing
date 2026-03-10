#include "flightlib/envs/quadrotor_env/quadrotor_dot_env.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <utility>

namespace flightlib {
namespace {

bool parseTagCenters(const YAML::Node &centers_node,
                     std::array<Vector<3>, quaddotenv::kNumTags> *centers) {
  if (!centers_node || !centers_node.IsSequence() ||
      static_cast<int>(centers_node.size()) != quaddotenv::kNumTags) {
    return false;
  }
  for (int i = 0; i < quaddotenv::kNumTags; i++) {
    const YAML::Node center = centers_node[i];
    if (!center.IsSequence() || center.size() != 3) return false;
    (*centers)[i] << center[0].as<Scalar>(), center[1].as<Scalar>(),
      center[2].as<Scalar>();
  }
  return true;
}

bool parseTagCorners(const YAML::Node &corners_node,
                     std::array<Matrix<3, 4>, quaddotenv::kNumTags> *corners) {
  if (!corners_node || !corners_node.IsSequence() ||
      static_cast<int>(corners_node.size()) != quaddotenv::kNumTags) {
    return false;
  }
  for (int tag_idx = 0; tag_idx < quaddotenv::kNumTags; tag_idx++) {
    const YAML::Node per_tag = corners_node[tag_idx];
    if (!per_tag.IsSequence() || per_tag.size() != 4) return false;
    for (int corner_idx = 0; corner_idx < 4; corner_idx++) {
      const YAML::Node corner = per_tag[corner_idx];
      if (!corner.IsSequence() || corner.size() != 3) return false;
      (*corners)[tag_idx].col(corner_idx) << corner[0].as<Scalar>(),
        corner[1].as<Scalar>(), corner[2].as<Scalar>();
    }
  }
  return true;
}

bool parseWorldBox(const YAML::Node &world_box_node, Matrix<3, 2> *world_box) {
  if (!world_box_node || !world_box_node.IsSequence()) {
    return false;
  }
  // Format A: [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
  if (world_box_node.size() == 3) {
    for (int i = 0; i < 3; i++) {
      const YAML::Node axis = world_box_node[i];
      if (!axis.IsSequence() || axis.size() != 2) return false;
      (*world_box)(i, 0) = axis[0].as<Scalar>();
      (*world_box)(i, 1) = axis[1].as<Scalar>();
    }
    return true;
  }
  // Format B: [xmin, xmax, ymin, ymax, zmin, zmax]
  if (world_box_node.size() == 6) {
    (*world_box) << world_box_node[0].as<Scalar>(), world_box_node[1].as<Scalar>(),
      world_box_node[2].as<Scalar>(), world_box_node[3].as<Scalar>(),
      world_box_node[4].as<Scalar>(), world_box_node[5].as<Scalar>();
    return true;
  }
  return false;
}

Scalar estimateTagScale(const Matrix<3, 4> &corners) {
  const Scalar l01 = (corners.col(1) - corners.col(0)).norm();
  const Scalar l12 = (corners.col(2) - corners.col(1)).norm();
  const Scalar l23 = (corners.col(3) - corners.col(2)).norm();
  const Scalar l30 = (corners.col(0) - corners.col(3)).norm();
  const Scalar mean_edge = (l01 + l12 + l23 + l30) * Scalar(0.25);
  return mean_edge;
}

}  // namespace

QuadrotorDotEnv::QuadrotorDotEnv()
  : QuadrotorDotEnv(getenv("FLIGHTMARE_PATH") +
                    std::string("/flightlib/configs/quadrotor_env.yaml")) {}

QuadrotorDotEnv::QuadrotorDotEnv(const std::string &cfg_path)
  : EnvBase(),
    pos_coeff_(0.0),
    ori_coeff_(0.0),
    lin_vel_coeff_(0.0),
    ang_vel_coeff_(0.0),
    act_coeff_(0.0),
    goal_pos_((Vector<3>() << 5.0, 7.0, 3.0).finished()),
    goal_ori_(Vector<3>::Zero()),
    goal_lin_vel_(Vector<3>::Zero()),
    goal_ang_vel_(Vector<3>::Zero()) {
  for (int i = 0; i < quaddotenv::kNumTags; i++) {
    tag_center_world_[i] = goal_pos_;
  }
  tag_center_world_[1].x() += 1.5;
  tag_center_world_[2].x() -= 1.5;
  for (int i = 0; i < quaddotenv::kNumTags; i++) {
    const Scalar cx = tag_center_world_[i].x();
    const Scalar cy = tag_center_world_[i].y();
    const Scalar cz = tag_center_world_[i].z();
    tag_corner_world_[i] <<
      cx - 0.5, cx + 0.5, cx + 0.5, cx - 0.5,
      cy - 0.5, cy - 0.5, cy + 0.5, cy + 0.5,
      cz,       cz,       cz,       cz;
  }

  // load configuration file
  YAML::Node cfg_ = YAML::LoadFile(cfg_path);

  quadrotor_ptr_ = std::make_shared<Quadrotor>();
  // update dynamics
  QuadrotorDynamics dynamics;
  dynamics.updateParams(cfg_);
  quadrotor_ptr_->updateDynamics(dynamics);

  // add RGB camera so the same extrinsics are available to Unity (optional)
  rgb_camera_ = std::make_shared<RGBCamera>();

  // Camera defaults (body -> camera). Can be overridden from YAML:
  // quadrotor_env.camera.{rel_pos, rel_quat_wxyz, width, height, fov,
  // fx, fy, cx, cy, intrinsics}
  Quaternion q_BC(0.7071, -0.7071, 0.0, 0.0);  // w, x, y, z
  Scalar cam_fov = 70.0;

  if (cfg_["quadrotor_env"] && cfg_["quadrotor_env"]["camera"]) {
    const YAML::Node cam_cfg = cfg_["quadrotor_env"]["camera"];

    if (cam_cfg["rel_pos"] && cam_cfg["rel_pos"].IsSequence() &&
        cam_cfg["rel_pos"].size() == 3) {
      B_r_BC_ << cam_cfg["rel_pos"][0].as<Scalar>(),
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

    if (cam_cfg["width"]) cam_width_ = cam_cfg["width"].as<int>();
    if (cam_cfg["height"]) cam_height_ = cam_cfg["height"].as<int>();
    if (cam_cfg["fov"]) cam_fov = cam_cfg["fov"].as<Scalar>();
    if (cam_cfg["log_missing_rgb"]) {
      log_missing_rgb_ = cam_cfg["log_missing_rgb"].as<bool>();
    }
    if (cam_cfg["log_world_pose"]) {
      log_world_pose_ = cam_cfg["log_world_pose"].as<bool>();
    }
    if (cam_cfg["log_world_pose_interval_steps"]) {
      log_world_pose_interval_steps_ =
        std::max(1, cam_cfg["log_world_pose_interval_steps"].as<int>());
    }

    if (cam_cfg["intrinsics"] && cam_cfg["intrinsics"].IsSequence() &&
        cam_cfg["intrinsics"].size() == 4) {
      fx_ = cam_cfg["intrinsics"][0].as<Scalar>();
      fy_ = cam_cfg["intrinsics"][1].as<Scalar>();
      cx_ = cam_cfg["intrinsics"][2].as<Scalar>();
      cy_ = cam_cfg["intrinsics"][3].as<Scalar>();
    } else {
      if (cam_cfg["fx"]) fx_ = cam_cfg["fx"].as<Scalar>();
      if (cam_cfg["fy"]) fy_ = cam_cfg["fy"].as<Scalar>();
      if (cam_cfg["cx"]) cx_ = cam_cfg["cx"].as<Scalar>();
      if (cam_cfg["cy"]) cy_ = cam_cfg["cy"].as<Scalar>();


    }

    if (parseTagCenters(cam_cfg["tag_centers_world"], &tag_center_world_)) {
      goal_pos_ = tag_center_world_[0];
    } else if (cam_cfg["tag_centers_world"]) {
      logger_.warn("tag_centers_world must be [[x,y,z] x3]. Using existing defaults.");
    }
    if (parseTagCorners(cam_cfg["tag_corners_world"], &tag_corner_world_)) {
      // parsed successfully
    } else if (cam_cfg["tag_corners_world"]) {
      logger_.warn("tag_corners_world must be [[[x,y,z] x4] x3]. Using existing defaults.");
    }
    // Backward compatibility for single-tag keys.
    if (cam_cfg["dot_world_pos"] && cam_cfg["dot_world_pos"].IsSequence() &&
        cam_cfg["dot_world_pos"].size() == 3) {
      tag_center_world_[0] << cam_cfg["dot_world_pos"][0].as<Scalar>(),
        cam_cfg["dot_world_pos"][1].as<Scalar>(),
        cam_cfg["dot_world_pos"][2].as<Scalar>();
      goal_pos_ = tag_center_world_[0];
    }
    if (cam_cfg["tag_center_world"] && cam_cfg["tag_center_world"].IsSequence() &&
        cam_cfg["tag_center_world"].size() == 3) {
      tag_center_world_[0] << cam_cfg["tag_center_world"][0].as<Scalar>(),
        cam_cfg["tag_center_world"][1].as<Scalar>(),
        cam_cfg["tag_center_world"][2].as<Scalar>();
      goal_pos_ = tag_center_world_[0];
    }
    if (cam_cfg["tag_corner_world"] && cam_cfg["tag_corner_world"].IsSequence() &&
        cam_cfg["tag_corner_world"].size() == 4) {
      bool corners_valid = true;
      for (int i = 0; i < 4; i++) {
        if (!cam_cfg["tag_corner_world"][i].IsSequence() ||
            cam_cfg["tag_corner_world"][i].size() != 3) {
          corners_valid = false;
          break;
        }
        tag_corner_world_[0].col(i) << cam_cfg["tag_corner_world"][i][0].as<Scalar>(),
          cam_cfg["tag_corner_world"][i][1].as<Scalar>(),
          cam_cfg["tag_corner_world"][i][2].as<Scalar>();
      }
      if (!corners_valid) {
        logger_.warn("tag_corner_world must be [[x,y,z] x4]. Using defaults.");
      }
    }
  }

  R_BC_ = q_BC.toRotationMatrix();

  // If intrinsics are not fully specified, derive from fov and image size.
  if (fx_ <= 0.0 || fy_ <= 0.0) {
    constexpr Scalar kPi = static_cast<Scalar>(3.14159265358979323846);
    const Scalar fov_rad = cam_fov * kPi / 180.0;
    const Scalar f_from_fov =
      (0.5 * static_cast<Scalar>(cam_height_)) / std::tan(0.5 * fov_rad);
    fx_ = f_from_fov;
    fy_ = f_from_fov;
  }
  if (cx_ == 0.0 && cy_ == 0.0) {
    cx_ = 0.5 * static_cast<Scalar>(cam_width_);
    cy_ = 0.5 * static_cast<Scalar>(cam_height_);
  }

  rgb_camera_->setWidth(cam_width_);
  rgb_camera_->setHeight(cam_height_);
  rgb_camera_->setFOV(cam_fov);
  rgb_camera_->setRelPose(B_r_BC_, R_BC_);
  rgb_camera_->setPostProcesscing(std::vector<bool>{false, false, false});
  quadrotor_ptr_->addRGBCamera(rgb_camera_);

  // define a bounding box
  world_box_ << -30, 30, -30, 30, 0, 30;
  if (cfg_["quadrotor_env"] && cfg_["quadrotor_env"]["world_box"]) {
    if (!parseWorldBox(cfg_["quadrotor_env"]["world_box"], &world_box_)) {
      logger_.warn("world_box must be [[xmin,xmax],[ymin,ymax],[zmin,zmax]] or [xmin,xmax,ymin,ymax,zmin,zmax]. Using default.");
      world_box_ << -30, 30, -30, 30, 0, 30;
    }
  }
  if (!quadrotor_ptr_->setWorldBox(world_box_)) {
    logger_.error("cannot set wolrd box");
  };

  // define input and output dimension for the environment
  obs_dim_ = quaddotenv::kNObs;
  act_dim_ = quaddotenv::kNAct;

  // Select control interpretation from YAML:
  // - motor: [m0, m1, m2, m3] rotor thrust commands
  // - ctbr: [collective_thrust, body_rate_x, body_rate_y, body_rate_z]
  std::string control_mode = "ctbr";
  if (cfg_["quadrotor_env"] && cfg_["quadrotor_env"]["control_mode"]) {
    control_mode = cfg_["quadrotor_env"]["control_mode"].as<std::string>();
  }
  use_ctbr_ = (control_mode == "ctbr");

  if (use_ctbr_) {
    const Scalar hover_acc = -Gz;
    Vector<3> omega_max = Vector<3>::Constant(6.0);
    if (cfg_["quadrotor_dynamics"] && cfg_["quadrotor_dynamics"]["omega_max"]) {
      const std::vector<Scalar> omega_max_cfg =
        cfg_["quadrotor_dynamics"]["omega_max"].as<std::vector<Scalar>>();
      if (omega_max_cfg.size() == 3) {
        omega_max = Map<const Vector<3>>(omega_max_cfg.data());
      }
    }
    act_mean_ << hover_acc, 0.0, 0.0, 0.0;
    act_std_ << hover_acc, omega_max.x(), omega_max.y(), omega_max.z();
  } else {
    Scalar mass = quadrotor_ptr_->getMass();
    act_mean_ = Vector<quaddotenv::kNAct>::Ones() * (-mass * Gz) / 4;
    act_std_ = Vector<quaddotenv::kNAct>::Ones() * (-mass * 2 * Gz) / 4;
  }

  // reasonable normalization defaults for [dot_uv, dot_duv + rgb]
  obs_mean_.setZero();
  obs_std_.setOnes();

  // load parameters
  loadParam(cfg_);
}

QuadrotorDotEnv::~QuadrotorDotEnv() {}

bool QuadrotorDotEnv::reset(Ref<Vector<>> obs, const bool random) {
  quad_state_.setZero();
  quad_obs_.setZero();
  quad_act_.setZero();
  prev_uv_.setZero();
  prev_uv_valid_ = false;
  stage_ = 0;
  miss_count_ = 0;
  last_visible_area_ = -1.0;
  miss_start_prev_area_ = -1.0;

  // deterministic base state from YAML
  quad_state_.x(QS::POSX) = init_pos_(0) + spawn_offset_(0); //spawn_offset_ : hpp 파일에 정의됨. 0,0,0이 기본값. setSpawnOffset 으로 변경 가능 
  quad_state_.x(QS::POSY) = init_pos_(1) + spawn_offset_(1);
  quad_state_.x(QS::POSZ) = init_pos_(2) + spawn_offset_(2);
  quad_state_.x(QS::ATTW) = 1.0;
  quad_state_.x(QS::ATTX) = 0.0;
  quad_state_.x(QS::ATTY) = 0.0;
  quad_state_.x(QS::ATTZ) = 0.0;

  if (random) {
    // randomly reset the quadrotor state
    // reset position around init_pos
    if (randomize_position_on_reset_) {
      quad_state_.x(QS::POSX) += uniform_dist_(random_gen_) * randomize_position_scale_;
       //random_gen_ : env_base.hpp 에 정의된 난수 생성기. 균등분포 uniform_dist_{-1.0,1.0} 로 정의됨 
       //즉, [-1.1] 범위의 난수를 하나 뽑아서 위치 램덤하게 함. 
       //randomize_position_scale_ : yaml에서 설정 가능. 위치 램덤하게 하는 정도. 1 이 기본값 
      quad_state_.x(QS::POSY) += uniform_dist_(random_gen_) * randomize_position_scale_;
      quad_state_.x(QS::POSZ) += uniform_dist_(random_gen_) * randomize_position_scale_;
    }
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
    if (randomize_velocity_on_reset_) {
      quad_state_.x(QS::VELX) = uniform_dist_(random_gen_) * randomize_velocity_scale_;
      quad_state_.x(QS::VELY) = uniform_dist_(random_gen_) * randomize_velocity_scale_;
      quad_state_.x(QS::VELZ) = uniform_dist_(random_gen_) * randomize_velocity_scale_;
    }
    // reset orientation
    if (randomize_attitude_on_reset_) {
      quad_state_.x(QS::ATTW) = uniform_dist_(random_gen_) * randomize_attitude_scale_;
      quad_state_.x(QS::ATTX) = uniform_dist_(random_gen_) * randomize_attitude_scale_;
      quad_state_.x(QS::ATTY) = uniform_dist_(random_gen_) * randomize_attitude_scale_;
      quad_state_.x(QS::ATTZ) = uniform_dist_(random_gen_) * randomize_attitude_scale_;
      if (quad_state_.qx.norm() > 1e-9) {
        quad_state_.qx /= quad_state_.qx.norm();
      } else {
        quad_state_.x(QS::ATTW) = 1.0;
        quad_state_.x(QS::ATTX) = 0.0;
        quad_state_.x(QS::ATTY) = 0.0;
        quad_state_.x(QS::ATTZ) = 0.0;
      }
    }
  }
  // reset quadrotor with random states
  quadrotor_ptr_->reset(quad_state_);

  // reset control command
  cmd_.t = 0.0;
  if (use_ctbr_) {
    cmd_.collective_thrust = -Gz;
    cmd_.omega.setZero();
    cmd_.thrusts = Vector<4>::Constant(NAN);
  } else {
    cmd_.collective_thrust = NAN;
    cmd_.omega = Vector<3>::Constant(NAN);
    cmd_.thrusts.setZero();
  }

  // obtain observations
  getObs(obs);
  return true;
}

bool QuadrotorDotEnv::projectWorldPointToImage(const Ref<const Vector<3>> p_W,
                                               Ref<Vector<2>> pixel_uv,
                                               bool *in_front,
                                               bool *in_image) const {
  // R_WB rotates body-frame vectors into world frame.
  const Matrix<3, 3> R_WB = quad_state_.q().toRotationMatrix(); //drone orientation
  const Vector<3> p_WB = quad_state_.p; //drone position

  const Matrix<3, 3> R_WC = R_WB * R_BC_; //camera world orientation
  const Vector<3> p_WC = p_WB + R_WB * B_r_BC_; //camera world POSITION

  // Transform world point into camera coordinates.
  const Vector<3> p_C = R_WC.transpose() * (p_W - p_WC); //p_W = target world coordinates

  constexpr Scalar kMinDepth = 1e-2;
  const Scalar depth = p_C.y();
  const bool point_in_front = (depth > kMinDepth);
  if (in_front != nullptr) {
    *in_front = point_in_front;
  }
  if (!point_in_front) {
    return false;
  }

  const Scalar inv_depth = 1.0 / depth;
  pixel_uv.x() = fx_ * (p_C.x() * inv_depth);
  pixel_uv.y() = -(fy_ * (p_C.z() * inv_depth));

  if (!pixel_uv.allFinite()) {
    return false;
  }

  // Treat out-of-image projections as invisible.
  const bool is_in_image =
    (pixel_uv.x() >= -cx_) &&
    (pixel_uv.x() < (static_cast<Scalar>(cam_width_) - cx_)) &&
    (pixel_uv.y() >= -cy_) &&
    (pixel_uv.y() < (static_cast<Scalar>(cam_height_) - cy_));
  if (in_image != nullptr) {
    *in_image = is_in_image;
  }
  return true;
}

bool QuadrotorDotEnv::getObs(Ref<Vector<>> obs) {
  quadrotor_ptr_->getState(&quad_state_);

  quad_obs_.segment<quaddotenv::kTagObs>(quaddotenv::kObs).setConstant(-1.0);

  if (log_world_pose_) {
    world_pose_log_counter_++;
    if (world_pose_log_counter_ % log_world_pose_interval_steps_ == 0) {
      const Matrix<3, 3> R_WB = quad_state_.q().toRotationMatrix();
      const Vector<3> p_WB = quad_state_.p;
      const Vector<3> p_WC = p_WB + R_WB * B_r_BC_;
      const Matrix<3, 3> R_WC = R_WB * R_BC_;
      const Vector<3> p_C_dbg =
        R_WC.transpose() * (tag_center_world_[0] - p_WC);
      Vector<2> uv_dbg;
      uv_dbg << (-cx_ - Scalar(1.0)), (-cy_ - Scalar(1.0));
      bool dot_in_front = false;
      bool dot_in_image = false;
      const bool dot_projected = projectWorldPointToImage(
        tag_center_world_[0], uv_dbg, &dot_in_front, &dot_in_image);
      logger_.info(
        "world pose | drone p_WB=[%.3f %.3f %.3f], camera p_WC=[%.3f %.3f %.3f]",
        p_WB.x(), p_WB.y(), p_WB.z(),
        p_WC.x(), p_WC.y(), p_WC.z());
      logger_.info(
        "R_WB | [%.4f %.4f %.4f; %.4f %.4f %.4f; %.4f %.4f %.4f]",
        R_WB(0, 0), R_WB(0, 1), R_WB(0, 2),
        R_WB(1, 0), R_WB(1, 1), R_WB(1, 2),
        R_WB(2, 0), R_WB(2, 1), R_WB(2, 2));
      logger_.info(
        "R_WC | [%.4f %.4f %.4f; %.4f %.4f %.4f; %.4f %.4f %.4f]",
        R_WC(0, 0), R_WC(0, 1), R_WC(0, 2),
        R_WC(1, 0), R_WC(1, 1), R_WC(1, 2),
        R_WC(2, 0), R_WC(2, 1), R_WC(2, 2));
      logger_.info(
        "dot uv | projected=%d in_front=%d in_image=%d uv=[%.2f %.2f]",
        static_cast<int>(dot_projected), static_cast<int>(dot_in_front),
        static_cast<int>(dot_in_image), uv_dbg.x(), uv_dbg.y());
      logger_.info(
        "dot p_C | [%.3f %.3f %.3f]",
        p_C_dbg.x(), p_C_dbg.y(), p_C_dbg.z());
    }
  }

  const Scalar sx = static_cast<Scalar>(quaddotenv::kImgWidth - 1) /
                    std::max(Scalar(1.0), static_cast<Scalar>(cam_width_ - 1));
  const Scalar sy = static_cast<Scalar>(quaddotenv::kImgHeight - 1) /
                    std::max(Scalar(1.0), static_cast<Scalar>(cam_height_ - 1));
  auto set_obs_from_center_uv = [&](const Vector<2> &uv_centered, int obs_x_idx) {
    const Scalar px = uv_centered.x() + cx_;
    const Scalar py = uv_centered.y() + cy_;
    quad_obs_(obs_x_idx) = px * sx;
    quad_obs_(obs_x_idx + 1) = py * sy;
  };

  for (int tag_idx = 0; tag_idx < quaddotenv::kNumTags; tag_idx++) {
    const int obs_base = quaddotenv::kObs + tag_idx * quaddotenv::kTagFeat;
    bool center_in_image = false;
    Vector<2> center_uv;
    center_uv.setZero();
    const bool center_projected = projectWorldPointToImage(
      tag_center_world_[tag_idx], center_uv, nullptr, &center_in_image);

    bool all_corners_projected = true;
    bool all_corners_in_image = true;
    Vector<2> corner_uv[4];
    for (int i = 0; i < 4; i++) {
      bool in_image = false;
      corner_uv[i].setZero();
      const bool projected = projectWorldPointToImage(
        tag_corner_world_[tag_idx].col(i), corner_uv[i], nullptr, &in_image);
      all_corners_projected = all_corners_projected && projected;
      all_corners_in_image = all_corners_in_image && in_image;
    }

    const bool tag_visible =
      center_projected && center_in_image && all_corners_projected && all_corners_in_image;
    if (tag_visible) {
      set_obs_from_center_uv(center_uv, obs_base + quaddotenv::kCenterX);
      set_obs_from_center_uv(corner_uv[0], obs_base + quaddotenv::kCorner0X);
      set_obs_from_center_uv(corner_uv[1], obs_base + quaddotenv::kCorner1X);
      set_obs_from_center_uv(corner_uv[2], obs_base + quaddotenv::kCorner2X);
      set_obs_from_center_uv(corner_uv[3], obs_base + quaddotenv::kCorner3X);
      quad_obs_(obs_base + quaddotenv::kTagId) = static_cast<Scalar>(tag_idx);
    }
  }

  cv::Mat rgb_image;
  if (rgb_camera_ != nullptr && rgb_camera_->getRGBImage(rgb_image) &&
      !rgb_image.empty()) {
    cv::Mat resized = rgb_image;
    if (rgb_image.cols != quaddotenv::kImgWidth ||
        rgb_image.rows != quaddotenv::kImgHeight) {
      cv::resize(rgb_image, resized,
                 cv::Size(quaddotenv::kImgWidth, quaddotenv::kImgHeight), 0.0, 0.0,
                 cv::INTER_AREA);
    }
    if (!resized.isContinuous()) resized = resized.clone();

    int flat_idx = quaddotenv::kImg;
    for (int r = 0; r < resized.rows; r++) {
      const cv::Vec3b *row = resized.ptr<cv::Vec3b>(r);
      for (int c = 0; c < resized.cols; c++) {
        quad_obs_(flat_idx++) = static_cast<Scalar>(row[c][0]);
        quad_obs_(flat_idx++) = static_cast<Scalar>(row[c][1]);
        quad_obs_(flat_idx++) = static_cast<Scalar>(row[c][2]);
      }
    }
  }

  obs.segment<quaddotenv::kNObs>(quaddotenv::kObs) = quad_obs_;
  return true;
}

void QuadrotorDotEnv::updateExtraInfo() {
  quadrotor_ptr_->getState(&quad_state_);
  extra_info_["drone_pos_x"] = quad_state_.x(QS::POSX);
  extra_info_["drone_pos_y"] = quad_state_.x(QS::POSY);
  extra_info_["drone_pos_z"] = quad_state_.x(QS::POSZ);
  extra_info_["drone_vel_x"] = quad_state_.x(QS::VELX);
  extra_info_["drone_vel_y"] = quad_state_.x(QS::VELY);
  extra_info_["drone_vel_z"] = quad_state_.x(QS::VELZ);

  extra_info_["reward_total"] = last_total_reward_;
  extra_info_["reward_xy"] = last_r_xy_;
  extra_info_["reward_vis"] = last_r_vis_;
  extra_info_["reward_center"] = last_r_center_;
  extra_info_["reward_area"] = last_r_area_;
  extra_info_["reward_shape"] = last_r_shape_;
  extra_info_["reward_shape2"] = last_r_shape2_;
  extra_info_["reward_area_small"] = last_r_area_small_;
  extra_info_["reward_smooth"] = last_r_smooth_;
  extra_info_["reward_invisible"] = last_r_invisible_;
  extra_info_["reward_switch"] = last_r_switch_;
  extra_info_["tag_visible"] = last_tag_visible_ ? 1.0f : 0.0f;
  extra_info_["corners_visible"] = last_corners_visible_ ? 1.0f : 0.0f;
  extra_info_["observed_area"] = last_observed_area_;
  extra_info_["stage"] = static_cast<float>(stage_);
  extra_info_["miss_count"] = static_cast<float>(miss_count_);
}

Scalar QuadrotorDotEnv::step(const Ref<Vector<>> act, Ref<Vector<>> obs) {
  quad_act_ = act.cwiseProduct(act_std_) + act_mean_;
  cmd_.t += sim_dt_;
  if (use_ctbr_) {
    cmd_.collective_thrust = quad_act_(0);
    cmd_.omega = quad_act_.segment<3>(1);
    cmd_.thrusts = Vector<4>::Constant(NAN);
  } else {
    cmd_.collective_thrust = NAN;
    cmd_.omega = Vector<3>::Constant(NAN);
    cmd_.thrusts = quad_act_;
  }

  // simulate quadrotor
  quadrotor_ptr_->run(cmd_, sim_dt_);

  // update observations
  getObs(obs);

  if (!stage_switch_enabled_) {
    stage_ = 0;
  }

  // ---------------------- stage-based multi-tag reward (largest -> middle -> smallest)
  const Scalar half_w = static_cast<Scalar>(quaddotenv::kImgWidth - 1) * 0.5;
  const Scalar half_h = static_cast<Scalar>(quaddotenv::kImgHeight - 1) * 0.5;
  const Scalar eps = 1e-6;
  const int active_slot = std::max(0, std::min(stage_, quaddotenv::kNumTags - 1)); //현재 stage에 해당하는 QR코드 인덱스 (0, 1, 2)
  const int active_tag_idx = stage_switch_enabled_ ? tag_order_[active_slot] : 0; //stage switch 비활성화 시 항상 tag 0 사용
  //tag_order_ :  stage마다 reward 계산에 사용할 QR 태그의 순서를 정의하는 배열. 예를 들어, tag_order_ = {2, 0, 1}이면, stage 0에서는 tag 2가 active_tag_idx가 되고, stage 1에서는 tag 0이 active_tag_idx가 되고, stage 2에서는 tag 1이 active_tag_idx가 됨
  const int active_base = active_tag_idx * quaddotenv::kTagFeat; //active_base : quad_obs_에서 현재 active_tag_idx에 해당하는 QR 코드 관측치의 시작 인덱스. 
  // 예를 들어, active_tag_idx가 1이면, active_base는 1 * kTagFeat가 되어, quad_obs_에서 tag 1의 관측치가 시작되는 인덱스를 가리킴.
  //kTagFeat = QR 1개당 필요한 관측치 수 (center + 4 corners + tag id)- 11개.

  const Scalar tag_id = quad_obs_(active_base + quaddotenv::kTagId); //tag 별로 tag_id
  //kTagId : QR 1개 안에서 tag_id 가 있는 인덱스(위치)/, 11(0-10)차원중 10에 있음
  const bool tag_visible = tag_id >= 0.0; //tag_id 가 양수이면 tag_visible true 
  const bool corners_visible =
    quad_obs_(active_base + quaddotenv::kCorner0X) >= 0.0 && //코너0의 x 좌표가 양수인지 (즉, 관측치에 유효한 값이 있는지). 모든 코너는 0-83 사이의 값 
    quad_obs_(active_base + quaddotenv::kCorner0Y) >= 0.0 &&
    quad_obs_(active_base + quaddotenv::kCorner1X) >= 0.0 &&
    quad_obs_(active_base + quaddotenv::kCorner1Y) >= 0.0 &&
    quad_obs_(active_base + quaddotenv::kCorner2X) >= 0.0 &&
    quad_obs_(active_base + quaddotenv::kCorner2Y) >= 0.0 &&
    quad_obs_(active_base + quaddotenv::kCorner3X) >= 0.0 &&
    quad_obs_(active_base + quaddotenv::kCorner3Y) >= 0.0;
    //quad_obs_ = QuadrotorDotEnv 내부에서 쓰는 관측 벡터 버퍼 . Vector<quaddotenv::kNObs>. 태그 투영값들 + RGB 펼친 값
    //매 step/reset 때 getObs()에서 채운 뒤, 최종 obs로 복사됩니다.

  Scalar r_vis = tag_visible ? 1.0 : -1.0;
  const Scalar xy_error =
    (quad_state_.p.head<2>() - tag_center_world_[active_tag_idx].head<2>()).norm();
  Scalar r_xy = -(xy_error * xy_error);
  Scalar r_center = 0.0; // tag center - 이미지 중심 거리. 가까울수록 penalty 줄어듬 
  Scalar r_area = 0.0; // tag area 지수형 보상. target area에 가까울수록 1에 가까움
  Scalar r_shape = 0.0; // tag shape - edge length 균일성, 대각선 길이 균일성, 직각 정도, 작은 면적 패널티 종합. 실제 tag 모양이 정사각형에 가까울수록 penalty 줄어듬
  Scalar r_shape2 = 0.0; // tag axis alignment - 변이 이미지 x/y 축과 평행할수록 보상
  Scalar r_area_small = 0.0; // tag_min_area 미만일 때만 적용되는 별도 패널티
  Scalar r_smooth = -act.cast<Scalar>().squaredNorm(); // 행동의 크기에 대한 패널티. 작은 행동일수록 penalty 줄어듬 (즉, 행동이 너무 크면 패널티가 커짐)
  Scalar r_invisible = 0.0; // 태그가 보이지 않을 때 패널티. 보이지 않을수록 penalty 커짐
  Scalar observed_area = -1.0; // active tag 면적(관측 불가 시 -1)

  if (tag_visible && corners_visible) {
    const Scalar cx = quad_obs_(active_base + quaddotenv::kCenterX); //tag center의 x 좌표
    const Scalar cy = quad_obs_(active_base + quaddotenv::kCenterY); //tag center의 y 좌표
    const Scalar ex = (cx - half_w) / std::max(half_w, eps); //tag center의 x 좌표가 이미지 중심에서 멀어질수록 ex의 절댓값이 커짐. half_w로 나누어서 정규화 (0~1 사이). eps는 0으로 나누는 것을 방지하기 위한 작은 값
    const Scalar ey = (cy - half_h) / std::max(half_h, eps); //tag center의 y 좌표가 이미지 중심에서 멀어질수록 ey의 절댓값이 커짐. half_h로 나누어서 정규화 (0~1 사이). eps는 0으로 나누는 것을 방지하기 위한 작은 값
    const Scalar e_center = std::sqrt(ex * ex + ey * ey);
    r_center = -e_center;

    const Scalar x0 = quad_obs_(active_base + quaddotenv::kCorner0X); //코너0의 x 좌표
    const Scalar y0 = quad_obs_(active_base + quaddotenv::kCorner0Y); //코너0의 y 좌표
    const Scalar x1 = quad_obs_(active_base + quaddotenv::kCorner1X);
    const Scalar y1 = quad_obs_(active_base + quaddotenv::kCorner1Y);
    const Scalar x2 = quad_obs_(active_base + quaddotenv::kCorner2X);
    const Scalar y2 = quad_obs_(active_base + quaddotenv::kCorner2Y);
    const Scalar x3 = quad_obs_(active_base + quaddotenv::kCorner3X);
    const Scalar y3 = quad_obs_(active_base + quaddotenv::kCorner3Y);

    const Scalar area_twice =
      x0 * y1 + x1 * y2 + x2 * y3 + x3 * y0 -
      (y0 * x1 + y1 * x2 + y2 * x3 + y3 * x0); //사각형의 면적을 구하는 공식. (x0,y0), (x1,y1), (x2,y2), (x3,y3)가 사각형의 네 꼭짓점 좌표일 때, 공식에서 나오는 값은 실제 면적의 2배가 되므로, 최종적으로는 절댓값을 취한 후 0.5를 곱하여 실제 면적을 구합니다.
    const Scalar area = std::abs(area_twice) * 0.5;
    observed_area = area;
    last_visible_area_ = area;
    const Scalar target_area = std::max(stage_target_area_[active_slot], Scalar(1.0)); //stage마다 다른 target area 설정. target area가 0이 되는 것을 방지하기 위해 최소값을 1.0으로 설정
    const Scalar area_err =
      std::abs(area - target_area) / std::max(target_area, eps); //target 대비 상대 오차
    constexpr Scalar kAreaExpScale = 3.0; //클수록 target 근처에서만 높은 보상
    r_area = std::exp(-kAreaExpScale * area_err);

    //std::hypot(a, b) = sqrt(a*a + b*b)
    const Scalar l01 = std::hypot(x1 - x0, y1 - y0); //코너0과 코너1 사이의 거리 (edge length)
    const Scalar l12 = std::hypot(x2 - x1, y2 - y1); //코너1과 코너2 사이의 거리 (edge length)
    const Scalar l23 = std::hypot(x3 - x2, y3 - y2); //코너2와 코너3 사이의 거리 (edge length)
    const Scalar l30 = std::hypot(x0 - x3, y0 - y3); //코너3과 코너0 사이의 거리 (edge length)
    const Scalar d02 = std::hypot(x2 - x0, y2 - y0); //코너0과 코너2 사이의 거리 (대각선 길이)
    const Scalar d13 = std::hypot(x3 - x1, y3 - y1); //코너1과 코너3 사이의 거리 (대각선 길이)
    const Scalar e_edge_adj =
      std::abs(l01 - l12) / (l01 + l12 + eps) +
      std::abs(l12 - l23) / (l12 + l23 + eps) +
      std::abs(l23 - l30) / (l23 + l30 + eps) +
      std::abs(l30 - l01) / (l30 + l01 + eps); //인접 edge 길이의 차이에 대한 패널티. 인접 edge 길이가 비슷할수록 패널티가 작아짐. eps는 0으로 나누는 것을 방지하기 위한 작은 값
    const Scalar e_edge_opp =
      std::abs(l01 - l23) / (l01 + l23 + eps) +
      std::abs(l12 - l30) / (l12 + l30 + eps); //반대편 edge 길이의 차이에 대한 패널티. 마주보는 edge 길이가 비슷할수록 패널티가 작아짐. eps는 0으로 나누는 것을 방지하기 위한 작은 값
    const Scalar e_diag = std::abs(d02 - d13) / (d02 + d13 + eps); //대각선 길이의 차이에 대한 패널티. d02와 d13은 서로 마주보는 대각선. 마주보는 대각선 길이가 비슷할수록 패널티가 작아짐. eps는 0으로 나누는 것을 방지하기 위한 작은 값
    const Scalar l_mean = (l01 + l12 + l23 + l30) * Scalar(0.25); //edge 길이의 평균. edge 길이들이 평균에 가까울수록 패널티가 작아짐. eps는 0으로 나누는 것을 방지하기 위한 작은 값
    const Scalar e_edge_all =
      (std::abs(l01 - l_mean) + std::abs(l12 - l_mean) +
       std::abs(l23 - l_mean) + std::abs(l30 - l_mean)) /
      (l01 + l12 + l23 + l30 + eps); //모든 edge 길이가 평균 edge 길이에서 벗어나는 정도에 대한 패널티. 모든 edge 길이가 평균에 가까울수록 패널티가 작아짐. eps는 0으로 나누는 것을 방지하기 위한 작은 값
    const Scalar v01x = x1 - x0, v01y = y1 - y0; //코너0에서 코너1로 향하는 벡터의 x와 y 성분. edge 벡터
    const Scalar v12x = x2 - x1, v12y = y2 - y1; //코너1에서 코너2로 향하는 벡터의 x와 y 성분. edge 벡터
    const Scalar v23x = x3 - x2, v23y = y3 - y2; //코너2에서 코너3로 향하는 벡터의 x와 y 성분. edge 벡터
    const Scalar v30x = x0 - x3, v30y = y0 - y3; //코너3에서 코너0로 향하는 벡터의 x와 y 성분. edge 벡터
    const Scalar c0 = std::abs(v01x * v30x + v01y * v30y) / (l01 * l30 + eps); //코너0에서 코너1로 향하는 벡터와 코너3에서 코너0로 향하는 벡터의 내적을 edge 길이의 곱으로 나눈 값. 두 벡터가 직각에 가까울수록 패널티가 작아짐. eps는 0으로 나누는 것을 방지하기 위한 작은 값
    const Scalar c1 = std::abs(v12x * v01x + v12y * v01y) / (l12 * l01 + eps); //코너1에서 코너2로 향하는 벡터와 코너0에서 코너1로 향하는 벡터의 내적을 edge 길이의 곱으로 나눈 값. 두 벡터가 직각에 가까울수록 패널티가 작아짐. eps는 0으로 나누는 것을 방지하기 위한 작은 값
    const Scalar c2 = std::abs(v23x * v12x + v23y * v12y) / (l23 * l12 + eps); //코너2에서 코너3로 향하는 벡터와 코너1에서 코너2로 향하는 벡터의 내적을 edge 길이의 곱으로 나눈 값. 두 벡터가 직각에 가까울수록 패널티가 작아짐. eps는 0으로 나누는 것을 방지하기 위한 작은 값
    const Scalar c3 = std::abs(v30x * v23x + v30y * v23y) / (l30 * l23 + eps); //코너3에서 코너0로 향하는 벡터와 코너2에서 코너3로 향하는 벡터의 내적을 edge 길이의 곱으로 나눈 값. 두 벡터가 직각에 가까울수록 패널티가 작아짐. eps는 0으로 나누는 것을 방지하기 위한 작은 값
    const Scalar e_right_angle = (c0 + c1 + c2 + c3) * Scalar(0.25); //네 코너에서의 직각 정도에 대한 패널티. 네 코너 모두에서 직각에 가까울수록 패널티가 작아짐. eps는 0으로 나누는 것을 방지하기 위한 작은 값
    const Scalar min_area = tag_min_area_;
    const Scalar e_area_small =
      std::max(Scalar(0.0), (min_area - area) / std::max(min_area, eps)); //태그 면적이 너무 작은 경우에 대한 패널티. 면적이 min_area보다 작을수록 패널티가 커짐. min_area로 나누어서 정규화. eps는 0으로 나누는 것을 방지하기 위한 작은 값
    r_area_small = -e_area_small;
    const Scalar e_shape =
      e_edge_adj + e_edge_opp + e_diag + e_edge_all + e_right_angle;
    r_shape = -e_shape;

    // axis alignment: each edge should be parallel to either image x-axis or y-axis
    const Scalar u01x = v01x / (l01 + eps), u01y = v01y / (l01 + eps); 
    //v01x : 코너0에서 코너1로 향하는 벡터의 x 성분. 
    //l01 : 코너0에서 코너1 사이의 거리 (edge length)
    //u01x, u01y : 코너0에서 코너1로 향하는 단위 벡터의 x와 y 성분. edge 벡터를 edge 길이로 나누어서 정규화. eps는 0으로 나누는 것을 방지
    const Scalar u12x = v12x / (l12 + eps), u12y = v12y / (l12 + eps);
    const Scalar u23x = v23x / (l23 + eps), u23y = v23y / (l23 + eps);
    const Scalar u30x = v30x / (l30 + eps), u30y = v30y / (l30 + eps);
    // edge 01 is explicitly encouraged to align with image x-axis
    const Scalar a01 = std::abs(u01y); //edge 01 -> x-axis 평행 한지. 0 일수록 좋음 
    const Scalar a12 = std::abs(u12x); // edge 12 -> y-axis 평행 하지 
    const Scalar a23 = std::abs(u23y); // edge 23 -> x-axis 평행 한지
    const Scalar a30 = std::abs(u30x); // edge 30 -> y-axis 평행 한지 
    const Scalar e_axis_align = (a01 + a12 + a23 + a30) * Scalar(0.25);
    r_shape2 = -e_axis_align;
  }
  if (!(tag_visible && corners_visible)) {
    if (miss_count_ == 0) {
      // area on the last visible step before this invisible streak starts
      miss_start_prev_area_ = last_visible_area_;
    }
    const Scalar miss_steps = static_cast<Scalar>(miss_count_ + 1); //태그가 보이지 않는 상태가 몇 step 지속되었는지를 나타내는 값. miss_count_는 현재까지 태그가 보이지 않는 상태가 지속된 step 수를 카운트하는 변수. 여기에 1을 더하는 이유는 현재 step도 포함하기 위함. 태그가 보이지 않는 상태가 지속될수록 miss_steps의 값이 커지며, 이를 통해 r_invisible에 점점 더 큰 패널티를 주게 됨
    const Scalar stage_scale = static_cast<Scalar>(stage_); //현재 stage에 대한 스케일 값. stage_는 현재 stage를 나타내는 변수. 이 값을 사용하여 r_invisible에 stage에 따라 다른 패널티를 주게 됨
    const Scalar invisible_cost = (
      invisible_base_penalty_ +
      invisible_miss_penalty_ * miss_steps +
      invisible_stage_penalty_ * stage_scale
    );
    r_invisible = -invisible_cost;
    if (miss_start_prev_area_ >= invisible_positive_area_threshold_) {
      r_invisible = invisible_cost;
    }
  }

  // keep miss_count_ for invisible penalties
  if (!corners_visible) {
    miss_count_++;
  } else {
    miss_count_ = 0;
    miss_start_prev_area_ = -1.0;
  }
  Scalar r_switch = 0.0;
  if (stage_switch_enabled_ && stage_ < (quaddotenv::kNumTags - 1)) { //현재 stage가 마지막 stage보다 작은 경우에만 다음 stage로 넘어갈 수 있는지 평가. 마지막 stage에서는 다음 stage가 없으므로, 다음 stage로 넘어갈 수 있는지 평가할 필요가 없음
    bool can_advance = false;
    {
      const bool active_corners_visible =
        quad_obs_(active_base + quaddotenv::kCorner0X) >= 0.0 &&
        quad_obs_(active_base + quaddotenv::kCorner0Y) >= 0.0 &&
        quad_obs_(active_base + quaddotenv::kCorner1X) >= 0.0 &&
        quad_obs_(active_base + quaddotenv::kCorner1Y) >= 0.0 &&
        quad_obs_(active_base + quaddotenv::kCorner2X) >= 0.0 &&
        quad_obs_(active_base + quaddotenv::kCorner2Y) >= 0.0 &&
        quad_obs_(active_base + quaddotenv::kCorner3X) >= 0.0 &&
        quad_obs_(active_base + quaddotenv::kCorner3Y) >= 0.0;
      if (active_corners_visible) { //4개 코너가 모두 보이는 경우에만 다음 stage로 넘어갈 수 있는지 평가. 4개 코너 중 하나라도 보이지 않으면 다음 stage로 넘어갈 수 없음
        const Scalar ax0 = quad_obs_(active_base + quaddotenv::kCorner0X);
        const Scalar ay0 = quad_obs_(active_base + quaddotenv::kCorner0Y);
        const Scalar ax1 = quad_obs_(active_base + quaddotenv::kCorner1X);
        const Scalar ay1 = quad_obs_(active_base + quaddotenv::kCorner1Y);
        const Scalar ax2 = quad_obs_(active_base + quaddotenv::kCorner2X);
        const Scalar ay2 = quad_obs_(active_base + quaddotenv::kCorner2Y);
        const Scalar ax3 = quad_obs_(active_base + quaddotenv::kCorner3X);
        const Scalar ay3 = quad_obs_(active_base + quaddotenv::kCorner3Y);
        const Scalar active_area_twice =
          ax0 * ay1 + ax1 * ay2 + ax2 * ay3 + ax3 * ay0 -
          (ay0 * ax1 + ay1 * ax2 + ay2 * ax3 + ay3 * ax0);
        const Scalar active_area = std::abs(active_area_twice) * 0.5;
        can_advance = active_area > 50.0; //현재 stage의 QR 코드가 충분히 크게 보이는 경우에만 다음 stage로 넘어갈 수 있도록 하는 조건. 
      }
    }
    if (can_advance) {
      stage_++;
      miss_count_ = 0; //stage가 바뀌면 태그가 보이지 않는 상태도 초기화
      miss_start_prev_area_ = -1.0;
      r_switch = stage_switch_bonus_;   
    }
  }

  Scalar total_reward = 0.0;
  total_reward +=
    landing_w_xy_ * r_xy +
    tag_vis_coeff_ * r_vis +
    tag_center_coeff_ * r_center +
    tag_area_coeff_ * r_area +
    tag_shape_coeff_ * r_shape +
    tag_shape2_coeff_ * r_shape2 +
    tag_area_small_coeff_ * r_area_small +
    tag_smooth_coeff_ * r_smooth +
    r_invisible +
    r_switch;

  last_total_reward_ = total_reward;
  last_r_xy_ = landing_w_xy_ * r_xy;
  last_r_vis_ = tag_vis_coeff_ * r_vis;
  last_r_center_ = tag_center_coeff_ * r_center;
  last_r_area_ = tag_area_coeff_ * r_area;
  last_r_shape_ = tag_shape_coeff_ * r_shape;
  last_r_shape2_ = tag_shape2_coeff_ * r_shape2;
  last_r_area_small_ = tag_area_small_coeff_ * r_area_small;
  last_r_smooth_ = tag_smooth_coeff_ * r_smooth;
  last_r_invisible_ = r_invisible;
  last_r_switch_ = r_switch;
  last_observed_area_ = observed_area;
  last_tag_visible_ = tag_visible;
  last_corners_visible_ = corners_visible;

  log_counter_++;
  if (log_counter_ % std::max(1, log_interval_steps_) == 0) {
    logger_.info(
      "quad pos | x=%.3f y=%.3f z=%.3f",
      quad_state_.x(QS::POSX), quad_state_.x(QS::POSY), quad_state_.x(QS::POSZ));
    logger_.info(
      "tag area | stage=%d active_tag=%d visible=%d corners=%d area=%.3f target=%.3f",
      stage_, active_tag_idx, static_cast<int>(tag_visible),
      static_cast<int>(corners_visible), observed_area,
      std::max(stage_target_area_[active_slot], Scalar(1.0)));
    logger_.info(
      "reward | total=%.4f xy=%.4f vis=%.4f center=%.4f area=%.4f shape=%.4f shape2=%.4f area_small=%.4f smooth=%.4f invisible=%.4f switch=%.4f",
      total_reward, landing_w_xy_ * r_xy, tag_vis_coeff_ * r_vis,
      tag_center_coeff_ * r_center, tag_area_coeff_ * r_area,
      tag_shape_coeff_ * r_shape, tag_shape2_coeff_ * r_shape2,
      tag_area_small_coeff_ * r_area_small, tag_smooth_coeff_ * r_smooth,
      r_invisible, r_switch);
  }

  return total_reward;
}

bool QuadrotorDotEnv::isTerminalState(Scalar &reward) {
  const bool hit_world_box =
    (quad_state_.x(QS::POSX) <= world_box_(0, 0)+0.001) ||
    (quad_state_.x(QS::POSX) >= world_box_(0, 1)-0.001) ||
    (quad_state_.x(QS::POSY) <= world_box_(1, 0)+0.001) ||
    (quad_state_.x(QS::POSY) >= world_box_(1, 1)-0.001) ||
    (quad_state_.x(QS::POSZ) <= world_box_(2, 0)+0.001) ||
    (quad_state_.x(QS::POSZ) >= world_box_(2, 1)-0.001);
  if (hit_world_box) {
    reward = landing_failure_reward_;
    return true;
  }

  // Early terminate on excessive tilt (flip-like behavior).
  {
    const Vector<3> euler_zyx =
      quad_state_.q().toRotationMatrix().eulerAngles(2, 1, 0);
    const Scalar tilt = std::sqrt(
      euler_zyx(1) * euler_zyx(1) + euler_zyx(2) * euler_zyx(2));
    if (tilt > landing_tilt_hard_) {
      reward = landing_tilt_hard_penalty_;
      return true;
    }
  }

  // Ground plane is assumed around z=3.0.
  if (quad_state_.x(QS::POSZ) <= landing_terminal_z_) {
    const int terminal_tag_idx =
      stage_switch_enabled_ ? tag_order_[quaddotenv::kNumTags - 1] : 0;
    const Scalar xy_error =
      (quad_state_.p.head<2>() - tag_center_world_[terminal_tag_idx].head<2>()).norm();
    const Scalar vxy = quad_state_.v.head<2>().norm();
    const Scalar vz = std::abs(quad_state_.x(QS::VELZ));
    const Scalar body_rate = quad_state_.w.norm();
    const Vector<3> euler_zyx =
      quad_state_.q().toRotationMatrix().eulerAngles(2, 1, 0);
    const Scalar tilt = std::sqrt(
      euler_zyx(1) * euler_zyx(1) + euler_zyx(2) * euler_zyx(2));

    const bool success = (xy_error < landing_success_xy_error_) &&
                         (vxy < landing_success_vxy_) &&
                         (vz < landing_success_vz_) &&
                         (tilt < landing_success_tilt_) &&
                         (body_rate < landing_success_body_rate_);
    reward = success ? landing_success_reward_ : landing_failure_reward_;
    return true;
  }
  reward = 0.0;
  return false;
}

bool QuadrotorDotEnv::loadParam(const YAML::Node &cfg) {
  if (cfg["quadrotor_env"]) {
    sim_dt_ = cfg["quadrotor_env"]["sim_dt"].as<Scalar>();
    max_t_ = cfg["quadrotor_env"]["max_t"].as<Scalar>();
    if (cfg["quadrotor_env"]["init_pos"]) {
      const std::vector<Scalar> init_pos =
        cfg["quadrotor_env"]["init_pos"].as<std::vector<Scalar>>();
      if (init_pos.size() == 3) {
        init_pos_ = Map<const Vector<3>>(init_pos.data());
      } else {
        logger_.warn("init_pos must have 3 elements. Using [0,0,20].");
      }
    }
    if (cfg["quadrotor_env"]["randomize_position_on_reset"]) {
      randomize_position_on_reset_ =
        cfg["quadrotor_env"]["randomize_position_on_reset"].as<bool>();
    }
    if (cfg["quadrotor_env"]["randomize_position_scale"]) {
      randomize_position_scale_ =
        cfg["quadrotor_env"]["randomize_position_scale"].as<Scalar>();
      if (randomize_position_scale_ < 0.0) {
        logger_.warn("randomize_position_scale must be >= 0. Using 1.0.");
        randomize_position_scale_ = 1.0;
      }
    }
    if (cfg["quadrotor_env"]["randomize_velocity_on_reset"]) {
      randomize_velocity_on_reset_ =
        cfg["quadrotor_env"]["randomize_velocity_on_reset"].as<bool>();
    }
    if (cfg["quadrotor_env"]["randomize_velocity_scale"]) {
      randomize_velocity_scale_ =
        cfg["quadrotor_env"]["randomize_velocity_scale"].as<Scalar>();
      if (randomize_velocity_scale_ < 0.0) {
        logger_.warn("randomize_velocity_scale must be >= 0. Using 1.0.");
        randomize_velocity_scale_ = 1.0;
      }
    }
    if (cfg["quadrotor_env"]["randomize_attitude_on_reset"]) {
      randomize_attitude_on_reset_ =
        cfg["quadrotor_env"]["randomize_attitude_on_reset"].as<bool>();
    }
    if (cfg["quadrotor_env"]["randomize_attitude_scale"]) {
      randomize_attitude_scale_ =
        cfg["quadrotor_env"]["randomize_attitude_scale"].as<Scalar>();
      if (randomize_attitude_scale_ < 0.0) {
        logger_.warn("randomize_attitude_scale must be >= 0. Using 1.0.");
        randomize_attitude_scale_ = 1.0;
      }
    }

    if (cfg["quadrotor_env"]["camera"]) {
      const YAML::Node cam_cfg = cfg["quadrotor_env"]["camera"];
      if (parseTagCenters(cam_cfg["tag_centers_world"], &tag_center_world_)) {
        goal_pos_ = tag_center_world_[0];
      } else if (cam_cfg["tag_centers_world"]) {
        logger_.warn("tag_centers_world must be [[x,y,z] x3]. Using existing values.");
      }
      if (parseTagCorners(cam_cfg["tag_corners_world"], &tag_corner_world_)) {
        // parsed successfully
      } else if (cam_cfg["tag_corners_world"]) {
        logger_.warn("tag_corners_world must be [[[x,y,z] x4] x3]. Using existing values.");
      }
      // Backward compatibility for single-tag keys.
      if (cam_cfg["dot_world_pos"] && cam_cfg["dot_world_pos"].IsSequence() &&
          cam_cfg["dot_world_pos"].size() == 3) {
        tag_center_world_[0] << cam_cfg["dot_world_pos"][0].as<Scalar>(),
          cam_cfg["dot_world_pos"][1].as<Scalar>(),
          cam_cfg["dot_world_pos"][2].as<Scalar>();
        goal_pos_ = tag_center_world_[0];
      }
      if (cam_cfg["tag_center_world"] && cam_cfg["tag_center_world"].IsSequence() &&
          cam_cfg["tag_center_world"].size() == 3) {
        tag_center_world_[0] << cam_cfg["tag_center_world"][0].as<Scalar>(),
          cam_cfg["tag_center_world"][1].as<Scalar>(),
          cam_cfg["tag_center_world"][2].as<Scalar>();
        goal_pos_ = tag_center_world_[0];
      }
      if (cam_cfg["tag_corner_world"] && cam_cfg["tag_corner_world"].IsSequence() &&
          cam_cfg["tag_corner_world"].size() == 4) {
        bool corners_valid = true;
        for (int i = 0; i < 4; i++) {
          const YAML::Node corner = cam_cfg["tag_corner_world"][i];
          if (!corner.IsSequence() || corner.size() != 3) {
            corners_valid = false;
            break;
          }
          tag_corner_world_[0].col(i) << corner[0].as<Scalar>(),
            corner[1].as<Scalar>(), corner[2].as<Scalar>();
        }
        if (!corners_valid) {
          logger_.warn("tag_corner_world must be [[x,y,z] x4]. Using existing values.");
        }
      }
    }
    if (cfg["quadrotor_env"]["camera"] &&
        cfg["quadrotor_env"]["camera"]["log_world_pose"]) {
      log_world_pose_ =
        cfg["quadrotor_env"]["camera"]["log_world_pose"].as<bool>();
    }
    if (cfg["quadrotor_env"]["camera"] &&
        cfg["quadrotor_env"]["camera"]["log_world_pose_interval_steps"]) {
      log_world_pose_interval_steps_ = std::max(
        1,
        cfg["quadrotor_env"]["camera"]["log_world_pose_interval_steps"].as<int>());
    }
    // Sort tags by physical size (largest -> smallest) for stage progression.
    std::array<std::pair<Scalar, int>, quaddotenv::kNumTags> scales;
    for (int i = 0; i < quaddotenv::kNumTags; i++) {
      scales[i] = {estimateTagScale(tag_corner_world_[i]), i};
    }
    std::sort(scales.begin(), scales.end(),
              [](const std::pair<Scalar, int> &a, const std::pair<Scalar, int> &b) {
                return a.first > b.first;
              });
    for (int i = 0; i < quaddotenv::kNumTags; i++) {
      tag_order_[i] = scales[i].second;
    }
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
    if (cfg["rl"]["landing_w_xy"]) {
      landing_w_xy_ = cfg["rl"]["landing_w_xy"].as<Scalar>();
    }
    if (cfg["rl"]["landing_w_z"]) landing_w_z_ = cfg["rl"]["landing_w_z"].as<Scalar>();
    // Backward compatibility: shared velocity weights set both near/far terms.
    if (cfg["rl"]["landing_w_vel_xy"]) {
      const Scalar w = cfg["rl"]["landing_w_vel_xy"].as<Scalar>();
      landing_w_vel_xy_near_ = w;
      landing_w_vel_xy_far_ = w;
    }
    if (cfg["rl"]["landing_w_vel_z"]) {
      const Scalar w = cfg["rl"]["landing_w_vel_z"].as<Scalar>();
      landing_w_vel_z_near_ = w;
      landing_w_vel_z_far_ = w;
    }
    if (cfg["rl"]["landing_w_vel_xy_near"]) {
      landing_w_vel_xy_near_ = cfg["rl"]["landing_w_vel_xy_near"].as<Scalar>();
    }
    if (cfg["rl"]["landing_w_vel_xy_far"]) {
      landing_w_vel_xy_far_ = cfg["rl"]["landing_w_vel_xy_far"].as<Scalar>();
    }
    if (cfg["rl"]["landing_w_vel_z_near"]) {
      landing_w_vel_z_near_ = cfg["rl"]["landing_w_vel_z_near"].as<Scalar>();
    }
    if (cfg["rl"]["landing_w_vel_z_far"]) {
      landing_w_vel_z_far_ = cfg["rl"]["landing_w_vel_z_far"].as<Scalar>();
    }
    if (cfg["rl"]["landing_near_ground_z"]) {
      landing_near_ground_z_ = cfg["rl"]["landing_near_ground_z"].as<Scalar>();
    }

    if (cfg["rl"]["landing_w_tilt"]) {
      landing_w_tilt_ = cfg["rl"]["landing_w_tilt"].as<Scalar>();
    }
    if (cfg["rl"]["landing_tilt_hard"]) {
      landing_tilt_hard_ = cfg["rl"]["landing_tilt_hard"].as<Scalar>();
    }
    if (cfg["rl"]["landing_tilt_hard_penalty"]) {
      landing_tilt_hard_penalty_ =
        cfg["rl"]["landing_tilt_hard_penalty"].as<Scalar>();
    }
    if (cfg["rl"]["landing_w_yaw"]) {
      landing_w_yaw_ = cfg["rl"]["landing_w_yaw"].as<Scalar>();
    }

    if (cfg["rl"]["landing_time_penalty"]) {
      landing_time_penalty_ = cfg["rl"]["landing_time_penalty"].as<Scalar>();
    }
    if (cfg["rl"]["landing_w_body_rate"]) {
      landing_w_body_rate_ = cfg["rl"]["landing_w_body_rate"].as<Scalar>();
    }
    if (cfg["rl"]["landing_w_rate_cmd_xy"]) {
      landing_w_rate_cmd_xy_ = cfg["rl"]["landing_w_rate_cmd_xy"].as<Scalar>();
    }
    if (cfg["rl"]["landing_w_duv"]) {
      landing_w_duv_ = cfg["rl"]["landing_w_duv"].as<Scalar>();
    }
    if (cfg["rl"]["landing_z_safe_margin"]) {
      landing_z_safe_margin_ = cfg["rl"]["landing_z_safe_margin"].as<Scalar>();
    }

    if (cfg["rl"]["landing_w_img_center"]) {
      landing_w_img_center_ = cfg["rl"]["landing_w_img_center"].as<Scalar>();
    }
    if (cfg["rl"]["landing_dot_not_visible_penalty"]) {
      landing_dot_not_visible_penalty_ =
        cfg["rl"]["landing_dot_not_visible_penalty"].as<Scalar>();
    }
    if (cfg["rl"]["tag_vis_coeff"]) {
      tag_vis_coeff_ = cfg["rl"]["tag_vis_coeff"].as<Scalar>();
    }
    if (cfg["rl"]["tag_center_coeff"]) {
      tag_center_coeff_ = cfg["rl"]["tag_center_coeff"].as<Scalar>();
    }
    if (cfg["rl"]["tag_area_coeff"]) {
      tag_area_coeff_ = cfg["rl"]["tag_area_coeff"].as<Scalar>();
    }
    if (cfg["rl"]["tag_shape_coeff"]) {
      tag_shape_coeff_ = cfg["rl"]["tag_shape_coeff"].as<Scalar>();
    }
    if (cfg["rl"]["tag_shape2_coeff"]) {
      tag_shape2_coeff_ = cfg["rl"]["tag_shape2_coeff"].as<Scalar>();
    }
    if (cfg["rl"]["tag_area_small_coeff"]) {
      tag_area_small_coeff_ = cfg["rl"]["tag_area_small_coeff"].as<Scalar>();
    }
    if (cfg["rl"]["tag_smooth_coeff"]) {
      tag_smooth_coeff_ = cfg["rl"]["tag_smooth_coeff"].as<Scalar>();
    }
 
    
    if (cfg["rl"]["tag_min_area"]) {
      tag_min_area_ = cfg["rl"]["tag_min_area"].as<Scalar>();
    }
    if (cfg["rl"]["stage_switch_enabled"]) {
      stage_switch_enabled_ = cfg["rl"]["stage_switch_enabled"].as<bool>();
    }
    if (cfg["rl"]["stage_miss_threshold"]) {
      stage_miss_threshold_ = std::max(1, cfg["rl"]["stage_miss_threshold"].as<int>());
    }
    if (cfg["rl"]["stage_require_next_visible"]) {
      stage_require_next_visible_ = cfg["rl"]["stage_require_next_visible"].as<bool>();
    }
    if (cfg["rl"]["stage_switch_bonus"]) {
      stage_switch_bonus_ = cfg["rl"]["stage_switch_bonus"].as<Scalar>();
    }
    if (cfg["rl"]["stage_target_area"] &&
        cfg["rl"]["stage_target_area"].IsSequence() &&
        cfg["rl"]["stage_target_area"].size() == quaddotenv::kNumTags) {
      for (int i = 0; i < quaddotenv::kNumTags; i++) {
        stage_target_area_[i] = cfg["rl"]["stage_target_area"][i].as<Scalar>();
      }
    }
    if (cfg["rl"]["invisible_base_penalty"]) {
      invisible_base_penalty_ = std::max(Scalar(0.0),
                                         cfg["rl"]["invisible_base_penalty"].as<Scalar>());
    }
    if (cfg["rl"]["invisible_miss_penalty"]) {
      invisible_miss_penalty_ = std::max(Scalar(0.0),
                                         cfg["rl"]["invisible_miss_penalty"].as<Scalar>());
    }
    if (cfg["rl"]["invisible_stage_penalty"]) {
      invisible_stage_penalty_ = std::max(Scalar(0.0),
                                          cfg["rl"]["invisible_stage_penalty"].as<Scalar>());
    }
    if (cfg["rl"]["invisible_positive_area_threshold"]) {
      invisible_positive_area_threshold_ = std::max(
        Scalar(0.0), cfg["rl"]["invisible_positive_area_threshold"].as<Scalar>());
    }
    if (cfg["rl"]["landing_center_u_gate"]) {
      landing_center_u_gate_ = cfg["rl"]["landing_center_u_gate"].as<Scalar>();
    }
    if (cfg["rl"]["landing_center_v_gate"]) {
      landing_center_v_gate_ = cfg["rl"]["landing_center_v_gate"].as<Scalar>();
    }
    if (cfg["rl"]["landing_center_hold_bonus"]) {
      landing_center_hold_bonus_ = cfg["rl"]["landing_center_hold_bonus"].as<Scalar>();
    }

    if (cfg["rl"]["landing_w_early_descend"]) {
      landing_w_early_descend_ = cfg["rl"]["landing_w_early_descend"].as<Scalar>();
    }
    if (cfg["rl"]["landing_terminal_z"]) {
      landing_terminal_z_ = cfg["rl"]["landing_terminal_z"].as<Scalar>();
    }
    if (cfg["rl"]["landing_success_xy_error"]) {
      landing_success_xy_error_ = cfg["rl"]["landing_success_xy_error"].as<Scalar>();
    }
    if (cfg["rl"]["landing_success_vz"]) {
      landing_success_vz_ = cfg["rl"]["landing_success_vz"].as<Scalar>();
    }
    if (cfg["rl"]["landing_success_tilt"]) {
      landing_success_tilt_ = cfg["rl"]["landing_success_tilt"].as<Scalar>();
    }
    if (cfg["rl"]["landing_success_vxy"]) {
      landing_success_vxy_ = cfg["rl"]["landing_success_vxy"].as<Scalar>();
    }
    if (cfg["rl"]["landing_success_body_rate"]) {
      landing_success_body_rate_ =
        cfg["rl"]["landing_success_body_rate"].as<Scalar>();
    }
    if (cfg["rl"]["landing_success_reward"]) {
      landing_success_reward_ = cfg["rl"]["landing_success_reward"].as<Scalar>();
    }
    if (cfg["rl"]["landing_failure_reward"]) {
      landing_failure_reward_ = cfg["rl"]["landing_failure_reward"].as<Scalar>();
    }
    if (cfg["rl"]["log_interval_steps"]) {
      log_interval_steps_ = std::max(1, cfg["rl"]["log_interval_steps"].as<int>());
    }
  } else {
    return false;
  }
  return true;
}

bool QuadrotorDotEnv::getAct(Ref<Vector<>> act) const {
  if (cmd_.t >= 0.0 && quad_act_.allFinite()) {
    act = quad_act_;
    return true;
  }
  return false;
}

bool QuadrotorDotEnv::getAct(Command *const cmd) const {
  if (!cmd_.valid()) return false;
  *cmd = cmd_;
  return true;
}

void QuadrotorDotEnv::addObjectsToUnity(std::shared_ptr<UnityBridge> bridge) {
  bridge->addQuadrotor(quadrotor_ptr_);
}

std::ostream &operator<<(std::ostream &os, const QuadrotorDotEnv &quad_env) {
  os.precision(3);
  os << "Quadrotor Dot Environment:\n"
     << "obs dim =            [" << quad_env.obs_dim_ << "]\n"
     << "act dim =            [" << quad_env.act_dim_ << "]\n"
     << "sim dt =             [" << quad_env.sim_dt_ << "]\n"
     << "max_t =              [" << quad_env.max_t_ << "]\n"
     << "tag_center_world[0]= [" << quad_env.tag_center_world_[0].transpose() << "]\n"
     << "tag_center_world[1]= [" << quad_env.tag_center_world_[1].transpose() << "]\n"
     << "tag_center_world[2]= [" << quad_env.tag_center_world_[2].transpose() << "]\n"
     << "tag_order(l->s)=    [" << quad_env.tag_order_[0] << " "
     << quad_env.tag_order_[1] << " " << quad_env.tag_order_[2] << "]\n"
     << "camera fx fy cx cy = [" << quad_env.fx_ << " " << quad_env.fy_ << " "
     << quad_env.cx_ << " " << quad_env.cy_ << "]\n"
     << "act_mean =           [" << quad_env.act_mean_.transpose() << "]\n"
     << "act_std =            [" << quad_env.act_std_.transpose() << "]\n"
     << "obs_mean =           [" << quad_env.obs_mean_.transpose() << "]\n"
     << "obs_std =            [" << quad_env.obs_std_.transpose() << std::endl;
  os.precision();
  return os;
}

}  // namespace flightlib
