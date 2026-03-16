#include "flightlib/envs/quadrotor_env/quadrotor_pos_env.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <utility>

namespace flightlib {
namespace {

bool parseTagCenters(const YAML::Node &centers_node,
                     std::array<Vector<3>, quadposenv::kNumTags> *centers) {
  if (!centers_node || !centers_node.IsSequence() ||
      static_cast<int>(centers_node.size()) != quadposenv::kNumTags) {
    return false;
  }
  for (int i = 0; i < quadposenv::kNumTags; i++) {
    const YAML::Node center = centers_node[i];
    if (!center.IsSequence() || center.size() != 3) return false;
    (*centers)[i] << center[0].as<Scalar>(), center[1].as<Scalar>(),
      center[2].as<Scalar>();
  }
  return true;
}

bool parseTagCorners(const YAML::Node &corners_node,
                     std::array<Matrix<3, 4>, quadposenv::kNumTags> *corners) {
  if (!corners_node || !corners_node.IsSequence() ||
      static_cast<int>(corners_node.size()) != quadposenv::kNumTags) {
    return false;
  }
  for (int tag_idx = 0; tag_idx < quadposenv::kNumTags; tag_idx++) {
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

Scalar polygonArea(const std::array<Vector<2>, 4> &pts) {
  Scalar twice_area = 0.0;
  for (int i = 0; i < 4; i++) {
    const Vector<2> &a = pts[i];
    const Vector<2> &b = pts[(i + 1) % 4];
    twice_area += a.x() * b.y() - b.x() * a.y();
  }
  return std::abs(twice_area) * Scalar(0.5);
}

Scalar axisAlignmentPenalty(const std::array<Vector<2>, 4> &pts) {
  constexpr Scalar eps = static_cast<Scalar>(1e-6);
  const Vector<2> v01 = pts[1] - pts[0];
  const Vector<2> v12 = pts[2] - pts[1];
  const Vector<2> v23 = pts[3] - pts[2];
  const Vector<2> v30 = pts[0] - pts[3];
  const Scalar l01 = v01.norm();
  const Scalar l12 = v12.norm();
  const Scalar l23 = v23.norm();
  const Scalar l30 = v30.norm();
  const Vector<2> u01 = v01 / (l01 + eps);
  const Vector<2> u12 = v12 / (l12 + eps);
  const Vector<2> u23 = v23 / (l23 + eps);
  const Vector<2> u30 = v30 / (l30 + eps);
  const Scalar a01 = std::abs(u01.y());
  const Scalar a12 = std::abs(u12.x());
  const Scalar a23 = std::abs(u23.y());
  const Scalar a30 = std::abs(u30.x());
  return (a01 + a12 + a23 + a30) * Scalar(0.25);
}

bool tagObsCornersFromQuadObs(const Vector<quadposenv::kNObs> &quad_obs,
                              int tag_idx,
                              std::array<Vector<2>, 4> *corners) {
  const int obs_base = quadposenv::kObs + tag_idx * quadposenv::kTagFeat;
  if (quad_obs(obs_base + quadposenv::kTagId) < 0.0) {
    return false;
  }
  (*corners)[0] << quad_obs(obs_base + quadposenv::kCorner0X),
    quad_obs(obs_base + quadposenv::kCorner0Y);
  (*corners)[1] << quad_obs(obs_base + quadposenv::kCorner1X),
    quad_obs(obs_base + quadposenv::kCorner1Y);
  (*corners)[2] << quad_obs(obs_base + quadposenv::kCorner2X),
    quad_obs(obs_base + quadposenv::kCorner2Y);
  (*corners)[3] << quad_obs(obs_base + quadposenv::kCorner3X),
    quad_obs(obs_base + quadposenv::kCorner3Y);
  return true;
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

QuadrotorPosEnv::QuadrotorPosEnv()
  : QuadrotorPosEnv(getenv("FLIGHTMARE_PATH") +
                    std::string("/flightlib/configs/quadrotor_pos_env.yaml")) {}

QuadrotorPosEnv::QuadrotorPosEnv(const std::string &cfg_path)
  : EnvBase()  {
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
  obs_dim_ = quadposenv::kNObs;
  act_dim_ = quadposenv::kNAct;

  // Select control interpretation from YAML:
  // - motor: [m0, m1, m2, m3] rotor thrust commands
  // - ctbr: [collective_thrust, body_rate_x, body_rate_y, body_rate_z]
  std::string control_mode = "ctbr";
  if (cfg_["quadrotor_env"] && cfg_["quadrotor_env"]["control_mode"]) {
    control_mode = cfg_["quadrotor_env"]["control_mode"].as<std::string>();
  }
  use_ctbr_ = (control_mode == "ctbr");

  if (use_ctbr_) {
    const Scalar hover_acc = -Gz; //types.hpp 에서 -9.81 로 정의되어 있음 
    Vector<3> omega_max = Vector<3>::Constant(6.0);
    if (cfg_["quadrotor_dynamics"] && cfg_["quadrotor_dynamics"]["omega_max"]) {
      const std::vector<Scalar> omega_max_cfg =
        cfg_["quadrotor_dynamics"]["omega_max"].as<std::vector<Scalar>>();
      if (omega_max_cfg.size() == 3) {
        omega_max = Map<const Vector<3>>(omega_max_cfg.data());
      }
    }
    //CTBR 일때 
    act_mean_ << hover_acc, 0.0, 0.0, 0.0;
    act_std_ << hover_acc, omega_max.x(), omega_max.y(), omega_max.z();
  } else {
    Scalar mass = quadrotor_ptr_->getMass();
    act_mean_ = Vector<quadposenv::kNAct>::Ones() * (-mass * Gz) / 4;
    act_std_ = Vector<quadposenv::kNAct>::Ones() * (-mass * 2 * Gz) / 4;
  }

  // reasonable normalization defaults for [tag_uv, tag_rgb]
  obs_mean_.setZero();
  obs_std_.setOnes();

  // load parameters
  loadParam(cfg_);
}

QuadrotorPosEnv::~QuadrotorPosEnv() {}

bool QuadrotorPosEnv::reset(Ref<Vector<>> obs, const bool random) {
  quad_state_.setZero();
  quad_obs_.setZero();
  quad_obs_.segment<quadposenv::kTagObs>(quadposenv::kObs).setConstant(-1.0);
  quad_act_.setZero();
  prev_corner_uv_.setZero();
  prev_corner_uv_valid_ = false;
  curr_tag_visible_.fill(false);
  estimated_p_C_.setZero();
  estimated_p_C_valid_ = false;
  last_observed_area_ = -1.0;
  last_tag_visible_ = false;
  last_corners_visible_ = false;
  prev_center_error_ = -1.0;
  prev_xy_error_ = -1.0;
  stage_ = 0;
  area_reward_mode_active_ = false;
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

bool QuadrotorPosEnv::worldPointToCamera(const Ref<const Vector<3>> p_W,
                                         Ref<Vector<3>> p_C) const {
  const Matrix<3, 3> R_WB = quad_state_.q().toRotationMatrix();
  const Vector<3> p_WB = quad_state_.p;
  const Matrix<3, 3> R_WC = R_WB * R_BC_;
  const Vector<3> p_WC = p_WB + R_WB * B_r_BC_;
  p_C = R_WC.transpose() * (p_W - p_WC);
  return p_C.allFinite();
}

bool QuadrotorPosEnv::projectWorldPointToImage(const Ref<const Vector<3>> p_W,
                                               Ref<Vector<2>> pixel_uv,
                                               bool *in_front,
                                               bool *in_image) const {
  Vector<3> p_C;
  if (!worldPointToCamera(p_W, p_C)) {
    return false;
  }

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

bool QuadrotorPosEnv::getObs(Ref<Vector<>> obs) {
  quadrotor_ptr_->getState(&quad_state_);

  if (!hold_last_tag_obs_) {
    quad_obs_.segment<quadposenv::kTagObs>(quadposenv::kObs).setConstant(-1.0);
  }
  curr_tag_visible_.fill(false);

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
      bool tag_in_front = false;
      bool tag_in_image = false;
      const bool tag_projected = projectWorldPointToImage(
        tag_center_world_[0], uv_dbg, &tag_in_front, &tag_in_image);
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
        "tag uv | projected=%d in_front=%d in_image=%d uv=[%.2f %.2f]",
        static_cast<int>(tag_projected), static_cast<int>(tag_in_front),
        static_cast<int>(tag_in_image), uv_dbg.x(), uv_dbg.y());
      logger_.info(
        "tag p_C | [%.3f %.3f %.3f]",
        p_C_dbg.x(), p_C_dbg.y(), p_C_dbg.z());
    }
  }

  const Scalar sx = static_cast<Scalar>(quadposenv::kImgWidth - 1) /
                    std::max(Scalar(1.0), static_cast<Scalar>(cam_width_ - 1));
  const Scalar sy = static_cast<Scalar>(quadposenv::kImgHeight - 1) /
                    std::max(Scalar(1.0), static_cast<Scalar>(cam_height_ - 1));
  auto set_obs_from_center_uv = [&](const Vector<2> &uv_centered, int obs_x_idx) {
    const Scalar px = uv_centered.x() + cx_;
    const Scalar py = uv_centered.y() + cy_;
    quad_obs_(obs_x_idx) = px * sx;
    quad_obs_(obs_x_idx + 1) = py * sy;
  };

  last_tag_visible_ = false;
  last_corners_visible_ = false;

  for (int tag_idx = 0; tag_idx < quadposenv::kNumTags; tag_idx++) {
    const int obs_base = quadposenv::kObs + tag_idx * quadposenv::kTagFeat;
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
    const bool tag_obs_available =
      tag_visible ||
      (use_projected_uv_out_of_view_ && center_projected && all_corners_projected);
    curr_tag_visible_[tag_idx] = tag_visible;
    if (tag_obs_available) {
      set_obs_from_center_uv(center_uv, obs_base + quadposenv::kCenterX);
      set_obs_from_center_uv(corner_uv[0], obs_base + quadposenv::kCorner0X);
      set_obs_from_center_uv(corner_uv[1], obs_base + quadposenv::kCorner1X);
      set_obs_from_center_uv(corner_uv[2], obs_base + quadposenv::kCorner2X);
      set_obs_from_center_uv(corner_uv[3], obs_base + quadposenv::kCorner3X);
      quad_obs_(obs_base + quadposenv::kTagId) = static_cast<Scalar>(tag_idx);

      if (tag_idx == 0) {
        Vector<3> measured_p_C;
        if (worldPointToCamera(tag_center_world_[0], measured_p_C)) {
          estimated_p_C_ = measured_p_C;
          estimated_p_C_valid_ = true;
        }
        last_tag_visible_ = true;
        last_corners_visible_ = true;
      }
    }
  }

  if (!curr_tag_visible_[0] && estimated_p_C_valid_) {
    const Vector<3> omega_cmd = quad_act_.segment<3>(1);
    estimated_p_C_ -= sim_dt_ * omega_cmd.cross(estimated_p_C_);
  }

  std::array<Vector<2>, 4> tag0_corners;
  if (tagObsCornersFromQuadObs(quad_obs_, 0, &tag0_corners)) {
    last_metric_area_ = polygonArea(tag0_corners);
    last_metric_shape2_ = -axisAlignmentPenalty(tag0_corners);
    last_observed_area_ = last_metric_area_;
  } else {
    last_metric_area_ = 0.0;
    last_metric_shape2_ = 0.0;
    last_observed_area_ = -1.0;
  }

  cv::Mat rgb_image;
  if (rgb_camera_ != nullptr && rgb_camera_->getRGBImage(rgb_image) &&
      !rgb_image.empty()) {
    cv::Mat resized = rgb_image;
    if (rgb_image.cols != quadposenv::kImgWidth ||
        rgb_image.rows != quadposenv::kImgHeight) {
      cv::resize(rgb_image, resized,
                 cv::Size(quadposenv::kImgWidth, quadposenv::kImgHeight), 0.0, 0.0,
                 cv::INTER_AREA);
    }
    if (!resized.isContinuous()) resized = resized.clone();

    int flat_idx = quadposenv::kImg;
    for (int r = 0; r < resized.rows; r++) {
      const cv::Vec3b *row = resized.ptr<cv::Vec3b>(r);
      for (int c = 0; c < resized.cols; c++) {
        quad_obs_(flat_idx++) = static_cast<Scalar>(row[c][0]);
        quad_obs_(flat_idx++) = static_cast<Scalar>(row[c][1]);
        quad_obs_(flat_idx++) = static_cast<Scalar>(row[c][2]);
      }
    }
  }

  obs.segment<quadposenv::kNObs>(quadposenv::kObs) = quad_obs_;
  return true;
}

void QuadrotorPosEnv::updateExtraInfo() {
  quadrotor_ptr_->getState(&quad_state_);
  extra_info_["drone_pos_x"] = quad_state_.x(QS::POSX);
  extra_info_["drone_pos_y"] = quad_state_.x(QS::POSY);
  extra_info_["drone_pos_z"] = quad_state_.x(QS::POSZ);
  extra_info_["drone_vel_x"] = quad_state_.x(QS::VELX);
  extra_info_["drone_vel_y"] = quad_state_.x(QS::VELY);
  extra_info_["drone_vel_z"] = quad_state_.x(QS::VELZ);

  extra_info_["reward_total"] = last_total_reward_;
  extra_info_["reward_xy"] = last_r_xy_;
  extra_info_["reward_center"] = last_r_center_;
  extra_info_["reward_z"] = last_r_z_;
  extra_info_["reward_survival"] = last_r_survival_;
  extra_info_["reward_action_hover"] = last_r_vis_;
  extra_info_["metric_area"] = last_metric_area_;
  extra_info_["metric_shape2"] = last_metric_shape2_;
  extra_info_["reward_area"] = last_r_area_;
  extra_info_["reward_shape"] = last_r_shape_;
  extra_info_["reward_shape2"] = last_r_shape2_;
  extra_info_["reward_center_progress"] = last_r_center_progress_;
  extra_info_["reward_missing_tag"] = last_r_missing_tag_;
  extra_info_["estimated_p_c_x"] = estimated_p_C_valid_ ? estimated_p_C_.x() : 0.0f;
  extra_info_["estimated_p_c_y"] = estimated_p_C_valid_ ? estimated_p_C_.y() : 0.0f;
  extra_info_["estimated_p_c_z"] = estimated_p_C_valid_ ? estimated_p_C_.z() : 0.0f;
  Vector<3> real_p_C;
  const bool real_p_C_valid = worldPointToCamera(tag_center_world_[0], real_p_C);
  extra_info_["real_p_c_x"] = real_p_C_valid ? real_p_C.x() : 0.0f;
  extra_info_["real_p_c_y"] = real_p_C_valid ? real_p_C.y() : 0.0f;
  extra_info_["real_p_c_z"] = real_p_C_valid ? real_p_C.z() : 0.0f;
}

Scalar QuadrotorPosEnv::step(const Ref<Vector<>> act, Ref<Vector<>> obs) {
  quad_act_ = act.cwiseProduct(act_std_) + act_mean_;
  cmd_.t += sim_dt_; //sim_dt_ : 한  step 마다 시간이 sim_dt 만큼 증가 
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

  const int target_tag_idx = 0;
  const Scalar xy_error =
    (quad_state_.p.head<2>() - tag_center_world_[target_tag_idx].head<2>()).norm();
  const Scalar prev_xy_error_for_log = prev_xy_error_;
  Scalar xy_error_delta = 0.0;
  const Scalar z_error =
    quad_state_.x(QS::POSZ) - (tag_center_world_[target_tag_idx].z()+37.0);
  const Matrix<3, 3> R_WB = quad_state_.q().toRotationMatrix();
  const Scalar cos_tilt =
    std::max(Scalar(-1.0), std::min(Scalar(1.0), R_WB(2, 2)));
  const Scalar tilt = std::acos(cos_tilt);
  Scalar reward_xy = 0.0;
  if (prev_xy_error_ >= Scalar(0.0)) {
    xy_error_delta = prev_xy_error_ - xy_error;
    reward_xy = centering_w_xy_ * xy_error_delta;
  }
  prev_xy_error_ = xy_error;
  const Scalar reward_z = -centering_w_z_ * (z_error * z_error);
  const Scalar reward_tilt = -centering_w_tilt_ * (tilt * tilt);
  const Scalar reward_survival = centering_survival_reward_;
  const Scalar action_hover_error =
    (quad_act_(0) - Scalar(0)) * (quad_act_(0) - Scalar(0)) +
    quad_act_(1) * quad_act_(1) +
    quad_act_(2) * quad_act_(2) +
    quad_act_(3) * quad_act_(3);
  const Scalar reward_action_hover = -centering_w_action_hover_ * action_hover_error;
  const Scalar half_w = static_cast<Scalar>(quadposenv::kImgWidth - 1) * Scalar(0.5);
  const Scalar half_h = static_cast<Scalar>(quadposenv::kImgHeight - 1) * Scalar(0.5);
  const Scalar eps = static_cast<Scalar>(1e-6);

  Scalar reward_center = 0.0;
  Scalar reward_center_progress = 0.0;
  Scalar reward_area = 0.0;
  Scalar reward_shape = 0.0;
  Scalar reward_missing_tag = 0.0;
  Scalar metric_area = 0.0;
  Scalar metric_shape = 0.0;
  Scalar center_error = -1.0;
  Scalar prev_center_error_for_log = prev_center_error_;
  Scalar center_error_delta = 0.0;

  std::array<Vector<2>, 4> tag0_corners;
  if (tagObsCornersFromQuadObs(quad_obs_, 0, &tag0_corners)) {
    const Scalar center_x = quad_obs_(quadposenv::kCenterX);
    const Scalar center_y = quad_obs_(quadposenv::kCenterY);
    const Scalar ex = (center_x - half_w) / std::max(half_w, eps);
    const Scalar ey = (center_y - half_h) / std::max(half_h, eps);
    center_error = std::sqrt(ex * ex + ey * ey);
    const Scalar center_score = Scalar(1.0) / (Scalar(1.0) + center_error);
    if (prev_center_error_ >= Scalar(0.0)) {
      center_error_delta = prev_center_error_ - center_error;
      reward_center_progress =
        tag_center_progress_coeff_ * center_error_delta;
    }
    prev_center_error_ = center_error;

    metric_area = polygonArea(tag0_corners);
    const Scalar target_area = std::max(tag_target_area_, Scalar(1.0));
    const Scalar area_error =
      std::abs(metric_area - target_area) / std::max(target_area, eps);
    const Scalar area_score = std::exp(-tag_area_error_scale_ * area_error);

    const Scalar shape_penalty = axisAlignmentPenalty(tag0_corners);
    const Scalar shape_score =
      std::exp(-tag_shape_error_scale_ * shape_penalty);

    reward_center = tag_center_coeff_ * center_score;
    reward_area = tag_area_coeff_ * area_score;
    reward_shape = tag_shape_coeff_ * shape_score;
    metric_shape = shape_score;
  } else {
    prev_center_error_ = -1.0;
    reward_missing_tag = -tag_missing_penalty_;
  }

  const Scalar total_reward =
    reward_xy + reward_z + reward_tilt + reward_survival + reward_action_hover +
    reward_center + reward_center_progress + reward_area + reward_shape +
    reward_missing_tag;

  last_total_reward_ = total_reward;
  last_r_xy_ = reward_xy;
  last_r_z_ = reward_z;
  last_r_vis_ = reward_action_hover;
  last_r_center_ = reward_center;
  last_r_survival_ = reward_survival;
  last_metric_area_ = metric_area;
  last_metric_shape2_ = metric_shape;
  last_r_area_ = reward_area;
  last_r_shape_ = reward_shape;
  last_r_shape2_ = tag_shape2_coeff_ * metric_shape;
  last_r_center_progress_ = reward_center_progress;
  last_r_missing_tag_ = reward_missing_tag;
  last_r_area_small_ = 0.0;
  last_r_smooth_ = 0.0;
  last_r_invisible_ = 0.0;
  last_r_switch_ = 0.0;

  const Vector<3> euler_zyx_log = R_WB.eulerAngles(2, 1, 0);
  const Scalar yaw_log = euler_zyx_log(0);

  log_counter_++;
  if (log_counter_ % std::max(1, log_interval_steps_) == 0) {
    logger_.info(
      "quad pos | x=%.3f y=%.3f z=%.3f yaw=%.4f tilt=%.4f",
      quad_state_.x(QS::POSX), quad_state_.x(QS::POSY), quad_state_.x(QS::POSZ),
      yaw_log, tilt);
    logger_.info(
      "reward | tot=%.2f xy=%.2f pxy=%.3f dxy=%.3f z=%.2f c=%.2f cp=%.2f cerr=%.3f pc=%.3f d=%.3f a=%.2f s=%.2f miss=%.2f tilt=%.2f act=%.2f area=%.1f shape=%.2f",
      total_reward, reward_xy, prev_xy_error_for_log, xy_error_delta, reward_z, reward_center, reward_center_progress,
      center_error, prev_center_error_for_log, center_error_delta,
      reward_area, reward_shape, reward_missing_tag,
      reward_tilt, reward_action_hover, metric_area, metric_shape);
    logger_.info(
      "uv | c=(%.1f,%.1f) p0=(%.1f,%.1f) p1=(%.1f,%.1f) p2=(%.1f,%.1f) p3=(%.1f,%.1f)",
      quad_obs_(quadposenv::kCenterX), quad_obs_(quadposenv::kCenterY),
      quad_obs_(quadposenv::kCorner0X), quad_obs_(quadposenv::kCorner0Y),
      quad_obs_(quadposenv::kCorner1X), quad_obs_(quadposenv::kCorner1Y),
      quad_obs_(quadposenv::kCorner2X), quad_obs_(quadposenv::kCorner2Y),
      quad_obs_(quadposenv::kCorner3X), quad_obs_(quadposenv::kCorner3Y));
  }

  return total_reward;
}

bool QuadrotorPosEnv::isTerminalState(Scalar &reward) {
  const bool hit_world_box =
    (quad_state_.x(QS::POSX) <= world_box_(0, 0)+0.001) ||
    (quad_state_.x(QS::POSX) >= world_box_(0, 1)-0.001) ||
    (quad_state_.x(QS::POSY) <= world_box_(1, 0)+0.001) ||
    (quad_state_.x(QS::POSY) >= world_box_(1, 1)-0.001) ||
    (quad_state_.x(QS::POSZ) <= world_box_(2, 0)+0.001) ||
    (quad_state_.x(QS::POSZ) >= world_box_(2, 1)-0.001);
  if (hit_world_box) {
    if (log_counter_ % std::max(1, log_interval_steps_) == 0){
      logger_.warn(
        "terminate reason=world_box "
        "pos=(" + std::to_string(quad_state_.x(QS::POSX)) + ", " +
        std::to_string(quad_state_.x(QS::POSY)) + ", " +
        std::to_string(quad_state_.x(QS::POSZ)) + ") "
        "box=[(" + std::to_string(world_box_(0, 0)) + ", " +
        std::to_string(world_box_(0, 1)) + "), (" +
        std::to_string(world_box_(1, 0)) + ", " +
        std::to_string(world_box_(1, 1)) + "), (" +
        std::to_string(world_box_(2, 0)) + ", " +
        std::to_string(world_box_(2, 1)) + ")]");}
    reward = landing_failure_reward_;
    return true;
  }

  // Early terminate on excessive tilt computed from quaternion / rotation matrix.
  {
    const Matrix<3, 3> R_WB = quad_state_.q().toRotationMatrix();
    const Vector<3> euler_zyx = R_WB.eulerAngles(2, 1, 0);
    const Scalar cos_tilt =
      std::max(Scalar(-1.0), std::min(Scalar(1.0), R_WB(2, 2)));
    const Scalar tilt = std::acos(cos_tilt);
    if (tilt > landing_tilt_hard_) {
      if (log_counter_ % std::max(1, log_interval_steps_) == 0){
        logger_.warn(
          "terminate reason=attitude_limit "
          "yaw=" + std::to_string(euler_zyx(0)) +
          " tilt=" + std::to_string(tilt) +
          " threshold=" + std::to_string(landing_tilt_hard_));}
      reward = landing_tilt_hard_penalty_;
      return true;
    }
  }

  // Ground plane is assumed around z=3.0.
  if (quad_state_.x(QS::POSZ) <= landing_terminal_z_) {
    const int terminal_tag_idx = 0;
    const Scalar xy_error =
      (quad_state_.p.head<2>() - tag_center_world_[terminal_tag_idx].head<2>()).norm();
    const Scalar vxy = quad_state_.v.head<2>().norm();
    const Scalar vz = std::abs(quad_state_.x(QS::VELZ));
    const Scalar body_rate = quad_state_.w.norm();
    const Matrix<3, 3> R_WB = quad_state_.q().toRotationMatrix();
    const Vector<3> euler_zyx = R_WB.eulerAngles(2, 1, 0);
    const Scalar cos_tilt =
      std::max(Scalar(-1.0), std::min(Scalar(1.0), R_WB(2, 2)));
    const Scalar tilt = std::acos(cos_tilt);
    if (log_counter_ % std::max(1, log_interval_steps_) == 0){
      logger_.warn(
        std::string("terminate reason=landing_terminal ") +
        "z=" + std::to_string(quad_state_.x(QS::POSZ)) +
        " terminal_z=" + std::to_string(landing_terminal_z_) +
        " xy_error=" + std::to_string(xy_error) +
        " vxy=" + std::to_string(vxy) +
        " vz=" + std::to_string(vz) +
        " yaw=" + std::to_string(euler_zyx(0)) +
        " tilt=" + std::to_string(tilt) +
        " body_rate=" + std::to_string(body_rate));}
    reward =  landing_failure_reward_;
    return true;
  }
  reward = 0.0;
  return false;
}

bool QuadrotorPosEnv::loadParam(const YAML::Node &cfg) {
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
      if (cam_cfg["tag_world_pos"] && cam_cfg["tag_world_pos"].IsSequence() &&
          cam_cfg["tag_world_pos"].size() == 3) {
        tag_center_world_[0] << cam_cfg["tag_world_pos"][0].as<Scalar>(),
          cam_cfg["tag_world_pos"][1].as<Scalar>(),
          cam_cfg["tag_world_pos"][2].as<Scalar>();
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
    std::array<std::pair<Scalar, int>, quadposenv::kNumTags> scales;
    for (int i = 0; i < quadposenv::kNumTags; i++) {
      scales[i] = {estimateTagScale(tag_corner_world_[i]), i};
    }
    std::sort(scales.begin(), scales.end(),
              [](const std::pair<Scalar, int> &a, const std::pair<Scalar, int> &b) {
                return a.first > b.first;
              });
    for (int i = 0; i < quadposenv::kNumTags; i++) {
      tag_order_[i] = scales[i].second;
    }
  } else {
    return false;
  }

  if (cfg["rl"]) {
    // load reinforcement learning related parameters
    if (cfg["rl"]["centering_w_xy"]) {
      centering_w_xy_ = cfg["rl"]["centering_w_xy"].as<Scalar>();
    } else if (cfg["rl"]["landing_w_xy"]) {
      centering_w_xy_ = cfg["rl"]["landing_w_xy"].as<Scalar>();
    }
    if (cfg["rl"]["centering_xy_reward_scale"]) {
      centering_xy_reward_scale_ = cfg["rl"]["centering_xy_reward_scale"].as<Scalar>();
    } else if (cfg["rl"]["landing_xy_reward_scale"]) {
      centering_xy_reward_scale_ = cfg["rl"]["landing_xy_reward_scale"].as<Scalar>();
    }
    if (cfg["rl"]["centering_w_z"]) {
      centering_w_z_ = cfg["rl"]["centering_w_z"].as<Scalar>();
    } else if (cfg["rl"]["landing_w_z"]) {
      centering_w_z_ = cfg["rl"]["landing_w_z"].as<Scalar>();
    }
    if (cfg["rl"]["centering_w_tilt"]) {
      centering_w_tilt_ = cfg["rl"]["centering_w_tilt"].as<Scalar>();
    } else if (cfg["rl"]["landing_w_tilt"]) {
      centering_w_tilt_ = cfg["rl"]["landing_w_tilt"].as<Scalar>();
    }
    if (cfg["rl"]["centering_survival_reward"]) {
      centering_survival_reward_ = cfg["rl"]["centering_survival_reward"].as<Scalar>();
    } else if (cfg["rl"]["landing_survival_reward"]) {
      centering_survival_reward_ = cfg["rl"]["landing_survival_reward"].as<Scalar>();
    }
    if (cfg["rl"]["centering_w_action_hover"]) {
      centering_w_action_hover_ = cfg["rl"]["centering_w_action_hover"].as<Scalar>();
    } else if (cfg["rl"]["landing_w_action_hover"]) {
      centering_w_action_hover_ = cfg["rl"]["landing_w_action_hover"].as<Scalar>();
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
    } else if (cfg["rl"]["tag_shape_coeff"]) {
      tag_shape2_coeff_ = cfg["rl"]["tag_shape_coeff"].as<Scalar>();
    }
    if (cfg["rl"]["tag_target_area"]) {
      tag_target_area_ = cfg["rl"]["tag_target_area"].as<Scalar>();
    }
    if (cfg["rl"]["tag_area_error_scale"]) {
      tag_area_error_scale_ = cfg["rl"]["tag_area_error_scale"].as<Scalar>();
    }
    if (cfg["rl"]["tag_shape_error_scale"]) {
      tag_shape_error_scale_ = cfg["rl"]["tag_shape_error_scale"].as<Scalar>();
    }
    if (cfg["rl"]["tag_center_progress_coeff"]) {
      tag_center_progress_coeff_ =
        cfg["rl"]["tag_center_progress_coeff"].as<Scalar>();
    }
    if (cfg["rl"]["tag_missing_penalty"]) {
      tag_missing_penalty_ = cfg["rl"]["tag_missing_penalty"].as<Scalar>();
    }

    if (cfg["rl"]["hold_last_tag_obs"]) {
      hold_last_tag_obs_ = cfg["rl"]["hold_last_tag_obs"].as<bool>();
    }
    if (cfg["rl"]["use_projected_uv_out_of_view"]) {
      use_projected_uv_out_of_view_ =
        cfg["rl"]["use_projected_uv_out_of_view"].as<bool>();
    }
    if (cfg["rl"]["landing_tilt_hard_penalty"]) {
      landing_tilt_hard_penalty_ = cfg["rl"]["landing_tilt_hard_penalty"].as<Scalar>();
    }
    if (cfg["rl"]["landing_tilt_hard"]) {
      landing_tilt_hard_ = cfg["rl"]["landing_tilt_hard"].as<Scalar>();
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

bool QuadrotorPosEnv::getAct(Ref<Vector<>> act) const {
  if (cmd_.t >= 0.0 && quad_act_.allFinite()) {
    act = quad_act_;
    return true;
  }
  return false;
}

bool QuadrotorPosEnv::getAct(Command *const cmd) const {
  if (!cmd_.valid()) return false;
  *cmd = cmd_;
  return true;
}

void QuadrotorPosEnv::addObjectsToUnity(std::shared_ptr<UnityBridge> bridge) {
  bridge->addQuadrotor(quadrotor_ptr_);
}

std::ostream &operator<<(std::ostream &os, const QuadrotorPosEnv &quad_env) {
  os.precision(3);
  os << "Quadrotor Pos Environment:\n"
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
