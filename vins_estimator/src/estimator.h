#pragma once

#include <std_msgs/Header.h>

#include <eigen3/Eigen/Eigen>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <unordered_map>

#include "factor/imu_factor.h"
#include "factor/marginalization_factor.h"
#include "factor/pose_local_parameterization.h"
#include "factor/projection_factor.h"
#include "factor/projection_td_factor.h"
#include "feature_manager.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"
#include "initial/initial_sfm.h"
#include "initial/solve_5pts.h"
#include "parameters.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"

namespace vins {

class Estimator {
 public:
  Estimator();

  void setParameter();

  // interface
  void processIMU(double t, const Eigen::Vector3d &linear_acceleration, const Eigen::Vector3d &angular_velocity);
  void processImage(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header);
  void setReloFrame(double _frame_stamp, int _frame_index, std::vector<Eigen::Vector3d> &_match_points, Eigen::Vector3d _relo_t,
                    Eigen::Matrix3d _relo_r);
  // internal
  void clearState();

 private:
  bool initialStructure();
  bool visualInitialAlign();
  bool relativePose(Eigen::Matrix3d &relative_R, Eigen::Vector3d &relative_T, int &l);
  void slideWindow();
  void solveOdometry();
  void slideWindowNew();
  void slideWindowOld();
  void optimization();
  void vector2double();
  void double2vector();
  bool failureDetection();

 public:
  enum SolverFlag { INITIAL, NON_LINEAR };
  enum MarginalizationFlag { MARGIN_OLD = 0, MARGIN_SECOND_NEW = 1 };

  SolverFlag solver_flag;
  MarginalizationFlag marginalization_flag;

  Eigen::Vector3d g;                       // gravity vector
  Eigen::Matrix3d ric[NUM_OF_CAM];         // rotation from other camera to camera
  Eigen::Vector3d tic[NUM_OF_CAM];         // translation from other camera to camera
  Eigen::Vector3d Ps[(WINDOW_SIZE + 1)];   // position/translation from camera to global
  Eigen::Vector3d Vs[(WINDOW_SIZE + 1)];   // velocity
  Eigen::Matrix3d Rs[(WINDOW_SIZE + 1)];   // rotation from camera to global
  Eigen::Vector3d Bas[(WINDOW_SIZE + 1)];  // bias of acceleration
  Eigen::Vector3d Bgs[(WINDOW_SIZE + 1)];  // bias of gyroscope
  double td;

  Eigen::Vector3d acc_0, gyr_0;

 public:  // only because of visualization
  std_msgs::Header Headers[(WINDOW_SIZE + 1)];

  FeatureManager f_manager;
  std::vector<Eigen::Vector3d> key_poses;  // only for visualization

  // relocalization
  double relo_frame_stamp;
  double relo_frame_index;
  Eigen::Matrix3d drift_correct_r;
  Eigen::Vector3d drift_correct_t;
  Eigen::Vector3d relo_relative_t;
  Eigen::Quaterniond relo_relative_q;
  double relo_relative_yaw;

 private:
  bool first_imu;
  bool failure_occur;
  double initial_timestamp;

  Eigen::Matrix3d back_R0, last_R, last_R0;
  Eigen::Vector3d back_P0, last_P, last_P0;

  // important
  std::map<double, ImageFrame> all_image_frame;
  IntegrationBase *tmp_pre_integration;
  IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];

  // mainly for storing data
  std::vector<double> dt_buf[(WINDOW_SIZE + 1)];
  std::vector<Eigen::Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
  std::vector<Eigen::Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];
  int frame_count;

  MotionEstimator m_estimator;            // for solving relative pose
  InitialEXRotation initial_ex_rotation;  // initialize ric

  // for ceres-solver
  double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];            // related to Ps, Rs
  double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];  // related to Vs, Bas, Bgs
  double para_Feature[NUM_OF_F][SIZE_FEATURE];             //
  double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];              // related to ric, tic
  double para_Td[1][1];                                    // related to td
  MarginalizationInfo *last_marginalization_info;
  std::vector<double *> last_marginalization_parameter_blocks;

  // relocalization
  bool relocalization_info;
  int relo_frame_local_index;
  std::vector<Eigen::Vector3d> match_points;
  double relo_Pose[SIZE_POSE];

  Eigen::Vector3d prev_relo_t;
  Eigen::Matrix3d prev_relo_r;
};

}  // namespace vins
