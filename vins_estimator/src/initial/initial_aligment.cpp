#include <ros/ros.h>

#include "../factor/imu_factor.h"
#include "../feature_manager.h"
#include "../utility/utility.h"
#include "initial_alignment.h"

using namespace std;

void solveGyroscopeBias(std::map<double, ImageFrame> &all_image_frame, Eigen::Vector3d *Bgs) {
  Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
  Eigen::Vector3d b = Eigen::Vector3d::Zero();
  for (auto frame_i = all_image_frame.begin(); frame_i != std::prev(all_image_frame.end()); frame_i = std::next(frame_i)) {
    const auto frame_j = std::next(frame_i);
    const Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);
    const Eigen::Matrix3d tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
    const Eigen::Vector3d tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();
    A += tmp_A.transpose() * tmp_A;
    b += tmp_A.transpose() * tmp_b;
  }
  const Eigen::Vector3d delta_bg = A.ldlt().solve(b);
  ROS_WARN_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());

  for (int i = 0; i <= WINDOW_SIZE; i++) {
    Bgs[i] += delta_bg;
  }

  for (auto frame_i = std::next(all_image_frame.begin()); frame_i != all_image_frame.end(); frame_i = std::next(frame_i)) {
    frame_i->second.pre_integration->repropagate(Eigen::Vector3d::Zero(), Bgs[0]);
  }
}

Eigen::MatrixXd TangentBasis(const Eigen::Vector3d &g0) {
  const Eigen::Vector3d a = g0.normalized();
  const Eigen::Vector3d tmp = a == Eigen::Vector3d(0., 0., 1.) ? Eigen::Vector3d(1., 0., 0.) : Eigen::Vector3d(0., 0., 1.);
  const Eigen::Vector3d b = (tmp - a * (a.transpose() * tmp)).normalized();
  const Eigen::Vector3d c = a.cross(b);

  Eigen::MatrixXd bc(3, 2);
  bc.block<3, 1>(0, 0) = b;
  bc.block<3, 1>(0, 1) = c;
  return bc;
}

void RefineGravity(const std::map<double, ImageFrame> &all_image_frame, Eigen::Vector3d &g, Eigen::VectorXd &x) {
  Eigen::Vector3d g0 = g.normalized() * G.norm();
  int all_frame_count = all_image_frame.size();
  int n_state = all_frame_count * 3 + 2 + 1;

  Eigen::MatrixXd A{n_state, n_state};
  A.setZero();
  Eigen::VectorXd b{n_state};
  b.setZero();

  std::map<double, ImageFrame>::iterator frame_i;
  std::map<double, ImageFrame>::iterator frame_j;
  for (int k = 0; k < 4; k++) {
    const Eigen::MatrixXd lxly = TangentBasis(g0);
    int i = 0;
    for (auto frame_i = all_image_frame.begin(); frame_i != std::prev(all_image_frame.end()); frame_i = std::next(frame_i), ++i) {
      const auto frame_j = std::next(frame_i);
      const double dt = frame_j->second.pre_integration->sum_dt;

      Eigen::Matrix<double, 6, 9> tmp_A = Eigen::Matrix<double, 6, 9>::Zero();
      Eigen::Matrix<double, 6, 1> tmp_b = Eigen::Matrix<double, 6, 1>::Zero();

      tmp_A.block<3, 3>(0, 0) = -dt * Eigen::Matrix3d::Identity();
      tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Eigen::Matrix3d::Identity() * lxly;
      tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;
      tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] -
                                frame_i->second.R.transpose() * dt * dt / 2 * g0;

      tmp_A.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
      tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
      tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Eigen::Matrix3d::Identity() * lxly;
      tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * Eigen::Matrix3d::Identity() * g0;

      const Eigen::Matrix<double, 6, 6> cov_inv = Eigen::Matrix<double, 6, 6>::Identity();
      const Eigen::MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
      const Eigen::VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

      A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
      b.segment<6>(i * 3) += r_b.head<6>();

      A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
      b.tail<3>() += r_b.tail<3>();

      A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
      A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);
    Eigen::VectorXd dg = x.segment<2>(n_state - 3);
    g0 = (g0 + lxly * dg).normalized() * G.norm();
  }
  g = g0;
}

bool LinearAlignment(std::map<double, ImageFrame> &all_image_frame, Eigen::Vector3d &g, Eigen::VectorXd &x) {
  int all_frame_count = all_image_frame.size();
  int n_state = all_frame_count * 3 + 3 + 1;

  Eigen::MatrixXd A{n_state, n_state};
  A.setZero();
  Eigen::VectorXd b{n_state};
  b.setZero();

  std::map<double, ImageFrame>::iterator frame_i;
  std::map<double, ImageFrame>::iterator frame_j;
  int i = 0;
  for (auto frame_i = all_image_frame.begin(); frame_i != std::prev(all_image_frame.end()); frame_i = std::next(frame_i), ++i) {
    const auto frame_j = std::next(frame_i);
    const double dt = frame_j->second.pre_integration->sum_dt;

    Eigen::Matrix<double, 6, 10> tmp_A = Eigen::Matrix<double, 6, 10>::Zero();
    Eigen::Matrix<double, 6, 1> tmp_b = Eigen::Matrix<double, 6, 1>::Zero();

    tmp_A.block<3, 3>(0, 0) = -dt * Eigen::Matrix3d::Identity();
    tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Eigen::Matrix3d::Identity();
    tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;
    tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];

    tmp_A.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
    tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
    tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Eigen::Matrix3d::Identity();
    tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;

    const Eigen::Matrix<double, 6, 6> cov_inv = Eigen::Matrix<double, 6, 6>::Identity();
    const Eigen::MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
    const Eigen::VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

    A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
    b.segment<6>(i * 3) += r_b.head<6>();

    A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
    b.tail<4>() += r_b.tail<4>();

    A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
    A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
  }

  A = A * 1000.0;
  b = b * 1000.0;
  x = A.ldlt().solve(b);
  double s = x(n_state - 1) / 100.0;
  ROS_DEBUG("estimated scale: %f", s);
  g = x.segment<3>(n_state - 4);
  ROS_DEBUG_STREAM(" result g     " << g.norm() << " " << g.transpose());
  if (fabs(g.norm() - G.norm()) > 1.0 || s < 0) {
    return false;
  }

  RefineGravity(all_image_frame, g, x);
  s = (x.tail<1>())(0) / 100.0;
  (x.tail<1>())(0) = s;
  ROS_DEBUG_STREAM(" refine     " << g.norm() << " " << g.transpose());
  return s >= 0.0;
}

bool VisualIMUAlignment(std::map<double, ImageFrame> &all_image_frame, Eigen::Vector3d *Bgs, Eigen::Vector3d &g, Eigen::VectorXd &x) {
  solveGyroscopeBias(all_image_frame, Bgs);
  return LinearAlignment(all_image_frame, g, x);
}
