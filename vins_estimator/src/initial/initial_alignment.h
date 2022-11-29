#pragma once

#include <eigen3/Eigen/Eigen>

class ImageFrame {
 public:
  ImageFrame() {}
  ImageFrame(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>& _points, double _t) : t{_t}, points{_points} {}

  bool is_key_frame = false;
  double t;
  std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> points;

  Eigen::Matrix3d R;
  Eigen::Vector3d T;
  IntegrationBase* pre_integration;
};

bool VisualIMUAlignment(std::map<double, ImageFrame>& all_image_frame, Eigen::Vector3d* Bgs, Eigen::Vector3d& g, Eigen::VectorXd& x);