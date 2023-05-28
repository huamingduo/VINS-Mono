#pragma once

#include <algorithm>
#include <eigen3/Eigen/Eigen>
#include <list>
#include <numeric>
#include <vector>

#include "parameters.h"

namespace vins {

class FeaturePerFrame {
 public:
  FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td)
      : cur_td(td), point(_point(0), _point(1), _point(2)), uv(_point(3), _point(4)), velocity(_point(5), _point(6)) {}

  double cur_td;
  Eigen::Vector3d point;
  Eigen::Vector2d uv;
  Eigen::Vector2d velocity;

  // check utility
  double z;
  bool is_used;
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
};

class FeaturePerId {
 public:
  FeaturePerId(int _feature_id, int _start_frame) : feature_id(_feature_id), start_frame(_start_frame) {}
  inline int endFrame() { return start_frame + feature_per_frame.size() - 1; }

  const int feature_id;
  int start_frame;
  std::vector<FeaturePerFrame> feature_per_frame;

  int used_num = 0;
  int solve_flag = 0;  // 0 haven't solve yet; 1 solve succ; 2 solve fail;
  double estimated_depth = -1.;

  // check utility
  bool is_outlier;
};

class FeatureManager {
 public:
  FeatureManager(Eigen::Matrix3d _Rs[]);

  void setRic(Eigen::Matrix3d _ric[]);
  void clearState() { feature.clear(); }
  int getFeatureCount();
  bool addFeatureCheckParallax(int frame_count, const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

  void setDepth(const Eigen::VectorXd &x);
  void clearDepth(const Eigen::VectorXd &x);
  Eigen::VectorXd getDepthVector();
  void triangulate(Eigen::Vector3d Ps[], Eigen::Vector3d tic[], Eigen::Matrix3d ric[]);

  void removeFailures();
  void RemoveEarliestFrameAndShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
  void RemoveEarliestFrame();
  void RemoveLatestFrame(int frame_count);
  void removeOutlier();

  void debugShow();

 private:
  double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count) const;

 public:
  std::list<FeaturePerId> feature;
  int last_track_num;

 private:
  const Eigen::Matrix3d *Rs;
  Eigen::Matrix3d ric[NUM_OF_CAM];
};

}  // namespace vins