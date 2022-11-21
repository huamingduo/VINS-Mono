#pragma once

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "parameters.h"
#include "tic_toc.h"

namespace vins {

inline bool inBorder(const cv::Point2f &pt) {
  const int BORDER_SIZE = 1;
  int img_x = cvRound(pt.x);
  int img_y = cvRound(pt.y);
  return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

template <typename T>
void reduceVector(std::vector<T> &v, const std::vector<uchar> &status) {
  int j = 0;
  for (int i = 0; i < int(v.size()); i++)
    if (status[i]) v[j++] = v[i];
  v.resize(j);
}

class FeatureTracker {
 public:
  FeatureTracker() {}
  ~FeatureTracker() {}

  void readImage(const cv::Mat &_img, double _cur_time);
  bool updateID(unsigned int i);
  void readIntrinsicParameter(const std::string &calib_file);
#ifdef SHOW_UNDISTORTION
  void showUndistortion(const std::string &name);
#endif

 private:
  void rejectWithF();
  void setMask();
  void detectFeatures();
  void addPoints();
  void undistortedPoints();

  cv::Mat EqualizeIfNeeded(const cv::Mat &image, const double &clip_limit = 40., const cv::Size &tile_grid_size = cv::Size(8, 8)) const;

 public:
  cv::Mat fisheye_mask, cur_img;
  std::vector<cv::Point2f> cur_pts;
  std::vector<cv::Point2f> cur_un_pts;
  std::vector<cv::Point2f> pts_velocity;
  std::vector<int> ids;
  std::vector<int> track_cnt;
  camodocal::CameraPtr m_camera;

 private:
  static int n_id;

  double cur_time;
  double prev_time;

  cv::Mat mask, forw_img;
  std::vector<cv::Point2f> n_pts, forw_pts;
};

}  // namespace vins
