#pragma once

#include <eigen3/Eigen/Eigen>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "parameters.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"

#define MIN_LOOP_NUM 25

namespace vins {

class BriefExtractor {
 public:
  BriefExtractor(const std::string &pattern_file) {
    // The DVision::BRIEF extractor computes a random pattern by default when
    // the object is created.
    // We load the pattern that we used to build the vocabulary, to make
    // the descriptors compatible with the predefined vocabulary

    // loads the pattern
    cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
    if (!fs.isOpened()) {
      throw string("Could not open file ") + pattern_file;
    }

    std::vector<int> x1, y1, x2, y2;
    fs["x1"] >> x1;
    fs["x2"] >> x2;
    fs["y1"] >> y1;
    fs["y2"] >> y2;

    m_brief.importPairs(x1, y1, x2, y2);
  }

  virtual void operator()(const cv::Mat &im, std::vector<cv::KeyPoint> &keys, std::vector<DVision::BRIEF::bitset> &descriptors) const {
    m_brief.compute(im, keys, descriptors);
  }

 private:
  DVision::BRIEF m_brief;
};

class KeyFrame {
 public:
  KeyFrame(double _time_stamp, int _index, Eigen::Vector3d &_vio_T_w_i, Eigen::Matrix3d &_vio_R_w_i, cv::Mat &_image,
           std::vector<cv::Point3f> &_point_3d, std::vector<cv::Point2f> &_point_2d_uv, std::vector<cv::Point2f> &_point_2d_normal,
           std::vector<double> &_point_id, int _sequence);
  KeyFrame(double _time_stamp, int _index, Eigen::Vector3d &_vio_T_w_i, Eigen::Matrix3d &_vio_R_w_i, Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i,
           cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1> &_loop_info, std::vector<cv::KeyPoint> &_keypoints,
           std::vector<cv::KeyPoint> &_keypoints_norm, std::vector<DVision::BRIEF::bitset> &_brief_descriptors);

  bool findConnection(KeyFrame *old_kf);

  void updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i) {
    T_w_i = _T_w_i;
    R_w_i = _R_w_i;
  }
  void updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i) {
    vio_T_w_i = _T_w_i;
    vio_R_w_i = _R_w_i;
    T_w_i = vio_T_w_i;
    R_w_i = vio_R_w_i;
  }
  void updateLoop(Eigen::Matrix<double, 8, 1> &_loop_info) {
    if (std::abs(_loop_info(7)) < 30.0 && Eigen::Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0) {
      loop_info = _loop_info;
    }
  }

  void getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i) const {
    _T_w_i = T_w_i;
    _R_w_i = R_w_i;
  }
  void getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i) const {
    _T_w_i = vio_T_w_i;
    _R_w_i = vio_R_w_i;
  }
  const Eigen::Vector3d getLoopRelativeT() const { return Eigen::Vector3d(loop_info(0), loop_info(1), loop_info(2)); }
  const Eigen::Quaterniond getLoopRelativeQ() const { return Eigen::Quaterniond(loop_info(3), loop_info(4), loop_info(5), loop_info(6)); }
  const double &getLoopRelativeYaw() const { return loop_info(7); }

 private:
  inline int HammingDis(const DVision::BRIEF::bitset &a, const DVision::BRIEF::bitset &b) {
    DVision::BRIEF::bitset xor_of_bitset = a ^ b;
    int dis = xor_of_bitset.count();
    return dis;
  }

  void computeWindowBRIEFPoint();
  void computeBRIEFPoint();

  bool searchInAera(const DVision::BRIEF::bitset window_descriptor, const std::vector<DVision::BRIEF::bitset> &descriptors_old,
                    const std::vector<cv::KeyPoint> &keypoints_old, const std::vector<cv::KeyPoint> &keypoints_old_norm, cv::Point2f &best_match,
                    cv::Point2f &best_match_norm);
  void searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old, std::vector<cv::Point2f> &matched_2d_old_norm, std::vector<uchar> &status,
                        const std::vector<DVision::BRIEF::bitset> &descriptors_old, const std::vector<cv::KeyPoint> &keypoints_old,
                        const std::vector<cv::KeyPoint> &keypoints_old_norm);
  void FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm, const std::vector<cv::Point2f> &matched_2d_old_norm,
                              std::vector<uchar> &status);
  void PnPRANSAC(const std::vector<cv::Point2f> &matched_2d_old_norm, const std::vector<cv::Point3f> &matched_3d, std::vector<uchar> &status,
                 Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old);

 public:
  bool has_loop = false;
  int index;
  int local_index;
  int loop_index = -1;
  int sequence;
  double time_stamp;

  Eigen::Vector3d vio_T_w_i;
  Eigen::Matrix3d vio_R_w_i;
  Eigen::Vector3d T_w_i;
  Eigen::Matrix3d R_w_i;
  Eigen::Matrix<double, 8, 1> loop_info;

  cv::Mat image;
  std::vector<cv::KeyPoint> keypoints;
  std::vector<cv::KeyPoint> keypoints_norm;
  std::vector<DVision::BRIEF::bitset> brief_descriptors;

 private:
  Eigen::Vector3d origin_vio_T;
  Eigen::Matrix3d origin_vio_R;

  cv::Mat thumbnail;
  std::vector<double> point_id;
  std::vector<cv::Point3f> point_3d;
  std::vector<cv::Point2f> point_2d_uv;
  std::vector<cv::Point2f> point_2d_norm;

  std::vector<cv::KeyPoint> window_keypoints;
  std::vector<DVision::BRIEF::bitset> window_brief_descriptors;
};

}  // namespace vins
