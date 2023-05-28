#pragma once

#include <eigen3/Eigen/Eigen>

namespace vins {

struct Observation {
  double stamp = -1.;
  int frame_id = -1;

  Eigen::Vector3d point = -Eigen::Vector3d::Ones();  // normalized feature position (with z = 1) in current camera frame
  Eigen::Vector2d pixel = -Eigen::Vector2d::Ones();  // feature pixel position in current camera frame
  Eigen::Vector2d flow = Eigen::Vector2d::Zero();    // feature pixel flow speed

  Observation() = delete;
  Observation(const double& t) : stamp(t) {}
  Observation(const int& f) : frame_id(f) {}
};

class Feature {
 public:
  Feature() = delete;
  Feature(const int& feature_id) : feature_id_(feature_id) {}
  ~Feature() {}

  const int& feature_id() const { return feature_id_; }
  const int& start_frame_id() const { return start_frame_id_; }
  const int& end_frame_id() const { return end_frame_id_; }
  const double& start_stamp() const { return start_stamp_; }
  const double& end_stamp() const { return end_stamp_; }
  const Eigen::Vector3d& global_position() const { return global_position_; }
  const std::vector<Observation>& observations() const { return observations_; }

  Eigen::Vector3d& mutable_global_position() { return global_position_; }

  void InsertObservation(const int& frame_id, const Eigen::Vector3d& point) {
    if (start_frame_id_ == -1) {
      start_frame_id_ = frame_id;
    }
    end_frame_id_ = std::max(end_frame_id_, frame_id);

    Observation observation{frame_id};
    observation.point = point;
    observations_.push_back(observation);
  }

  void RemoveObservationAtFrame(const int& frame_id = 0) {
    if (start_frame_id_ == -1 || frame_id < start_frame_id_ || frame_id > end_frame_id_) {
      return;
    }

    for (auto it = observations_.begin(); it != observations_.end();) {
      if (it->frame_id != frame_id) {
        if (it->frame_id > frame_id) {
          --it->frame_id;
        }
        it = std::next(it);
        continue;
      }

      it = observations_.erase(it);
    }

    --end_frame_id_;
    if (start_frame_id_ > end_frame_id_) {
      start_frame_id_ = -1;
      end_frame_id_ = -1;
      global_position_ = Eigen::Vector3d::Zero();
      observations_.clear();
    }
  }

 private:
  const int feature_id_;
  int start_frame_id_ = -1;
  int end_frame_id_ = -1;
  double start_stamp_ = -1.;
  double end_stamp_ = -1.;

  Eigen::Vector3d global_position_ = Eigen::Vector3d::Zero();
  std::vector<Observation> observations_;
};

}  // namespace vins