#include "initial_sfm.h"

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

namespace vins {

void GlobalSFM::triangulatePoint(const Eigen::Matrix<double, 3, 4> &Pose0, const Eigen::Matrix<double, 3, 4> &Pose1, const Eigen::Vector2d &point0,
                                 const Eigen::Vector2d &point1, Eigen::Vector3d &point_3d) {
  Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
  design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
  design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
  design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
  design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
  Eigen::Vector4d triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
  point_3d(0) = triangulated_point(0) / triangulated_point(3);
  point_3d(1) = triangulated_point(1) / triangulated_point(3);
  point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

bool GlobalSFM::solveFrameByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial, int i, std::vector<Feature> &sfm_f) {
  std::vector<cv::Point2f> pts_2_vector;
  std::vector<cv::Point3f> pts_3_vector;
  for (const auto &feature : sfm_f) {
    if (feature.global_position().isZero()) {
      continue;
    }

    for (const auto &observation : feature.observations()) {
      if (observation.frame_id == i) {
        pts_2_vector.push_back(cv::Point2f(observation.point.x(), observation.point.y()));
        pts_3_vector.push_back(cv::Point3f(feature.global_position().x(), feature.global_position().y(), feature.global_position().z()));
        break;
      }
    }
  }

  if (pts_2_vector.size() < 15) {
    printf("unstable features tracking, please slowly move you device!\n");
    if (pts_2_vector.size() < 10) {
      return false;
    }
  }

  cv::Mat r, rvec, t, D, tmp_r;
  cv::eigen2cv(R_initial, tmp_r);
  cv::Rodrigues(tmp_r, rvec);
  cv::eigen2cv(P_initial, t);
  cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
  bool pnp_succ;
  pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
  if (!pnp_succ) {
    return false;
  }
  cv::Rodrigues(rvec, r);
  // cout << "r " << endl << r << endl;
  Eigen::MatrixXd R_pnp;
  cv::cv2eigen(r, R_pnp);
  Eigen::MatrixXd T_pnp;
  cv::cv2eigen(t, T_pnp);
  R_initial = R_pnp;
  P_initial = T_pnp;
  return true;
}

void GlobalSFM::triangulateTwoFrames(int frame0, const Eigen::Matrix<double, 3, 4> &Pose0, int frame1, const Eigen::Matrix<double, 3, 4> &Pose1,
                                     std::vector<Feature> &sfm_f) {
  assert(frame0 != frame1);
  for (auto &feature : sfm_f) {
    if (!feature.global_position().isZero()) {
      continue;
    }

    bool has_0 = false, has_1 = false;
    Eigen::Vector2d point0, point1;
    for (const auto &item : feature.observations()) {
      if (item.frame_id == frame0) {
        point0 = item.point.head<2>();
        has_0 = true;
      } else if (item.frame_id == frame1) {
        point1 = item.point.head<2>();
        has_1 = true;
      }
    }

    if (has_0 && has_1) {
      Eigen::Vector3d point_3d;
      triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
      feature.mutable_global_position() = point_3d;
    }
  }
}

// 	 q w_R_cam t w_R_cam
//  c_rotation cam_R_w
//  c_translation cam_R_w
// relative_q[i][j]  j_q_i
// relative_t[i][j]  j_t_ji  (j < i)
bool GlobalSFM::construct(int frame_num, Eigen::Quaterniond *q, Eigen::Vector3d *T, int l, const Eigen::Matrix3d relative_R,
                          const Eigen::Vector3d relative_T, std::vector<Feature> &sfm_f, std::map<int, Eigen::Vector3d> &sfm_tracked_points) {
  feature_num = sfm_f.size();
  // cout << "set 0 and " << l << " as known " << endl;
  //  have relative_r relative_t
  //  intial two view
  q[l].setIdentity();
  T[l].setZero();
  q[frame_num - 1] = Eigen::Quaterniond(relative_R);
  T[frame_num - 1] = relative_T;

  // rotate to cam frame
  Eigen::Matrix3d c_Rotation[frame_num];
  Eigen::Vector3d c_Translation[frame_num];
  Eigen::Quaterniond c_Quat[frame_num];
  double c_rotation[frame_num][4];
  double c_translation[frame_num][3];
  Eigen::Matrix<double, 3, 4> Pose[frame_num];

  c_Quat[l] = q[l].inverse();
  c_Rotation[l] = c_Quat[l].toRotationMatrix();
  c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
  Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
  Pose[l].block<3, 1>(0, 3) = c_Translation[l];

  c_Quat[frame_num - 1] = q[frame_num - 1].inverse();
  c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
  c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
  Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
  Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];

  // 1: trangulate between l ----- frame_num - 1
  // 2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1;
  for (int i = l; i < frame_num - 1; i++) {
    // solve pnp
    if (i > l) {
      Eigen::Matrix3d R_initial = c_Rotation[i - 1];
      Eigen::Vector3d P_initial = c_Translation[i - 1];
      if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f)) {
        return false;
      }
      c_Rotation[i] = R_initial;
      c_Translation[i] = P_initial;
      c_Quat[i] = c_Rotation[i];
      Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
      Pose[i].block<3, 1>(0, 3) = c_Translation[i];
    }

    // triangulate point based on the solve pnp result
    triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
  }
  // 3: triangulate l-----l+1 l+2 ... frame_num -2
  for (int i = l + 1; i < frame_num - 1; i++) {
    triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);
  }
  // 4: solve pnp l-1; triangulate l-1 ----- l
  //              l-2              l-2 ----- l
  for (int i = l - 1; i >= 0; i--) {
    // solve pnp
    Eigen::Matrix3d R_initial = c_Rotation[i + 1];
    Eigen::Vector3d P_initial = c_Translation[i + 1];
    if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f)) {
      return false;
    }
    c_Rotation[i] = R_initial;
    c_Translation[i] = P_initial;
    c_Quat[i] = c_Rotation[i];
    Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
    Pose[i].block<3, 1>(0, 3) = c_Translation[i];
    // triangulate
    triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
  }
  // 5: triangulate all other points
  for (auto &feature : sfm_f) {
    if (!feature.global_position().isZero() || feature.observations().size() < 2) {
      continue;
    }

    const int prev_frame = feature.observations().front().frame_id;
    const int next_frame = feature.observations().back().frame_id;
    const Eigen::Vector2d prev_point = feature.observations().front().point.head<2>();
    const Eigen::Vector2d next_point = feature.observations().back().point.head<2>();

    Eigen::Vector3d point_3d;
    triangulatePoint(Pose[prev_frame], Pose[next_frame], prev_point, next_point, point_3d);

    feature.mutable_global_position() = point_3d;
  }

  // full BA
  ceres::Problem problem;
  ceres::LocalParameterization *local_parameterization = new ceres::QuaternionParameterization();
  for (int i = 0; i < frame_num; i++) {
    // double array for ceres
    c_translation[i][0] = c_Translation[i].x();
    c_translation[i][1] = c_Translation[i].y();
    c_translation[i][2] = c_Translation[i].z();
    c_rotation[i][0] = c_Quat[i].w();
    c_rotation[i][1] = c_Quat[i].x();
    c_rotation[i][2] = c_Quat[i].y();
    c_rotation[i][3] = c_Quat[i].z();
    problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
    problem.AddParameterBlock(c_translation[i], 3);
    if (i == l) {
      problem.SetParameterBlockConstant(c_rotation[i]);
    }
    if (i == l || i == frame_num - 1) {
      problem.SetParameterBlockConstant(c_translation[i]);
    }
  }

  for (auto &feature : sfm_f) {
    if (feature.global_position().isZero()) {
      continue;
    }

    for (const auto &observation : feature.observations()) {
      const int index = observation.frame_id;
      ceres::CostFunction *cost_function = ReprojectionError3D::Create(observation.point.x(), observation.point.y());
      problem.AddResidualBlock(cost_function, NULL, c_rotation[index], c_translation[index], feature.mutable_global_position().data());
    }
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  // options.minimizer_progress_to_stdout = true;
  options.max_solver_time_in_seconds = 0.2;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  // std::cout << summary.BriefReport() << "\n";
  if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03) {
    // cout << "vision only BA converge" << endl;
  } else {
    // cout << "vision only BA not converge " << endl;
    return false;
  }

  for (int i = 0; i < frame_num; i++) {
    q[i].w() = c_rotation[i][0];
    q[i].x() = c_rotation[i][1];
    q[i].y() = c_rotation[i][2];
    q[i].z() = c_rotation[i][3];
    q[i] = q[i].inverse();
  }

  for (int i = 0; i < frame_num; i++) {
    T[i] = -1 * (q[i] * Eigen::Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
  }

  for (const auto &feature : sfm_f) {
    if (feature.global_position().isZero()) {
      continue;
    }

    sfm_tracked_points[feature.feature_id()] = feature.global_position();
  }

  return true;
}

}  // namespace vins
