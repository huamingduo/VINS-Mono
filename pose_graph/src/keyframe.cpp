#include "keyframe.h"

using namespace Eigen;
using namespace std;
using namespace DVision;

namespace vins {

template <typename Derived>
static void reduceVector(std::vector<Derived> &v, std::vector<uchar> status) {
  int j = 0;
  for (int i = 0; i < int(v.size()); i++) {
    if (status[i]) {
      v[j++] = v[i];
    }
  }
  v.resize(j);
}

// create keyframe online
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image, std::vector<cv::Point3f> &_point_3d,
                   std::vector<cv::Point2f> &_point_2d_uv, std::vector<cv::Point2f> &_point_2d_norm, std::vector<double> &_point_id, int _sequence) {
  time_stamp = _time_stamp;
  index = _index;
  vio_T_w_i = _vio_T_w_i;
  vio_R_w_i = _vio_R_w_i;
  T_w_i = vio_T_w_i;
  R_w_i = vio_R_w_i;
  origin_vio_T = vio_T_w_i;
  origin_vio_R = vio_R_w_i;
  image = _image.clone();
  cv::resize(image, thumbnail, cv::Size(80, 60));
  point_3d = _point_3d;
  point_2d_uv = _point_2d_uv;
  point_2d_norm = _point_2d_norm;
  point_id = _point_id;
  loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
  sequence = _sequence;
  computeWindowBRIEFPoint();
  computeBRIEFPoint();
  if (!DEBUG_IMAGE) image.release();
}

// load previous keyframe
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i, cv::Mat &_image,
                   int _loop_index, Eigen::Matrix<double, 8, 1> &_loop_info, std::vector<cv::KeyPoint> &_keypoints,
                   std::vector<cv::KeyPoint> &_keypoints_norm, std::vector<BRIEF::bitset> &_brief_descriptors) {
  time_stamp = _time_stamp;
  index = _index;
  // vio_T_w_i = _vio_T_w_i;
  // vio_R_w_i = _vio_R_w_i;
  vio_T_w_i = _T_w_i;
  vio_R_w_i = _R_w_i;
  T_w_i = _T_w_i;
  R_w_i = _R_w_i;
  if (DEBUG_IMAGE) {
    image = _image.clone();
    cv::resize(image, thumbnail, cv::Size(80, 60));
  }
  if (_loop_index != -1)
    has_loop = true;
  else
    has_loop = false;
  loop_index = _loop_index;
  loop_info = _loop_info;
  sequence = 0;
  keypoints = _keypoints;
  keypoints_norm = _keypoints_norm;
  brief_descriptors = _brief_descriptors;
}

void KeyFrame::computeWindowBRIEFPoint() {
  BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
  for (int i = 0; i < (int)point_2d_uv.size(); i++) {
    cv::KeyPoint key;
    key.pt = point_2d_uv[i];
    window_keypoints.push_back(key);
  }
  extractor(image, window_keypoints, window_brief_descriptors);
}

void KeyFrame::computeBRIEFPoint() {
  BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
  const int fast_th = 20;  // corner detector response threshold
  if (1)
    cv::FAST(image, keypoints, fast_th, true);
  else {
    std::vector<cv::Point2f> tmp_pts;
    cv::goodFeaturesToTrack(image, tmp_pts, 500, 0.01, 10);
    for (int i = 0; i < (int)tmp_pts.size(); i++) {
      cv::KeyPoint key;
      key.pt = tmp_pts[i];
      keypoints.push_back(key);
    }
  }
  extractor(image, keypoints, brief_descriptors);
  for (int i = 0; i < (int)keypoints.size(); i++) {
    Eigen::Vector3d tmp_p;
    m_camera->liftProjective(Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y), tmp_p);
    cv::KeyPoint tmp_norm;
    tmp_norm.pt = cv::Point2f(tmp_p.x() / tmp_p.z(), tmp_p.y() / tmp_p.z());
    keypoints_norm.push_back(tmp_norm);
  }
}

bool KeyFrame::searchInAera(const BRIEF::bitset window_descriptor, const std::vector<BRIEF::bitset> &descriptors_old,
                            const std::vector<cv::KeyPoint> &keypoints_old, const std::vector<cv::KeyPoint> &keypoints_old_norm,
                            cv::Point2f &best_match, cv::Point2f &best_match_norm) {
  cv::Point2f best_pt;
  int bestDist = 128;
  int bestIndex = -1;
  for (int i = 0; i < (int)descriptors_old.size(); i++) {
    int dis = HammingDis(window_descriptor, descriptors_old[i]);
    if (dis < bestDist) {
      bestDist = dis;
      bestIndex = i;
    }
  }
  // printf("best dist %d", bestDist);
  if (bestIndex != -1 && bestDist < 80) {
    best_match = keypoints_old[bestIndex].pt;
    best_match_norm = keypoints_old_norm[bestIndex].pt;
    return true;
  } else
    return false;
}

void KeyFrame::searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old, std::vector<cv::Point2f> &matched_2d_old_norm, std::vector<uchar> &status,
                                const std::vector<BRIEF::bitset> &descriptors_old, const std::vector<cv::KeyPoint> &keypoints_old,
                                const std::vector<cv::KeyPoint> &keypoints_old_norm) {
  for (int i = 0; i < (int)window_brief_descriptors.size(); i++) {
    cv::Point2f pt(0.f, 0.f);
    cv::Point2f pt_norm(0.f, 0.f);
    if (searchInAera(window_brief_descriptors[i], descriptors_old, keypoints_old, keypoints_old_norm, pt, pt_norm))
      status.push_back(1);
    else
      status.push_back(0);
    matched_2d_old.push_back(pt);
    matched_2d_old_norm.push_back(pt_norm);
  }
}

void KeyFrame::FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm, const std::vector<cv::Point2f> &matched_2d_old_norm,
                                      std::vector<uchar> &status) {
  int n = (int)matched_2d_cur_norm.size();
  for (int i = 0; i < n; i++) status.push_back(0);
  if (n >= 8) {
    std::vector<cv::Point2f> tmp_cur(n), tmp_old(n);
    for (int i = 0; i < (int)matched_2d_cur_norm.size(); i++) {
      double FOCAL_LENGTH = 460.0;
      double tmp_x, tmp_y;
      tmp_x = FOCAL_LENGTH * matched_2d_cur_norm[i].x + COL / 2.0;
      tmp_y = FOCAL_LENGTH * matched_2d_cur_norm[i].y + ROW / 2.0;
      tmp_cur[i] = cv::Point2f(tmp_x, tmp_y);

      tmp_x = FOCAL_LENGTH * matched_2d_old_norm[i].x + COL / 2.0;
      tmp_y = FOCAL_LENGTH * matched_2d_old_norm[i].y + ROW / 2.0;
      tmp_old[i] = cv::Point2f(tmp_x, tmp_y);
    }
    cv::findFundamentalMat(tmp_cur, tmp_old, cv::FM_RANSAC, 3.0, 0.9, status);
  }
}

void KeyFrame::PnPRANSAC(const std::vector<cv::Point2f> &matched_2d_old_norm, const std::vector<cv::Point3f> &matched_3d, std::vector<uchar> &status,
                         Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old) {
  // for (int i = 0; i < matched_3d.size(); i++)
  //	printf("3d x: %f, y: %f, z: %f\n",matched_3d[i].x, matched_3d[i].y, matched_3d[i].z );
  // printf("match size %d \n", matched_3d.size());
  cv::Mat r, rvec, t, D, tmp_r;
  cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
  Matrix3d R_inital;
  Vector3d P_inital;
  Matrix3d R_w_c = origin_vio_R * qic;
  Vector3d T_w_c = origin_vio_T + origin_vio_R * tic;

  R_inital = R_w_c.inverse();
  P_inital = -(R_inital * T_w_c);

  cv::eigen2cv(R_inital, tmp_r);
  cv::Rodrigues(tmp_r, rvec);
  cv::eigen2cv(P_inital, t);

  cv::Mat inliers;
  TicToc t_pnp_ransac;

  if (CV_MAJOR_VERSION < 3)
    solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 100, inliers);
  else {
    if (CV_MINOR_VERSION < 2)
      solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, sqrt(10.0 / 460.0), 0.99, inliers);
    else
      solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 0.99, inliers);
  }

  for (int i = 0; i < (int)matched_2d_old_norm.size(); i++) status.push_back(0);

  for (int i = 0; i < inliers.rows; i++) {
    int n = inliers.at<int>(i);
    status[n] = 1;
  }

  cv::Rodrigues(rvec, r);
  Matrix3d R_pnp, R_w_c_old;
  cv::cv2eigen(r, R_pnp);
  R_w_c_old = R_pnp.transpose();
  Vector3d T_pnp, T_w_c_old;
  cv::cv2eigen(t, T_pnp);
  T_w_c_old = R_w_c_old * (-T_pnp);

  PnP_R_old = R_w_c_old * qic.transpose();
  PnP_T_old = T_w_c_old - PnP_R_old * tic;
}

bool KeyFrame::findConnection(KeyFrame *old_kf) {
  // printf("find Connection\n");
  std::vector<cv::Point3f> matched_3d = point_3d;
  std::vector<cv::Point2f> matched_2d_cur = point_2d_uv;
  std::vector<cv::Point2f> matched_2d_cur_norm = point_2d_norm;
  std::vector<double> matched_id = point_id;

  // printf("search by des\n");
  std::vector<uchar> status;
  std::vector<cv::Point2f> matched_2d_old;
  std::vector<cv::Point2f> matched_2d_old_norm;
  searchByBRIEFDes(matched_2d_old, matched_2d_old_norm, status, old_kf->brief_descriptors, old_kf->keypoints, old_kf->keypoints_norm);
  reduceVector(matched_2d_cur, status);
  reduceVector(matched_2d_old, status);
  reduceVector(matched_2d_cur_norm, status);
  reduceVector(matched_2d_old_norm, status);
  reduceVector(matched_3d, status);
  reduceVector(matched_id, status);
  // printf("search by des finish\n");

  status.clear();

  Eigen::Vector3d PnP_T_old;
  Eigen::Matrix3d PnP_R_old;
  if ((int)matched_2d_cur.size() > MIN_LOOP_NUM) {
    PnPRANSAC(matched_2d_old_norm, matched_3d, status, PnP_T_old, PnP_R_old);
    reduceVector(matched_2d_cur, status);
    reduceVector(matched_2d_old, status);
    reduceVector(matched_2d_cur_norm, status);
    reduceVector(matched_2d_old_norm, status);
    reduceVector(matched_3d, status);
    reduceVector(matched_id, status);
#if 1
    if (DEBUG_IMAGE) {
      int gap = 10;
      cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
      cv::Mat gray_img, loop_match_img;
      cv::Mat old_img = old_kf->image;
      cv::hconcat(image, gap_image, gap_image);
      cv::hconcat(gap_image, old_img, gray_img);
      cvtColor(gray_img, loop_match_img, cv::COLOR_GRAY2RGB);
      for (int i = 0; i < (int)matched_2d_cur.size(); i++) {
        cv::Point2f cur_pt = matched_2d_cur[i];
        cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
      }
      for (int i = 0; i < (int)matched_2d_old.size(); i++) {
        cv::Point2f old_pt = matched_2d_old[i];
        old_pt.x += (COL + gap);
        cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
      }
      for (int i = 0; i < (int)matched_2d_cur.size(); i++) {
        cv::Point2f old_pt = matched_2d_old[i];
        old_pt.x += (COL + gap);
        cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 2, 8, 0);
      }
      cv::Mat notation(50, COL + gap + COL, CV_8UC3, cv::Scalar(255, 255, 255));
      putText(notation, "current frame: " + to_string(index) + "  sequence: " + to_string(sequence), cv::Point2f(20, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
              cv::Scalar(255), 3);

      putText(notation, "previous frame: " + to_string(old_kf->index) + "  sequence: " + to_string(old_kf->sequence), cv::Point2f(20 + COL + gap, 30),
              cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
      cv::vconcat(notation, loop_match_img, loop_match_img);

      /*
      ostringstream path;
      path <<  "/home/tony-ws1/raw_data/loop_image/"
              << index << "-"
              << old_kf->index << "-" << "3pnp_match.jpg";
      cv::imwrite( path.str().c_str(), loop_match_img);
      */
      if ((int)matched_2d_cur.size() > MIN_LOOP_NUM) {
        /*
        cv::imshow("loop connection",loop_match_img);
        cv::waitKey(10);
        */
        cv::Mat thumbimage;
        cv::resize(loop_match_img, thumbimage, cv::Size(loop_match_img.cols / 2, loop_match_img.rows / 2));
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", thumbimage).toImageMsg();
        msg->header.stamp = ros::Time(time_stamp);
        pub_match_img.publish(msg);
      }
    }
#endif
  }

  if ((int)matched_2d_cur.size() > MIN_LOOP_NUM) {
    Eigen::Vector3d relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old);
    Eigen::Quaterniond relative_q{PnP_R_old.transpose() * origin_vio_R};
    double relative_yaw = Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).x() - Utility::R2ypr(PnP_R_old).x());
    // printf("PNP relative\n");
    // cout << "pnp relative_t " << relative_t.transpose() << endl;
    // cout << "pnp relative_yaw " << relative_yaw << endl;
    if (abs(relative_yaw) < 30.0 && relative_t.norm() < 20.0) {
      has_loop = true;
      loop_index = old_kf->index;
      loop_info << relative_t.x(), relative_t.y(), relative_t.z(), relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(), relative_yaw;
      if (FAST_RELOCALIZATION) {
        sensor_msgs::PointCloud msg_match_points;
        msg_match_points.header.stamp = ros::Time(time_stamp);
        for (int i = 0; i < (int)matched_2d_old_norm.size(); i++) {
          geometry_msgs::Point32 p;
          p.x = matched_2d_old_norm[i].x;
          p.y = matched_2d_old_norm[i].y;
          p.z = matched_id[i];
          msg_match_points.points.push_back(p);
        }
        Eigen::Vector3d T = old_kf->T_w_i;
        Eigen::Matrix3d R = old_kf->R_w_i;
        Quaterniond Q(R);
        sensor_msgs::ChannelFloat32 t_q_index;
        t_q_index.values.push_back(T.x());
        t_q_index.values.push_back(T.y());
        t_q_index.values.push_back(T.z());
        t_q_index.values.push_back(Q.w());
        t_q_index.values.push_back(Q.x());
        t_q_index.values.push_back(Q.y());
        t_q_index.values.push_back(Q.z());
        t_q_index.values.push_back(index);
        msg_match_points.channels.push_back(t_q_index);
        pub_match_points.publish(msg_match_points);
      }
      return true;
    }
  }
  return false;
}

}  // namespace vins
