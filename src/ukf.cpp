#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Radar measurement dimension
  n_z_radar_ = 3;

  // Lidar measurement dimension
  n_z_lidar_ = 2;

  // Sigma point spreading parameter
  lambda_ = 3 - n_x_;

  // Augmented Sigma point spreading parameter
  lambda_aug_ = 3 - n_aug_;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial augmented mean vector
  x_aug_ = VectorXd(n_aug_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // initial augmented state covariance
  P_aug_ = MatrixXd(n_aug_, n_aug_);
  
  // initial sigma points matrix
  Xsig_ = MatrixXd(n_x_, 2 * n_x_ + 1);

  // initial augmented sigma point matrix
  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // initial predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // initial radar mean predicted measurement
  z_radar_pred_ = VectorXd(n_z_radar_);

  // initial radar measurement sigma points matrix
  Zsig_radar_ = MatrixXd(n_z_radar_, 2 * n_aug_ + 1);

  // initial predicted radar measurement covariance matrix
  S_radar_pred_ = MatrixXd(n_z_radar_, n_z_radar_);

  // initial radar measurement noise covariance matrix
  R_radar_ = MatrixXd(n_z_radar_,n_z_radar_);

  // initial lidar mean predicted measurement
  z_lidar_pred_ = VectorXd(n_z_lidar_);

  // initial lidar measurement sigma points matrix
  Zsig_lidar_ = MatrixXd(n_z_lidar_, 2 * n_aug_ + 1);

  // initial predicted lidar measurement covariance matrix
  S_lidar_pred_ = MatrixXd(n_z_lidar_, n_z_lidar_);

  // initial lidar measurement noise covariance matrix
  R_lidar_ = MatrixXd(n_z_lidar_,n_z_lidar_);


  // initial weights of sigma points
  weights_ = VectorXd(2 * n_aug_ + 1);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.125 * M_PI;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03; // 0.0175

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3; // 0.1
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
   GenerateWeights();
   GenerateMeasurementNoiseCovarianceMatrices();
}

UKF::~UKF() {}

void UKF::GenerateSigmaPoints() {
  // calculate square root of P_
  MatrixXd A = P_.llt().matrixL();

  // set first column of sigma point matrix
  Xsig_.col(0) = x_;

  // set remaining sigma points
  for (int i = 0; i < n_x_; ++i) {
    Xsig_.col(i + 1)        = x_ + sqrt(lambda_ + n_x_) * A.col(i);
    Xsig_.col(i + 1 + n_x_) = x_ - sqrt(lambda_ + n_x_) * A.col(i);
  }
}

void UKF::GenerateAugmentedSigmaPoints() {
  // create augmented mean state
  x_aug_.head(n_x_) = x_;
  x_aug_(n_x_) = 0;
  x_aug_(n_x_ + 1) = 0;

  // create augmented covariance matrix
  P_aug_.fill(0.0);
  P_aug_.topLeftCorner(n_x_,n_x_) = P_;
  P_aug_(n_x_,n_x_) = std_a_ * std_a_;
  P_aug_(n_x_ + 1,n_x_ + 1) = std_yawdd_ * std_yawdd_;

  // create square root matrix
  MatrixXd L = P_aug_.llt().matrixL();

  // create augmented sigma points
  Xsig_aug_.col(0)  = x_aug_;
  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug_.col(i + 1)          = x_aug_ + sqrt(lambda_aug_ + n_aug_) * L.col(i);
    Xsig_aug_.col(i + 1 + n_aug_) = x_aug_ - sqrt(lambda_aug_ + n_aug_) * L.col(i);
  }
}

void UKF::GenerateWeights() {
  double weight_0 = lambda_aug_/(lambda_aug_ + n_aug_);
  weights_(0) = weight_0;
  for (int i = 1; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 weights_
    weights_(i) = 0.5 / (n_aug_ + lambda_aug_);
  }
}

void UKF::GenerateMeasurementNoiseCovarianceMatrices() {
  R_radar_ <<  std_radr_ * std_radr_, 0, 0,
               0, std_radphi_ * std_radphi_, 0,
               0, 0, std_radrd_ * std_radrd_;
  
  R_lidar_ << std_laspx_ * std_laspx_, 0,
              0, std_laspy_ * std_laspy_;
}

void UKF::PredictSigmaPoints(double delta_t) {
  // predict sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    // extract values for better readability
    double p_x = Xsig_aug_(0, i);
    double p_y = Xsig_aug_(1, i);
    double v = Xsig_aug_(2, i);
    double yaw = Xsig_aug_(3, i);
    double yawd = Xsig_aug_(4, i);
    double nu_a = Xsig_aug_(5, i);
    double nu_yawdd = Xsig_aug_(6, i);

    // predicted state values
    double px_pred, py_pred;

    // avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_pred = p_x + v / yawd * (sin (yaw + yawd * delta_t) - sin(yaw));
        py_pred = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    } else {
        px_pred = p_x + v * delta_t * cos(yaw);
        py_pred = p_y + v * delta_t * sin(yaw);
    }

    double v_pred = v;
    double yaw_pred = yaw + yawd*delta_t;
    double yawd_pred = yawd;

    // add noise
    px_pred += 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_pred += 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_pred += nu_a * delta_t;
    yaw_pred += 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_pred += nu_yawdd * delta_t;

    // write predicted sigma point into right column
    Xsig_pred_(0, i) = px_pred;
    Xsig_pred_(1, i) = py_pred;
    Xsig_pred_(2, i) = v_pred;
    Xsig_pred_(3, i) = yaw_pred;
    Xsig_pred_(4, i) = yawd_pred;
  }
}

void UKF::PredictMeanStateAndCovarianceMatrix() {
  // predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // iterate over sigma points
    x_ += weights_(i) * Xsig_pred_.col(i);
  }

  // predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3) > M_PI)  x_diff(3) -= 2.0 * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2.0 * M_PI;

    P_ += weights_(i) * x_diff * x_diff.transpose() ;
  }
}

void UKF::PredictRadarMeasurement() {
  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // extract values for better readability
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig_radar_(0,i) = sqrt(p_x * p_x + p_y * p_y);                // r
    Zsig_radar_(1,i) = atan2(p_y, p_x);                            // phi
    Zsig_radar_(2,i) = (p_x * v1 + p_y * v2) / Zsig_radar_(0,i);   // r_dot
  }

  // mean predicted measurement
  z_radar_pred_.fill(0.0);
  for (int i=0; i < 2 * n_aug_ + 1; ++i) {
    z_radar_pred_ += weights_(i) * Zsig_radar_.col(i);
  }

  // innovation covariance matrix S
  S_radar_pred_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig_radar_.col(i) - z_radar_pred_;

    // angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2.0 * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;

    S_radar_pred_ += weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  S_radar_pred_ += R_radar_;
}

void UKF::PredictLidarMeasurement() {
  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    // extract values for better readability
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);

    // measurement model
    Zsig_lidar_(0, i) = p_x;
    Zsig_lidar_(1, i) = p_y;
  }

  // mean predicted measurement
  z_lidar_pred_.fill(0.0);
  for (int i=0; i < 2 * n_aug_ + 1; ++i) {
    z_lidar_pred_ += weights_(i) * Zsig_lidar_.col(i);
  }

  // innovation covariance matrix S
  S_lidar_pred_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig_lidar_.col(i) - z_lidar_pred_;

    // angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2.0 * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;

    S_lidar_pred_ += weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  S_lidar_pred_ += R_lidar_;
}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
   if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER) {
     PredictLidarMeasurement();
   }
   else if (meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR) {
     PredictRadarMeasurement();
   }
   else {
     std::cout << "Error: Unknown sensor type.\n";
   }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
   GenerateSigmaPoints();
   GenerateAugmentedSigmaPoints();
   PredictSigmaPoints(delta_t);
   PredictMeanStateAndCovarianceMatrix();
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_lidar_);

  // calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig_lidar_.col(i) - z_lidar_pred_;
    // angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2.0 * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3) > M_PI) x_diff(3) -= 2.0 * M_PI;
    while (x_diff(3) <- M_PI) x_diff(3) += 2.0 * M_PI;

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc * S_lidar_pred_.inverse();

  // residual
  VectorXd z_diff = meas_package.raw_measurements_ - z_lidar_pred_;

  // angle normalization
  while (z_diff(1) > M_PI) z_diff(1) -= 2.0 * M_PI;
  while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;

  // update state mean and covariance matrix
  x_ += K * z_diff;
  P_ -= K * S_lidar_pred_ * K.transpose();
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_radar_);

  // calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig_radar_.col(i) - z_radar_pred_;
    // angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2.0 * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3) > M_PI) x_diff(3) -= 2.0 * M_PI;
    while (x_diff(3) <- M_PI) x_diff(3) += 2.0 * M_PI;

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc * S_radar_pred_.inverse();

  // residual
  VectorXd z_diff = meas_package.raw_measurements_ - z_radar_pred_;

  // angle normalization
  while (z_diff(1) > M_PI) z_diff(1) -= 2.0 * M_PI;
  while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;

  // update state mean and covariance matrix
  x_ += K * z_diff;
  P_ -= K * S_radar_pred_ * K.transpose();
}