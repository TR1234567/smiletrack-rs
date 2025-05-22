#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_kalman_filter_initialization() {
        let kf = KalmanFilter::new();
        
        // Check dimensions
        assert_eq!(kf.mean.nrows(), 8);
        assert_eq!(kf.covariance.nrows(), 8);
        assert_eq!(kf.covariance.ncols(), 8);
        
        // Check initial values
        assert_eq!(kf.mean, SVector::<f32, 8>::zeros());
        
        // Covariance should be diagonal with high uncertainty
        for i in 0..8 {
            for j in 0..8 {
                if i == j {
                    assert!(kf.covariance[(i, j)] > 0.0);
                } else {
                    assert_eq!(kf.covariance[(i, j)], 0.0);
                }
            }
        }
    }

    #[test]
    fn test_kalman_predict() {
        let mut kf = KalmanFilter::new();
        
        // Set initial state
        let initial_pos = SVector::<f32, 4>::new(100.0, 100.0, 50.0, 50.0);
        let initial_vel = SVector::<f32, 4>::new(10.0, 5.0, 0.0, 0.0);
        kf.mean.fixed_view_mut::<4, 1>(0, 0).copy_from(&initial_pos);
        kf.mean.fixed_view_mut::<4, 1>(4, 0).copy_from(&initial_vel);

        // Predict
        kf.predict();

        // Position should be updated based on velocity
        assert_relative_eq!(kf.mean[0], 110.0); // x + vx
        assert_relative_eq!(kf.mean[1], 105.0); // y + vy
        assert_relative_eq!(kf.mean[2], 50.0);  // w (unchanged)
        assert_relative_eq!(kf.mean[3], 50.0);  // h (unchanged)

        // Velocities should remain the same
        assert_relative_eq!(kf.mean[4], 10.0);
        assert_relative_eq!(kf.mean[5], 5.0);
        assert_relative_eq!(kf.mean[6], 0.0);
        assert_relative_eq!(kf.mean[7], 0.0);

        // Covariance should increase due to process noise
        let initial_cov = kf.covariance.clone();
        kf.predict();
        assert!(kf.covariance.diagonal().norm() > initial_cov.diagonal().norm());
    }

    #[test]
    fn test_kalman_update() {
        let mut kf = KalmanFilter::new();
        
        // Set initial state with some uncertainty
        let initial_pos = SVector::<f32, 4>::new(100.0, 100.0, 50.0, 50.0);
        kf.mean.fixed_view_mut::<4, 1>(0, 0).copy_from(&initial_pos);
        
        // Create measurement slightly offset from prediction
        let measurement = SVector::<f32, 4>::new(110.0, 105.0, 52.0, 48.0);
        
        // Update
        kf.update(&measurement);

        // State should be somewhere between prediction and measurement
        assert!(kf.mean[0] > 100.0 && kf.mean[0] < 110.0);
        assert!(kf.mean[1] > 100.0 && kf.mean[1] < 105.0);
        assert!(kf.mean[2] > 50.0 && kf.mean[2] < 52.0);
        assert!(kf.mean[3] > 48.0 && kf.mean[3] < 50.0);

        // Covariance should decrease after measurement update
        let pre_update_cov = kf.covariance.clone();
        kf.update(&measurement);
        assert!(kf.covariance.diagonal().norm() < pre_update_cov.diagonal().norm());
    }

    #[test]
    fn test_chi2inv95() {
        // Test chi-square 95th percentile values
        assert!(chi2inv95(1) > 0.0);
        assert!(chi2inv95(2) > chi2inv95(1));
        assert!(chi2inv95(3) > chi2inv95(2));
        assert!(chi2inv95(4) > chi2inv95(3));
    }
} 