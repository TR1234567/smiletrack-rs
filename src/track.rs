#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{Point, Rect, Size};

    #[test]
    fn test_track_initialization() {
        let bbox = Rect::new(100, 100, 50, 50);
        let track = Track::new(bbox, 1);

        assert_eq!(track.id, 1);
        assert_eq!(track.bbox, bbox);
        assert_eq!(track.time_since_update, 0);
        assert_eq!(track.hits, 1);
        assert!(!track.is_deleted);
    }

    #[test]
    fn test_track_predict() {
        let bbox = Rect::new(100, 100, 50, 50);
        let mut track = Track::new(bbox, 1);

        // Initial prediction should move based on zero velocity
        track.predict();
        assert_eq!(track.bbox, bbox); // Position should remain same initially

        // Set some velocity and predict again
        track.set_velocity(Point::new(10, 5));
        track.predict();
        
        // Check if position updated according to velocity
        assert_eq!(track.bbox.x, 110);
        assert_eq!(track.bbox.y, 105);
        assert_eq!(track.bbox.width, 50);
        assert_eq!(track.bbox.height, 50);
    }

    #[test]
    fn test_track_update() {
        let init_bbox = Rect::new(100, 100, 50, 50);
        let mut track = Track::new(init_bbox, 1);

        // Update with new detection
        let new_bbox = Rect::new(110, 105, 52, 48);
        track.update(new_bbox);

        assert_eq!(track.bbox, new_bbox);
        assert_eq!(track.time_since_update, 0);
        assert_eq!(track.hits, 2);
        assert!(!track.is_deleted);

        // Test velocity calculation
        let velocity = track.get_velocity();
        assert_eq!(velocity.x, 10); // dx = 110 - 100
        assert_eq!(velocity.y, 5);  // dy = 105 - 100
    }

    #[test]
    fn test_track_mark_missed() {
        let bbox = Rect::new(100, 100, 50, 50);
        let mut track = Track::new(bbox, 1);

        // Mark as missed multiple times
        for i in 1..=MAX_AGE {
            track.mark_missed();
            assert_eq!(track.time_since_update, i);
            assert!(!track.is_deleted);
        }

        // One more miss should mark as deleted
        track.mark_missed();
        assert!(track.is_deleted);
    }

    #[test]
    fn test_track_state_machine() {
        let bbox = Rect::new(100, 100, 50, 50);
        let mut track = Track::new(bbox, 1);

        // New track should be tentative
        assert!(!track.is_confirmed());

        // Update until confirmed
        for _ in 0..N_INIT-1 {
            track.update(bbox);
            assert!(!track.is_confirmed());
        }

        // One more update should confirm the track
        track.update(bbox);
        assert!(track.is_confirmed());

        // Missing updates should eventually delete the track
        for _ in 0..MAX_AGE {
            track.mark_missed();
        }
        track.mark_missed();
        assert!(track.is_deleted);
    }
} 