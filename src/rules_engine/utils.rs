use super::state::{Planet, Point, BOARD_SIZE, CENTER, ROTATION_RADIUS_LIMIT, SUN_RADIUS};

const TARGET_EPS: f64 = 1e-6;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticTargetRay {
    pub angle: f64,
    pub end: Point,
}

pub fn distance(a: Point, b: Point) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

pub fn is_orbiting(position: Point, radius: f64) -> bool {
    distance(position, Point::new(CENTER, CENTER)) + radius < ROTATION_RADIUS_LIMIT
}

pub fn orbit_position(initial_position: Point, angular_velocity: f64, step: f64) -> Point {
    let dx = initial_position.x - CENTER;
    let dy = initial_position.y - CENTER;
    let orbital_radius = (dx * dx + dy * dy).sqrt();
    let initial_angle = dy.atan2(dx);
    let current_angle = initial_angle + angular_velocity * step;
    Point::new(
        CENTER + orbital_radius * current_angle.cos(),
        CENTER + orbital_radius * current_angle.sin(),
    )
}

pub fn fourfold_symmetric_points(point: Point) -> [Point; 4] {
    [
        Point::new(point.y, point.x),
        Point::new(BOARD_SIZE - point.x, point.y),
        Point::new(point.x, BOARD_SIZE - point.y),
        Point::new(BOARD_SIZE - point.y, BOARD_SIZE - point.x),
    ]
}

pub fn point_to_segment_distance(point: Point, start: Point, end: Point) -> f64 {
    let dx = start.x - end.x;
    let dy = start.y - end.y;
    let length_squared = dx * dx + dy * dy;
    if length_squared == 0.0 {
        return distance(point, start);
    }

    let t = (((point.x - start.x) * (end.x - start.x) + (point.y - start.y) * (end.y - start.y))
        / length_squared)
        .clamp(0.0, 1.0);
    let projection = Point::new(
        start.x + t * (end.x - start.x),
        start.y + t * (end.y - start.y),
    );
    distance(point, projection)
}

pub fn angle_between(start: Point, end: Point) -> f64 {
    (end.y - start.y).atan2(end.x - start.x)
}

pub fn point_along(start: Point, angle: f64, distance: f64) -> Point {
    Point::new(
        start.x + angle.cos() * distance,
        start.y + angle.sin() * distance,
    )
}

pub fn launch_start(source: &Planet, angle: f64) -> Point {
    point_along(source.position(), angle, source.radius + 0.1)
}

pub fn static_target_rays(source: &Planet, target: &Planet) -> Vec<StaticTargetRay> {
    let source_pos = source.position();
    let target_pos = target.position();
    let base_angle = angle_between(source_pos, target_pos);
    let mut rays = vec![static_target_ray_for_angle(source, target, base_angle)];
    let distance_to_target = distance(source_pos, target_pos);
    let radius = (target.radius - TARGET_EPS).max(0.0);
    if distance_to_target > radius && radius > 0.0 {
        let half_angle = (radius / distance_to_target).asin();
        rays.push(static_target_ray_for_angle(
            source,
            target,
            base_angle + half_angle,
        ));
        rays.push(static_target_ray_for_angle(
            source,
            target,
            base_angle - half_angle,
        ));
    }
    rays
}

pub fn best_static_target_angle<'a, I>(
    source: &Planet,
    target: &Planet,
    static_blockers: I,
) -> Option<f64>
where
    I: Iterator<Item = &'a Planet> + Clone,
{
    let rays = static_target_rays(source, target);
    rays.into_iter()
        .find(|ray| {
            !static_ray_hits_sun(source, *ray)
                && !static_blockers
                    .clone()
                    .any(|blocker| static_ray_hits_planet(source, *ray, blocker))
        })
        .map(|ray| ray.angle)
}

pub fn static_ray_hits_sun(source: &Planet, ray: StaticTargetRay) -> bool {
    point_to_segment_distance(
        Point::new(CENTER, CENTER),
        launch_start(source, ray.angle),
        ray.end,
    ) < SUN_RADIUS
}

pub fn static_ray_hits_planet(source: &Planet, ray: StaticTargetRay, planet: &Planet) -> bool {
    point_to_segment_distance(planet.position(), launch_start(source, ray.angle), ray.end)
        < planet.radius
}

fn static_target_ray_for_angle(source: &Planet, target: &Planet, angle: f64) -> StaticTargetRay {
    let start = launch_start(source, angle);
    let dir = Point::new(angle.cos(), angle.sin());
    let target_pos = target.position();
    let to_target = Point::new(target_pos.x - start.x, target_pos.y - start.y);
    let projection = to_target.x * dir.x + to_target.y * dir.y;
    let perpendicular_squared =
        (to_target.x * to_target.x + to_target.y * to_target.y) - projection * projection;
    let hit_distance = if perpendicular_squared < target.radius * target.radius {
        projection - (target.radius * target.radius - perpendicular_squared.max(0.0)).sqrt()
    } else {
        projection
    }
    .max(0.0);
    StaticTargetRay {
        angle,
        end: point_along(start, angle, hit_distance),
    }
}

pub fn swept_pair_hit(
    fleet_start: Point,
    fleet_end: Point,
    planet_start: Point,
    planet_end: Point,
    radius: f64,
) -> bool {
    let d0x = fleet_start.x - planet_start.x;
    let d0y = fleet_start.y - planet_start.y;
    let dvx = (fleet_end.x - fleet_start.x) - (planet_end.x - planet_start.x);
    let dvy = (fleet_end.y - fleet_start.y) - (planet_end.y - planet_start.y);
    let a = dvx * dvx + dvy * dvy;
    let b = 2.0 * (d0x * dvx + d0y * dvy);
    let c = d0x * d0x + d0y * d0y - radius * radius;
    if a < 1e-12 {
        return c <= 0.0;
    }
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return false;
    }
    let root = discriminant.sqrt();
    let t1 = (-b - root) / (2.0 * a);
    let t2 = (-b + root) / (2.0 * a);
    t2 >= 0.0 && t1 <= 1.0
}

pub fn fleet_speed(ships: i32, max_speed: f64) -> f64 {
    assert!(ships > 0, "fleet speed requires a positive ship count");

    let speed = 1.0 + (max_speed - 1.0) * (f64::from(ships).ln() / 1000.0_f64.ln()).powf(1.5);
    speed.min(max_speed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-12,
            "actual {actual} != expected {expected}"
        );
    }

    #[test]
    fn distance_matches_reference_formula() {
        close(distance(Point::new(1.0, 2.0), Point::new(4.0, 6.0)), 5.0);
    }

    #[test]
    fn point_to_segment_distance_projects_inside_segment() {
        close(
            point_to_segment_distance(
                Point::new(3.0, 4.0),
                Point::new(0.0, 0.0),
                Point::new(6.0, 0.0),
            ),
            4.0,
        );
    }

    #[test]
    fn point_to_segment_distance_clamps_to_endpoint() {
        close(
            point_to_segment_distance(
                Point::new(8.0, 4.0),
                Point::new(0.0, 0.0),
                Point::new(6.0, 0.0),
            ),
            (20.0_f64).sqrt(),
        );
    }

    #[test]
    fn point_to_segment_distance_handles_zero_length_segment() {
        close(
            point_to_segment_distance(
                Point::new(3.0, 4.0),
                Point::new(0.0, 0.0),
                Point::new(0.0, 0.0),
            ),
            5.0,
        );
    }

    #[test]
    fn swept_pair_hit_matches_reference_formula_examples() {
        assert!(swept_pair_hit(
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(5.0, 2.0),
            Point::new(5.0, -2.0),
            1.0,
        ));
        assert!(swept_pair_hit(
            Point::new(0.0, 1.0),
            Point::new(10.0, 1.0),
            Point::new(5.0, 0.0),
            Point::new(5.0, 0.0),
            1.0,
        ));
        assert!(!swept_pair_hit(
            Point::new(0.0, 2.0),
            Point::new(10.0, 2.0),
            Point::new(5.0, 0.0),
            Point::new(5.0, 0.0),
            1.0,
        ));
    }

    #[test]
    fn fleet_speed_matches_python_formula_examples() {
        close(fleet_speed(1, 6.0), 1.0);
        close(fleet_speed(1000, 6.0), 6.0);

        let expected_500 = 1.0 + 5.0 * (500.0_f64.ln() / 1000.0_f64.ln()).powf(1.5);
        close(fleet_speed(500, 6.0), expected_500);
    }

    #[test]
    fn fleet_speed_caps_above_max() {
        close(fleet_speed(10_000, 6.0), 6.0);
    }
}
