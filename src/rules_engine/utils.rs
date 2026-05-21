use std::f64::consts::PI;

use super::state::{
    Planet, Point, StaticTargetArc, BOARD_SIZE, CENTER, ROTATION_RADIUS_LIMIT, SUN_RADIUS,
};

const TARGET_EPS: f64 = 1e-6;
const AVOIDANCE_EPS: f64 = 1e-6;
const ANGLE_EPS: f64 = 1e-9;
const ANGLE_CHOICE_EPS: f64 = 1e-4;
const TAU: f64 = PI * 2.0;

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
    let base_angle = angle_between(source.position(), target.position());
    let feasible = static_target_angle_spans(source, target, static_blockers.clone());
    if let Some(relative_angle) = choose_center_first(&feasible) {
        return Some(normalize_angle(base_angle + relative_angle));
    }
    None
}

pub fn static_target_arcs<'a, I>(
    source: &Planet,
    target: &Planet,
    static_blockers: I,
) -> Vec<StaticTargetArc>
where
    I: Iterator<Item = &'a Planet>,
{
    static_target_angle_spans(source, target, static_blockers)
        .into_iter()
        .map(Into::into)
        .collect()
}

fn static_target_angle_spans<'a, I>(
    source: &Planet,
    target: &Planet,
    static_blockers: I,
) -> Vec<AngleSpan>
where
    I: Iterator<Item = &'a Planet>,
{
    let source_pos = source.position();
    let target_pos = target.position();
    let base_angle = angle_between(source_pos, target_pos);
    let max_ray_distance = distance(source_pos, target_pos) + target.radius;
    let mut feasible = static_target_sun_safe_angle_spans(source, target);
    for blocker in static_blockers {
        let blocker_arc = forbidden_circle_arc(
            source_pos,
            blocker.position(),
            blocker.radius + AVOIDANCE_EPS,
            max_ray_distance,
            base_angle,
        );
        feasible = subtract_spans(&feasible, &blocker_arc);
        if feasible.is_empty() {
            break;
        }
    }
    feasible
}

fn static_target_sun_safe_angle_spans(source: &Planet, target: &Planet) -> Vec<AngleSpan> {
    let source_pos = source.position();
    let target_pos = target.position();
    let base_angle = angle_between(source_pos, target_pos);
    let target_distance = distance(source_pos, target_pos);
    let target_radius = (target.radius - TARGET_EPS).max(0.0);
    let target_half_angle = if target_distance <= target_radius {
        PI
    } else if target_radius > 0.0 {
        (target_radius / target_distance).asin()
    } else {
        0.0
    };
    let target_arc = vec![AngleSpan {
        start: -target_half_angle,
        end: target_half_angle,
    }];
    let max_ray_distance = target_distance + target.radius;

    let sun_arc = forbidden_circle_arc(
        source_pos,
        Point::new(CENTER, CENTER),
        SUN_RADIUS + AVOIDANCE_EPS,
        max_ray_distance,
        base_angle,
    );
    subtract_spans(&target_arc, &sun_arc)
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

#[derive(Clone, Copy, Debug)]
struct AngleSpan {
    start: f64,
    end: f64,
}

impl From<AngleSpan> for StaticTargetArc {
    fn from(span: AngleSpan) -> Self {
        Self {
            start: span.start,
            end: span.end,
        }
    }
}

impl From<StaticTargetArc> for AngleSpan {
    fn from(span: StaticTargetArc) -> Self {
        Self {
            start: span.start,
            end: span.end,
        }
    }
}

fn forbidden_circle_arc(
    source_pos: Point,
    center: Point,
    radius: f64,
    max_ray_distance: f64,
    reference_angle: f64,
) -> Vec<AngleSpan> {
    let center_distance = distance(source_pos, center);
    if center_distance <= radius {
        return vec![AngleSpan {
            start: -PI,
            end: PI,
        }];
    }
    if center_distance - radius > max_ray_distance {
        return Vec::new();
    }
    let half_angle = (radius / center_distance).clamp(-1.0, 1.0).asin();
    arc_to_spans(
        angle_between(source_pos, center),
        half_angle,
        reference_angle,
    )
}

fn arc_to_spans(center: f64, half_angle: f64, reference_angle: f64) -> Vec<AngleSpan> {
    if half_angle >= PI {
        return vec![AngleSpan {
            start: -PI,
            end: PI,
        }];
    }

    let center = angle_delta(center, reference_angle);
    let raw_start = center - half_angle;
    let raw_end = center + half_angle;
    let mut spans = Vec::with_capacity(2);
    for offset in [-TAU, 0.0, TAU] {
        let start = raw_start + offset;
        let end = raw_end + offset;
        let clipped_start = start.max(-PI);
        let clipped_end = end.min(PI);
        if clipped_start <= clipped_end {
            spans.push(AngleSpan {
                start: clipped_start,
                end: clipped_end,
            });
        }
    }
    merge_spans(spans)
}

fn subtract_spans(spans: &[AngleSpan], forbidden: &[AngleSpan]) -> Vec<AngleSpan> {
    let mut current = spans.to_vec();
    for forbidden_span in forbidden {
        let mut next = Vec::with_capacity(current.len() + 1);
        for span in current {
            if forbidden_span.end <= span.start || forbidden_span.start >= span.end {
                next.push(span);
                continue;
            }
            if forbidden_span.start > span.start {
                next.push(AngleSpan {
                    start: span.start,
                    end: forbidden_span.start,
                });
            }
            if forbidden_span.end < span.end {
                next.push(AngleSpan {
                    start: forbidden_span.end,
                    end: span.end,
                });
            }
        }
        current = next;
        if current.is_empty() {
            break;
        }
    }
    current
}

fn merge_spans(mut spans: Vec<AngleSpan>) -> Vec<AngleSpan> {
    spans.sort_by(|left, right| left.start.total_cmp(&right.start));
    let mut merged: Vec<AngleSpan> = Vec::with_capacity(spans.len());
    for span in spans {
        if span.end < span.start {
            continue;
        }
        if let Some(last) = merged.last_mut() {
            if span.start <= last.end + ANGLE_EPS {
                last.end = last.end.max(span.end);
                continue;
            }
        }
        merged.push(span);
    }
    merged
}

fn choose_center_first(spans: &[AngleSpan]) -> Option<f64> {
    spans
        .iter()
        .filter_map(|span| {
            let width = span.end - span.start;
            (width > ANGLE_EPS).then(|| {
                let candidate = 0.0_f64.clamp(span.start, span.end);
                let candidate = if (candidate - span.start).abs() <= ANGLE_EPS {
                    (span.start + ANGLE_CHOICE_EPS).min((span.start + span.end) / 2.0)
                } else if (candidate - span.end).abs() <= ANGLE_EPS {
                    (span.end - ANGLE_CHOICE_EPS).max((span.start + span.end) / 2.0)
                } else {
                    candidate
                };
                (candidate.abs(), candidate)
            })
        })
        .min_by(|left, right| left.0.total_cmp(&right.0))
        .map(|(_, angle)| angle)
}

fn normalize_angle(angle: f64) -> f64 {
    (angle + PI).rem_euclid(TAU) - PI
}

fn angle_delta(angle: f64, reference_angle: f64) -> f64 {
    normalize_angle(angle - reference_angle)
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
