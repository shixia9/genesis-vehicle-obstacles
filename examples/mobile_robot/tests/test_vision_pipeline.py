from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from examples.mobile_robot.vision import (
    Detection,
    DisabledDetector,
    FramePacket,
    GroundTruthDetector,
    ObjectTracker,
    VisionResult,
    dominant_color,
    enrich_approximate_depth,
)
from examples.mobile_robot.room_navigation_observable import (
    CarConfig,
    DifferentialDriveController,
    VISION_ROUTE_WAYPOINTS,
)


def make_frame(*, frame_id: int = 0, pose=(0.0, 0.0, 0.22, 0.0)) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        sim_time=frame_id * 0.02,
        image=np.zeros((96, 128, 3), dtype=np.uint8),
        camera_pose=pose,
        intrinsics=(128.0, 96.0, 90.0),
    )


def test_frame_and_disabled_detector_are_explicit() -> None:
    frame = make_frame()
    result = DisabledDetector().detect(frame)
    assert frame.to_metadata()["encoding"] == "rgb8"
    assert result.available is False
    assert result.status == "disabled"
    assert result.frame_id == frame.frame_id


def test_ground_truth_detector_projects_only_forward_objects() -> None:
    objects = (
        SimpleNamespace(
            category="car",
            color="yellow",
            shape="vehicle",
            position=(2.0, 0.0, 0.3),
            size=(0.8, 0.5, 0.4),
        ),
        SimpleNamespace(
            category="box",
            color="red",
            shape="box",
            position=(-2.0, 0.0, 0.3),
            size=(0.6, 0.6, 0.6),
        ),
    )
    result = GroundTruthDetector(objects).detect(make_frame())
    assert result.status == "ground_truth"
    assert [d.label for d in result.detections] == ["car"]
    assert result.detections[0].distance_m == 2.0


def test_tracker_keeps_id_for_same_world_object() -> None:
    tracker = ObjectTracker()
    first = VisionResult(
        frame_id=1,
        sim_time=0.02,
        model_name="test",
        latency_ms=1.0,
        detections=(
            Detection(
                class_id=0,
                label="car",
                confidence=0.9,
                bbox_xyxy=(20, 20, 40, 40),
                color="yellow",
                position_world=(1.0, 2.0, 0.2),
            ),
        ),
    )
    second = VisionResult(
        frame_id=2,
        sim_time=0.04,
        model_name="test",
        latency_ms=1.0,
        detections=(
            Detection(
                class_id=0,
                label="car",
                confidence=0.92,
                bbox_xyxy=(21, 20, 41, 40),
                color="yellow",
                position_world=(1.1, 2.0, 0.2),
            ),
        ),
    )
    first_tracked = tracker.update(first)
    second_tracked = tracker.update(second)
    assert first_tracked.detections[0].track_id == "track-001"
    assert second_tracked.detections[0].track_id == "track-001"
    assert tracker.summaries()[0]["visible_frame_count"] == 2


def test_color_and_depth_helpers_are_conservative_and_json_safe() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[5:15, 5:15] = (240, 220, 20)
    color, confidence = dominant_color(image, (5, 5, 15, 15))
    assert color == "yellow"
    assert confidence > 0.9

    result = VisionResult(
        frame_id=0,
        sim_time=0.0,
        model_name="test",
        latency_ms=0.0,
        detections=(Detection(class_id=0, label="car", confidence=0.9, bbox_xyxy=(5, 5, 15, 15)),),
    )
    depth = np.full((10, 10), 2.0, dtype=np.float32)
    fused = enrich_approximate_depth(result, depth, rgb_shape=(20, 20))
    assert fused.detections[0].distance_m == 2.0


def test_waypoint_controller_keeps_lidar_safety_priority() -> None:
    controller = DifferentialDriveController(CarConfig(), VISION_ROUTE_WAYPOINTS)
    clear = controller.command(
        np.asarray((-2.8, -1.8), dtype=np.float32),
        0.0,
        np.full(72, 6.0, dtype=np.float32),
    )
    blocked = controller.command(
        np.asarray((-2.8, -1.8), dtype=np.float32),
        0.0,
        np.concatenate((np.full(36, 0.3, dtype=np.float32), np.full(36, 6.0, dtype=np.float32))),
    )
    assert np.isfinite(clear[0]) and np.isfinite(clear[1])
    assert blocked[0] == 0.0
    assert abs(blocked[1]) == CarConfig().max_angular_speed
