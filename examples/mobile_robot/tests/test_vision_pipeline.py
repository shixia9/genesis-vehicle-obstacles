from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

import numpy as np

from examples.mobile_robot.vision import (
    AnnotatedRgbView,
    Detection,
    DisabledDetector,
    FramePacket,
    GroundTruthDetector,
    ObjectTracker,
    PinholeIntrinsics,
    RgbdCalibration,
    VisionResult,
    GroundingCandidate,
    Owlv2Grounder,
    OpenVocabularyDetector,
    resolve_open_vocab_device,
    YoloWorldGrounder,
    enrich_calibrated_depth,
    dominant_color,
    enrich_approximate_depth,
    sample_aligned_depth,
)
from examples.mobile_robot.room_navigation_observable import (
    CarConfig,
    DifferentialDriveController,
    VISION_ROUTE_WAYPOINTS,
)
from examples.mobile_robot.vision.evaluate_runtime import _match_frame
from examples.mobile_robot.vision.open_vocab_validation import region_evidence


def make_frame(*, frame_id: int = 0, pose=(0.0, 0.0, 0.22, 0.0)) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        sim_time=frame_id * 0.02,
        image=np.zeros((96, 128, 3), dtype=np.uint8),
        camera_pose=pose,
        intrinsics=(128.0, 96.0, 90.0),
    )


class _FakeCv2:
    WINDOW_NORMAL = 0
    COLOR_RGB2BGR = 1
    WND_PROP_VISIBLE = 2
    error = RuntimeError

    def __init__(self) -> None:
        self.named_windows: list[str] = []
        self.frames: list[np.ndarray] = []
        self.destroyed: list[str] = []

    def namedWindow(self, name: str, flags: int) -> None:
        self.named_windows.append(name)

    def cvtColor(self, image: np.ndarray, code: int) -> np.ndarray:
        assert code == self.COLOR_RGB2BGR
        return image[..., ::-1].copy()

    def imshow(self, name: str, image: np.ndarray) -> None:
        assert name in self.named_windows
        self.frames.append(image)

    def waitKey(self, delay: int) -> int:
        assert delay == 1
        return -1

    def getWindowProperty(self, name: str, prop: int) -> float:
        assert name in self.named_windows
        assert prop == self.WND_PROP_VISIBLE
        return 1.0

    def destroyWindow(self, name: str) -> None:
        self.destroyed.append(name)


def test_annotated_view_renders_rgb_overlay_without_reusing_gui_texture() -> None:
    view = AnnotatedRgbView("test annotated view")
    fake_cv2 = _FakeCv2()
    view._cv2 = fake_cv2
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[..., 0] = 255
    result = VisionResult(
        frame_id=1,
        sim_time=0.02,
        model_name="test",
        latency_ms=1.0,
        detections=(Detection(class_id=0, label="car", confidence=0.9, bbox_xyxy=(2, 2, 12, 12)),),
    )

    assert view.show(image, result) is True
    assert fake_cv2.named_windows == ["test annotated view"]
    assert fake_cv2.frames[0].shape == image.shape
    assert tuple(fake_cv2.frames[0][0, 0]) == (0, 0, 255)
    view.close()
    assert fake_cv2.destroyed == ["test annotated view"]


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

    # Genesis's attached camera can render the same material much darker than
    # the nominal RGB swatch; color classification must remain illumination
    # tolerant for the "yellow car" target.
    dark_yellow = np.zeros((20, 20, 3), dtype=np.uint8)
    dark_yellow[5:15, 5:15] = (80, 70, 13)
    color, confidence = dominant_color(dark_yellow, (5, 5, 15, 15))
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


def test_calibrated_rgbd_mapping_uses_genesis_intrinsics() -> None:
    calibration = RgbdCalibration(
        rgb=PinholeIntrinsics(256, 192, 96.0, 96.0, 128.0, 96.0),
        depth=PinholeIntrinsics(128, 96, 64.0, 64.0, 64.0, 48.0),
        depth_from_rgb=np.eye(4),
        robot_from_depth=np.array(
            [[1.0, 0.0, 0.0, 0.6], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.23], [0, 0, 0, 1]],
            dtype=np.float64,
        ),
    )
    mapped = calibration.project_rgb_pixels_to_depth(np.array([[128.0, 96.0], [32.0, 24.0]]))
    assert np.allclose(mapped[0], (64.0, 48.0))
    assert np.allclose(mapped[1], (0.0, 0.0))

    depth = np.full((96, 128), 2.0, dtype=np.float32)
    distance, spread, count = sample_aligned_depth(depth, (96.0, 72.0, 160.0, 120.0), calibration)
    assert distance == 2.0
    assert spread == 0.0
    assert count > 0

    result = VisionResult(
        frame_id=1,
        sim_time=0.02,
        model_name="test",
        latency_ms=1.0,
        detections=(Detection(class_id=0, label="car", confidence=0.9, bbox_xyxy=(96, 72, 160, 120)),),
    )
    fused = enrich_calibrated_depth(result, depth, calibration, robot_pose=(0.0, 0.0, 0.0, 0.0))
    detection = fused.detections[0]
    assert detection.distance_m == 2.0
    assert detection.distance_confidence == 1.0
    assert detection.position_robot is not None
    assert np.allclose(detection.position_robot, (2.6, 0.0, 0.23))
    assert np.allclose(detection.position_world, (2.6, 0.0, 0.23))

    camera_pose = calibration.camera_pose_from_robot_pose((1.0, 2.0, 0.2, 0.0))
    assert np.allclose(camera_pose, (1.6, 2.0, 0.43, 0.0))


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


def test_runtime_evaluator_can_match_by_world_position() -> None:
    prediction = {
        "label": "car",
        "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
        "position_world": [1.2, 2.0, 0.2],
    }
    truth = {
        "label": "car",
        "bbox_xyxy": [20.0, 20.0, 30.0, 30.0],
        "position_world": [1.0, 2.0, 0.2],
    }
    matches, unmatched_predictions, unmatched_truth = _match_frame(
        [prediction], [truth], iou_threshold=0.5, match_mode="position", position_threshold_m=0.5
    )
    assert len(matches) == 1
    assert unmatched_predictions == []
    assert unmatched_truth == []


def test_open_vocab_device_prefers_available_backend_or_cpu_fallback() -> None:
    device, reason = resolve_open_vocab_device("auto")
    assert device in {"mps", "cpu"}
    assert reason
    cpu_device, cpu_reason = resolve_open_vocab_device("cpu")
    assert cpu_device == "cpu"
    assert cpu_reason == "requested_cpu"
    with pytest.raises(ValueError):
        resolve_open_vocab_device("tpu")


def test_open_vocab_candidate_is_json_safe_and_preserves_prompt() -> None:
    candidate = GroundingCandidate(
        prompt="a green pillar",
        prompt_index=0,
        confidence=0.73,
        bbox_xyxy=(1, 2, 30, 40),
        frame_id=4,
        sim_time=0.08,
        model_name="yolov8s-world.pt",
        latency_ms=12.5,
    )
    data = candidate.to_dict()
    assert data["prompt"] == "a green pillar"
    assert data["bbox_xyxy"] == [1.0, 2.0, 30.0, 40.0]
    with pytest.raises(ValueError):
        GroundingCandidate(
            prompt="bad",
            prompt_index=0,
            confidence=1.1,
            bbox_xyxy=(0, 0, 1, 1),
            frame_id=0,
            sim_time=0.0,
            model_name="test",
            latency_ms=0.0,
        )


def test_open_vocab_roi_evidence_is_truth_free_and_prompt_conditioned() -> None:
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[10:60, 20:40] = (235, 220, 20)  # yellow, tall ROI
    yellow = region_evidence(image, (20, 10, 40, 60), "yellow pillar")
    blue = region_evidence(image, (20, 10, 40, 60), "blue pillar")
    assert yellow.colour_score > blue.colour_score
    assert yellow.shape_score > 0.5
    with pytest.raises(ValueError):
        region_evidence(np.zeros((10, 10), dtype=np.uint8), (0, 0, 2, 2), "yellow object")


def test_open_vocab_detector_maps_candidates_without_touching_control() -> None:
    class _FakeGrounder:
        model_name = "fake-open-vocab"
        device = "cpu"
        device_reason = "test"
        last_latency_ms = 3.5

        def ground(self, frame, prompts):
            return (
                GroundingCandidate(
                    prompt="an unknown platform",
                    prompt_index=0,
                    confidence=0.08,
                    bbox_xyxy=(4, 5, 30, 40),
                    frame_id=frame.frame_id,
                    sim_time=frame.sim_time,
                    model_name=self.model_name,
                    latency_ms=self.last_latency_ms,
                ),
            )

    detector = OpenVocabularyDetector.__new__(OpenVocabularyDetector)
    detector.grounder = _FakeGrounder()
    detector.prompts = ("an unknown platform",)
    detector.decision_confidence = 0.05
    result = detector.detect(make_frame(frame_id=8))
    assert result.status == "candidate"
    assert result.available is True
    assert result.detections[0].label == "an unknown platform"
    assert result.detections[0].confidence == 0.08


def test_open_vocab_grounder_rejects_missing_local_weight_before_model_load(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        YoloWorldGrounder(tmp_path / "missing.pt", device="cpu")
    with pytest.raises(FileNotFoundError, match="directory does not exist"):
        Owlv2Grounder(tmp_path / "missing_owlv2", device="cpu")
