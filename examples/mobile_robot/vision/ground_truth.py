"""Deterministic semantic detector for simulator bring-up and evaluation.

This adapter projects authored scene metadata into approximate camera boxes. It
is intentionally explicit about being ground truth: it is useful for testing
the runtime, overlay, tracker, and logging paths before a local model weight is
available, but it must not be presented as a YOLO result.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from .types import Detection, FramePacket, VisionResult


class GroundTruthDetector:
    """Project static semantic objects into the robot camera view."""

    def __init__(self, objects: Iterable[Any], *, fov_degrees: float = 90.0):
        self.objects = tuple(objects)
        self.fov_degrees = float(fov_degrees)
        if not 0.0 < self.fov_degrees < 180.0:
            raise ValueError("fov_degrees must be in (0, 180)")

    def detect(self, frame: FramePacket) -> VisionResult:
        if frame.camera_pose is None:
            return VisionResult(
                frame_id=frame.frame_id,
                sim_time=frame.sim_time,
                model_name="genesis_ground_truth",
                latency_ms=0.0,
                detections=(),
                status="missing_camera_pose",
                available=False,
            )

        camera_pose = tuple(frame.camera_pose)
        if len(camera_pose) < 4:
            raise ValueError("camera_pose must contain [x, y, z, yaw]")
        robot_x, robot_y, robot_z, yaw = camera_pose[:4]
        width, height = frame.width, frame.height
        if frame.intrinsics is not None and len(frame.intrinsics) >= 4:
            fx, fy, cx, cy = (float(value) for value in frame.intrinsics[:4])
            horizontal_half_fov = math.atan2(cx, fx)
        else:
            fx = fy = width / (2.0 * math.tan(math.radians(self.fov_degrees) / 2.0))
            cx, cy = width / 2.0, height * 0.58
            horizontal_half_fov = math.radians(self.fov_degrees) / 2.0
        detections: list[Detection] = []

        for class_id, obj in enumerate(self.objects):
            object_x, object_y, object_z = tuple(float(value) for value in obj.position)
            delta_x = object_x - float(robot_x)
            delta_y = object_y - float(robot_y)
            forward = math.cos(float(yaw)) * delta_x + math.sin(float(yaw)) * delta_y
            lateral = -math.sin(float(yaw)) * delta_x + math.cos(float(yaw)) * delta_y
            if forward <= 0.15 or abs(math.atan2(lateral, forward)) > horizontal_half_fov:
                continue

            size_x, size_y, size_z = (float(value) for value in obj.size)
            projected_width = max(4.0, fx * max(size_x, size_y) / forward)
            projected_height = max(4.0, fy * size_z / forward)
            # Genesis' camera image points right along negative robot-lateral
            # and down along negative robot-up.
            center_x = cx - fx * lateral / forward
            center_y = cy - fy * (object_z - float(robot_z)) / forward
            x1 = max(0.0, center_x - projected_width / 2.0)
            y1 = max(0.0, center_y - projected_height / 2.0)
            x2 = min(float(width - 1), center_x + projected_width / 2.0)
            y2 = min(float(height - 1), center_y + projected_height / 2.0)
            if x2 <= x1 or y2 <= y1:
                continue

            distance = math.hypot(delta_x, delta_y)
            detections.append(
                Detection(
                    class_id=class_id,
                    label=str(obj.category),
                    confidence=1.0,
                    bbox_xyxy=(x1, y1, x2, y2),
                    color=str(obj.color),
                    color_confidence=1.0,
                    shape=str(obj.shape),
                    distance_m=distance,
                    position_camera=(forward, lateral, object_z - float(robot_z)),
                    position_robot=(forward, lateral, object_z - float(robot_z)),
                    position_world=(object_x, object_y, object_z),
                )
            )

        return VisionResult(
            frame_id=frame.frame_id,
            sim_time=frame.sim_time,
            model_name="genesis_ground_truth",
            latency_ms=0.0,
            detections=tuple(detections),
            status="ground_truth",
            available=True,
        )
