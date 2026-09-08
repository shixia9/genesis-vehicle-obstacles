"""Exact pinhole RGB-D calibration for the Genesis showcase cameras.

Genesis uses two different camera APIs in this example: the RGB camera is a
rendered pinhole camera whose ``fov`` is vertical, while ``DepthCameraPattern``
accepts a horizontal FOV and returns ray distances.  This module keeps those
conventions explicit and maps RGB pixels into the depth camera's canonical
robotics frame ``(forward, lateral-left, up)``.

The current scene mounts both sensors at the same optical centre and with the
same orientation.  A calibrated affine pixel map is therefore exact (up to
pixel-centre quantisation), unlike the previous resolution-ratio heuristic.
The calibration record still stores the relative extrinsic matrix and rejects
non-co-located cameras rather than silently applying an invalid approximation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np

from .types import Detection, VisionResult


@dataclass(frozen=True)
class PinholeIntrinsics:
    """Pinhole intrinsics in pixel units for one image stream."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera resolution must be positive")
        for name in ("fx", "fy", "cx", "cy"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0 and name in {"fx", "fy"}:
                raise ValueError(f"invalid camera intrinsic {name}={value!r}")

    @classmethod
    def from_matrix(cls, matrix: Any, *, width: int, height: int) -> "PinholeIntrinsics":
        values = np.asarray(matrix, dtype=np.float64)
        if values.shape != (3, 3):
            raise ValueError(f"intrinsic matrix must be 3x3, got {values.shape}")
        return cls(
            width=int(width),
            height=int(height),
            fx=float(values[0, 0]),
            fy=float(values[1, 1]),
            cx=float(values[0, 2]),
            cy=float(values[1, 2]),
        )

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "matrix": self.matrix.tolist(),
        }

    def pixel_to_robotics_ray(self, pixels: np.ndarray) -> np.ndarray:
        """Convert ``(..., 2)`` image pixels to unit ``(forward,left,up)`` rays."""

        uv = np.asarray(pixels, dtype=np.float64)
        if uv.shape[-1] != 2:
            raise ValueError("pixels must have a final (u, v) dimension")
        # Genesis' DepthCameraPattern uses x_c=(u-cx)/fx,
        # y_c=(v-cy)/fy, dirs=(z_c,-x_c,-y_c).
        lateral = -(uv[..., 0] - self.cx) / self.fx
        up = -(uv[..., 1] - self.cy) / self.fy
        rays = np.stack((np.ones_like(lateral), lateral, up), axis=-1)
        norm = np.linalg.norm(rays, axis=-1, keepdims=True)
        return rays / np.maximum(norm, np.finfo(np.float64).eps)

    def robotics_ray_to_pixel(self, rays: np.ndarray) -> np.ndarray:
        """Project ``(..., 3)`` robotics-frame rays into image pixels."""

        values = np.asarray(rays, dtype=np.float64)
        if values.shape[-1] != 3:
            raise ValueError("rays must have a final (forward, lateral, up) dimension")
        forward = values[..., 0]
        if np.any(forward <= 0.0):
            raise ValueError("cannot project rays with non-positive forward component")
        u = self.cx - self.fx * values[..., 1] / forward
        v = self.cy - self.fy * values[..., 2] / forward
        return np.stack((u, v), axis=-1)


def _homogeneous(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


@dataclass(frozen=True)
class RgbdCalibration:
    """RGB/depth intrinsics and relative extrinsics."""

    rgb: PinholeIntrinsics
    depth: PinholeIntrinsics
    depth_from_rgb: np.ndarray
    robot_from_depth: np.ndarray
    source: str = "genesis_known_intrinsics_extrinsics"

    def __post_init__(self) -> None:
        for name in ("depth_from_rgb", "robot_from_depth"):
            matrix = np.asarray(getattr(self, name), dtype=np.float64)
            if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                raise ValueError(f"{name} must be a finite 4x4 matrix")
            object.__setattr__(self, name, matrix)

    @property
    def co_located(self) -> bool:
        transform = self.depth_from_rgb
        return bool(np.linalg.norm(transform[:3, 3]) <= 1e-5)

    def project_rgb_pixels_to_depth(self, pixels: np.ndarray) -> np.ndarray:
        """Project RGB pixels to depth pixels using the calibrated relative pose.

        The current exact path requires a shared optical centre.  If future
        sensors are translated relative to one another, depth-dependent
        reprojection must be implemented instead of silently using this map.
        """

        if not self.co_located:
            raise ValueError("RGB/depth optical centres are translated; depth-dependent reprojection is required")
        rgb_rays = self.rgb.pixel_to_robotics_ray(pixels)
        rotation = self.depth_from_rgb[:3, :3]
        depth_rays = np.einsum("ij,...j->...i", rotation, rgb_rays)
        return self.depth.robotics_ray_to_pixel(depth_rays)

    def depth_pixel_to_robot_point(self, pixel: tuple[float, float], distance_m: float) -> np.ndarray:
        ray = self.depth.pixel_to_robotics_ray(np.asarray(pixel, dtype=np.float64))
        point_depth = ray * float(distance_m)
        point_h = np.concatenate((point_depth, [1.0]))
        return (self.robot_from_depth @ point_h)[:3]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "rgb": self.rgb.to_dict(),
            "depth": self.depth.to_dict(),
            "depth_from_rgb": self.depth_from_rgb.tolist(),
            "robot_from_depth": self.robot_from_depth.tolist(),
            "co_located": self.co_located,
        }


def build_genesis_rgbd_calibration(rgb_camera: Any, depth_camera: Any, body: Any | None = None) -> RgbdCalibration:
    """Read Genesis' actual camera models and relative mount transforms."""

    rgb_model = PinholeIntrinsics.from_matrix(
        rgb_camera.intrinsics,
        width=int(rgb_camera.res[0]),
        height=int(rgb_camera.res[1]),
    )
    pattern = depth_camera._options.pattern
    depth_model = PinholeIntrinsics(
        width=int(pattern.width),
        height=int(pattern.height),
        fx=float(pattern.fx),
        fy=float(pattern.fy),
        cx=float(pattern.cx),
        cy=float(pattern.cy),
    )

    # The DepthCamera pattern is expressed in the robot's canonical
    # (forward, lateral-left, up) frame.  Its options are the known sensor
    # extrinsics relative to the base link.
    try:
        from genesis.utils.geom import euler_to_R

        depth_rotation = np.asarray(euler_to_R(np.asarray(depth_camera._options.euler_offset)), dtype=np.float64)
    except Exception:
        depth_rotation = np.eye(3, dtype=np.float64)
    robot_from_depth = _homogeneous(
        depth_rotation,
        np.asarray(depth_camera._options.pos_offset, dtype=np.float64),
    )

    # Genesis rendered Camera coordinates are (right, up, backward). Convert
    # them to the same canonical robotics camera frame (forward, left, up).
    rgb_transform = np.asarray(rgb_camera.transform, dtype=np.float64)
    rgb_from_canonical = np.array(
        [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    world_from_rgb_canonical = np.eye(4, dtype=np.float64)
    world_from_rgb_canonical[:3, :3] = rgb_transform[:3, :3] @ rgb_from_canonical
    world_from_rgb_canonical[:3, 3] = rgb_transform[:3, 3]

    if body is not None:
        try:
            from genesis.utils.geom import quat_to_R

            body_rotation = np.asarray(quat_to_R(np.asarray(body.get_quat()).reshape(4)), dtype=np.float64)
            body_position = np.asarray(body.get_pos(), dtype=np.float64).reshape(3)
            world_from_robot = _homogeneous(body_rotation, body_position)
            world_from_depth = world_from_robot @ robot_from_depth
            depth_from_rgb = np.linalg.inv(world_from_depth) @ world_from_rgb_canonical
        except Exception:
            depth_from_rgb = np.eye(4, dtype=np.float64)
    else:
        depth_from_rgb = np.eye(4, dtype=np.float64)

    return RgbdCalibration(
        rgb=rgb_model,
        depth=depth_model,
        depth_from_rgb=depth_from_rgb,
        robot_from_depth=robot_from_depth,
    )


def _depth_roi_from_bbox(
    calibration: RgbdCalibration,
    bbox_xyxy: tuple[float, float, float, float],
    *,
    inner_fraction: float = 0.2,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox_xyxy
    if x2 <= x1 or y2 <= y1:
        return None
    corners = np.array(((x1, y1), (x2, y1), (x1, y2), (x2, y2)), dtype=np.float64)
    projected = calibration.project_rgb_pixels_to_depth(corners)
    left = float(np.min(projected[:, 0]))
    right = float(np.max(projected[:, 0]))
    top = float(np.min(projected[:, 1]))
    bottom = float(np.max(projected[:, 1]))
    width = right - left
    height = bottom - top
    left += inner_fraction * width
    right -= inner_fraction * width
    top += inner_fraction * height
    bottom -= inner_fraction * height
    left_i = max(0, min(calibration.depth.width - 1, int(math.floor(left))))
    right_i = max(left_i + 1, min(calibration.depth.width, int(math.ceil(right)) + 1))
    top_i = max(0, min(calibration.depth.height - 1, int(math.floor(top))))
    bottom_i = max(top_i + 1, min(calibration.depth.height, int(math.ceil(bottom)) + 1))
    if right_i <= left_i or bottom_i <= top_i:
        return None
    return left_i, top_i, right_i, bottom_i


def sample_aligned_depth(
    depth: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    calibration: RgbdCalibration,
    *,
    max_range_m: float = 6.0,
    inner_fraction: float = 0.2,
) -> tuple[float | None, float | None, int]:
    """Return median ray distance, MAD spread, and valid sample count."""

    values = np.asarray(depth, dtype=np.float32)
    if values.shape != (calibration.depth.height, calibration.depth.width):
        raise ValueError(
            f"depth shape {values.shape} does not match calibrated resolution "
            f"{(calibration.depth.height, calibration.depth.width)}"
        )
    roi = _depth_roi_from_bbox(calibration, bbox_xyxy, inner_fraction=inner_fraction)
    if roi is None:
        return None, None, 0
    left, top, right, bottom = roi
    samples = values[top:bottom, left:right]
    valid = samples[np.isfinite(samples) & (samples > 0.05) & (samples <= max_range_m)]
    if valid.size == 0:
        return None, None, 0
    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    return median, mad, int(valid.size)


def enrich_calibrated_depth(
    result: VisionResult,
    depth: np.ndarray,
    calibration: RgbdCalibration,
    *,
    robot_pose: tuple[float, ...] | np.ndarray | None = None,
    max_range_m: float = 6.0,
) -> VisionResult:
    """Attach exact aligned depth and robot/world coordinates to detections."""

    detections: list[Detection] = []
    for detection in result.detections:
        if detection.distance_m is not None:
            detections.append(detection)
            continue
        distance, mad, valid_count = sample_aligned_depth(
            depth,
            detection.bbox_xyxy,
            calibration,
            max_range_m=max_range_m,
        )
        if distance is None:
            detections.append(
                replace(
                    detection,
                    distance_confidence=0.0,
                    depth_valid_pixel_count=valid_count,
                    depth_spread_m=None,
                )
            )
            continue

        center_rgb = np.asarray(
            ((detection.bbox_xyxy[0] + detection.bbox_xyxy[2]) / 2.0,
             (detection.bbox_xyxy[1] + detection.bbox_xyxy[3]) / 2.0),
            dtype=np.float64,
        )
        center_depth = calibration.project_rgb_pixels_to_depth(center_rgb)
        center_depth_pixel = (
            float(np.clip(center_depth[0], 0.0, calibration.depth.width - 1.0)),
            float(np.clip(center_depth[1], 0.0, calibration.depth.height - 1.0)),
        )
        point_robot = calibration.depth_pixel_to_robot_point(center_depth_pixel, distance)
        point_depth_h = np.linalg.inv(calibration.robot_from_depth) @ np.concatenate((point_robot, [1.0]))
        point_camera = point_depth_h[:3]
        point_world = None
        if robot_pose is not None:
            pose = np.asarray(robot_pose, dtype=np.float64).reshape(-1)
            if pose.size < 4:
                raise ValueError("robot_pose must contain x, y, z, yaw")
            cos_yaw = math.cos(float(pose[3]))
            sin_yaw = math.sin(float(pose[3]))
            point_world = np.array(
                [
                    pose[0] + cos_yaw * point_robot[0] - sin_yaw * point_robot[1],
                    pose[1] + sin_yaw * point_robot[0] + cos_yaw * point_robot[1],
                    pose[2] + point_robot[2],
                ],
                dtype=np.float64,
            )
        confidence = float(min(1.0, valid_count / 20.0) * math.exp(-mad / 0.20))
        detections.append(
            replace(
                detection,
                distance_m=distance,
                distance_confidence=confidence,
                depth_valid_pixel_count=valid_count,
                depth_spread_m=mad,
                position_camera=tuple(float(value) for value in point_camera),
                position_robot=tuple(float(value) for value in point_robot),
                position_world=None if point_world is None else tuple(float(value) for value in point_world),
            )
        )
    return replace(result, detections=tuple(detections))
