"""Generate an evaluation-only Genesis OOD set for open-vocabulary grounding.

This command does **not** create a YOLO class list and does not train a
detector.  It renders RGB/depth/instance masks and writes scene truth plus
natural-language prompt manifests so a text-conditioned model can be scored
on objects, materials, distractors and missing-target cases that are kept
outside the existing closed-set training data.

Example (small smoke set)::

    .venv/bin/python examples/mobile_robot/vision/generate_open_vocab_ood.py \
      --layouts 1 --steps 10 --sample-every 5 \
      --output-dir out/mobile_robot_open_vocab_ood_smoke

The full set should use several layouts and be generated on a host with a
working Genesis OpenGL context.  Splits are layout-level, never adjacent-frame
level: ``dev``, ``negative_test`` and ``ood_test``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

import genesis as gs

if __package__:
    from ..room_navigation_observable import (
        CarConfig,
        DifferentialDriveController,
        INITIAL_POSITION,
        Odometry,
        RoomConfig,
        SemanticObjectSpec,
        build_scene,
        geometry_collision,
        read_sensor_observation,
        waypoints_for_scenario,
    )
    from .rgbd_calibration import build_genesis_rgbd_calibration
else:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from room_navigation_observable import (  # type: ignore[no-redef]
        CarConfig,
        DifferentialDriveController,
        INITIAL_POSITION,
        Odometry,
        RoomConfig,
        SemanticObjectSpec,
        build_scene,
        geometry_collision,
        read_sensor_observation,
        waypoints_for_scenario,
    )
    from vision.rgbd_calibration import build_genesis_rgbd_calibration  # type: ignore[no-redef]


SCENARIO = "vision_route_showcase"
SPLITS = ("dev", "negative_test", "ood_test")


@dataclass(frozen=True)
class OodObject:
    """Authored object metadata used only by the dataset generator/evaluator."""

    object_id: str
    category: str
    color: str
    shape: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    asset_variant: str

    def to_spec(self) -> Any:
        return SemanticObjectSpec(
            object_id=self.object_id,
            category=self.category,
            color=self.color,
            shape=self.shape,
            position=self.position,
            size=self.size,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "category": self.category,
            "color": self.color,
            "shape": self.shape,
            "position": list(self.position),
            "size": list(self.size),
            "asset_variant": self.asset_variant,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", type=int, default=12)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--sample-every", type=int, default=25)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--gpu", action="store_true", help="Use Genesis GPU backend when available.")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/mobile_robot_open_vocab_ood"))
    return parser.parse_args()


def _split_for_layout(layout_index: int, total_layouts: int) -> str:
    """Assign whole layouts to splits so adjacent frames never cross splits."""

    if total_layouts <= 1:
        return "ood_test"
    dev_count = max(1, total_layouts // 4)
    negative_count = max(1, total_layouts // 4)
    if layout_index < dev_count:
        return "dev"
    if layout_index < dev_count + negative_count:
        return "negative_test"
    return "ood_test"


def _objects_for_layout(layout_index: int, seed: int) -> tuple[OodObject, ...]:
    """Create varied assets and compositional distractors in safe corridors."""

    rng = np.random.default_rng(seed + layout_index * 1009)
    variant = layout_index % 4
    pillar_size = ((0.38, 0.38, 0.90), (0.52, 0.52, 1.15), (0.44, 0.44, 0.72), (0.62, 0.62, 1.30))[variant]
    platform_size = ((0.90, 0.70, 0.18), (1.20, 0.85, 0.22), (0.72, 0.72, 0.30), (1.40, 0.60, 0.16))[variant]

    def jitter(center: tuple[float, float], radius: tuple[float, float] = (0.12, 0.12)) -> tuple[float, float]:
        return (
            float(center[0] + rng.uniform(-radius[0], radius[0])),
            float(center[1] + rng.uniform(-radius[1], radius[1])),
        )

    pillar_a = jitter((-1.65, 0.35), (0.18, 0.20))
    # Keep the second pillar away from the final x=2.8 vertical corridor.
    pillar_b = jitter((1.70, 0.45), (0.18, 0.20))
    platform = jitter((0.65, 2.05), (0.18, 0.12))
    unknown = jitter((1.55, -1.45), (0.18, 0.12))
    # Keep the reference car outside the first vertical corridor; otherwise
    # the conservative robot footprint check can classify the route as a
    # collision while the car is still visually useful from later viewpoints.
    yellow_car = jitter((-1.45, -1.10), (0.12, 0.10))
    return (
        OodObject("green_pillar_asset_a", "pillar", "green", "cylinder", (*pillar_a, pillar_size[2] / 2), pillar_size, f"pillar_v{variant}_a"),
        OodObject("green_pillar_asset_b", "pillar", "green", "cylinder", (*pillar_b, pillar_size[2] / 2), pillar_size, f"pillar_v{variant}_b"),
        OodObject("orange_platform_asset", "platform", "orange", "box", (*platform, platform_size[2] / 2), platform_size, f"platform_v{variant}"),
        OodObject("purple_unknown_asset", "sculpture", "purple", "box", (*unknown, 0.30), (0.58, 0.42, 0.60), f"sculpture_v{variant}"),
        OodObject("yellow_car_reference", "car", "yellow", "vehicle", (*yellow_car, 0.22), (0.80, 0.52, 0.38), f"car_reference_v{variant}"),
    )


def _entity_idx(value: Any) -> int | None:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, tuple) and value:
        return _entity_idx(value[0])
    if hasattr(value, "idx"):
        return int(value.idx)
    return None


def _segmentation_map(scene: Any, registry: tuple[Any, ...]) -> dict[int, Any]:
    by_entity = {int(entity.idx): spec for spec, entity in registry}
    result: dict[int, Any] = {}
    for seg_id, value in scene.segmentation_idx_dict.items():
        entity_idx = _entity_idx(value)
        if entity_idx in by_entity:
            result[int(seg_id)] = by_entity[entity_idx]
    if len(result) != len(registry):
        raise RuntimeError(f"segmentation mapped {len(result)} objects, expected {len(registry)}")
    return result


def _segmentation_values(segmentation: np.ndarray, height: int, width: int) -> np.ndarray:
    values = np.asarray(segmentation)
    while values.ndim > 2:
        values = values[..., 0]
    if values.shape != (height, width):
        raise ValueError(f"unexpected segmentation shape {values.shape}; expected {(height, width)}")
    return values


def _truth_from_segmentation(
    segmentation: np.ndarray,
    seg_to_spec: dict[int, Any],
    *,
    width: int,
    height: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    values = _segmentation_values(segmentation, height, width)
    instance_mask = np.zeros((height, width), dtype=np.uint16)
    records: list[dict[str, Any]] = []
    for instance_id, (seg_id, spec) in enumerate(sorted(seg_to_spec.items()), start=1):
        ys, xs = np.where(values == seg_id)
        visible = int(xs.size)
        record = {
            "object_id": str(spec.object_id),
            "category": str(spec.category),
            "color": str(spec.color),
            "shape": str(spec.shape),
            "position_world": [float(value) for value in spec.position],
            "size": [float(value) for value in spec.size],
            "segmentation_id": int(seg_id),
            "instance_id": instance_id,
            "visible_pixel_count": visible,
            "visible": visible > 0,
            "bbox_xyxy": None,
            "truncated": False,
        }
        if visible:
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            record["bbox_xyxy"] = [x1, y1, x2, y2]
            record["truncated"] = bool(x1 == 0 or y1 == 0 or x2 == width or y2 == height)
            instance_mask[values == seg_id] = instance_id
        records.append(record)
    return records, instance_mask


def _prompts_for_objects(objects: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    by_object = {item["object_id"]: item for item in objects}
    by_category: dict[str, list[str]] = {}
    for item in by_object.values():
        by_category.setdefault(str(item["category"]), []).append(str(item["object_id"]))
    prompts: list[dict[str, Any]] = []
    for category, ids in sorted(by_category.items()):
        colors = sorted({str(by_object[item_id]["color"]) for item_id in ids})
        color = colors[0] if colors else ""
        prompt = f"{color} {category}".strip()
        visible_ids = [item_id for item_id in ids if by_object[item_id]["visible"]]
        prompts.append(
            {
                "prompt": prompt,
                "kind": "ambiguous" if len(visible_ids) > 1 else "positive" if visible_ids else "out_of_view",
                "target_object_ids": visible_ids,
                "target_exists_in_scene": True,
                "target_visible": bool(visible_ids),
            }
        )
    # These prompts are intentionally absent from the scene.  They measure
    # false positives and must never become fixed detector classes.
    prompts.extend(
        {
            "prompt": prompt,
            "kind": "negative",
            "target_object_ids": [],
            "target_exists_in_scene": False,
            "target_visible": False,
        }
        for prompt in ("striped statue", "glass arch", "red traffic cone")
    )
    return prompts


def _prepare_dirs(root: Path) -> None:
    for split in SPLITS:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "depth" / split).mkdir(parents=True, exist_ok=True)
        (root / "masks" / split).mkdir(parents=True, exist_ok=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.layouts <= 0 or args.steps <= 0 or args.sample_every <= 0 or args.dt <= 0.0:
        raise ValueError("layouts, steps, sample-every and dt must be positive")
    root = args.output_dir
    _prepare_dirs(root)
    metadata_path = root / "metadata.jsonl"
    prompt_path = root / "prompts.jsonl"
    total_images = 0
    split_counts = {split: 0 for split in SPLITS}
    object_counts: dict[str, int] = {}
    calibration: dict[str, Any] | None = None
    layout_manifest: list[dict[str, Any]] = []
    with metadata_path.open("w", encoding="utf-8") as metadata_file, prompt_path.open("w", encoding="utf-8") as prompt_file:
        for layout_index in range(args.layouts):
            split = _split_for_layout(layout_index, args.layouts)
            layout_seed = args.seed + layout_index * 1009
            authored = _objects_for_layout(layout_index, args.seed)
            specs = tuple(item.to_spec() for item in authored)
            scene_args = SimpleNamespace(
                gpu=args.gpu,
                dt=args.dt,
                seed=layout_seed,
                vis=False,
                robot_view=False,
                scenario=SCENARIO,
                segmentation_level="entity",
                return_semantic_registry=True,
                semantic_specs_override=specs,
            )
            try:
                built = build_scene(scene_args)
                (
                    scene,
                    car,
                    lidar,
                    depth_camera,
                    imu,
                    _overview_camera,
                    robot_rgb_camera,
                    obstacle_specs,
                    registry,
                ) = built
                if calibration is None:
                    calibration = build_genesis_rgbd_calibration(
                        robot_rgb_camera, depth_camera, getattr(car, "body", None)
                    ).to_dict()
                seg_to_spec = _segmentation_map(scene, registry)
                controller = DifferentialDriveController(
                    CarConfig(), waypoints=waypoints_for_scenario(SCENARIO), dt=args.dt
                )
                odometry = Odometry(position=np.array(INITIAL_POSITION[:2], dtype=np.float32), yaw=0.0)
                room_config = RoomConfig()
                scene.step()
                observation = read_sensor_observation(
                    step=0,
                    dt=args.dt,
                    car=car,
                    lidar=lidar,
                    depth_camera=depth_camera,
                    imu=imu,
                    odometry=odometry,
                    render=False,
                )
                for step in range(args.steps):
                    linear, angular, reached = controller.command_from_observation(observation)
                    car.step(linear, angular, args.dt)
                    odometry.integrate(linear, angular, args.dt)
                    scene.step()
                    observation_step = step + 1
                    observation = read_sensor_observation(
                        step=observation_step,
                        dt=args.dt,
                        car=car,
                        lidar=lidar,
                        depth_camera=depth_camera,
                        imu=imu,
                        odometry=odometry,
                        render=False,
                    )
                    if observation_step % args.sample_every and not reached:
                        if geometry_collision(np.asarray(observation["robot_pose"][:3]), room_config, obstacle_specs):
                            raise RuntimeError(f"layout {layout_index} collided at step {observation_step}")
                        continue

                    rgb = np.asarray(robot_rgb_camera.render(rgb=True, force_render=True)[0])
                    if rgb.ndim == 4:
                        rgb = rgb[0]
                    segmentation = np.asarray(
                        robot_rgb_camera.render(rgb=False, segmentation=True, force_render=True)[2]
                    )
                    height, width = int(rgb.shape[0]), int(rgb.shape[1])
                    objects, instance_mask = _truth_from_segmentation(
                        segmentation, seg_to_spec, width=width, height=height
                    )
                    stem = f"layout_{layout_index:03d}_frame_{observation_step:05d}"
                    image_rel = Path("images") / split / f"{stem}.png"
                    depth_rel = Path("depth") / split / f"{stem}.npy"
                    mask_rel = Path("masks") / split / f"{stem}.png"
                    Image.fromarray(rgb.astype(np.uint8)).save(root / image_rel)
                    np.save(root / depth_rel, np.asarray(observation["depth"], dtype=np.float32))
                    Image.fromarray(instance_mask).save(root / mask_rel)
                    frame_record = {
                        "layout_index": layout_index,
                        "layout_seed": layout_seed,
                        "split": split,
                        "step": observation_step,
                        "sim_time": observation_step * args.dt,
                        "image": str(image_rel),
                        "depth": str(depth_rel),
                        "mask": str(mask_rel),
                        "camera_pose": list(
                            build_genesis_rgbd_calibration(
                                robot_rgb_camera, depth_camera, getattr(car, "body", None)
                            ).camera_pose_from_robot_pose(observation["robot_pose"])
                        ),
                        "robot_pose": np.asarray(observation["robot_pose"]).tolist(),
                        "objects": objects,
                    }
                    metadata_file.write(json.dumps(frame_record, ensure_ascii=False) + "\n")
                    prompts = _prompts_for_objects(objects, split)
                    prompt_file.write(
                        json.dumps(
                            {
                                "image": str(image_rel),
                                "split": split,
                                "layout_index": layout_index,
                                "prompts": prompts,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    total_images += 1
                    split_counts[split] += 1
                    for obj in authored:
                        object_counts[obj.asset_variant] = object_counts.get(obj.asset_variant, 0) + 1
                    if total_images % 10 == 0:
                        print(f"generated images={total_images} layout={layout_index} split={split}")
                    if reached:
                        break
            finally:
                gs.destroy()
            layout_manifest.append(
                {
                    "layout_index": layout_index,
                    "layout_seed": layout_seed,
                    "split": split,
                    "objects": [item.to_dict() for item in authored],
                }
            )

    if calibration is None:
        raise RuntimeError("no layout was generated")
    (root / "camera_calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    (root / "layouts.json").write_text(json.dumps(layout_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    config = {
        "purpose": "open_vocabulary_evaluation_only",
        "closed_set_training": False,
        "scenario": SCENARIO,
        "layouts": args.layouts,
        "steps": args.steps,
        "sample_every": args.sample_every,
        "dt": args.dt,
        "master_seed": args.seed,
        "splits": SPLITS,
        "split_policy": "layout_level_no_adjacent_frame_leakage",
        "images": total_images,
        "split_counts": split_counts,
        "asset_variant_counts": object_counts,
        "metadata": str(metadata_path),
        "prompts": str(prompt_path),
        "camera_calibration": str(root / "camera_calibration.json"),
        "segmentation_source": "genesis_scene.segmentation_idx_dict",
        "runtime_truth_access": False,
        "note": "Truth masks and prompt targets are evaluation annotations; do not convert them into a fixed detector class list.",
    }
    (root / "generation_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return config


def main() -> None:
    try:
        result = run(parse_args())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
