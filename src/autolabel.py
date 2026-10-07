"""Evaluate LiDAR-assisted 2D auto-label proposals on KITTI.

The pipeline compares two deterministic proposals against KITTI's annotated 2D
box: (1) the envelope of the projected 3D-box corners and (2) the envelope of
LiDAR returns inside that 3D box.  It also sweeps calibration yaw drift and
creates CSV/figure evidence for Topic F.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from starter.datasets import list_frames, load_frame
from starter.kitti_io import KittiCalib, KittiObject
from starter.projection import box3d_corners_cam, draw_box2d, perturb_extrinsic, velo_to_cam


@dataclass(frozen=True)
class Proposal:
    bbox: np.ndarray | None
    point_count: int
    status: str


def clip_bbox(bbox: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    """Clip an xyxy box to an image while preserving floating-point precision."""
    height, width = image_shape[:2]
    result = np.asarray(bbox, dtype=np.float64).copy()
    result[[0, 2]] = np.clip(result[[0, 2]], 0, width - 1)
    result[[1, 3]] = np.clip(result[[1, 3]], 0, height - 1)
    return result


def bbox_iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Return IoU for two xyxy boxes; missing or degenerate boxes score zero."""
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def _bbox_from_uv(uv: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray | None:
    if len(uv) == 0 or not np.isfinite(uv).all():
        return None
    bbox = np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])
    bbox = clip_bbox(bbox, image_shape)
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox


def project_cam_unclipped(points_cam: np.ndarray, P2: np.ndarray, min_depth: float = 0.1) -> np.ndarray:
    """Project valid camera-frame points without discarding off-image pixels."""
    points_cam = np.asarray(points_cam, dtype=np.float64)
    valid = np.isfinite(points_cam).all(axis=1) & (points_cam[:, 2] > min_depth)
    selected = points_cam[valid]
    if not len(selected):
        return np.empty((0, 2), dtype=np.float64)
    homogeneous = np.column_stack([selected, np.ones(len(selected))])
    projected = (P2 @ homogeneous.T).T
    finite = np.isfinite(projected).all(axis=1) & (np.abs(projected[:, 2]) > 1e-12)
    return projected[finite, :2] / projected[finite, 2, None]


def corner_proposal(obj: KittiObject, calib: KittiCalib, image_shape: tuple[int, ...]) -> Proposal:
    """Create a 2D proposal from the envelope of eight projected 3D corners."""
    corners = box3d_corners_cam(obj)
    uv = project_cam_unclipped(corners, calib.P2)
    bbox = _bbox_from_uv(uv, image_shape) if len(uv) == 8 else None
    return Proposal(bbox, 8 if bbox is not None else len(uv), "ok" if bbox is not None else "invalid_corners")


def points_in_box(points_cam: np.ndarray, obj: KittiObject, tolerance_m: float = 0.05) -> np.ndarray:
    """Mask camera-frame points inside a KITTI oriented 3D box."""
    relative = np.asarray(points_cam, dtype=np.float64) - obj.location
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    local = relative @ rotation
    height, width, length = obj.dimensions
    return (
        (np.abs(local[:, 0]) <= length / 2 + tolerance_m)
        & (local[:, 1] >= -height - tolerance_m)
        & (local[:, 1] <= tolerance_m)
        & (np.abs(local[:, 2]) <= width / 2 + tolerance_m)
        & np.isfinite(local).all(axis=1)
    )


def lidar_proposal(
    points_cam: np.ndarray,
    obj: KittiObject,
    calib: KittiCalib,
    image_shape: tuple[int, ...],
    min_points: int = 3,
) -> Proposal:
    """Create a tight 2D proposal from projected returns inside a 3D box."""
    inside = points_in_box(points_cam, obj)
    count = int(inside.sum())
    if count < min_points:
        return Proposal(None, count, "too_few_points")
    uv = project_cam_unclipped(points_cam[inside], calib.P2)
    bbox = _bbox_from_uv(uv, image_shape)
    if bbox is None:
        return Proposal(None, count, "outside_image")
    return Proposal(bbox, count, "ok")


def _box_columns(prefix: str, box: np.ndarray | None) -> dict[str, float | str]:
    if box is None:
        return {f"{prefix}_{name}": "" for name in ("x1", "y1", "x2", "y2")}
    return {f"{prefix}_{name}": float(value) for name, value in zip(("x1", "y1", "x2", "y2"), box)}


def evaluate_frame(data_root: str, frame_id: str, yaw_deg: float, min_points: int) -> tuple[list[dict], dict]:
    frame = load_frame(data_root, frame_id)
    calib = perturb_extrinsic(frame["calib"], yaw_deg=yaw_deg)
    points_cam = velo_to_cam(frame["points"][:, :3], calib)
    rows: list[dict] = []
    for object_index, obj in enumerate(frame["labels"]):
        gt = clip_bbox(obj.bbox, frame["image"].shape)
        corner = corner_proposal(obj, frame["calib"], frame["image"].shape)
        lidar = lidar_proposal(points_cam, obj, calib, frame["image"].shape, min_points)
        row = {
            "frame_id": frame_id,
            "object_index": object_index,
            "class": obj.type,
            "distance_m": float(np.linalg.norm(obj.location)),
            "truncated": obj.truncated,
            "occluded": obj.occluded,
            "yaw_drift_deg": yaw_deg,
            "corner_iou": bbox_iou(corner.bbox, gt),
            "lidar_iou": bbox_iou(lidar.bbox, gt),
            "lidar_points": lidar.point_count,
            "lidar_status": lidar.status,
        }
        row.update(_box_columns("gt", gt))
        row.update(_box_columns("corner", corner.bbox))
        row.update(_box_columns("lidar", lidar.bbox))
        rows.append(row)
    return rows, frame


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _draw_proposals(image: np.ndarray, rows: Iterable[dict], title: str) -> np.ndarray:
    vis = image.copy()
    for row in rows:
        gt = np.array([row[f"gt_{key}"] for key in ("x1", "y1", "x2", "y2")], dtype=float)
        vis = draw_box2d(vis, gt, (0, 255, 0), f"GT {row['class']}")
        if row["corner_x1"] != "":
            corner = np.array([row[f"corner_{key}"] for key in ("x1", "y1", "x2", "y2")], dtype=float)
            vis = draw_box2d(vis, corner, (255, 80, 0), f"3D {row['corner_iou']:.2f}")
        if row["lidar_x1"] != "":
            lidar = np.array([row[f"lidar_{key}"] for key in ("x1", "y1", "x2", "y2")], dtype=float)
            vis = draw_box2d(vis, lidar, (0, 220, 255), f"LiDAR {row['lidar_iou']:.2f}")
    cv2.rectangle(vis, (0, 0), (min(vis.shape[1] - 1, 730), 28), (0, 0, 0), -1)
    cv2.putText(vis, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return vis


def _plot_method_comparison(rows: list[dict], path: Path) -> None:
    corner = np.array([row["corner_iou"] for row in rows], dtype=float)
    lidar = np.array([row["lidar_iou"] for row in rows if row["lidar_status"] == "ok"], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].boxplot([corner, lidar], tick_labels=["3D corners", "LiDAR tight"])
    axes[0].set_ylabel("IoU with KITTI 2D label")
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_title("Proposal quality (valid proposals)")
    distances = np.array([row["distance_m"] for row in rows], dtype=float)
    points = np.array([row["lidar_points"] for row in rows], dtype=float)
    axes[1].scatter(distances, points + 1, s=18, alpha=0.7)
    axes[1].axhline(4, color="crimson", linestyle="--", label="minimum = 3 returns")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Object distance (m)")
    axes[1].set_ylabel("LiDAR returns + 1 (log scale)")
    axes[1].set_title("Sparse returns create failures")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _summarize_drift(rows: list[dict], review_iou: float) -> list[dict]:
    summary = []
    for yaw in sorted({float(row["yaw_drift_deg"]) for row in rows}):
        group = [row for row in rows if float(row["yaw_drift_deg"]) == yaw]
        valid = [row for row in group if row["lidar_status"] == "ok"]
        ious = np.array([row["lidar_iou"] for row in valid], dtype=float)
        review = sum(row["lidar_status"] != "ok" or row["lidar_iou"] < review_iou for row in group)
        summary.append({
            "yaw_drift_deg": yaw,
            "objects": len(group),
            "valid_proposals": len(valid),
            "coverage": len(valid) / len(group) if group else 0.0,
            "median_lidar_iou": float(np.median(ious)) if len(ious) else 0.0,
            "mean_lidar_iou": float(np.mean(ious)) if len(ious) else 0.0,
            "review_iou_threshold": review_iou,
            "review_flag_rate": review / len(group) if group else 0.0,
        })
    return summary


def _plot_drift(summary: list[dict], path: Path) -> None:
    yaw = [row["yaw_drift_deg"] for row in summary]
    median = [row["median_lidar_iou"] for row in summary]
    coverage = [row["coverage"] for row in summary]
    review = [row["review_flag_rate"] for row in summary]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(yaw, median, "o-", label="median LiDAR-box IoU")
    ax.plot(yaw, coverage, "s-", label="proposal coverage")
    ax.plot(yaw, review, "^-", label="review-flag rate")
    ax.axhline(summary[0]["review_iou_threshold"], color="gray", linestyle="--", alpha=0.7, label="IoU review threshold")
    ax.set_xlabel("Injected LiDAR yaw drift (degrees)")
    ax.set_ylabel("Metric (0–1)")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    ax.set_title("Auto-label QA sensitivity to calibration drift")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    frame_ids = list_frames(args.data_root)
    if args.frames:
        unknown = sorted(set(args.frames) - set(frame_ids))
        if unknown:
            raise ValueError(f"Unknown frame(s): {', '.join(unknown)}")
        frame_ids = args.frames
    if not frame_ids:
        raise ValueError("No frames found")

    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    baseline: list[dict] = []
    frames: dict[str, dict] = {}
    for frame_id in frame_ids:
        rows, frame = evaluate_frame(args.data_root, frame_id, 0.0, args.min_points)
        baseline.extend(rows)
        frames[frame_id] = frame
    _write_csv(out_dir / "autolabel_objects.csv", baseline)

    drift_rows: list[dict] = []
    for yaw in args.yaw_levels:
        for frame_id in frame_ids:
            rows, _ = evaluate_frame(args.data_root, frame_id, yaw, args.min_points)
            drift_rows.extend(rows)
    _write_csv(out_dir / "autolabel_drift_objects.csv", drift_rows)
    summary = _summarize_drift(drift_rows, args.review_iou)
    _write_csv(out_dir / "autolabel_drift_summary.csv", summary)

    demo_frame = args.demo_frame if args.demo_frame in frames else max(
        frame_ids,
        key=lambda fid: sum(row["lidar_status"] == "ok" for row in baseline if row["frame_id"] == fid),
    )
    demo_rows = [row for row in baseline if row["frame_id"] == demo_frame]
    demo = _draw_proposals(
        frames[demo_frame]["image"], demo_rows,
        "KITTI: GT=green, projected 3D=blue, LiDAR-tight=yellow",
    )
    cv2.imwrite(str(figures_dir / f"demo_autolabel_{demo_frame}.png"), demo)

    failures = sorted(
        baseline,
        key=lambda row: (row["lidar_status"] == "ok", row["lidar_points"], row["lidar_iou"]),
    )
    failure = failures[0]
    failure_image = _draw_proposals(
        frames[failure["frame_id"]]["image"], [failure],
        f"FAIL: {failure['class']} at {failure['distance_m']:.1f} m; {failure['lidar_points']} LiDAR points; {failure['lidar_status']}",
    )
    cv2.imwrite(str(figures_dir / f"fail_sparse_lidar_{failure['frame_id']}.png"), failure_image)

    _plot_method_comparison(baseline, figures_dir / "iou_method_comparison.png")
    _plot_drift(summary, figures_dir / "calibration_drift.png")

    valid = [row for row in baseline if row["lidar_status"] == "ok"]
    print(f"frames={len(frame_ids)} objects={len(baseline)} valid_lidar_proposals={len(valid)}")
    print(f"corner_median_iou={np.median([row['corner_iou'] for row in baseline]):.3f}")
    if valid:
        print(f"lidar_median_iou={np.median([row['lidar_iou'] for row in valid]):.3f}")
    print(f"wrote CSV files and figures to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark 3D-corner and LiDAR-tight 2D label proposals on KITTI-format data."
    )
    parser.add_argument("--data-root", default="data/kitti_mini", help="KITTI-format dataset root")
    parser.add_argument("--out-dir", default="results", help="output directory for CSV and figures")
    parser.add_argument("--frames", nargs="*", help="optional frame IDs; default uses every frame")
    parser.add_argument("--demo-frame", default="000011", help="frame used for the qualitative demo")
    parser.add_argument("--min-points", type=int, default=3, help="minimum LiDAR returns required for a tight box")
    parser.add_argument("--review-iou", type=float, default=0.5, help="IoU below which a proposal is flagged for review")
    parser.add_argument(
        "--yaw-levels", type=float, nargs="+", default=[-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3],
        help="calibration yaw drift levels in degrees",
    )
    args = parser.parse_args()
    if args.min_points < 1:
        parser.error("--min-points must be at least 1")
    if not 0 <= args.review_iou <= 1:
        parser.error("--review-iou must be in [0, 1]")
    return args


if __name__ == "__main__":
    run(parse_args())
