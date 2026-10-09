"""Bonus analyses for Day 23 (RUBRIC.md §2): visualisation, calibration, CVAT export.

Run from the repo root with the lab environment active:

    python student/bonus/bonus_analysis.py cache  --config student/config/paths.yaml
    python student/bonus/bonus_analysis.py verify
    python student/bonus/bonus_analysis.py viz
    python student/bonus/bonus_analysis.py calib
    python student/bonus/bonus_analysis.py cvat

``cache`` runs the provided detector (Part A–D) once per frame and stores the
detections, valid ground truth, FRONT camera labels, BEV maps and FRONT images
under ``data/cache/`` (git-ignored: it contains Waymo-derived data).

Every other stage replays the per-frame loop of ``fusion_lab.scripts.run_lab``
on that cache with the same platform classes (``Sensor``, ``Filter``,
``TrackManager``, ``evaluation``) and the student workspace (Part E–H).
``verify`` checks that the replay reproduces ``student/artifacts/metrics_*.json``
exactly, so the bonus numbers come from the same tracker as the graded run.
No stage writes to ``student/artifacts/``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BONUS = ROOT / "student" / "bonus"
CACHE = ROOT / "data" / "cache"
CACHE_JSON = CACHE / "frames.json"

FRONT = 1  # dataset_pb2.CameraName.FRONT
TYPE_VEHICLE = 1  # label_pb2.Label.Type.TYPE_VEHICLE


# ---------------------------------------------------------------------------
# Stage 1: run the provided detector once and cache everything per frame.
# ---------------------------------------------------------------------------


def _box_dict(label: Any) -> dict[str, Any]:
    box = label.box
    return {
        "id": label.id, "x": box.center_x, "y": box.center_y, "z": box.center_z,
        "l": box.length, "w": box.width, "h": box.height, "yaw": box.heading,
    }


def build_cache(config_path: Path) -> None:
    """Run PCL → BEV → FPN once per frame and cache what the tracker consumes."""
    import cv2
    import torch

    from fusion_lab.evaluation import detection_counts, valid_ground_truth
    from fusion_lab.lidar_pcl import pcl_from_range_image
    from fusion_lab.scripts import run_lab
    from fusion_lab.workspace_loader import load_workspace

    run_lab._setup_import_paths()
    from simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2
    from simple_waymo_open_dataset_reader import utils as waymo_utils

    cfg = run_lab._load_paths_config(config_path)
    ws = load_workspace()
    bev, det_pipe, det_metrics = ws["bev_mapping"], ws["detection_pipeline"], ws["detection_metrics"]
    weights = run_lab._resolve_weights(cfg)
    det_cfg = det_pipe.load_fpn_resnet_config(str(weights) if weights else None)
    model = det_pipe.create_fpn_model(det_cfg, str(weights) if weights else None)

    segment = cfg["segment"]
    frame_start, frame_end = int(cfg.get("frame_start", 0)), int(cfg.get("frame_end", 20))
    (CACHE / "bev").mkdir(parents=True, exist_ok=True)
    (CACHE / "front").mkdir(parents=True, exist_ok=True)

    reader = WaymoDataFileReader(str(Path(cfg["waymo_dir"]) / segment))
    frames, camera_calib = [], None
    for cnt, frame in enumerate(reader):
        if cnt < frame_start:
            continue
        if cnt > frame_end:
            break
        if camera_calib is None:
            calib = waymo_utils.get(frame.context.camera_calibrations, dataset_pb2.CameraName.FRONT)
            camera_calib = {
                "intrinsic": list(calib.intrinsic), "extrinsic": list(calib.extrinsic.transform),
                "width": calib.width, "height": calib.height,
            }
        points = pcl_from_range_image(frame, dataset_pb2.LaserName.TOP)
        bev_map = bev.bev_maps_from_pcl(points, det_cfg)
        tensor = torch.from_numpy(bev_map).unsqueeze(0).float()
        detections = det_pipe.detect_objects_from_bev(tensor, model, det_cfg)
        labels = valid_ground_truth(frame.laser_labels, det_cfg, label_pb2.Label.Type.TYPE_VEHICLE)
        group = next((g for g in frame.camera_labels if g.name == dataset_pb2.CameraName.FRONT), None)
        frames.append({
            "frame": cnt,
            "detections": [[float(v) for v in det] for det in detections],
            "det_counts": detection_counts(labels, detections, det_metrics),
            "valid_gt": [_box_dict(label) for label in labels],
            "front": None if group is None else [
                {"id": l.id, "type": int(l.type), "cx": l.box.center_x, "cy": l.box.center_y,
                 "w": l.box.width, "l": l.box.length} for l in group.labels
            ],
        })
        # Display copies only: BEV as an 8-bit RGB image, FRONT as the raw JPEG.
        bev_rgb = (np.clip(np.transpose(bev_map, (1, 2, 0)), 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(str(CACHE / "bev" / f"{cnt:04d}.png"), bev_rgb[:, :, ::-1])
        image = waymo_utils.get(frame.images, dataset_pb2.CameraName.FRONT)
        (CACHE / "front" / f"{cnt:04d}.jpg").write_bytes(image.image)
        print(f"cached frame {cnt}: {len(detections)} detections, {len(labels)} valid GT", flush=True)

    payload = {"segment": segment, "frames_range": [frame_start, frames[-1]["frame"]],
               "camera_calib": camera_calib, "det_cfg": {
                   "lim_x": list(det_cfg.lim_x), "lim_y": list(det_cfg.lim_y),
                   "lim_z": list(det_cfg.lim_z), "bev_height": det_cfg.bev_height,
                   "bev_width": det_cfg.bev_width}, "frames": frames}
    CACHE_JSON.write_text(json.dumps(payload))
    print(f"wrote {CACHE_JSON}")


def load_cache() -> dict[str, Any]:
    if not CACHE_JSON.is_file():
        sys.exit(f"Missing {CACHE_JSON}; run the `cache` stage first.")
    return json.loads(CACHE_JSON.read_text())


# ---------------------------------------------------------------------------
# Stage 2: replay the official tracking loop on the cache, with probes.
# ---------------------------------------------------------------------------


def _ns_label(box: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(id=box["id"], box=SimpleNamespace(
        center_x=box["x"], center_y=box["y"], center_z=box["z"]))


def _fake_frame(front: list[dict[str, Any]] | None) -> SimpleNamespace:
    if front is None:
        return SimpleNamespace(camera_labels=[])
    labels = [SimpleNamespace(type=l["type"], box=SimpleNamespace(center_x=l["cx"], center_y=l["cy"]))
              for l in front]
    return SimpleNamespace(camera_labels=[SimpleNamespace(name=FRONT, labels=labels)])


def _camera_calibration(cache: dict[str, Any]) -> SimpleNamespace:
    c = cache["camera_calib"]
    return SimpleNamespace(intrinsic=c["intrinsic"], width=c["width"], height=c["height"],
                           extrinsic=SimpleNamespace(transform=c["extrinsic"]))


def perturbed_extrinsic(transform: np.ndarray, yaw_deg: float = 0.0, pitch_deg: float = 0.0,
                        lateral_m: float = 0.0) -> np.ndarray:
    """Return a camera→vehicle transform the tracker *believes*, offset in the camera frame.

    Waymo camera axes are x forward, y left, z up. A yaw error rotates about the
    camera z axis, a pitch error about its y axis; ``lateral_m`` shifts the
    believed mounting position along the camera y axis.
    """
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    delta = np.eye(4)
    delta[:3, :3] = rz @ ry
    delta[:3, 3] = [0.0, lateral_m, 0.0]
    return np.asarray(transform, dtype=float).reshape(4, 4) @ delta


def _snapshot(tracks: list[Any]) -> dict[int, dict[str, Any]]:
    return {t.id: {"x": np.asarray(t.x, dtype=float).reshape(-1).tolist(),
                   "P": np.asarray(t.P, dtype=float).tolist(), "state": t.state,
                   "score": float(t.score), "l": t.length, "w": t.width, "h": t.height,
                   "yaw": t.yaw} for t in tracks}


def replay(cache: dict[str, Any], mode: str, seed: int = 0,
           camera_transform: np.ndarray | None = None, probe: bool = False) -> dict[str, Any]:
    """Replay ``run_lab.run`` for one mode on cached detections; optionally probe the camera pass."""
    from fusion_lab import tracking_params
    from fusion_lab.evaluation import aggregate_records, tracking_counts
    from fusion_lab.scripts.run_lab import _front_observations, _lidar_observations
    from fusion_lab.tracking.filter import Filter
    from fusion_lab.tracking.manager import TrackManager
    from fusion_lab.tracking.sensors import Sensor
    from fusion_lab.workspace_loader import load_workspace

    ws = load_workspace()
    kalman, assoc, cam = ws["kalman"], ws["association"], ws["camera_fusion"]
    det_cfg = SimpleNamespace(**cache["det_cfg"])
    rng = np.random.default_rng(seed)
    innovations: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    current = {"frame": -1}

    class ProbeFilter(Filter):
        """Record the camera innovation γ, S and NIS right before each EKF update."""

        def update(self, track: Any, meas: Any) -> None:
            if probe and meas.sensor.name == "camera":
                H = meas.sensor.get_H(track.x)
                gamma = kalman.innovation(track.x, meas)
                S = kalman.innovation_covariance(track.P, meas, H)
                innovations.append({
                    "frame": current["frame"], "track": track.id,
                    "gamma": np.asarray(gamma).reshape(-1).tolist(),
                    "S": np.asarray(S).tolist(),
                    "nis": float((gamma.T @ np.linalg.inv(S) @ gamma).item()),
                })
            super().update(track, meas)

    original_cost = assoc.association_cost_matrix

    def probing_cost(track_list, meas_list):
        """Wrap Part F: log, per visible track, its nearest camera candidate before gating."""
        costs = original_cost(track_list, meas_list)
        if probe and meas_list and meas_list[0].sensor.name == "camera":
            for i, track in enumerate(track_list):
                if not meas_list[0].sensor.in_fov(track.x):
                    continue
                d2 = [assoc.mahalanobis_distance(track, m) for m in meas_list]
                j = int(np.argmin(d2))
                gamma = kalman.innovation(track.x, meas_list[j])
                candidates.append({
                    "frame": current["frame"], "track": track.id, "state": track.state,
                    "d2": float(d2[j]), "accepted": bool(np.isfinite(costs[i, j])),
                    "gamma": np.asarray(gamma).reshape(-1).tolist(),
                })
        return costs

    assoc.association_cost_matrix = probing_cost
    try:
        KF = ProbeFilter(kalman)
        manager = TrackManager(ws["track_management"])
        lidar_sensor = Sensor("lidar", None, cam)
        camera_sensor = None
        if mode == "fused":
            camera_sensor = Sensor("camera", _camera_calibration(cache), cam)
            if camera_transform is not None:
                camera_sensor.sens_to_veh = np.asmatrix(camera_transform)
                camera_sensor.veh_to_sens = np.asmatrix(np.linalg.inv(camera_transform))
        records, history = [], []
        for fr in cache["frames"]:
            cnt = current["frame"] = fr["frame"]
            labels = [_ns_label(b) for b in fr["valid_gt"]]
            observations = _lidar_observations(cnt, fr["detections"], lidar_sensor, det_cfg)
            for track in manager.track_list:
                KF.predict(track)
                track.set_t(cnt * tracking_params.dt)
            assoc.associate_and_update(manager, observations, KF, lidar_sensor)
            after_lidar = _snapshot(manager.track_list)
            camera_obs = None
            if camera_sensor is not None:
                camera_obs = _front_observations(_fake_frame(fr["front"]), cnt, camera_sensor, rng,
                                                 FRONT, TYPE_VEHICLE)
                if camera_obs is not None:
                    assoc.associate_and_update(manager, camera_obs, KF, camera_sensor)
            record = {"mode": mode, "frame": cnt, **fr["det_counts"], "valid_gt": len(labels),
                      **tracking_counts(manager.track_list, labels)}
            records.append(record)
            history.append({
                "frame": cnt, "after_lidar": after_lidar, "final": _snapshot(manager.track_list),
                "camera_meas": [] if not camera_obs else
                [np.asarray(m.z).reshape(-1).tolist() for m in camera_obs],
            })
    finally:
        assoc.association_cost_matrix = original_cost
    metrics = aggregate_records(records, mode, cache["frames_range"], seed, cache["segment"])
    return {"metrics": metrics, "records": records, "history": history,
            "innovations": innovations, "candidates": candidates,
            "camera_sensor": camera_sensor}


def verify(cache: dict[str, Any]) -> None:
    """Check that the replay reproduces the graded per-mode metrics bit for bit."""
    artifacts = ROOT / "student" / "artifacts"
    for mode in ("lidar", "fused"):
        official = json.loads((artifacts / f"metrics_{mode}.json").read_text())
        replayed = replay(cache, mode)["metrics"]
        status = "IDENTICAL" if replayed == official else "DIFFERENT"
        print(f"{mode}: replay vs artifacts/metrics_{mode}.json → {status}")
        if replayed != official:
            print(json.dumps({"official": official["tracking"], "replay": replayed["tracking"]}, indent=2))
            sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers shared by the visual stages.
# ---------------------------------------------------------------------------


def match_tracks_to_gt(snapshot: dict[int, dict[str, Any]], gt: list[dict[str, Any]]):
    """Match confirmed tracks to GT exactly like ``evaluation.tracking_counts`` (XY gate 2 m)."""
    from fusion_lab.evaluation import TRACK_GATE_METERS, _partial_assignment

    ids = [tid for tid, t in snapshot.items() if t["state"] == "confirmed"]
    pos = np.array([snapshot[t]["x"][:3] for t in ids]).reshape(-1, 3)
    centres = np.array([(b["x"], b["y"], b["z"]) for b in gt]).reshape(-1, 3)
    delta = pos[:, None, :] - centres[None, :, :]
    dist = np.linalg.norm(delta[:, :, :2], axis=2)
    pairs = _partial_assignment(dist, np.isfinite(dist) & (dist <= TRACK_GATE_METERS))
    matched = {ids[i]: (gt[j]["id"], delta[i, j]) for i, j in pairs}
    ghosts = [t for t in ids if t not in matched]
    return matched, ghosts


def _bev_extent(cache: dict[str, Any]) -> list[float]:
    c = cache["det_cfg"]
    return [c["lim_y"][0], c["lim_y"][1], c["lim_x"][0], c["lim_x"][1]]


def _box_polygon(x, y, l, w, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    corners = [(l / 2, w / 2), (l / 2, -w / 2), (-l / 2, -w / 2), (-l / 2, w / 2), (l / 2, w / 2)]
    return [(x + c * a - s * b, y + s * a + c * b) for a, b in corners]


# ---------------------------------------------------------------------------
# Stage 3: visualisations (bonus +3).
# ---------------------------------------------------------------------------


def visualise(cache: dict[str, Any]) -> None:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = BONUS / "viz"
    out.mkdir(parents=True, exist_ok=True)
    lidar = replay(cache, "lidar")
    fused = replay(cache, "fused", probe=True)
    cam_fusion = __import__("fusion_lab.workspace_loader", fromlist=["x"]).load_workspace()["camera_fusion"]
    camera = fused["camera_sensor"]
    gt_by_frame = {fr["frame"]: fr["valid_gt"] for fr in cache["frames"]}

    # Per GT object: 3D error of its matched confirmed track in each mode.
    errors: dict[str, dict[str, dict[int, np.ndarray]]] = {"lidar": {}, "fused": {}}
    for name, run in (("lidar", lidar), ("fused", fused)):
        for h in run["history"]:
            matched, _ = match_tracks_to_gt(h["final"], gt_by_frame[h["frame"]])
            for tid, (gid, delta) in matched.items():
                errors[name].setdefault(gid, {})[h["frame"]] = delta

    # Camera effect per fused track and frame: state shift produced by the camera update.
    shifts = []
    for h in fused["history"]:
        matched, _ = match_tracks_to_gt(h["final"], gt_by_frame[h["frame"]])
        for tid, after in h["final"].items():
            before = h["after_lidar"].get(tid)
            if before is None or tid not in matched:
                continue
            shift = np.array(after["x"][:3]) - np.array(before["x"][:3])
            if np.linalg.norm(shift) > 0:
                gid = matched[tid][0]
                gt = next(b for b in gt_by_frame[h["frame"]] if b["id"] == gid)
                err_before = np.array(before["x"][:3]) - np.array([gt["x"], gt["y"], gt["z"]])
                err_after = np.array(after["x"][:3]) - np.array([gt["x"], gt["y"], gt["z"]])
                shifts.append({"frame": h["frame"], "track": tid, "gt": gid, "shift": shift,
                               "err_before": err_before, "err_after": err_after})
    summary: dict[str, Any] = {"camera_updates_on_matched_tracks": len(shifts)}
    if shifts:
        eb = np.array([s["err_before"] for s in shifts])
        ea = np.array([s["err_after"] for s in shifts])
        summary["mean_abs_err_before_camera_xyz"] = np.abs(eb).mean(axis=0).round(4).tolist()
        summary["mean_abs_err_after_camera_xyz"] = np.abs(ea).mean(axis=0).round(4).tolist()
        summary["rms_3d_err_before_camera"] = float(np.sqrt((eb**2).sum(axis=1).mean()))
        summary["rms_3d_err_after_camera"] = float(np.sqrt((ea**2).sum(axis=1).mean()))
        summary["mean_abs_shift_xyz"] = np.abs(np.array([s["shift"] for s in shifts])).mean(axis=0).round(4).tolist()

    # Figure 1 — BEV of one frame: detections, GT, lidar-only vs fused confirmed tracks.
    frame_id = max(shifts, key=lambda s: np.linalg.norm(s["shift"][:2]))["frame"] if shifts else cache["frames"][0]["frame"]
    fr = next(f for f in cache["frames"] if f["frame"] == frame_id)
    bev = cv2.imread(str(CACHE / "bev" / f"{frame_id:04d}.png"))[:, :, ::-1]
    lidar_final = next(h for h in lidar["history"] if h["frame"] == frame_id)["final"]
    fused_h = next(h for h in fused["history"] if h["frame"] == frame_id)
    fused_confirmed = [tid for tid, t in fused_h["final"].items() if t["state"] == "confirmed"]
    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(len(fused_confirmed) or 1, 2, width_ratios=[1.6, 1])
    ax = fig.add_subplot(grid[:, 0])
    zooms = [fig.add_subplot(grid[k, 1]) for k in range(len(fused_confirmed))]

    def draw_bev(axis, markersize=1.0):
        axis.imshow(bev, origin="lower", extent=_bev_extent(cache))
        for b in fr["valid_gt"]:
            poly = _box_polygon(b["x"], b["y"], b["l"], b["w"], b["yaw"])
            axis.plot([p[1] for p in poly], [p[0] for p in poly], color="lime", lw=1.2)
            axis.plot(b["y"], b["x"], ".", color="lime", ms=8 * markersize)
        for d in fr["detections"]:
            axis.plot(d[2], d[1], "rx", ms=7 * markersize, mew=1.5)
        for t in lidar_final.values():
            if t["state"] == "confirmed":
                axis.plot(t["x"][1], t["x"][0], "o", mfc="none", mec="deepskyblue",
                          ms=11 * markersize, mew=1.8)
        for tid in fused_confirmed:
            t, before = fused_h["final"][tid], fused_h["after_lidar"][tid]
            axis.plot(before["x"][1], before["x"][0], "^", mfc="none", mec="magenta",
                      ms=9 * markersize, mew=1.5)
            axis.plot(t["x"][1], t["x"][0], "+", color="orange", ms=12 * markersize, mew=2)

    draw_bev(ax)
    for tid in fused_confirmed:
        t = fused_h["final"][tid]
        ax.annotate(f"F{tid}", (t["x"][1], t["x"][0]), xytext=(6, 6), textcoords="offset points",
                    color="orange", fontsize=9)
    ax.invert_xaxis()
    ax.set_xlabel("y vehicle [m] (left is +y)")
    ax.set_ylabel("x vehicle [m] (forward)")
    ax.set_title(f"Frame {frame_id}: GT box/centre (green), detection (red x), lidar-only track (blue o),\n"
                 "fused track after lidar update (magenta ^) and after camera update (orange +)",
                 fontsize=10)
    for axis, tid in zip(zooms, fused_confirmed):
        draw_bev(axis, markersize=1.6)
        t, before = fused_h["final"][tid], fused_h["after_lidar"][tid]
        axis.annotate("", xy=(t["x"][1], t["x"][0]), xytext=(before["x"][1], before["x"][0]),
                      arrowprops={"arrowstyle": "->", "color": "white", "lw": 1.5})
        axis.set_xlim(t["x"][1] + 0.6, t["x"][1] - 0.6)
        axis.set_ylim(t["x"][0] - 0.6, t["x"][0] + 0.6)
        shift = np.array(t["x"][:3]) - np.array(before["x"][:3])
        axis.set_title(f"zoom F{tid} (±0.6 m): camera shift Δx={shift[0]:+.3f}, Δy={shift[1]:+.3f}, "
                       f"Δz={shift[2]:+.3f} m", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "01_bev_tracks_lidar_vs_fused.png", dpi=130)
    plt.close(fig)

    # Figure 2 — FRONT image: camera measurements and projected track before/after camera update.
    image = cv2.imread(str(CACHE / "front" / f"{frame_id:04d}.jpg"))[:, :, ::-1]
    projected = []
    for tid, after in fused_h["final"].items():
        before = fused_h["after_lidar"].get(tid)
        if after["state"] != "confirmed" or before is None:
            continue
        xb = np.asmatrix(before["x"]).T
        xa = np.asmatrix(after["x"]).T
        if not (cam_fusion.is_in_field_of_view(xb, camera) and cam_fusion.is_in_field_of_view(xa, camera)):
            continue
        pb = np.asarray(cam_fusion.camera_measurement_prediction(xb, camera)).ravel()
        pa = np.asarray(cam_fusion.camera_measurement_prediction(xa, camera)).ravel()
        meas = min(fused_h["camera_meas"], key=lambda z: math.dist(z, pb)) if fused_h["camera_meas"] else None
        projected.append((tid, pb, pa, meas))
    fig = plt.figure(figsize=(13, 11))
    grid = fig.add_gridspec(2, max(1, len(projected)), height_ratios=[2, 1.1])
    ax = fig.add_subplot(grid[0, :])

    def draw_front(axis, size=1.0):
        axis.imshow(image)
        for z in fused_h["camera_meas"]:
            axis.plot(z[0], z[1], "s", mfc="none", mec="yellow", ms=10 * size, mew=1.5)
        for tid, pb, pa, _ in projected:
            axis.plot(pb[0], pb[1], "o", mfc="none", mec="deepskyblue", ms=9 * size, mew=1.8)
            axis.plot(pa[0], pa[1], "+", color="orange", ms=12 * size, mew=2)

    draw_front(ax)
    for tid, _, pa, _ in projected:
        ax.annotate(f"F{tid}", (pa[0], pa[1]), xytext=(8, -8), textcoords="offset points",
                    color="orange", fontsize=9)
    ax.set_title(f"Frame {frame_id} FRONT: camera measurement z (yellow square = noisy GT 2D centre),\n"
                 "h(x) after lidar update (blue o) and after camera update (orange +)")
    ax.axis("off")
    for k, (tid, pb, pa, meas) in enumerate(projected):
        axis = fig.add_subplot(grid[1, k])
        draw_front(axis, size=1.8)
        if meas is not None:
            axis.annotate("", xy=meas, xytext=pb,
                          arrowprops={"arrowstyle": "->", "color": "red", "lw": 1.5})
            gamma = np.subtract(meas, pb)
            residual = np.subtract(meas, pa)
            axis.set_title(f"F{tid}: γ = z − h(x) = ({gamma[0]:+.1f}, {gamma[1]:+.1f}) px\n"
                           f"after camera update z − h(x) = ({residual[0]:+.1f}, {residual[1]:+.1f}) px",
                           fontsize=9)
        points = np.array([pb, pa] + ([meas] if meas is not None else []))
        centre = (points.min(axis=0) + points.max(axis=0)) / 2
        half = max(25.0, float(np.abs(points - centre).max()) + 12.0)
        axis.set_xlim(centre[0] - half, centre[0] + half)
        axis.set_ylim(centre[1] + half * 0.75, centre[1] - half * 0.75)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(out / "02_front_projection_before_after_camera.png", dpi=110)
    plt.close(fig)

    # Figure 3 — one GT vehicle over time: lidar-only vs fused error and camera shift.
    common = {g: (errors["lidar"][g], errors["fused"].get(g, {})) for g in errors["lidar"]}
    best_gid = max(common, key=lambda g: len(set(common[g][0]) & set(common[g][1])))
    le, fe = common[best_gid]
    frames_l = sorted(le)
    frames_f = sorted(fe)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for k, axis_name in enumerate(("x (depth)", "y (lateral)", "z (height)")):
        axes[k].plot(frames_l, [le[f][k] for f in frames_l], "-", color="deepskyblue", label="lidar-only")
        axes[k].plot(frames_f, [fe[f][k] for f in frames_f], "-", color="orange", label="fused")
        axes[k].axhline(0, color="gray", lw=0.6)
        axes[k].set_ylabel(f"error {axis_name} [m]")
    axes[0].legend()
    axes[0].set_title(f"GT vehicle {best_gid[:8]}…: track − GT position error per axis")
    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(out / "03_track_error_timeseries.png", dpi=130)
    plt.close(fig)

    # Figure 4 — effect of the camera update on all matched tracks, per axis.
    if shifts:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for k, axis_name in enumerate(("x (depth)", "y (lateral)", "z (height)")):
            bins = np.linspace(-0.6, 0.6, 49)
            axes[k].hist(eb[:, k], bins=bins, alpha=0.6, color="deepskyblue", label="after lidar update")
            axes[k].hist(ea[:, k], bins=bins, alpha=0.6, color="orange", label="after camera update")
            axes[k].set_title(f"{axis_name}: mean |err| {np.abs(eb[:, k]).mean():.3f} → "
                              f"{np.abs(ea[:, k]).mean():.3f} m")
            axes[k].set_xlabel("track − GT [m]")
        axes[0].legend()
        fig.suptitle(f"Same fused tracks, same frames (n={len(shifts)} camera updates): "
                     "position error before vs after the camera update")
        fig.tight_layout()
        fig.savefig(out / "04_camera_update_error_histogram.png", dpi=130)
        plt.close(fig)

    summary.update({"figure_frame": frame_id, "timeseries_gt_id": best_gid,
                    "timeseries_frames_lidar": len(frames_l), "timeseries_frames_fused": len(frames_f),
                    "lidar_metrics": lidar["metrics"]["tracking"]["lidar"],
                    "fused_metrics": fused["metrics"]["tracking"]["fused"]})
    (out / "viz_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


# ---------------------------------------------------------------------------
# Stage 4: camera calibration sensitivity (bonus +4).
# ---------------------------------------------------------------------------


CALIBRATION_LEVELS = [
    ("baseline", 0.0, 0.0, 0.0),
    ("yaw +0.25°", 0.25, 0.0, 0.0),
    ("yaw +0.5°", 0.5, 0.0, 0.0),
    ("yaw +1°", 1.0, 0.0, 0.0),
    ("yaw +2°", 2.0, 0.0, 0.0),
    ("yaw +5°", 5.0, 0.0, 0.0),
    ("pitch +1°", 0.0, 1.0, 0.0),
    ("lateral +0.3 m", 0.0, 0.0, 0.3),
]


def calibration(cache: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fusion_lab import tracking_params
    from scipy.stats import chi2

    gate = float(chi2.ppf(tracking_params.gating_threshold, df=2))
    true_tf = np.asarray(cache["camera_calib"]["extrinsic"], dtype=float).reshape(4, 4)
    lidar = replay(cache, "lidar")["metrics"]["tracking"]["lidar"]
    rows = []
    for name, yaw, pitch, lateral in CALIBRATION_LEVELS:
        tf = perturbed_extrinsic(true_tf, yaw, pitch, lateral)
        run = replay(cache, "fused", camera_transform=tf, probe=True)
        m = run["metrics"]["tracking"]["fused"]
        inn = run["innovations"]
        cand = [c for c in run["candidates"] if c["state"] == "confirmed"]
        gam = np.array([i["gamma"] for i in inn]).reshape(-1, 2)
        cgam = np.array([c["gamma"] for c in cand]).reshape(-1, 2)
        rows.append({
            "level": name, "yaw_deg": yaw, "pitch_deg": pitch, "lateral_m": lateral,
            "camera_updates": len(inn),
            "confirmed_candidates": len(cand),
            "gate_accept_rate": float(np.mean([c["accepted"] for c in cand])) if cand else None,
            "mean_gamma_u_px": float(gam[:, 0].mean()) if len(gam) else None,
            "mean_gamma_v_px": float(gam[:, 1].mean()) if len(gam) else None,
            "nearest_mean_gamma_u_px": float(cgam[:, 0].mean()) if len(cgam) else None,
            "nearest_mean_gamma_v_px": float(cgam[:, 1].mean()) if len(cgam) else None,
            "mean_nis_accepted": float(np.mean([i["nis"] for i in inn])) if inn else None,
            "median_d2_nearest": float(np.median([c["d2"] for c in cand])) if cand else None,
            "rmse_fused": m["rmse"], "rmse_minus_lidar": None if m["rmse"] is None else m["rmse"] - lidar["rmse"],
            "matches": m["matches"], "ghost_track_frames": m["ghost_track_frames"],
            "missed_gt_frames": m["missed_gt_frames"],
        })
        print(json.dumps(rows[-1]), flush=True)
    result = {"chi2_gate_2dof": gate, "lidar_only": lidar, "levels": rows,
              "note": "Camera measurements are always generated from the TRUE calibration; "
                      "only the extrinsic the tracker uses (h(x), H) is perturbed."}
    (BONUS / "calibration_results.json").write_text(json.dumps(result, indent=2))

    yaw_rows = [r for r in rows if r["pitch_deg"] == 0 and r["lateral_m"] == 0]
    yaws = [r["yaw_deg"] for r in yaw_rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(yaws, [r["nearest_mean_gamma_u_px"] for r in yaw_rows], "o-", label="nearest candidate")
    axes[0].plot(yaws, [r["mean_gamma_u_px"] for r in yaw_rows], "s--", label="accepted updates")
    axes[0].set_xlabel("yaw error [deg]")
    axes[0].set_ylabel("mean innovation γ_u [px]")
    axes[0].legend()
    axes[1].plot(yaws, [r["gate_accept_rate"] for r in yaw_rows], "o-")
    axes[1].set_xlabel("yaw error [deg]")
    axes[1].set_ylabel(f"χ² gate accept rate (d² < {gate:.2f})")
    axes[2].plot(yaws, [r["rmse_fused"] for r in yaw_rows], "o-", color="orange", label="fused")
    axes[2].axhline(lidar["rmse"], color="deepskyblue", ls="--", label="lidar-only")
    axes[2].set_xlabel("yaw error [deg]")
    axes[2].set_ylabel("RMSE [m]")
    axes[2].legend()
    fig.suptitle("Camera extrinsic yaw error: innovation bias, gating, and tracking RMSE")
    fig.tight_layout()
    fig.savefig(BONUS / "viz" / "05_calibration_yaw_sweep.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stage 5: CVAT export (bonus +3).
# ---------------------------------------------------------------------------


def _bev_pixel_box(cache: dict[str, Any], x, y, l, w, yaw):
    """Box in the CVAT frame image: BEV rotated 180° so forward (+x) is up and +y is left.

    Returns ``(xtl, ytl, xbr, ybr, rotation)``; CVAT rotates clockwise in degrees, while
    the vehicle yaw is counter-clockwise seen from above, hence ``-yaw``.
    """
    c = cache["det_cfg"]
    cell = (c["lim_x"][1] - c["lim_x"][0]) / c["bev_height"]
    col = (c["lim_y"][1] - y) / cell
    row = (c["lim_x"][1] - x) / cell
    half_w, half_l = w / cell / 2, l / cell / 2  # width across columns, length along rows
    return col - half_w, row - half_l, col + half_w, row + half_l, (-math.degrees(yaw)) % 360


def _cvat_frames(cache: dict[str, Any]) -> Path:
    """Write forward-up, brightened BEV frames for the CVAT task (git-ignored cache)."""
    import cv2

    target = CACHE / "bev_cvat"
    target.mkdir(parents=True, exist_ok=True)
    for fr in cache["frames"]:
        image = cv2.imread(str(CACHE / "bev" / f"{fr['frame']:04d}.png"))
        image = cv2.rotate(image, cv2.ROTATE_180)
        image = cv2.convertScaleAbs(image, alpha=2.2, beta=10)
        cv2.imwrite(str(target / f"{fr['frame']:04d}.png"), image)
    return target


def cvat_export(cache: dict[str, Any]) -> None:
    """Export fused tracks (export_tracks_json) and a CVAT for video 1.1 XML on BEV frames."""
    from xml.sax.saxutils import escape

    from fusion_lab.evaluation import TRACK_GATE_METERS, _partial_assignment
    from fusion_lab.export_cvat import export_tracks_json

    out = BONUS / "cvat"
    out.mkdir(parents=True, exist_ok=True)
    fused = replay(cache, "fused")
    gt_by_frame = {fr["frame"]: fr["valid_gt"] for fr in cache["frames"]}

    # Match *every* track (any lifecycle state) to GT with the 2 m XY gate, so ghost
    # hypotheses that never get confirmed and identity changes are both visible.
    track_results, ghost_frames, owners = [], {}, {}
    for h in fused["history"]:
        snap, gt = h["final"], gt_by_frame[h["frame"]]
        ids = list(snap)
        pos = np.array([snap[t]["x"][:3] for t in ids]).reshape(-1, 3)
        centres = np.array([(b["x"], b["y"], b["z"]) for b in gt]).reshape(-1, 3)
        dist = np.linalg.norm(pos[:, None, :2] - centres[None, :, :2], axis=2)
        pairs = _partial_assignment(dist, np.isfinite(dist) & (dist <= TRACK_GATE_METERS))
        matched = {ids[i]: gt[j]["id"] for i, j in pairs}
        tracks = []
        for tid, t in snap.items():
            gid = matched.get(tid)
            if gid is None:
                ghost_frames.setdefault(tid, []).append(h["frame"])
            else:
                history = owners.setdefault(gid, [])
                if not history or history[-1]["track"] != tid:
                    history.append({"track": tid, "from_frame": h["frame"], "state": t["state"]})
            tracks.append({"id": tid, "state": t["state"], "score": round(t["score"], 4),
                           "x": t["x"][0], "y": t["x"][1], "z": t["x"][2],
                           "vx": t["x"][3], "vy": t["x"][4], "vz": t["x"][5],
                           "length": t["l"], "width": t["w"], "height": t["h"], "yaw": t["yaw"],
                           "matched_gt_id": gid, "ghost": gid is None})
        track_results.append({"frame": h["frame"], "tracks": tracks})
    export_tracks_json(track_results, out / "tracks_fused.json")

    frames = [fr["frame"] for fr in cache["frames"]]
    first, last = frames[0], frames[-1]
    states = "initialized\ntentative\nconfirmed"
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>', "<annotations>", "  <version>1.1</version>",
        "  <meta>", "    <task>", "      <name>day23_bev_fused_tracks</name>",
        f"      <size>{len(frames)}</size>", "      <mode>interpolation</mode>",
        "      <overlap>0</overlap>", "      <flipped>False</flipped>",
        "      <start_frame>0</start_frame>", f"      <stop_frame>{len(frames) - 1}</stop_frame>",
        "      <labels>",
        "        <label><name>fusion_track</name><color>#ff9900</color><type>rectangle</type><attributes>"
        "<attribute><name>track_id</name><mutable>False</mutable><input_type>text</input_type>"
        "<default_value>-1</default_value><values>-1</values></attribute>"
        "<attribute><name>state</name><mutable>True</mutable><input_type>select</input_type>"
        f"<default_value>initialized</default_value><values>{states}</values></attribute>"
        "<attribute><name>ghost</name><mutable>True</mutable><input_type>checkbox</input_type>"
        "<default_value>false</default_value><values>false</values></attribute>"
        "</attributes></label>",
        "        <label><name>gt_vehicle</name><color>#33dd33</color><type>rectangle</type><attributes>"
        "<attribute><name>gt_id</name><mutable>False</mutable><input_type>text</input_type>"
        "<default_value>-</default_value><values>-</values></attribute>"
        "</attributes></label>",
        "      </labels>", "    </task>", "  </meta>",
    ]
    next_id = 0

    def box_xml(frame: int, box: tuple, outside: int, attrs: dict[str, str]) -> None:
        xtl, ytl, xbr, ybr, rot = box
        lines.append(f'    <box frame="{frame - first}" keyframe="1" outside="{outside}" occluded="0" '
                     f'xtl="{xtl:.2f}" ytl="{ytl:.2f}" xbr="{xbr:.2f}" ybr="{ybr:.2f}" '
                     f'rotation="{rot:.2f}" z_order="0">')
        for name, value in attrs.items():
            lines.append(f'      <attribute name="{name}">{escape(value)}</attribute>')
        lines.append("    </box>")

    def emit_track(label: str, boxes: dict[int, tuple], attrs: dict[int, dict[str, str]]) -> None:
        nonlocal next_id
        lines.append(f'  <track id="{next_id}" label="{label}" source="file">')
        next_id += 1
        ordered = sorted(boxes)
        for k, f in enumerate(ordered):
            box_xml(f, boxes[f], 0, attrs[f])
            if (k + 1 == len(ordered) or ordered[k + 1] != f + 1) and f < last:
                box_xml(f + 1, boxes[f], 1, attrs[f])  # hide until the object reappears
        lines.append("  </track>")

    by_track: dict[int, dict[int, tuple]] = {}
    track_attrs: dict[int, dict[int, dict[str, str]]] = {}
    for fr in track_results:
        for t in fr["tracks"]:
            by_track.setdefault(t["id"], {})[fr["frame"]] = _bev_pixel_box(
                cache, t["x"], t["y"], t["length"], t["width"], t["yaw"])
            track_attrs.setdefault(t["id"], {})[fr["frame"]] = {
                "track_id": str(t["id"]), "state": t["state"],
                "ghost": "true" if t["ghost"] else "false"}
    for tid in sorted(by_track):
        emit_track("fusion_track", by_track[tid], track_attrs[tid])
    by_gt: dict[str, dict[int, tuple]] = {}
    for fr in cache["frames"]:
        for b in fr["valid_gt"]:
            by_gt.setdefault(b["id"], {})[fr["frame"]] = _bev_pixel_box(
                cache, b["x"], b["y"], b["l"], b["w"], b["yaw"])
    for gid in sorted(by_gt):
        emit_track("gt_vehicle", by_gt[gid], {f: {"gt_id": gid} for f in by_gt[gid]})
    lines.append("</annotations>")
    (out / "annotations_cvat_video_1.1.xml").write_text("\n".join(lines), encoding="utf-8")
    frames_dir = _cvat_frames(cache)

    lifetimes = {tid: {"frames": [min(b), max(b)],
                       "states": sorted({a["state"] for a in track_attrs[tid].values()})}
                 for tid, b in by_track.items()}
    report = {
        "frames": [first, last],
        "tracks": lifetimes,
        "ghost_tracks": {tid: {"frames": f, "states": lifetimes[tid]["states"]}
                         for tid, f in ghost_frames.items()},
        "confirmed_ghost_frames": sum(1 for fr in track_results for t in fr["tracks"]
                                      if t["ghost"] and t["state"] == "confirmed"),
        "id_changes": {gid: seq for gid, seq in owners.items() if len(seq) > 1},
        "cvat_images": frames_dir.relative_to(ROOT).as_posix(),
        "image_layout": "BEV rotated 180°: forward (+x) up, +y (left) on the left, 608x608 px, "
                        "1 px = 50/608 m",
    }
    (out / "cvat_events.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ("ghost_tracks", "confirmed_ghost_frames", "id_changes")},
                     indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["cache", "verify", "viz", "calib", "cvat"])
    parser.add_argument("--config", type=Path, default=ROOT / "student" / "config" / "paths.yaml")
    args = parser.parse_args()
    if args.stage == "cache":
        build_cache(args.config)
        return
    cache = load_cache()
    {"verify": verify, "viz": visualise, "calib": calibration, "cvat": cvat_export}[args.stage](cache)


if __name__ == "__main__":
    main()
