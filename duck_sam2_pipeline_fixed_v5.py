#!/usr/bin/env python3
"""
Duck 3D Point Cloud Extraction Pipeline - v5
核心修复（相对v4）：
  1. Colored ICP 替代 Point-to-Plane ICP → 打破圆柱/杯体旋转对称歧义
  2. 旋转方向一致性检查 → 拦截对称跳变帧
  3. bad_count 重置时使用 last_good_world_tf → 防止漂移污染
  4. RANSAC 地面检测加入底部位置约束 → 防止误删杯底
  5. 其余逻辑（分割/滤波/DBSCAN/关键帧）完整保留
"""

import argparse
import os
import sys
import numpy as np
import cv2
import open3d as o3d
from collections import deque

try:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import rosbag2_py
except ImportError:
    print("[ERROR] ROS2 environment required.")
    sys.exit(1)

SAM2_AVAILABLE = False
try:
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    SAM2_AVAILABLE = True
    print("[INFO] SAM2 found")
except ImportError:
    print("[ERROR] SAM2 not installed")
    sys.exit(1)


# ══════════════════════════════════════════════════════
# Utility
# ══════════════════════════════════════════════════════
def _extract_field_fast(data, offset, step, n, dtype):
    itemsize = np.dtype(dtype).itemsize
    row_starts = np.arange(n) * step + offset
    indices = (row_starts[:, None] + np.arange(itemsize)[None, :]).ravel()
    return np.frombuffer(data[indices].tobytes(), dtype=dtype)


# ══════════════════════════════════════════════════════
# 1. Bag Reader
# ══════════════════════════════════════════════════════
class BagReader:
    def __init__(self, bag_path):
        self.bag_path = bag_path
        sr = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
        cr = rosbag2_py.ConverterOptions("", "")
        self.reader = rosbag2_py.SequentialReader()
        self.reader.open(sr, cr)
        self.type_map = {t.name: t.type
                         for t in self.reader.get_all_topics_and_types()}

    def read_all(self, max_frames=None):
        color_buf, points_buf = deque(), deque()
        paired = []
        while self.reader.has_next():
            topic, raw, ts = self.reader.read_next()
            if topic == "/camera/color/image_raw":
                msg = deserialize_message(raw, get_message(self.type_map[topic]))
                img = self._decode_image(msg)
                if img is not None:
                    color_buf.append((ts, img))
            elif topic == "/camera/depth/points":
                msg = deserialize_message(raw, get_message(self.type_map[topic]))
                pts = self._decode_pointcloud2(msg)
                if pts is not None:
                    points_buf.append((ts, pts))

            while color_buf and points_buf:
                tc, ic = color_buf[0]
                tp, pp = points_buf[0]
                dt = abs(tc - tp) / 1e6
                if dt < 33.0:
                    paired.append((ic, pp, tc))
                    color_buf.popleft()
                    points_buf.popleft()
                    if max_frames and len(paired) >= max_frames:
                        print(f"[BAG] Reached {max_frames} frames")
                        return paired
                elif tc < tp:
                    color_buf.popleft()
                else:
                    points_buf.popleft()

        print(f"[BAG] Total paired frames: {len(paired)}")
        return paired

    @staticmethod
    def _decode_image(msg):
        enc = msg.encoding.lower()
        data = np.frombuffer(msg.data, dtype=np.uint8)
        h, w = msg.height, msg.width
        if enc in ("rgb8", "rgb"):
            return data.reshape(h, w, 3)
        elif enc in ("bgr8", "bgr"):
            return data.reshape(h, w, 3)[:, :, ::-1].copy()
        return None

    @staticmethod
    def _decode_pointcloud2(msg):
        fields = {f.name: f for f in msg.fields}
        if not all(k in fields for k in ("x", "y", "z")):
            return None
        step = msg.point_step
        data = np.frombuffer(msg.data, dtype=np.uint8)
        n = msg.width * msg.height
        result = np.zeros((n, 6), dtype=np.float32)
        for i, name in enumerate(["x", "y", "z"]):
            result[:, i] = _extract_field_fast(
                data, fields[name].offset, step, n, np.float32)
        if "rgb" in fields:
            raw = _extract_field_fast(
                data, fields["rgb"].offset, step, n, np.float32)
            rgb_int = raw.view(np.uint32)
            result[:, 3] = ((rgb_int >> 16) & 0xFF).astype(np.float32)
            result[:, 4] = ((rgb_int >> 8) & 0xFF).astype(np.float32)
            result[:, 5] = (rgb_int & 0xFF).astype(np.float32)
        else:
            result[:, 3:] = 200.0
        valid = np.isfinite(result[:, :3]).all(axis=1) & (result[:, 2] > 0.01)
        return result[valid]


# ══════════════════════════════════════════════════════
# 2. Camera Parameters
# ══════════════════════════════════════════════════════
def get_color_intrinsics(bag_path):
    sr = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    reader = rosbag2_py.SequentialReader()
    reader.open(sr, rosbag2_py.ConverterOptions("", ""))
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    while reader.has_next():
        topic, raw, _ = reader.read_next()
        if topic == "/camera/color/camera_info":
            msg = deserialize_message(raw, get_message(type_map[topic]))
            K = msg.k
            print(f"[INTR] Color camera: fx={K[0]:.2f}, fy={K[4]:.2f}, "
                  f"cx={K[2]:.2f}, cy={K[5]:.2f}, res={msg.width}x{msg.height}")
            return (K[0], K[4], K[2], K[5])
    print("[WARN] Color intrinsics not found, using defaults")
    return (691.33, 691.51, 643.92, 362.12)


def get_extrinsics_depth_to_color(bag_path):
    sr = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    reader = rosbag2_py.SequentialReader()
    reader.open(sr, rosbag2_py.ConverterOptions("", ""))
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    adj = {}

    def quat_to_mat(tx, ty, tz, qx, qy, qz, qw):
        mat = np.eye(4, dtype=np.float64)
        mat[0, 0] = 1 - 2*qy**2 - 2*qz**2
        mat[0, 1] = 2*qx*qy - 2*qz*qw
        mat[0, 2] = 2*qx*qz + 2*qy*qw
        mat[1, 0] = 2*qx*qy + 2*qz*qw
        mat[1, 1] = 1 - 2*qx**2 - 2*qz**2
        mat[1, 2] = 2*qy*qz - 2*qx*qw
        mat[2, 0] = 2*qx*qz - 2*qy*qw
        mat[2, 1] = 2*qy*qz + 2*qx*qw
        mat[2, 2] = 1 - 2*qx**2 - 2*qy**2
        mat[0, 3], mat[1, 3], mat[2, 3] = tx, ty, tz
        return mat

    while reader.has_next():
        topic, raw, _ = reader.read_next()
        if topic == "/tf_static":
            msg = deserialize_message(raw, get_message(type_map[topic]))
            for tf in msg.transforms:
                p = tf.header.frame_id
                c = tf.child_frame_id
                t, r = tf.transform.translation, tf.transform.rotation
                mat = quat_to_mat(t.x, t.y, t.z, r.x, r.y, r.z, r.w)
                adj.setdefault(p, []).append((c, mat))
                adj.setdefault(c, []).append((p, np.linalg.inv(mat)))
            break

    src = "camera_depth_optical_frame"
    dst = "camera_color_optical_frame"
    queue = deque([(src, np.eye(4, dtype=np.float64))])
    visited = {src}
    while queue:
        curr, mat = queue.popleft()
        if curr == dst:
            print(f"[TF] Found extrinsics {src} -> {dst}")
            print(f"[TF] Extrinsics:\n{np.round(mat, 3)}")
            return mat
        for nxt, step_mat in adj.get(curr, []):
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, step_mat @ mat))

    print("[WARN] TF extrinsics not found, using identity")
    return np.eye(4, dtype=np.float64)


# ══════════════════════════════════════════════════════
# 3. SAM2 Segmenter
# ══════════════════════════════════════════════════════
class DuckSegmenter:
    def __init__(self, sam2_checkpoint):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[SEG] Loading SAM2, device: {device}")
        if "tiny" in sam2_checkpoint:
            cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        elif "small" in sam2_checkpoint:
            cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        elif "base_plus" in sam2_checkpoint:
            cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        else:
            cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.predictor = build_sam2_video_predictor(
            cfg, sam2_checkpoint, device=device)

    def get_clicks(self, rgb_image):
        h, w = rgb_image.shape[:2]
        display = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        scale = min(1.0, 1280/w, 720/h)
        disp = cv2.resize(display, (int(w*scale), int(h*scale)))
        pos, neg, all_pts = [], [], []

        print("\nLClick=positive(object) | RClick=negative(bg/table) | ENTER=confirm | Z=undo")
        canvas = [disp.copy()]

        def redraw():
            img = canvas[0].copy()
            for p in pos:
                cv2.circle(img, (int(p[0]*scale), int(p[1]*scale)), 8, (0, 255, 0), -1)
                cv2.circle(img, (int(p[0]*scale), int(p[1]*scale)), 9, (255, 255, 255), 2)
            for p in neg:
                cv2.circle(img, (int(p[0]*scale), int(p[1]*scale)), 8, (0, 0, 255), -1)
                cv2.circle(img, (int(p[0]*scale), int(p[1]*scale)), 9, (255, 255, 255), 2)
            cv2.putText(img, f"Green(pos):{len(pos)} Red(neg):{len(neg)} [ENTER=confirm]",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return img

        def on_mouse(event, x, y, flags, param):
            rx, ry = x / scale, y / scale
            if event == cv2.EVENT_LBUTTONDOWN:
                pos.append([rx, ry]); all_pts.append((rx, ry, True))
            elif event == cv2.EVENT_RBUTTONDOWN:
                neg.append([rx, ry]); all_pts.append((rx, ry, False))

        cv2.namedWindow("Annotate Frame 0", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Annotate Frame 0", on_mouse)
        while True:
            cv2.imshow("Annotate Frame 0", redraw())
            key = cv2.waitKey(30) & 0xFF
            if key in [13, 32] and len(pos) > 0:
                break
            elif key in [ord('z'), ord('Z')] and all_pts:
                _, _, is_pos = all_pts.pop()
                pos.pop() if is_pos else neg.pop()
        cv2.destroyAllWindows()
        return np.array(pos), np.array(neg)

    def segment_all(self, frames_rgb, pos_pts, neg_pts):
        import tempfile, shutil
        from PIL import Image
        tmp = tempfile.mkdtemp(prefix="sam2_")
        try:
            for i, rgb in enumerate(frames_rgb):
                Image.fromarray(rgb).save(os.path.join(tmp, f"{i:05d}.jpg"), quality=95)
            with torch.inference_mode():
                state = self.predictor.init_state(video_path=tmp)
                pts = np.vstack([pos_pts, neg_pts]).astype(np.float32)
                lbs = np.array([1]*len(pos_pts) + [0]*len(neg_pts), dtype=np.int32)
                self.predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=0, obj_id=1,
                    points=pts, labels=lbs)
                masks = [None] * len(frames_rgb)
                for idx, _, logits in self.predictor.propagate_in_video(state):
                    masks[idx] = (logits[0, 0] > 0.0).cpu().numpy()
                    if idx % 10 == 0:
                        cov = masks[idx].sum() / masks[idx].size * 100
                        print(f"[SAM2] frame {idx}: mask coverage {cov:.1f}%")
            h, w = frames_rgb[0].shape[:2]
            return [m if m is not None else np.zeros((h, w), bool) for m in masks]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ══════════════════════════════════════════════════════
# 4. Point Cloud Filters
# ══════════════════════════════════════════════════════
def filter_duck_points(pcd_nx6, rgb_image, mask, color_intrinsics, extrinsics_d2c):
    fx, fy, cx, cy = color_intrinsics
    pts = pcd_nx6[:, :3].astype(np.float64)
    ones = np.ones((len(pts), 1))
    pts_h = np.hstack([pts, ones])
    pts_color = (extrinsics_d2c @ pts_h.T).T
    Xc, Yc, Zc = pts_color[:, 0], pts_color[:, 1], pts_color[:, 2]
    valid_z = Zc > 0.05
    u = np.zeros(len(pcd_nx6), dtype=np.int32)
    v = np.zeros(len(pcd_nx6), dtype=np.int32)
    u[valid_z] = (Xc[valid_z] / Zc[valid_z] * fx + cx).astype(np.int32)
    v[valid_z] = (Yc[valid_z] / Zc[valid_z] * fy + cy).astype(np.int32)
    H, W = mask.shape
    in_bounds = valid_z & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    in_mask = np.zeros(len(pcd_nx6), dtype=bool)
    idx = np.where(in_bounds)[0]
    in_mask[idx] = mask[v[idx], u[idx]]
    result = pcd_nx6[in_mask].copy()
    if result.shape[1] >= 6:
        result[:, 3:6] = rgb_image[v[in_mask], u[in_mask]].astype(np.float32)
    return result


def remove_floor_ransac(pcd_nx6, dist_thresh=0.012):
    """
    修复版：增加底部位置约束，防止误删杯底。
    只有当检测到的平面法向量接近竖直(|ny|>0.7)
    且该平面确实在点云底部1/3区域时，才删除。
    """
    if len(pcd_nx6) < 50:
        return pcd_nx6
    o3pcd = o3d.geometry.PointCloud()
    o3pcd.points = o3d.utility.Vector3dVector(pcd_nx6[:, :3].astype(np.float64))
    try:
        plane_model, inliers = o3pcd.segment_plane(
            distance_threshold=dist_thresh, ransac_n=3, num_iterations=500)
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal /= np.linalg.norm(normal)
        inlier_ratio = len(inliers) / len(pcd_nx6)

        # 更严格的法向量约束（原来是 |ny|>0.6 or |nz|>0.6，容易误触发）
        is_floor_normal = (abs(normal[1]) > 0.70)

        if is_floor_normal and inlier_ratio > 0.15:
            # 额外检查：地面应该在点云 Y 轴底部 1/3
            inlier_y = pcd_nx6[inliers, 1]
            y_min = pcd_nx6[:, 1].min()
            y_max = pcd_nx6[:, 1].max()
            y_thresh = y_min + (y_max - y_min) * 0.33
            is_bottom = inlier_y.mean() < y_thresh

            if is_bottom:
                print(f"[RANSAC] Floor detected: normal={normal.round(3)}, "
                      f"ratio={inlier_ratio:.1%}, removing")
                keep = np.ones(len(pcd_nx6), dtype=bool)
                keep[inliers] = False
                return pcd_nx6[keep]
            else:
                print(f"[RANSAC] Plane found but not at bottom (mean_y={inlier_y.mean():.3f}), "
                      f"skipping removal")
        else:
            print(f"[RANSAC] No clear floor detected (normal={normal.round(3)}, "
                  f"ratio={inlier_ratio:.1%}), keeping")
    except Exception as e:
        print(f"[RANSAC] Failed: {e}")
    return pcd_nx6


def statistical_filter(pcd_nx6, nb=20, std=1.5):
    if len(pcd_nx6) < nb + 1:
        return pcd_nx6
    o3pcd = o3d.geometry.PointCloud()
    o3pcd.points = o3d.utility.Vector3dVector(pcd_nx6[:, :3].astype(np.float64))
    _, ind = o3pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)
    print(f"[FILTER] Statistical filter: {len(pcd_nx6)} -> {len(ind)} pts")
    return pcd_nx6[ind]


def voxel_down(pcd_nx6, voxel_size):
    if len(pcd_nx6) < 10:
        return pcd_nx6
    o3pcd = o3d.geometry.PointCloud()
    o3pcd.points = o3d.utility.Vector3dVector(pcd_nx6[:, :3].astype(np.float64))
    if pcd_nx6.shape[1] >= 6:
        o3pcd.colors = o3d.utility.Vector3dVector(
            np.clip(pcd_nx6[:, 3:6] / 255.0, 0, 1).astype(np.float64))
    down = o3pcd.voxel_down_sample(voxel_size)
    pts = np.asarray(down.points, dtype=np.float32)
    if down.has_colors():
        cols = (np.asarray(down.colors) * 255).astype(np.float32)
        return np.hstack([pts, cols])
    return pts


def keep_largest_cluster(pcd_nx6, eps=0.05, min_pts=10):
    if len(pcd_nx6) < min_pts * 2:
        return pcd_nx6
    o3pcd = o3d.geometry.PointCloud()
    o3pcd.points = o3d.utility.Vector3dVector(pcd_nx6[:, :3].astype(np.float64))
    labels = np.array(o3pcd.cluster_dbscan(
        eps=eps, min_points=min_pts, print_progress=False))
    if labels.max() < 0:
        print("[DBSCAN] No clusters found, returning original")
        return pcd_nx6
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    largest_label = unique[np.argmax(counts)]
    mask = (labels == largest_label)
    kept = pcd_nx6[mask]
    print(f"[DBSCAN] {len(pcd_nx6)} -> {len(kept)} pts "
          f"({len(unique)} clusters, kept #{largest_label}, ratio {len(kept)/len(pcd_nx6):.1%})")
    return kept


# ══════════════════════════════════════════════════════
# 5. v5 Core: Colored ICP + Rotation Consistency Check
# ══════════════════════════════════════════════════════

def _nx6_to_pcd(nx6, voxel_size):
    """
    nx6 → open3d PointCloud，保留颜色（Colored ICP 必须）。
    """
    if len(nx6) < 30:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(nx6[:, :3].astype(np.float64))
    if nx6.shape[1] >= 6:
        colors = np.clip(nx6[:, 3:6] / 255.0, 0, 1).astype(np.float64)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5, max_nn=30))
    pcd.orient_normals_towards_camera_location(np.array([0., 0., 0.]))
    return pcd


def _check_transform(tf, max_trans_m, max_angle_deg):
    trans_m = float(np.linalg.norm(tf[:3, 3]))
    cos_a = np.clip((np.trace(tf[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(cos_a)))
    ok = (trans_m < max_trans_m) and (angle_deg < max_angle_deg)
    return ok, trans_m, angle_deg


def _check_rotation_consistency(tf_curr, tf_prev, frob_thresh=1.5):
    """
    【v5新增】检查当前帧变换与上一帧变换旋转方向是否一致。
    圆柱对称跳变时，旋转矩阵会突然翻转，Frobenius 范数差会超过阈值。
    frob_thresh=1.5 为经验值（旋转矩阵元素范围[-1,1]，最大差值约3.46）。
    """
    R_curr = tf_curr[:3, :3]
    R_prev = tf_prev[:3, :3]
    diff = np.linalg.norm(R_curr - R_prev, 'fro')
    return diff < frob_thresh, diff


def _duck_icp_colored(src, tgt, voxel_size):
    """
    【v5核心】Colored ICP：同时优化几何距离 + RGB 颜色差异。
    对圆柱/杯体的旋转对称歧义有显著抑制效果，
    因为即使两帧几何距离相同，颜色分布会区分旋转方向。

    若 src/tgt 颜色缺失则自动回退到 Point-to-Plane ICP。
    """
    use_colored = src.has_colors() and tgt.has_colors()

    if use_colored:
        try:
            result = o3d.pipelines.registration.registration_colored_icp(
                src, tgt,
                max_correspondence_distance=voxel_size * 4,
                init=np.eye(4),
                estimation_method=(
                    o3d.pipelines.registration
                    .TransformationEstimationForColoredICP(lambda_geometric=0.968)),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=100,
                    relative_fitness=1e-6,
                    relative_rmse=1e-6))
            return result.transformation, result.fitness
        except Exception as e:
            print(f"[ICP] Colored ICP failed ({e}), falling back to P2Plane")

    # Fallback: Point-to-Plane
    result = o3d.pipelines.registration.registration_icp(
        src, tgt,
        max_correspondence_distance=voxel_size * 4,
        init=np.eye(4),
        estimation_method=(
            o3d.pipelines.registration
            .TransformationEstimationPointToPlane()),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=100,
            relative_fitness=1e-6,
            relative_rmse=1e-6))
    return result.transformation, result.fitness


def merge_duck_icp_v5(duck_arrays, voxel_size_duck=0.001,
                      max_trans_m=0.05, max_angle_deg=8.0,
                      keyframe_interval=8):
    """
    v5 多帧 Duck-to-Duck Colored ICP 融合。

    相对 v4 的核心变化：
    ① Colored ICP 替代 Point-to-Plane → 打破圆柱对称
    ② 旋转一致性检查 → 拦截对称跳变帧
    ③ bad_count 重置时使用 last_good_world_tf（而非当前位置）
       → 防止漂移帧位置污染后续配准
    """
    if not duck_arrays:
        return np.array([])

    n = len(duck_arrays)
    print(f"\n[ICP] Duck-to-Duck Colored ICP, {n} frames")
    print(f"[ICP] Params: max_trans={max_trans_m*100:.1f}cm, "
          f"max_angle={max_angle_deg:.1f}deg, kf_interval={keyframe_interval}")

    icp_voxel = max(voxel_size_duck * 4, 0.004)

    # ── 初始化 ──
    kf_raw = duck_arrays[0]
    kf_pcd = _nx6_to_pcd(kf_raw, icp_voxel)
    if kf_pcd is None:
        print("[ICP] Frame 0 invalid")
        return np.array([])

    kf_world_tf = np.eye(4)

    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(kf_raw[:, :3].astype(np.float64))
    if kf_raw.shape[1] >= 6:
        merged.colors = o3d.utility.Vector3dVector(
            np.clip(kf_raw[:, 3:6] / 255.0, 0, 1))

    bad_count = 0
    bad_reset_thresh = 5
    frames_since_kf = 0
    accepted = 0

    # 【v5新增】记录上一帧被接受时的变换（用于一致性检查）和最后成功的世界坐标
    prev_accepted_tf = np.eye(4)
    last_good_world_tf = np.eye(4)

    for i in range(1, n):
        src_raw = duck_arrays[i]
        src_pcd = _nx6_to_pcd(src_raw, icp_voxel)
        if src_pcd is None or len(src_pcd.points) < 20:
            bad_count += 1
            continue

        # ── Colored ICP：当前帧 → 关键帧 ──
        tf, fitness = _duck_icp_colored(src_pcd, kf_pcd, icp_voxel)

        # ── 校验 fitness ──
        if fitness < 0.40:
            print(f"[ICP] Frame {i:3d}: fitness={fitness:.3f} too low -> reject")
            bad_count += 1
            continue

        # ── 校验变换幅度 ──
        ok, t_m, ang = _check_transform(tf, max_trans_m, max_angle_deg)
        if not ok:
            print(f"[ICP] Frame {i:3d}: fitness={fitness:.3f} "
                  f"dt={t_m*100:.1f}cm dR={ang:.1f}deg over limit -> reject")
            bad_count += 1
            # ── bad_count 重置处理 ──
            if bad_count >= bad_reset_thresh:
                new_kf = _nx6_to_pcd(src_raw, icp_voxel)
                if new_kf is not None:
                    kf_pcd = new_kf
                    # 【v5修复】使用最后一次成功的 world_tf，而不是当前帧位置
                    # 这样关键帧在世界坐标中仍锚定在已知正确位置，不会漂移
                    kf_world_tf = last_good_world_tf.copy()
                    frames_since_kf = 0
                    bad_count = 0
                    print(f"[ICP] Frame {i:3d}: keyframe timer reset (frame rejected)")
            continue

        # ── 【v5新增】旋转方向一致性检查 ──
        consistent, frob_diff = _check_rotation_consistency(tf, prev_accepted_tf)
        if not consistent:
            print(f"[ICP] Frame {i:3d}: rotation jump detected "
                  f"(Frobenius diff={frob_diff:.3f}) -> reject (cylinder symmetry guard)")
            bad_count += 1
            if bad_count >= bad_reset_thresh:
                new_kf = _nx6_to_pcd(src_raw, icp_voxel)
                if new_kf is not None:
                    kf_pcd = new_kf
                    kf_world_tf = last_good_world_tf.copy()
                    frames_since_kf = 0
                    bad_count = 0
                    print(f"[ICP] Frame {i:3d}: keyframe reset after rotation jump")
            continue

        # ── 接受此帧 ──
        bad_count = 0
        accepted += 1
        frames_since_kf += 1
        prev_accepted_tf = tf.copy()

        world_tf = kf_world_tf @ tf
        last_good_world_tf = world_tf.copy()  # 【v5】记录最后成功位置

        # 变换到世界坐标，融入全局点云
        curr = o3d.geometry.PointCloud()
        curr.points = o3d.utility.Vector3dVector(src_raw[:, :3].astype(np.float64))
        if src_raw.shape[1] >= 6:
            curr.colors = o3d.utility.Vector3dVector(
                np.clip(src_raw[:, 3:6] / 255.0, 0, 1))
        curr.transform(world_tf)

        merged += curr
        merged = merged.voxel_down_sample(voxel_size_duck)

        if i % 10 == 0 or i == n - 1:
            print(f"[ICP] Frame {i:3d}/{n}: fitness={fitness:.3f}, "
                  f"dt={t_m*100:.1f}cm, dR={ang:.1f}deg, "
                  f"accepted={accepted}, pts={len(merged.points)}")

        # ── 更新关键帧 ──
        if frames_since_kf >= keyframe_interval:
            new_kf = _nx6_to_pcd(src_raw, icp_voxel)
            if new_kf is not None:
                kf_pcd = new_kf
                kf_world_tf = world_tf.copy()
                frames_since_kf = 0
                print(f"[ICP] Frame {i:3d}: keyframe updated (normal)")

        # bad_count 重置检查（放在最后，已通过的帧不触发）
        if bad_count >= bad_reset_thresh:
            new_kf = _nx6_to_pcd(src_raw, icp_voxel)
            if new_kf is not None:
                kf_pcd = new_kf
                kf_world_tf = last_good_world_tf.copy()
                frames_since_kf = 0
                bad_count = 0
                print(f"[ICP] Frame {i:3d}: keyframe reset (bad_count threshold)")

    print(f"\n[ICP] Done: accepted={accepted}/{n-1} ({accepted/(n-1)*100:.1f}%), "
          f"final pts={len(merged.points)}")

    pts = np.asarray(merged.points, dtype=np.float32)
    if merged.has_colors():
        cols = (np.asarray(merged.colors) * 255).astype(np.float32)
        return np.hstack([pts, cols])
    return pts


# ══════════════════════════════════════════════════════
# 6. Main
# ══════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Mug/Object 3D PCD Pipeline v5 - Colored ICP + Rotation Guard")
    parser.add_argument("--bag", required=True)
    parser.add_argument("--output", default="output_v5.pcd")
    parser.add_argument("--sam2_checkpoint", required=True)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--skip_frames", type=int, default=1)
    parser.add_argument("--voxel_size", type=float, default=0.001)
    parser.add_argument("--no_ransac", action="store_true")
    parser.add_argument("--save_masks", action="store_true")
    parser.add_argument("--max_trans", type=float, default=0.05,
                        help="Max translation per frame (m), default=5cm")
    parser.add_argument("--max_angle", type=float, default=8.0,
                        help="Max rotation per frame (deg), default=8deg")
    parser.add_argument("--kf_interval", type=int, default=8,
                        help="Keyframe update interval (frames), default=8")
    parser.add_argument("--dbscan_eps", type=float, default=0.05,
                        help="DBSCAN neighborhood radius (m), default=5cm")
    parser.add_argument("--rot_frob_thresh", type=float, default=1.5,
                        help="Frobenius norm threshold for rotation consistency check, default=1.5")
    args = parser.parse_args()

    print("=" * 60)
    print("  Duck/Mug PCD Pipeline - v5 (Colored ICP + Rotation Guard)")
    print("=" * 60)

    # ── 读数据 ──
    reader = BagReader(args.bag)
    frames = reader.read_all(max_frames=args.max_frames)
    if not frames:
        print("[ERROR] No paired frames found")
        sys.exit(1)
    if args.skip_frames > 1:
        frames = frames[::args.skip_frames]
        print(f"[INFO] After skip: {len(frames)} frames")

    frames_rgb = [f[0] for f in frames]
    frames_pcd = [f[1] for f in frames]
    print(f"[INFO] RGB: {frames_rgb[0].shape}, first frame pts: {len(frames_pcd[0])}")

    # ── 相机参数 ──
    color_intr = get_color_intrinsics(args.bag)
    extrinsics = get_extrinsics_depth_to_color(args.bag)

    # ── SAM2 分割 ──
    seg = DuckSegmenter(args.sam2_checkpoint)
    pos_pts, neg_pts = seg.get_clicks(frames_rgb[0])
    print(f"[SEG] Positive: {len(pos_pts)}, Negative: {len(neg_pts)}")
    masks = seg.segment_all(frames_rgb, pos_pts, neg_pts)

    if args.save_masks:
        os.makedirs("masks_preview", exist_ok=True)
        for i in range(min(30, len(masks))):
            prev = frames_rgb[i].copy()
            prev[masks[i]] = (prev[masks[i]] * 0.4 +
                              np.array([0, 255, 0]) * 0.6).astype(np.uint8)
            cv2.imwrite(f"masks_preview/frame_{i:04d}.jpg",
                        cv2.cvtColor(prev, cv2.COLOR_RGB2BGR))
        print("[INFO] Mask previews saved to masks_preview/")

    # ── 逐帧提取目标点云 ──
    valid_ducks = []
    print(f"\n[PCD] Extracting per-frame, {len(frames)} frames...")
    for i, (pcd_raw, mask) in enumerate(zip(frames_pcd, masks)):
        if mask is None or mask.sum() < 100:
            continue
        h, w = frames_rgb[i].shape[:2]
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST).astype(bool)

        duck = filter_duck_points(
            pcd_raw, frames_rgb[i], mask, color_intr, extrinsics)
        if len(duck) < 20:
            continue

        duck = voxel_down(duck, args.voxel_size)
        valid_ducks.append(duck)

        if i % 10 == 0:
            with_floor_pts = len(pcd_raw)
            print(f"[PCD] Frame {i}: duck={len(duck)} pts, "
                  f"with_floor={with_floor_pts} pts, mask_px={mask.sum()}")

    print(f"\n[PCD] Valid frames: {len(valid_ducks)}")
    if not valid_ducks:
        print("[ERROR] No object point cloud extracted!")
        sys.exit(1)

    # ── v5 Colored ICP 多帧融合 ──
    if len(valid_ducks) == 1:
        merged = valid_ducks[0]
        print("[INFO] Only one frame, skipping ICP")
    else:
        merged = merge_duck_icp_v5(
            valid_ducks,
            voxel_size_duck=args.voxel_size,
            max_trans_m=args.max_trans,
            max_angle_deg=args.max_angle,
            keyframe_interval=args.kf_interval)

    if len(merged) == 0:
        print("[ERROR] Point cloud empty after ICP")
        sys.exit(1)

    print(f"[PCD] After ICP: {len(merged)} pts")

    # ── 统计滤波 ──
    merged = statistical_filter(merged, nb=20, std=2.0)

    # ── RANSAC 去地面（修复版：带底部位置约束）──
    if not args.no_ransac:
        merged = remove_floor_ransac(merged, dist_thresh=0.012)

    # ── DBSCAN 保留最大连通块 ──
    merged = keep_largest_cluster(merged, eps=args.dbscan_eps, min_pts=10)

    # ── 再次统计滤波 ──
    merged = statistical_filter(merged, nb=15, std=1.5)

    # ── 最终降采样 ──
    merged = voxel_down(merged, args.voxel_size)
    print(f"[PCD] Final: {len(merged)} pts")

    # ── 保存 ──
    final = o3d.geometry.PointCloud()
    final.points = o3d.utility.Vector3dVector(merged[:, :3].astype(np.float64))
    if merged.shape[1] >= 6:
        final.colors = o3d.utility.Vector3dVector(
            np.clip(merged[:, 3:6] / 255.0, 0, 1).astype(np.float64))
    final.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    final.orient_normals_consistent_tangent_plane(15)
    o3d.io.write_point_cloud(args.output, final)

    pts = np.asarray(final.points)
    sx = pts[:, 0].max() - pts[:, 0].min()
    sy = pts[:, 1].max() - pts[:, 1].min()
    sz = pts[:, 2].max() - pts[:, 2].min()
    print(f"\n[OK] Saved: {args.output}")
    print(f"   Total pts: {len(pts)}")
    print(f"   X: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}] m  (span {sx:.3f}m)")
    print(f"   Y: [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}] m  (span {sy:.3f}m)")
    print(f"   Z: [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}] m  (span {sz:.3f}m)")
    if max(sx, sy, sz) > 0.5:
        print("⚠️  Span > 50cm, possible drift remaining.")
        print(f"   Try: --max_trans {args.max_trans*0.6:.3f} or "
              f"--max_angle {args.max_angle*0.7:.1f} or "
              f"--rot_frob_thresh {args.rot_frob_thresh*0.8:.2f}")


if __name__ == "__main__":
    main()
