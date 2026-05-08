#!/usr/bin/env python3
"""
Duck 3D Point Cloud Extraction Pipeline - v4
核心修复（对应MD文档）：
  1. 修复最关键问题：ICP阶段保留桌面，融合完成后再RANSAC删地面
  2. 修复max_frames参数被忽略的bug
  3. 修复RANSAC法向量判定（兼容Y轴朝下坐标系）
  4. putText改为纯英文，避免Linux中文渲染崩溃
  5. 法线估计半径改为 voxel_size*10（MD文档建议）
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
# 工具函数
# ══════════════════════════════════════════════════════
def _extract_field_fast(data, offset, step, n, dtype):
    itemsize = np.dtype(dtype).itemsize
    row_starts = np.arange(n) * step + offset
    indices = (row_starts[:, None] + np.arange(itemsize)[None, :]).ravel()
    return np.frombuffer(data[indices].tobytes(), dtype=dtype)


# ══════════════════════════════════════════════════════
# 1. Bag读取器
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
                if dt < 200.0:
                    paired.append((ic, pp, tc))
                    color_buf.popleft()
                    points_buf.popleft()
                    # ★ 修复：正确使用 max_frames 参数
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
# 2. 相机参数读取
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
            return mat
        for nxt, step_mat in adj.get(curr, []):
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, step_mat @ mat))

    print("[WARN] TF extrinsics not found, using identity")
    return np.eye(4, dtype=np.float64)


# ══════════════════════════════════════════════════════
# 3. SAM2分割器
# ★ 修复：putText改为纯英文，防止Linux中文渲染崩溃
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

        # ★ 修复：提示信息改为英文，避免Linux字体缺失导致崩溃
        print("\nLClick=positive(duck) | RClick=negative(bg/table) | ENTER=confirm | Z=undo")
        canvas = [disp.copy()]

        def redraw():
            img = canvas[0].copy()
            for p in pos:
                cv2.circle(img, (int(p[0]*scale), int(p[1]*scale)), 8, (0, 255, 0), -1)
                cv2.circle(img, (int(p[0]*scale), int(p[1]*scale)), 9, (255, 255, 255), 2)
            for p in neg:
                cv2.circle(img, (int(p[0]*scale), int(p[1]*scale)), 8, (0, 0, 255), -1)
                cv2.circle(img, (int(p[0]*scale), int(p[1]*scale)), 9, (255, 255, 255), 2)
            # ★ 纯英文文字
            cv2.putText(img, f"Pos(G):{len(pos)} Neg(R):{len(neg)} [ENTER=confirm]",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return img

        def on_mouse(event, x, y, flags, param):
            rx, ry = x / scale, y / scale
            if event == cv2.EVENT_LBUTTONDOWN:
                pos.append([rx, ry])
                all_pts.append((rx, ry, True))
            elif event == cv2.EVENT_RBUTTONDOWN:
                neg.append([rx, ry])
                all_pts.append((rx, ry, False))

        cv2.namedWindow("Duck_Annotation", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Duck_Annotation", on_mouse)
        while True:
            cv2.imshow("Duck_Annotation", redraw())
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
                lbs = np.array([1] * len(pos_pts) + [0] * len(neg_pts), dtype=np.int32)
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
# 4. 点云过滤
# ══════════════════════════════════════════════════════
def filter_duck_points(pcd_nx6, rgb_image, mask,
                       color_intrinsics, extrinsics_d2c):
    """返回 Mask 内的点云（纯鸭子，不含桌面）"""
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


def filter_duck_points_with_floor(pcd_nx6, rgb_image, mask,
                                  color_intrinsics, extrinsics_d2c,
                                  floor_margin=0.04):
    """
    ★ 新增函数：返回"鸭子点云 + 周围桌面"用于ICP约束。
    策略：保留 Mask 内的点（鸭子）+ Mask 外但高度接近桌面的点（桌面支撑）。
    floor_margin: 桌面高度向上扩展的容忍量(m)，用于捕捉桌面点
    """
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

    # 鸭子Mask内的点
    in_mask = np.zeros(len(pcd_nx6), dtype=bool)
    idx = np.where(in_bounds)[0]
    in_mask[idx] = mask[v[idx], u[idx]]

    # 桌面点：不在Mask内，且Y坐标（垂直方向）接近桌面高度
    # Orbbec坐标系：Y轴朝下，桌面是一个高Y值的水平面
    duck_pts = pcd_nx6[in_mask, :3]
    if len(duck_pts) > 10:
        # 估算桌面高度：鸭子最低点附近
        duck_y_max = np.percentile(duck_pts[:, 1], 95)  # Y朝下，鸭子底部≈最大Y
        floor_y_min = duck_y_max - floor_margin
        is_floor = (pcd_nx6[:, 1] > floor_y_min) & (~in_mask)
        combined_mask = in_mask | is_floor
    else:
        combined_mask = in_mask

    result = pcd_nx6[combined_mask].copy()
    # 颜色赋值（只对in_bounds的点赋真实颜色）
    in_bounds_combined = combined_mask & in_bounds
    result_idx = np.where(combined_mask)[0]
    ib_idx = np.where(in_bounds_combined)[0]
    if result.shape[1] >= 6 and len(ib_idx) > 0:
        # 找出在combined里哪些位置是in_bounds的
        combined_positions = np.searchsorted(result_idx, ib_idx)
        result[combined_positions, 3:6] = rgb_image[
            v[ib_idx], u[ib_idx]].astype(np.float32)

    duck_only = pcd_nx6[in_mask].copy()
    if duck_only.shape[1] >= 6:
        duck_idx = np.where(in_mask)[0]
        ib_duck = np.where(in_mask & in_bounds)[0]
        for ii in ib_duck:
            duck_only[np.searchsorted(duck_idx, ii), 3:6] = \
                rgb_image[v[ii], u[ii]].astype(np.float32)

    return result, duck_only  # (带桌面的用于ICP, 纯鸭子用于最终保存)


def remove_floor_ransac(pcd_nx6, dist_thresh=0.012):
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

        # ★ 修复：Orbbec Y轴朝下，桌面法向量主要在Y方向
        # 同时兼容Z朝上的情况（abs(normal[2]) > 0.7）
        is_floor = (abs(normal[1]) > 0.7 or abs(normal[2]) > 0.7)

        if is_floor and inlier_ratio > 0.10:
            print(f"[RANSAC] Floor detected: normal={normal.round(3)}, "
                  f"ratio={inlier_ratio:.1%}, removing")
            keep = np.ones(len(pcd_nx6), dtype=bool)
            keep[inliers] = False
            return pcd_nx6[keep]
        else:
            print(f"[RANSAC] No dominant floor found (normal={normal.round(3)}, "
                  f"ratio={inlier_ratio:.1%}), keeping all")
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


# ══════════════════════════════════════════════════════
# 5. Duck-to-Duck ICP（含桌面约束版）
# ══════════════════════════════════════════════════════

def _nx6_to_pcd(nx6, voxel_size):
    if len(nx6) < 30:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(nx6[:, :3].astype(np.float64))
    if nx6.shape[1] >= 6:
        pcd.colors = o3d.utility.Vector3dVector(
            np.clip(nx6[:, 3:6] / 255.0, 0, 1).astype(np.float64))
    pcd = pcd.voxel_down_sample(voxel_size)
    # ★ 修复：法线估计半径改为 voxel_size*10（MD文档明确建议）
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 10, max_nn=30))
    pcd.orient_normals_towards_camera_location(np.array([0., 0., 0.]))
    return pcd


def _check_transform(tf, max_trans_m, max_angle_deg):
    trans_m = float(np.linalg.norm(tf[:3, 3]))
    cos_a = np.clip((np.trace(tf[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(abs(cos_a))))
    ok = (trans_m < max_trans_m) and (angle_deg < max_angle_deg)
    return ok, trans_m, angle_deg


def _duck_icp(src, tgt, voxel_size):
    result = o3d.pipelines.registration.registration_icp(
        src, tgt,
        max_correspondence_distance=voxel_size * 4,
        init=np.eye(4),
        estimation_method=
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=100,
            relative_fitness=1e-6,
            relative_rmse=1e-6))
    return result.transformation, result.fitness


def keep_largest_cluster(pcd_nx6, eps=0.05, min_pts=10):
    if len(pcd_nx6) < min_pts * 2:
        return pcd_nx6
    o3pcd = o3d.geometry.PointCloud()
    o3pcd.points = o3d.utility.Vector3dVector(pcd_nx6[:, :3].astype(np.float64))
    labels = np.array(o3pcd.cluster_dbscan(eps=eps, min_points=min_pts,
                                            print_progress=False))
    if labels.max() < 0:
        print("[DBSCAN] No cluster found, returning original")
        return pcd_nx6
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    largest_label = unique[np.argmax(counts)]
    mask = (labels == largest_label)
    kept = pcd_nx6[mask]
    print(f"[DBSCAN] {len(pcd_nx6)} -> {len(kept)} pts "
          f"({len(unique)} clusters, kept #{largest_label}, "
          f"ratio {len(kept)/len(pcd_nx6):.1%})")
    return kept


def merge_duck_icp(duck_with_floor, duck_only_list,
                   voxel_size_duck=0.001,
                   max_trans_m=0.05,
                   max_angle_deg=8.0,
                   keyframe_interval=5):
    """
    ★ 核心修复：ICP使用带桌面的点云做配准，融合时只累积纯鸭子点云。

    duck_with_floor: List[nx6] 每帧"鸭子+桌面"点云，用于ICP计算变换
    duck_only_list:  List[nx6] 每帧纯鸭子点云，用于累积到最终结果

    原理（对应MD文档3.1解决方案）：
    - 桌面是大平面，提供 Z轴 和 Pitch/Roll 约束，防止圆柱体在垂直方向滑移
    - ICP算完变换后，把这个变换只应用到纯鸭子点云上累积
    - 最后输出不含桌面的纯鸭子点云
    """
    n = len(duck_with_floor)
    assert len(duck_only_list) == n
    if n == 0:
        return np.array([])

    print(f"\n[ICP] Duck-to-Duck ICP with floor constraint, {n} frames")
    print(f"[ICP] Params: max_trans={max_trans_m*100:.1f}cm, "
          f"max_angle={max_angle_deg:.1f}deg, kf_interval={keyframe_interval}")

    icp_voxel = max(voxel_size_duck * 4, 0.004)

    # 关键帧用带桌面的点云
    kf_raw_floor = duck_with_floor[0]
    kf_pcd = _nx6_to_pcd(kf_raw_floor, icp_voxel)
    if kf_pcd is None:
        print("[ICP] Frame 0 invalid")
        return np.array([])

    kf_idx = 0
    kf_world_tf = np.eye(4)

    # 全局点云从第0帧纯鸭子点云开始（不含桌面）
    merged = o3d.geometry.PointCloud()
    d0 = duck_only_list[0]
    merged.points = o3d.utility.Vector3dVector(d0[:, :3].astype(np.float64))
    if d0.shape[1] >= 6:
        merged.colors = o3d.utility.Vector3dVector(
            np.clip(d0[:, 3:6] / 255.0, 0, 1))

    accepted = rejected = 0

    for i in range(1, n):
        src_floor = duck_with_floor[i]
        src_duck  = duck_only_list[i]

        # ICP用带桌面的点云（更多约束）
        src_pcd = _nx6_to_pcd(src_floor, icp_voxel)
        if src_pcd is None or len(src_pcd.points) < 20:
            rejected += 1
            continue

        tf, fitness = _duck_icp(src_pcd, kf_pcd, icp_voxel)

        accepted_this = False

        if fitness < 0.35:
            print(f"[ICP] Frame {i:3d}: fitness={fitness:.3f} too low -> reject")
            rejected += 1
        else:
            ok, t_m, ang = _check_transform(tf, max_trans_m, max_angle_deg)
            if not ok:
                print(f"[ICP] Frame {i:3d}: fitness={fitness:.3f} "
                      f"dt={t_m*100:.1f}cm dR={ang:.1f}deg over limit -> reject")
                rejected += 1
            else:
                accepted_this = True
                accepted += 1

                world_tf = kf_world_tf @ tf

                # ★ 累积纯鸭子点云（不含桌面）
                curr = o3d.geometry.PointCloud()
                curr.points = o3d.utility.Vector3dVector(
                    src_duck[:, :3].astype(np.float64))
                if src_duck.shape[1] >= 6:
                    curr.colors = o3d.utility.Vector3dVector(
                        np.clip(src_duck[:, 3:6] / 255.0, 0, 1))
                curr.transform(world_tf)

                merged += curr
                merged = merged.voxel_down_sample(voxel_size_duck)

                if i % 5 == 0:
                    print(f"[ICP] Frame {i:3d}/{n}: fitness={fitness:.3f}, "
                          f"dt={t_m*100:.1f}cm, dR={ang:.1f}deg, "
                          f"accepted={accepted}, pts={len(merged.points)}")

        # 关键帧更新
        if (i - kf_idx) >= keyframe_interval:
            if accepted_this:
                new_kf = _nx6_to_pcd(src_floor, icp_voxel)  # ★ 关键帧也用带桌面版
                if new_kf is not None:
                    kf_pcd = new_kf
                    kf_world_tf = world_tf.copy()
                    kf_idx = i
                    print(f"[ICP] Frame {i:3d}: keyframe updated (normal)")
            else:
                kf_idx = i
                print(f"[ICP] Frame {i:3d}: keyframe timer reset (frame rejected)")

    ratio = accepted / (n - 1) if n > 1 else 1.0
    print(f"\n[ICP] Done: accepted={accepted}/{n-1} ({ratio*100:.1f}%), "
          f"final pts={len(merged.points)}")

    if ratio < 0.5:
        print(f"[WARN] Accept rate < 50%! Try:")
        print(f"  --max_trans {max_trans_m + 0.02:.2f}")
        print(f"  --max_angle {max_angle_deg + 3:.1f}")
        print(f"  --kf_interval {max(3, keyframe_interval - 2)}")

    pts = np.asarray(merged.points, dtype=np.float32)
    if merged.has_colors():
        cols = (np.asarray(merged.colors) * 255).astype(np.float32)
        return np.hstack([pts, cols])
    return pts


# ══════════════════════════════════════════════════════
# 6. 主流程
# ══════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--output", default="duck_fixed.pcd")
    parser.add_argument("--sam2_checkpoint", required=True)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--skip_frames", type=int, default=1)
    parser.add_argument("--voxel_size", type=float, default=0.001)
    parser.add_argument("--no_ransac", action="store_true")
    parser.add_argument("--save_masks", action="store_true")
    parser.add_argument("--max_trans", type=float, default=0.05)
    parser.add_argument("--max_angle", type=float, default=8.0)
    parser.add_argument("--kf_interval", type=int, default=5)
    parser.add_argument("--dbscan_eps", type=float, default=0.03)
    parser.add_argument("--floor_margin", type=float, default=0.04,
                        help="Floor capture margin above duck bottom (m)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Duck PCD Pipeline - v4 (Floor-constrained ICP)")
    print("=" * 60)

    # ★ 修复：正确传入 args.max_frames
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

    color_intr = get_color_intrinsics(args.bag)
    extrinsics = get_extrinsics_depth_to_color(args.bag)
    print(f"[TF] Extrinsics:\n{np.round(extrinsics, 4)}")

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

    # ★ 核心修复：每帧同时提取"带桌面版"和"纯鸭子版"
    valid_with_floor = []
    valid_duck_only  = []

    print(f"\n[PCD] Extracting per-frame, {len(frames)} frames...")
    for i, (pcd_raw, mask) in enumerate(zip(frames_pcd, masks)):
        if mask is None or mask.sum() < 100:
            continue
        h, w = frames_rgb[i].shape[:2]
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST).astype(bool)

        # 同时获取带桌面版和纯鸭子版
        with_floor, duck_only = filter_duck_points_with_floor(
            pcd_raw, frames_rgb[i], mask, color_intr, extrinsics,
            floor_margin=args.floor_margin)

        if len(duck_only) < 20:
            continue

        with_floor = voxel_down(with_floor, args.voxel_size)
        duck_only  = voxel_down(duck_only, args.voxel_size)

        valid_with_floor.append(with_floor)
        valid_duck_only.append(duck_only)

        if i % 10 == 0:
            print(f"[PCD] Frame {i}: duck={len(duck_only)} pts, "
                  f"with_floor={len(with_floor)} pts, mask_px={mask.sum()}")

    print(f"\n[PCD] Valid frames: {len(valid_duck_only)}")
    if not valid_duck_only:
        print("[ERROR] No duck points extracted!")
        sys.exit(1)

    # ICP融合（带桌面约束，输出纯鸭子）
    if len(valid_duck_only) == 1:
        merged = valid_duck_only[0]
        print("[INFO] Only 1 frame, skipping ICP")
    else:
        merged = merge_duck_icp(
            valid_with_floor,
            valid_duck_only,
            voxel_size_duck=args.voxel_size,
            max_trans_m=args.max_trans,
            max_angle_deg=args.max_angle,
            keyframe_interval=args.kf_interval)

    if len(merged) == 0:
        print("[ERROR] Empty point cloud after ICP")
        sys.exit(1)

    print(f"[PCD] After ICP: {len(merged)} pts")

    # 统计滤波
    merged = statistical_filter(merged, nb=20, std=2.0)

    # ★ 修复：RANSAC现在在ICP之后执行（MD文档核心建议）
    # 虽然 merge_duck_icp 输出已经是纯鸭子，不含桌面，
    # 但若有少量桌面点漏进来，这里再清一遍
    if not args.no_ransac:
        merged = remove_floor_ransac(merged, dist_thresh=0.012)

    # DBSCAN保留最大连通块
    merged = keep_largest_cluster(merged, eps=args.dbscan_eps, min_pts=10)

    # 再次统计滤波
    merged = statistical_filter(merged, nb=15, std=1.5)

    # 最终降采样
    merged = voxel_down(merged, args.voxel_size)
    print(f"[PCD] Final: {len(merged)} pts")

    # 保存
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
    if max(sx, sy, sz) > 0.3:
        print(f"[WARN] Span > 30cm, possible drift. Try:")
        print(f"  --max_trans {args.max_trans - 0.01:.2f}  "
              f"--max_angle {args.max_angle - 1:.1f}")


if __name__ == "__main__":
    main()
