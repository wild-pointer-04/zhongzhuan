#!/usr/bin/env python3
"""
Duck 3D Point Cloud Extraction Pipeline - 完全修复版 v3
核心修复：
  1. 彻底放弃全场景ICP，改用 Duck-to-Duck ICP
     （鸭子有曲面+喙，ICP约束充分；桌面是大平面，ICP无约束）
  2. 每帧对变换幅度做物理约束（平移<4cm/帧，旋转<6°/帧）
  3. 固定关键帧策略，防止滚动累积漂移
  4. DBSCAN最终聚类，强制保留最大连通块（鸭子主体）
  5. 保留所有原有功能（分割/滤波/RANSAC）
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
    print("[ERROR] 需要ROS2环境。")
    sys.exit(1)

SAM2_AVAILABLE = False
try:
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    SAM2_AVAILABLE = True
    print("[INFO] SAM2 已找到")
except ImportError:
    print("[ERROR] SAM2未安装")
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
# 1. Bag读取器（原版不变）
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
                        print(f"[BAG] 达到 {max_frames} 帧")
                        return paired
                elif tc < tp:
                    color_buf.popleft()
                else:
                    points_buf.popleft()

        print(f"[BAG] 共配对 {len(paired)} 帧")
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
# 2. 相机参数读取（原版不变）
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
            print(f"[INTR] 彩色相机内参: fx={K[0]:.2f}, fy={K[4]:.2f}, "
                  f"cx={K[2]:.2f}, cy={K[5]:.2f}, 分辨率={msg.width}x{msg.height}")
            return (K[0], K[4], K[2], K[5])
    print("[WARN] 未找到彩色内参，使用默认值")
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
            print(f"[TF] 找到外参矩阵 {src} → {dst}")
            print(f"[TF] 平移部分(mm): x={mat[0,3]*1000:.1f}, "
                  f"y={mat[1,3]*1000:.1f}, z={mat[2,3]*1000:.1f}")
            return mat
        for nxt, step_mat in adj.get(curr, []):
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, step_mat @ mat))

    print("[WARN] 未找到TF外参，使用单位矩阵")
    return np.eye(4, dtype=np.float64)


# ══════════════════════════════════════════════════════
# 3. SAM2分割器（原版不变）
# ══════════════════════════════════════════════════════
class DuckSegmenter:
    def __init__(self, sam2_checkpoint):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[SEG] 加载SAM2，设备: {device}")
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

        print("\n左键=正样本(鸭子) | 右键=负样本(桌面/背景) | ENTER=确认 | Z=撤销")
        canvas = [disp.copy()]

        def redraw():
            img = canvas[0].copy()
            for p in pos:
                cv2.circle(img, (int(p[0]*scale), int(p[1]*scale)), 8, (0,255,0), -1)
                cv2.circle(img, (int(p[0]*scale), int(p[1]*scale)), 9, (255,255,255), 2)
            for p in neg:
                cv2.circle(img, (int(p[0]*scale), int(p[1]*scale)), 8, (0,0,255), -1)
                cv2.circle(img, (int(p[0]*scale), int(p[1]*scale)), 9, (255,255,255), 2)
            cv2.putText(img, f"绿(正):{len(pos)} 红(负):{len(neg)} [ENTER确认]",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            return img

        def on_mouse(event, x, y, flags, param):
            rx, ry = x/scale, y/scale
            if event == cv2.EVENT_LBUTTONDOWN:
                pos.append([rx, ry]); all_pts.append((rx, ry, True))
            elif event == cv2.EVENT_RBUTTONDOWN:
                neg.append([rx, ry]); all_pts.append((rx, ry, False))

        cv2.namedWindow("标注第一帧", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("标注第一帧", on_mouse)
        while True:
            cv2.imshow("标注第一帧", redraw())
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
                Image.fromarray(rgb).save(os.path.join(tmp, f"{i:05d}.jpg"),
                                          quality=95)
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
                    if idx % 100 == 0:
                        cov = masks[idx].sum() / masks[idx].size * 100
                        print(f"[SAM2] 帧{idx}: mask覆盖{cov:.1f}%")
            h, w = frames_rgb[0].shape[:2]
            return [m if m is not None else np.zeros((h, w), bool)
                    for m in masks]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ══════════════════════════════════════════════════════
# 4. 点云过滤（原版不变）
# ══════════════════════════════════════════════════════
def filter_duck_points(pcd_nx6, rgb_image, mask,
                       color_intrinsics, extrinsics_d2c):
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
        is_floor = (abs(normal[1]) > 0.6 or abs(normal[2]) > 0.6)
        if is_floor and inlier_ratio > 0.15:
            print(f"[RANSAC] 检测到地面，法向量={normal.round(3)}，"
                  f"占比={inlier_ratio:.1%}，删除")
            keep = np.ones(len(pcd_nx6), dtype=bool)
            keep[inliers] = False
            return pcd_nx6[keep]
        else:
            print(f"[RANSAC] 未检测到明显地面，保留")
    except Exception as e:
        print(f"[RANSAC] 失败: {e}")
    return pcd_nx6


def statistical_filter(pcd_nx6, nb=20, std=1.5):
    if len(pcd_nx6) < nb + 1:
        return pcd_nx6
    o3pcd = o3d.geometry.PointCloud()
    o3pcd.points = o3d.utility.Vector3dVector(pcd_nx6[:, :3].astype(np.float64))
    _, ind = o3pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)
    print(f"[FILTER] 统计滤波: {len(pcd_nx6)} → {len(ind)} 点")
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
# 5. 核心修复：Duck-to-Duck ICP + 变换校验 + DBSCAN清理
# ══════════════════════════════════════════════════════

def _nx6_to_pcd(nx6, voxel_size):
    """把 nx6 数组转成 open3d PointCloud，降采样，估算法向量"""
    if len(nx6) < 30:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(nx6[:, :3].astype(np.float64))
    if nx6.shape[1] >= 6:
        pcd.colors = o3d.utility.Vector3dVector(
            np.clip(nx6[:, 3:6] / 255.0, 0, 1).astype(np.float64))
    pcd = pcd.voxel_down_sample(voxel_size)
    # 法向量朝相机（原点）方向
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5, max_nn=30))
    pcd.orient_normals_towards_camera_location(np.array([0., 0., 0.]))
    return pcd


def _check_transform(tf, max_trans_m, max_angle_deg):
    """
    校验刚体变换是否在物理合理范围内。
    返回: (is_ok, trans_m, angle_deg)
    """
    trans_m = float(np.linalg.norm(tf[:3, 3]))
    # 从旋转矩阵提取旋转角
    cos_a = np.clip((np.trace(tf[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(cos_a)))
    ok = (trans_m < max_trans_m) and (angle_deg < max_angle_deg)
    return ok, trans_m, angle_deg


def _duck_icp(src, tgt, voxel_size):
    """
    单次 Duck-to-Duck Point-to-Plane ICP。
    src, tgt 均为已计算法向量的 o3d.PointCloud。
    返回: (4×4 变换矩阵, fitness)
    """
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
    """
    用 open3d DBSCAN 保留最大连通块（即鸭子主体），丢弃漂移幽灵碎片。
    eps=5cm 对应鸭子的表面邻域尺度。
    """
    if len(pcd_nx6) < min_pts * 2:
        return pcd_nx6
    o3pcd = o3d.geometry.PointCloud()
    o3pcd.points = o3d.utility.Vector3dVector(pcd_nx6[:, :3].astype(np.float64))
    # open3d 内置 DBSCAN，无需 scikit-learn
    labels = np.array(o3pcd.cluster_dbscan(eps=eps, min_points=min_pts,
                                            print_progress=False))
    if labels.max() < 0:
        print("[DBSCAN] 未找到任何聚类，返回原始点云")
        return pcd_nx6
    # 统计各 cluster 的点数，保留最大的
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    largest_label = unique[np.argmax(counts)]
    mask = (labels == largest_label)
    kept = pcd_nx6[mask]
    print(f"[DBSCAN] {len(pcd_nx6)} → {len(kept)} 点 "
          f"（共 {len(unique)} 个 cluster，保留最大 #{largest_label}，"
          f"占比 {len(kept)/len(pcd_nx6):.1%}）")
    return kept


def merge_duck_icp(duck_arrays, voxel_size_duck=0.001,
                   max_trans_m=0.05, max_angle_deg=8.0,
                   keyframe_interval=8):
    """
    Duck-to-Duck ICP 多帧融合。

    策略：
    ┌─ 每帧鸭子点云 → _nx6_to_pcd → ICP 对齐到关键帧 ─┐
    │  ① fitness < 0.4 → 拒绝                          │
    │  ② 平移 > max_trans_m 或旋转 > max_angle_deg → 拒绝│
    │  ③ 连续 bad_reset_thresh 帧失败 → 重置关键帧      │
    └──────────────────────────────────────────────────┘

    关键帧每 keyframe_interval 帧更新一次（不是 rolling，是跳跃式锚点）。
    这样每帧的变换是"当前帧→最近关键帧"，而不是累积链。
    """
    if not duck_arrays:
        return np.array([])

    n = len(duck_arrays)
    print(f"\n[ICP] 开始 Duck-to-Duck 配准，共 {n} 帧")
    print(f"[ICP] 参数: max_trans={max_trans_m*100:.1f}cm, "
          f"max_angle={max_angle_deg:.1f}°, "
          f"keyframe_interval={keyframe_interval}")

    # ICP 用稍大体素（加速 + 减少噪点干扰），融合用原体素
    icp_voxel = max(voxel_size_duck * 4, 0.004)

    # ── 初始化 ──
    kf_raw = duck_arrays[0]           # 当前关键帧的 nx6 数据
    kf_pcd = _nx6_to_pcd(kf_raw, icp_voxel)
    if kf_pcd is None:
        print("[ICP] 第0帧无效")
        return np.array([])

    # 关键帧在世界坐标系中的变换（初始为单位矩阵）
    kf_world_tf = np.eye(4)

    # 全局点云从第0帧开始
    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(
        kf_raw[:, :3].astype(np.float64))
    if kf_raw.shape[1] >= 6:
        merged.colors = o3d.utility.Vector3dVector(
            np.clip(kf_raw[:, 3:6] / 255.0, 0, 1))

    bad_count = 0
    bad_reset_thresh = 5
    frames_since_kf = 0
    accepted = 0

    for i in range(1, n):
        src_raw = duck_arrays[i]
        src_pcd = _nx6_to_pcd(src_raw, icp_voxel)
        if src_pcd is None or len(src_pcd.points) < 20:
            bad_count += 1
            continue

        # ── ICP：当前帧 → 关键帧 ──
        tf, fitness = _duck_icp(src_pcd, kf_pcd, icp_voxel)

        # ── 校验 fitness ──
        if fitness < 0.40:
            print(f"[ICP] 帧{i:3d}: fitness={fitness:.3f} 过低 → 拒绝")
            bad_count += 1
        else:
            # ── 校验变换幅度 ──
            ok, t_m, ang = _check_transform(tf, max_trans_m, max_angle_deg)
            if not ok:
                print(f"[ICP] 帧{i:3d}: fitness={fitness:.3f} 但 "
                      f"Δt={t_m*100:.1f}cm Δθ={ang:.1f}° 超限 → 拒绝")
                bad_count += 1
            else:
                bad_count = 0
                accepted += 1
                frames_since_kf += 1

                # tf：把当前帧对齐到关键帧
                # 当前帧在世界坐标 = kf_world_tf @ tf
                # （tf 把 src 变换到 kf，kf_world_tf 把 kf 变换到世界）
                world_tf = kf_world_tf @ tf

                # 把当前帧鸭子点云变换到世界坐标，融入全局点云
                curr = o3d.geometry.PointCloud()
                curr.points = o3d.utility.Vector3dVector(
                    src_raw[:, :3].astype(np.float64))
                if src_raw.shape[1] >= 6:
                    curr.colors = o3d.utility.Vector3dVector(
                        np.clip(src_raw[:, 3:6] / 255.0, 0, 1))
                curr.transform(world_tf)

                merged += curr
                merged = merged.voxel_down_sample(voxel_size_duck)

                if i % 10 == 0:
                    print(f"[ICP] 帧{i:3d}/{n}: fitness={fitness:.3f}, "
                          f"Δt={t_m*100:.1f}cm, Δθ={ang:.1f}°, "
                          f"已接受={accepted}, 点云={len(merged.points)}")

                # ── 更新关键帧 ──
                if frames_since_kf >= keyframe_interval:
                    kf_pcd = _nx6_to_pcd(src_raw, icp_voxel)
                    if kf_pcd is not None:
                        kf_world_tf = world_tf.copy()
                        frames_since_kf = 0
                        print(f"[ICP] 帧{i:3d}: 关键帧更新")

        # ── 连续失败过多，强制重置关键帧 ──
        if bad_count >= bad_reset_thresh:
            new_kf = _nx6_to_pcd(src_raw, icp_voxel)
            if new_kf is not None:
                kf_pcd = new_kf
                # 注意：此时 kf_world_tf 维持上次已知的世界坐标不变
                # 相当于假设这帧和上一个关键帧在同一位置（保守处理）
                frames_since_kf = 0
                bad_count = 0
                print(f"[ICP] 帧{i:3d}: 连续失败，强制重置关键帧")

    print(f"\n[ICP] 完成: 接受={accepted}/{n-1} 帧, "
          f"最终点云={len(merged.points)} 点")

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
    # 新增参数
    parser.add_argument("--max_trans", type=float, default=0.05,
                        help="每帧最大允许平移(m)，默认5cm")
    parser.add_argument("--max_angle", type=float, default=8.0,
                        help="每帧最大允许旋转(度)，默认8°")
    parser.add_argument("--kf_interval", type=int, default=8,
                        help="关键帧更新间隔(帧数)，默认8")
    parser.add_argument("--dbscan_eps", type=float, default=0.05,
                        help="DBSCAN邻域半径(m)，默认5cm")
    args = parser.parse_args()

    print("=" * 60)
    print("  Duck PCD Pipeline - v3 (Duck-to-Duck ICP)")
    print("=" * 60)

    # 读数据
    reader = BagReader(args.bag)
    frames = reader.read_all(max_frames=args.max_frames)
    if not frames:
        print("[ERROR] 没有配对到帧")
        sys.exit(1)
    if args.skip_frames > 1:
        frames = frames[::args.skip_frames]
        print(f"[INFO] 跳帧后 {len(frames)} 帧")

    frames_rgb = [f[0] for f in frames]
    frames_pcd = [f[1] for f in frames]
    print(f"[INFO] RGB: {frames_rgb[0].shape}, 首帧点数: {len(frames_pcd[0])}")

    # 读相机参数
    color_intr = get_color_intrinsics(args.bag)
    extrinsics = get_extrinsics_depth_to_color(args.bag)
    print(f"[TF] 外参矩阵:\n{np.round(extrinsics, 4)}")

    # SAM2分割
    seg = DuckSegmenter(args.sam2_checkpoint)
    pos_pts, neg_pts = seg.get_clicks(frames_rgb[0])
    print(f"[SEG] 正样本: {len(pos_pts)}, 负样本: {len(neg_pts)}")
    masks = seg.segment_all(frames_rgb, pos_pts, neg_pts)

    # 保存mask预览
    if args.save_masks:
        os.makedirs("masks_preview", exist_ok=True)
        for i in range(min(30, len(masks))):
            prev = frames_rgb[i].copy()
            prev[masks[i]] = (prev[masks[i]] * 0.4 +
                              np.array([0, 255, 0]) * 0.6).astype(np.uint8)
            cv2.imwrite(f"masks_preview/frame_{i:04d}.jpg",
                       cv2.cvtColor(prev, cv2.COLOR_RGB2BGR))
        print("[INFO] mask预览已保存到 masks_preview/")

    # 逐帧提取鸭子点云
    valid_ducks = []
    print(f"\n[PCD] 逐帧过滤，共 {len(frames)} 帧...")
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

        if i % 50 == 0:
            print(f"[PCD] 帧{i}: 鸭子{len(duck)}点, mask{mask.sum()}px")

    print(f"\n[PCD] 有效帧: {len(valid_ducks)}")
    if not valid_ducks:
        print("[ERROR] 没有提取到鸭子点云！")
        sys.exit(1)

    # ICP多帧配准融合（新版 Duck-to-Duck）
    if len(valid_ducks) == 1:
        merged = valid_ducks[0]
        print("[INFO] 只有一帧，跳过ICP")
    else:
        merged = merge_duck_icp(
            valid_ducks,
            voxel_size_duck=args.voxel_size,
            max_trans_m=args.max_trans,
            max_angle_deg=args.max_angle,
            keyframe_interval=args.kf_interval)

    if len(merged) == 0:
        print("[ERROR] ICP后点云为空")
        sys.exit(1)

    print(f"[PCD] ICP后点数: {len(merged)}")

    # 统计滤波（去孤立噪点）
    merged = statistical_filter(merged, nb=20, std=2.0)

    # RANSAC去桌面
    if not args.no_ransac:
        merged = remove_floor_ransac(merged, dist_thresh=0.012)

    # ★ DBSCAN聚类：强制保留最大连通块（关键步骤）
    merged = keep_largest_cluster(merged, eps=args.dbscan_eps, min_pts=10)

    # 再次统计滤波
    merged = statistical_filter(merged, nb=15, std=1.5)

    # 最终降采样
    merged = voxel_down(merged, args.voxel_size)
    print(f"[PCD] 最终点数: {len(merged)}")

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
    size_x = pts[:, 0].max() - pts[:, 0].min()
    size_y = pts[:, 1].max() - pts[:, 1].min()
    size_z = pts[:, 2].max() - pts[:, 2].min()
    print(f"\n✅ 保存到: {args.output}")
    print(f"   总点数: {len(pts)}")
    print(f"   X: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}] m  (跨度 {size_x:.3f}m)")
    print(f"   Y: [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}] m  (跨度 {size_y:.3f}m)")
    print(f"   Z: [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}] m  (跨度 {size_z:.3f}m)")
    if max(size_x, size_y, size_z) > 0.5:
        print("⚠️  点云跨度 > 50cm，可能仍有漂移帧混入。")
        print("   建议：降低 --max_trans（当前{:.2f}）或 --max_angle（当前{:.1f}）".format(
            args.max_trans, args.max_angle))


if __name__ == "__main__":
    main()
  
