#!/usr/bin/env python3
"""
静态单面点云提取 - 针对"只拍一个面、物体静止"场景
核心改动：
  1. 去掉ICP，改为直接堆叠+体素降采样（物体静止，无需配准）
  2. filter_duck_points 改为直接用深度图投影，不依赖外参
  3. 增加深度图可视化调试，方便确认mask对齐
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
except ImportError:
    print("[ERROR] SAM2未安装")
    sys.exit(1)


# ══════════════════════════════════════════
# 工具
# ══════════════════════════════════════════
def _extract_field_fast(data, offset, step, n, dtype):
    itemsize = np.dtype(dtype).itemsize
    row_starts = np.arange(n) * step + offset
    indices = (row_starts[:, None] + np.arange(itemsize)[None, :]).ravel()
    return np.frombuffer(data[indices].tobytes(), dtype=dtype)


# ══════════════════════════════════════════
# Bag读取（与原版相同）
# ══════════════════════════════════════════
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
                    if max_frames and len(paired) >= max_frames:
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


# ══════════════════════════════════════════
# 相机参数
# ══════════════════════════════════════════
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
            return (K[0], K[4], K[2], K[5], msg.width, msg.height)
    print("[WARN] 未找到彩色内参，使用默认值")
    return (691.33, 691.51, 643.92, 362.12, 1280, 720)


def get_depth_intrinsics(bag_path):
    """读取深度相机内参（新增）"""
    sr = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    reader = rosbag2_py.SequentialReader()
    reader.open(sr, rosbag2_py.ConverterOptions("", ""))
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    while reader.has_next():
        topic, raw, _ = reader.read_next()
        if topic == "/camera/depth/camera_info":
            msg = deserialize_message(raw, get_message(type_map[topic]))
            K = msg.k
            print(f"[INTR] 深度相机内参: fx={K[0]:.2f}, fy={K[4]:.2f}, "
                  f"cx={K[2]:.2f}, cy={K[5]:.2f}, 分辨率={msg.width}x{msg.height}")
            return (K[0], K[4], K[2], K[5], msg.width, msg.height)
    print("[WARN] 未找到深度内参，使用彩色内参代替")
    return None


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
            print(f"[TF] depth→color 外参矩阵找到，平移="
                  f"[{mat[0,3]*1000:.1f}, {mat[1,3]*1000:.1f}, {mat[2,3]*1000:.1f}]mm")
            return mat
        for nxt, step_mat in adj.get(curr, []):
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, step_mat @ mat))

    print("[WARN] 未找到TF外参，使用单位矩阵")
    return np.eye(4, dtype=np.float64)


# ══════════════════════════════════════════
# SAM2（与原版相同）
# ══════════════════════════════════════════
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

        print("\n左键=正样本(目标物体) | 右键=负样本(背景) | ENTER=确认 | Z=撤销")
        canvas = [disp.copy()]

        def redraw():
            img = canvas[0].copy()
            for p in pos:
                cv2.circle(img, (int(p[0]*scale), int(p[1]*scale)), 8, (0,255,0), -1)
            for p in neg:
                cv2.circle(img, (int(p[0]*scale), int(p[1]*scale)), 8, (0,0,255), -1)
            cv2.putText(img, f"绿(正):{len(pos)} 红(负):{len(neg)} [ENTER确认]",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            return img

        def on_mouse(event, x, y, flags, param):
            rx, ry = x/scale, y/scale
            if event == cv2.EVENT_LBUTTONDOWN:
                pos.append([rx, ry]); all_pts.append((rx, ry, True))
            elif event == cv2.EVENT_RBUTTONDOWN:
                neg.append([rx, ry]); all_pts.append((rx, ry, False))

        cv2.namedWindow("Annotation", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Annotation", on_mouse)
        while True:
            cv2.imshow("Annotation", redraw())
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
            h, w = frames_rgb[0].shape[:2]
            return [m if m is not None else np.zeros((h, w), bool) for m in masks]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ══════════════════════════════════════════
# ★ 核心修改：新的点云过滤函数
#   策略：点云本身带RGB，直接用3D点投影到彩色图
#   关键：depth/points topic里的点已经在深度相机坐标系下
#   需要用 depth→color 外参转换后，再用彩色内参投影
# ══════════════════════════════════════════
def filter_points_by_mask(pcd_nx6, rgb_image, mask,
                          color_intrinsics, extrinsics_d2c,
                          debug=False):
    """
    将深度点云投影到彩色图像，用mask过滤。
    pcd_nx6: [N, 6] float32，前3列是XYZ（深度相机坐标系，单位m）
    color_intrinsics: (fx, fy, cx, cy, W, H)
    extrinsics_d2c: 4x4矩阵，将深度坐标系点转到彩色坐标系
    """
    fx, fy, cx, cy, W, H = color_intrinsics

    pts3d = pcd_nx6[:, :3].astype(np.float64)  # [N, 3]
    N = len(pts3d)

    # 转换到彩色相机坐标系
    ones = np.ones((N, 1))
    pts_h = np.hstack([pts3d, ones])             # [N, 4]
    pts_c = (extrinsics_d2c @ pts_h.T).T         # [N, 4]
    Xc, Yc, Zc = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]

    # 只保留在相机前方的点
    valid_z = Zc > 0.05
    u = np.full(N, -1, dtype=np.int32)
    v = np.full(N, -1, dtype=np.int32)
    u[valid_z] = np.round(Xc[valid_z] / Zc[valid_z] * fx + cx).astype(np.int32)
    v[valid_z] = np.round(Yc[valid_z] / Zc[valid_z] * fy + cy).astype(np.int32)

    in_bounds = valid_z & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    # ★ 调试：把投影点画在图上，确认对齐
    if debug:
        dbg = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR).copy()
        idx_all = np.where(in_bounds)[0][::10]  # 每10个点画一个
        for k in idx_all:
            cv2.circle(dbg, (u[k], v[k]), 1, (0, 255, 0), -1)
        # 叠加mask
        mask_vis = np.zeros_like(dbg)
        mask_vis[mask] = (0, 0, 255)
        dbg = cv2.addWeighted(dbg, 0.7, mask_vis, 0.3, 0)
        cv2.imshow("投影调试（绿=深度点，红=mask）", dbg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 用mask过滤
    in_mask = np.zeros(N, dtype=bool)
    idx = np.where(in_bounds)[0]
    in_mask[idx] = mask[v[idx], u[idx]]

    result = pcd_nx6[in_mask].copy()

    # 用彩色图像的真实颜色替换点云颜色（更准确）
    if result.shape[1] >= 6 and len(result) > 0:
        idx_kept = np.where(in_mask)[0]
        result[:, 3] = rgb_image[v[idx_kept], u[idx_kept], 0].astype(np.float32)
        result[:, 4] = rgb_image[v[idx_kept], u[idx_kept], 1].astype(np.float32)
        result[:, 5] = rgb_image[v[idx_kept], u[idx_kept], 2].astype(np.float32)

    return result


# ══════════════════════════════════════════
# 辅助处理函数（与原版基本相同）
# ══════════════════════════════════════════
def statistical_filter(pcd_nx6, nb=20, std=1.5):
    if len(pcd_nx6) < nb + 1:
        return pcd_nx6
    o3pcd = o3d.geometry.PointCloud()
    o3pcd.points = o3d.utility.Vector3dVector(pcd_nx6[:, :3].astype(np.float64))
    _, ind = o3pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)
    print(f"[FILTER] 统计滤波: {len(pcd_nx6)} → {len(ind)} 点")
    return pcd_nx6[np.array(ind)]


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
            print(f"[RANSAC] 删除地面，法向量={normal.round(3)}，占比={inlier_ratio:.1%}")
            keep = np.ones(len(pcd_nx6), dtype=bool)
            keep[inliers] = False
            return pcd_nx6[keep]
    except Exception as e:
        print(f"[RANSAC] 失败: {e}")
    return pcd_nx6


def keep_largest_cluster(pcd_nx6, eps=0.05, min_pts=10):
    if len(pcd_nx6) < min_pts * 2:
        return pcd_nx6
    o3pcd = o3d.geometry.PointCloud()
    o3pcd.points = o3d.utility.Vector3dVector(pcd_nx6[:, :3].astype(np.float64))
    labels = np.array(o3pcd.cluster_dbscan(eps=eps, min_points=min_pts,
                                            print_progress=False))
    if labels.max() < 0:
        return pcd_nx6
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    largest_label = unique[np.argmax(counts)]
    mask = (labels == largest_label)
    kept = pcd_nx6[mask]
    print(f"[DBSCAN] {len(pcd_nx6)} → {len(kept)} 点")
    return kept


# ══════════════════════════════════════════
# ★ 新增：静态多帧直接融合（不需要ICP）
#   物体静止 → 直接堆叠所有帧的点云 → 体素降采样去重
# ══════════════════════════════════════════
def merge_static(obj_arrays, voxel_size):
    """
    静态场景融合：直接合并所有帧的点云，用体素降采样去重。
    不需要任何配准，适用于相机和物体都静止（或只拍一个面）的情况。
    """
    print(f"\n[MERGE] 静态融合 {len(obj_arrays)} 帧...")
    merged = o3d.geometry.PointCloud()
    for i, arr in enumerate(obj_arrays):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(arr[:, :3].astype(np.float64))
        if arr.shape[1] >= 6:
            pcd.colors = o3d.utility.Vector3dVector(
                np.clip(arr[:, 3:6] / 255.0, 0, 1))
        merged += pcd

    print(f"[MERGE] 合并前总点数: {len(merged.points)}")
    merged = merged.voxel_down_sample(voxel_size)
    print(f"[MERGE] 体素降采样后: {len(merged.points)} 点")

    pts = np.asarray(merged.points, dtype=np.float32)
    if merged.has_colors():
        cols = (np.asarray(merged.colors) * 255).astype(np.float32)
        return np.hstack([pts, cols])
    return pts


# ══════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="静态单面点云提取")
    parser.add_argument("--bag", required=True)
    parser.add_argument("--output", default="object_static.pcd")
    parser.add_argument("--sam2_checkpoint", required=True)
    parser.add_argument("--max_frames", type=int, default=50,
                        help="最多取多少帧（静态场景不需要太多，默认50）")
    parser.add_argument("--skip_frames", type=int, default=1)
    parser.add_argument("--voxel_size", type=float, default=0.001)
    parser.add_argument("--no_ransac", action="store_true")
    parser.add_argument("--dbscan_eps", type=float, default=0.03)
    parser.add_argument("--debug_projection", action="store_true",
                        help="显示第一帧的投影调试图，确认mask和点云对齐")
    parser.add_argument("--save_masks", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  静态单面点云提取 Pipeline")
    print("=" * 60)

    # 读数据
    reader = BagReader(args.bag)
    frames = reader.read_all(max_frames=args.max_frames)
    if not frames:
        print("[ERROR] 没有配对到帧")
        sys.exit(1)
    if args.skip_frames > 1:
        frames = frames[::args.skip_frames]
    print(f"[INFO] 共 {len(frames)} 帧")

    frames_rgb = [f[0] for f in frames]
    frames_pcd = [f[1] for f in frames]

    # 读相机参数
    color_intr = get_color_intrinsics(args.bag)
    extrinsics = get_extrinsics_depth_to_color(args.bag)
    fx, fy, cx, cy, W, H = color_intr
    print(f"[INTR] 彩色: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}, {W}x{H}")

    # SAM2分割
    seg = DuckSegmenter(args.sam2_checkpoint)
    pos_pts, neg_pts = seg.get_clicks(frames_rgb[0])
    masks = seg.segment_all(frames_rgb, pos_pts, neg_pts)

    if args.save_masks:
        os.makedirs("masks_preview", exist_ok=True)
        for i in range(min(20, len(masks))):
            prev = frames_rgb[i].copy()
            prev[masks[i]] = (prev[masks[i]] * 0.4 +
                              np.array([0, 255, 0]) * 0.6).astype(np.uint8)
            cv2.imwrite(f"masks_preview/frame_{i:04d}.jpg",
                        cv2.cvtColor(prev, cv2.COLOR_RGB2BGR))

    # 逐帧提取物体点云
    valid_objs = []
    debug_done = False
    for i, (pcd_raw, mask) in enumerate(zip(frames_pcd, masks)):
        if mask is None or mask.sum() < 100:
            continue

        h, w = frames_rgb[i].shape[:2]
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST).astype(bool)

        # ★ 第一帧可选显示调试图
        do_debug = args.debug_projection and not debug_done
        if do_debug:
            debug_done = True

        obj = filter_points_by_mask(
            pcd_raw, frames_rgb[i], mask,
            color_intr, extrinsics,
            debug=do_debug)

        if len(obj) < 20:
            print(f"[WARN] 帧{i}: 过滤后只有{len(obj)}点，跳过")
            continue

        obj = voxel_down(obj, args.voxel_size)
        valid_objs.append(obj)

        if i % 10 == 0:
            coverage = mask.sum() / mask.size * 100
            print(f"[PCD] 帧{i}: 物体{len(obj)}点, mask覆盖{coverage:.1f}%")

    print(f"\n[PCD] 有效帧: {len(valid_objs)}")
    if not valid_objs:
        print("[ERROR] 没有提取到物体点云！")
        print("  请尝试：--debug_projection 查看投影是否对齐")
        sys.exit(1)

    # ★ 静态融合（不用ICP）
    merged = merge_static(valid_objs, args.voxel_size)

    # 统计滤波
    merged = statistical_filter(merged, nb=20, std=2.0)

    # RANSAC去地面
    if not args.no_ransac:
        merged = remove_floor_ransac(merged, dist_thresh=0.012)

    # DBSCAN保留主体
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
    sx = pts[:, 0].max() - pts[:, 0].min()
    sy = pts[:, 1].max() - pts[:, 1].min()
    sz = pts[:, 2].max() - pts[:, 2].min()
    print(f"\n✅ 保存: {args.output}")
    print(f"   总点数: {len(pts)}")
    print(f"   X跨度: {sx:.3f}m  Y跨度: {sy:.3f}m  Z跨度: {sz:.3f}m")

    if max(sx, sy, sz) < 0.01:
        print("⚠️  点云跨度 < 1cm，可能提取到的点极少，建议用 --debug_projection 检查")


if __name__ == "__main__":
    main()
