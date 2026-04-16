#!/usr/bin/env python3
"""
Duck 3D Point Cloud Extraction Pipeline - 修复版
修复清单：
  1. 外参矩阵方向确认，只读彩色相机内参
  2. ICP改为Point-to-Plane + FPFH特征预对齐（解决大平面滑动）
  3. 加入桌面RANSAC剔除（解决底部粘连）
  4. 加入统计滤波（解决杂物尾巴）
  5. 清理死代码
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
# 2. 相机参数读取（修复：明确区分彩色/深度内参）
# ══════════════════════════════════════════════════════
def get_color_intrinsics(bag_path):
    """只读彩色相机内参，不混用深度内参"""
    sr = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    reader = rosbag2_py.SequentialReader()
    reader.open(sr, rosbag2_py.ConverterOptions("", ""))
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    while reader.has_next():
        topic, raw, _ = reader.read_next()
        if topic == "/camera/color/camera_info":  # 明确只取彩色
            msg = deserialize_message(raw, get_message(type_map[topic]))
            K = msg.k
            print(f"[INTR] 彩色相机内参: fx={K[0]:.2f}, fy={K[4]:.2f}, "
                  f"cx={K[2]:.2f}, cy={K[5]:.2f}, 分辨率={msg.width}x{msg.height}")
            return (K[0], K[4], K[2], K[5])
    print("[WARN] 未找到彩色内参，使用默认值")
    return (691.33, 691.51, 643.92, 362.12)


def get_extrinsics_depth_to_color(bag_path):
    """
    从TF链计算 depth_optical → color_optical 的4×4变换矩阵
    方向：把深度相机坐标系下的XYZ，变换到彩色相机坐标系下
    这样才能用彩色内参正确投影
    """
    sr = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    reader = rosbag2_py.SequentialReader()
    reader.open(sr, rosbag2_py.ConverterOptions("", ""))
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}

    # 构建TF图（双向）
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
            break  # tf_static只需读一次

    # BFS: depth_optical_frame → color_optical_frame
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

    print("[WARN] 未找到TF外参，使用单位矩阵（投影可能有偏差）")
    return np.eye(4, dtype=np.float64)


# ══════════════════════════════════════════════════════
# 3. SAM2分割器
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
        print("建议：鸭子头、身体各点1-2个正样本；鸭子脚正下方桌面点2-3个负样本")

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
# 4. 点云过滤（修复：外参变换 + 只用彩色内参）
# ══════════════════════════════════════════════════════
def filter_duck_points(pcd_nx6, rgb_image, mask,
                       color_intrinsics, extrinsics_d2c):
    """
    正确流程：
      XYZ (深度坐标系) → [外参] → XYZ' (彩色坐标系) → [彩色内参] → u,v → 查mask
    """
    fx, fy, cx, cy = color_intrinsics

    pts = pcd_nx6[:, :3].astype(np.float64)
    ones = np.ones((len(pts), 1))
    pts_h = np.hstack([pts, ones])  # (N, 4)

    # 外参变换：深度坐标系 → 彩色坐标系
    pts_color = (extrinsics_d2c @ pts_h.T).T  # (N, 4)
    Xc, Yc, Zc = pts_color[:, 0], pts_color[:, 1], pts_color[:, 2]

    # 用彩色相机内参投影
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

    # 从彩色图取正确颜色
    if result.shape[1] >= 6:
        result[:, 3:6] = rgb_image[v[in_mask], u[in_mask]].astype(np.float32)

    return result


def remove_floor_ransac(pcd_nx6, dist_thresh=0.012):
    """用RANSAC检测并去除地面/桌面点（解决底部粘连）"""
    if len(pcd_nx6) < 50:
        return pcd_nx6
    o3pcd = o3d.geometry.PointCloud()
    o3pcd.points = o3d.utility.Vector3dVector(pcd_nx6[:, :3].astype(np.float64))
    try:
        plane_model, inliers = o3pcd.segment_plane(
            distance_threshold=dist_thresh,
            ransac_n=3,
            num_iterations=500)
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal /= np.linalg.norm(normal)
        inlier_ratio = len(inliers) / len(pcd_nx6)
        # 只有法向量接近水平（桌面/地面）且占比>15%才删
        # 相机俯视时，桌面法向量的Y或Z分量会比较大
        is_floor = (abs(normal[1]) > 0.6 or abs(normal[2]) > 0.6)
        if is_floor and inlier_ratio > 0.15:
            print(f"[RANSAC] 检测到地面，法向量={normal.round(3)}，"
                  f"占比={inlier_ratio:.1%}，删除")
            keep = np.ones(len(pcd_nx6), dtype=bool)
            keep[inliers] = False
            return pcd_nx6[keep]
        else:
            print(f"[RANSAC] 未检测到明显地面（法向量={normal.round(3)}），保留")
    except Exception as e:
        print(f"[RANSAC] 失败: {e}")
    return pcd_nx6


def statistical_filter(pcd_nx6, nb=20, std=1.5):
    """统计滤波去除孤立噪点（解决尾巴杂物）"""
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
# 5. ICP多帧配准（修复：FPFH全局预对齐 + Point-to-Plane局部）
# ══════════════════════════════════════════════════════
def make_scene_pcd(nx6, voxel_size=0.01):
    """准备场景点云（只取近距离，降采样，计算法向量）"""
    # 只取深度 0.1~1.5m 的点（去掉地板以外的远处噪声）
    valid = (nx6[:, 2] > 0.1) & (nx6[:, 2] < 1.5)
    pts = nx6[valid]
    if len(pts) < 100:
        return None
    o3pcd = o3d.geometry.PointCloud()
    o3pcd.points = o3d.utility.Vector3dVector(pts[:, :3].astype(np.float64))
    if pts.shape[1] >= 6:
        o3pcd.colors = o3d.utility.Vector3dVector(
            np.clip(pts[:, 3:6] / 255.0, 0, 1).astype(np.float64))
    o3pcd = o3pcd.voxel_down_sample(voxel_size)
    o3pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 3, max_nn=30))
    o3pcd.orient_normals_consistent_tangent_plane(10)
    return o3pcd


def compute_fpfh(pcd, voxel_size):
    """计算FPFH特征描述子（用于全局配准）"""
    radius_feature = voxel_size * 5
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))


def global_registration(src, dst, voxel_size):
    """FPFH + RANSAC全局配准，给ICP提供好的初始值"""
    src_fpfh = compute_fpfh(src, voxel_size)
    dst_fpfh = compute_fpfh(dst, voxel_size)
    dist_thresh = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src, dst, src_fpfh, dst_fpfh,
        mutual_filter=True,
        max_correspondence_distance=dist_thresh,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    return result.transformation


def icp_refine(src, dst, init_transform, voxel_size):
    """Point-to-Plane ICP精细配准"""
    result = o3d.pipelines.registration.registration_icp(
        src, dst,
        max_correspondence_distance=voxel_size * 2,
        init=init_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=100, relative_fitness=1e-6, relative_rmse=1e-6)
    )
    return result.transformation, result.fitness


def merge_duck_icp(scene_arrays, duck_arrays, voxel_size_duck=0.001,
                   use_global_reg=True):
    """
    帧间ICP：用全场景点云估计相机运动，把运动施加到鸭子点云上
    
    改进：
    - 场景用 1cm 体素 + Point-to-Plane ICP（比 Colored ICP 在均匀色彩场景更稳）
    - 每隔N帧尝试FPFH全局配准，防止累积漂移
    - fitness阈值拒绝坏帧
    """
    if not duck_arrays:
        return np.array([])

    n = len(scene_arrays)
    print(f"\n[ICP] 开始配准，共 {n} 帧...")

    scene_voxel = 0.01  # 场景1cm体素

    target = make_scene_pcd(scene_arrays[0], scene_voxel)
    if target is None:
        print("[ICP] 第0帧场景点云无效")
        return np.array([])

    # 初始化全局鸭子点云
    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(
        duck_arrays[0][:, :3].astype(np.float64))
    if duck_arrays[0].shape[1] >= 6:
        merged.colors = o3d.utility.Vector3dVector(
            np.clip(duck_arrays[0][:, 3:6] / 255.0, 0, 1))

    trans_global = np.eye(4)
    bad_frame_count = 0

    for i in range(1, n):
        source = make_scene_pcd(scene_arrays[i], scene_voxel)
        if source is None:
            continue

        # 全局配准（每20帧或者上一帧很差时触发）
        init_tf = np.eye(4)
        if use_global_reg and (i % 20 == 1 or bad_frame_count >= 3):
            try:
                init_tf = global_registration(source, target, scene_voxel)
                print(f"[ICP] 帧{i}: 触发全局FPFH配准")
                bad_frame_count = 0
            except Exception as e:
                print(f"[ICP] 帧{i}: 全局配准失败({e})，用单位矩阵初始化")

        # 局部ICP精细配准
        tf, fitness = icp_refine(source, target, init_tf, scene_voxel)

        # 拒绝差帧
        if fitness < 0.3:
            print(f"[ICP] 帧{i}: fitness={fitness:.3f} 过低，跳过")
            bad_frame_count += 1
            continue

        bad_frame_count = 0
        trans_global = trans_global @ tf

        # 把当前帧鸭子变换到全局坐标系
        curr_duck = o3d.geometry.PointCloud()
        curr_duck.points = o3d.utility.Vector3dVector(
            duck_arrays[i][:, :3].astype(np.float64))
        if duck_arrays[i].shape[1] >= 6:
            curr_duck.colors = o3d.utility.Vector3dVector(
                np.clip(duck_arrays[i][:, 3:6] / 255.0, 0, 1))
        curr_duck.transform(trans_global)

        merged += curr_duck
        merged = merged.voxel_down_sample(voxel_size_duck)

        target = source  # 滚动更新target

        if i % 20 == 0:
            print(f"[ICP] 帧{i}/{n}: fitness={fitness:.3f}, "
                  f"鸭子点数={len(merged.points)}")

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
    parser.add_argument("--no_global_reg", action="store_true",
                        help="禁用FPFH全局配准（更快但可能漂移）")
    args = parser.parse_args()

    print("=" * 60)
    print("  Duck PCD Pipeline - 修复版（外参+内参+ICP）")
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
    print(f"[INFO] RGB: {frames_rgb[0].shape}, "
          f"首帧点数: {len(frames_pcd[0])}")

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
    valid_scenes = []

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
        valid_scenes.append(pcd_raw)

        if i % 50 == 0:
            print(f"[PCD] 帧{i}: 鸭子{len(duck)}点, mask{mask.sum()}px")

    print(f"\n[PCD] 有效帧: {len(valid_ducks)}")
    if not valid_ducks:
        print("[ERROR] 没有提取到鸭子点云！")
        sys.exit(1)

    # ICP多帧配准融合
    if len(valid_ducks) == 1:
        merged = valid_ducks[0]
        print("[INFO] 只有一帧，跳过ICP")
    else:
        merged = merge_duck_icp(
            valid_scenes, valid_ducks,
            voxel_size_duck=args.voxel_size,
            use_global_reg=not args.no_global_reg)

    if len(merged) == 0:
        print("[ERROR] ICP后点云为空")
        sys.exit(1)

    print(f"[PCD] ICP后点数: {len(merged)}")

    # 统计滤波（去尾巴杂物）
    merged = statistical_filter(merged, nb=20, std=2.0)

    # RANSAC去桌面（去底部粘连）
    if not args.no_ransac:
        merged = remove_floor_ransac(merged, dist_thresh=0.012)

    # 再次统计滤波（去RANSAC产生的边缘碎片）
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
    print(f"\n✅ 保存到: {args.output}")
    print(f"   总点数: {len(pts)}")
    print(f"   X: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}] m")
    print(f"   Y: [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}] m")
    print(f"   Z: [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}] m")


if __name__ == "__main__":
    main()
