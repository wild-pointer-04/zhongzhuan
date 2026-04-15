#!/usr/bin/env python3
"""
Duck 3D Point Cloud Extraction Pipeline (TF + Scene ICP + Cleanup)
================================================================
"""

import argparse
import os
import sys
import numpy as np
import cv2
import open3d as o3d
from collections import deque

# ─── ROS2 bag reading ────────────────────────────────────────────────────────
try:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import rosbag2_py
except ImportError:
    print("[ERROR] 需要ROS2环境。请在 ros_sam_310 conda 环境中运行。")
    sys.exit(1)

# ─── SAM2 ────────────────────────────────────────────────────────────────────
SAM2_AVAILABLE = False
try:
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    SAM2_AVAILABLE = True
    print("[INFO] SAM2 已找到")
except ImportError:
    print("[WARN] SAM2 未安装，将使用 SAM1 fallback")
    try:
        from segment_anything import sam_model_registry, SamPredictor
        SAM1_AVAILABLE = True
        print("[INFO] SAM1 已找到")
    except ImportError:
        print("[ERROR] SAM1 和 SAM2 都未安装！")
        sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# 1. BAG 读取器
# ══════════════════════════════════════════════════════════════════════════════
class BagReader:
    TOPICS = {
        "color": "/camera/color/image_raw",
        "depth_points": "/camera/depth/points",
    }

    def __init__(self, bag_path: str):
        self.bag_path = bag_path
        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
        converter_options = rosbag2_py.ConverterOptions("", "")
        self.reader = rosbag2_py.SequentialReader()
        self.reader.open(storage_options, converter_options)

        topic_types = self.reader.get_all_topics_and_types()
        self.type_map = {t.name: t.type for t in topic_types}

    def read_all(self, max_frames=None):
        color_buf = deque()
        points_buf = deque()
        paired = []
        frame_count = 0

        print("[BAG] 开始读取并配对帧...")
        while self.reader.has_next():
            topic, raw, ts_ns = self.reader.read_next()

            if topic == self.TOPICS["color"]:
                msg_type = get_message(self.type_map[topic])
                msg = deserialize_message(raw, msg_type)
                img = self._decode_image(msg)
                if img is not None:
                    color_buf.append((ts_ns, img))

            elif topic == self.TOPICS["depth_points"]:
                msg_type = get_message(self.type_map[topic])
                msg = deserialize_message(raw, msg_type)
                pts = self._decode_pointcloud2(msg)
                if pts is not None:
                    points_buf.append((ts_ns, pts))

            while color_buf and points_buf:
                tc, ic = color_buf[0]
                tp, pp = points_buf[0]
                dt_ms = abs(tc - tp) / 1e6

                if dt_ms < 33.0:
                    paired.append((ic, pp, tc))
                    color_buf.popleft()
                    points_buf.popleft()
                    frame_count += 1
                    if frame_count % 50 == 0:
                        print(f"[BAG] 已配对 {frame_count} 帧，ΔT={dt_ms:.1f}ms")
                    if max_frames and frame_count >= max_frames:
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
            img = data.reshape(h, w, 3)
        elif enc in ("bgr8", "bgr"):
            img = data.reshape(h, w, 3)[:, :, ::-1].copy()
        elif enc in ("mono8", "8uc1"):
            img = cv2.cvtColor(data.reshape(h, w), cv2.COLOR_GRAY2RGB)
        else:
            return None
        return img

    @staticmethod
    def _decode_pointcloud2(msg):
        fields = {f.name: f for f in msg.fields}
        if not all(k in fields for k in ("x", "y", "z")): return None

        point_step = msg.point_step
        data = np.frombuffer(msg.data, dtype=np.uint8)
        n_points = msg.width * msg.height
        result = np.zeros((n_points, 6), dtype=np.float32)

        for i, name in enumerate(["x", "y", "z"]):
            f = fields[name]
            result[:, i] = _extract_field_fast(data, f.offset, point_step, n_points, np.float32)

        if "rgb" in fields:
            f = fields["rgb"]
            raw_rgb = _extract_field_fast(data, f.offset, point_step, n_points, np.float32)
            rgb_int = raw_rgb.view(np.uint32)
            result[:, 3] = ((rgb_int >> 16) & 0xFF).astype(np.float32)
            result[:, 4] = ((rgb_int >> 8) & 0xFF).astype(np.float32)
            result[:, 5] = (rgb_int & 0xFF).astype(np.float32)
        else:
            result[:, 3:] = 255.0

        valid = np.isfinite(result[:, :3]).all(axis=1)
        valid &= (result[:, 2] > 0.01)
        return result[valid]

def _extract_field_fast(data: np.ndarray, offset: int, step: int, n: int, dtype):
    itemsize = np.dtype(dtype).itemsize
    row_starts = np.arange(n) * step + offset
    indices = (row_starts[:, None] + np.arange(itemsize)[None, :]).ravel()
    return np.frombuffer(data[indices].tobytes(), dtype=dtype)

# ══════════════════════════════════════════════════════════════════════════════
# 2. SAM2 视频分割器
# ══════════════════════════════════════════════════════════════════════════════
class DuckSegmenter:
    def __init__(self, sam2_checkpoint: str = None, sam1_checkpoint: str = None):
        self.use_sam2 = SAM2_AVAILABLE and sam2_checkpoint and os.path.exists(sam2_checkpoint)
        if self.use_sam2:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if "tiny" in sam2_checkpoint: cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
            elif "small" in sam2_checkpoint: cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
            elif "base_plus" in sam2_checkpoint: cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
            else: cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            self.predictor = build_sam2_video_predictor(cfg, sam2_checkpoint, device=device)
        else:
            device = "cpu"
            sam = sam_model_registry["vit_h"](checkpoint=sam1_checkpoint)
            sam.to(device)
            self.predictor = SamPredictor(sam)

    def get_first_frame_clicks(self, rgb_image: np.ndarray):
        h, w = rgb_image.shape[:2]
        display = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        scale = min(1.0, 1280 / w, 720 / h)
        if scale < 1.0: display = cv2.resize(display, (int(w * scale), int(h * scale)))

        pos_points, neg_points, all_points = [], [], []
        print("\n" + "="*60)
        print("  第一帧标注说明：左键=正样本(鸭子) | 右键=负样本(桌面) | ENTER=确认 | Z=撤销")
        print("="*60 + "\n")

        canvas = [display.copy()]
        def draw_points():
            img = canvas[0].copy()
            for p in pos_points:
                px, py = int(p[0] * scale), int(p[1] * scale)
                cv2.circle(img, (px, py), 8, (0, 255, 0), -1)
            for p in neg_points:
                px, py = int(p[0] * scale), int(p[1] * scale)
                cv2.circle(img, (px, py), 8, (0, 0, 255), -1)
            cv2.putText(img, f"Pos: {len(pos_points)} Neg: {len(neg_points)} [ENTER]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            return img

        def on_mouse(event, x, y, flags, param):
            rx, ry = x / scale, y / scale
            if event == cv2.EVENT_LBUTTONDOWN:
                pos_points.append([rx, ry]); all_points.append((rx, ry, True))
            elif event == cv2.EVENT_RBUTTONDOWN:
                neg_points.append([rx, ry]); all_points.append((rx, ry, False))

        cv2.namedWindow("Duck Annotation - Frame 0", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Duck Annotation - Frame 0", on_mouse)

        while True:
            cv2.imshow("Duck Annotation - Frame 0", draw_points())
            key = cv2.waitKey(30) & 0xFF
            if key in [13, ord(' ')] and len(pos_points) > 0: break
            elif key in [ord('z'), ord('Z')] and all_points:
                _, _, is_pos = all_points.pop()
                pos_points.pop() if is_pos else neg_points.pop()
        cv2.destroyAllWindows()
        return np.array(pos_points), np.array(neg_points)

    def segment_all_frames_sam2(self, frames_rgb: list, pos_points: np.ndarray, neg_points: np.ndarray):
        import tempfile, shutil
        from PIL import Image
        tmp_dir = tempfile.mkdtemp(prefix="sam2_frames_")
        try:
            for i, rgb in enumerate(frames_rgb):
                Image.fromarray(rgb).save(os.path.join(tmp_dir, f"{i:05d}.jpg"), quality=95)
            with torch.inference_mode():
                inference_state = self.predictor.init_state(video_path=tmp_dir)
                points = list(pos_points) + list(neg_points)
                labels = [1]*len(pos_points) + [0]*len(neg_points)
                self.predictor.add_new_points_or_box(
                    inference_state=inference_state, frame_idx=0, obj_id=1,
                    points=np.array(points, dtype=np.float32), labels=np.array(labels, dtype=np.int32)
                )
                masks = [None] * len(frames_rgb)
                for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                    masks[out_frame_idx] = (out_mask_logits[0, 0] > 0.0).cpu().numpy()
            return [m if m is not None else np.zeros(frames_rgb[0].shape[:2], dtype=bool) for m in masks]
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3. 点云过滤器 (TF外参修正版 + 飞线清理)
# ══════════════════════════════════════════════════════════════════════════════
class PointCloudFilter:
    @staticmethod
    def filter_by_mask_and_depth(pcd_nx6, rgb_image, mask, color_camera_intrinsics, extrinsics_matrix):
        fx, fy, cx, cy = color_camera_intrinsics
        pts_3d = pcd_nx6[:, :3]
        ones = np.ones((pts_3d.shape[0], 1), dtype=np.float32)
        pts_4d = np.hstack([pts_3d, ones])
        
        pts_color_opt = (extrinsics_matrix @ pts_4d.T).T
        X, Y, Z = pts_color_opt[:, 0], pts_color_opt[:, 1], pts_color_opt[:, 2]

        u = (X / Z * fx + cx).astype(np.int32)
        v = (Y / Z * fy + cy).astype(np.int32)

        H, W = mask.shape
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (Z > 0.05)

        u_valid, v_valid = u[in_bounds], v[in_bounds]
        in_mask = mask[v_valid, u_valid]

        final_mask = np.zeros(len(pcd_nx6), dtype=bool)
        in_bounds_indices = np.where(in_bounds)[0]
        duck_indices = in_bounds_indices[in_mask]
        final_mask[duck_indices] = True

        filtered_pcd = pcd_nx6[final_mask].copy()
        filtered_pcd[:, 3:6] = rgb_image[v_valid[in_mask], u_valid[in_mask]]
        return filtered_pcd

    @staticmethod
    def voxel_downsample(pcd_nx6: np.ndarray, voxel_size=0.003):
        if len(pcd_nx6) < 10: return pcd_nx6
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pcd_nx6[:, :3].astype(np.float64))
        if pcd_nx6.shape[1] >= 6: o3d_pcd.colors = o3d.utility.Vector3dVector(pcd_nx6[:, 3:6] / 255.0)
        down = o3d_pcd.voxel_down_sample(voxel_size)
        pts = np.asarray(down.points, dtype=np.float32)
        if down.has_colors(): return np.hstack([pts, (np.asarray(down.colors) * 255).astype(np.float32)])
        return pts

    # 【新增刀法 1】：切除底座
    @staticmethod
    def remove_table_plane(pcd_nx6: np.ndarray, distance_threshold=0.012):
        if len(pcd_nx6) < 100: return pcd_nx6
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pcd_nx6[:, :3].astype(np.float64))
        try:
            # 找到最大的平面（即被误框进来的桌面）并切除
            plane_model, inliers = o3d_pcd.segment_plane(distance_threshold, 3, 1000)
            outlier_mask = np.ones(len(pcd_nx6), dtype=bool)
            outlier_mask[inliers] = False
            return pcd_nx6[outlier_mask]
        except:
            return pcd_nx6

    # 【新增刀法 2】：消灭飞线和悬空杂物
    @staticmethod
    def statistical_filter(pcd_nx6: np.ndarray, nb_neighbors=30, std_ratio=1.0):
        if len(pcd_nx6) < nb_neighbors + 1: return pcd_nx6
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pcd_nx6[:, :3].astype(np.float64))
        if pcd_nx6.shape[1] >= 6:
            o3d_pcd.colors = o3d.utility.Vector3dVector(pcd_nx6[:, 3:6] / 255.0)
        # 严格过滤掉不合群的点
        _, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        return pcd_nx6[ind]

# ══════════════════════════════════════════════════════════════════════════════
# 4. TF外参解析引擎 & 场景级 ICP 合并核心
# ══════════════════════════════════════════════════════════════════════════════
def get_extrinsics_matrix(bag_path: str, source_frame="camera_depth_optical_frame", target_frame="camera_color_optical_frame"):
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    
    if "/tf_static" not in type_map: return np.eye(4)

    adj = {}
    while reader.has_next():
        topic, raw, _ = reader.read_next()
        if topic == "/tf_static":
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(raw, msg_type)
            for transform in msg.transforms:
                p, c = transform.header.frame_id, transform.child_frame_id
                t, r = transform.transform.translation, transform.transform.rotation
                qx, qy, qz, qw = r.x, r.y, r.z, r.w
                mat = np.array([
                    [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw, t.x],
                    [    2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2,     2*qy*qz - 2*qx*qw, t.y],
                    [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2, t.z],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                if p not in adj: adj[p] = []
                if c not in adj: adj[c] = []
                adj[c].append((p, mat))
                adj[p].append((c, np.linalg.inv(mat)))
            break

    queue = deque([(source_frame, np.eye(4, dtype=np.float32))])
    visited = set([source_frame])
    while queue:
        curr, curr_mat = queue.popleft()
        if curr == target_frame: return curr_mat
        for nxt, step_mat in adj.get(curr, []):
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, step_mat @ curr_mat))
    return np.eye(4)

def get_intrinsics_from_bag(bag_path: str):
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}

    while reader.has_next():
        topic, raw, _ = reader.read_next()
        if topic in ("/camera/color/camera_info", "/camera/depth/camera_info"):
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(raw, msg_type)
            return (msg.k[0], msg.k[4], msg.k[2], msg.k[5])
    return (412.857, 412.857, 424.0, 237.019)

def merge_duck_with_scene_icp(scene_arrays, duck_arrays, duck_voxel_size=0.001):
    if not duck_arrays: return np.array([])
    
    def make_clean_scene_pcd(nx6_array, voxel_size):
        valid_pts = nx6_array[(nx6_array[:, 2] > 0.1) & (nx6_array[:, 2] < 1.0)]
        if len(valid_pts) < 100: return None
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(valid_pts[:, :3].astype(np.float64))
        if valid_pts.shape[1] >= 6:
            pcd.colors = o3d.utility.Vector3dVector(valid_pts[:, 3:6] / 255.0)
        pcd = pcd.voxel_down_sample(voxel_size)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
        return pcd

    scene_voxel = 0.01 
    target_scene = make_clean_scene_pcd(scene_arrays[0], scene_voxel)
    
    merged_duck_pcd = o3d.geometry.PointCloud()
    merged_duck_pcd.points = o3d.utility.Vector3dVector(duck_arrays[0][:, :3].astype(np.float64))
    if duck_arrays[0].shape[1] >= 6:
        merged_duck_pcd.colors = o3d.utility.Vector3dVector(duck_arrays[0][:, 3:6] / 255.0)

    trans_global = np.eye(4)

    for i in range(1, len(scene_arrays)):
        source_scene = make_clean_scene_pcd(scene_arrays[i], scene_voxel)
        if source_scene is None: continue

        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_scene, 
            target_scene, 
            0.03, 
            np.eye(4), 
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50)
        )

        if result_icp.fitness < 0.15:
            continue

        trans_global = trans_global @ result_icp.transformation

        current_duck = o3d.geometry.PointCloud()
        current_duck.points = o3d.utility.Vector3dVector(duck_arrays[i][:, :3].astype(np.float64))
        if duck_arrays[i].shape[1] >= 6:
            current_duck.colors = o3d.utility.Vector3dVector(duck_arrays[i][:, 3:6] / 255.0)
            
        current_duck.transform(trans_global)
        merged_duck_pcd += current_duck
        merged_duck_pcd = merged_duck_pcd.voxel_down_sample(duck_voxel_size)
        target_scene = source_scene

    pts = np.asarray(merged_duck_pcd.points, dtype=np.float32)
    if merged_duck_pcd.has_colors():
        cols = (np.asarray(merged_duck_pcd.colors) * 255.0).astype(np.float32)
        return np.hstack([pts, cols])
    return pts

# ══════════════════════════════════════════════════════════════════════════════
# 5. 主流程
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", default="duck_1_0.db3")
    parser.add_argument("--output", default="duck_final_icp.pcd")
    parser.add_argument("--sam2_checkpoint", default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--skip_frames", type=int, default=1)
    parser.add_argument("--voxel_size", type=float, default=0.001)
    args = parser.parse_args()

    reader = BagReader(args.bag)
    all_frames = reader.read_all(max_frames=args.max_frames)
    if not all_frames: sys.exit(1)
    if args.skip_frames > 1: all_frames = all_frames[::args.skip_frames]

    frames_rgb = [f[0] for f in all_frames]
    frames_pcd = [f[1] for f in all_frames]

    intrinsics = get_intrinsics_from_bag(args.bag)
    extrinsics_matrix = get_extrinsics_matrix(args.bag)

    segmenter = DuckSegmenter(sam2_checkpoint=args.sam2_checkpoint)
    pos, neg = segmenter.get_first_frame_clicks(frames_rgb[0])
    masks = segmenter.segment_all_frames_sam2(frames_rgb, pos, neg)

    pf = PointCloudFilter()
    valid_scenes, valid_ducks = [], []
    
    for i, (pcd_raw, mask) in enumerate(zip(frames_pcd, masks)):
        if mask is None or mask.sum() == 0: continue
        h, w = frames_rgb[i].shape[:2]
        if mask.shape != (h, w): mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

        filtered = pf.filter_by_mask_and_depth(pcd_raw, frames_rgb[i], mask, intrinsics, extrinsics_matrix)
        if len(filtered) < 10: continue

        filtered = pf.voxel_downsample(filtered, voxel_size=args.voxel_size)
        valid_ducks.append(filtered)
        valid_scenes.append(pcd_raw) 

    # 1. 拼接
    merged = merge_duck_with_scene_icp(valid_scenes, valid_ducks, duck_voxel_size=args.voxel_size)

    # 2. 手术刀介入：切底座 + 去飞线
    print("\n[PCD] 拼接完成，正在切除底部粘连的桌面并清理飞线噪点...")
    merged = pf.remove_table_plane(merged, distance_threshold=0.012)
    merged = pf.statistical_filter(merged, nb_neighbors=30, std_ratio=1.0)

    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(merged[:, :3].astype(np.float64))
    if merged.shape[1] >= 6: final_pcd.colors = o3d.utility.Vector3dVector(np.clip(merged[:, 3:6] / 255.0, 0, 1).astype(np.float64))

    final_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    o3d.io.write_point_cloud(args.output, final_pcd)
    print(f"\n✅ 成功保存最终完美版点云到: {args.output} (共 {len(final_pcd.points)} 点)")

if __name__ == "__main__":
    main()
