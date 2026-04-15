#!/usr/bin/env python3
"""
Duck 3D Point Cloud Extraction Pipeline (TF Extrinsics Aligned)
================================================================
策略：
  1. 读 /camera/depth/points 与 /camera/color/image_raw 并做时间戳同步
  2. 提取 /tf_static 构建坐标变换树，获取深度到彩色镜头的外参 (Extrinsics)
  3. 用 SAM2/SAM1 进行视频级目标追踪和 Mask 分割
  4. 将 3D 点乘以外参矩阵，转换到彩色光学坐标系后，再用内参进行 2D 投影
  5. 严格过滤鸭子点云，上色，去噪，去桌面，最终合并输出 PCD
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
        print("[INFO] SAM1 已找到，使用逐帧模式（注意：相机运动时可能漂移）")
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
        print("[BAG] 话题列表：")
        for t in topic_types:
            print(f"      {t.name}  ({t.type})")

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

            # 时间戳配对
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
                        print(f"[BAG] 达到最大帧数 {max_frames}，停止读取")
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
            print(f"[SEG] 加载 SAM2: {sam2_checkpoint}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if "tiny" in sam2_checkpoint: cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
            elif "small" in sam2_checkpoint: cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
            elif "base_plus" in sam2_checkpoint: cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
            else: cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            self.predictor = build_sam2_video_predictor(cfg, sam2_checkpoint, device=device)
        else:
            print(f"[SEG] 使用 SAM1: {sam1_checkpoint}")
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
# 3. 点云过滤器 (TF外参修正版)
# ══════════════════════════════════════════════════════════════════════════════
class PointCloudFilter:
    @staticmethod
    def filter_by_mask_and_depth(
        pcd_nx6: np.ndarray,
        rgb_image: np.ndarray,
        mask: np.ndarray,
        color_camera_intrinsics: tuple,
        extrinsics_matrix: np.ndarray
    ) -> np.ndarray:
        fx, fy, cx, cy = color_camera_intrinsics
        
        # 1. 提取原始 3D 坐标
        pts_3d = pcd_nx6[:, :3]
        
        # 2. 将 3D 点云转换为齐次坐标 (N x 4)，并应用外参矩阵
        ones = np.ones((pts_3d.shape[0], 1), dtype=np.float32)
        pts_4d = np.hstack([pts_3d, ones])
        
        # 公式: P_color_opt = Extrinsics * P_depth_opt
        pts_color_opt = (extrinsics_matrix @ pts_4d.T).T
        X = pts_color_opt[:, 0]
        Y = pts_color_opt[:, 1]
        Z = pts_color_opt[:, 2]

        # 3. 使用转换后的坐标进行 2D 图像投影
        u = (X / Z * fx + cx).astype(np.int32)
        v = (Y / Z * fy + cy).astype(np.int32)

        H, W = mask.shape
        min_z = Z[Z > 0.05].min() if len(Z[Z > 0.05]) > 0 else 0.1
        max_z_threshold = min_z + 0.15 

        # 边界与截断过滤
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (Z > 0) & (Z < max_z_threshold)

        u_valid = u[in_bounds]
        v_valid = v[in_bounds]
        in_mask = mask[v_valid, u_valid]

        # 4. 构建最终 Mask 并提取点云 (保持原深度坐标系的数据不变)
        final_mask = np.zeros(len(pcd_nx6), dtype=bool)
        in_bounds_indices = np.where(in_bounds)[0]
        duck_indices = in_bounds_indices[in_mask]
        final_mask[duck_indices] = True

        filtered_pcd = pcd_nx6[final_mask].copy()

        # 5. 上色
        valid_u = u_valid[in_mask]
        valid_v = v_valid[in_mask]
        filtered_pcd[:, 3:6] = rgb_image[valid_v, valid_u]

        return filtered_pcd

    @staticmethod
    def remove_table_plane(pcd_nx6: np.ndarray, distance_threshold=0.015):
        if len(pcd_nx6) < 100: return pcd_nx6
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pcd_nx6[:, :3].astype(np.float64))
        try:
            plane_model, inliers = o3d_pcd.segment_plane(distance_threshold, 3, 1000)
            outlier_mask = np.ones(len(pcd_nx6), dtype=bool)
            outlier_mask[inliers] = False
            return pcd_nx6[outlier_mask]
        except:
            return pcd_nx6

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


# ══════════════════════════════════════════════════════════════════════════════
# 4. TF 外参解析引擎
# ══════════════════════════════════════════════════════════════════════════════
def get_extrinsics_matrix(bag_path: str, source_frame="camera_depth_optical_frame", target_frame="camera_color_optical_frame"):
    """
    通过读取 /tf_static 构建变换树，并通过 BFS 搜索两个 Frame 之间的齐次变换矩阵。
    """
    print(f"[TF] 正在从 {bag_path} 解析坐标系关系...")
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    
    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}
    
    if "/tf_static" not in type_map:
        print("[WARN] Bag包中没有 /tf_static 话题，将使用单位矩阵。")
        return np.eye(4)

    # 提取静态TF图
    adj = {}
    while reader.has_next():
        topic, raw, _ = reader.read_next()
        if topic == "/tf_static":
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(raw, msg_type)
            for transform in msg.transforms:
                p = transform.header.frame_id
                c = transform.child_frame_id
                t = transform.transform.translation
                r = transform.transform.rotation
                
                # 四元数转 4x4 旋转矩阵
                qx, qy, qz, qw = r.x, r.y, r.z, r.w
                mat = np.array([
                    [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw, t.x],
                    [    2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2,     2*qy*qz - 2*qx*qw, t.y],
                    [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2, t.z],
                    [0, 0, 0, 1]
                ], dtype=np.float32)

                if p not in adj: adj[p] = []
                if c not in adj: adj[c] = []
                # P_parent = mat * P_child (若点在child系，转到parent系需乘以mat)
                adj[c].append((p, mat))
                # P_child = inv(mat) * P_parent
                adj[p].append((c, np.linalg.inv(mat)))
            break

    # BFS 寻找路径
    from collections import deque
    queue = deque([(source_frame, np.eye(4, dtype=np.float32))])
    visited = set([source_frame])
    
    while queue:
        curr, curr_mat = queue.popleft()
        if curr == target_frame:
            print(f"[TF] 成功找到 {source_frame} -> {target_frame} 的变换矩阵！")
            return curr_mat
            
        for nxt, step_mat in adj.get(curr, []):
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, step_mat @ curr_mat))
                
    print(f"[WARN] 未能在TF树中找到 {source_frame} 到 {target_frame} 的路径！使用单位矩阵。")
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


# ══════════════════════════════════════════════════════════════════════════════
# 5. 主流程
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", default="duck_1_0.db3")
    parser.add_argument("--output", default="duck_final_tf.pcd")
    parser.add_argument("--sam2_checkpoint", default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--skip_frames", type=int, default=1)
    parser.add_argument("--voxel_size", type=float, default=0.003)
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  Duck 3D Point Cloud Pipeline (TF Extrinsics Aligned)")
    print("="*60)

    reader = BagReader(args.bag)
    all_frames = reader.read_all(max_frames=args.max_frames)
    if not all_frames: sys.exit(1)
    if args.skip_frames > 1: all_frames = all_frames[::args.skip_frames]

    frames_rgb = [f[0] for f in all_frames]
    frames_pcd = [f[1] for f in all_frames]

    intrinsics = get_intrinsics_from_bag(args.bag)
    print(f"[INFO] 相机内参: fx={intrinsics[0]:.2f}, fy={intrinsics[1]:.2f}, cx={intrinsics[2]:.2f}, cy={intrinsics[3]:.2f}")

    # 获取动态 TF 外参变换矩阵
    extrinsics_matrix = get_extrinsics_matrix(args.bag)

    segmenter = DuckSegmenter(sam2_checkpoint=args.sam2_checkpoint)
    pos, neg = segmenter.get_first_frame_clicks(frames_rgb[0])
    masks = segmenter.segment_all_frames_sam2(frames_rgb, pos, neg)

    pf = PointCloudFilter()
    all_duck_points = []
    
    print(f"\n[PCD] 开始逐帧点云过滤 (应用 TF 外参修正)...")
    for i, (pcd_raw, mask) in enumerate(zip(frames_pcd, masks)):
        if mask is None or mask.sum() == 0: continue
        h, w = frames_rgb[i].shape[:2]
        if mask.shape != (h, w): mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

        filtered = pf.filter_by_mask_and_depth(pcd_raw, frames_rgb[i], mask, intrinsics, extrinsics_matrix)
        if len(filtered) < 10: continue

        filtered = pf.voxel_downsample(filtered, voxel_size=args.voxel_size)
        all_duck_points.append(filtered)
        if i % 50 == 0: print(f"[PCD] 帧 {i}: 提取 {len(filtered)} 点")

    merged = np.vstack(all_duck_points)
    merged = pf.remove_table_plane(merged)
    merged = pf.voxel_downsample(merged, voxel_size=args.voxel_size)

    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(merged[:, :3].astype(np.float64))
    if merged.shape[1] >= 6: final_pcd.colors = o3d.utility.Vector3dVector(np.clip(merged[:, 3:6] / 255.0, 0, 1).astype(np.float64))

    final_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    o3d.io.write_point_cloud(args.output, final_pcd)
    print(f"\n✅ 成功保存 TF 对齐点云到: {args.output} (共 {len(final_pcd.points)} 点)")

if __name__ == "__main__":
    main()
