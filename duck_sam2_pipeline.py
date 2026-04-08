#!/usr/bin/env python3
"""
Duck 3D Point Cloud Extraction Pipeline
========================================
策略：
  1. 直接读 /camera/depth/points (XYZRGB, 已硬件对齐) → 跳过内参反投影，避免尺寸不匹配
  2. 同时读 /camera/color/image_raw → 给SAM2做视觉分割
  3. 时间戳缓冲区配对 (ΔT < 33ms)
  4. SAM2 视频传播：第1帧手动点击，后续帧自动追踪（解决相机运动漂移）
  5. 对每帧XYZRGB点云，用SAM2 mask的RGB对应关系过滤出鸭子点
  6. 桌面RANSAC平面剔除 + 统计滤波去噪
  7. 所有帧累积 → 最终致密点云

依赖安装：
  pip install rclpy rosbag2_py sensor_msgs_py open3d numpy opencv-python
  pip install torch torchvision
  # SAM2: pip install git+https://github.com/facebookresearch/sam2.git
  # 或者: pip install sam-2

用法：
  python duck_sam2_pipeline.py --bag duck_1_0.db3 --output duck_final_v2.pcd
  python duck_sam2_pipeline.py --bag duck_1_0.db3 --output duck_final_v2.pcd --max_frames 100
  python duck_sam2_pipeline.py --bag duck_1_0.db3 --output duck_final_v2.pcd --skip_frames 2
"""

import argparse
import os
import sys
import time
import struct
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
    """顺序读取ROS2 bag，返回按话题分类的消息"""

    TOPICS = {
        "color": "/camera/color/image_raw",
        "depth_points": "/camera/depth/points",   # XYZRGB点云（已对齐）
    }

    def __init__(self, bag_path: str):
        self.bag_path = bag_path
        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
        converter_options = rosbag2_py.ConverterOptions("", "")
        self.reader = rosbag2_py.SequentialReader()
        self.reader.open(storage_options, converter_options)

        # 获取话题类型映射
        topic_types = self.reader.get_all_topics_and_types()
        self.type_map = {t.name: t.type for t in topic_types}
        print("[BAG] 话题列表：")
        for t in topic_types:
            print(f"      {t.name}  ({t.type})")

    def read_all(self, max_frames=None):
        """
        时间戳配对读取，返回 [(color_img_rgb, pcd_points_Nx6), ...]
        pcd_points_Nx6: [X, Y, Z, R, G, B]  (float32, RGB 0-255)
        """
        # 缓冲区
        color_buf = deque()   # (timestamp_ns, np.ndarray HxWx3 RGB)
        points_buf = deque()  # (timestamp_ns, np.ndarray Nx6)

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

            # 尝试配对
            while color_buf and points_buf:
                tc, ic = color_buf[0]
                tp, pp = points_buf[0]
                dt_ms = abs(tc - tp) / 1e6  # ns → ms

                if dt_ms < 33.0:  # 配对成功
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
                    color_buf.popleft()  # color太老，丢弃
                else:
                    points_buf.popleft()  # points太老，丢弃

        print(f"[BAG] 共配对 {len(paired)} 帧")
        return paired

    @staticmethod
    def _decode_image(msg):
        """将ROS Image msg解码为RGB numpy数组"""
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
            print(f"[WARN] 未知图像编码: {enc}")
            return None
        return img  # RGB

    @staticmethod
    def _decode_pointcloud2(msg):
        """
        将ROS PointCloud2 (XYZRGB) 解码为 Nx6 float32 数组
        列顺序: [X, Y, Z, R, G, B]  R/G/B 范围 0-255
        """
        # 解析字段偏移
        fields = {f.name: f for f in msg.fields}
        if not all(k in fields for k in ("x", "y", "z")):
            print("[WARN] 点云缺少XYZ字段")
            return None

        point_step = msg.point_step
        data = np.frombuffer(msg.data, dtype=np.uint8)
        n_points = msg.width * msg.height

        result = np.zeros((n_points, 6), dtype=np.float32)

        # XYZ
        for i, name in enumerate(["x", "y", "z"]):
            f = fields[name]
            offsets = np.arange(f.offset, len(data), point_step)
            result[:, i] = np.frombuffer(
                bytes(data[np.concatenate([np.arange(o, o+4) for o in offsets])]),
                dtype=np.float32
            ) if False else _extract_field_fast(data, f.offset, point_step, n_points, np.float32)

        # RGB (packed as float32 or uint32)
        if "rgb" in fields:
            f = fields["rgb"]
            raw_rgb = _extract_field_fast(data, f.offset, point_step, n_points, np.float32)
            rgb_int = raw_rgb.view(np.uint32)
            result[:, 3] = ((rgb_int >> 16) & 0xFF).astype(np.float32)  # R
            result[:, 4] = ((rgb_int >> 8) & 0xFF).astype(np.float32)   # G
            result[:, 5] = (rgb_int & 0xFF).astype(np.float32)          # B
        else:
            # 无RGB信息，用白色填充
            result[:, 3:] = 255.0

        # 过滤NaN/Inf
        valid = np.isfinite(result[:, :3]).all(axis=1)
        # 过滤深度为0的点
        valid &= (result[:, 2] > 0.01)
        return result[valid]


def _extract_field_fast(data: np.ndarray, offset: int, step: int, n: int, dtype):
    """从packed点云数据中快速提取单个字段"""
    itemsize = np.dtype(dtype).itemsize
    # 构建索引
    row_starts = np.arange(n) * step + offset
    indices = (row_starts[:, None] + np.arange(itemsize)[None, :]).ravel()
    return np.frombuffer(data[indices].tobytes(), dtype=dtype)


# ══════════════════════════════════════════════════════════════════════════════
# 2. SAM2 视频分割器
# ══════════════════════════════════════════════════════════════════════════════
class DuckSegmenter:
    """
    第1帧：CLI提示用户点击坐标（或自动用鼠标点击）→ 记录正负样本点
    后续帧：SAM2自动传播 mask
    """

    def __init__(self, sam2_checkpoint: str = None, sam1_checkpoint: str = None):
        self.use_sam2 = SAM2_AVAILABLE and sam2_checkpoint and os.path.exists(sam2_checkpoint)

        if self.use_sam2:
            print(f"[SEG] 加载 SAM2: {sam2_checkpoint}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[SEG] 使用设备: {device}")
            # SAM2 视频预测器
            # 根据checkpoint名自动选择model_cfg
            if "tiny" in sam2_checkpoint:
                cfg = "sam2_hiera_t.yaml"
            elif "small" in sam2_checkpoint:
                cfg = "sam2_hiera_s.yaml"
            elif "base_plus" in sam2_checkpoint:
                cfg = "sam2_hiera_b+.yaml"
            else:
                cfg = "sam2_hiera_l.yaml"
            self.predictor = build_sam2_video_predictor(cfg, sam2_checkpoint, device=device)
            self.device = device
            self.inference_state = None
        else:
            print(f"[SEG] 使用 SAM1 逐帧模式: {sam1_checkpoint}")
            device = "cpu"  # RTX 5060 Ti (sm_120) 不兼容，强制CPU
            print(f"[SEG] 使用设备: cpu (RTX 5060 Ti sm_120 不兼容)")
            sam = sam_model_registry["vit_h"](checkpoint=sam1_checkpoint)
            sam.to(device)
            from segment_anything import SamPredictor
            self.predictor = SamPredictor(sam)
            self.device = device
            self.inference_state = None

    def get_first_frame_clicks(self, rgb_image: np.ndarray):
        """
        在第一帧上收集用户点击的正/负样本点。
        支持两种方式：
          1. GUI (cv2 窗口)：左键=正样本，右键=负样本
          2. CLI fallback：输入坐标
        返回：
          pos_points: [[x,y], ...] 正样本（鸭子）
          neg_points: [[x,y], ...] 负样本（桌面/背景）
        """
        h, w = rgb_image.shape[:2]
        display = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        # 缩放显示（如果太大）
        scale = min(1.0, 1280 / w, 720 / h)
        if scale < 1.0:
            display = cv2.resize(display, (int(w * scale), int(h * scale)))

        pos_points = []
        neg_points = []
        confirmed = [False]

        print("\n" + "="*60)
        print("  第一帧标注说明：")
        print("  左键点击  → 正样本（鸭子身体，建议点3-5个）")
        print("  右键点击  → 负样本（桌面/键盘，建议点2-3个）")
        print("  按 ENTER  → 确认并开始处理")
        print("  按 Z      → 撤销最后一个点")
        print("="*60 + "\n")

        canvas = [display.copy()]

        def draw_points():
            img = canvas[0].copy()
            for p in pos_points:
                px, py = int(p[0] * scale), int(p[1] * scale)
                cv2.circle(img, (px, py), 8, (0, 255, 0), -1)
                cv2.circle(img, (px, py), 9, (255, 255, 255), 2)
            for p in neg_points:
                px, py = int(p[0] * scale), int(p[1] * scale)
                cv2.circle(img, (px, py), 8, (0, 0, 255), -1)
                cv2.circle(img, (px, py), 9, (255, 255, 255), 2)
            label = f"正样本(绿): {len(pos_points)}  负样本(红): {len(neg_points)}  [ENTER确认] [Z撤销]"
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            return img

        all_points = []  # [(x, y, is_pos), ...]

        def on_mouse(event, x, y, flags, param):
            rx, ry = x / scale, y / scale  # 还原真实坐标
            if event == cv2.EVENT_LBUTTONDOWN:
                pos_points.append([rx, ry])
                all_points.append((rx, ry, True))
                print(f"  ✅ 正样本: ({rx:.0f}, {ry:.0f})")
            elif event == cv2.EVENT_RBUTTONDOWN:
                neg_points.append([rx, ry])
                all_points.append((rx, ry, False))
                print(f"  ❌ 负样本: ({rx:.0f}, {ry:.0f})")

        cv2.namedWindow("Duck Annotation - Frame 0", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Duck Annotation - Frame 0", on_mouse)

        gui_ok = True
        try:
            while True:
                cv2.imshow("Duck Annotation - Frame 0", draw_points())
                key = cv2.waitKey(30) & 0xFF
                if key == 13 or key == ord(' '):  # ENTER 或空格
                    if len(pos_points) == 0:
                        print("[WARN] 请至少点击一个正样本（鸭子）！")
                        continue
                    break
                elif key == ord('z') or key == ord('Z'):
                    if all_points:
                        rx, ry, is_pos = all_points.pop()
                        if is_pos:
                            pos_points.pop()
                            print(f"  撤销正样本: ({rx:.0f}, {ry:.0f})")
                        else:
                            neg_points.pop()
                            print(f"  撤销负样本: ({rx:.0f}, {ry:.0f})")
                elif key == 27:  # ESC
                    print("[INFO] 用户取消，切换到CLI模式")
                    gui_ok = False
                    break
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"[WARN] GUI模式失败: {e}，切换到CLI模式")
            gui_ok = False

        if not gui_ok or len(pos_points) == 0:
            pos_points, neg_points = self._cli_input(h, w)

        return np.array(pos_points), np.array(neg_points)

    @staticmethod
    def _cli_input(h, w):
        """CLI方式输入坐标"""
        print(f"\n图像尺寸: {w}x{h}")
        print("请输入正样本坐标（鸭子身体），格式: x,y  一行一个，输入空行结束")
        pos_points = []
        while True:
            s = input("  正样本 x,y (或直接回车结束): ").strip()
            if not s:
                if pos_points:
                    break
                print("  至少输入一个正样本！")
                continue
            try:
                x, y = map(float, s.split(","))
                pos_points.append([x, y])
            except:
                print("  格式错误，请输入 x,y")

        print("请输入负样本坐标（桌面/背景），格式: x,y  一行一个，输入空行结束（可跳过）")
        neg_points = []
        while True:
            s = input("  负样本 x,y (或直接回车结束): ").strip()
            if not s:
                break
            try:
                x, y = map(float, s.split(","))
                neg_points.append([x, y])
            except:
                print("  格式错误，请输入 x,y")

        return np.array(pos_points), np.array(neg_points)

    def segment_all_frames_sam2(self, frames_rgb: list, pos_points: np.ndarray, neg_points: np.ndarray):
        """
        SAM2视频模式：用第一帧的标注传播到所有帧
        返回：masks列表，每个元素是HxW bool数组
        """
        import tempfile, shutil
        from PIL import Image

        print(f"[SAM2] 初始化视频推理，共 {len(frames_rgb)} 帧...")

        # SAM2视频预测器需要帧保存为jpg文件
        tmp_dir = tempfile.mkdtemp(prefix="sam2_frames_")
        try:
            print(f"[SAM2] 临时帧目录: {tmp_dir}")
            for i, rgb in enumerate(frames_rgb):
                Image.fromarray(rgb).save(os.path.join(tmp_dir, f"{i:05d}.jpg"), quality=95)

            # 初始化推理状态
            with torch.inference_mode():
                inference_state = self.predictor.init_state(video_path=tmp_dir)

                # 在第0帧添加正/负样本点
                points = []
                labels = []
                for p in pos_points:
                    points.append(p)
                    labels.append(1)
                for p in neg_points:
                    points.append(p)
                    labels.append(0)

                points_tensor = np.array(points, dtype=np.float32)
                labels_tensor = np.array(labels, dtype=np.int32)

                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=1,  # 鸭子的对象ID
                    points=points_tensor,
                    labels=labels_tensor,
                )
                print(f"[SAM2] 第0帧初始化完成，对象ID={out_obj_ids}")

                # 传播到所有帧
                masks = [None] * len(frames_rgb)
                print("[SAM2] 开始视频传播...")

                for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                    # out_mask_logits: [n_obj, 1, H, W]
                    mask = (out_mask_logits[0, 0] > 0.0).cpu().numpy()
                    masks[out_frame_idx] = mask
                    if out_frame_idx % 100 == 0:
                        coverage = mask.sum() / mask.size * 100
                        print(f"[SAM2] 帧 {out_frame_idx}/{len(frames_rgb)}: mask覆盖 {coverage:.1f}%")

            # 处理None（未覆盖的帧）
            h, w = frames_rgb[0].shape[:2]
            for i, m in enumerate(masks):
                if m is None:
                    masks[i] = np.zeros((h, w), dtype=bool)
            return masks

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def segment_frame_sam1(self, rgb_image: np.ndarray, pos_points: np.ndarray, neg_points: np.ndarray):
        """
        SAM1逐帧模式（不推荐，相机运动时会漂移）
        """
        self.predictor.set_image(rgb_image)

        all_pts = []
        all_lbs = []
        for p in pos_points:
            all_pts.append(p)
            all_lbs.append(1)
        for p in neg_points:
            all_pts.append(p)
            all_lbs.append(0)

        masks, scores, _ = self.predictor.predict(
            point_coords=np.array(all_pts, dtype=np.float32),
            point_labels=np.array(all_lbs, dtype=np.int32),
            multimask_output=True,
        )
        # 选得分最高的mask
        best = masks[np.argmax(scores)]
        return best


# ══════════════════════════════════════════════════════════════════════════════
# 3. 点云过滤器
# ══════════════════════════════════════════════════════════════════════════════
class PointCloudFilter:
    """
    对XYZRGB点云做多级过滤，只保留鸭子点云
    """

    @staticmethod
    def filter_by_mask_and_depth(
        pcd_nx6: np.ndarray,     # [X,Y,Z,R,G,B] 原始点云
        rgb_image: np.ndarray,   # HxW RGB图像（与mask对应）
        mask: np.ndarray,        # HxW bool，True=鸭子
        color_camera_intrinsics,  # (fx,fy,cx,cy)
    ) -> np.ndarray:
        """
        方法：将3D点云投影到RGB图像坐标，查询该像素是否在mask内
        这样可以精确地只保留mask内的点，不依赖颜色相似度
        
        返回：过滤后的Nx6数组
        """
        fx, fy, cx, cy = color_camera_intrinsics
        X = pcd_nx6[:, 0]
        Y = pcd_nx6[:, 1]
        Z = pcd_nx6[:, 2]

        # 投影到图像坐标
        u = (X / Z * fx + cx).astype(np.int32)
        v = (Y / Z * fy + cy).astype(np.int32)

        H, W = mask.shape
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (Z > 0)
        u_valid = u[in_bounds]
        v_valid = v[in_bounds]

        in_mask = mask[v_valid, u_valid]

        # 构建最终mask（在原始数组大小上）
        final_mask = np.zeros(len(pcd_nx6), dtype=bool)
        in_bounds_indices = np.where(in_bounds)[0]
        final_mask[in_bounds_indices[in_mask]] = True

        return pcd_nx6[final_mask]

    @staticmethod
    def remove_table_plane(pcd_nx6: np.ndarray, distance_threshold=0.015, n_iterations=1000):
        """
        用RANSAC检测并去除桌面平面
        通常桌面是点云中最大的平面
        """
        if len(pcd_nx6) < 100:
            return pcd_nx6

        pts = pcd_nx6[:, :3]
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

        try:
            plane_model, inliers = o3d_pcd.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=3,
                num_iterations=n_iterations,
            )
            a, b, c, d = plane_model
            # 只有法向量接近竖直（桌面）才去除
            # 桌面法向量应该接近 (0, -1, 0) 或 (0, 0, -1)（取决于相机朝向）
            normal = np.array([a, b, c])
            normal /= np.linalg.norm(normal)

            # 如果是水平面（法向量Y或Z分量大）才去除
            is_horizontal = abs(normal[1]) > 0.7 or abs(normal[2]) > 0.7
            inlier_ratio = len(inliers) / len(pcd_nx6)

            if is_horizontal and inlier_ratio > 0.1:
                print(f"[FILTER] RANSAC检测到桌面平面，法向量={normal.round(2)}, 内点比例={inlier_ratio:.1%}")
                outlier_mask = np.ones(len(pcd_nx6), dtype=bool)
                outlier_mask[inliers] = False
                return pcd_nx6[outlier_mask]
            else:
                print(f"[FILTER] 检测到平面但不像桌面（法向量={normal.round(2)}），跳过")
                return pcd_nx6
        except Exception as e:
            print(f"[FILTER] RANSAC失败: {e}")
            return pcd_nx6

    @staticmethod
    def statistical_filter(pcd_nx6: np.ndarray, nb_neighbors=20, std_ratio=2.0):
        """统计滤波去除孤立噪点"""
        if len(pcd_nx6) < nb_neighbors + 1:
            return pcd_nx6

        pts = pcd_nx6[:, :3]
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

        _, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        return pcd_nx6[ind]

    @staticmethod
    def voxel_downsample(pcd_nx6: np.ndarray, voxel_size=0.003):
        """体素降采样，使点云更均匀"""
        if len(pcd_nx6) < 10:
            return pcd_nx6

        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pcd_nx6[:, :3].astype(np.float64))
        if pcd_nx6.shape[1] >= 6:
            colors = pcd_nx6[:, 3:6] / 255.0
            o3d_pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        down = o3d_pcd.voxel_down_sample(voxel_size=voxel_size)
        pts = np.asarray(down.points, dtype=np.float32)
        if down.has_colors():
            colors = (np.asarray(down.colors) * 255).astype(np.float32)
            return np.hstack([pts, colors])
        return pts


# ══════════════════════════════════════════════════════════════════════════════
# 4. 主流程
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Duck 3D Point Cloud Extraction")
    parser.add_argument("--bag", default="duck_1_0.db3", help="ROS2 bag路径")
    parser.add_argument("--output", default="duck_clean_v2.pcd", help="输出PCD文件")
    parser.add_argument("--sam2_checkpoint", default=None, help="SAM2权重路径")
    parser.add_argument("--sam1_checkpoint", default="sam_vit_h_4b8939.pth", help="SAM1权重路径")
    parser.add_argument("--max_frames", type=int, default=None, help="最多处理帧数（调试用）")
    parser.add_argument("--skip_frames", type=int, default=1, help="每N帧取1帧（加速）")
    parser.add_argument("--voxel_size", type=float, default=0.003, help="体素降采样大小(m)")
    parser.add_argument("--no_ransac", action="store_true", help="跳过RANSAC桌面去除")
    parser.add_argument("--save_masks", action="store_true", help="保存分割mask图片")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  Duck 3D Point Cloud Pipeline  (SAM2 + Direct PointCloud)")
    print("="*60)

    # ── 读取Bag ──
    reader = BagReader(args.bag)
    all_frames = reader.read_all(max_frames=args.max_frames)

    if not all_frames:
        print("[ERROR] 没有配对到任何帧！检查bag话题名称。")
        sys.exit(1)

    # 跳帧
    if args.skip_frames > 1:
        all_frames = all_frames[::args.skip_frames]
        print(f"[INFO] 跳帧后剩余 {len(all_frames)} 帧")

    frames_rgb = [f[0] for f in all_frames]    # list of HxW3 RGB
    frames_pcd = [f[1] for f in all_frames]    # list of Nx6

    print(f"[INFO] RGB图像尺寸: {frames_rgb[0].shape}")
    print(f"[INFO] 首帧点云点数: {len(frames_pcd[0])}")

    # ── 内参（用于点云→图像投影） ──
    # 从bag的camera_info读取
    intrinsics = get_intrinsics_from_bag(args.bag)
    print(f"[INFO] 相机内参: fx={intrinsics[0]:.2f}, fy={intrinsics[1]:.2f}, "
          f"cx={intrinsics[2]:.2f}, cy={intrinsics[3]:.2f}")

    # ── 初始化分割器 ──
    segmenter = DuckSegmenter(
        sam2_checkpoint=args.sam2_checkpoint,
        sam1_checkpoint=args.sam1_checkpoint,
    )

    # ── 第一帧标注 ──
    print("\n[SEG] 请在第一帧图像上标注鸭子位置...")
    pos_points, neg_points = segmenter.get_first_frame_clicks(frames_rgb[0])
    print(f"[SEG] 正样本: {len(pos_points)} 个，负样本: {len(neg_points)} 个")

    # ── 分割所有帧 ──
    if segmenter.use_sam2:
        masks = segmenter.segment_all_frames_sam2(frames_rgb, pos_points, neg_points)
    else:
        print("[SEG] 使用SAM1逐帧模式（相机运动时可能漂移，建议用SAM2）")
        masks = []
        for i, rgb in enumerate(frames_rgb):
            if i % 50 == 0:
                print(f"[SEG] 分割帧 {i}/{len(frames_rgb)}...")
            m = segmenter.segment_frame_sam1(rgb, pos_points, neg_points)
            masks.append(m)

    # 保存mask预览
    if args.save_masks:
        os.makedirs("duck_masks_preview", exist_ok=True)
        for i, (rgb, mask) in enumerate(zip(frames_rgb[:20], masks[:20])):
            preview = rgb.copy()
            preview[mask] = (preview[mask] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
            cv2.imwrite(f"duck_masks_preview/frame_{i:04d}.jpg",
                       cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        print("[INFO] mask预览已保存到 duck_masks_preview/")

    # ── 逐帧过滤点云 ──
    pf = PointCloudFilter()
    all_duck_points = []

    print(f"\n[PCD] 开始逐帧点云过滤...")
    for i, (pcd_raw, mask) in enumerate(zip(frames_pcd, masks)):
        if mask is None or mask.sum() == 0:
            continue

        # 确保mask与RGB图像同尺寸
        h, w = frames_rgb[i].shape[:2]
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h),
                             interpolation=cv2.INTER_NEAREST).astype(bool)

        # 用mask+投影过滤点云
        filtered = pf.filter_by_mask_and_depth(pcd_raw, frames_rgb[i], mask, intrinsics)

        if len(filtered) < 10:
            continue

        # 体素降采样（减少内存）
        filtered = pf.voxel_downsample(filtered, voxel_size=args.voxel_size)
        all_duck_points.append(filtered)

        if i % 100 == 0:
            total_so_far = sum(len(p) for p in all_duck_points)
            print(f"[PCD] 帧 {i}/{len(frames_pcd)}: 本帧{len(filtered)}点, "
                  f"累计{total_so_far}点, mask面积{mask.sum()}px")

    if not all_duck_points:
        print("[ERROR] 没有提取到任何鸭子点！检查mask和内参。")
        sys.exit(1)

    # ── 合并所有帧 ──
    merged = np.vstack(all_duck_points)
    print(f"\n[PCD] 合并后总点数: {len(merged)}")

    # ── 全局统计滤波 ──
    print("[PCD] 统计滤波去噪...")
    merged = pf.statistical_filter(merged, nb_neighbors=20, std_ratio=1.5)
    print(f"[PCD] 滤波后点数: {len(merged)}")

    # ── RANSAC去桌面 ──
    if not args.no_ransac:
        print("[PCD] RANSAC去除桌面...")
        merged = pf.remove_table_plane(merged)
        print(f"[PCD] 去桌面后点数: {len(merged)}")

    # ── 最终体素降采样 ──
    print("[PCD] 最终体素降采样...")
    merged = pf.voxel_downsample(merged, voxel_size=args.voxel_size)
    print(f"[PCD] 最终点数: {len(merged)}")

    # ── 保存PCD ──
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(merged[:, :3].astype(np.float64))
    if merged.shape[1] >= 6:
        colors = merged[:, 3:6] / 255.0
        final_pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1).astype(np.float64))

    # 计算法向量（可选，某些网络需要）
    print("[PCD] 计算法向量...")
    final_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )
    final_pcd.orient_normals_consistent_tangent_plane(k=15)

    o3d.io.write_point_cloud(args.output, final_pcd)
    print(f"\n✅ 保存到: {args.output}")
    print(f"   总点数: {len(final_pcd.points)}")

    # ── 统计信息 ──
    pts = np.asarray(final_pcd.points)
    print(f"   边界框: X [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}] m")
    print(f"           Y [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}] m")
    print(f"           Z [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}] m")
    bbox_vol = (pts.max(0) - pts.min(0)).prod()
    print(f"   边界盒体积: {bbox_vol*1e6:.1f} cm³")
    print(f"   点密度: {len(pts)/bbox_vol:.0f} 点/m³")


def get_intrinsics_from_bag(bag_path: str):
    """从bag的camera_info话题读取内参"""
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import rosbag2_py

    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    # 优先用color camera info
    target = "/camera/color/camera_info"
    fallback = "/camera/depth/camera_info"

    while reader.has_next():
        topic, raw, _ = reader.read_next()
        if topic in (target, fallback):
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(raw, msg_type)
            K = msg.k  # 3x3 row-major
            fx, fy = K[0], K[4]
            cx, cy = K[2], K[5]
            if fx > 0:
                return (fx, fy, cx, cy)

    # Fallback到你已知的内参
    print("[WARN] 无法从bag读取内参，使用硬编码值")
    return (412.857, 412.857, 424.0, 237.019)


if __name__ == "__main__":
    main()
