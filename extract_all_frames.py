#!/usr/bin/env python3
"""
从 rosbag 提取所有帧目标物体点云并融合
策略：SAM2 视频模式（只点击第一帧，自动传播） + 所有帧叠加 + 体素降采样去重
"""

import argparse
import os
import sys
import shutil
import tempfile
import numpy as np
import cv2
import open3d as o3d
from collections import deque
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py


# ══════════════════════════════════════════
# 1. 工具函数
# ══════════════════════════════════════════
def _extract_field_fast(data, offset, step, n, dtype):
    itemsize = np.dtype(dtype).itemsize
    row_starts = np.arange(n) * step + offset
    indices = (row_starts[:, None] + np.arange(itemsize)[None, :]).ravel()
    return np.frombuffer(data[indices].tobytes(), dtype=dtype)


# ══════════════════════════════════════════
# 2. Bag 读取：提取所有配对帧
# ══════════════════════════════════════════
class BagReader:
    def __init__(self, bag_path):
        self.bag_path = bag_path
        sr = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
        cr = rosbag2_py.ConverterOptions("", "")
        self.reader = rosbag2_py.SequentialReader()
        self.reader.open(sr, cr)
        self.type_map = {t.name: t.type for t in self.reader.get_all_topics_and_types()}

    def read_all_frames(self, sync_thresh_ms=200.0, max_frames=None):
        """
        读取所有配对帧，返回 list of (rgb_image, pcd_nx6)
        sync_thresh_ms: 图像与点云时间戳差阈值（毫秒）
        max_frames: 最多读取多少帧（None=全部）
        """
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

            # 尝试配对
            while color_buf and points_buf:
                tc, ic = color_buf[0]
                tp, pp = points_buf[0]
                dt = abs(tc - tp) / 1e6  # ns -> ms
                if dt < sync_thresh_ms:
                    paired.append((ic, pp))
                    color_buf.popleft()
                    points_buf.popleft()
                    print(f"  配对第 {len(paired)} 帧，时间差 {dt:.1f}ms，点数 {len(pp)}")
                    if max_frames and len(paired) >= max_frames:
                        return paired
                elif tc < tp:
                    color_buf.popleft()
                else:
                    points_buf.popleft()

        return paired

    @staticmethod
    def _decode_image(msg):
        data = np.frombuffer(msg.data, dtype=np.uint8)
        enc = msg.encoding.lower()
        if enc in ("rgb8", "rgb"):
            return data.reshape(msg.height, msg.width, 3)
        elif enc in ("bgr8", "bgr"):
            return data.reshape(msg.height, msg.width, 3)[:, :, ::-1].copy()
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
            result[:, i] = _extract_field_fast(data, fields[name].offset, step, n, np.float32)
        if "rgb" in fields:
            raw = _extract_field_fast(data, fields["rgb"].offset, step, n, np.float32)
            rgb_int = raw.view(np.uint32)
            result[:, 3] = ((rgb_int >> 16) & 0xFF).astype(np.float32)
            result[:, 4] = ((rgb_int >> 8) & 0xFF).astype(np.float32)
            result[:, 5] = (rgb_int & 0xFF).astype(np.float32)
        else:
            result[:, 3:] = 200.0
        valid = np.isfinite(result[:, :3]).all(axis=1) & (result[:, 2] > 0.01)
        return result[valid]


# ══════════════════════════════════════════
# 3. 相机参数（从 bag 的 camera_info 话题读取）
# ══════════════════════════════════════════
def get_color_intrinsics_from_bag(bag_path):
    """从 /camera/color/camera_info 读取真实内参"""
    sr = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    cr = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(sr, cr)
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}

    while reader.has_next():
        topic, raw, _ = reader.read_next()
        if topic == "/camera/color/camera_info":
            msg = deserialize_message(raw, get_message(type_map[topic]))
            K = msg.k  # 行主序 3x3
            fx, fy = K[0], K[4]
            cx, cy = K[2], K[5]
            W, H = msg.width, msg.height
            print(f"[内参] fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f} W={W} H={H}")
            return (fx, fy, cx, cy, W, H)

    # 默认回退值
    print("[WARN] 未找到 camera_info，使用默认内参")
    return (691.33, 691.51, 643.92, 362.12, 1280, 720)


def get_extrinsics_depth_to_color():
    """depth_optical_frame -> color_optical_frame 外参（从 /tf_static 或硬编码）"""
    # 你的设备实测值：平移很小（毫米级），旋转接近单位阵
    mat = np.eye(4, dtype=np.float64)
    mat[0, 3] = -0.0001   # -0.1 mm
    mat[1, 3] = -0.0020   # -2.0 mm
    mat[2, 3] =  0.0134   # 13.4 mm
    return mat


# ══════════════════════════════════════════
# 4. SAM2 视频推理：只点击第一帧，传播全部
# ══════════════════════════════════════════
class VideoSegmenter:
    def __init__(self, sam2_checkpoint):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[SAM2] 使用设备: {device}")
        # 根据 checkpoint 名称自动选配置
        if "tiny" in sam2_checkpoint:
            cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        elif "small" in sam2_checkpoint:
            cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        elif "base" in sam2_checkpoint:
            cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        else:
            cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.predictor = build_sam2_video_predictor(cfg, sam2_checkpoint, device=device)
        self.device = device

    def segment_all_frames(self, frames_rgb, tmp_dir):
        """
        输入: frames_rgb - list of np.ndarray (H,W,3) RGB
        输出: masks - list of np.ndarray (H,W) bool，与输入帧一一对应
        流程: 保存为 JPEG 序列 -> SAM2 视频模式 -> 只点击第一帧 -> 传播
        """
        # ---- 保存帧到临时目录 ----
        video_dir = os.path.join(tmp_dir, "frames")
        os.makedirs(video_dir, exist_ok=True)
        for i, img in enumerate(frames_rgb):
            Image.fromarray(img).save(os.path.join(video_dir, f"{i:05d}.jpg"), quality=95)
        print(f"[SAM2] 已保存 {len(frames_rgb)} 帧到临时目录")

        # ---- 在第一帧上交互式点击 ----
        pos_pts, neg_pts = self._interactive_click(frames_rgb[0])
        if not pos_pts:
            raise RuntimeError("未提供正样本点，无法分割")

        # ---- SAM2 视频推理 ----
        print("[SAM2] 开始视频推理，传播所有帧...")
        masks = [None] * len(frames_rgb)

        with torch.inference_mode():
            state = self.predictor.init_state(video_path=video_dir)

            pts_arr = np.array(pos_pts + neg_pts, dtype=np.float32)
            lbs_arr = np.array([1] * len(pos_pts) + [0] * len(neg_pts), dtype=np.int32)
            self.predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                points=pts_arr,
                labels=lbs_arr,
            )

            for frame_idx, obj_ids, logits in self.predictor.propagate_in_video(state):
                # logits shape: (n_obj, 1, H, W)
                mask = (logits[0, 0] > 0.0).cpu().numpy().astype(bool)
                masks[frame_idx] = mask
                if frame_idx % 10 == 0:
                    print(f"  传播进度: {frame_idx+1}/{len(frames_rgb)} 帧，mask像素={mask.sum()}")

        # 填补 None（极少情况 SAM2 跳帧）
        last_valid = None
        for i in range(len(masks)):
            if masks[i] is not None:
                last_valid = masks[i]
            elif last_valid is not None:
                masks[i] = last_valid
            else:
                masks[i] = np.zeros(frames_rgb[0].shape[:2], dtype=bool)

        print(f"[SAM2] 推理完成，共 {len(masks)} 帧 mask")
        return masks

    @staticmethod
    def _interactive_click(rgb_image):
        """交互式点击，返回 (pos_pts, neg_pts)"""
        h, w = rgb_image.shape[:2]
        scale = min(1.0, 1280 / w, 720 / h)
        disp = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        disp = cv2.resize(disp, (int(w * scale), int(h * scale)))
        pos, neg = [], []

        def on_mouse(event, x, y, flags, param):
            ox, oy = x / scale, y / scale
            if event == cv2.EVENT_LBUTTONDOWN:
                pos.append([ox, oy])
            elif event == cv2.EVENT_RBUTTONDOWN:
                neg.append([ox, oy])

        cv2.namedWindow("第一帧点击 (左键=目标 右键=背景 Enter=确认)", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("第一帧点击 (左键=目标 右键=背景 Enter=确认)", on_mouse)

        while True:
            img = disp.copy()
            for p in pos:
                cv2.circle(img, (int(p[0] * scale), int(p[1] * scale)), 8, (0, 255, 0), -1)
            for p in neg:
                cv2.circle(img, (int(p[0] * scale), int(p[1] * scale)), 8, (0, 0, 255), -1)
            cv2.putText(img, "Left=Pos  Right=Neg  Enter=Confirm  R=Reset",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("第一帧点击 (左键=目标 右键=背景 Enter=确认)", img)
            key = cv2.waitKey(30) & 0xFF
            if key == 13 and pos:   # Enter
                break
            elif key == ord('r'):   # R 键重置
                pos.clear()
                neg.clear()
                print("  已重置点击")

        cv2.destroyAllWindows()
        print(f"[交互] 正样本 {len(pos)} 个，负样本 {len(neg)} 个")
        return pos, neg


# ══════════════════════════════════════════
# 5. 点云过滤：用 mask 投影过滤单帧点云
# ══════════════════════════════════════════
def filter_points_by_mask(pcd_nx6, rgb_image, mask, color_intrinsics, extrinsics_d2c):
    fx, fy, cx, cy, W, H = color_intrinsics
    pts3d = pcd_nx6[:, :3].astype(np.float64)
    N = len(pts3d)

    # depth 坐标系 -> color 坐标系
    pts_h = np.hstack([pts3d, np.ones((N, 1))])
    pts_c = (extrinsics_d2c @ pts_h.T).T
    Xc, Yc, Zc = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]

    valid_z = Zc > 0.05
    u = np.full(N, -1, dtype=np.int32)
    v = np.full(N, -1, dtype=np.int32)
    u[valid_z] = np.round(Xc[valid_z] / Zc[valid_z] * fx + cx).astype(np.int32)
    v[valid_z] = np.round(Yc[valid_z] / Zc[valid_z] * fy + cy).astype(np.int32)

    in_bounds = valid_z & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    in_mask = np.zeros(N, dtype=bool)
    idx = np.where(in_bounds)[0]
    in_mask[idx] = mask[v[idx], u[idx]]

    result = pcd_nx6[in_mask].copy()
    # 用彩色图像重新着色（比点云自带 RGB 更准）
    if result.shape[1] >= 6 and len(result) > 0:
        kept = np.where(in_mask)[0]
        result[:, 3] = rgb_image[v[kept], u[kept], 0].astype(np.float32)
        result[:, 4] = rgb_image[v[kept], u[kept], 1].astype(np.float32)
        result[:, 5] = rgb_image[v[kept], u[kept], 2].astype(np.float32)
    return result


# ══════════════════════════════════════════
# 6. 多帧融合：叠加 + 体素降采样 + 统计滤波
# ══════════════════════════════════════════
def merge_and_denoise(all_pts_list, voxel_size=0.002, stat_nb=30, stat_ratio=2.0):
    """
    all_pts_list: list of np.ndarray (N,6)
    voxel_size: 体素大小（米），控制最终点云密度，越小越密但越慢
    """
    if not all_pts_list:
        return None

    # 拼接所有帧
    merged = np.vstack(all_pts_list)
    print(f"[融合] 所有帧叠加后总点数: {len(merged)}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged[:, :3].astype(np.float64))
    if merged.shape[1] >= 6:
        pcd.colors = o3d.utility.Vector3dVector(
            np.clip(merged[:, 3:6] / 255.0, 0, 1).astype(np.float64)
        )

    # 体素降采样去重（消除重叠帧的冗余点，并做颜色平均）
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"[融合] 体素降采样后 (voxel={voxel_size}m): {len(pcd.points)} 点")

    # 统计滤波去噪（去掉孤立的噪声点）
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=stat_nb, std_ratio=stat_ratio)
    print(f"[融合] 统计滤波后: {len(pcd.points)} 点")

    return pcd


# ══════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="从 rosbag 提取所有帧目标点云并融合")
    parser.add_argument("--bag",              required=True,  help="rosbag .db3 路径")
    parser.add_argument("--sam2_checkpoint",  required=True,  help="SAM2 checkpoint 路径")
    parser.add_argument("--output",           default="merged_object.pcd", help="输出 .pcd 路径")
    parser.add_argument("--voxel_size",       type=float, default=0.002,   help="体素大小(m)，默认2mm")
    parser.add_argument("--max_frames",       type=int,   default=None,    help="最多处理多少帧（调试用）")
    parser.add_argument("--sync_thresh_ms",   type=float, default=200.0,   help="图像与点云同步时间阈值(ms)")
    parser.add_argument("--skip_frames",      type=int,   default=1,       help="每隔 N 帧取一帧（1=全取）")
    args = parser.parse_args()

    tmp_dir = tempfile.mkdtemp(prefix="sam2_multiframe_")
    print(f"[临时目录] {tmp_dir}")

    try:
        # ── Step 1: 读取所有配对帧 ──────────────────────────────
        print("\n[Step 1] 读取 Bag 所有配对帧...")
        reader = BagReader(args.bag)
        all_frames = reader.read_all_frames(
            sync_thresh_ms=args.sync_thresh_ms,
            max_frames=args.max_frames
        )

        if not all_frames:
            print("[ERROR] 未找到任何配对帧，请检查话题名和时间戳同步")
            return

        # 跳帧采样
        if args.skip_frames > 1:
            all_frames = all_frames[::args.skip_frames]
        print(f"[Step 1] 共 {len(all_frames)} 帧将被处理")

        frames_rgb = [f[0] for f in all_frames]
        frames_pcd = [f[1] for f in all_frames]

        # ── Step 2: 读取相机参数 ────────────────────────────────
        print("\n[Step 2] 读取相机参数...")
        color_intr = get_color_intrinsics_from_bag(args.bag)
        extrinsics = get_extrinsics_depth_to_color()

        # ── Step 3: SAM2 视频分割（只点击第一帧）──────────────
        print("\n[Step 3] SAM2 视频分割...")
        segmenter = VideoSegmenter(args.sam2_checkpoint)
        masks = segmenter.segment_all_frames(frames_rgb, tmp_dir)

        # ── Step 4: 逐帧提取目标点云 ───────────────────────────
        print("\n[Step 4] 逐帧提取目标点云...")
        all_pts = []
        for i, (rgb, pcd_raw, mask) in enumerate(zip(frames_rgb, frames_pcd, masks)):
            if mask is None or mask.sum() < 30:
                print(f"  帧 {i:03d}: mask 为空，跳过")
                continue
            pts = filter_points_by_mask(pcd_raw, rgb, mask, color_intr, extrinsics)
            if len(pts) > 0:
                all_pts.append(pts)
                if i % 10 == 0:
                    print(f"  帧 {i:03d}: 提取 {len(pts)} 点，mask 像素 {mask.sum()}")

        print(f"\n共 {len(all_pts)}/{len(frames_rgb)} 帧提取到有效点云")

        if not all_pts:
            print("[ERROR] 所有帧均未提取到点云！检查 mask 或外参。")
            return

        # ── Step 5: 融合、降采样、去噪 ─────────────────────────
        print("\n[Step 5] 融合所有帧点云...")
        final_pcd = merge_and_denoise(
            all_pts,
            voxel_size=args.voxel_size,
            stat_nb=30,
            stat_ratio=2.0
        )

        if final_pcd is None or len(final_pcd.points) == 0:
            print("[ERROR] 融合后点云为空")
            return

        # ── Step 6: 保存 ────────────────────────────────────────
        o3d.io.write_point_cloud(args.output, final_pcd)
        print(f"\n✅ 完成！最终点云点数: {len(final_pcd.points)}")
        print(f"   保存路径: {args.output}")

        # 可选：可视化预览
        print("\n[预览] 关闭窗口以退出...")
        o3d.visualization.draw_geometries(
            [final_pcd],
            window_name="融合点云预览",
            width=1280, height=720
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("[清理] 临时目录已删除")


if __name__ == "__main__":
    main()
