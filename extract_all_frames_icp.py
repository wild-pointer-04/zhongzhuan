#!/usr/bin/env python3
"""
从 rosbag 提取所有帧目标物体点云并融合（位姿补偿版）

策略：
  - 用【背景点云】（mask之外的区域）做帧间ICP，估计相机相对第0帧的变换T
  - 用T把每帧【目标点云】变换到第0帧坐标系
  - 所有帧叠加 → 体素降采样 → 统计滤波
"""

import argparse
import os
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
# 1. 工具
# ══════════════════════════════════════════
def _extract_field_fast(data, offset, step, n, dtype):
    itemsize = np.dtype(dtype).itemsize
    row_starts = np.arange(n) * step + offset
    indices = (row_starts[:, None] + np.arange(itemsize)[None, :]).ravel()
    return np.frombuffer(data[indices].tobytes(), dtype=dtype)


# ══════════════════════════════════════════
# 2. Bag 读取
# ══════════════════════════════════════════
class BagReader:
    def __init__(self, bag_path):
        sr = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
        cr = rosbag2_py.ConverterOptions("", "")
        self.reader = rosbag2_py.SequentialReader()
        self.reader.open(sr, cr)
        self.type_map = {t.name: t.type for t in self.reader.get_all_topics_and_types()}

    def read_all_frames(self, sync_thresh_ms=200.0, max_frames=None):
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
                if dt < sync_thresh_ms:
                    paired.append((ic, pp))
                    color_buf.popleft()
                    points_buf.popleft()
                    if len(paired) % 10 == 0:
                        print(f"  已配对 {len(paired)} 帧...")
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
        xyz = np.zeros((n, 3), dtype=np.float32)
        for i, name in enumerate(["x", "y", "z"]):
            xyz[:, i] = _extract_field_fast(data, fields[name].offset, step, n, np.float32)
        valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0.01)
        return xyz[valid]


# ══════════════════════════════════════════
# 3. 相机内参
# ══════════════════════════════════════════
def get_color_intrinsics_from_bag(bag_path):
    sr = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    cr = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(sr, cr)
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    while reader.has_next():
        topic, raw, _ = reader.read_next()
        if topic == "/camera/color/camera_info":
            msg = deserialize_message(raw, get_message(type_map[topic]))
            K = msg.k
            fx, fy, cx, cy = K[0], K[4], K[2], K[5]
            W, H = msg.width, msg.height
            print(f"[内参] fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f} W={W} H={H}")
            return fx, fy, cx, cy, W, H
    return 691.33, 691.51, 643.92, 362.12, 1280, 720


# ══════════════════════════════════════════
# 4. SAM2
# ══════════════════════════════════════════
class VideoSegmenter:
    def __init__(self, sam2_checkpoint):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[SAM2] 使用设备: {device}")
        if "tiny" in sam2_checkpoint:
            cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        elif "small" in sam2_checkpoint:
            cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        elif "base" in sam2_checkpoint:
            cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        else:
            cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.predictor = build_sam2_video_predictor(cfg, sam2_checkpoint, device=device)

    def segment_all_frames(self, frames_rgb, tmp_dir):
        video_dir = os.path.join(tmp_dir, "frames")
        os.makedirs(video_dir, exist_ok=True)
        for i, img in enumerate(frames_rgb):
            Image.fromarray(img).save(os.path.join(video_dir, f"{i:05d}.jpg"), quality=95)
        print(f"[SAM2] 已保存 {len(frames_rgb)} 帧")
        pos_pts, neg_pts = self._interactive_click(frames_rgb[0])
        if not pos_pts:
            raise RuntimeError("未提供正样本点")
        masks = [None] * len(frames_rgb)
        with torch.inference_mode():
            state = self.predictor.init_state(video_path=video_dir)
            pts_arr = np.array(pos_pts + neg_pts, dtype=np.float32)
            lbs_arr = np.array([1]*len(pos_pts) + [0]*len(neg_pts), dtype=np.int32)
            self.predictor.add_new_points_or_box(
                inference_state=state, frame_idx=0, obj_id=1,
                points=pts_arr, labels=lbs_arr)
            for frame_idx, _, logits in self.predictor.propagate_in_video(state):
                masks[frame_idx] = (logits[0, 0] > 0.0).cpu().numpy().astype(bool)
                if frame_idx % 10 == 0:
                    print(f"  传播 {frame_idx+1}/{len(frames_rgb)}，像素={masks[frame_idx].sum()}")
        last = None
        for i in range(len(masks)):
            if masks[i] is not None:
                last = masks[i]
            elif last is not None:
                masks[i] = last
            else:
                masks[i] = np.zeros(frames_rgb[0].shape[:2], dtype=bool)
        print("[SAM2] 完成")
        return masks

    @staticmethod
    def _interactive_click(rgb_image):
        h, w = rgb_image.shape[:2]
        scale = min(1.0, 1280/w, 720/h)
        disp = cv2.resize(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR),
                          (int(w*scale), int(h*scale)))
        pos, neg = [], []
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                pos.append([x/scale, y/scale])
            elif event == cv2.EVENT_RBUTTONDOWN:
                neg.append([x/scale, y/scale])
        cv2.namedWindow("左键=目标 右键=背景 Enter=确认 R=重置", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("左键=目标 右键=背景 Enter=确认 R=重置", on_mouse)
        while True:
            img = disp.copy()
            for p in pos:
                cv2.circle(img, (int(p[0]*scale), int(p[1]*scale)), 8, (0,255,0), -1)
            for p in neg:
                cv2.circle(img, (int(p[0]*scale), int(p[1]*scale)), 8, (0,0,255), -1)
            cv2.putText(img, "Left=Pos  Right=Neg  Enter=OK  R=Reset",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.imshow("左键=目标 右键=背景 Enter=确认 R=重置", img)
            key = cv2.waitKey(30) & 0xFF
            if key == 13 and pos:
                break
            elif key == ord('r'):
                pos.clear(); neg.clear()
        cv2.destroyAllWindows()
        print(f"[交互] 正={len(pos)} 负={len(neg)}")
        return pos, neg


# ══════════════════════════════════════════
# 5. 点云分离：目标 / 背景
# ══════════════════════════════════════════
def split_foreground_background(pcd_xyz, rgb_image, mask, fx, fy, cx, cy, W, H,
                                 min_coverage=0.10, bg_sample_n=30000):
    """
    返回:
      fg_pts  (M,6) 目标点云 + RGB，或 None（盲区）
      bg_pcd  open3d.PointCloud 背景点云（用于ICP）
      coverage float
    """
    N = len(pcd_xyz)
    X = pcd_xyz[:, 0].astype(np.float64)
    Y = pcd_xyz[:, 1].astype(np.float64)
    Z = pcd_xyz[:, 2].astype(np.float64)

    valid_z = Z > 0.05
    u = np.full(N, -1, dtype=np.int32)
    v = np.full(N, -1, dtype=np.int32)
    u[valid_z] = np.round(X[valid_z] / Z[valid_z] * fx + cx).astype(np.int32)
    v[valid_z] = np.round(Y[valid_z] / Z[valid_z] * fy + cy).astype(np.int32)
    in_bounds = valid_z & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    idx_ib = np.where(in_bounds)[0]
    mask_flags = mask[v[idx_ib], u[idx_ib]]

    # 覆盖率检测
    pts_in_mask = mask_flags.sum()
    coverage = pts_in_mask / max(mask.sum(), 1)

    # 前景（目标）
    if coverage < min_coverage:
        fg_pts = None
    else:
        fg_idx = idx_ib[mask_flags]
        fg_pts = np.zeros((len(fg_idx), 6), dtype=np.float32)
        fg_pts[:, :3] = pcd_xyz[fg_idx]
        fg_pts[:, 3] = rgb_image[v[fg_idx], u[fg_idx], 0]
        fg_pts[:, 4] = rgb_image[v[fg_idx], u[fg_idx], 1]
        fg_pts[:, 5] = rgb_image[v[fg_idx], u[fg_idx], 2]

    # 背景（mask之外，用于ICP）
    bg_idx = idx_ib[~mask_flags]
    if len(bg_idx) > bg_sample_n:
        bg_idx = np.random.choice(bg_idx, bg_sample_n, replace=False)
    bg_pcd = o3d.geometry.PointCloud()
    bg_pcd.points = o3d.utility.Vector3dVector(pcd_xyz[bg_idx].astype(np.float64))

    return fg_pts, bg_pcd, coverage


# ══════════════════════════════════════════
# 6. 背景ICP：估计当前帧相对参考帧的变换
# ══════════════════════════════════════════
def estimate_transform_icp(src_bg: o3d.geometry.PointCloud,
                            ref_bg: o3d.geometry.PointCloud,
                            voxel_size=0.01):
    """
    src_bg: 当前帧背景点云
    ref_bg: 参考帧（第0帧）背景点云
    返回 4x4 变换矩阵 T，使得 T @ src ≈ ref
    """
    # 降采样
    src_d = src_bg.voxel_down_sample(voxel_size)
    ref_d = ref_bg.voxel_down_sample(voxel_size)

    if len(src_d.points) < 100 or len(ref_d.points) < 100:
        return np.eye(4)

    # 先用粗配准（point-to-point ICP，大阈值）
    threshold_coarse = voxel_size * 15
    reg_coarse = o3d.pipelines.registration.registration_icp(
        src_d, ref_d, threshold_coarse,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )

    # 再精配准（point-to-plane，小阈值）
    src_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*3, max_nn=30))
    ref_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*3, max_nn=30))
    threshold_fine = voxel_size * 3
    reg_fine = o3d.pipelines.registration.registration_icp(
        src_d, ref_d, threshold_fine,
        reg_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )

    fitness = reg_fine.fitness
    rmse = reg_fine.inlier_rmse
    T = reg_fine.transformation

    return T, fitness, rmse


# ══════════════════════════════════════════
# 7. 融合
# ══════════════════════════════════════════
def merge_and_denoise(all_pts_list, voxel_size=0.002, stat_nb=30, stat_ratio=2.0):
    merged = np.vstack(all_pts_list)
    print(f"[融合] 叠加后总点数: {len(merged)}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged[:, :3].astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(
        np.clip(merged[:, 3:6] / 255.0, 0, 1).astype(np.float64))
    pcd = pcd.voxel_down_sample(voxel_size)
    print(f"[融合] 体素降采样后: {len(pcd.points)} 点")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=stat_nb, std_ratio=stat_ratio)
    print(f"[融合] 统计滤波后: {len(pcd.points)} 点")
    return pcd


# ══════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag",             required=True)
    parser.add_argument("--sam2_checkpoint", required=True)
    parser.add_argument("--output",          default="merged_object.pcd")
    parser.add_argument("--voxel_size",      type=float, default=0.002,
                        help="最终点云体素大小(m)，默认2mm")
    parser.add_argument("--icp_voxel",       type=float, default=0.015,
                        help="ICP配准用的体素大小(m)，默认15mm")
    parser.add_argument("--min_coverage",    type=float, default=0.10,
                        help="mask内点云覆盖率阈值，低于此值丢弃（默认10%%）")
    parser.add_argument("--min_fitness",     type=float, default=0.3,
                        help="ICP fitness阈值，低于此值认为配准失败（默认0.3）")
    parser.add_argument("--max_frames",      type=int,   default=None)
    parser.add_argument("--sync_thresh_ms",  type=float, default=200.0)
    parser.add_argument("--skip_frames",     type=int,   default=1)
    args = parser.parse_args()

    tmp_dir = tempfile.mkdtemp(prefix="sam2_pose_")
    print(f"[临时目录] {tmp_dir}")

    try:
        # Step 1: 读帧
        print("\n[Step 1] 读取 Bag 所有配对帧...")
        reader = BagReader(args.bag)
        all_frames = reader.read_all_frames(
            sync_thresh_ms=args.sync_thresh_ms,
            max_frames=args.max_frames)
        if not all_frames:
            print("[ERROR] 未找到配对帧"); return
        if args.skip_frames > 1:
            all_frames = all_frames[::args.skip_frames]
        print(f"[Step 1] 共 {len(all_frames)} 帧")

        frames_rgb = [f[0] for f in all_frames]
        frames_pcd = [f[1] for f in all_frames]

        # Step 2: 内参
        print("\n[Step 2] 读取相机内参...")
        fx, fy, cx, cy, W, H = get_color_intrinsics_from_bag(args.bag)

        # Step 3: SAM2
        print("\n[Step 3] SAM2 视频分割...")
        segmenter = VideoSegmenter(args.sam2_checkpoint)
        masks = segmenter.segment_all_frames(frames_rgb, tmp_dir)

        # Step 4: 逐帧分离前景/背景
        print("\n[Step 4] 分离前景/背景...")
        fg_list  = []   # (i, fg_pts_nx6)
        bg_list  = []   # (i, bg_pcd)

        for i, (rgb, pcd_raw, mask) in enumerate(zip(frames_rgb, frames_pcd, masks)):
            if mask.sum() < 30:
                continue
            fg, bg, cov = split_foreground_background(
                pcd_raw, rgb, mask, fx, fy, cx, cy, W, H,
                min_coverage=args.min_coverage)
            if fg is None:
                if i % 10 == 0:
                    print(f"  帧{i:03d}: 覆盖率{cov:.1%} ⚠ 盲区")
                continue
            fg_list.append((i, fg))
            bg_list.append((i, bg))
            if i % 10 == 0:
                print(f"  帧{i:03d}: 覆盖率{cov:.1%} 前景{len(fg)}点 背景{len(bg.points)}点")

        print(f"有效帧: {len(fg_list)}")
        if not fg_list:
            print("[ERROR] 无有效帧"); return

        # Step 5: 背景ICP 估计位姿，变换前景点云到第0帧坐标系
        print(f"\n[Step 5] 背景ICP位姿估计（icp_voxel={args.icp_voxel}m）...")
        print(f"{'帧':>4}  {'fitness':>8}  {'RMSE(mm)':>9}  {'状态':>6}")
        print("-" * 40)

        ref_bg = bg_list[0][1]   # 第0有效帧作为参考

        all_fg_transformed = []

        for idx, ((fi, fg), (_, bg)) in enumerate(zip(fg_list, bg_list)):
            if idx == 0:
                # 参考帧直接加入
                all_fg_transformed.append(fg)
                print(f"{fi:>4}  {'ref':>8}  {'ref':>9}  参考帧")
                continue

            T, fitness, rmse = estimate_transform_icp(bg, ref_bg, voxel_size=args.icp_voxel)

            if fitness < args.min_fitness:
                print(f"{fi:>4}  {fitness:>8.3f}  {rmse*1000:>9.2f}  ✗ 配准差，跳过")
                continue

            # 变换前景点云
            pts_h = np.hstack([fg[:, :3], np.ones((len(fg), 1), dtype=np.float64)])
            pts_transformed = (T @ pts_h.astype(np.float64).T).T[:, :3]

            fg_t = fg.copy()
            fg_t[:, :3] = pts_transformed.astype(np.float32)
            all_fg_transformed.append(fg_t)

            status = "✓" if fitness > 0.6 else "△"
            print(f"{fi:>4}  {fitness:>8.3f}  {rmse*1000:>9.2f}  {status}")

        print(f"\n成功配准帧数: {len(all_fg_transformed)}")
        if not all_fg_transformed:
            print("[ERROR] 所有帧配准失败，尝试降低 --min_fitness 或增大 --icp_voxel")
            return

        # Step 6: 融合
        print("\n[Step 6] 融合所有帧...")
        final_pcd = merge_and_denoise(all_fg_transformed, voxel_size=args.voxel_size)

        # Step 7: 保存
        o3d.io.write_point_cloud(args.output, final_pcd)
        print(f"\n✅ 完成！最终点数: {len(final_pcd.points)}")
        print(f"   保存路径: {args.output}")

        print("\n[预览] 关闭窗口退出...")
        o3d.visualization.draw_geometries(
            [final_pcd], window_name="融合点云预览", width=1280, height=720)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("[清理] 完成")


if __name__ == "__main__":
    main()
