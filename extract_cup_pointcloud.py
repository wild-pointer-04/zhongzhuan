#!/usr/bin/env python3
"""
杯子点云提取工具 —— 移动相机版（无 /tf 位姿）
策略：SAM2手动分割每帧物体 → ICP帧间配准估计位姿 → 多帧融合

话题（Orbbec）：
  /camera/color/image_raw        RGB图
  /camera/depth/image_raw        深度图（uint16，单位mm）
  /camera/color/camera_info      彩色相机内参
  /camera/depth/camera_info      深度相机内参
  /tf_static                     深度→彩色外参

运行示例：
  # 有SAM2：
  python extract_cup_pointcloud.py /path/to/bag_dir \
      --sam2-checkpoint ./sam2_hiera_large.pt \
      --annotate-interval 15 --output cup.pcd

  # 无SAM2（用鼠标框选）：
  python extract_cup_pointcloud.py /path/to/bag_dir \
      --annotate-interval 15 --output cup.pcd

  # 静止相机（跳过ICP）：
  python extract_cup_pointcloud.py /path/to/bag_dir --no-icp
"""

import sys
import os
import argparse
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path

# ── ROS2 ──
try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError:
    print("[ERROR] 请先执行: source /opt/ros/humble/setup.bash")
    sys.exit(1)

# ── SAM2 ──
try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("[WARN] SAM2未安装，将使用鼠标框选(ROI)作为备用分割")


# ════════════════════════════════════════════════════════════════
# 1. ROS2 bag 读取
# ════════════════════════════════════════════════════════════════

def open_reader(bag_path: str) -> rosbag2_py.SequentialReader:
    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path, storage_id="sqlite3"
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def read_all_messages(bag_path: str, topics: list) -> dict:
    """读指定话题全部消息，返回 {topic: [(timestamp_ns, msg), ...]}"""
    reader = open_reader(bag_path)
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}

    missing = [t for t in topics if t not in topic_types]
    if missing:
        print(f"[WARN] bag中不存在以下话题: {missing}")

    data = {t: [] for t in topics}
    while reader.has_next():
        topic, raw, ts = reader.read_next()
        if topic in topics:
            msg_type = get_message(topic_types[topic])
            msg = deserialize_message(raw, msg_type)
            data[topic].append((ts, msg))

    del reader
    return data


def decode_rgb(msg) -> np.ndarray:
    """sensor_msgs/Image → BGR uint8"""
    raw = bytes(msg.data)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    enc = msg.encoding.lower()
    if enc in ("rgb8",):
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    # bgr8 / bgra8 / yuv 等按需扩展
    return arr.copy()


def decode_depth(msg) -> np.ndarray:
    """sensor_msgs/Image → float32 深度图（单位：米）"""
    raw = bytes(msg.data)
    enc = msg.encoding.lower()
    if enc in ("16uc1", "mono16"):
        arr = np.frombuffer(raw, dtype=np.uint16).reshape(msg.height, msg.width)
        return arr.astype(np.float32) * 0.001      # mm → m（Orbbec默认）
    elif enc in ("32fc1",):
        arr = np.frombuffer(raw, dtype=np.float32).reshape(msg.height, msg.width)
        return arr.copy()
    else:
        raise ValueError(f"不支持的深度编码: {msg.encoding}")


def get_intrinsics(bag_path: str, topic: str) -> dict:
    data = read_all_messages(bag_path, [topic])
    msgs = data[topic]
    if not msgs:
        raise RuntimeError(f"未找到话题: {topic}")
    _, msg = msgs[0]
    K = np.array(msg.k).reshape(3, 3)
    return {
        "fx": K[0, 0], "fy": K[1, 1],
        "cx": K[0, 2], "cy": K[1, 2],
        "width": msg.width, "height": msg.height
    }


def quaternion_to_matrix(qx, qy, qz, qw) -> np.ndarray:
    T = np.eye(4)
    T[0, 0] = 1 - 2*(qy**2 + qz**2)
    T[0, 1] = 2*(qx*qy - qz*qw)
    T[0, 2] = 2*(qx*qz + qy*qw)
    T[1, 0] = 2*(qx*qy + qz*qw)
    T[1, 1] = 1 - 2*(qx**2 + qz**2)
    T[1, 2] = 2*(qy*qz - qx*qw)
    T[2, 0] = 2*(qx*qz - qy*qw)
    T[2, 1] = 2*(qy*qz + qx*qw)
    T[2, 2] = 1 - 2*(qx**2 + qy**2)
    return T


def get_extrinsics_depth_to_color(bag_path: str) -> np.ndarray:
    """从 /tf_static 读深度→彩色外参，找不到则返回单位矩阵。"""
    data = read_all_messages(bag_path, ["/tf_static"])
    msgs = data["/tf_static"]
    for _, msg in msgs:
        for tf in msg.transforms:
            src = tf.header.frame_id
            dst = tf.child_frame_id
            # 尝试找到任意含 depth→color 的变换
            is_depth_to_color = ("depth" in src and "color" in dst) or \
                                 ("color" in src and "depth" in dst)
            if not is_depth_to_color:
                continue
            t = tf.transform.translation
            r = tf.transform.rotation
            T = quaternion_to_matrix(r.x, r.y, r.z, r.w)
            T[:3, 3] = [t.x, t.y, t.z]
            if "color" in src:
                T = np.linalg.inv(T)   # 翻转方向
            print(f"[外参] {src} → {dst}  T=\n{np.round(T, 4)}")
            return T

    print("[外参] 未找到深度→彩色TF，假设深度图已对齐（单位矩阵）")
    return np.eye(4)


def sync_frames(color_msgs: list, depth_msgs: list,
                max_dt_ns: int = 50_000_000) -> list:
    """最近邻时间戳匹配，返回 [(color_msg, depth_msg), ...]"""
    synced = []
    depth_ts = np.array([ts for ts, _ in depth_msgs], dtype=np.int64)
    for c_ts, c_msg in color_msgs:
        idx = int(np.argmin(np.abs(depth_ts - c_ts)))
        d_ts, d_msg = depth_msgs[idx]
        if abs(int(c_ts) - int(d_ts)) < max_dt_ns:
            synced.append((c_msg, d_msg))
    print(f"[同步] 彩色{len(color_msgs)} 深度{len(depth_msgs)} → 匹配{len(synced)}对")
    return synced


# ════════════════════════════════════════════════════════════════
# 2. 分割器
# ════════════════════════════════════════════════════════════════

class Segmenter:
    def __init__(self, sam2_ckpt: str = None, sam2_cfg: str = "sam2_hiera_l.yaml"):
        self.use_sam2 = SAM2_AVAILABLE and sam2_ckpt and os.path.exists(sam2_ckpt)
        if self.use_sam2:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = build_sam2(sam2_cfg, sam2_ckpt, device=device)
            self.predictor = SAM2ImagePredictor(model)
            print(f"[SAM2] 已加载 on {device}")
        else:
            self.predictor = None
            print("[分割] 使用框选模式（未安装SAM2或未指定权重）")

    def annotate(self, bgr: np.ndarray, frame_idx: int) -> np.ndarray:
        """交互标注，返回 bool mask (H, W)，用户按ESC/取消返回None。"""
        if self.use_sam2:
            return self._sam2(bgr, frame_idx)
        else:
            return self._roi(bgr, frame_idx)

    def _sam2(self, bgr: np.ndarray, frame_idx: int):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)

        pts_xy, labels = [], []
        mask_show = None
        win = f"[帧{frame_idx}] 左键前景 右键背景 Enter确认 ESC跳过"

        def on_mouse(event, x, y, flags, param):
            nonlocal mask_show
            if event == cv2.EVENT_LBUTTONDOWN:
                pts_xy.append((x, y)); labels.append(1)
            elif event == cv2.EVENT_RBUTTONDOWN:
                pts_xy.append((x, y)); labels.append(0)

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, on_mouse)

        while True:
            disp = bgr.copy()
            # 推断当前mask
            if pts_xy:
                m, scores, _ = self.predictor.predict(
                    point_coords=np.array(pts_xy, dtype=np.float32),
                    point_labels=np.array(labels, dtype=np.int32),
                    multimask_output=True
                )
                mask_show = m[int(np.argmax(scores))].astype(bool)
                # 绿色半透明覆盖
                overlay = disp.copy()
                overlay[mask_show] = (
                    overlay[mask_show] * 0.4 +
                    np.array([0, 220, 0]) * 0.6
                ).astype(np.uint8)
                disp = overlay
            # 画点
            for (px, py), lbl in zip(pts_xy, labels):
                c = (0, 255, 0) if lbl else (0, 0, 255)
                cv2.circle(disp, (px, py), 7, c, -1)
                cv2.circle(disp, (px, py), 7, (255, 255, 255), 1)

            cv2.putText(disp, "L:前景  R:背景  Enter:确认  ESC:跳过",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2)
            cv2.imshow(win, disp)
            key = cv2.waitKey(30) & 0xFF
            if key == 13 and mask_show is not None:   # Enter
                break
            if key == 27:                              # ESC
                cv2.destroyWindow(win)
                return None

        cv2.destroyWindow(win)
        return mask_show

    def _roi(self, bgr: np.ndarray, frame_idx: int):
        win = f"[帧{frame_idx}] 框选目标后 Enter/Space确认，ESC跳过"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        roi = cv2.selectROI(win, bgr, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(win)
        if roi == (0, 0, 0, 0):
            return None
        x, y, w, h = map(int, roi)
        mask = np.zeros(bgr.shape[:2], dtype=bool)
        mask[y:y+h, x:x+w] = True
        return mask


# ════════════════════════════════════════════════════════════════
# 3. 点云工具
# ════════════════════════════════════════════════════════════════

def mask_to_pointcloud(depth: np.ndarray, color: np.ndarray,
                       mask: np.ndarray,
                       c_intr: dict, d_intr: dict,
                       T_d2c: np.ndarray,
                       depth_min: float, depth_max: float
                       ) -> o3d.geometry.PointCloud:
    """
    mask区域 → 彩色相机坐标系3D点云（带颜色）。
    自动处理深度图与彩色图尺寸不一致的情况。
    """
    H_c, W_c = color.shape[:2]
    H_d, W_d = depth.shape[:2]
    aligned = (H_c == H_d and W_c == W_d)

    v_c, u_c = np.where(mask)
    if len(v_c) == 0:
        return o3d.geometry.PointCloud()

    if aligned:
        d_vals = depth[v_c, u_c]
        valid = (d_vals > depth_min) & (d_vals < depth_max) & np.isfinite(d_vals)
        v_c, u_c, d_vals = v_c[valid], u_c[valid], d_vals[valid]
        fx, fy, cx, cy = c_intr["fx"], c_intr["fy"], c_intr["cx"], c_intr["cy"]
        X = (u_c - cx) * d_vals / fx
        Y = (v_c - cy) * d_vals / fy
        Z = d_vals
    else:
        # 彩色坐标 → 深度图坐标（近似映射）
        u_d = (u_c * (W_d / W_c)).astype(int).clip(0, W_d - 1)
        v_d = (v_c * (H_d / H_c)).astype(int).clip(0, H_d - 1)
        d_vals = depth[v_d, u_d]
        valid = (d_vals > depth_min) & (d_vals < depth_max) & np.isfinite(d_vals)
        v_c, u_c = v_c[valid], u_c[valid]
        u_d, v_d = u_d[valid], v_d[valid]
        d_vals = d_vals[valid]

        dfx, dfy = d_intr["fx"], d_intr["fy"]
        dcx, dcy = d_intr["cx"], d_intr["cy"]
        xd = (u_d - dcx) * d_vals / dfx
        yd = (v_d - dcy) * d_vals / dfy

        # 深度坐标系 → 彩色坐标系
        pts_d = np.stack([xd, yd, d_vals, np.ones_like(d_vals)], axis=1)
        pts_c = (T_d2c @ pts_d.T).T
        X, Y, Z = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]

    if len(X) == 0:
        return o3d.geometry.PointCloud()

    colors_bgr = color[v_c, u_c].astype(np.float32) / 255.0
    colors_rgb = colors_bgr[:, ::-1]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack([X, Y, Z], axis=1))
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
    return pcd


def icp_register(source: o3d.geometry.PointCloud,
                 target: o3d.geometry.PointCloud,
                 voxel: float) -> tuple:
    """
    FPFH粗配准 + Point-to-Plane ICP精配准。
    返回 (4×4变换矩阵, fitness分数)
    """
    def preprocess(pcd):
        down = pcd.voxel_down_sample(voxel)
        down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=30)
        )
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 15, max_nn=100)
        )
        return down, fpfh

    src_d, src_f = preprocess(source)
    tgt_d, tgt_f = preprocess(target)

    # RANSAC粗配准
    result_coarse = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_d, tgt_d, src_f, tgt_f,
        mutual_filter=True,
        max_correspondence_distance=voxel * 15,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceEstimationNormalShooting() if False else
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 15),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )

    # ICP精配准
    # 需要法线（Point-to-Plane），直接用原始点云
    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=30)
    )
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=30)
    )
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=voxel * 3,
        init=result_coarse.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=60
        )
    )
    return result_icp.transformation, result_icp.fitness


def fuse_and_clean(pcd_list: list, voxel: float) -> o3d.geometry.PointCloud:
    all_pts, all_col = [], []
    for p in pcd_list:
        all_pts.append(np.asarray(p.points))
        if p.has_colors():
            all_col.append(np.asarray(p.colors))

    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(np.vstack(all_pts))
    if all_col and len(all_col) == len(all_pts):
        merged.colors = o3d.utility.Vector3dVector(np.vstack(all_col))

    print(f"  合并前总点数: {len(merged.points)}")
    merged = merged.voxel_down_sample(voxel)
    print(f"  体素降采样后: {len(merged.points)}")
    merged, _ = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"  统计滤波后:   {len(merged.points)}")
    return merged


# ════════════════════════════════════════════════════════════════
# 4. 主流程
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag",
                        help="ROS2 bag路径（目录或.db3文件）")
    parser.add_argument("--sam2-checkpoint", default=None,
                        help="SAM2权重文件，如 ./sam2_hiera_large.pt")
    parser.add_argument("--sam2-cfg", default="sam2_hiera_l.yaml",
                        help="SAM2配置名（默认 sam2_hiera_l.yaml）")
    parser.add_argument("--annotate-interval", type=int, default=15,
                        help="手动标注间隔（帧数，默认15）")
    parser.add_argument("--depth-min", type=float, default=0.1,
                        help="最近有效深度（米，默认0.1）")
    parser.add_argument("--depth-max", type=float, default=2.0,
                        help="最远有效深度（米，默认2.0）")
    parser.add_argument("--voxel", type=float, default=0.003,
                        help="融合体素大小（米，默认0.003）")
    parser.add_argument("--icp-voxel", type=float, default=0.005,
                        help="ICP用体素大小（米，默认0.005）")
    parser.add_argument("--icp-fitness-min", type=float, default=0.3,
                        help="ICP最低fitness阈值，低于此值丢弃该帧（默认0.3）")
    parser.add_argument("--no-icp", action="store_true",
                        help="跳过ICP（仅相机完全静止时使用）")
    parser.add_argument("--output", default="cup_fused.pcd",
                        help="输出PCD路径（默认 cup_fused.pcd）")
    args = parser.parse_args()

    # ── 解析bag路径 ──
    bag_path = args.bag
    if os.path.isdir(bag_path):
        db3s = sorted(Path(bag_path).glob("*.db3"))
        if not db3s:
            print(f"[ERROR] 目录中未找到.db3文件: {bag_path}")
            sys.exit(1)
        bag_path = str(db3s[0])
    print(f"[INFO] bag文件: {bag_path}")

    # ── 相机参数 ──
    print("\n[1] 读取相机参数...")
    c_intr = get_intrinsics(bag_path, "/camera/color/camera_info")
    d_intr = get_intrinsics(bag_path, "/camera/depth/camera_info")
    T_d2c  = get_extrinsics_depth_to_color(bag_path)
    print(f"    彩色内参: fx={c_intr['fx']:.1f} fy={c_intr['fy']:.1f} "
          f"cx={c_intr['cx']:.1f} cy={c_intr['cy']:.1f} "
          f"({c_intr['width']}×{c_intr['height']})")
    print(f"    深度内参: fx={d_intr['fx']:.1f} fy={d_intr['fy']:.1f} "
          f"({d_intr['width']}×{d_intr['height']})")

    # ── 读取所有帧 ──
    print("\n[2] 读取图像帧...")
    all_data = read_all_messages(
        bag_path,
        ["/camera/color/image_raw", "/camera/depth/image_raw"]
    )
    color_msgs = all_data["/camera/color/image_raw"]
    depth_msgs = all_data["/camera/depth/image_raw"]

    if not color_msgs:
        print("[ERROR] 未读到彩色帧，检查话题名")
        sys.exit(1)
    if not depth_msgs:
        print("[ERROR] 未读到深度帧，检查话题名")
        sys.exit(1)

    synced = sync_frames(color_msgs, depth_msgs)
    if not synced:
        print("[ERROR] 时间同步失败，尝试放宽 max_dt_ns")
        sys.exit(1)

    total_frames = len(synced)
    print(f"    共 {total_frames} 对同步帧")

    # ── 初始化分割器 ──
    segmenter = Segmenter(sam2_ckpt=args.sam2_checkpoint, sam2_cfg=args.sam2_cfg)

    # ── 逐帧处理 ──
    print(f"\n[3] 逐帧提取（共{total_frames}帧，每{args.annotate_interval}帧手动标注）")
    print("    提示：标注窗口 左键=前景 右键=背景 Enter=确认 ESC=跳过本帧\n")

    reference_pcd = None     # 第一帧，作为ICP目标（世界坐标系基准）
    accumulated   = []       # 已对齐到第一帧坐标系的点云列表
    current_mask  = None     # 当前有效mask（非标注帧复用）
    skipped = 0

    for idx, (c_msg, d_msg) in enumerate(synced):
        rgb   = decode_rgb(c_msg)
        depth = decode_depth(d_msg)

        # 判断是否需要手动标注
        need_annotate = (idx == 0) or (idx % args.annotate_interval == 0)

        if need_annotate:
            print(f"  ── 帧 {idx:3d}/{total_frames-1}  [需要标注] ──")
            mask = segmenter.annotate(rgb, frame_idx=idx)
            if mask is None or mask.sum() < 50:
                print(f"     → 跳过（mask无效或面积太小）")
                skipped += 1
                continue
            current_mask = mask
            print(f"     → mask面积: {mask.sum()} 像素")
        else:
            if current_mask is None:
                continue
            mask = current_mask   # 复用最近一次标注

        # 反投影 → 单帧物体点云
        pcd_frame = mask_to_pointcloud(
            depth, rgb, mask,
            c_intr, d_intr, T_d2c,
            depth_min=args.depth_min,
            depth_max=args.depth_max
        )

        n_pts = len(pcd_frame.points)
        if n_pts < 30:
            print(f"  帧{idx:3d}: 有效点过少({n_pts})，跳过")
            skipped += 1
            continue

        # 第一帧直接作为基准
        if reference_pcd is None:
            reference_pcd = pcd_frame
            accumulated.append(pcd_frame)
            print(f"  帧{idx:3d}: 基准帧，{n_pts}点")
            continue

        if args.no_icp:
            accumulated.append(pcd_frame)
            if idx % args.annotate_interval == 0:
                print(f"  帧{idx:3d}: 直接累积，{n_pts}点（no-icp模式）")
        else:
            print(f"  帧{idx:3d}: ICP配准中 ({n_pts}点)...", end=" ", flush=True)
            try:
                T_rel, fitness = icp_register(pcd_frame, reference_pcd,
                                              voxel=args.icp_voxel)
                print(f"fitness={fitness:.4f}", end="  ")
                if fitness < args.icp_fitness_min:
                    print(f"→ 丢弃（fitness低于{args.icp_fitness_min}）")
                    skipped += 1
                    continue
                pcd_frame.transform(T_rel)
                accumulated.append(pcd_frame)
                print(f"→ OK，累积至{len(accumulated)}帧")
            except Exception as e:
                print(f"\n     → ICP失败: {e}，跳过")
                skipped += 1

    cv2.destroyAllWindows()

    print(f"\n  处理完成：成功{len(accumulated)}帧，跳过{skipped}帧")

    if not accumulated:
        print("[ERROR] 未成功提取任何点云")
        sys.exit(1)

    # ── 融合 & 去噪 ──
    print(f"\n[4] 融合{len(accumulated)}帧点云...")
    fused = fuse_and_clean(accumulated, voxel=args.voxel)

    # ── 保存 ──
    o3d.io.write_point_cloud(args.output, fused)
    print(f"\n[5] 已保存 → {args.output}  ({len(fused.points)}点)")

    # ── 可视化 ──
    print("[6] 可视化（按 Q 关闭）...")
    vis = o3d.visualization.Visualizer()
    vis.create_window("融合点云", width=1280, height=720)
    vis.add_geometry(fused)
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.1, 0.1, 0.1])
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
