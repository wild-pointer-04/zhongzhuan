"""
duck_pcd_pipeline.py
====================
从 ROS 2 Bag 中重建鸭子的致密彩色 3D 点云。

管线：
  Bag → 时间戳同步配对帧 → SAM分割(带中心跟踪) → 单帧PCD + SOR去噪
      → ICP多帧融合(FPFH粗配准 + ICP精配准) → Voxel下采样 → 保存 + 可视化

用法：
  python duck_pcd_pipeline.py <bag_path> <sam_checkpoint> [选项]

选项：
  --max_frames   最多处理多少帧 (默认 900, 即全部)
  --frame_step   每隔几帧处理一次 (默认 5, 即每5帧取1帧)
  --voxel_size   体素下采样尺寸，单位米 (默认 0.001 = 1mm)
  --device       cpu 或 cuda (默认 cpu)
  --output       输出 PCD 文件路径 (默认 duck_final.pcd)
"""

import os
import sys
import argparse
import numpy as np
import cv2
import open3d as o3d
import torch
from segment_anything import sam_model_registry, SamPredictor

import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

# ─────────────────────────────────────────────
# 相机内参 (848×480, RealSense D系列)
# ─────────────────────────────────────────────
FX = 412.8570556640625
FY = 412.8570556640625
CX = 424.0
CY = 237.01876831054688

DEPTH_SCALE = 1000.0        # ROS depth 单位: mm → m
DEPTH_MIN   = 0.15          # 近端截断 (m)
DEPTH_MAX   = 1.0           # 远端截断 (m)
MIN_POINTS  = 200           # 单帧有效点最少数量，否则丢弃

# ─────────────────────────────────────────────
# 全局点击坐标
# ─────────────────────────────────────────────
_click_x, _click_y = -1, -1

def _mouse_callback(event, x, y, flags, param):
    global _click_x, _click_y
    if event == cv2.EVENT_LBUTTONDOWN:
        _click_x, _click_y = x, y
        print(f"  [点击] ({x}, {y})")


# ══════════════════════════════════════════════
# 1. Bag 读取：时间戳同步配对
# ══════════════════════════════════════════════

def load_synced_frames(bag_path: str,
                       rgb_topic: str  = '/camera/color/image_raw',
                       depth_topic: str = '/camera/depth/image_raw',
                       max_frames: int = 9999,
                       frame_step: int = 5) -> list[tuple]:
    """
    从 Bag 中读取 RGB + Depth 的时间戳同步帧对。

    策略：维护两个最新消息缓冲，每当两者时间差 < 33ms（约1帧），
    就认为是同一时刻，配成一对。

    返回：[(rgb_np_uint8_RGB, depth_np_uint16), ...]
    """
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    reader.open(storage_options, rosbag2_py.ConverterOptions('', ''))

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: get_message(t.type) for t in topic_types}

    # 确保话题存在
    available = [t.name for t in topic_types]
    for t in [rgb_topic, depth_topic]:
        if t not in available:
            print(f"[警告] 话题 '{t}' 不在 Bag 中！可用话题: {available}")

    buf_rgb   = None   # (timestamp_ns, np_array)
    buf_depth = None
    pairs     = []
    SYNC_THRESH_NS = 33_000_000  # 33ms

    count_raw = 0
    print("正在读取 Bag 并同步帧对...")

    while reader.has_next():
        topic, data, ts_ns = reader.read_next()

        if topic == rgb_topic:
            msg = deserialize_message(data, type_map[topic])
            # ROS image_raw encoding 通常是 'rgb8'，直接作为 RGB 存储
            arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3).copy()
            if msg.encoding.lower() in ('bgr8', 'bgr'):
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            buf_rgb = (ts_ns, arr)

        elif topic == depth_topic:
            msg = deserialize_message(data, type_map[topic])
            arr = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width).copy()
            buf_depth = (ts_ns, arr)

        # 尝试配对
        if buf_rgb is not None and buf_depth is not None:
            dt = abs(buf_rgb[0] - buf_depth[0])
            if dt < SYNC_THRESH_NS:
                count_raw += 1
                # frame_step 抽帧
                if count_raw % frame_step == 0:
                    pairs.append((buf_rgb[1], buf_depth[1]))
                    if len(pairs) % 20 == 0:
                        print(f"  已配对 {len(pairs)} 帧 (原始帧 {count_raw})")
                buf_rgb = None   # 消费掉，避免重复配对
                buf_depth = None
                if len(pairs) >= max_frames:
                    break

    print(f"共获得 {len(pairs)} 个同步帧对（原始 {count_raw} 对）")
    return pairs


# ══════════════════════════════════════════════
# 2. 交互选点：在第一帧上点击鸭子中心
# ══════════════════════════════════════════════

def _try_gui_select(rgb_img_rgb: np.ndarray) -> tuple[int, int] | None:
    """
    尝试用 OpenCV GUI 窗口让用户点击选点。
    如果 GUI 后端初始化失败（Qt字体缺失等），返回 None。
    """
    global _click_x, _click_y
    _click_x, _click_y = -1, -1

    win = 'Select Duck Center'
    try:
        # WINDOW_NORMAL 兼容性更好，避免 Qt 特定问题
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        # 测试窗口是否真的创建成功
        ret = cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE)
        if ret < 0:
            raise RuntimeError("窗口创建失败 (getWindowProperty < 0)")
        cv2.setMouseCallback(win, _mouse_callback)
    except Exception as e:
        print(f"  [GUI] 窗口初始化失败: {e}")
        cv2.destroyAllWindows()
        return None

    print("\n>>> 请在弹出窗口中点击鸭子中心，然后按【空格键】确认 (ESC退出)...")
    while True:
        view = cv2.cvtColor(rgb_img_rgb, cv2.COLOR_RGB2BGR)
        if _click_x != -1:
            cv2.circle(view, (_click_x, _click_y), 7, (0, 0, 255), -1)
            cv2.putText(view, f"({_click_x},{_click_y})", (_click_x + 10, _click_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(view, "Press SPACE to confirm", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(win, view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and _click_x != -1:
            break
        if key == 27:
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyAllWindows()
    return _click_x, _click_y


def _save_first_frame_for_reference(rgb_img_rgb: np.ndarray, path: str = "first_frame.png"):
    """把第一帧存成 PNG，方便用户用图片查看器确认坐标。"""
    bgr = cv2.cvtColor(rgb_img_rgb, cv2.COLOR_RGB2BGR)
    # 在图上画网格线方便读坐标
    h, w = bgr.shape[:2]
    for x in range(0, w, 100):
        cv2.line(bgr, (x, 0), (x, h), (80, 80, 80), 1)
        cv2.putText(bgr, str(x), (x + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    for y in range(0, h, 100):
        cv2.line(bgr, (0, y), (w, y), (80, 80, 80), 1)
        cv2.putText(bgr, str(y), (2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.imwrite(path, bgr)
    print(f"  [参考图] 第一帧已保存到: {os.path.abspath(path)}")
    print(f"  图片尺寸: {w}×{h}，请用图片查看器打开，找到鸭子中心像素坐标。")


def _cli_select(rgb_img_rgb: np.ndarray) -> tuple[int, int]:
    """
    GUI 不可用时，保存第一帧图片，然后让用户在终端输入坐标。
    """
    ref_path = "first_frame_grid.png"
    _save_first_frame_for_reference(rgb_img_rgb, ref_path)

    h, w = rgb_img_rgb.shape[:2]
    print(f"\n  图像尺寸: 宽={w}, 高={h}")
    print("  请打开上面保存的图片，找到鸭子中心的像素坐标，然后在此输入：")
    while True:
        try:
            raw = input("  请输入坐标，格式为 x,y (例如 420,240): ").strip()
            x_s, y_s = raw.split(',')
            x, y = int(x_s.strip()), int(y_s.strip())
            if 0 <= x < w and 0 <= y < h:
                return x, y
            else:
                print(f"  坐标超出范围！请输入 x∈[0,{w-1}], y∈[0,{h-1}]")
        except (ValueError, KeyboardInterrupt):
            print("  格式错误，请重新输入（格式：x,y）")


def interactive_select_point(rgb_img_rgb: np.ndarray) -> tuple[int, int]:
    """
    优先尝试 GUI 点击选点；GUI 初始化失败时自动降级为：
      保存带网格的参考图 → 终端输入坐标。
    """
    result = _try_gui_select(rgb_img_rgb)
    if result is not None:
        return result

    print("\n  [降级] GUI 不可用，切换到命令行选点模式...")
    return _cli_select(rgb_img_rgb)


# ══════════════════════════════════════════════
# 3. 单帧处理：SAM分割 → 3D反投影 → SOR去噪
# ══════════════════════════════════════════════

def estimate_duck_depth(depth_img: np.ndarray, px: int, py: int,
                         patch: int = 10) -> tuple[float, float] | None:
    """
    在选点坐标周围取 patch×patch 的小窗口，
    统计有效深度值的中位数，估算鸭子所在深度范围。
    返回 (z_min, z_max)，单位米；若窗口无有效深度则返回 None。
    """
    h, w = depth_img.shape
    y0, y1 = max(0, py - patch), min(h, py + patch)
    x0, x1 = max(0, px - patch), min(w, px + patch)
    region = depth_img[y0:y1, x0:x1].astype(np.float32) / DEPTH_SCALE
    valid = region[(region > 0.05) & (region < 2.0)]
    if len(valid) == 0:
        return None
    z_med = float(np.median(valid))
    # 鸭子大约 10cm 高，前后各留 8cm 余量
    margin = 0.08
    return max(0.05, z_med - margin), z_med + margin


def pick_best_mask(masks: np.ndarray, scores: np.ndarray,
                   prompt_x: int, prompt_y: int,
                   duck_z_min: float, duck_z_max: float,
                   depth_img: np.ndarray) -> np.ndarray | None:
    """
    从 SAM 返回的多个 Mask 中选出最符合鸭子的那个。

    综合三个条件打分：
      1. SAM 置信度分数
      2. Mask 面积惩罚（太大说明包含了桌面）
      3. Mask 内部深度值与鸭子深度范围的吻合度（核心新增）

    返回最优 mask（bool 数组）或 None。
    """
    h_d, w_d = depth_img.shape
    z = depth_img.astype(np.float32) / DEPTH_SCALE

    best_score = -1.0
    best_mask  = None

    for i, (mask, sam_score) in enumerate(zip(masks, scores)):
        if sam_score < 0.7:
            continue

        # 把 mask 缩放到深度图尺寸
        mask_d = cv2.resize(
            mask.astype(np.uint8),
            (w_d, h_d), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        area = np.sum(mask_d)
        if area < 100:
            continue

        # ── 深度吻合度：mask 内落在鸭子深度范围内的像素比例 ──
        z_in_mask = z[mask_d & (depth_img > 0)]
        if len(z_in_mask) == 0:
            continue
        depth_hit_ratio = np.mean((z_in_mask >= duck_z_min) & (z_in_mask <= duck_z_max))

        # ── 面积惩罚：面积越大分越低（对数归一化）──
        # 假设鸭子不超过图像的 15%
        img_area = h_d * w_d
        area_penalty = max(0.0, 1.0 - (area / (img_area * 0.15)))

        # ── 综合分 ──────────────────────────────────
        composite = sam_score * 0.3 + depth_hit_ratio * 0.5 + area_penalty * 0.2

        if composite > best_score:
            best_score = composite
            best_mask  = mask   # 返回原始尺寸的 mask（RGB尺寸）

    return best_mask


def frame_to_pcd(rgb_img: np.ndarray,
                 depth_img: np.ndarray,
                 predictor: SamPredictor,
                 prompt_x: int,
                 prompt_y: int,
                 duck_z_min: float,
                 duck_z_max: float) -> tuple[o3d.geometry.PointCloud | None, tuple[int, int]]:
    """
    处理单帧，返回 (点云 or None, 下一帧的追踪提示点坐标)。

    改进点：
      - SAM 使用「正提示点（鸭子中心）+ 负提示点（图像四角，排除大背景）」
      - 用深度吻合度替代"面积最小"策略选 mask
      - 深度过滤使用动态鸭子深度范围而非固定全局范围
    """

    h_r, w_r = rgb_img.shape[:2]
    h_d, w_d = depth_img.shape

    # ── 3.1 构建 SAM 提示点 ──────────────────────────
    # 正样本：鸭子中心
    pos_pts = [[prompt_x, prompt_y]]
    pos_labels = [1]

    # 负样本：图像四角 + 下边缘中点（桌面通常在画面下方）
    # 注意：负样本坐标要在 RGB 尺寸内
    neg_pts = [
        [10,          10         ],   # 左上角
        [w_r - 10,    10         ],   # 右上角
        [10,          h_r - 10   ],   # 左下角
        [w_r - 10,    h_r - 10   ],   # 右下角
        [w_r // 2,    h_r - 10   ],   # 下边缘中点（桌面）
    ]
    neg_labels = [0] * len(neg_pts)

    all_pts    = np.array(pos_pts + neg_pts,    dtype=np.float32)
    all_labels = np.array(pos_labels + neg_labels, dtype=np.int32)

    # ── 3.2 SAM 分割 ─────────────────────────────────
    predictor.set_image(rgb_img)
    masks, scores, _ = predictor.predict(
        point_coords=all_pts,
        point_labels=all_labels,
        multimask_output=True,
    )

    # ── 3.3 用深度吻合度选最优 Mask ──────────────────
    mask_rgb = pick_best_mask(masks, scores, prompt_x, prompt_y,
                              duck_z_min, duck_z_max, depth_img)
    if mask_rgb is None:
        return None, (prompt_x, prompt_y)

    # ── 3.4 计算下一帧跟踪点（本帧 Mask 质心）────────
    ys, xs = np.where(mask_rgb)
    if len(xs) > 0:
        next_px = int(np.mean(xs))
        next_py = int(np.mean(ys))
    else:
        next_px, next_py = prompt_x, prompt_y

    # ── 3.5 将 Mask 和 RGB 对齐到深度图尺寸 ──────────
    if (h_r, w_r) != (h_d, w_d):
        rgb_at_depth = cv2.resize(rgb_img, (w_d, h_d), interpolation=cv2.INTER_LINEAR)
    else:
        rgb_at_depth = rgb_img

    mask_at_depth = cv2.resize(
        mask_rgb.astype(np.uint8), (w_d, h_d),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    # ── 3.6 深度过滤（用动态鸭子深度范围）────────────
    z = depth_img.astype(np.float32) / DEPTH_SCALE
    valid_mask = (
        mask_at_depth       &
        (depth_img > 0)     &   # 剔除无效深度像素
        (z >= duck_z_min)   &   # 动态鸭子深度下界
        (z <= duck_z_max)       # 动态鸭子深度上界
    )

    if np.sum(valid_mask) < MIN_POINTS:
        return None, (next_px, next_py)

    # ── 3.7 3D 反投影 ─────────────────────────────────
    u, v = np.meshgrid(np.arange(w_d), np.arange(h_d))
    x3d = (u[valid_mask] - CX) * z[valid_mask] / FX
    y3d = (v[valid_mask] - CY) * z[valid_mask] / FY
    z3d = z[valid_mask]
    pts = np.stack([x3d, y3d, z3d], axis=1)

    # ── 3.8 提取颜色 ──────────────────────────────────
    colors = rgb_at_depth[valid_mask].astype(np.float64) / 255.0

    # ── 3.9 构建点云 + SOR 去噪 ───────────────────────
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.5)

    return pcd, (next_px, next_py)


# ══════════════════════════════════════════════
# 4. ICP 多帧融合
# ══════════════════════════════════════════════

def pairwise_registration(src: o3d.geometry.PointCloud,
                           tgt: o3d.geometry.PointCloud,
                           voxel_size: float) -> np.ndarray:
    """
    FPFH 粗配准 + ICP 精配准，返回从 src 到 tgt 的 4×4 变换矩阵。
    """
    # 体素下采样（用于特征提取）
    def preprocess(pcd):
        down = pcd.voxel_down_sample(voxel_size * 5)
        down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 10, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 25, max_nn=100))
        return down, fpfh

    src_down, src_fpfh = preprocess(src)
    tgt_down, tgt_fpfh = preprocess(tgt)

    # FPFH 粗配准
    dist_thresh = voxel_size * 15
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=dist_thresh,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4_000_000, 500),
    )

    # ICP 精配准
    result_icp = o3d.pipelines.registration.registration_icp(
        src, tgt,
        max_correspondence_distance=voxel_size * 2,
        init=result_ransac.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )

    return result_icp.transformation


def merge_point_clouds(pcd_list: list[o3d.geometry.PointCloud],
                       voxel_size: float) -> o3d.geometry.PointCloud:
    """
    将所有单帧点云融合到第一帧坐标系中。
    使用滚动配准：每帧与上一帧配准，然后拼入累积点云。
    """
    if not pcd_list:
        return o3d.geometry.PointCloud()

    print(f"\n开始 ICP 多帧融合，共 {len(pcd_list)} 帧...")
    merged = pcd_list[0]

    for i in range(1, len(pcd_list)):
        src = pcd_list[i]
        tgt = merged

        try:
            T = pairwise_registration(src, tgt, voxel_size)
            src_transformed = src.transform(T)
            merged = merged + src_transformed
            # 每融合 10 帧做一次体素下采样，防止内存爆炸
            if i % 10 == 0:
                merged = merged.voxel_down_sample(voxel_size)
                print(f"  已融合 {i+1}/{len(pcd_list)} 帧，当前点数: {len(merged.points)}")
        except Exception as e:
            print(f"  [跳过] 第 {i} 帧配准失败: {e}")
            continue

    return merged


# ══════════════════════════════════════════════
# 5. 主流程
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="鸭子 3D 点云重建管线")
    parser.add_argument("bag_path",       help="ROS 2 Bag 路径")
    parser.add_argument("sam_checkpoint", help="SAM 权重文件路径 (.pth)")
    parser.add_argument("--max_frames",   type=int,   default=900,        help="最多处理帧数")
    parser.add_argument("--frame_step",   type=int,   default=5,          help="抽帧间隔")
    parser.add_argument("--voxel_size",   type=float, default=0.001,      help="体素尺寸 (m)")
    parser.add_argument("--device",       type=str,   default="cpu",      help="cpu 或 cuda")
    parser.add_argument("--output",       type=str,   default="duck_final.pcd", help="输出路径")
    args = parser.parse_args()

    # ── 5.1 加载 SAM ────────────────────────────────
    print(f"\n[1/5] 加载 SAM 模型 (device={args.device})...")
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    # ── 5.2 读取 Bag，获取同步帧对 ──────────────────
    print("\n[2/5] 读取 Bag 文件...")
    pairs = load_synced_frames(
        args.bag_path,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
    )
    if not pairs:
        print("错误：未找到有效帧对，请检查 Bag 文件和话题名称。")
        sys.exit(1)

    first_rgb, _ = pairs[0]

    # ── 5.3 交互选点 ─────────────────────────────────
    print("\n[3/5] 请在第一帧上点击鸭子中心...")
    px, py = interactive_select_point(first_rgb)
    print(f"  确认选点: ({px}, {py})")

    # ── 5.3.5 估算鸭子深度范围 ───────────────────────
    _, first_depth = pairs[0]

    # 选点坐标是 RGB 图上的，深度图可能尺寸不同，需要映射
    h_r, w_r = first_rgb.shape[:2]
    h_d, w_d = first_depth.shape
    px_d = int(px * w_d / w_r)
    py_d = int(py * h_d / h_r)

    depth_range = estimate_duck_depth(first_depth, px_d, py_d)
    if depth_range is None:
        print("  [警告] 选点处深度值无效，将使用默认深度范围 0.15m ~ 1.0m")
        duck_z_min, duck_z_max = DEPTH_MIN, DEPTH_MAX
    else:
        duck_z_min, duck_z_max = depth_range
        print(f"  鸭子深度范围自动估算: {duck_z_min:.3f}m ~ {duck_z_max:.3f}m")

    # ── 5.4 逐帧处理 ─────────────────────────────────
    print(f"\n[4/5] 逐帧提取点云，共 {len(pairs)} 帧...")
    pcd_list = []
    curr_px, curr_py = px, py  # 追踪点，会随帧更新

    for i, (rgb, depth) in enumerate(pairs):
        pcd, (curr_px, curr_py) = frame_to_pcd(
            rgb, depth, predictor, curr_px, curr_py,
            duck_z_min, duck_z_max
        )
        if pcd is not None and len(pcd.points) >= MIN_POINTS:
            pcd_list.append(pcd)
        if (i + 1) % 10 == 0:
            print(f"  处理进度: {i+1}/{len(pairs)} 帧，有效帧: {len(pcd_list)}")

    print(f"  逐帧处理完成，有效帧: {len(pcd_list)}/{len(pairs)}")

    if not pcd_list:
        print("错误：所有帧处理均失败，请检查选点是否准确。")
        sys.exit(1)

    # 如果只有1帧（比如测试），直接跳过融合
    if len(pcd_list) == 1:
        final_pcd = pcd_list[0]
        print("  只有1帧有效点云，跳过 ICP 融合。")
    else:
        # ── 5.5 ICP 多帧融合 ─────────────────────────
        print("\n[5/5] ICP 多帧融合...")
        merged = merge_point_clouds(pcd_list, args.voxel_size)

        # 最终体素下采样（1mm网格，致密不冗余）
        final_pcd = merged.voxel_down_sample(args.voxel_size)

    # 计算法线（让可视化更有立体感）
    final_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30)
    )

    # ── 5.6 保存 & 可视化 ───────────────────────────
    print(f"\n[完成] 点云点数: {len(final_pcd.points)}")
    print(f"       保存到: {args.output}")
    o3d.io.write_point_cloud(args.output, final_pcd)

    print("\n>>> 弹出可视化窗口")
    print("    操作提示: 鼠标左键旋转 / 滚轮缩放 / 按 '1'~'3' 切换渲染模式")
    o3d.visualization.draw_geometries(
        [final_pcd],
        window_name=f"鸭子点云 ({len(final_pcd.points)} 点)",
        point_show_normal=False,
    )


if __name__ == "__main__":
    main()
