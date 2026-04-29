"""
validate_duck.py  (v4 - 侧面抓取版 / side-grasp from camera-facing surface)
============================================================================
相对 v3 的核心改动：
  1. 识别点云密集面（相机拍摄面）= Y 轴负方向最小值面（相机在物体正前方）
  2. 手掌从 Y 轴负方向（正面）水平接近，手掌法向量朝 +Y（即朝向物体）
  3. 悬停偏移从 Z 方向改为 Y 方向（HOVER_DIST = 手掌距物体正面的距离）
  4. 过滤条件：检查手掌 Y 坐标 < 物体正面 Y 值（确保从前方来），以及手掌朝向

坐标系说明（Orbbec 侧拍，相机在物体正前方）：
  中心化后：
    Y 轴：相机朝向轴（Y_min = 相机侧 = 正面，Y_max = 背面）
    Z 轴：朝上（正值为顶部）
    X 轴：水平左右

  "从正面抓" = 手掌 Y 坐标 < 物体 Y_min（悬停在前方）
               手掌局部 Z 轴经旋转后指向 +Y（即手掌面向物体）

使用方法：
    python scripts/validate_duck.py \
        --pcd_path /home/eureka/duck_point/duck_v6.pcd \
        --ckpt_name model_shadowhand \
        --batch_size 10 \
        --num_rounds 3

可选参数：
    --camera_axis y          # 相机拍摄轴（默认 y，可选 x）
    --camera_dir neg         # 相机在负方向侧（默认 neg，可选 pos）
    --hover_dist 0.06        # 手掌悬停距正面的距离（米）
    --side_angle_deg 60      # 手掌朝向容忍角度（与相机轴对齐角度）
"""

import os, sys, time, argparse, warnings
import numpy as np
from tqdm import tqdm
from termcolor import cprint
from types import SimpleNamespace
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from model.network import create_network
from utils.multilateration import multilateration
from utils.se3_transform import compute_link_pose
from utils.optimization import process_transform, create_problem, optimization
from utils.hand_model import create_hand_model
from validation.validate_utils import validate_isaac

OBJECT_NAME = "contactdb+water_bottle"
ROBOT_NAME  = "wujihand"
NUM_POINTS  = 512

# ── 侧面抓取默认参数 ────────────────────────────────────────
# 手掌悬停在物体正面前方的距离（沿相机轴方向）
HOVER_DIST = 0.06          # 6cm
# 判断"从正面抓"的角度容忍度：手掌法向量与相机轴方向夹角 < 这个值
SIDE_GRASP_ANGLE_DEG = 60.0
# ────────────────────────────────────────────────────────────


def get_camera_axis_info(camera_axis: str, camera_dir: str):
    """
    返回相机轴索引和符号。

    camera_axis: 'x' | 'y' | 'z'（点云中心化后，相机指向的轴）
    camera_dir:  'neg' | 'pos'（相机在负值侧还是正值侧）

    返回：
        axis_idx   : int，轴索引 (0=X, 1=Y, 2=Z)
        front_sign : +1 or -1（front_surface 在轴的哪一端）
                     neg → 相机在负侧 → 物体正面 = 轴最小值 → front_sign = -1
                     pos → 相机在正侧 → 物体正面 = 轴最大值 → front_sign = +1
        approach_vec: np.ndarray shape (3,)，手掌接近方向（从手掌到物体）
    """
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map[camera_axis.lower()]
    if camera_dir == 'neg':
        # 相机在负侧，物体正面是轴最小值，手掌从更负的位置接近 → 接近方向 = +axis
        front_sign   = -1
        approach_vec = np.zeros(3); approach_vec[axis_idx] = +1.0
    else:
        # 相机在正侧，物体正面是轴最大值，手掌从更正的位置接近 → 接近方向 = -axis
        front_sign   = +1
        approach_vec = np.zeros(3); approach_vec[axis_idx] = -1.0
    return axis_idx, front_sign, approach_vec


def load_side_pc(pcd_path, num_points=NUM_POINTS,
                 camera_axis='y', camera_dir='neg'):
    """
    加载点云，坐标系对齐，采样。

    Orbbec 侧拍坐标系对齐规则（相机在 Y 负方向，镜头朝 +Y）：
      相机 Z 朝上 → Isaac Z 朝上（保持不变，或按实际标定调整）
      相机 Y 朝物体（相机光轴）→ Isaac Y 朝物体（前方）
      相机 X 水平 → Isaac X 水平

    如果实际相机安装方向不同，修改下方翻转逻辑即可。
    """
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    if len(pts) == 0:
        raise ValueError(f"点云为空：{pcd_path}")

    cprint(f"[PC] 原始点数: {len(pts)}", "cyan")

    # 1. 中心化
    center = pts.mean(axis=0)
    pts    = pts - center
    cprint(f"[PC] 中心化后 X:[{pts[:,0].min():.3f},{pts[:,0].max():.3f}]"
           f" Y:[{pts[:,1].min():.3f},{pts[:,1].max():.3f}]"
           f" Z:[{pts[:,2].min():.3f},{pts[:,2].max():.3f}]", "cyan")

    # 2. 坐标系对齐
    #    根据实际相机安装方式调整。
    #    Orbbec 侧拍常见情况：相机 Z 朝上，Y 朝物体，X 朝右
    #    → 对齐到 Isaac：无需翻转（如需翻转取消下方注释）
    # pts[:, 2] = -pts[:, 2]   # 若 Z 需要翻转
    # pts[:, 0] = -pts[:, 0]   # 若 X 需要翻转

    # 3. 确定正面位置
    axis_idx, front_sign, approach_vec = get_camera_axis_info(camera_axis, camera_dir)
    axis_name = ['X', 'Y', 'Z'][axis_idx]

    if front_sign == -1:
        # 相机在负侧，正面 = 轴最小值
        front_val = pts[:, axis_idx].min()
        back_val  = pts[:, axis_idx].max()
    else:
        # 相机在正侧，正面 = 轴最大值
        front_val = pts[:, axis_idx].max()
        back_val  = pts[:, axis_idx].min()

    cprint(f"[PC] 相机轴: {axis_name}, 方向: {camera_dir}", "cyan")
    cprint(f"[PC] 物体正面 {axis_name}={front_val:.4f}m，背面 {axis_name}={back_val:.4f}m", "cyan")
    cprint(f"[PC] 手掌悬停目标: {axis_name} = {front_val + front_sign * (-HOVER_DIST):.4f}m", "cyan")

    n   = len(pts)
    idx = np.random.choice(n, num_points, replace=(n < num_points))
    return torch.from_numpy(pts[idx]), float(front_val), axis_idx, front_sign, approach_vec


def make_side_grasp_q(hand, device, obj_pc, front_val,
                      axis_idx, front_sign, approach_vec,
                      batch_size, hover_dist=HOVER_DIST):
    """
    构造"从侧面抓"的初始 q batch。

    策略：
    - 手掌位置：沿接近轴在正面前方 hover_dist 处
    - 手掌在另两个轴的位置：物体中心附近 + 随机小扰动
    - 旋转：手掌局部 Z 轴对齐到 approach_vec（即朝向物体）
      基础旋转 = 将局部 Z [0,0,1] 旋转到 approach_vec
    - 手指关节：从 hand.get_initial_q() 借用（张开状态）
    """
    iq_list, rpc_list, opc_list = [], [], []
    num_points = obj_pc.shape[0]

    # 物体中心（另两个轴的均值）
    other_axes  = [i for i in range(3) if i != axis_idx]
    center_vals = obj_pc[:, other_axes].mean(dim=0).numpy()  # [2]

    # 手掌悬停位置：在正面法向量方向退出 hover_dist
    palm_on_axis = front_val + front_sign * (-hover_dist)  # front_sign=-1 → 更负

    # 基础旋转：将手掌局部 Z 轴 [0,0,1] 对齐到 approach_vec
    local_z = np.array([0.0, 0.0, 1.0])
    base_rotvec = _align_rotvec(local_z, approach_vec)

    for _ in range(batch_size):
        q_new = hand.get_initial_q().clone()

        # ── 手掌平移 ─────────────────────────────────────
        pos = np.zeros(3)
        pos[axis_idx]      = palm_on_axis
        # 另两轴在物体中心附近随机抖动 ±2cm
        pos[other_axes[0]] = float(center_vals[0]) + (np.random.rand() - 0.5) * 0.04
        pos[other_axes[1]] = float(center_vals[1]) + (np.random.rand() - 0.5) * 0.04
        q_new[0] = float(pos[0])
        q_new[1] = float(pos[1])
        q_new[2] = float(pos[2])

        # ── 手掌旋转 ─────────────────────────────────────
        r_base    = R.from_rotvec(base_rotvec)
        # 随机小扰动 ±20°，接近轴方向扰动减半
        perturb   = np.random.uniform(-np.pi/9, np.pi/9, size=3)
        perturb[axis_idx] *= 0.5
        r_perturb = R.from_rotvec(perturb)
        r_final   = r_perturb * r_base
        rv = r_final.as_rotvec()
        q_new[3] = float(rv[0])
        q_new[4] = float(rv[1])
        q_new[5] = float(rv[2])

        # ── 生成 robot_pc ─────────────────────────────────
        robot_pc = hand.get_transformed_links_pc(q_new)[:, :3]
        rn = robot_pc.shape[0]
        if rn < num_points:
            pad      = torch.randint(0, rn, (num_points - rn,))
            robot_pc = torch.cat([robot_pc, robot_pc[pad]], dim=0)
        else:
            robot_pc = robot_pc[:num_points]

        # ── 物体点云加噪声 ───────────────────────────────
        obj_noisy = obj_pc + torch.randn_like(obj_pc) * 0.002

        iq_list.append(q_new)
        rpc_list.append(robot_pc)
        opc_list.append(obj_noisy)

    return (
        torch.stack(iq_list).to(device),
        torch.stack(rpc_list).to(device),
        torch.stack(opc_list).to(device),
    )


def _align_rotvec(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
    """
    计算将 from_vec 旋转到 to_vec 的旋转向量（axis-angle）。
    两向量均为单位向量。
    """
    from_vec = from_vec / np.linalg.norm(from_vec)
    to_vec   = to_vec   / np.linalg.norm(to_vec)
    cross    = np.cross(from_vec, to_vec)
    dot      = np.dot(from_vec, to_vec)
    sin_a    = np.linalg.norm(cross)
    cos_a    = dot

    if sin_a < 1e-6:
        if cos_a > 0:
            return np.zeros(3)                       # 已对齐
        else:
            # 方向完全相反，选任意垂直轴转 180°
            perp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(from_vec, perp)) > 0.9:
                perp = np.array([0.0, 1.0, 0.0])
            return perp * np.pi

    axis  = cross / sin_a
    angle = np.arctan2(sin_a, cos_a)
    return axis * angle


def filter_side_grasp(predict_q_batch, front_val, axis_idx, front_sign,
                      approach_vec, angle_thresh_deg=SIDE_GRASP_ANGLE_DEG):
    """
    过滤掉不是从正面侧抓的预测姿态。

    判断标准：
    1. 手掌在接近轴上的位置必须在物体正面之前（未穿入物体）
    2. 手掌局部 Z 轴经旋转后，与 approach_vec 的夹角 < angle_thresh_deg

    返回：过滤后的 q_batch 以及有效的索引
    """
    q_np       = predict_q_batch.detach().cpu().numpy()
    valid_mask = []

    for i in range(q_np.shape[0]):
        q       = q_np[i]
        palm_ax = q[axis_idx]   # 手掌在接近轴上的坐标

        # 条件1：手掌未穿入物体
        # front_sign=-1 → 正面在轴最小值 → 手掌应 < front_val + 容差
        # front_sign=+1 → 正面在轴最大值 → 手掌应 > front_val - 容差
        margin = 0.05   # 5cm 容差，防止误判
        if front_sign == -1:
            pos_ok = palm_ax < (front_val + margin)
        else:
            pos_ok = palm_ax > (front_val - margin)

        if not pos_ok:
            valid_mask.append(False)
            continue

        # 条件2：手掌朝向物体
        try:
            rot_mat    = R.from_rotvec(q[3:6]).as_matrix()
            local_z_w  = rot_mat @ np.array([0, 0, 1])   # 手掌法向量（世界系）
            cos_a      = np.clip(np.dot(local_z_w, approach_vec), -1, 1)
            angle      = np.degrees(np.arccos(cos_a))
            valid_mask.append(angle < angle_thresh_deg)
        except Exception:
            valid_mask.append(False)

    valid_mask  = np.array(valid_mask)
    valid_count = valid_mask.sum()
    total       = len(valid_mask)

    if valid_count == 0:
        cprint(f"   [Filter] 无姿态通过侧面约束（{total}个全部过滤），保留原始 batch", "yellow")
        return predict_q_batch, np.arange(total)

    cprint(f"   [Filter] 侧面抓取过滤: {valid_count}/{total} 个通过", "green")
    indices = np.where(valid_mask)[0]
    return predict_q_batch[torch.from_numpy(indices)], indices


def run_local_refinement(network, hand, robot_pc, local_pc, initial_q, device):
    with torch.no_grad():
        if local_pc.shape[1] < 512:
            idx = torch.randint(0, local_pc.shape[1], (1, 512), device=device)
            lpc = torch.gather(local_pc, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        else:
            lpc = local_pc[:, :512, :]
        dro = network(robot_pc, lpc)["dro"].detach()
    mlat_pc = multilateration(dro, lpc)
    tf, _   = compute_link_pose(hand.links_pc, mlat_pc, is_train=False)
    otf     = process_transform(hand.pk_chain, tf)
    layer   = create_problem(hand.pk_chain, otf.keys())
    refined = optimization(hand.pk_chain, layer, initial_q, otf)
    return refined, otf


def infer_batch(network, hand, iq_batch, rpc_batch, opc_batch, batch_size, device):
    pq_list, t_list = [], []
    for i in tqdm(range(batch_size), desc="推理", leave=False):
        iq  = iq_batch[i:i+1];  rpc = rpc_batch[i:i+1];  opc = opc_batch[i:i+1]
        with torch.no_grad():
            dro = network(rpc, opc)["dro"].detach()
        mlat  = multilateration(dro, opc)
        tf, _ = compute_link_pose(hand.links_pc, mlat, is_train=False)
        otf   = process_transform(hand.pk_chain, tf)
        layer = create_problem(hand.pk_chain, otf.keys())
        t0    = time.time()
        pq    = optimization(hand.pk_chain, layer, iq, otf)
        t_list.append(time.time() - t0)
        pq_list.append(pq)
    return torch.cat(pq_list, dim=0), t_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd_path",          type=str,   required=True)
    parser.add_argument("--ckpt_name",          type=str,   default="model_shadowhand")
    parser.add_argument("--batch_size",         type=int,   default=10)
    parser.add_argument("--num_rounds",         type=int,   default=3)
    parser.add_argument("--gpu",                type=int,   default=0)
    parser.add_argument("--num_points",         type=int,   default=NUM_POINTS)
    # ── 侧面抓取参数 ─────────────────────────────────────
    parser.add_argument("--camera_axis",        type=str,   default="y",
                        choices=["x", "y", "z"],
                        help="中心化后点云最密集面所在的轴（默认 y）")
    parser.add_argument("--camera_dir",         type=str,   default="neg",
                        choices=["neg", "pos"],
                        help="相机在轴的负侧(neg)还是正侧(pos)（默认 neg）")
    parser.add_argument("--hover_dist",         type=float, default=HOVER_DIST,
                        help="手掌悬停在正面前方的距离，单位 m（默认 0.06）")
    parser.add_argument("--side_angle_deg",     type=float, default=SIDE_GRASP_ANGLE_DEG,
                        help="手掌朝向容忍角度（默认 60°）")
    # ── 控制开关 ─────────────────────────────────────────
    parser.add_argument("--no_side_filter",     action="store_true",
                        help="禁用侧面抓取过滤（对比实验用）")
    parser.add_argument("--use_side_prior",     action="store_true",
                        help="用手工构造的侧面接近初始位姿替代随机初始位姿")
    args   = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    # 1. 点云
    cprint(f"=> 加载点云: {args.pcd_path}", "cyan")
    obj_pc, front_val, axis_idx, front_sign, approach_vec = load_side_pc(
        args.pcd_path, args.num_points, args.camera_axis, args.camera_dir)
    debug_pc  = obj_pc.to(device)
    axis_name = ['X', 'Y', 'Z'][axis_idx]
    cprint(f"   物体正面 {axis_name} = {front_val:.4f}m", "green")
    cprint(f"   手掌接近方向: {approach_vec}", "green")

    # 2. 网络
    cprint(f"=> 加载网络: {args.ckpt_name}", "cyan")
    network = create_network(
        SimpleNamespace(emb_dim=512, latent_dim=64, pretrain=None,
                        center_pc=True, block_computing=True),
        mode="validate").to(device)
    network.load_state_dict(torch.load(
        os.path.join(ROOT_DIR, f"ckpt/model/{args.ckpt_name}.pth"),
        map_location=device))
    network.eval()
    cprint("   网络加载完成", "green")

    # 3. 手部
    cprint(f"=> 加载手部: {ROBOT_NAME}", "cyan")
    hand = create_hand_model(ROBOT_NAME, device)

    success_num = 0;  total_num = 0;  all_time = []
    mode_str   = "侧面先验" if args.use_side_prior else "网络预测"
    filter_str = "无过滤"   if args.no_side_filter  else "侧面过滤"
    cprint(f"\n=> 验证: {args.num_rounds}轮×{args.batch_size}姿态 | "
           f"模式:{mode_str} | {filter_str}\n", "cyan")

    for rnd in range(args.num_rounds):
        cprint(f"── Round {rnd+1}/{args.num_rounds} ──", "yellow")

        # ── 构造 batch ────────────────────────────────────
        if args.use_side_prior:
            iq_b, rpc_b, opc_b = make_side_grasp_q(
                hand, device, obj_pc, front_val,
                axis_idx, front_sign, approach_vec,
                args.batch_size, args.hover_dist)
            cprint(f"   使用侧面先验初始位姿，手掌{axis_name}={front_val + front_sign * (-args.hover_dist):.4f}m",
                   "magenta")
        else:
            iq_list, rpc_list, opc_list = [], [], []
            for _ in range(args.batch_size):
                iq  = hand.get_initial_q()
                rpc = hand.get_transformed_links_pc(iq)[:, :3]
                opc = obj_pc + torch.randn_like(obj_pc) * 0.002
                rn  = rpc.shape[0]
                if rn < args.num_points:
                    pad = torch.randint(0, rn, (args.num_points - rn,))
                    rpc = torch.cat([rpc, rpc[pad]], dim=0)
                else:
                    rpc = rpc[:args.num_points]
                iq_list.append(iq);  rpc_list.append(rpc);  opc_list.append(opc)
            iq_b  = torch.stack(iq_list).to(device)
            rpc_b = torch.stack(rpc_list).to(device)
            opc_b = torch.stack(opc_list).to(device)

        # ── 网络推理 + IK ─────────────────────────────────
        pq_batch, tlist = infer_batch(
            network, hand, iq_b, rpc_b, opc_b, args.batch_size, device)
        all_time.extend(tlist)

        # ── 侧面抓取过滤 ──────────────────────────────────
        if not args.no_side_filter:
            pq_filtered, valid_idx = filter_side_grasp(
                pq_batch, front_val, axis_idx, front_sign,
                approach_vec, args.side_angle_deg)
            # 打印每个样本的调试信息
            for ii, qi in enumerate(pq_batch.detach().cpu().numpy()):
                try:
                    rot  = R.from_rotvec(qi[3:6]).as_matrix()
                    lzw  = rot @ np.array([0, 0, 1])
                    ang  = np.degrees(np.arccos(np.clip(np.dot(lzw, approach_vec), -1, 1)))
                    mark = "✓" if ii in valid_idx else "✗"
                    cprint(f"   样本{ii:2d}: {axis_name}={qi[axis_idx]:.3f}m, "
                           f"朝向角={ang:.1f}°  {mark}", "white")
                except Exception:
                    pass
        else:
            pq_filtered = pq_batch
            valid_idx   = np.arange(args.batch_size)

        if len(pq_filtered) == 0:
            cprint("   所有姿态被过滤，跳过本轮", "red")
            total_num += args.batch_size
            continue

        saved_rpc0 = rpc_b[0:1].clone()

        # ── IsaacGym 首次验证 ─────────────────────────────
        cprint(f"   → IsaacGym 首次验证（{len(pq_filtered)}个姿态）...", "cyan")
        success, isaac_ret = validate_isaac(
            ROBOT_NAME, OBJECT_NAME, pq_filtered,
            gpu=args.gpu, debug_pc=debug_pc)

        # ── TRIGGERED 处理 ────────────────────────────────
        if isinstance(success, str) and success == "TRIGGERED":
            info = isaac_ret
            cprint(f"==> 触觉触发(Step:{info.get('step')})，启动精调...",
                   "magenta", attrs=["bold"])

            tp = info["tip_pos"]
            if isinstance(tp, dict):
                tv = torch.tensor([tp["x"], tp["y"], tp["z"]], device=device)
            elif hasattr(tp, "x"):
                tv = torch.tensor([float(tp.x), float(tp.y), float(tp.z)], device=device)
            else:
                tv = torch.as_tensor(tp, dtype=torch.float32, device=device)
                if tv.shape != (3,):
                    cprint("WARN: tip_pos 异常，跳过", "red")
                    total_num += len(pq_filtered);  continue

            dist = torch.norm(debug_pc - tv, dim=-1)
            _, li = torch.topk(dist, k=min(256, debug_pc.shape[0]), largest=False)
            lpc   = debug_pc[li].unsqueeze(0)
            cprint(f"   局部点云{lpc.shape[1]}点，触碰@{tv.cpu().numpy().round(4)}", "magenta")

            print(f"   精调前 Q[:5]: {pq_filtered[0,:5].detach().cpu().numpy()}")
            refined_q, _ = run_local_refinement(
                network, hand, saved_rpc0, lpc,
                pq_filtered[0:1].clone(), device)
            print(f"   精调后 Q[:5]: {refined_q[0,:5].detach().cpu().numpy()}")

            cprint("   → IsaacGym 二次验证（关闭触觉中断）...", "cyan")
            s2, _ = validate_isaac(ROBOT_NAME, OBJECT_NAME, refined_q,
                                   gpu=args.gpu, debug_pc=debug_pc,
                                   is_refinement=True)
            if isinstance(s2, torch.Tensor) and s2[0].item():
                cprint("   精调: ✓ 成功", "green", attrs=["bold"]);  success_num += 1
            else:
                cprint("   精调: ✗ 失败", "red")

            total_num += 1
            cprint(f"   SR: {success_num}/{total_num} "
                   f"({success_num/total_num*100:.1f}%)", "yellow")
            continue

        # ── 正常批次统计 ──────────────────────────────────
        succ_num = success.sum().item() if isinstance(success, torch.Tensor) else 0
        cprint(f"   Round {rnd+1} 成功: {succ_num}/{len(pq_filtered)}",
               "green" if succ_num > 0 else "red")
        success_num += succ_num
        total_num   += len(pq_filtered)

    # 汇总
    cprint("\n" + "="*50, "yellow")
    if total_num > 0:
        cprint(f"[Final] 成功率: {success_num}/{total_num} "
               f"({success_num/total_num*100:.1f}%)",
               "yellow", attrs=["bold"])
        if all_time:
            cprint(f"[Final] 平均IK时间: {np.mean(all_time)*1000:.1f}ms", "yellow")
    else:
        cprint("[Final] 未评估任何样本", "red")


if __name__ == "__main__":
    warnings.simplefilter(action="ignore", category=FutureWarning)
    torch.set_num_threads(8)
    main()
