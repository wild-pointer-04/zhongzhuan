"""
validate_duck_v7.py  —— 侧面抓取（全面修复版）
================================================================
相比 v6b 的核心改动：

  修复1：palm_forward_axis 默认改为 'y'
         wujihand 手心朝 +Y（local_Y→world 最接近 approach=[0,1,0]），
         -y 才是 180° 反向（手背朝物体），z/-z 是手侧躺（up_angle=90°）

  修复2：tip_pos 坐标系对齐
         Isaac 返回世界坐标，点云已中心化。触碰点距离检测前先把 tip_pos
         减去点云均值（obj_center_offset）才能正确比较。

  修复3：Phase1 手指闭合不足
         IK 优化会把手指关节往 initial_q 方向拉（张开），导致 Phase1 时
         手指几乎是张开的。新增 --joint_target_scale 参数（默认 0.3），
         在送入 IsaacGym 前把 q[6:] 乘以该系数，让手指更卷曲。

  修复4：初始穿透过滤
         首次验证时 Step0 力 > MAX_INIT_FORCE(80N) 的 env 不计入统计，
         并在 TRIGGERED 路径增加 init_force 检测。

  修复5：局部精调坐标系修复
         精调时 local_pc 用中心化点云（已与 tip_pos 世界坐标对齐后的版本）

坐标系约定（中心化后）：
  Y_min（负侧）= 相机/正面侧
  approach_vec = [0, +1, 0]（从正面向背面接近）
  hand palm 朝 +Y（local_Y 轴）

用法：
    python scripts/validate_duck_v7.py \
        --pcd_path /home/eureka/duck_point/duck_v6.pcd \
        --ckpt_name model_shadowhand \
        --batch_size 20 \
        --num_rounds 5 \
        --camera_axis y \
        --camera_dir neg \
        --hover_dist 0.06 \
        --finger_len 0.08 \
        --finger_bend 0.55 \
        --joint_target_scale 0.3 \
        --palm_forward_axis y \
        --use_side_prior
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

# ── 默认值 ────────────────────────────────────────────────────
OBJECT_NAME          = "contactdb+cup"
ROBOT_NAME           = "wujihand"
NUM_POINTS           = 512
HOVER_DIST           = 0.06    # 手掌基座到物体正面距离（m）
FINGER_LEN           = 0.08    # 手指估计长度（m），加到 hover 上
FINGER_BEND          = 0.55    # 预弯比例（0=张开, 1=完全弯曲）
JOINT_TARGET_SCALE   = 0.3     # Phase1 送入 Isaac 前关节目标缩放（越小越卷曲）
SIDE_GRASP_ANGLE_DEG = 40.0    # 过滤角度阈值（更严格）
MAX_INIT_FORCE       = 80.0    # 穿透判定阈值（N）
MAX_TIP_DIST         = 0.08    # 触碰点到点云最大距离（m），修复坐标系后可收紧
# ─────────────────────────────────────────────────────────────

PALM_AXIS_MAP = {
    'x':  np.array([ 1., 0., 0.]),
    '-x': np.array([-1., 0., 0.]),
    'y':  np.array([ 0., 1., 0.]),
    '-y': np.array([ 0.,-1., 0.]),
    'z':  np.array([ 0., 0., 1.]),
    '-z': np.array([ 0., 0.,-1.]),
}


# ══════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════

def get_camera_axis_info(camera_axis, camera_dir):
    idx = {'x': 0, 'y': 1, 'z': 2}[camera_axis.lower()]
    if camera_dir == 'neg':
        front_sign   = -1
        approach_vec = np.eye(3)[idx]      # +Y 方向接近
    else:
        front_sign   = +1
        approach_vec = -np.eye(3)[idx]
    return idx, front_sign, approach_vec.astype(float)


def align_rotvec(from_vec, to_vec):
    """计算将 from_vec 旋转到 to_vec 的旋转向量"""
    fv = from_vec / (np.linalg.norm(from_vec) + 1e-8)
    tv = to_vec   / (np.linalg.norm(to_vec)   + 1e-8)
    c  = np.cross(fv, tv)
    d  = np.clip(np.dot(fv, tv), -1., 1.)
    s  = np.linalg.norm(c)
    if s < 1e-6:
        return np.zeros(3) if d > 0 else np.array([0., 0., 1.]) * np.pi
    return (c / s) * np.arctan2(s, d)


# ══════════════════════════════════════════════════════════════
#  点云加载
# ══════════════════════════════════════════════════════════════

def load_side_pc(pcd_path, num_points=NUM_POINTS, camera_axis='y', camera_dir='neg'):
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    if len(pts) == 0:
        raise ValueError(f"点云为空：{pcd_path}")
    cprint(f"[PC] 原始点数: {len(pts)}", "cyan")

    # 中心化，保存偏移量用于坐标系修复
    obj_center = pts.mean(axis=0)
    pts -= obj_center
    cprint(f"[PC] 中心化偏移 obj_center={obj_center.round(4)}", "cyan")
    cprint(f"[PC] X:[{pts[:,0].min():.3f},{pts[:,0].max():.3f}]"
           f"  Y:[{pts[:,1].min():.3f},{pts[:,1].max():.3f}]"
           f"  Z:[{pts[:,2].min():.3f},{pts[:,2].max():.3f}]", "cyan")

    axis_idx, front_sign, approach_vec = get_camera_axis_info(camera_axis, camera_dir)
    front_val = float(pts[:, axis_idx].min() if front_sign == -1
                      else pts[:, axis_idx].max())
    axis_name = 'XYZ'[axis_idx]
    cprint(f"[PC] 相机轴:{axis_name}({camera_dir})  正面={front_val:.4f}m", "cyan")

    idx = np.random.choice(len(pts), num_points, replace=(len(pts) < num_points))
    # 返回中心化点云张量 + 原始中心（用于世界坐标对齐）
    return torch.from_numpy(pts[idx]), front_val, axis_idx, front_sign, approach_vec, obj_center


# ══════════════════════════════════════════════════════════════
#  侧面先验初始位姿
# ══════════════════════════════════════════════════════════════

def make_side_grasp_q(hand, device, obj_pc, front_val,
                      axis_idx, front_sign, approach_vec,
                      batch_size,
                      hover_dist=HOVER_DIST,
                      finger_len=FINGER_LEN,
                      finger_bend=FINGER_BEND,
                      joint_target_scale=JOINT_TARGET_SCALE,
                      palm_fwd_vec=None):
    """
    手掌真实悬停距离 = hover_dist + finger_len
    关节目标额外用 joint_target_scale 缩放以保证 Phase1 时手指卷曲。
    """
    if palm_fwd_vec is None:
        palm_fwd_vec = np.array([0., 1., 0.])   # 默认 +Y

    total_dist   = hover_dist + finger_len
    # front_sign=-1（相机在Y负侧）: palm 在正面更负的位置
    palm_on_axis = front_val + front_sign * total_dist

    iq_list, rpc_list, opc_list = [], [], []
    num_points = obj_pc.shape[0]
    other_axes = [i for i in range(3) if i != axis_idx]

    center_oa = obj_pc[:, other_axes].mean(dim=0).numpy()
    extent_oa = (obj_pc[:, other_axes].max(dim=0).values
               - obj_pc[:, other_axes].min(dim=0).values).numpy()

    # Z 高度：物体中段（20%~80%分位）
    z_vals = obj_pc[:, 2].numpy()
    z_lo   = float(np.percentile(z_vals, 20))
    z_hi   = float(np.percentile(z_vals, 80))
    z_mid  = (z_lo + z_hi) / 2.0

    axis_name = 'XYZ'[axis_idx]
    cprint(f"   [make_q] front_val={front_val:.4f}  front_sign={front_sign}"
           f"  hover={hover_dist}+finger={finger_len}  palm_{axis_name}={palm_on_axis:.4f}", "magenta")
    cprint(f"   [make_q] Z范围 [{z_lo:.3f}, {z_hi:.3f}]m  palm_fwd={palm_fwd_vec}", "magenta")

    # 计算基础旋转：将手掌局部朝前轴对齐到接近方向
    base_rv = align_rotvec(palm_fwd_vec, approach_vec)
    cprint(f"   [make_q] base_rotvec={base_rv.round(4)}  norm={np.linalg.norm(base_rv):.3f}", "magenta")

    for _ in range(batch_size):
        q_new = hand.get_initial_q().clone()

        # ── 位置 ──────────────────────────────────────────────
        pos = np.zeros(3)
        pos[axis_idx] = palm_on_axis

        # 水平方向（非相机轴、非Z轴）小幅随机
        horiz_axes = [a for a in other_axes if a != 2]
        if horiz_axes:
            ha = horiz_axes[0]
            ha_local = other_axes.index(ha)
            pos[ha] = float(center_oa[ha_local]) \
                    + (np.random.rand() - 0.5) * float(extent_oa[ha_local]) * 0.4

        # Z：物体中段随机
        pos[2] = z_mid + (np.random.rand() - 0.5) * (z_hi - z_lo) * 0.4

        q_new[0] = float(pos[0])
        q_new[1] = float(pos[1])
        q_new[2] = float(pos[2])

        # ── 旋转 ──────────────────────────────────────────────
        # ±8° 扰动，接近轴严格禁止扰动
        r_base  = R.from_rotvec(base_rv)
        perturb = np.random.uniform(-np.pi / 22, np.pi / 22, size=3)
        perturb[axis_idx] = 0.0                # 接近轴不扰动
        r_final = R.from_rotvec(perturb) * r_base
        rv      = r_final.as_rotvec()

        q_new[3] = float(rv[0])
        q_new[4] = float(rv[1])
        q_new[5] = float(rv[2])

        # ── 手指 ──────────────────────────────────────────────
        # Step1: 预弯（finger_bend=0.55 → 关节值×0.45，接近闭合）
        q_new[6:] = q_new[6:] * (1.0 - finger_bend)
        # Step2: joint_target_scale 额外缩放（Phase1 送入 Isaac 时也用这个）
        # 这里先存原始预弯值，Isaac 那边再缩放
        # （joint_target_scale 在 pq_batch 送出前统一处理）

        # ── robot_pc ──────────────────────────────────────────
        robot_pc = hand.get_transformed_links_pc(q_new)[:, :3]
        rn = robot_pc.shape[0]
        if rn < num_points:
            pad      = torch.randint(0, rn, (num_points - rn,))
            robot_pc = torch.cat([robot_pc, robot_pc[pad]], dim=0)
        else:
            robot_pc = robot_pc[:num_points]

        obj_noisy = obj_pc + torch.randn_like(obj_pc) * 0.001

        iq_list.append(q_new)
        rpc_list.append(robot_pc)
        opc_list.append(obj_noisy)

    return (
        torch.stack(iq_list).to(device),
        torch.stack(rpc_list).to(device),
        torch.stack(opc_list).to(device),
    )


def apply_joint_target_scale(pq_batch, scale):
    """在送入 IsaacGym 前将手指关节乘以 scale（使手指更卷曲）"""
    pq_scaled = pq_batch.clone()
    pq_scaled[:, 6:] = pq_scaled[:, 6:] * scale
    return pq_scaled


# ══════════════════════════════════════════════════════════════
#  侧面过滤
# ══════════════════════════════════════════════════════════════

def filter_side_grasp(pq_batch, front_val, axis_idx, front_sign,
                      approach_vec, palm_fwd_vec,
                      angle_thresh=SIDE_GRASP_ANGLE_DEG):
    q_np       = pq_batch.detach().cpu().numpy()
    valid_mask = []
    margin     = 0.02   # 允许少量超出正面（IK 可能轻微调整）
    axis_name  = 'XYZ'[axis_idx]

    for q in q_np:
        palm_ax = q[axis_idx]
        # 手掌必须在物体正面侧（含少量 margin）
        pos_ok  = (palm_ax < front_val + margin) if front_sign == -1 \
                  else (palm_ax > front_val - margin)
        if not pos_ok:
            valid_mask.append(False); continue

        # 旋转向量范数检查
        rv_norm = np.linalg.norm(q[3:6])
        if rv_norm > np.pi * 1.5:
            valid_mask.append(False); continue

        try:
            rot    = R.from_rotvec(q[3:6]).as_matrix()
            fwd_w  = rot @ palm_fwd_vec
            angle  = np.degrees(np.arccos(np.clip(np.dot(fwd_w, approach_vec), -1., 1.)))
            valid_mask.append(angle < angle_thresh)
        except Exception:
            valid_mask.append(False)

    valid_mask  = np.array(valid_mask)
    valid_count = int(valid_mask.sum())
    total       = len(valid_mask)

    if valid_count == 0:
        cprint(f"   [Filter] 全部过滤，保留原始 batch", "yellow")
        return pq_batch, np.arange(total)

    cprint(f"   [Filter] 侧面过滤: {valid_count}/{total} 通过", "green")
    indices = np.where(valid_mask)[0]
    return pq_batch[torch.from_numpy(indices)], indices


# ══════════════════════════════════════════════════════════════
#  推理 & 精调
# ══════════════════════════════════════════════════════════════

def infer_batch(network, hand, iq_b, rpc_b, opc_b, batch_size, device):
    pq_list, t_list = [], []
    for i in tqdm(range(batch_size), desc="推理", leave=False):
        iq = iq_b[i:i+1]; rpc = rpc_b[i:i+1]; opc = opc_b[i:i+1]
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


def side_prior_only_batch(iq_b, batch_size):
    cprint("   [NoNetwork] 跳过网络推理，直接使用先验位姿", "yellow")
    return iq_b, [0.0] * batch_size


def run_local_refinement(network, hand, robot_pc, local_pc, initial_q, device):
    with torch.no_grad():
        if local_pc.shape[1] < 512:
            idx = torch.randint(0, local_pc.shape[1], (1, 512), device=device)
            lpc = torch.gather(local_pc, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        else:
            lpc = local_pc[:, :512, :]
        dro = network(robot_pc, lpc)["dro"].detach()
    mlat  = multilateration(dro, lpc)
    tf, _ = compute_link_pose(hand.links_pc, mlat, is_train=False)
    otf   = process_transform(hand.pk_chain, tf)
    layer = create_problem(hand.pk_chain, otf.keys())
    return optimization(hand.pk_chain, layer, initial_q, otf), otf


# ══════════════════════════════════════════════════════════════
#  main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="侧面抓取 v7（全面修复版）")

    parser.add_argument("--pcd_path",          type=str,   required=True)
    parser.add_argument("--ckpt_name",          type=str,   default="model_shadowhand")
    parser.add_argument("--batch_size",         type=int,   default=20)
    parser.add_argument("--num_rounds",         type=int,   default=5)
    parser.add_argument("--gpu",                type=int,   default=0)
    parser.add_argument("--num_points",         type=int,   default=NUM_POINTS)

    # 相机/接近
    parser.add_argument("--camera_axis",        type=str,   default="y",
                        choices=["x","y","z"])
    parser.add_argument("--camera_dir",         type=str,   default="neg",
                        choices=["neg","pos"])
    parser.add_argument("--hover_dist",         type=float, default=HOVER_DIST,
                        help="手掌基座到物体正面的距离 m（不含手指，默认 0.06）")
    parser.add_argument("--finger_len",         type=float, default=FINGER_LEN,
                        help="手指估计长度 m（默认 0.08）")
    parser.add_argument("--finger_bend",        type=float, default=FINGER_BEND,
                        help="初始预弯比例 0~1（默认 0.55）")
    parser.add_argument("--joint_target_scale", type=float, default=JOINT_TARGET_SCALE,
                        help="送入 Isaac 前关节目标缩放（默认 0.3，越小手指越卷曲）")
    parser.add_argument("--side_angle_deg",     type=float, default=SIDE_GRASP_ANGLE_DEG,
                        help="朝向过滤角度阈值（默认 40°）")
    parser.add_argument("--palm_forward_axis",  type=str,   default="y",
                        choices=list(PALM_AXIS_MAP.keys()),
                        help="手掌局部'朝前'轴（wujihand 正确值为 y）")

    # 控制
    parser.add_argument("--no_side_filter",     action="store_true")
    parser.add_argument("--use_side_prior",     action="store_true")
    parser.add_argument("--no_network",         action="store_true",
                        help="跳过 DRO 网络，直接用先验位姿")

    args   = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    palm_fwd_vec = PALM_AXIS_MAP[args.palm_forward_axis]
    axis_name    = 'XYZ'[{'x':0,'y':1,'z':2}[args.camera_axis.lower()]]

    # ── 点云 ──────────────────────────────────────────────────
    cprint(f"\n=> 加载点云: {args.pcd_path}", "cyan")
    obj_pc, front_val, axis_idx, front_sign, approach_vec, obj_center = load_side_pc(
        args.pcd_path, args.num_points, args.camera_axis, args.camera_dir)
    debug_pc = obj_pc.to(device)

    # ★ 修复2：保存点云中心，用于 tip_pos 坐标系对齐
    obj_center_t = torch.from_numpy(obj_center).float().to(device)
    cprint(f"   [坐标系] obj_center={obj_center.round(4)}（Isaac世界坐标→点云坐标需减此值）", "green")

    total_hover = args.hover_dist + args.finger_len
    palm_target = front_val + front_sign * total_hover
    cprint(f"   正面 {axis_name}={front_val:.4f}m", "green")
    cprint(f"   手掌目标 {axis_name}={palm_target:.4f}m  "
           f"(hover={args.hover_dist}+finger={args.finger_len}={total_hover:.3f}m)", "green")
    cprint(f"   palm_fwd={args.palm_forward_axis}  bend={args.finger_bend}"
           f"  joint_scale={args.joint_target_scale}", "green")

    # ── 网络 ──────────────────────────────────────────────────
    if not args.no_network:
        cprint(f"\n=> 加载网络: {args.ckpt_name}", "cyan")
        network = create_network(
            SimpleNamespace(emb_dim=512, latent_dim=64, pretrain=None,
                            center_pc=True, block_computing=True),
            mode="validate").to(device)
        network.load_state_dict(torch.load(
            os.path.join(ROOT_DIR, f"ckpt/model/{args.ckpt_name}.pth"),
            map_location=device))
        network.eval()
        cprint("   网络加载完成", "green")
    else:
        network = None
        cprint("\n=> [NoNetwork 模式] 跳过网络加载", "yellow")

    # ── 手部 ──────────────────────────────────────────────────
    cprint(f"\n=> 加载手部: {ROBOT_NAME}", "cyan")
    hand = create_hand_model(ROBOT_NAME, device)

    # ── 主循环 ────────────────────────────────────────────────
    success_num, total_num, all_time = 0, 0, []
    cprint(f"\n=> 验证: {args.num_rounds}轮×{args.batch_size}  "
           f"{'先验' if args.use_side_prior else '随机'}  "
           f"{'网络OFF' if args.no_network else '网络ON'}  "
           f"palm:{args.palm_forward_axis}  bend:{args.finger_bend}"
           f"  joint_scale:{args.joint_target_scale}\n", "cyan")

    for rnd in range(args.num_rounds):
        cprint(f"── Round {rnd+1}/{args.num_rounds} ──", "yellow")

        # ── 构造 batch ────────────────────────────────────────
        if args.use_side_prior:
            iq_b, rpc_b, opc_b = make_side_grasp_q(
                hand, device, obj_pc, front_val,
                axis_idx, front_sign, approach_vec,
                args.batch_size,
                hover_dist=args.hover_dist,
                finger_len=args.finger_len,
                finger_bend=args.finger_bend,
                joint_target_scale=args.joint_target_scale,
                palm_fwd_vec=palm_fwd_vec)
        else:
            iq_list, rpc_list, opc_list = [], [], []
            for _ in range(args.batch_size):
                iq  = hand.get_initial_q()
                rpc = hand.get_transformed_links_pc(iq)[:, :3]
                opc = obj_pc + torch.randn_like(obj_pc) * 0.001
                rn  = rpc.shape[0]
                if rn < args.num_points:
                    pad = torch.randint(0, rn, (args.num_points - rn,))
                    rpc = torch.cat([rpc, rpc[pad]], dim=0)
                else:
                    rpc = rpc[:args.num_points]
                iq_list.append(iq); rpc_list.append(rpc); opc_list.append(opc)
            iq_b  = torch.stack(iq_list).to(device)
            rpc_b = torch.stack(rpc_list).to(device)
            opc_b = torch.stack(opc_list).to(device)

        # ── 推理 ──────────────────────────────────────────────
        if args.no_network:
            pq_batch, tlist = side_prior_only_batch(iq_b, args.batch_size)
        else:
            pq_batch, tlist = infer_batch(
                network, hand, iq_b, rpc_b, opc_b, args.batch_size, device)
        all_time.extend(tlist)

        # ★ 修复3：关节目标缩放（让手指更卷曲送入 Isaac）
        pq_for_isaac = apply_joint_target_scale(pq_batch, args.joint_target_scale)

        # ── 过滤 ──────────────────────────────────────────────
        if not args.no_side_filter:
            pq_filtered, valid_idx = filter_side_grasp(
                pq_for_isaac, front_val, axis_idx, front_sign,
                approach_vec, palm_fwd_vec, args.side_angle_deg)
            q_np = pq_for_isaac.detach().cpu().numpy()
            for ii in range(q_np.shape[0]):
                try:
                    rot   = R.from_rotvec(q_np[ii,3:6]).as_matrix()
                    fwd_w = rot @ palm_fwd_vec
                    ang   = np.degrees(np.arccos(
                                np.clip(np.dot(fwd_w, approach_vec), -1., 1.)))
                    mark  = "✓" if ii in valid_idx else "✗"
                    cprint(f"   样本{ii:2d}: {axis_name}={q_np[ii,axis_idx]:.3f}m"
                           f"  朝向角={ang:.1f}°  {mark}", "white")
                except Exception:
                    pass
        else:
            pq_filtered = pq_for_isaac
            valid_idx   = np.arange(args.batch_size)

        if len(pq_filtered) == 0:
            cprint("   所有姿态被过滤", "red")
            total_num += args.batch_size; continue

        saved_rpc0 = rpc_b[0:1].clone()

        # ── IsaacGym 首次验证 ─────────────────────────────────
        cprint(f"   → IsaacGym 首次验证（{len(pq_filtered)}个）...", "cyan")
        success, isaac_ret = validate_isaac(
            ROBOT_NAME, OBJECT_NAME, pq_filtered,
            gpu=args.gpu, debug_pc=debug_pc)

        # ── TRIGGERED 路径 ────────────────────────────────────
        if isinstance(success, str) and success == "TRIGGERED":
            info       = isaac_ret
            step       = info.get("step", -1)
            init_force = info.get("init_force", 0.0)
            cprint(f"==> 触觉触发(Step:{step}, init_force:{init_force:.1f}N)", "magenta", attrs=["bold"])

            # ★ 修复4：穿透检测
            if init_force > MAX_INIT_FORCE:
                cprint(f"   初始力过大({init_force:.1f}N > {MAX_INIT_FORCE}N)，穿透跳过", "red")
                total_num += len(pq_filtered); continue

            # ★ 修复2：tip_pos 坐标系对齐
            tp = info["tip_pos"]
            if isinstance(tp, dict):
                tv_world = torch.tensor([tp["x"], tp["y"], tp["z"]], dtype=torch.float32, device=device)
            elif hasattr(tp, "x"):
                tv_world = torch.tensor([float(tp.x), float(tp.y), float(tp.z)],
                                        dtype=torch.float32, device=device)
            else:
                tv_world = torch.as_tensor(tp, dtype=torch.float32, device=device)
                if tv_world.shape != (3,):
                    cprint("   tip_pos 格式异常，跳过", "red")
                    total_num += len(pq_filtered); continue

            # Isaac 世界坐标 → 中心化点云坐标
            tv = tv_world - obj_center_t
            cprint(f"   tip_world={tv_world.cpu().numpy().round(4)}"
                   f"  tip_centered={tv.cpu().numpy().round(4)}", "magenta")

            dist_to_obj = torch.norm(debug_pc - tv, dim=-1).min().item()
            cprint(f"   触碰点到点云最近距离: {dist_to_obj:.4f}m", "magenta")
            if dist_to_obj > MAX_TIP_DIST:
                cprint(f"   触碰点太远({dist_to_obj:.3f}m > {MAX_TIP_DIST}m)，跳过", "red")
                total_num += len(pq_filtered); continue

            rv_norm_before = float(torch.norm(pq_filtered[0, 3:6]).item())
            if rv_norm_before > np.pi * 1.5:
                cprint(f"   精调前旋转异常(norm={rv_norm_before:.2f})，跳过", "red")
                total_num += 1; continue

            # 局部点云（中心化坐标）
            dist  = torch.norm(debug_pc - tv, dim=-1)
            _, li = torch.topk(dist, k=min(256, debug_pc.shape[0]), largest=False)
            lpc   = debug_pc[li].unsqueeze(0)
            cprint(f"   局部点云{lpc.shape[1]}点  触碰@{tv.cpu().numpy().round(4)}", "magenta")
            cprint(f"   精调前: {pq_filtered[0,:6].detach().cpu().numpy().round(4)}", "white")

            if args.no_network:
                refined_q = pq_filtered[0:1].clone()
                cprint("   [NoNetwork] 跳过精调", "yellow")
            else:
                refined_q, _ = run_local_refinement(
                    network, hand, saved_rpc0, lpc,
                    pq_filtered[0:1].clone(), device)
                # 精调后也应用 joint_target_scale
                refined_q = apply_joint_target_scale(refined_q, args.joint_target_scale)
                cprint(f"   精调后: {refined_q[0,:6].detach().cpu().numpy().round(4)}", "white")

            rv_norm_after = float(torch.norm(refined_q[0, 3:6]).item())
            if rv_norm_after > np.pi * 1.5:
                cprint(f"   精调后旋转爆炸(norm={rv_norm_after:.2f})，跳过", "red")
                total_num += 1
                cprint(f"   SR: {success_num}/{total_num} ({success_num/total_num*100:.1f}%)", "yellow")
                continue

            cprint("   → 二次验证...", "cyan")
            s2, _ = validate_isaac(
                ROBOT_NAME, OBJECT_NAME, refined_q,
                gpu=args.gpu, debug_pc=debug_pc, is_refinement=True)

            if isinstance(s2, torch.Tensor) and s2[0].item():
                cprint("   精调: ✓ 成功", "green", attrs=["bold"]); success_num += 1
            else:
                cprint("   精调: ✗ 失败", "red")

            total_num += 1
            cprint(f"   SR: {success_num}/{total_num} ({success_num/total_num*100:.1f}%)", "yellow")
            continue

        # ── 正常批次统计 ──────────────────────────────────────
        if isinstance(success, torch.Tensor):
            succ_num = int(success.sum().item())
        else:
            succ_num = 0
        cprint(f"   Round {rnd+1} 成功: {succ_num}/{len(pq_filtered)}",
               "green" if succ_num > 0 else "red")
        success_num += succ_num
        total_num   += len(pq_filtered)

    # ── 汇总 ──────────────────────────────────────────────────
    cprint("\n" + "=" * 55, "yellow")
    if total_num > 0:
        cprint(f"[Final] 成功率: {success_num}/{total_num} ({success_num/total_num*100:.1f}%)",
               "yellow", attrs=["bold"])
        if all_time and not args.no_network:
            cprint(f"[Final] 平均IK时间: {np.mean(all_time)*1000:.1f}ms", "yellow")
    else:
        cprint("[Final] 未评估任何样本", "red")


if __name__ == "__main__":
    warnings.simplefilter(action="ignore", category=FutureWarning)
    torch.set_num_threads(8)
    main()
