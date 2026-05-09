"""
validate_duck_v6.py  —— 侧面抓取版（半侧点云，修正手掌朝向与位置）
================================================================
相比 v5 的核心修改：
  1. 修复 palm_on_axis 符号 bug：front_sign=-1 时手掌应在 front_val 更负处
     旧：palm_on_axis = front_val - front_sign * hover_dist  （符号混乱）
     新：palm_on_axis = front_val + front_sign * hover_dist  （front_sign=-1 → 更负）
     注意：front_sign=-1，hover_dist=+0.08 → palm = front_val + (-1)*0.08 = front_val - 0.08 ✓

  2. 修复手掌朝向：wujihand 手掌"朝前"轴需要实测确认，
     提供 --palm_forward_axis 参数（默认 z，可选 x/y/-x/-y/-z）
     用 _align_rotvec(palm_fwd, approach_vec) 替代固定 local_z

  3. 手掌 Z 高度限制在物体中段（不去夹杯口边缘）：
     Z 范围限制在 [obj_z_min + 20%, obj_z_max - 20%]

  4. 精调前检查 Q 旋转向量 norm（> π 说明旋转异常）

  5. 新增 --debug_palm_axis 开关：打印手掌各局部轴的世界方向，方便确认朝向

坐标系（中心化后）：
  Y_min = 相机侧（正面），Y_max = 背面
  Z = 朝上，X = 水平

用法：
    python scripts/validate_duck_v6.py \
        --pcd_path /home/eureka/duck_point/duck_v6.pcd \
        --ckpt_name model_shadowhand \
        --batch_size 20 \
        --num_rounds 5 \
        --camera_axis y \
        --camera_dir neg \
        --hover_dist 0.10 \
        --side_angle_deg 45 \
        --palm_forward_axis z \
        --use_side_prior \
        --debug_palm_axis
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

# ── 全局默认值 ────────────────────────────────────────────────
OBJECT_NAME          = "contactdb+cup"
ROBOT_NAME           = "wujihand"
NUM_POINTS           = 512
HOVER_DIST           = 0.10          # 手掌距正面距离（m），加大避免穿透
SIDE_GRASP_ANGLE_DEG = 45.0
MAX_INIT_FORCE       = 100.0         # Step0 穿透判定阈值（N）
MAX_TIP_DIST         = 0.15          # 触碰点合法距离（m）
# ─────────────────────────────────────────────────────────────

# 手掌局部"朝前"轴映射（字符串 → np.ndarray）
PALM_AXIS_MAP = {
    'x':  np.array([ 1.0,  0.0,  0.0]),
    '-x': np.array([-1.0,  0.0,  0.0]),
    'y':  np.array([ 0.0,  1.0,  0.0]),
    '-y': np.array([ 0.0, -1.0,  0.0]),
    'z':  np.array([ 0.0,  0.0,  1.0]),
    '-z': np.array([ 0.0,  0.0, -1.0]),
}


# ══════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════

def get_camera_axis_info(camera_axis: str, camera_dir: str):
    """
    返回轴索引、正面符号、手掌接近方向向量。

    front_sign=-1 → 正面是轴最小值（相机在负侧）
    approach_vec  → 手掌朝向物体的方向（从手掌指向物体）
    """
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map[camera_axis.lower()]
    if camera_dir == 'neg':
        # 相机在负侧 → 正面=轴最小值 → 手掌从更负处来 → 朝向物体=+axis
        front_sign   = -1
        approach_vec = np.zeros(3); approach_vec[axis_idx] = +1.0
    else:
        front_sign   = +1
        approach_vec = np.zeros(3); approach_vec[axis_idx] = -1.0
    return axis_idx, front_sign, approach_vec


def _align_rotvec(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
    """将 from_vec 旋转到 to_vec 的旋转向量。"""
    from_vec = from_vec / (np.linalg.norm(from_vec) + 1e-8)
    to_vec   = to_vec   / (np.linalg.norm(to_vec)   + 1e-8)
    cross    = np.cross(from_vec, to_vec)
    dot      = np.clip(np.dot(from_vec, to_vec), -1.0, 1.0)
    sin_a    = np.linalg.norm(cross)

    if sin_a < 1e-6:
        return np.zeros(3) if dot > 0 else (
            np.array([0.0, 1.0, 0.0]) * np.pi
            if abs(from_vec[0]) < 0.9
            else np.array([0.0, 0.0, 1.0]) * np.pi
        )

    axis  = cross / sin_a
    angle = np.arctan2(sin_a, dot)
    return axis * angle


def debug_palm_orientation(q_sample: np.ndarray, approach_vec: np.ndarray,
                           palm_fwd: np.ndarray):
    """打印手掌三个局部轴在世界坐标系的方向，帮助确认朝向。"""
    try:
        rot = R.from_rotvec(q_sample[3:6]).as_matrix()
        lx  = rot @ np.array([1.0, 0.0, 0.0])
        ly  = rot @ np.array([0.0, 1.0, 0.0])
        lz  = rot @ np.array([0.0, 0.0, 1.0])
        cprint(f"   [DebugPalm] local_X→world: {lx.round(3)}", "blue")
        cprint(f"   [DebugPalm] local_Y→world: {ly.round(3)}", "blue")
        cprint(f"   [DebugPalm] local_Z→world: {lz.round(3)}", "blue")
        cprint(f"   [DebugPalm] approach_vec : {approach_vec.round(3)}", "blue")
        fwd_w = rot @ palm_fwd
        ang   = np.degrees(np.arccos(np.clip(np.dot(fwd_w, approach_vec), -1, 1)))
        cprint(f"   [DebugPalm] palm_fwd→world: {fwd_w.round(3)}  夹角={ang:.1f}°", "blue")
    except Exception as e:
        cprint(f"   [DebugPalm] 失败: {e}", "red")


# ══════════════════════════════════════════════════════════════
#  点云加载
# ══════════════════════════════════════════════════════════════

def load_side_pc(pcd_path, num_points=NUM_POINTS,
                 camera_axis='y', camera_dir='neg'):
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    if len(pts) == 0:
        raise ValueError(f"点云为空：{pcd_path}")

    cprint(f"[PC] 原始点数: {len(pts)}", "cyan")
    center = pts.mean(axis=0)
    pts    = pts - center

    cprint(f"[PC] 中心化后"
           f"  X:[{pts[:,0].min():.3f},{pts[:,0].max():.3f}]"
           f"  Y:[{pts[:,1].min():.3f},{pts[:,1].max():.3f}]"
           f"  Z:[{pts[:,2].min():.3f},{pts[:,2].max():.3f}]", "cyan")

    axis_idx, front_sign, approach_vec = get_camera_axis_info(camera_axis, camera_dir)
    axis_name = ['X', 'Y', 'Z'][axis_idx]

    front_val = float(pts[:, axis_idx].min() if front_sign == -1
                      else pts[:, axis_idx].max())
    back_val  = float(pts[:, axis_idx].max() if front_sign == -1
                      else pts[:, axis_idx].min())

    cprint(f"[PC] 相机轴:{axis_name}({camera_dir})  "
           f"正面={front_val:.4f}m  背面={back_val:.4f}m", "cyan")

    n   = len(pts)
    idx = np.random.choice(n, num_points, replace=(n < num_points))
    return torch.from_numpy(pts[idx]), front_val, axis_idx, front_sign, approach_vec


# ══════════════════════════════════════════════════════════════
#  侧面先验初始位姿
# ══════════════════════════════════════════════════════════════

def make_side_grasp_q(hand, device, obj_pc, front_val,
                      axis_idx, front_sign, approach_vec,
                      batch_size, hover_dist=HOVER_DIST,
                      palm_fwd_vec=None, debug_palm=False):
    """
    构造从正面接近的初始抓取位姿 batch。

    手掌位置：
      palm_on_axis = front_val + front_sign * hover_dist
        front_sign=-1, hover_dist=+0.10 → palm = front_val - 0.10（在正面更外侧）✓
        front_sign=+1, hover_dist=+0.10 → palm = front_val + 0.10（在正面更外侧）✓

    手掌朝向：
      将 palm_fwd_vec（手掌"朝前"局部轴）对齐到 approach_vec（朝向物体）

    手掌高度（Z轴）：
      限制在物体中段 [z_20%, z_80%]，避免夹杯口边缘
    """
    if palm_fwd_vec is None:
        palm_fwd_vec = np.array([0.0, 0.0, 1.0])

    iq_list, rpc_list, opc_list = [], [], []
    num_points = obj_pc.shape[0]
    other_axes = [i for i in range(3) if i != axis_idx]

    # 物体在另两轴的统计
    center_oa = obj_pc[:, other_axes].mean(dim=0).numpy()
    extent_oa = (obj_pc[:, other_axes].max(dim=0).values
               - obj_pc[:, other_axes].min(dim=0).values).numpy()

    # ── 关键修正：手掌在接近轴上的坐标 ──────────────────────
    # front_sign=-1 → 正面是 front_val（负数）→ 手掌更负 → palm = front_val + (-1)*hover_dist
    # front_sign=+1 → 正面是 front_val（正数）→ 手掌更正 → palm = front_val + (+1)*hover_dist
    palm_on_axis = front_val + front_sign * hover_dist
    cprint(f"   [make_q] front_val={front_val:.4f}  front_sign={front_sign}"
           f"  hover={hover_dist}  palm_axis={palm_on_axis:.4f}", "magenta")

    # 物体 Z 范围中段（避免夹顶部边缘）
    z_vals  = obj_pc[:, 2].numpy()   # Z 轴（朝上）
    z_lo    = float(np.percentile(z_vals, 20))
    z_hi    = float(np.percentile(z_vals, 80))
    z_mid   = (z_lo + z_hi) / 2
    cprint(f"   [make_q] Z范围限制 [{z_lo:.3f}, {z_hi:.3f}]m（中段）", "magenta")

    # 基础旋转：将手掌局部"朝前"轴对齐到 approach_vec
    base_rotvec = _align_rotvec(palm_fwd_vec, approach_vec)

    for i in range(batch_size):
        q_new = hand.get_initial_q().clone()

        # ── 手掌平移 ──────────────────────────────────────
        pos = np.zeros(3)
        pos[axis_idx]      = palm_on_axis
        # 水平轴（另一个非Z轴）在物体中心附近随机
        horiz_ax = other_axes[0] if other_axes[0] != 2 else other_axes[1]
        pos[horiz_ax]      = float(center_oa[other_axes.index(horiz_ax)]) \
                             + (np.random.rand() - 0.5) * float(extent_oa[other_axes.index(horiz_ax)]) * 0.5
        # Z 轴：限制在物体中段
        pos[2]             = z_mid + (np.random.rand() - 0.5) * (z_hi - z_lo) * 0.6

        q_new[0] = float(pos[0])
        q_new[1] = float(pos[1])
        q_new[2] = float(pos[2])

        # ── 手掌旋转 ──────────────────────────────────────
        r_base  = R.from_rotvec(base_rotvec)
        # 小扰动 ±8°，接近轴方向不扰动
        perturb = np.random.uniform(-np.pi / 22, np.pi / 22, size=3)
        perturb[axis_idx] = 0.0
        r_final = R.from_rotvec(perturb) * r_base
        rv      = r_final.as_rotvec()
        q_new[3] = float(rv[0])
        q_new[4] = float(rv[1])
        q_new[5] = float(rv[2])

        # ── 手指轻微预弯 ──────────────────────────────────
        q_new[6:] = q_new[6:] * 0.75

        # 打印第一个样本的手掌朝向（调试用）
        if debug_palm and i == 0:
            debug_palm_orientation(q_new.numpy(), approach_vec, palm_fwd_vec)

        # ── robot_pc ──────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════
#  侧面过滤
# ══════════════════════════════════════════════════════════════

def filter_side_grasp(predict_q_batch, front_val, axis_idx, front_sign,
                      approach_vec, palm_fwd_vec,
                      angle_thresh_deg=SIDE_GRASP_ANGLE_DEG):
    """
    过滤条件：
    1. 手掌未穿入物体（在正面外侧 1cm 容差内）
    2. 手掌"朝前"轴（经旋转后）与 approach_vec 夹角 < angle_thresh_deg
    """
    q_np       = predict_q_batch.detach().cpu().numpy()
    valid_mask = []
    margin     = 0.01

    for q in q_np:
        palm_ax = q[axis_idx]
        pos_ok  = (palm_ax < front_val + margin) if front_sign == -1 \
                  else (palm_ax > front_val - margin)

        if not pos_ok:
            valid_mask.append(False)
            continue

        try:
            rot_mat  = R.from_rotvec(q[3:6]).as_matrix()
            fwd_w    = rot_mat @ palm_fwd_vec
            cos_a    = np.clip(np.dot(fwd_w, approach_vec), -1.0, 1.0)
            angle    = np.degrees(np.arccos(cos_a))
            valid_mask.append(angle < angle_thresh_deg)
        except Exception:
            valid_mask.append(False)

    valid_mask  = np.array(valid_mask)
    valid_count = int(valid_mask.sum())
    total       = len(valid_mask)

    if valid_count == 0:
        cprint(f"   [Filter] 全部过滤（{total}个），保留原始 batch", "yellow")
        return predict_q_batch, np.arange(total)

    cprint(f"   [Filter] 侧面过滤: {valid_count}/{total} 通过", "green")
    indices = np.where(valid_mask)[0]
    return predict_q_batch[torch.from_numpy(indices)], indices


# ══════════════════════════════════════════════════════════════
#  推理 & 精调
# ══════════════════════════════════════════════════════════════

def infer_batch(network, hand, iq_batch, rpc_batch, opc_batch, batch_size, device):
    pq_list, t_list = [], []
    for i in tqdm(range(batch_size), desc="推理", leave=False):
        iq  = iq_batch[i:i+1]
        rpc = rpc_batch[i:i+1]
        opc = opc_batch[i:i+1]
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
    parser = argparse.ArgumentParser(description="侧面抓取验证 v6（半侧点云）")

    parser.add_argument("--pcd_path",         type=str,   required=True)
    parser.add_argument("--ckpt_name",         type=str,   default="model_shadowhand")
    parser.add_argument("--batch_size",        type=int,   default=20)
    parser.add_argument("--num_rounds",        type=int,   default=5)
    parser.add_argument("--gpu",               type=int,   default=0)
    parser.add_argument("--num_points",        type=int,   default=NUM_POINTS)

    # 相机/接近方向
    parser.add_argument("--camera_axis",       type=str,   default="y",
                        choices=["x", "y", "z"])
    parser.add_argument("--camera_dir",        type=str,   default="neg",
                        choices=["neg", "pos"])
    parser.add_argument("--hover_dist",        type=float, default=HOVER_DIST)
    parser.add_argument("--side_angle_deg",    type=float, default=SIDE_GRASP_ANGLE_DEG)

    # ★ 新增：手掌朝前轴（需根据 wujihand URDF 确认）
    parser.add_argument("--palm_forward_axis", type=str,   default="z",
                        choices=["x", "-x", "y", "-y", "z", "-z"],
                        help="手掌局部坐标系中'朝前'（指向物体）的轴，"
                             "需与 wujihand URDF 手掌 link 的朝向一致（默认 z）")

    # 控制开关
    parser.add_argument("--no_side_filter",   action="store_true")
    parser.add_argument("--use_side_prior",   action="store_true")
    parser.add_argument("--debug_palm_axis",  action="store_true",
                        help="打印手掌局部轴的世界方向（用于确认 palm_forward_axis）")

    args   = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    palm_fwd_vec = PALM_AXIS_MAP[args.palm_forward_axis]
    axis_name    = ['X', 'Y', 'Z'][{'x':0,'y':1,'z':2}[args.camera_axis.lower()]]

    # ── 1. 点云 ───────────────────────────────────────────────
    cprint(f"\n=> 加载点云: {args.pcd_path}", "cyan")
    obj_pc, front_val, axis_idx, front_sign, approach_vec = load_side_pc(
        args.pcd_path, args.num_points, args.camera_axis, args.camera_dir)
    debug_pc = obj_pc.to(device)

    palm_target = front_val + front_sign * args.hover_dist
    cprint(f"   物体正面 {axis_name}={front_val:.4f}m", "green")
    cprint(f"   手掌目标 {axis_name}={palm_target:.4f}m  (hover={args.hover_dist}m)", "green")
    cprint(f"   手掌接近方向: {approach_vec}", "green")
    cprint(f"   手掌局部朝前轴: {args.palm_forward_axis} → {palm_fwd_vec}", "green")

    # ── 2. 网络 ───────────────────────────────────────────────
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

    # ── 3. 手部 ───────────────────────────────────────────────
    cprint(f"\n=> 加载手部: {ROBOT_NAME}", "cyan")
    hand = create_hand_model(ROBOT_NAME, device)

    # ── 4. 主循环 ─────────────────────────────────────────────
    success_num, total_num, all_time = 0, 0, []
    cprint(f"\n=> 验证: {args.num_rounds}轮×{args.batch_size}  "
           f"模式:{'先验' if args.use_side_prior else '随机'}  "
           f"过滤:{'关' if args.no_side_filter else f'开({args.side_angle_deg}°)'}  "
           f"palm_fwd:{args.palm_forward_axis}\n", "cyan")

    for rnd in range(args.num_rounds):
        cprint(f"── Round {rnd+1}/{args.num_rounds} ──", "yellow")

        # ── 构造 batch ─────────────────────────────────────
        if args.use_side_prior:
            iq_b, rpc_b, opc_b = make_side_grasp_q(
                hand, device, obj_pc, front_val,
                axis_idx, front_sign, approach_vec,
                args.batch_size, args.hover_dist,
                palm_fwd_vec=palm_fwd_vec,
                debug_palm=args.debug_palm_axis and rnd == 0)
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

        # ── 推理 + IK ──────────────────────────────────────
        pq_batch, tlist = infer_batch(
            network, hand, iq_b, rpc_b, opc_b, args.batch_size, device)
        all_time.extend(tlist)

        # ── 侧面过滤 ───────────────────────────────────────
        if not args.no_side_filter:
            pq_filtered, valid_idx = filter_side_grasp(
                pq_batch, front_val, axis_idx, front_sign,
                approach_vec, palm_fwd_vec, args.side_angle_deg)
            q_np = pq_batch.detach().cpu().numpy()
            for ii in range(q_np.shape[0]):
                try:
                    rot  = R.from_rotvec(q_np[ii, 3:6]).as_matrix()
                    fwd_w = rot @ palm_fwd_vec
                    ang   = np.degrees(np.arccos(
                                np.clip(np.dot(fwd_w, approach_vec), -1.0, 1.0)))
                    mark  = "✓" if ii in valid_idx else "✗"
                    cprint(f"   样本{ii:2d}: {axis_name}={q_np[ii,axis_idx]:.3f}m"
                           f"  朝向角={ang:.1f}°  {mark}", "white")
                except Exception:
                    pass
        else:
            pq_filtered = pq_batch
            valid_idx   = np.arange(args.batch_size)

        if len(pq_filtered) == 0:
            cprint("   所有姿态被过滤，跳过", "red")
            total_num += args.batch_size
            continue

        saved_rpc0 = rpc_b[0:1].clone()

        # ── IsaacGym 首次验证 ──────────────────────────────
        cprint(f"   → IsaacGym 首次验证（{len(pq_filtered)}个）...", "cyan")
        success, isaac_ret = validate_isaac(
            ROBOT_NAME, OBJECT_NAME, pq_filtered,
            gpu=args.gpu, debug_pc=debug_pc)

        # ── TRIGGERED ─────────────────────────────────────
        if isinstance(success, str) and success == "TRIGGERED":
            info       = isaac_ret
            step       = info.get("step", -1)
            init_force = info.get("init_force", 0.0)
            cprint(f"==> 触觉触发(Step:{step}, init_force:{init_force:.1f}N)，检查...",
                   "magenta", attrs=["bold"])

            # 检查初始穿透
            if init_force > MAX_INIT_FORCE:
                cprint(f"   初始力过大({init_force:.1f}N)，穿透跳过", "red")
                total_num += len(pq_filtered)
                continue

            # 解析触碰点
            tp = info["tip_pos"]
            if isinstance(tp, dict):
                tv = torch.tensor([tp["x"], tp["y"], tp["z"]], device=device)
            elif hasattr(tp, "x"):
                tv = torch.tensor([float(tp.x), float(tp.y), float(tp.z)], device=device)
            else:
                tv = torch.as_tensor(tp, dtype=torch.float32, device=device)
                if tv.shape != (3,):
                    cprint("   tip_pos 格式异常，跳过", "red")
                    total_num += len(pq_filtered); continue

            # 检查触碰点合法性
            dist_to_obj = torch.norm(debug_pc - tv, dim=-1).min().item()
            if dist_to_obj > MAX_TIP_DIST:
                cprint(f"   触碰点离物体太远({dist_to_obj:.3f}m)，跳过", "red")
                total_num += len(pq_filtered); continue

            # 精调前旋转向量 norm 检查
            rv_norm_before = float(torch.norm(pq_filtered[0, 3:6]).item())
            if rv_norm_before > np.pi * 1.5:
                cprint(f"   精调前旋转异常(norm={rv_norm_before:.2f})，跳过", "red")
                total_num += 1; continue

            # 构建局部点云
            dist  = torch.norm(debug_pc - tv, dim=-1)
            _, li = torch.topk(dist, k=min(256, debug_pc.shape[0]), largest=False)
            lpc   = debug_pc[li].unsqueeze(0)
            cprint(f"   局部点云{lpc.shape[1]}点  触碰@{tv.cpu().numpy().round(4)}", "magenta")

            cprint(f"   精调前 Q[:6]: {pq_filtered[0,:6].detach().cpu().numpy().round(4)}", "white")
            refined_q, _ = run_local_refinement(
                network, hand, saved_rpc0, lpc,
                pq_filtered[0:1].clone(), device)
            cprint(f"   精调后 Q[:6]: {refined_q[0,:6].detach().cpu().numpy().round(4)}", "white")

            # 精调后旋转检查
            rv_norm_after = float(torch.norm(refined_q[0, 3:6]).item())
            if rv_norm_after > np.pi * 1.5:
                cprint(f"   精调后旋转爆炸(norm={rv_norm_after:.2f})，跳过", "red")
                total_num += 1
                cprint(f"   SR: {success_num}/{total_num} ({success_num/total_num*100:.1f}%)", "yellow")
                continue

            cprint("   → IsaacGym 二次验证...", "cyan")
            s2, _ = validate_isaac(
                ROBOT_NAME, OBJECT_NAME, refined_q,
                gpu=args.gpu, debug_pc=debug_pc, is_refinement=True)

            if isinstance(s2, torch.Tensor) and s2[0].item():
                cprint("   精调: ✓ 成功", "green", attrs=["bold"])
                success_num += 1
            else:
                cprint("   精调: ✗ 失败", "red")

            total_num += 1
            cprint(f"   SR: {success_num}/{total_num} ({success_num/total_num*100:.1f}%)", "yellow")
            continue

        # ── 正常统计 ───────────────────────────────────────
        succ_num = int(success.sum().item()) if isinstance(success, torch.Tensor) else 0
        cprint(f"   Round {rnd+1} 成功: {succ_num}/{len(pq_filtered)}",
               "green" if succ_num > 0 else "red")
        success_num += succ_num
        total_num   += len(pq_filtered)

    # ── 汇总 ──────────────────────────────────────────────────
    cprint("\n" + "=" * 55, "yellow")
    if total_num > 0:
        cprint(f"[Final] 成功率: {success_num}/{total_num} "
               f"({success_num/total_num*100:.1f}%)", "yellow", attrs=["bold"])
        if all_time:
            cprint(f"[Final] 平均IK时间: {np.mean(all_time)*1000:.1f}ms", "yellow")
    else:
        cprint("[Final] 未评估任何样本", "red")


if __name__ == "__main__":
    warnings.simplefilter(action="ignore", category=FutureWarning)
    torch.set_num_threads(8)
    main()
