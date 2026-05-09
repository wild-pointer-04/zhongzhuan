"""
validate_duck_v8.py  —— 侧面抓取（v8 关键修复）
================================================================
相比 v7 的修复：

  修复A：彻底解决旋转问题
    不再用 align_rotvec(palm_fwd, approach)，因为：
    1. palm_fwd=y 时两向量重合，旋转为零
    2. initial_q 本身就是歪的，局部轴没有天然朝 +Y 的
    改为：直接硬编码"手掌朝 +Y"的目标旋转矩阵，
    即让机械手从正前方（Y 负侧）水平接近物体：
      手掌朝 +Y: rotation_matrix = Rx(0) 且手心面朝 +Y
    具体做法：先从 URDF 的 initial_q 算出手掌的默认旋转，
    再计算"让手掌朝 +Y 且手指朝 -Z（向下）"所需的修正旋转。

  修复B：彻底解决 tip_pos 坐标系问题
    Isaac 世界坐标里物体中心由 [Retract] object_center 给出，
    但我们在 validate_utils 里拿不到这个值。
    改为：完全跳过 tip_pos 距离检测，只检查 init_force。
    触发后直接用全局点云做精调，不做局部采样。
    （精调质量会略降，但不会因坐标系错误而全部跳过）

  修复C：过滤器改用实际接近轴位置判断
    Y 坐标必须 < front_val（确保手在物体正面侧）

坐标系：camera_axis=y, camera_dir=neg
  物体正面 Y 最小值（约 -0.058m）
  手从 Y 负方向接近（approach_vec=[0,+1,0]）
  手掌目标位置 Y = front_val - (hover_dist + finger_len)

用法：
    python scripts/validate_duck_v8.py \
        --pcd_path /home/eureka/duck_point/duck_v6.pcd \
        --ckpt_name model_shadowhand \
        --batch_size 20 \
        --num_rounds 5 \
        --camera_axis y \
        --camera_dir neg \
        --hover_dist 0.05 \
        --finger_len 0.09 \
        --finger_bend 0.55 \
        --joint_target_scale 0.3 \
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
HOVER_DIST           = 0.05
FINGER_LEN           = 0.09
FINGER_BEND          = 0.55
JOINT_TARGET_SCALE   = 0.3
SIDE_GRASP_ANGLE_DEG = 45.0
MAX_INIT_FORCE       = 80.0
# ─────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════

def get_camera_axis_info(camera_axis, camera_dir):
    idx = {'x': 0, 'y': 1, 'z': 2}[camera_axis.lower()]
    front_sign   = -1 if camera_dir == 'neg' else +1
    approach_vec = np.eye(3)[idx] if camera_dir == 'neg' else -np.eye(3)[idx]
    return idx, front_sign, approach_vec.astype(float)


def compute_palm_approach_rotvec(hand, axis_idx, front_sign):
    """
    计算让手掌朝向接近方向的旋转向量。

    策略：在 initial_q 的基础上，找"手掌基座（link0）坐标系中，
    哪个局部轴与 approach_vec 最对齐"，然后旋转使该轴对齐 approach_vec。

    对 wujihand 的具体做法：
    initial_q[3:6] 是旋转向量。我们直接覆盖为让手掌朝 approach_vec 的旋转。
    即：构造旋转矩阵 R 使得 R @ palm_local_fwd = approach_vec。

    由于不知道 palm_local_fwd，我们用暴力搜索：
    对 6 个候选局部轴，分别计算对齐旋转，取旋转角最小的（最接近 initial_q）。
    """
    approach_vec = np.zeros(3)
    approach_vec[axis_idx] = 1.0 if front_sign == -1 else -1.0

    # 获取 initial_q 的旋转矩阵
    iq = hand.get_initial_q().cpu().numpy()
    rv_init = iq[3:6]
    R_init  = R.from_rotvec(rv_init).as_matrix()

    # 6个候选局部轴
    candidates = {
        'x':  np.array([1., 0., 0.]),
        '-x': np.array([-1., 0., 0.]),
        'y':  np.array([0., 1., 0.]),
        '-y': np.array([0., -1., 0.]),
        'z':  np.array([0., 0., 1.]),
        '-z': np.array([0., 0., -1.]),
    }

    best_axis = None
    best_angle = float('inf')
    best_rv = None

    cprint("\n   [PalmAxis] 搜索最优手掌局部轴：", "cyan")
    for name, local_vec in candidates.items():
        world_vec = R_init @ local_vec  # 初始 q 下该局部轴的世界方向
        dot = np.clip(np.dot(world_vec, approach_vec), -1., 1.)
        angle_deg = np.degrees(np.arccos(dot))

        # 计算对齐旋转
        cross = np.cross(world_vec, approach_vec)
        s = np.linalg.norm(cross)
        d = dot
        if s < 1e-6:
            rv_align = np.zeros(3) if d > 0 else np.array([0., 0., 1.]) * np.pi
        else:
            rv_align = (cross / s) * np.arctan2(s, d)

        # 对齐旋转 + 初始旋转 = 最终旋转
        R_final = R.from_rotvec(rv_align) * R.from_rotvec(rv_init)
        rv_final = R_final.as_rotvec()
        final_angle = np.linalg.norm(rv_final) * 180 / np.pi

        cprint(f"      {name:3s}: world={world_vec.round(3)}  "
               f"与approach夹角={angle_deg:.1f}°  "
               f"对齐后旋转norm={np.linalg.norm(rv_final):.3f}rad", "white")

        if angle_deg < best_angle:
            best_angle = angle_deg
            best_axis  = name
            best_rv    = rv_final

    cprint(f"   [PalmAxis] 最优: {best_axis}  初始夹角={best_angle:.1f}°", "green")
    cprint(f"   [PalmAxis] 目标旋转向量={best_rv.round(4)}", "green")
    return best_rv, best_axis


# ══════════════════════════════════════════════════════════════
#  点云加载
# ══════════════════════════════════════════════════════════════

def load_side_pc(pcd_path, num_points=NUM_POINTS, camera_axis='y', camera_dir='neg'):
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    if len(pts) == 0:
        raise ValueError(f"点云为空：{pcd_path}")
    cprint(f"[PC] 原始点数: {len(pts)}", "cyan")
    pts -= pts.mean(axis=0)
    cprint(f"[PC] X:[{pts[:,0].min():.3f},{pts[:,0].max():.3f}]"
           f"  Y:[{pts[:,1].min():.3f},{pts[:,1].max():.3f}]"
           f"  Z:[{pts[:,2].min():.3f},{pts[:,2].max():.3f}]", "cyan")

    axis_idx, front_sign, approach_vec = get_camera_axis_info(camera_axis, camera_dir)
    front_val = float(pts[:, axis_idx].min() if front_sign == -1
                      else pts[:, axis_idx].max())
    axis_name = 'XYZ'[axis_idx]
    cprint(f"[PC] 相机轴:{axis_name}({camera_dir})  正面={front_val:.4f}m", "cyan")

    idx = np.random.choice(len(pts), num_points, replace=(len(pts) < num_points))
    return torch.from_numpy(pts[idx]), front_val, axis_idx, front_sign, approach_vec


# ══════════════════════════════════════════════════════════════
#  侧面先验初始位姿
# ══════════════════════════════════════════════════════════════

def make_side_grasp_q(hand, device, obj_pc, front_val,
                      axis_idx, front_sign, approach_vec,
                      batch_size, target_rv,
                      hover_dist=HOVER_DIST,
                      finger_len=FINGER_LEN,
                      finger_bend=FINGER_BEND):

    total_dist   = hover_dist + finger_len
    # front_sign=-1: palm 在正面更负（如 Y = -0.058 + (-1)*0.14 = -0.198）
    palm_on_axis = front_val + front_sign * total_dist

    iq_list, rpc_list, opc_list = [], [], []
    num_points = obj_pc.shape[0]
    other_axes = [i for i in range(3) if i != axis_idx]

    center_oa = obj_pc[:, other_axes].mean(dim=0).numpy()
    extent_oa = (obj_pc[:, other_axes].max(dim=0).values
               - obj_pc[:, other_axes].min(dim=0).values).numpy()

    z_vals = obj_pc[:, 2].numpy()
    z_lo   = float(np.percentile(z_vals, 20))
    z_hi   = float(np.percentile(z_vals, 80))
    z_mid  = (z_lo + z_hi) / 2.0

    axis_name = 'XYZ'[axis_idx]
    cprint(f"   [make_q] front_val={front_val:.4f}  front_sign={front_sign}"
           f"  total_dist={total_dist:.3f}  palm_{axis_name}={palm_on_axis:.4f}", "magenta")
    cprint(f"   [make_q] Z范围 [{z_lo:.3f}, {z_hi:.3f}]m  target_rv={target_rv.round(4)}", "magenta")

    for _ in range(batch_size):
        q_new = hand.get_initial_q().clone()

        # ── 位置 ──────────────────────────────────────────────
        pos = np.zeros(3)
        pos[axis_idx] = palm_on_axis

        horiz_axes = [a for a in other_axes if a != 2]
        if horiz_axes:
            ha = horiz_axes[0]
            ha_local = other_axes.index(ha)
            pos[ha] = float(center_oa[ha_local]) \
                    + (np.random.rand() - 0.5) * float(extent_oa[ha_local]) * 0.4

        pos[2] = z_mid + (np.random.rand() - 0.5) * (z_hi - z_lo) * 0.4

        q_new[0] = float(pos[0])
        q_new[1] = float(pos[1])
        q_new[2] = float(pos[2])

        # ── 旋转：使用计算出的目标旋转 + 小扰动 ──────────────
        perturb = np.random.uniform(-np.pi / 24, np.pi / 24, size=3)
        perturb[axis_idx] = 0.0   # 接近轴不扰动
        R_target   = R.from_rotvec(target_rv)
        R_perturb  = R.from_rotvec(perturb)
        R_final    = R_perturb * R_target
        rv_final   = R_final.as_rotvec()

        q_new[3] = float(rv_final[0])
        q_new[4] = float(rv_final[1])
        q_new[5] = float(rv_final[2])

        # ── 手指预弯 ──────────────────────────────────────────
        q_new[6:] = q_new[6:] * (1.0 - finger_bend)

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
    pq_scaled = pq_batch.clone()
    pq_scaled[:, 6:] = pq_scaled[:, 6:] * scale
    return pq_scaled


# ══════════════════════════════════════════════════════════════
#  侧面过滤（只检查位置，不检查旋转——旋转已由 target_rv 保证）
# ══════════════════════════════════════════════════════════════

def filter_side_grasp(pq_batch, front_val, axis_idx, front_sign, approach_vec,
                      target_rv, angle_thresh=SIDE_GRASP_ANGLE_DEG):
    q_np       = pq_batch.detach().cpu().numpy()
    valid_mask = []
    axis_name  = 'XYZ'[axis_idx]

    for q in q_np:
        # 位置检查：手掌必须在物体正面侧（允许 2cm 余量）
        palm_ax = q[axis_idx]
        margin  = 0.02
        pos_ok  = (palm_ax < front_val + margin) if front_sign == -1 \
                  else (palm_ax > front_val - margin)
        if not pos_ok:
            valid_mask.append(False); continue

        # 旋转向量范数检查
        rv_norm = np.linalg.norm(q[3:6])
        if rv_norm > np.pi * 1.5:
            valid_mask.append(False); continue

        # 朝向角检查
        try:
            rot   = R.from_rotvec(q[3:6]).as_matrix()
            # 用 target_rv 对应的局部轴
            local_fwd = R.from_rotvec(target_rv).as_matrix().T @ approach_vec
            fwd_w = rot @ local_fwd
            angle = np.degrees(np.arccos(np.clip(np.dot(fwd_w, approach_vec), -1., 1.)))
            valid_mask.append(angle < angle_thresh)
        except Exception:
            valid_mask.append(False)

    valid_mask  = np.array(valid_mask)
    valid_count = int(valid_mask.sum())
    total       = len(valid_mask)

    if valid_count == 0:
        cprint(f"   [Filter] 全部过滤，保留原始 batch", "yellow")
        return pq_batch, np.arange(total)

    cprint(f"   [Filter] 过滤: {valid_count}/{total} 通过", "green")
    indices = np.where(valid_mask)[0]
    return pq_batch[torch.from_numpy(indices)], indices


# ══════════════════════════════════════════════════════════════
#  推理
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


def run_local_refinement(network, hand, robot_pc, local_pc, initial_q, device):
    with torch.no_grad():
        n = local_pc.shape[1]
        if n < 512:
            idx = torch.randint(0, n, (1, 512), device=device)
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
    parser = argparse.ArgumentParser(description="侧面抓取 v8")

    parser.add_argument("--pcd_path",          type=str,   required=True)
    parser.add_argument("--ckpt_name",          type=str,   default="model_shadowhand")
    parser.add_argument("--batch_size",         type=int,   default=20)
    parser.add_argument("--num_rounds",         type=int,   default=5)
    parser.add_argument("--gpu",                type=int,   default=0)
    parser.add_argument("--num_points",         type=int,   default=NUM_POINTS)
    parser.add_argument("--camera_axis",        type=str,   default="y", choices=["x","y","z"])
    parser.add_argument("--camera_dir",         type=str,   default="neg", choices=["neg","pos"])
    parser.add_argument("--hover_dist",         type=float, default=HOVER_DIST)
    parser.add_argument("--finger_len",         type=float, default=FINGER_LEN)
    parser.add_argument("--finger_bend",        type=float, default=FINGER_BEND)
    parser.add_argument("--joint_target_scale", type=float, default=JOINT_TARGET_SCALE)
    parser.add_argument("--side_angle_deg",     type=float, default=SIDE_GRASP_ANGLE_DEG)
    parser.add_argument("--no_side_filter",     action="store_true")
    parser.add_argument("--use_side_prior",     action="store_true")
    parser.add_argument("--no_network",         action="store_true")

    args   = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    axis_name = 'XYZ'[{'x':0,'y':1,'z':2}[args.camera_axis.lower()]]

    # ── 点云 ──────────────────────────────────────────────────
    cprint(f"\n=> 加载点云: {args.pcd_path}", "cyan")
    obj_pc, front_val, axis_idx, front_sign, approach_vec = load_side_pc(
        args.pcd_path, args.num_points, args.camera_axis, args.camera_dir)
    debug_pc = obj_pc.to(device)

    # ── 手部 ──────────────────────────────────────────────────
    cprint(f"\n=> 加载手部: {ROBOT_NAME}", "cyan")
    hand = create_hand_model(ROBOT_NAME, device)

    # ── 计算目标旋转向量（只算一次，打印供确认）─────────────
    cprint(f"\n=> 计算手掌朝向旋转...", "cyan")
    target_rv, best_axis = compute_palm_approach_rotvec(hand, axis_idx, front_sign)

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
        cprint("\n=> [NoNetwork 模式]", "yellow")

    # ── 参数汇总 ──────────────────────────────────────────────
    total_hover = args.hover_dist + args.finger_len
    cprint(f"\n   正面 {axis_name}={front_val:.4f}m", "green")
    cprint(f"   手掌目标 {axis_name}={front_val + front_sign*total_hover:.4f}m  "
           f"(hover={args.hover_dist}+finger={args.finger_len}={total_hover:.3f}m)", "green")
    cprint(f"   best_palm_local_axis={best_axis}  bend={args.finger_bend}"
           f"  joint_scale={args.joint_target_scale}\n", "green")

    # ── 主循环 ────────────────────────────────────────────────
    success_num, total_num, all_time = 0, 0, []

    for rnd in range(args.num_rounds):
        cprint(f"── Round {rnd+1}/{args.num_rounds} ──", "yellow")

        # 构造 batch
        if args.use_side_prior:
            iq_b, rpc_b, opc_b = make_side_grasp_q(
                hand, device, obj_pc, front_val,
                axis_idx, front_sign, approach_vec,
                args.batch_size, target_rv,
                hover_dist=args.hover_dist,
                finger_len=args.finger_len,
                finger_bend=args.finger_bend)
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

        # 推理
        if args.no_network:
            pq_batch = iq_b.clone()
            tlist    = [0.0] * args.batch_size
        else:
            pq_batch, tlist = infer_batch(
                network, hand, iq_b, rpc_b, opc_b, args.batch_size, device)
        all_time.extend(tlist)

        # 关节缩放
        pq_for_isaac = apply_joint_target_scale(pq_batch, args.joint_target_scale)

        # 过滤
        if not args.no_side_filter:
            pq_filtered, valid_idx = filter_side_grasp(
                pq_for_isaac, front_val, axis_idx, front_sign,
                approach_vec, target_rv, args.side_angle_deg)
            q_np = pq_for_isaac.detach().cpu().numpy()
            for ii in range(q_np.shape[0]):
                mark = "✓" if ii in valid_idx else "✗"
                rv_n = np.linalg.norm(q_np[ii, 3:6])
                cprint(f"   样本{ii:2d}: {axis_name}={q_np[ii,axis_idx]:.3f}m"
                       f"  rv_norm={rv_n:.2f}  {mark}", "white")
        else:
            pq_filtered = pq_for_isaac
            valid_idx   = np.arange(args.batch_size)

        if len(pq_filtered) == 0:
            cprint("   所有姿态被过滤", "red")
            total_num += args.batch_size; continue

        saved_rpc0 = rpc_b[0:1].clone()

        # IsaacGym 验证
        cprint(f"   → IsaacGym 验证（{len(pq_filtered)}个）...", "cyan")
        success, isaac_ret = validate_isaac(
            ROBOT_NAME, OBJECT_NAME, pq_filtered,
            gpu=args.gpu, debug_pc=debug_pc)

        # TRIGGERED 路径
        if isinstance(success, str) and success == "TRIGGERED":
            info       = isaac_ret
            step       = info.get("step", -1)
            init_force = info.get("init_force", 0.0)
            cprint(f"==> 触觉触发(Step:{step}, init_force:{init_force:.1f}N)", "magenta", attrs=["bold"])

            if init_force > MAX_INIT_FORCE:
                cprint(f"   初始力过大({init_force:.1f}N)，穿透跳过", "red")
                total_num += len(pq_filtered); continue

            # ★ 修复B：不做 tip_pos 坐标转换，直接用全局点云精调
            cprint(f"   跳过 tip_pos 距离检测，用全局点云精调", "yellow")

            rv_norm_before = float(torch.norm(pq_filtered[0, 3:6]).item())
            if rv_norm_before > np.pi * 1.5:
                cprint(f"   旋转异常(norm={rv_norm_before:.2f})，跳过", "red")
                total_num += 1; continue

            cprint(f"   精调前 q[:6]={pq_filtered[0,:6].detach().cpu().numpy().round(4)}", "white")

            if args.no_network:
                refined_q = pq_filtered[0:1].clone()
            else:
                # 用全局点云做精调
                global_lpc = debug_pc.unsqueeze(0)
                refined_q, _ = run_local_refinement(
                    network, hand, saved_rpc0, global_lpc,
                    pq_filtered[0:1].clone(), device)
                refined_q = apply_joint_target_scale(refined_q, args.joint_target_scale)
                cprint(f"   精调后 q[:6]={refined_q[0,:6].detach().cpu().numpy().round(4)}", "white")

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

        # 正常批次
        if isinstance(success, torch.Tensor):
            succ_num = int(success.sum().item())
        else:
            succ_num = 0
        cprint(f"   Round {rnd+1} 成功: {succ_num}/{len(pq_filtered)}",
               "green" if succ_num > 0 else "red")
        success_num += succ_num
        total_num   += len(pq_filtered)

    # 汇总
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
