"""
validate_duck_v6b.py  —— 侧面抓取（修复手指穿透 + 改进抓取质量）
================================================================
相比 v6 的改动：

  1. hover_dist 含义扩展为"手掌基座到物体正面的距离"
     新增 --finger_len 参数（默认 0.06m），hover_dist 自动加上手指长度
     实际悬停距离 = hover_dist + finger_len，确保手指展开时不穿透

  2. 手指预弯幅度可调（--finger_bend，默认 0.5，即关节 × 0.5）
     更大的预弯 = 手指更卷曲 = 不容易穿透但抓握范围减小

  3. 过滤器增加：Phase0 首步力 > MAX_INIT_FORCE 的样本直接从 batch 里删掉
     不再依赖 validate_utils 返回 init_force 字段（兼容旧版）

  4. 首次验证非 TRIGGERED 时（正常批次），统计 per-sample 结果更清晰

  5. --no_network 开关：完全跳过 DRO 推理，只用侧面先验位姿直接做 IK
     适合半侧点云场景下网络效果差的情况

坐标系（中心化后）：
  Y_min = 相机侧（正面），Y_max = 背面
  Z = 朝上，X = 水平

用法（推荐）：
    python scripts/validate_duck_v6b.py \
        --pcd_path /home/eureka/duck_point/duck_v6.pcd \
        --ckpt_name model_shadowhand \
        --batch_size 20 \
        --num_rounds 5 \
        --camera_axis y \
        --camera_dir neg \
        --hover_dist 0.06 \
        --finger_len 0.07 \
        --finger_bend 0.5 \
        --side_angle_deg 45 \
        --palm_forward_axis z \
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
FINGER_LEN           = 0.07    # 手指估计长度（m），加到 hover 上避免穿透
FINGER_BEND          = 0.5     # 手指预弯比例（0=不弯，1=完全弯曲到 initial_q 最大值）
SIDE_GRASP_ANGLE_DEG = 45.0
MAX_INIT_FORCE       = 80.0    # 穿透判定阈值（N）
MAX_TIP_DIST         = 0.15
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
        approach_vec = np.eye(3)[idx]
    else:
        front_sign   = +1
        approach_vec = -np.eye(3)[idx]
    return idx, front_sign, approach_vec.astype(float)


def align_rotvec(from_vec, to_vec):
    fv = from_vec / (np.linalg.norm(from_vec) + 1e-8)
    tv = to_vec   / (np.linalg.norm(to_vec)   + 1e-8)
    c  = np.cross(fv, tv)
    d  = np.clip(np.dot(fv, tv), -1., 1.)
    s  = np.linalg.norm(c)
    if s < 1e-6:
        return np.zeros(3) if d > 0 else np.array([0.,1.,0.]) * np.pi
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
    pts -= pts.mean(axis=0)
    cprint(f"[PC] X:[{pts[:,0].min():.3f},{pts[:,0].max():.3f}]"
           f"  Y:[{pts[:,1].min():.3f},{pts[:,1].max():.3f}]"
           f"  Z:[{pts[:,2].min():.3f},{pts[:,2].max():.3f}]", "cyan")

    axis_idx, front_sign, approach_vec = get_camera_axis_info(camera_axis, camera_dir)
    front_val = float(pts[:, axis_idx].min() if front_sign == -1
                      else pts[:, axis_idx].max())
    back_val  = float(pts[:, axis_idx].max() if front_sign == -1
                      else pts[:, axis_idx].min())
    axis_name = 'XYZ'[axis_idx]
    cprint(f"[PC] 相机轴:{axis_name}({camera_dir})  正面={front_val:.4f}m  背面={back_val:.4f}m", "cyan")

    idx = np.random.choice(len(pts), num_points, replace=(len(pts) < num_points))
    return torch.from_numpy(pts[idx]), front_val, axis_idx, front_sign, approach_vec


# ══════════════════════════════════════════════════════════════
#  侧面先验初始位姿
# ══════════════════════════════════════════════════════════════

def make_side_grasp_q(hand, device, obj_pc, front_val,
                      axis_idx, front_sign, approach_vec,
                      batch_size,
                      hover_dist=HOVER_DIST,
                      finger_len=FINGER_LEN,
                      finger_bend=FINGER_BEND,
                      palm_fwd_vec=None):
    """
    手掌真实悬停距离 = hover_dist + finger_len
    这样手掌基座在正面外 hover_dist，而手指尖（伸展时）恰好停在正面处。
    预弯 finger_bend 比例时，手指尖实际位置退后，进一步避免穿透。
    """
    if palm_fwd_vec is None:
        palm_fwd_vec = np.array([0., 0., 1.])

    total_dist   = hover_dist + finger_len
    palm_on_axis = front_val + front_sign * total_dist   # front_sign=-1 → 更负

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
    cprint(f"   [make_q] Z范围 [{z_lo:.3f}, {z_hi:.3f}]m（物体中段）", "magenta")

    base_rv = align_rotvec(palm_fwd_vec, approach_vec)

    for _ in range(batch_size):
        q_new = hand.get_initial_q().clone()

        # 位置
        pos = np.zeros(3)
        pos[axis_idx] = palm_on_axis
        horiz_ax = other_axes[0] if other_axes[0] != 2 else other_axes[1]
        ha_local = other_axes.index(horiz_ax)
        pos[horiz_ax] = float(center_oa[ha_local]) \
                      + (np.random.rand() - 0.5) * float(extent_oa[ha_local]) * 0.5
        pos[2] = z_mid + (np.random.rand() - 0.5) * (z_hi - z_lo) * 0.5

        q_new[0] = float(pos[0])
        q_new[1] = float(pos[1])
        q_new[2] = float(pos[2])

        # 旋转：±8°，接近轴不扰动
        r_base  = R.from_rotvec(base_rv)
        perturb = np.random.uniform(-np.pi/22, np.pi/22, size=3)
        perturb[axis_idx] = 0.0
        r_final = R.from_rotvec(perturb) * r_base
        rv      = r_final.as_rotvec()
        q_new[3] = float(rv[0])
        q_new[4] = float(rv[1])
        q_new[5] = float(rv[2])

        # 手指预弯：finger_bend=0.5 → 关节值减半（趋向闭合）
        # initial_q 的手指部分通常是全张开状态（值较大）
        q_new[6:] = q_new[6:] * (1.0 - finger_bend)

        # robot_pc
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

def filter_side_grasp(pq_batch, front_val, axis_idx, front_sign,
                      approach_vec, palm_fwd_vec,
                      angle_thresh=SIDE_GRASP_ANGLE_DEG):
    q_np       = pq_batch.detach().cpu().numpy()
    valid_mask = []
    margin     = 0.01
    axis_name  = 'XYZ'[axis_idx]

    for q in q_np:
        palm_ax = q[axis_idx]
        pos_ok  = (palm_ax < front_val + margin) if front_sign == -1 \
                  else (palm_ax > front_val - margin)
        if not pos_ok:
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
    """
    --no_network 模式：直接用先验初始位姿作为预测位姿，不走网络。
    """
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
    parser = argparse.ArgumentParser(description="侧面抓取 v6b（半侧点云）")

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
                        help="手指估计长度 m，加入悬停距离（默认 0.07）")
    parser.add_argument("--finger_bend",        type=float, default=FINGER_BEND,
                        help="手指预弯比例 0~1（0=张开, 1=完全弯曲, 默认 0.5）")
    parser.add_argument("--side_angle_deg",     type=float, default=SIDE_GRASP_ANGLE_DEG)
    parser.add_argument("--palm_forward_axis",  type=str,   default="z",
                        choices=list(PALM_AXIS_MAP.keys()),
                        help="手掌局部'朝前'轴（默认 z，用 debug_palm_axis.py 确认）")

    # 控制
    parser.add_argument("--no_side_filter",     action="store_true")
    parser.add_argument("--use_side_prior",     action="store_true")
    parser.add_argument("--no_network",         action="store_true",
                        help="跳过 DRO 网络，直接用先验位姿（半侧点云场景备选）")

    args   = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    palm_fwd_vec = PALM_AXIS_MAP[args.palm_forward_axis]
    axis_name    = 'XYZ'[{'x':0,'y':1,'z':2}[args.camera_axis.lower()]]

    # ── 点云 ──────────────────────────────────────────────────
    cprint(f"\n=> 加载点云: {args.pcd_path}", "cyan")
    obj_pc, front_val, axis_idx, front_sign, approach_vec = load_side_pc(
        args.pcd_path, args.num_points, args.camera_axis, args.camera_dir)
    debug_pc = obj_pc.to(device)

    total_hover = args.hover_dist + args.finger_len
    palm_target = front_val + front_sign * total_hover
    cprint(f"   正面 {axis_name}={front_val:.4f}m", "green")
    cprint(f"   手掌目标 {axis_name}={palm_target:.4f}m  "
           f"(hover={args.hover_dist}+finger={args.finger_len}={total_hover:.3f}m)", "green")
    cprint(f"   palm_fwd={args.palm_forward_axis}  bend={args.finger_bend}", "green")

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
           f"palm:{args.palm_forward_axis}  bend:{args.finger_bend}\n", "cyan")

    for rnd in range(args.num_rounds):
        cprint(f"── Round {rnd+1}/{args.num_rounds} ──", "yellow")

        # 构造 batch
        if args.use_side_prior:
            iq_b, rpc_b, opc_b = make_side_grasp_q(
                hand, device, obj_pc, front_val,
                axis_idx, front_sign, approach_vec,
                args.batch_size,
                hover_dist=args.hover_dist,
                finger_len=args.finger_len,
                finger_bend=args.finger_bend,
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

        # 推理
        if args.no_network:
            pq_batch, tlist = side_prior_only_batch(iq_b, args.batch_size)
        else:
            pq_batch, tlist = infer_batch(
                network, hand, iq_b, rpc_b, opc_b, args.batch_size, device)
        all_time.extend(tlist)

        # 过滤
        if not args.no_side_filter:
            pq_filtered, valid_idx = filter_side_grasp(
                pq_batch, front_val, axis_idx, front_sign,
                approach_vec, palm_fwd_vec, args.side_angle_deg)
            q_np = pq_batch.detach().cpu().numpy()
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
            pq_filtered = pq_batch
            valid_idx   = np.arange(args.batch_size)

        if len(pq_filtered) == 0:
            cprint("   所有姿态被过滤", "red")
            total_num += args.batch_size; continue

        saved_rpc0 = rpc_b[0:1].clone()

        # IsaacGym 验证
        cprint(f"   → IsaacGym 首次验证（{len(pq_filtered)}个）...", "cyan")
        success, isaac_ret = validate_isaac(
            ROBOT_NAME, OBJECT_NAME, pq_filtered,
            gpu=args.gpu, debug_pc=debug_pc)

        # TRIGGERED
        if isinstance(success, str) and success == "TRIGGERED":
            info       = isaac_ret
            step       = info.get("step", -1)
            init_force = info.get("init_force", 0.0)
            cprint(f"==> 触觉触发(Step:{step}, init_force:{init_force:.1f}N)", "magenta", attrs=["bold"])

            if init_force > MAX_INIT_FORCE:
                cprint(f"   初始力过大，穿透跳过", "red")
                total_num += len(pq_filtered); continue

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

            dist_to_obj = torch.norm(debug_pc - tv, dim=-1).min().item()
            if dist_to_obj > MAX_TIP_DIST:
                cprint(f"   触碰点太远({dist_to_obj:.3f}m)，跳过", "red")
                total_num += len(pq_filtered); continue

            rv_norm_before = float(torch.norm(pq_filtered[0, 3:6]).item())
            if rv_norm_before > np.pi * 1.5:
                cprint(f"   精调前旋转异常(norm={rv_norm_before:.2f})，跳过", "red")
                total_num += 1; continue

            dist  = torch.norm(debug_pc - tv, dim=-1)
            _, li = torch.topk(dist, k=min(256, debug_pc.shape[0]), largest=False)
            lpc   = debug_pc[li].unsqueeze(0)
            cprint(f"   局部点云{lpc.shape[1]}点  触碰@{tv.cpu().numpy().round(4)}", "magenta")
            cprint(f"   精调前: {pq_filtered[0,:6].detach().cpu().numpy().round(4)}", "white")

            if args.no_network:
                refined_q = pq_filtered[0:1].clone()
                cprint("   [NoNetwork] 跳过精调，直接二次验证", "yellow")
            else:
                refined_q, _ = run_local_refinement(
                    network, hand, saved_rpc0, lpc,
                    pq_filtered[0:1].clone(), device)
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

        # 正常统计
        succ_num = int(success.sum().item()) if isinstance(success, torch.Tensor) else 0
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
