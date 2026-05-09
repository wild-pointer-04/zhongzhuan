"""
validate_duck_v5.py  —— 侧面抓取版（半侧点云）
================================================================
核心设计：
  - 相机从 Y 负方向拍摄物体，点云只有正面（Y_min 侧）
  - 机械手从 Y 负方向水平接近，手掌法向量朝 +Y（朝向物体）
  - 过滤器：手掌 Y < front_val + margin，且手掌法向量与 +Y 夹角 < angle_thresh
  - 触觉触发后验证触碰点合法性再精调
  - Phase0 Step0 力 > 100N 视为初始穿透，直接跳过
 
坐标系（中心化后）：
  Y_min = 相机侧（正面），Y_max = 背面
  Z = 朝上，X = 水平
 
用法：
    python scripts/validate_duck_v5.py \
        --pcd_path /home/eureka/duck_point/duck_v6.pcd \
        --ckpt_name model_shadowhand \
        --batch_size 20 \
        --num_rounds 5 \
        --camera_axis y \
        --camera_dir neg \
        --hover_dist 0.08 \
        --side_angle_deg 45 \
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
 
# ── 全局默认值 ────────────────────────────────────────────────
OBJECT_NAME          = "contactdb+cup"
ROBOT_NAME           = "wujihand"
NUM_POINTS           = 512
HOVER_DIST           = 0.08          # 手掌距物体正面距离（m）
SIDE_GRASP_ANGLE_DEG = 45.0          # 手掌朝向容忍角度
MAX_INIT_FORCE       = 100.0         # Step0 力超过此值视为穿透（N）
MAX_TIP_DIST         = 0.15          # 触碰点离物体最大合法距离（m）
# ─────────────────────────────────────────────────────────────
 
 
# ══════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════
 
def get_camera_axis_info(camera_axis: str, camera_dir: str):
    """
    返回轴索引、正面符号、手掌接近方向向量。
 
    camera_dir='neg' → 相机在轴负侧 → 正面 = 轴最小值
                       手掌从更负处接近 → approach_vec = +axis
    camera_dir='pos' → 相机在轴正侧 → 正面 = 轴最大值
                       手掌从更正处接近 → approach_vec = -axis
    """
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map[camera_axis.lower()]
    if camera_dir == 'neg':
        front_sign   = -1
        approach_vec = np.zeros(3); approach_vec[axis_idx] = +1.0
    else:
        front_sign   = +1
        approach_vec = np.zeros(3); approach_vec[axis_idx] = -1.0
    return axis_idx, front_sign, approach_vec
 
 
def _align_rotvec(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
    """将 from_vec 旋转到 to_vec 的旋转向量（axis-angle）。"""
    from_vec = from_vec / (np.linalg.norm(from_vec) + 1e-8)
    to_vec   = to_vec   / (np.linalg.norm(to_vec)   + 1e-8)
    cross    = np.cross(from_vec, to_vec)
    dot      = np.dot(from_vec, to_vec)
    sin_a    = np.linalg.norm(cross)
    cos_a    = np.clip(dot, -1.0, 1.0)
 
    if sin_a < 1e-6:
        if cos_a > 0:
            return np.zeros(3)
        else:
            perp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(from_vec, perp)) > 0.9:
                perp = np.array([0.0, 1.0, 0.0])
            return perp * np.pi
 
    axis  = cross / sin_a
    angle = np.arctan2(sin_a, cos_a)
    return axis * angle
 
 
# ══════════════════════════════════════════════════════════════
#  点云加载
# ══════════════════════════════════════════════════════════════
 
def load_side_pc(pcd_path, num_points=NUM_POINTS,
                 camera_axis='y', camera_dir='neg'):
    """
    加载并中心化点云，返回：
        obj_pc      : Tensor [N, 3]
        front_val   : float，正面轴坐标值
        axis_idx    : int
        front_sign  : ±1
        approach_vec: np.ndarray [3]
    """
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    if len(pts) == 0:
        raise ValueError(f"点云为空：{pcd_path}")
 
    cprint(f"[PC] 原始点数: {len(pts)}", "cyan")
 
    # 中心化
    center = pts.mean(axis=0)
    pts    = pts - center
    cprint(f"[PC] 中心化后 X:[{pts[:,0].min():.3f},{pts[:,0].max():.3f}]"
           f"  Y:[{pts[:,1].min():.3f},{pts[:,1].max():.3f}]"
           f"  Z:[{pts[:,2].min():.3f},{pts[:,2].max():.3f}]", "cyan")
 
    axis_idx, front_sign, approach_vec = get_camera_axis_info(camera_axis, camera_dir)
    axis_name = ['X', 'Y', 'Z'][axis_idx]
 
    if front_sign == -1:
        front_val = float(pts[:, axis_idx].min())
        back_val  = float(pts[:, axis_idx].max())
    else:
        front_val = float(pts[:, axis_idx].max())
        back_val  = float(pts[:, axis_idx].min())
 
    cprint(f"[PC] 相机轴: {axis_name}({camera_dir})  正面={front_val:.4f}m  背面={back_val:.4f}m", "cyan")
 
    n   = len(pts)
    idx = np.random.choice(n, num_points, replace=(n < num_points))
    return torch.from_numpy(pts[idx]), front_val, axis_idx, front_sign, approach_vec
 
 
# ══════════════════════════════════════════════════════════════
#  侧面先验初始位姿生成
# ══════════════════════════════════════════════════════════════
 
def make_side_grasp_q(hand, device, obj_pc, front_val,
                      axis_idx, front_sign, approach_vec,
                      batch_size, hover_dist=HOVER_DIST):
    """
    从侧面（相机拍摄面）构造初始抓取位姿 batch。
 
    手掌位置：
      - 接近轴坐标 = front_val - front_sign * hover_dist（停在正面外侧）
      - 其余两轴在物体包围盒中心附近小幅随机
 
    手掌旋转：
      - 局部 Z [0,0,1] 对齐到 approach_vec（手掌面朝物体）
      - ±10° 小扰动，接近轴方向不扰动
 
    手指：initial_q × 0.8（轻微预弯，避免穿透）
    """
    iq_list, rpc_list, opc_list = [], [], []
    num_points  = obj_pc.shape[0]
    other_axes  = [i for i in range(3) if i != axis_idx]
 
    # 物体在另两轴的中心和范围
    center_oa   = obj_pc[:, other_axes].mean(dim=0).numpy()   # [2]
    extent_oa   = (obj_pc[:, other_axes].max(dim=0).values
                 - obj_pc[:, other_axes].min(dim=0).values).numpy()  # [2]
 
    # 手掌在接近轴上的停放坐标
    # front_sign=-1 → front_val 是负数 → 手掌更负 → palm_ax = front_val - hover_dist
    palm_on_axis = front_val - front_sign * hover_dist
 
    # 基础旋转：将手掌局部 Z 轴对齐到 approach_vec
    local_z     = np.array([0.0, 0.0, 1.0])
    base_rotvec = _align_rotvec(local_z, approach_vec)
 
    for _ in range(batch_size):
        q_new = hand.get_initial_q().clone()
 
        # ── 手掌平移 ──────────────────────────────────────
        pos = np.zeros(3)
        pos[axis_idx]      = palm_on_axis
        # 另两轴在物体中心 ± 30% 包围盒范围内随机
        pos[other_axes[0]] = float(center_oa[0]) + (np.random.rand() - 0.5) * float(extent_oa[0]) * 0.6
        pos[other_axes[1]] = float(center_oa[1]) + (np.random.rand() - 0.5) * float(extent_oa[1]) * 0.6
        q_new[0] = float(pos[0])
        q_new[1] = float(pos[1])
        q_new[2] = float(pos[2])
 
        # ── 手掌旋转 ──────────────────────────────────────
        r_base    = R.from_rotvec(base_rotvec)
        perturb   = np.random.uniform(-np.pi / 18, np.pi / 18, size=3)   # ±10°
        perturb[axis_idx] = 0.0     # 接近方向不扰动，保持面朝物体
        r_final   = R.from_rotvec(perturb) * r_base
        rv        = r_final.as_rotvec()
        q_new[3]  = float(rv[0])
        q_new[4]  = float(rv[1])
        q_new[5]  = float(rv[2])
 
        # ── 手指轻微预弯（减少穿透概率）────────────────────
        q_new[6:] = q_new[6:] * 0.8
 
        # ── 生成 robot_pc ─────────────────────────────────
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
#  侧面抓取过滤
# ══════════════════════════════════════════════════════════════
 
def filter_side_grasp(predict_q_batch, front_val, axis_idx, front_sign,
                      approach_vec, angle_thresh_deg=SIDE_GRASP_ANGLE_DEG):
    """
    过滤不满足侧面抓取约束的姿态。
 
    条件1：手掌未穿入物体（位置在正面外侧，margin=1cm）
    条件2：手掌法向量（局部 Z 轴）与 approach_vec 夹角 < angle_thresh_deg
    """
    q_np       = predict_q_batch.detach().cpu().numpy()
    valid_mask = []
    margin     = 0.01   # 1cm，防止误判（比之前 5cm 严格得多）
 
    for i in range(q_np.shape[0]):
        q       = q_np[i]
        palm_ax = q[axis_idx]
 
        # 条件1：位置合法
        # front_sign=-1 → 正面是轴最小值 → 手掌 palm_ax < front_val + margin
        # front_sign=+1 → 正面是轴最大值 → 手掌 palm_ax > front_val - margin
        if front_sign == -1:
            pos_ok = palm_ax < (front_val + margin)
        else:
            pos_ok = palm_ax > (front_val - margin)
 
        if not pos_ok:
            valid_mask.append(False)
            continue
 
        # 条件2：朝向合法
        try:
            rot_mat   = R.from_rotvec(q[3:6]).as_matrix()
            local_z_w = rot_mat @ np.array([0.0, 0.0, 1.0])
            cos_a     = np.clip(np.dot(local_z_w, approach_vec), -1.0, 1.0)
            angle     = np.degrees(np.arccos(cos_a))
            valid_mask.append(angle < angle_thresh_deg)
        except Exception:
            valid_mask.append(False)
 
    valid_mask  = np.array(valid_mask)
    valid_count = int(valid_mask.sum())
    total       = len(valid_mask)
 
    if valid_count == 0:
        cprint(f"   [Filter] 无姿态通过侧面约束（{total}个全部过滤），保留原始 batch", "yellow")
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
    parser = argparse.ArgumentParser(
        description="侧面抓取验证（半侧点云，相机正面接近）")
 
    # 基础参数
    parser.add_argument("--pcd_path",      type=str,   required=True,
                        help="点云文件路径（.pcd）")
    parser.add_argument("--ckpt_name",     type=str,   default="model_shadowhand",
                        help="模型权重名称（不含路径和后缀）")
    parser.add_argument("--batch_size",    type=int,   default=20)
    parser.add_argument("--num_rounds",    type=int,   default=5)
    parser.add_argument("--gpu",           type=int,   default=0)
    parser.add_argument("--num_points",    type=int,   default=NUM_POINTS)
 
    # 相机/接近方向参数
    parser.add_argument("--camera_axis",   type=str,   default="y",
                        choices=["x", "y", "z"],
                        help="点云中相机拍摄轴（默认 y）")
    parser.add_argument("--camera_dir",    type=str,   default="neg",
                        choices=["neg", "pos"],
                        help="相机在轴的哪侧：neg=负侧，pos=正侧（默认 neg）")
    parser.add_argument("--hover_dist",    type=float, default=HOVER_DIST,
                        help="手掌悬停在正面外的距离，单位 m（默认 0.08）")
    parser.add_argument("--side_angle_deg",type=float, default=SIDE_GRASP_ANGLE_DEG,
                        help="手掌朝向容忍角度（默认 45°）")
 
    # 控制开关
    parser.add_argument("--no_side_filter",action="store_true",
                        help="禁用侧面过滤（对比实验）")
    parser.add_argument("--use_side_prior",action="store_true",
                        help="使用侧面先验初始位姿（推荐开启）")
 
    args   = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
 
    axis_name = ['X', 'Y', 'Z'][{'x':0,'y':1,'z':2}[args.camera_axis.lower()]]
 
    # ── 1. 点云 ───────────────────────────────────────────────
    cprint(f"\n=> 加载点云: {args.pcd_path}", "cyan")
    obj_pc, front_val, axis_idx, front_sign, approach_vec = load_side_pc(
        args.pcd_path, args.num_points, args.camera_axis, args.camera_dir)
    debug_pc = obj_pc.to(device)
 
    cprint(f"   物体正面 {axis_name} = {front_val:.4f}m", "green")
    cprint(f"   手掌接近方向: {approach_vec}", "green")
    palm_target = front_val - front_sign * args.hover_dist
    cprint(f"   手掌目标 {axis_name} = {palm_target:.4f}m  (hover={args.hover_dist}m)", "green")
 
    # ── 2. 网络 ───────────────────────────────────────────────
    cprint(f"\n=> 加载网络: {args.ckpt_name}", "cyan")
    network = create_network(
        SimpleNamespace(emb_dim=512, latent_dim=64, pretrain=None,
                        center_pc=True, block_computing=True),
        mode="validate").to(device)
    ckpt_path = os.path.join(ROOT_DIR, f"ckpt/model/{args.ckpt_name}.pth")
    network.load_state_dict(torch.load(ckpt_path, map_location=device))
    network.eval()
    cprint("   网络加载完成", "green")
 
    # ── 3. 手部模型 ───────────────────────────────────────────
    cprint(f"\n=> 加载手部: {ROBOT_NAME}", "cyan")
    hand = create_hand_model(ROBOT_NAME, device)
 
    # ── 4. 主循环 ─────────────────────────────────────────────
    success_num = 0
    total_num   = 0
    all_time    = []
 
    mode_str   = "侧面先验" if args.use_side_prior else "随机初始"
    filter_str = "无过滤"   if args.no_side_filter  else f"侧面过滤(≤{args.side_angle_deg}°)"
    cprint(f"\n=> 开始验证: {args.num_rounds}轮×{args.batch_size}姿态 | "
           f"模式:{mode_str} | {filter_str}\n", "cyan")
 
    for rnd in range(args.num_rounds):
        cprint(f"── Round {rnd+1}/{args.num_rounds} ──", "yellow")
 
        # ── 构造 batch ─────────────────────────────────────
        if args.use_side_prior:
            iq_b, rpc_b, opc_b = make_side_grasp_q(
                hand, device, obj_pc, front_val,
                axis_idx, front_sign, approach_vec,
                args.batch_size, args.hover_dist)
            cprint(f"   [Prior] 手掌{axis_name}={palm_target:.4f}m", "magenta")
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
 
        # ── 网络推理 + IK ──────────────────────────────────
        pq_batch, tlist = infer_batch(
            network, hand, iq_b, rpc_b, opc_b, args.batch_size, device)
        all_time.extend(tlist)
 
        # ── 侧面过滤 ───────────────────────────────────────
        if not args.no_side_filter:
            pq_filtered, valid_idx = filter_side_grasp(
                pq_batch, front_val, axis_idx, front_sign,
                approach_vec, args.side_angle_deg)
 
            # 打印每个样本详情
            q_np = pq_batch.detach().cpu().numpy()
            for ii in range(q_np.shape[0]):
                try:
                    rot  = R.from_rotvec(q_np[ii, 3:6]).as_matrix()
                    lzw  = rot @ np.array([0.0, 0.0, 1.0])
                    ang  = np.degrees(np.arccos(
                               np.clip(np.dot(lzw, approach_vec), -1.0, 1.0)))
                    mark = "✓" if ii in valid_idx else "✗"
                    cprint(f"   样本{ii:2d}: {axis_name}={q_np[ii,axis_idx]:.3f}m  "
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
 
        # ── IsaacGym 首次验证 ──────────────────────────────
        cprint(f"   → IsaacGym 首次验证（{len(pq_filtered)}个姿态）...", "cyan")
        success, isaac_ret = validate_isaac(
            ROBOT_NAME, OBJECT_NAME, pq_filtered,
            gpu=args.gpu, debug_pc=debug_pc)
 
        # ── TRIGGERED 处理 ─────────────────────────────────
        if isinstance(success, str) and success == "TRIGGERED":
            info = isaac_ret
            step = info.get("step", -1)
            cprint(f"==> 触觉触发(Step:{step})，启动精调...", "magenta", attrs=["bold"])
 
            # 解析触碰点
            tp = info["tip_pos"]
            if isinstance(tp, dict):
                tv = torch.tensor([tp["x"], tp["y"], tp["z"]], device=device)
            elif hasattr(tp, "x"):
                tv = torch.tensor([float(tp.x), float(tp.y), float(tp.z)], device=device)
            else:
                tv = torch.as_tensor(tp, dtype=torch.float32, device=device)
                if tv.shape != (3,):
                    cprint("   WARN: tip_pos 格式异常，跳过", "red")
                    total_num += len(pq_filtered)
                    continue
 
            # ★ 检查1：触碰点是否在物体附近
            dist_to_obj = torch.norm(debug_pc - tv, dim=-1).min().item()
            if dist_to_obj > MAX_TIP_DIST:
                cprint(f"   触碰点离物体太远 ({dist_to_obj:.3f}m > {MAX_TIP_DIST}m)，跳过精调", "red")
                total_num += len(pq_filtered)
                continue
 
            # ★ 检查2：Phase0 Step0 力是否合理（防穿透）
            init_force = info.get("init_force", 0.0)
            if init_force > MAX_INIT_FORCE:
                cprint(f"   初始力过大 ({init_force:.1f}N > {MAX_INIT_FORCE}N)，初始穿透，跳过精调", "red")
                total_num += len(pq_filtered)
                continue
 
            # 构建局部点云
            dist    = torch.norm(debug_pc - tv, dim=-1)
            _, li   = torch.topk(dist, k=min(256, debug_pc.shape[0]), largest=False)
            lpc     = debug_pc[li].unsqueeze(0)
            cprint(f"   局部点云 {lpc.shape[1]} 点，触碰@{tv.cpu().numpy().round(4)}", "magenta")
 
            cprint(f"   精调前 Q[:6]: {pq_filtered[0,:6].detach().cpu().numpy().round(4)}", "white")
            refined_q, _ = run_local_refinement(
                network, hand, saved_rpc0, lpc,
                pq_filtered[0:1].clone(), device)
            cprint(f"   精调后 Q[:6]: {refined_q[0,:6].detach().cpu().numpy().round(4)}", "white")
 
            # 精调后旋转向量 norm 超过 π*1.5 说明旋转爆炸，跳过
            rv_norm = float(torch.norm(refined_q[0, 3:6]).item())
            if rv_norm > np.pi * 1.5:
                cprint(f"   精调旋转爆炸 (norm={rv_norm:.2f})，跳过", "red")
                total_num += 1
                cprint(f"   SR: {success_num}/{total_num} "
                       f"({success_num/total_num*100:.1f}%)", "yellow")
                continue
 
            cprint("   → IsaacGym 二次验证（关闭触觉中断）...", "cyan")
            s2, _ = validate_isaac(
                ROBOT_NAME, OBJECT_NAME, refined_q,
                gpu=args.gpu, debug_pc=debug_pc,
                is_refinement=True)
 
            if isinstance(s2, torch.Tensor) and s2[0].item():
                cprint("   精调: ✓ 成功", "green", attrs=["bold"])
                success_num += 1
            else:
                cprint("   精调: ✗ 失败", "red")
 
            total_num += 1
            cprint(f"   SR: {success_num}/{total_num} "
                   f"({success_num/total_num*100:.1f}%)", "yellow")
            continue
 
        # ── 正常批次统计 ───────────────────────────────────
        succ_num = int(success.sum().item()) if isinstance(success, torch.Tensor) else 0
        cprint(f"   Round {rnd+1} 成功: {succ_num}/{len(pq_filtered)}",
               "green" if succ_num > 0 else "red")
        success_num += succ_num
        total_num   += len(pq_filtered)
 
    # ── 最终汇总 ──────────────────────────────────────────────
    cprint("\n" + "=" * 55, "yellow")
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
