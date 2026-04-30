"""
validate_side_grasp_v2.py
=========================
相比 v1 的三处核心修复：

  修复1 【自动判断正面】
      detect_front_axis(pts) 把点云沿 X/Y/Z 三轴各切两半，
      比较两侧点密度，密集侧 = 相机拍摄面 = 正面。
      不再需要手动传 --camera_axis / --camera_dir。

  修复2 【过滤 bug】
      原代码 fallback 时把 valid_idx 改成 arange(total)，
      导致打印循环里全部显示 ✓，实际过滤完全失效。
      修复：先用原始 valid_mask 打印，再做 fallback 决策。

  修复3 【朝向约束强化】
      make_side_grasp_q() 中基础旋转用 _align_rotvec() 精确计算，
      确保手掌局部 Z 轴严格对齐到 approach_vec（正面法向量）。
      filter_side_grasp() 中 margin 逻辑也同步修正。

使用方法：
    python scripts/validate_side_grasp_v2.py \
        --pcd_path /home/eureka/duck_point/cup.pcd \
        --ckpt_name model_shadowhand \
        --batch_size 10 \
        --num_rounds 3 \
        --use_side_prior          # 推荐：使用正面先验初始位姿
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

HOVER_DIST         = 0.06    # 手掌悬停在正面前方的距离（m）
SIDE_ANGLE_DEG     = 60.0   # 手掌朝向容忍角度（°）


# ══════════════════════════════════════════════════════
#  修复1：自动判断正面
# ══════════════════════════════════════════════════════

def detect_front_axis(pts: np.ndarray):
    """
    通过点密度自动找出相机拍摄面（正面）。

    原理：
      对 X / Y / Z 三个轴，分别把点云切成"前半"和"后半"两组，
      比较两组点数。点更多的那侧说明相机在那侧（遮挡少、采样密）。
      选择两侧密度差异最大的轴作为接近轴。

    返回：
        axis_idx   : int      -- 0=X, 1=Y, 2=Z
        front_sign : +1/-1   -- +1 = 正面在正值侧（相机在正侧）
                                -1 = 正面在负值侧（相机在负侧）
        front_val  : float   -- 正面那一端的极值坐标
        approach_vec: np.ndarray [3] -- 手掌应面向的方向（从手掌指向物体）
    """
    best_axis = 0
    best_ratio = 0.0
    best_sign = -1

    for ax in range(3):
        vals  = pts[:, ax]
        mid   = vals.mean()
        n_neg = (vals < mid).sum()   # 负侧点数
        n_pos = (vals >= mid).sum()  # 正侧点数
        total = len(vals)

        # 密度比：多的那侧 / 总数，差异越大越好
        ratio = abs(n_neg - n_pos) / total
        if ratio > best_ratio:
            best_ratio = ratio
            best_axis  = ax
            # 哪侧点多，哪侧就是正面（相机侧）
            if n_neg > n_pos:
                best_sign = -1  # 正面在负值侧，相机在负侧
            else:
                best_sign = +1  # 正面在正值侧，相机在正侧

    axis_name = ['X', 'Y', 'Z'][best_axis]
    vals      = pts[:, best_axis]

    if best_sign == -1:
        front_val = vals.min()   # 正面 = 最小值
    else:
        front_val = vals.max()   # 正面 = 最大值

    # approach_vec：手掌从正面外侧接近物体，方向是"从手掌→物体"
    # 即正面法向量朝内（指向物体内部）
    approach_vec = np.zeros(3)
    approach_vec[best_axis] = -float(best_sign)   # 指向物体内部

    cprint(f"[AutoDetect] 最优轴: {axis_name}, 正面在{'负' if best_sign==-1 else '正'}值侧, "
           f"密度差: {best_ratio:.3f}", "cyan")
    cprint(f"[AutoDetect] 正面坐标: {axis_name}={front_val:.4f}m", "cyan")
    cprint(f"[AutoDetect] 手掌接近方向 approach_vec={approach_vec}", "cyan")

    return best_axis, best_sign, float(front_val), approach_vec


def load_pc(pcd_path: str, num_points: int = NUM_POINTS):
    """
    加载点云，中心化，自动检测正面。

    注意：如果你的 Orbbec 相机坐标系需要翻转（如 Z 轴朝下），
    在"坐标系对齐"那段取消对应的注释行。
    """
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    if len(pts) == 0:
        raise ValueError(f"点云为空: {pcd_path}")

    cprint(f"[PC] 原始点数: {len(pts)}", "cyan")

    # 1. 中心化
    pts = pts - pts.mean(axis=0)

    # 2. 坐标系对齐（根据实际相机安装调整）
    #    Orbbec 侧拍常见情况：相机 Z 朝下 → Isaac Z 朝上，需翻转
    # pts[:, 2] = -pts[:, 2]   # 取消注释以翻转 Z
    # pts[:, 1] = -pts[:, 1]   # 取消注释以翻转 Y

    cprint(f"[PC] 中心化后 X:[{pts[:,0].min():.3f},{pts[:,0].max():.3f}] "
           f"Y:[{pts[:,1].min():.3f},{pts[:,1].max():.3f}] "
           f"Z:[{pts[:,2].min():.3f},{pts[:,2].max():.3f}]", "cyan")

    # 3. 自动检测正面
    axis_idx, front_sign, front_val, approach_vec = detect_front_axis(pts)

    # 4. 采样
    n   = len(pts)
    idx = np.random.choice(n, num_points, replace=(n < num_points))
    return torch.from_numpy(pts[idx]), front_val, axis_idx, front_sign, approach_vec


# ══════════════════════════════════════════════════════
#  辅助：旋转向量计算
# ══════════════════════════════════════════════════════

def _align_rotvec(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
    """将 from_vec 旋转到 to_vec 的最短弧旋转向量（axis-angle）。"""
    from_vec = from_vec / (np.linalg.norm(from_vec) + 1e-9)
    to_vec   = to_vec   / (np.linalg.norm(to_vec)   + 1e-9)
    cross    = np.cross(from_vec, to_vec)
    dot      = np.dot(from_vec, to_vec)
    sin_a    = np.linalg.norm(cross)

    if sin_a < 1e-6:
        if dot > 0:
            return np.zeros(3)
        perp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(from_vec, perp)) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        return perp * np.pi

    axis  = cross / sin_a
    angle = np.arctan2(sin_a, dot)
    return axis * angle


# ══════════════════════════════════════════════════════
#  构造侧面抓取初始位姿
# ══════════════════════════════════════════════════════

def make_side_grasp_q(hand, device, obj_pc, front_val,
                      axis_idx, front_sign, approach_vec,
                      batch_size, hover_dist=HOVER_DIST):
    """
    构造从正面接近的初始 q batch。

    手掌位置：在正面外侧 hover_dist 处（沿接近轴反方向）
    手掌朝向：局部 Z 轴 = approach_vec（精确对齐 + 小扰动）
    """
    iq_list, rpc_list, opc_list = [], [], []
    num_points  = obj_pc.shape[0]
    other_axes  = [i for i in range(3) if i != axis_idx]
    center_vals = obj_pc[:, other_axes].mean(dim=0).numpy()

    # 手掌悬停坐标（在正面外侧）
    # front_sign=-1 → 正面在负侧 → 手掌在更负的位置 → palm = front_val - hover_dist
    # front_sign=+1 → 正面在正侧 → 手掌在更正的位置 → palm = front_val + hover_dist
    palm_on_axis = front_val + front_sign * hover_dist

    # 基础旋转：手掌局部 Z [0,0,1] → approach_vec
    base_rotvec = _align_rotvec(np.array([0.0, 0.0, 1.0]), approach_vec)

    for _ in range(batch_size):
        q_new = hand.get_initial_q().clone()

        # 手掌平移
        pos = np.zeros(3)
        pos[axis_idx]      = palm_on_axis
        pos[other_axes[0]] = float(center_vals[0]) + (np.random.rand() - 0.5) * 0.04
        pos[other_axes[1]] = float(center_vals[1]) + (np.random.rand() - 0.5) * 0.04
        q_new[0] = float(pos[0])
        q_new[1] = float(pos[1])
        q_new[2] = float(pos[2])

        # 手掌旋转 = 基础旋转 + 小扰动（±20°）
        perturb              = np.random.uniform(-np.pi / 9, np.pi / 9, size=3)
        perturb[axis_idx]   *= 0.3   # 接近轴方向扰动更小，保持朝向
        r_final              = R.from_rotvec(perturb) * R.from_rotvec(base_rotvec)
        rv                   = r_final.as_rotvec()
        q_new[3] = float(rv[0])
        q_new[4] = float(rv[1])
        q_new[5] = float(rv[2])

        # 生成 robot_pc
        robot_pc = hand.get_transformed_links_pc(q_new)[:, :3]
        rn = robot_pc.shape[0]
        if rn < num_points:
            pad      = torch.randint(0, rn, (num_points - rn,))
            robot_pc = torch.cat([robot_pc, robot_pc[pad]], dim=0)
        else:
            robot_pc = robot_pc[:num_points]

        iq_list.append(q_new)
        rpc_list.append(robot_pc)
        opc_list.append(obj_pc + torch.randn_like(obj_pc) * 0.002)

    return (
        torch.stack(iq_list).to(device),
        torch.stack(rpc_list).to(device),
        torch.stack(opc_list).to(device),
    )


# ══════════════════════════════════════════════════════
#  修复2 + 修复3：过滤函数（bug 修复 + 强化约束）
# ══════════════════════════════════════════════════════

def filter_side_grasp(predict_q_batch, front_val, axis_idx, front_sign,
                      approach_vec, angle_thresh_deg=SIDE_ANGLE_DEG,
                      hover_dist=HOVER_DIST):
    """
    过滤不是从正面来的姿态。

    判断标准：
      条件1：手掌在接近轴上的位置在物体外侧（未穿入物体）
             front_sign=-1 → hand_pos < front_val + margin
             front_sign=+1 → hand_pos > front_val - margin
      条件2：手掌局部Z轴经旋转后与 approach_vec 夹角 < angle_thresh_deg

    修复：先用原始 valid_mask 打印调试信息，再做 fallback 决策，
         避免 fallback 后打印显示错误的 ✓。
    """
    q_np       = predict_q_batch.detach().cpu().numpy()
    valid_mask = np.zeros(len(q_np), dtype=bool)
    angle_list = []
    pos_list   = []
    margin     = 0.03   # 3cm 容差

    for i, q in enumerate(q_np):
        palm_ax = q[axis_idx]
        pos_list.append(palm_ax)

        # 条件1：位置
        if front_sign == -1:
            pos_ok = palm_ax < (front_val + margin)
        else:
            pos_ok = palm_ax > (front_val - margin)

        # 条件2：朝向
        angle_ok = False
        ang      = 999.0
        if pos_ok:
            try:
                rot   = R.from_rotvec(q[3:6]).as_matrix()
                lzw   = rot @ np.array([0.0, 0.0, 1.0])
                cos_a = np.clip(np.dot(lzw, approach_vec), -1.0, 1.0)
                ang   = np.degrees(np.arccos(cos_a))
                angle_ok = ang < angle_thresh_deg
            except Exception:
                pass

        angle_list.append(ang)
        valid_mask[i] = pos_ok and angle_ok

    # ── 修复2：先打印（用原始 valid_mask），再决定 fallback ──
    axis_name   = ['X', 'Y', 'Z'][axis_idx]
    valid_idx_p = np.where(valid_mask)[0]
    for i in range(len(q_np)):
        mark = "✓" if valid_mask[i] else "✗"
        cprint(f"   样本{i:2d}: {axis_name}={pos_list[i]:.3f}m, "
               f"朝向角={angle_list[i]:.1f}°  {mark}", "white")

    valid_count = valid_mask.sum()
    total       = len(valid_mask)

    if valid_count == 0:
        # fallback：保留原始，但 valid_idx 仍基于原始 mask（全 False）
        # 不再用 arange，避免打印混乱
        cprint(f"   [Filter] 无姿态通过侧面约束（{total}个全部过滤），保留原始 batch", "yellow")
        return predict_q_batch, np.array([], dtype=int), True  # True = fallback

    cprint(f"   [Filter] 侧面抓取过滤: {valid_count}/{total} 个通过", "green")
    indices = np.where(valid_mask)[0]
    return predict_q_batch[torch.from_numpy(indices)], indices, False


# ══════════════════════════════════════════════════════
#  其他函数（保持不变）
# ══════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd_path",      type=str, required=True)
    parser.add_argument("--ckpt_name",     type=str, default="model_shadowhand")
    parser.add_argument("--batch_size",    type=int, default=10)
    parser.add_argument("--num_rounds",    type=int, default=3)
    parser.add_argument("--gpu",           type=int, default=0)
    parser.add_argument("--num_points",    type=int, default=NUM_POINTS)
    parser.add_argument("--hover_dist",    type=float, default=HOVER_DIST)
    parser.add_argument("--angle_deg",     type=float, default=SIDE_ANGLE_DEG)
    parser.add_argument("--no_filter",     action="store_true", help="禁用侧面过滤")
    parser.add_argument("--use_side_prior",action="store_true", help="用正面先验初始位姿")
    args   = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    # 1. 加载点云 + 自动检测正面
    cprint(f"=> 加载点云: {args.pcd_path}", "cyan")
    obj_pc, front_val, axis_idx, front_sign, approach_vec = load_pc(
        args.pcd_path, args.num_points)
    debug_pc  = obj_pc.to(device)
    axis_name = ['X', 'Y', 'Z'][axis_idx]
    cprint(f"   正面 {axis_name}={front_val:.4f}m  approach={approach_vec}", "green")

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

    # 3. 手部
    hand = create_hand_model(ROBOT_NAME, device)

    success_num = 0; total_num = 0; all_time = []
    cprint(f"\n=> 验证: {args.num_rounds}轮×{args.batch_size}姿态 | "
           f"过滤={'关' if args.no_filter else '开'} | "
           f"先验={'正面先验' if args.use_side_prior else '随机'}\n", "cyan")

    for rnd in range(args.num_rounds):
        cprint(f"── Round {rnd+1}/{args.num_rounds} ──", "yellow")

        # 构造 batch
        if args.use_side_prior:
            iq_b, rpc_b, opc_b = make_side_grasp_q(
                hand, device, obj_pc, front_val,
                axis_idx, front_sign, approach_vec,
                args.batch_size, args.hover_dist)
            cprint(f"   正面先验: 手掌{axis_name}="
                   f"{front_val + front_sign * args.hover_dist:.4f}m", "magenta")
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
                iq_list.append(iq); rpc_list.append(rpc); opc_list.append(opc)
            iq_b  = torch.stack(iq_list).to(device)
            rpc_b = torch.stack(rpc_list).to(device)
            opc_b = torch.stack(opc_list).to(device)

        # 推理
        pq_batch, tlist = infer_batch(
            network, hand, iq_b, rpc_b, opc_b, args.batch_size, device)
        all_time.extend(tlist)

        # 过滤（修复2：返回 fallback 标志）
        if not args.no_filter:
            pq_filtered, valid_idx, is_fallback = filter_side_grasp(
                pq_batch, front_val, axis_idx, front_sign,
                approach_vec, args.angle_deg, args.hover_dist)
            if is_fallback:
                # fallback 时仍用全部 batch，但标记清楚
                pq_filtered = pq_batch
        else:
            pq_filtered = pq_batch
            is_fallback = True

        if len(pq_filtered) == 0:
            cprint("   所有姿态被过滤，跳过本轮", "red")
            total_num += args.batch_size
            continue

        saved_rpc0 = rpc_b[0:1].clone()

        # IsaacGym 首次验证
        cprint(f"   → IsaacGym 首次验证（{len(pq_filtered)}个姿态）...", "cyan")
        success, isaac_ret = validate_isaac(
            ROBOT_NAME, OBJECT_NAME, pq_filtered,
            gpu=args.gpu, debug_pc=debug_pc)

        # TRIGGERED 处理
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
                    total_num += len(pq_filtered); continue

            dist = torch.norm(debug_pc - tv, dim=-1)
            _, li = torch.topk(dist, k=min(256, debug_pc.shape[0]), largest=False)
            lpc   = debug_pc[li].unsqueeze(0)
            cprint(f"   局部点云{lpc.shape[1]}点，触碰@{tv.cpu().numpy().round(4)}", "magenta")

            print(f"   精调前 Q[:5]: {pq_filtered[0,:5].detach().cpu().numpy()}")
            refined_q, _ = run_local_refinement(
                network, hand, saved_rpc0, lpc,
                pq_filtered[0:1].clone(), device)
            print(f"   精调后 Q[:5]: {refined_q[0,:5].detach().cpu().numpy()}")

            cprint("   → IsaacGym 二次验证...", "cyan")
            s2, _ = validate_isaac(ROBOT_NAME, OBJECT_NAME, refined_q,
                                   gpu=args.gpu, debug_pc=debug_pc,
                                   is_refinement=True)
            if isinstance(s2, torch.Tensor) and s2[0].item():
                cprint("   精调: ✓ 成功", "green", attrs=["bold"]); success_num += 1
            else:
                cprint("   精调: ✗ 失败", "red")

            total_num += 1
            cprint(f"   SR: {success_num}/{total_num} ({success_num/total_num*100:.1f}%)", "yellow")
            continue

        succ_num = success.sum().item() if isinstance(success, torch.Tensor) else 0
        cprint(f"   Round {rnd+1} 成功: {succ_num}/{len(pq_filtered)}",
               "green" if succ_num > 0 else "red")
        success_num += succ_num
        total_num   += len(pq_filtered)

    # 汇总
    cprint("\n" + "="*50, "yellow")
    if total_num > 0:
        cprint(f"[Final] 成功率: {success_num}/{total_num} ({success_num/total_num*100:.1f}%)",
               "yellow", attrs=["bold"])
        if all_time:
            cprint(f"[Final] 平均IK时间: {np.mean(all_time)*1000:.1f}ms", "yellow")
    else:
        cprint("[Final] 未评估任何样本", "red")


if __name__ == "__main__":
    warnings.simplefilter(action="ignore", category=FutureWarning)
    torch.set_num_threads(8)
    main()
