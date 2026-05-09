Copy

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
