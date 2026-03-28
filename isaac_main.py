import os
import sys
import json
import argparse
import warnings
import numpy as np
import trimesh
from termcolor import cprint
from types import SimpleNamespace
from scipy.spatial.transform import Rotation

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# IsaacGym 必须在 torch 之前 import
from validation.isaac_validator import IsaacValidator
from utils.hand_model import create_hand_model
from utils.rotation import q_rot6d_to_q_euler

import torch
import open3d as o3d

# 推理相关
from model.network import create_network
from utils.multilateration import multilateration
from utils.se3_transform import compute_link_pose
from utils.optimization import process_transform, create_problem, optimization


# ================================================================
# 工具函数：从 mesh + 当前位姿 生成点云
# ================================================================
def sample_pc_from_mesh_and_pose(object_name, object_pose_7, n_points=1024):
    """
    根据物体当前位姿，从 mesh 采样点云，模拟退后后相机能看到的点云。

    object_name: str, 如 'contactdb+apple'
    object_pose_7: Tensor [7] or [batch, 7]，xyz + quat(xyzw)
    n_points: 采样点数

    返回: Tensor [batch, n_points, 3]（世界坐标系）
    """
    name = object_name.split('+')
    mesh_path = os.path.join(ROOT_DIR,
        f'data/data_urdf/object/{name[0]}/{name[1]}/{name[1]}.stl')
    if not os.path.exists(mesh_path):
        # 尝试 obj 格式
        mesh_path = mesh_path.replace('.stl', '.obj')
    mesh = trimesh.load_mesh(mesh_path)

    # 在 mesh 局部坐标采样
    pc_local, _ = trimesh.sample.sample_surface(mesh, n_points)
    pc_local = torch.tensor(pc_local, dtype=torch.float32)  # [n_points, 3]

    pose = torch.atleast_2d(object_pose_7)  # [batch, 7]
    batch_size = pose.shape[0]
    pc_world_list = []

    for i in range(batch_size):
        xyz = pose[i, :3].numpy()
        quat_xyzw = pose[i, 3:7].numpy()
        rot = Rotation.from_quat(quat_xyzw).as_matrix()  # [3, 3]
        rot_t = torch.tensor(rot, dtype=torch.float32)
        pc_world = (rot_t @ pc_local.T).T + torch.tensor(xyz, dtype=torch.float32)
        pc_world_list.append(pc_world)

    return torch.stack(pc_world_list, dim=0)  # [batch, n_points, 3]


# ================================================================
# 工具函数：用接触点对点云做 ICP 精配准
# ================================================================
def icp_refine_pc(raw_pc, contact_points, contact_force_mag, threshold=0.02):
    """
    用接触点约束，对 raw_pc 做 ICP 精配准。

    raw_pc: Tensor [n_points, 3]，退后后从 mesh 采样的点云（单个 env）
    contact_points: Tensor [5, 3]，5 根指尖的世界坐标
    contact_force_mag: Tensor [5]，各指尖接触力大小（用于筛选有效接触点）
    threshold: ICP 最大对应距离

    返回: Tensor [n_points, 3]，精配准后的点云
    """
    # 只保留有实际接触的指尖（力 > 0.005N）
    valid_mask = contact_force_mag > 0.005
    if valid_mask.sum() == 0:
        print("[ICP] 没有有效接触点，跳过 ICP，直接返回原始点云")
        return raw_pc

    valid_contact = contact_points[valid_mask]  # [k, 3]
    print(f"[ICP] 有效接触指尖数: {valid_mask.sum().item()}, "
          f"接触点: {valid_contact.tolist()}")

    # 构建 open3d 点云
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(raw_pc.numpy())

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(valid_contact.numpy())

    # 估计 source 法线（ICP Point-to-Plane 需要）
    source_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )

    # 执行 ICP（Point-to-Point，因为 target 点少，Point-to-Plane 不稳定）
    result = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        max_correspondence_distance=threshold,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )

    T = torch.tensor(result.transformation, dtype=torch.float32)  # [4, 4]
    print(f"[ICP] fitness={result.fitness:.4f}, "
          f"inlier_rmse={result.inlier_rmse:.6f}")

    # 把变换应用到点云
    ones = torch.ones(raw_pc.shape[0], 1)
    pc_h = torch.cat([raw_pc, ones], dim=-1)  # [n, 4]
    pc_refined = (T @ pc_h.T).T[:, :3]        # [n, 3]
    return pc_refined


# ================================================================
# 推理函数：给定点云，返回新的关节角 q
# ================================================================
def run_inference(network, hand, robot_pc, object_pc, initial_q, device):
    """
    单次推理：network → multilateration → compute_link_pose → optimization

    robot_pc:  Tensor [1, n, 3]
    object_pc: Tensor [1, n, 3]
    initial_q: Tensor [1, DOF]
    返回: Tensor [1, DOF]
    """
    robot_pc = robot_pc.to(device)
    object_pc = object_pc.to(device)
    initial_q = initial_q.to(device)

    with torch.no_grad():
        dro = network(robot_pc, object_pc)['dro'].detach()

    mlat_pc = multilateration(dro, object_pc)
    transform, _ = compute_link_pose(hand.links_pc, mlat_pc, is_train=False)
    optim_transform = process_transform(hand.pk_chain, transform)
    layer = create_problem(hand.pk_chain, optim_transform.keys())
    predict_q = optimization(hand.pk_chain, layer, initial_q, optim_transform)
    return predict_q.cpu()


# ================================================================
# 主函数
# ================================================================
def isaac_main(
    mode: str,
    robot_name: str,
    object_name: str,
    batch_size: int,
    q_batch: torch.Tensor = None,
    gpu: int = 0,
    use_gui: bool = False,
    enable_tactile_loop: bool = False,   # 新增：是否开启触觉闭环
    ckpt_name: str = 'model_wujihand',   # 新增：重推理用的模型
):
    """
    主仿真入口。

    enable_tactile_loop=False → baseline，与原版行为完全相同
    enable_tactile_loop=True  → 触觉闭环模式
    """
    if mode == 'filter' and batch_size == 0:
        return 0, None
    if use_gui:
        gpu = 0

    device = torch.device(f'cuda:{gpu}')

    data_urdf_path = os.path.join(ROOT_DIR, 'data/data_urdf')
    urdf_assets_meta = json.load(open(os.path.join(data_urdf_path, 'robot/urdf_assets_meta.json')))
    robot_urdf_path = urdf_assets_meta['urdf_path'][robot_name]
    object_name_split = object_name.split('+') if object_name is not None else None
    object_urdf_path = (f'{object_name_split[0]}/{object_name_split[1]}'
                        f'/{object_name_split[1]}.urdf')

    hand = create_hand_model(robot_name)
    joint_orders = hand.get_joint_orders()

    simulator = IsaacValidator(
        robot_name=robot_name,
        joint_orders=joint_orders,
        batch_size=batch_size,
        gpu=gpu,
        is_filter=(mode == 'filter'),
        use_gui=use_gui,
        enable_tactile_loop=enable_tactile_loop,   # 传入开关
    )
    print("[Isaac] IsaacValidator is created.")

    simulator.set_asset(
        robot_path=os.path.join(data_urdf_path, 'robot'),
        robot_file=robot_urdf_path[21:],
        object_path=os.path.join(data_urdf_path, 'object'),
        object_file=object_urdf_path
    )
    simulator.create_envs()
    print("[Isaac] IsaacValidator preparation is done.")

    if mode == 'filter':
        dataset_path = os.path.join(ROOT_DIR, 'data/CMapDataset/cmap_dataset.pt')
        metadata = torch.load(dataset_path)['metadata']
        q_batch = [m[1] for m in metadata if m[2] == object_name and m[3] == robot_name]
        q_batch = torch.stack(q_batch, dim=0).to(torch.device('cpu'))
    if q_batch.shape[-1] != len(joint_orders):
        q_batch = q_rot6d_to_q_euler(q_batch)

    simulator.set_actor_pose_dof(q_batch.to(torch.device('cpu')))

    # ---- 运行仿真 ----
    result = simulator.run_sim()

    # ---- baseline 模式：result 直接是 (success, q_isaac) ----
    if not enable_tactile_loop or not isinstance(result, dict):
        success, q_isaac = result
        simulator.destroy()
        return success, q_isaac

    # ---- 触觉闭环模式 ----
    if not result.get('need_reinference', False):
        # APPROACH 阶段没有触发，已经跑完 closing+force，result 是 (success, q_isaac)
        success, q_isaac = result
        simulator.destroy()
        return success, q_isaac

    # ---- 需要重推理 ----
    print("[Tactile] 开始重推理流程...")

    triggered_mask = result['triggered_mask']           # [batch_size]
    contact_points = result['contact_points']           # [batch_size, 5, 3]
    contact_force_mag = result['contact_force_mag']     # [batch_size, 5]
    object_pose_after_retreat = result['object_pose_after_retreat']  # [batch_size, 7]

    # 1. 加载网络（只在触觉触发时才加载，节省内存）
    print(f"[Tactile] 加载推理网络: {ckpt_name}")
    network = create_network(
        SimpleNamespace(**{
            'emb_dim': 512,
            'latent_dim': 64,
            'pretrain': None,
            'center_pc': True,
            'block_computing': True
        }),
        mode='validate'
    ).to(device)
    ckpt_path = os.path.join(ROOT_DIR, f'ckpt/model/{ckpt_name}.pth')
    if not os.path.exists(ckpt_path):
        cprint(f"[Tactile] 警告：找不到权重文件 {ckpt_path}，跳过重推理", 'yellow')
        # 降级处理：用原始 q_batch 继续
        simulator.reset_target_and_close(q_batch)
        success, q_isaac = simulator.run_force_test()
        simulator.destroy()
        return success, q_isaac

    network.load_state_dict(torch.load(ckpt_path, map_location=device))
    network.eval()

    # 2. 从 hand model 拿 robot_pc（用于推理）
    hand_device = create_hand_model(robot_name, device)

    # 3. 对每个 env 做点云生成 + ICP + 重推理
    new_q_list = []
    for env_idx in range(batch_size):
        print(f"[Tactile] 处理 env {env_idx} ...")

        # 3a. 从 mesh + 当前物体位姿 生成点云（模拟退后后视觉扫描）
        raw_pc = sample_pc_from_mesh_and_pose(
            object_name,
            object_pose_after_retreat[env_idx],
            n_points=1024
        )  # [1, 1024, 3]

        # 3b. ICP 精配准（用接触点约束）
        refined_pc = icp_refine_pc(
            raw_pc[0],                        # [1024, 3]
            contact_points[env_idx],          # [5, 3]
            contact_force_mag[env_idx],       # [5]
        )  # [1024, 3]

        refined_pc_batch = refined_pc.unsqueeze(0)  # [1, 1024, 3]

        # 3c. 重推理
        robot_pc_single = hand_device.get_robot_pc().unsqueeze(0)  # [1, n, 3]，取手的标准点云
        initial_q_single = q_batch[env_idx].unsqueeze(0)           # [1, DOF]

        new_q = run_inference(
            network, hand_device,
            robot_pc_single,
            refined_pc_batch,
            initial_q_single,
            device
        )  # [1, DOF]
        new_q_list.append(new_q)
        print(f"[Tactile] env {env_idx} 重推理完成，new_q={new_q}")

    new_q_batch = torch.cat(new_q_list, dim=0)  # [batch_size, DOF]

    # 4. 用新的 q 重设手的位置并执行闭合
    simulator.reset_target_and_close(new_q_batch)

    # 5. force phase 判断成功
    success, q_isaac = simulator.run_force_test()
    simulator.destroy()
    return success, q_isaac


# ================================================================
# 命令行入口（供 validate_utils.py 子进程调用）
# ================================================================
if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--robot_name', type=str, required=True)
    parser.add_argument('--object_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--q_file', type=str)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--use_gui', action='store_true')
    # 新增触觉闭环开关
    parser.add_argument('--enable_tactile_loop', action='store_true',
                        help='开启触觉闭环模式（默认关闭，为 baseline）')
    parser.add_argument('--ckpt_name', type=str, default='model_wujihand',
                        help='重推理使用的模型权重名称')
    args = parser.parse_args()

    print(f'GPU: {args.gpu}')
    assert args.mode in ['filter', 'validation'], f"Unknown mode: {args.mode}!"
    q_batch = torch.load(args.q_file, map_location='cpu') if args.q_file is not None else None

    success, q_isaac = isaac_main(
        mode=args.mode,
        robot_name=args.robot_name,
        object_name=args.object_name,
        batch_size=args.batch_size,
        q_batch=q_batch,
        gpu=args.gpu,
        use_gui=args.use_gui,
        enable_tactile_loop=args.enable_tactile_loop,
        ckpt_name=args.ckpt_name,
    )

    success_num = success.sum().item()
    if args.mode == 'filter':
        print(f"<{args.robot_name}/{args.object_name}> before: {args.batch_size}, after: {success_num}")
        if success_num > 0:
            q_filtered = q_isaac[success]
            save_dir = str(os.path.join(ROOT_DIR, 'data/CMapDataset_filtered', args.robot_name))
            os.makedirs(save_dir, exist_ok=True)
            torch.save(q_filtered, os.path.join(save_dir, f'{args.object_name}_{success_num}.pt'))
    elif args.mode == 'validation':
        cprint(f"[{args.robot_name}/{args.object_name}] Result: {success_num}/{args.batch_size}", 'green')
        save_data = {
            'success': success,
            'q_isaac': q_isaac
        }
        os.makedirs(os.path.join(ROOT_DIR, 'tmp'), exist_ok=True)
        torch.save(save_data, os.path.join(ROOT_DIR, f'tmp/isaac_main_ret_{args.gpu}.pt'))
