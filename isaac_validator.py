from isaacgym import gymapi
from isaacgym import gymtorch

import os
import sys
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.controller import controller

# ============================================================
# 常量：wujihand 指尖在 robot asset 内的局部 body index
# 经过 Isaac Gym 实际查询确认（见项目文档）
# rigid_body_num = 33(robot) + 1(object) = 34
# object 在每个 env 里排第 0 位，robot body 从第 1 位开始
# 所以指尖的 env 内偏移 = 1 + local_index
# ============================================================
TIP_LOCAL_INDICES = [12, 17, 22, 27, 32]   # finger1~5 tip，robot asset 内局部 index
ROBOT_BODY_OFFSET  = 1                      # object 占了 index 0，robot 从 1 开始
NUM_ROBOT_BODIES   = 33
NUM_OBJECT_BODIES  = 1
# rigid_body_num 会在 set_asset 里赋值，这里先给默认值
_DEFAULT_RIGID_BODY_NUM = NUM_ROBOT_BODIES + NUM_OBJECT_BODIES  # 34


class IsaacValidator:
    def __init__(
        self,
        robot_name,
        joint_orders,
        batch_size,
        gpu=0,
        is_filter=False,
        use_gui=False,
        robot_friction=3.2,
        object_friction=3.2,
        steps_per_sec=100,
        grasp_step=100,
        debug_interval=0.01,
        # ---- 触觉闭环相关参数 ----
        tactile_threshold=0.01,      # 触发接触力阈值（N）
        approach_steps=40,           # APPROACH 阶段步数（前 N 步监听触觉）
        retreat_steps=30,            # 退后步数
        retreat_delta=0.05,          # 每步退后的距离（m），沿 virtual joint x 轴
        enable_tactile_loop=False,   # 是否开启触觉闭环（默认关闭，baseline 模式）
    ):
        self.gym = gymapi.acquire_gym()

        self.robot_name = robot_name
        self.joint_orders = joint_orders
        self.batch_size = batch_size
        self.gpu = gpu
        self.is_filter = is_filter
        self.robot_friction = robot_friction
        self.object_friction = object_friction
        self.steps_per_sec = steps_per_sec
        self.grasp_step = grasp_step
        self.debug_interval = debug_interval

        # 触觉闭环参数
        self.tactile_threshold = tactile_threshold
        self.approach_steps = approach_steps
        self.retreat_steps = retreat_steps
        self.retreat_delta = retreat_delta
        self.enable_tactile_loop = enable_tactile_loop

        self.envs = []
        self.robot_handles = []
        self.object_handles = []
        self.robot_asset = None
        self.object_asset = None
        self.rigid_body_num = _DEFAULT_RIGID_BODY_NUM
        self.object_force = None
        self.urdf2isaac_order = None
        self.isaac2urdf_order = None

        self.sim_params = gymapi.SimParams()
        self.sim_params.dt = 1 / steps_per_sec
        self.sim_params.substeps = 2
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
        self.sim_params.physx.use_gpu = True
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 8
        self.sim_params.physx.num_velocity_iterations = 0
        self.sim_params.physx.contact_offset = 0.012
        self.sim_params.physx.rest_offset = 0.0

        self.sim = self.gym.create_sim(self.gpu, self.gpu, gymapi.SIM_PHYSX, self.sim_params)
        self._rigid_body_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._dof_states = self.gym.acquire_dof_state_tensor(self.sim)

        self.viewer = None
        if use_gui:
            self.has_viewer = True
            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 1920
            self.camera_props.height = 1080
            self.camera_props.use_collision_geometry = True
            self.viewer = self.gym.create_viewer(self.sim, self.camera_props)
            self.gym.viewer_camera_look_at(self.viewer, None, gymapi.Vec3(1, 0, 0), gymapi.Vec3(0, 0, 0))
        else:
            self.has_viewer = False

        self.robot_asset_options = gymapi.AssetOptions()
        self.robot_asset_options.disable_gravity = True
        self.robot_asset_options.fix_base_link = True
        self.robot_asset_options.collapse_fixed_joints = True
        self.robot_asset_options.thickness = 0.001

        self.object_asset_options = gymapi.AssetOptions()
        self.object_asset_options.override_com = True
        self.object_asset_options.override_inertia = True
        self.object_asset_options.density = 1500

    # ------------------------------------------------------------------
    # set_asset / create_envs / set_actor_pose_dof  （与原版相同，不改动）
    # ------------------------------------------------------------------
    def set_asset(self, robot_path, robot_file, object_path, object_file):
        self.robot_asset = self.gym.load_asset(self.sim, robot_path, robot_file, self.robot_asset_options)
        self.object_asset = self.gym.load_asset(self.sim, object_path, object_file, self.object_asset_options)
        self.rigid_body_num = (self.gym.get_asset_rigid_body_count(self.robot_asset)
                               + self.gym.get_asset_rigid_body_count(self.object_asset))

    def create_envs(self):
        for env_idx in range(self.batch_size):
            env = self.gym.create_env(
                self.sim,
                gymapi.Vec3(-1, -1, -1),
                gymapi.Vec3(1, 1, 1),
                int(self.batch_size ** 0.5)
            )
            self.envs.append(env)

            if self.has_viewer:
                x_axis_dir = np.array([0, 0, 0, 1, 0, 0], dtype=np.float32)
                x_axis_color = np.array([1, 0, 0], dtype=np.float32)
                self.gym.add_lines(self.viewer, env, 1, x_axis_dir, x_axis_color)
                y_axis_dir = np.array([0, 0, 0, 0, 1, 0], dtype=np.float32)
                y_axis_color = np.array([0, 1, 0], dtype=np.float32)
                self.gym.add_lines(self.viewer, env, 1, y_axis_dir, y_axis_color)
                z_axis_dir = np.array([0, 0, 0, 0, 0, 1], dtype=np.float32)
                z_axis_color = np.array([0, 0, 1], dtype=np.float32)
                self.gym.add_lines(self.viewer, env, 1, z_axis_dir, z_axis_color)

            # object actor
            object_handle = self.gym.create_actor(
                env, self.object_asset, gymapi.Transform(),
                f'object_{env_idx}', env_idx
            )
            self.object_handles.append(object_handle)
            object_shape_properties = self.gym.get_actor_rigid_shape_properties(env, object_handle)
            for i in range(len(object_shape_properties)):
                object_shape_properties[i].friction = self.object_friction
            self.gym.set_actor_rigid_shape_properties(env, object_handle, object_shape_properties)

            # robot actor
            robot_handle = self.gym.create_actor(
                env, self.robot_asset, gymapi.Transform(),
                f'robot_{env_idx}', env_idx
            )
            self.robot_handles.append(robot_handle)
            robot_properties = self.gym.get_actor_dof_properties(env, robot_handle)
            robot_properties["driveMode"].fill(gymapi.DOF_MODE_POS)
            robot_properties["stiffness"].fill(400.0)
            robot_properties["damping"].fill(37.0)
            robot_properties["armature"].fill(0.001)
            self.gym.set_actor_dof_properties(env, robot_handle, robot_properties)

            object_shape_properties = self.gym.get_actor_rigid_shape_properties(env, robot_handle)
            for i in range(len(object_shape_properties)):
                object_shape_properties[i].friction = self.robot_friction
            self.gym.set_actor_rigid_shape_properties(env, robot_handle, object_shape_properties)

        obj_property = self.gym.get_actor_rigid_body_properties(self.envs[0], self.object_handles[0])
        object_mass = [obj_property[i].mass for i in range(len(obj_property))]
        object_mass = torch.tensor(object_mass)
        self.object_force = 0.5 * object_mass

        self.urdf2isaac_order = np.zeros(len(self.joint_orders), dtype=np.int32)
        self.isaac2urdf_order = np.zeros(len(self.joint_orders), dtype=np.int32)
        for urdf_idx, joint_name in enumerate(self.joint_orders):
            isaac_idx = self.gym.find_actor_dof_index(
                self.envs[0], self.robot_handles[0], joint_name, gymapi.DOMAIN_ACTOR
            )
            self.urdf2isaac_order[isaac_idx] = urdf_idx
            self.isaac2urdf_order[urdf_idx] = isaac_idx

    def set_actor_pose_dof(self, q):
        self.gym.prepare_sim(self.sim)

        _root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_state = gymtorch.wrap_tensor(_root_state)
        root_state[:] = torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
        self.gym.set_actor_root_state_tensor(self.sim, _root_state)

        outer_q, inner_q = controller(self.robot_name, q)

        for env_idx in range(len(self.envs)):
            env = self.envs[env_idx]
            robot_handle = self.robot_handles[env_idx]

            dof_states_initial = self.gym.get_actor_dof_states(env, robot_handle, gymapi.STATE_ALL).copy()
            dof_states_initial['pos'] = outer_q[env_idx, self.urdf2isaac_order]
            self.gym.set_actor_dof_states(env, robot_handle, dof_states_initial, gymapi.STATE_ALL)

            dof_states_target = self.gym.get_actor_dof_states(env, robot_handle, gymapi.STATE_ALL).copy()
            dof_states_target['pos'] = inner_q[env_idx, self.urdf2isaac_order]
            self.gym.set_actor_dof_position_targets(env, robot_handle, dof_states_target["pos"])

        # 保存 inner_q 供退后后恢复闭合目标使用
        self._inner_q = inner_q.clone()

    # ------------------------------------------------------------------
    # 内部工具：读取指尖世界坐标
    # ------------------------------------------------------------------
    def _get_tip_contact_forces(self):
        """
        返回每个 env 每根指尖的接触力模长。
        shape: [batch_size, 5]
        注意：需要在 gym.refresh_net_contact_force_tensor 之后调用。
        object 在 env 内占 index 0，robot body 从 index 1 开始。
        指尖的 env 内 index = ROBOT_BODY_OFFSET + tip_local_index。
        """
        contact_forces = gymtorch.wrap_tensor(
            self.gym.acquire_net_contact_force_tensor(self.sim)
        )  # shape: [batch_size * rigid_body_num, 3]

        tip_env_indices = [ROBOT_BODY_OFFSET + t for t in TIP_LOCAL_INDICES]  # [13,18,23,28,33]

        global_indices = []
        for env_id in range(self.batch_size):
            for tip_env_idx in tip_env_indices:
                global_indices.append(env_id * self.rigid_body_num + tip_env_idx)

        tip_forces = contact_forces[global_indices, :]          # [batch_size*5, 3]
        tip_forces = tip_forces.view(self.batch_size, 5, 3)     # [batch_size, 5, 3]
        magnitudes = tip_forces.norm(dim=-1)                    # [batch_size, 5]
        return magnitudes

    def _get_tip_world_positions(self):
        """
        从 rigid_body_state tensor 中读取指尖的世界坐标。
        shape: [batch_size, 5, 3]
        需要在 gym.refresh_rigid_body_state_tensor 之后调用。
        """
        rb_states = gymtorch.wrap_tensor(self._rigid_body_states)
        # rb_states shape: [batch_size * rigid_body_num, 13]  (pos3 + quat4 + vel3 + angvel3)

        tip_env_indices = [ROBOT_BODY_OFFSET + t for t in TIP_LOCAL_INDICES]

        global_indices = []
        for env_id in range(self.batch_size):
            for tip_env_idx in tip_env_indices:
                global_indices.append(env_id * self.rigid_body_num + tip_env_idx)

        tip_positions = rb_states[global_indices, :3]           # [batch_size*5, 3]
        tip_positions = tip_positions.view(self.batch_size, 5, 3)  # [batch_size, 5, 3]
        return tip_positions.clone()

    def _get_object_world_pose(self):
        """
        读取每个 env 中物体的世界位姿。
        object 在 env 内排第 0 位。
        返回 shape: [batch_size, 7]  (xyz + quat xyzw)
        """
        rb_states = gymtorch.wrap_tensor(self._rigid_body_states)
        object_global_indices = [env_id * self.rigid_body_num + 0
                                  for env_id in range(self.batch_size)]
        object_poses = rb_states[object_global_indices, :7]
        return object_poses.clone()

    def _viewer_step(self):
        if self.has_viewer:
            if self.gym.query_viewer_has_closed(self.viewer):
                return False
            t = time.time()
            while time.time() - t < self.debug_interval:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, render_collision=True)
        return True

    # ------------------------------------------------------------------
    # APPROACH 阶段：监听触觉，返回触发信息
    # ------------------------------------------------------------------
    def _run_approach_phase(self):
        """
        执行 APPROACH 阶段（前 approach_steps 步）。
        每步检查指尖接触力，一旦任意 env 的任意指尖超过阈值，立即触发。

        返回字典：
        {
            'triggered_mask': BoolTensor [batch_size],   哪些 env 被触发
            'contact_points': FloatTensor [batch_size, 5, 3],  触发时指尖世界坐标
            'triggered_step': int,  在第几步触发（未触发则为 approach_steps）
            'contact_force_mag': FloatTensor [batch_size, 5],  触发时接触力大小
        }
        """
        triggered_mask = torch.zeros(self.batch_size, dtype=torch.bool)
        contact_points = torch.zeros(self.batch_size, 5, 3)
        contact_force_mag = torch.zeros(self.batch_size, 5)
        triggered_step = self.approach_steps

        for step in range(self.approach_steps):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # 刷新 contact force tensor
            self.gym.refresh_net_contact_force_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            tip_force_mags = self._get_tip_contact_forces()  # [batch_size, 5]

            # 找到这一步新触发的 env（之前未触发 且 有指尖接触力超阈值）
            newly_triggered = (
                (~triggered_mask) &
                (tip_force_mags.max(dim=-1).values > self.tactile_threshold)
            )

            if newly_triggered.any():
                tip_positions = self._get_tip_world_positions()  # [batch_size, 5, 3]
                # 只记录新触发的 env
                contact_points[newly_triggered] = tip_positions[newly_triggered]
                contact_force_mag[newly_triggered] = tip_force_mags[newly_triggered]
                triggered_mask |= newly_triggered
                triggered_step = step
                print(f"[Tactile] 触发！step={step}, "
                      f"envs={newly_triggered.nonzero(as_tuple=False).squeeze(-1).tolist()}, "
                      f"max_force={tip_force_mags[newly_triggered].max():.4f} N")

            if not self._viewer_step():
                break

            # 如果所有 env 都已触发，提前退出
            if triggered_mask.all():
                break

        return {
            'triggered_mask': triggered_mask,
            'contact_points': contact_points,
            'triggered_step': triggered_step,
            'contact_force_mag': contact_force_mag,
        }

    # ------------------------------------------------------------------
    # CLOSING 阶段：不监听触觉，正常跑完剩余步数
    # ------------------------------------------------------------------
    def _run_closing_phase(self, remaining_steps):
        """
        执行 CLOSING 阶段，跑完剩余的 grasp_step 步。
        不监听触觉，让 PD 控制器把手指闭合到目标位置。
        """
        for step in range(remaining_steps):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            if not self._viewer_step():
                break

    # ------------------------------------------------------------------
    # 退后：让所有 env 的手沿 virtual_x 轴退后
    # ------------------------------------------------------------------
    def _run_retreat(self):
        """
        触发后让手退后：直接修改 virtual joint（DOF 0 对应 virtual_x）的目标位置。
        retreat_delta 为每步退后量，retreat_steps 为退后步数。
        退后完成后读取物体当前位姿（用于重推理）。
        """
        print(f"[Tactile] 开始退后，共 {self.retreat_steps} 步，每步 {self.retreat_delta} m")

        self.gym.refresh_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(self._dof_states).clone()
        # dof_states shape: [batch_size * num_dof, 2]  (pos, vel)
        # virtual_x 是第 0 个 DOF（在 extended URDF 里 world->virtual_x 排第一）
        num_dof = dof_states.shape[0] // self.batch_size

        # 计算退后目标：把 virtual_x DOF 的位置减去 retreat_delta * retreat_steps
        for env_idx in range(len(self.envs)):
            env = self.envs[env_idx]
            robot_handle = self.robot_handles[env_idx]
            dof_pos = self.gym.get_actor_dof_states(env, robot_handle, gymapi.STATE_ALL).copy()
            # virtual_x 在 isaac 顺序里的 index 需要通过 urdf2isaac_order 映射
            # 但 virtual joint 的 joint_name 以 'virtual' 开头，controller 里会跳过
            # 这里直接操作 DOF index 0（即 virtual_x）
            dof_pos['pos'][0] -= self.retreat_delta * self.retreat_steps
            self.gym.set_actor_dof_position_targets(env, robot_handle, dof_pos['pos'])

        for step in range(self.retreat_steps):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            if not self._viewer_step():
                break

        # 退后完成，读取物体当前位姿
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        object_pose_after_retreat = self._get_object_world_pose()  # [batch_size, 7]
        print(f"[Tactile] 退后完成，物体当前位姿（第0个env）: "
              f"xyz={object_pose_after_retreat[0, :3].tolist()}")
        return object_pose_after_retreat

    # ------------------------------------------------------------------
    # 重设目标并执行闭合（重推理后调用）
    # ------------------------------------------------------------------
    def reset_target_and_close(self, new_q):
        """
        重推理完成后，用新的关节角重新设置手的位置和目标，然后跑 closing phase。
        new_q: [batch_size, DOF]，来自重推理结果
        """
        print("[Tactile] 重设目标关节角，开始重新抓取...")
        outer_q, inner_q = controller(self.robot_name, new_q)

        for env_idx in range(len(self.envs)):
            env = self.envs[env_idx]
            robot_handle = self.robot_handles[env_idx]
            dof_states = self.gym.get_actor_dof_states(env, robot_handle, gymapi.STATE_ALL).copy()
            dof_states['pos'] = outer_q[env_idx, self.urdf2isaac_order]
            self.gym.set_actor_dof_states(env, robot_handle, dof_states, gymapi.STATE_ALL)
            dof_states_target = self.gym.get_actor_dof_states(env, robot_handle, gymapi.STATE_ALL).copy()
            dof_states_target['pos'] = inner_q[env_idx, self.urdf2isaac_order]
            self.gym.set_actor_dof_position_targets(env, robot_handle, dof_states_target['pos'])

        self._run_closing_phase(self.grasp_step)

    # ------------------------------------------------------------------
    # force phase：施加外力，判断成功（与原版完全相同）
    # ------------------------------------------------------------------
    def _run_force_phase(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        start_pos = gymtorch.wrap_tensor(self._rigid_body_states)[::self.rigid_body_num, :3].clone()

        force_tensor = torch.zeros([len(self.envs), self.rigid_body_num, 3])
        x_pos_force = force_tensor.clone(); x_pos_force[:, 0, 0] = self.object_force
        x_neg_force = force_tensor.clone(); x_neg_force[:, 0, 0] = -self.object_force
        y_pos_force = force_tensor.clone(); y_pos_force[:, 0, 1] = self.object_force
        y_neg_force = force_tensor.clone(); y_neg_force[:, 0, 1] = -self.object_force
        z_pos_force = force_tensor.clone(); z_pos_force[:, 0, 2] = self.object_force
        z_neg_force = force_tensor.clone(); z_neg_force[:, 0, 2] = -self.object_force
        force_list = [x_pos_force, y_pos_force, z_pos_force, x_neg_force, y_neg_force, z_neg_force]

        for step in range(self.steps_per_sec * 6):
            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(force_list[step // self.steps_per_sec]),
                None,
                gymapi.ENV_SPACE
            )
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            if not self._viewer_step():
                break

        self.gym.refresh_rigid_body_state_tensor(self.sim)
        end_pos = gymtorch.wrap_tensor(self._rigid_body_states)[::self.rigid_body_num, :3].clone()
        distance = (end_pos - start_pos).norm(dim=-1)

        DIST_THRESHOLD = 0.02
        FLY_THRESHOLD = 0.2

        if self.is_filter:
            success = (distance <= DIST_THRESHOLD) & (end_pos.norm(dim=-1) <= 0.05)
        else:
            success = (distance <= DIST_THRESHOLD)

        # 计算 q_isaac（与原版相同）
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        object_pose = gymtorch.wrap_tensor(self._rigid_body_states).clone()[::self.rigid_body_num, :7]
        object_transform = np.eye(4)[np.newaxis].repeat(self.batch_size, axis=0)
        object_transform[:, :3, 3] = object_pose[:, :3]
        object_transform[:, :3, :3] = Rotation.from_quat(object_pose[:, 3:7]).as_matrix()

        self.gym.refresh_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(self._dof_states).clone().reshape(len(self.envs), -1, 2)[:, :, 0]
        robot_transform = np.eye(4)[np.newaxis].repeat(self.batch_size, axis=0)
        robot_transform[:, :3, 3] = dof_states[:, :3]
        robot_transform[:, :3, :3] = Rotation.from_euler('XYZ', dof_states[:, 3:6]).as_matrix()
        robot_transform = np.linalg.inv(object_transform) @ robot_transform
        dof_states[:, :3] = torch.tensor(robot_transform[:, :3, 3])
        dof_states[:, 3:6] = torch.tensor(Rotation.from_matrix(robot_transform[:, :3, :3]).as_euler('XYZ'))
        q_isaac = dof_states[:, self.isaac2urdf_order].to(torch.device('cpu'))

        success_mask = distance <= DIST_THRESHOLD
        fly_mask = distance > FLY_THRESHOLD
        drop_mask = (distance > DIST_THRESHOLD) & (distance <= FLY_THRESHOLD)
        print(f"   [Debug] OK:{success_mask.sum().item()} | "
              f"Drop:{drop_mask.sum().item()} | Fly:{fly_mask.sum().item()}", flush=True)

        return success, q_isaac

    # ------------------------------------------------------------------
    # run_sim：主入口，兼容原版接口，同时支持触觉闭环
    # ------------------------------------------------------------------
    def run_sim(self):
        """
        主仿真循环。

        如果 enable_tactile_loop=False（默认），行为与原版完全相同。
        如果 enable_tactile_loop=True，执行触觉闭环逻辑：
            APPROACH → 触发检测 → 退后 → （外部重推理）→ 闭合 → force test

        返回：
            如果 enable_tactile_loop=False：
                (success, q_isaac)  与原版相同
            如果 enable_tactile_loop=True 且发生触发：
                {'need_reinference': True,
                 'triggered_mask': ...,
                 'contact_points': ...,
                 'object_pose_after_retreat': ...}
                 → 外部（isaac_main.py）完成重推理后，调用 reset_target_and_close(new_q)
                 → 再调用 run_force_test() 得到最终结果
            如果 enable_tactile_loop=True 且未发生触发：
                先跑完 closing phase，再返回 (success, q_isaac)
        """
        if not self.enable_tactile_loop:
            # ---- baseline 模式：与原版完全相同 ----
            for step in range(self.grasp_step):
                self.gym.simulate(self.sim)
                if self.has_viewer:
                    if self.gym.query_viewer_has_closed(self.viewer):
                        break
                    t = time.time()
                    while time.time() - t < self.debug_interval:
                        self.gym.step_graphics(self.sim)
                        self.gym.draw_viewer(self.viewer, self.sim, render_collision=True)
            return self._run_force_phase()

        # ---- 触觉闭环模式 ----
        # 1. APPROACH 阶段
        approach_result = self._run_approach_phase()
        triggered_mask = approach_result['triggered_mask']

        if not triggered_mask.any():
            # 没有任何 env 触发，正常跑完 closing
            print("[Tactile] APPROACH 阶段未检测到接触，直接进入 CLOSING 阶段")
            remaining = self.grasp_step - self.approach_steps
            self._run_closing_phase(remaining)
            return self._run_force_phase()

        # 2. 有触发：退后
        object_pose_after_retreat = self._run_retreat()

        # 3. 通知外部需要重推理，返回中间状态
        print(f"[Tactile] 需要重推理，触发的 env: "
              f"{triggered_mask.nonzero(as_tuple=False).squeeze(-1).tolist()}")
        return {
            'need_reinference': True,
            'triggered_mask': triggered_mask,
            'contact_points': approach_result['contact_points'],        # [batch_size, 5, 3]
            'contact_force_mag': approach_result['contact_force_mag'],  # [batch_size, 5]
            'object_pose_after_retreat': object_pose_after_retreat,     # [batch_size, 7]
        }

    def run_force_test(self):
        """供外部在重推理+reset_target_and_close之后调用，得到最终成功结果。"""
        return self._run_force_phase()

    def reset_simulator(self):
        self.gym.destroy_sim(self.sim)
        if self.has_viewer:
            self.gym.destroy_viewer(self.viewer)
            self.viewer = self.gym.create_viewer(self.sim, self.camera_props)
        self.sim = self.gym.create_sim(self.gpu, self.gpu, gymapi.SIM_PHYSX, self.sim_params)
        for env in self.envs:
            self.gym.destroy_env(env)
        self.envs = []
        self.robot_handles = []
        self.object_handles = []
        self.robot_asset = None
        self.object_asset = None

    def destroy(self):
        for env in self.envs:
            self.gym.destroy_env(env)
        self.gym.destroy_sim(self.sim)
        if self.has_viewer:
            self.gym.destroy_viewer(self.viewer)
        del self.gym
