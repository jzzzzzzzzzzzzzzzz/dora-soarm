# /home/jzzz/so100_dora/so100_dora_mujoco/nodes/so100_planner/so100_planner/main.py
import pyarrow as pa
import numpy as np
from dora import Node
import argparse
import os
import time

# --- Pinocchio and PyRoboPlan Imports ---
import pinocchio
import pyroboplan.core as prc # 主要用于 Pose (如果还用的话)
import pyroboplan.ik.differential_ik as pr_diff_ik # 明确导入 DifferentialIk 和 Options
# import pyroboplan.ik.nullspace_components as pr_nullspace # 如果要用空域控制器

from scipy.spatial.transform import Rotation as R # For converting Euler to Quaternion

# --- 配置 ---
BASE_LINK_DEFAULT = "Base" # URDF 中的基座连杆
EE_LINK_DEFAULT = "Fixed_Jaw"   # 末端执行器连杆
INTERPOLATION_STEPS = 20
STEP_DELAY = 0.05
NUM_ARM_JOINTS_FOR_IK = 5 # 我们用 IK 控制的臂部关节数量 (Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll)
TOTAL_MUJOCO_JOINTS = 6   # MuJoCo 模型总共控制的关节数 (包括夹爪)
# 配置文件路径，用于获取准确的关节名称 (假设与 MuJoCo 顺序一致)
# 这个路径需要根据实际情况调整，或者从环境变量/参数传入
# 我们这里先硬编码一个假设的路径，实际项目中应该更灵活
JOINT_CONFIG_PATH_FOR_NAMES = "../configs/so100_config.json" # <--- 确保这个路径正确或动态获取

# -------------

class Planner:
    def __init__(self, node_name: str, urdf_path: str, base_link_name: str, ee_link_name: str):
        self.node = Node(node_name)
        self.node_id_log_prefix = node_name
        self.urdf_path = urdf_path
        # base_link_name 在 Pinocchio 加载时不太直接用到，它会从 URDF 根开始
        self.ee_link_name = ee_link_name

        print(f"INFO ({self.node_id_log_prefix}): Initializing planner with Pinocchio and PyRoboPlan DifferentialIK.")
        print(f"INFO ({self.node_id_log_prefix}): Loading URDF from: {self.urdf_path}")
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")

        try:
            print(f"Attempting to load URDF using pinocchio (version {pinocchio.__version__})...")
            # 直接使用顶层函数
            self.pin_model = pinocchio.buildModelFromUrdf(self.urdf_path)
            print(f"INFO ({self.node_id_log_prefix}): Pinocchio model loaded successfully using pinocchio.buildModelFromUrdf. Name: {self.pin_model.name}")
            print(f"INFO ({self.node_id_log_prefix}): nq (config size): {self.pin_model.nq}, nv (velocity size): {self.pin_model.nv}")

            # 2. 创建 Pinocchio Data 对象
            self.pin_data = self.pin_model.createData()

            # 3. 获取用于IK的关节名称和在 Pinocchio 模型中的索引
            # 我们需要一个从 MuJoCo 关节顺序到 Pinocchio q 向量中对应位置的映射
            # 假设 so100_config.json 中的 "joint_names" 就是 MuJoCo 控制的顺序
            # 并且前 NUM_ARM_JOINTS_FOR_IK 个是手臂关节
            
            # --- 加载关节名 (用于映射) ---
            # 实际项目中，这个配置路径应该更灵活地传入
            # config_abs_path = os.path.join(os.path.dirname(__file__), JOINT_CONFIG_PATH_FOR_NAMES) # 假设config在planner节点同级目录的configs下
            # 为了简单，我们先假设一个固定的、从 graph 启动时的相对路径
            # CWD 在 graph 启动时是 graphs 目录
            # graph_dir = os.getcwd() # 这在 __init__ 中可能不是 graph 目录
            # urdf_dir = os.path.dirname(os.path.abspath(self.urdf_path)) # ../models/urdf
            # project_root = os.path.dirname(os.path.dirname(urdf_dir)) # ../
            # config_abs_path = os.path.join(project_root, "configs/so100_config.json") # 假设的路径

            # 让我们直接从 YAML 中获取 CONFIG 路径，就像 mujoco_client 那样
            # 这需要 planner 节点也能访问这个环境变量
            config_path_from_env = os.getenv("CONFIG_FOR_PLANNER_JOINT_NAMES") # 需要在 YAML 中为 planner 设置这个 env
            if not config_path_from_env:
                 # 如果环境变量没有，尝试一个硬编码的相对路径 (从 graph 目录出发)
                 # 这非常不推荐，只是作为后备的后备
                 config_path_from_env = "../configs/so100_config.json"
                 print(f"WARNING ({self.node_id_log_prefix}): CONFIG_FOR_PLANNER_JOINT_NAMES env not set. Falling back to hardcoded relative path: {config_path_from_env}")


            if not os.path.exists(config_path_from_env):
                 raise FileNotFoundError(f"Joint config file for planner NOT FOUND: {config_path_from_env}")
            with open(config_path_from_env) as file:
                joint_config_data = json.load(file)
            
            all_mujoco_joint_names_from_config = joint_config_data.get("joint_names", [])
            if len(all_mujoco_joint_names_from_config) < NUM_ARM_JOINTS_FOR_IK:
                raise ValueError(f"Config file '{config_path_from_env}' has too few joint names.")
            
            self.mujoco_arm_joint_names = all_mujoco_joint_names_from_config[:NUM_ARM_JOINTS_FOR_IK]
            print(f"INFO ({self.node_id_log_prefix}): MuJoCo arm joint names for IK mapping: {self.mujoco_arm_joint_names}")

            # 存储 Pinocchio 中这些关节的 q 和 v 向量的起始索引
            self.pin_q_indices_for_arm_joints = []
            self.pin_v_indices_for_arm_joints = [] # 虽然IK主要用q，但知道v的索引也有用

            for joint_name in self.mujoco_arm_joint_names:
                if self.pin_model.existJointName(joint_name):
                    joint_id = self.pin_model.getJointId(joint_name)
                    self.pin_q_indices_for_arm_joints.append(self.pin_model.idx_qs[joint_id])
                    self.pin_v_indices_for_arm_joints.append(self.pin_model.idx_vs[joint_id])
                else:
                    raise ValueError(f"Joint '{joint_name}' from config not found in Pinocchio model. Available Pinocchio joint names: {[self.pin_model.names[i] for i in range(1, self.pin_model.njoints)]}")
            
            print(f"INFO ({self.node_id_log_prefix}): Pinocchio q_vector indices for arm joints: {self.pin_q_indices_for_arm_joints}")


            # 4. 创建 Differential IK 求解器实例
            # 配置 DifferentialIkOptions
            # 这些参数需要根据你的机器人和需求进行调整
            ik_options = pr_diff_ik.DifferentialIkOptions(
                max_iters=200,
                max_retries=5, # 减少重试次数以加快响应
                max_translation_error=0.005, # m
                max_rotation_error=0.01,    # rad
                damping=0.5, # 阻尼项，防止奇异姿态附近步长过大
                min_step_size=0.01, # 最小步长
                max_step_size=0.2,  # 最大步长
                # ignore_joint_indices=[], # 如果有不想参与IK的Pinocchio关节索引，可以在这里指定
                # joint_weights=None, # 可以为不同关节设置权重
                rng_seed=None # 随机种子，用于随机重启
            )

            self.ik_solver = pr_diff_ik.DifferentialIk(
                self.pin_model,
                data=self.pin_data, # 需要传入 Pinocchio Data
                options=ik_options,
                # collision_model=None, # 暂时不加碰撞模型
                # collision_data=None,
                # visualizer=None
            )
            print(f"INFO ({self.node_id_log_prefix}): PyRoboPlan DifferentialIk solver created for EE: '{self.ee_link_name}'.")

            # 检查末端执行器连杆是否存在于 Pinocchio 模型中
            if not self.pin_model.existFrame(self.ee_link_name):
                 # 尝试添加一个与连杆同名的 frame (如果 URDF 中没有显式定义 frame)
                 # 这通常在 ee_link 是一个 link 而不是一个特定 frame 时需要
                 ee_link_id = self.pin_model.getFrameId(self.ee_link_name, pinocchio.BODY) # 尝试按BODY获取
                 if ee_link_id < self.pin_model.nframes:
                     print(f"INFO ({self.node_id_log_prefix}): EE link '{self.ee_link_name}' found as a body/link. IK will target this link's origin.")
                 else: # 如果连杆本身都找不到，就报错
                    available_frames = [self.pin_model.frames[i].name for i in range(self.pin_model.nframes)]
                    raise ValueError(f"End-effector frame '{self.ee_link_name}' not found in Pinocchio model. Available frames: {available_frames}")
            else:
                print(f"INFO ({self.node_id_log_prefix}): End-effector frame '{self.ee_link_name}' found in Pinocchio model.")


        except Exception as e:
            print(f"ERROR ({self.node_id_log_prefix}): Error initializing Pinocchio/PyRoboPlan: {e}")
            import traceback
            traceback.print_exc()
            raise

        self.current_mujoco_joint_positions = None

    def run(self):
        print(f"INFO ({self.node_id_log_prefix}): Planner run loop started.")
        for event in self.node:
            # ... (处理 current_joint_state_from_mujoco 的逻辑不变) ...
            if event["type"] == "INPUT":
                event_id = event["id"]
                value = event["value"]

                if event_id == "current_joint_state_from_mujoco":
                    try:
                        q_current_mujoco_full = value.to_numpy(zero_copy_only=False)
                        if len(q_current_mujoco_full) == TOTAL_MUJOCO_JOINTS:
                            self.current_mujoco_joint_positions = q_current_mujoco_full
                        else:
                            self.current_mujoco_joint_positions = None
                    except Exception as e:
                        self.current_mujoco_joint_positions = None

                elif event_id == "target_cartesian_pose_command":
                    # 1. 准备 IK 的初始猜测 (Pinocchio 的 q 向量)
                    initial_q_pin = pinocchio.neutral(self.pin_model) # 获取模型的零位姿态 (长度为 nq)
                    if self.current_mujoco_joint_positions is not None:
                        # 将 MuJoCo 的手臂关节角度填充到 Pinocchio q 向量的对应位置
                        for i in range(NUM_ARM_JOINTS_FOR_IK):
                            mujoco_val = self.current_mujoco_joint_positions[i]
                            pin_q_idx = self.pin_q_indices_for_arm_joints[i]
                            # 假设都是单自由度旋转关节
                            initial_q_pin[pin_q_idx] = mujoco_val
                        # print(f"DEBUG ({self.node_id_log_prefix}): Using initial q for Pinocchio IK: {initial_q_pin}")
                    else:
                        print(f"DEBUG ({self.node_id_log_prefix}): No current MuJoCo state, using Pinocchio neutral configuration as initial guess for IK.")


                    try:
                        command_str = value[0].as_py().strip()
                        parts = command_str.split()
                        if parts[0].lower() == "go_to_cartesian" and len(parts) == 7:
                            target_xyz_list = [float(p) for p in parts[1:4]]
                            target_rxyz_rad_list = [float(p) for p in parts[4:7]]
                            print(f"INFO ({self.node_id_log_prefix}): Parsed target: XYZ={target_xyz_list}, EulerRPY_rad={target_rxyz_rad_list}")

                            # 2. 将目标姿态转换为 pinocchio.SE3 对象
                            target_translation = np.array(target_xyz_list)
                            # 假设输入的 rxyz_rad 是 ZYX 顺序的欧拉角 (roll, pitch, yaw)
                            try:
                                rot_matrix = R.from_euler('zyx', target_rxyz_rad_list).as_matrix()
                            except Exception as e_rot:
                                print(f"ERROR ({self.node_id_log_prefix}): Could not convert Euler to Rotation Matrix: {e_rot}")
                                continue
                            
                            target_tform_pin = pinocchio.SE3(rot_matrix, target_translation)
                            # print(f"DEBUG ({self.node_id_log_prefix}): Target SE3 for Pinocchio IK: {target_tform_pin}")

                            # 3. 执行 IK
                            # nullspace_components 可以先为空列表
                            # 例如: joint_limits_ns = lambda m, q: pr_nullspace.joint_limit_nullspace_component(m, q, gain=0.1, padding=0.05)
                            q_solution_pin = self.ik_solver.solve(
                                self.ee_link_name,
                                target_tform_pin,
                                init_state=initial_q_pin,
                                nullspace_components=[], # 暂时不使用空域控制器
                                verbose=True # 可以设为 True 进行详细调试
                            )

                            if q_solution_pin is not None:
                                # 4. 从 Pinocchio 的 q 解中提取手臂关节角度
                                target_q_for_mujoco_arm = np.zeros(NUM_ARM_JOINTS_FOR_IK)
                                for i in range(NUM_ARM_JOINTS_FOR_IK):
                                    pin_q_idx = self.pin_q_indices_for_arm_joints[i]
                                    target_q_for_mujoco_arm[i] = q_solution_pin[pin_q_idx]
                                
                                print(f"INFO ({self.node_id_log_prefix}): DifferentialIK solution found. MuJoCo arm joints: {target_q_for_mujoco_arm}")

                                # --- 插值和发送逻辑 (与之前类似) ---
                                start_joints_for_interp = np.zeros(NUM_ARM_JOINTS_FOR_IK)
                                if self.current_mujoco_joint_positions is not None:
                                    start_joints_for_interp = self.current_mujoco_joint_positions[:NUM_ARM_JOINTS_FOR_IK]
                                
                                for i_step in range(INTERPOLATION_STEPS + 1):
                                    alpha = i_step / INTERPOLATION_STEPS
                                    interpolated_arm_joints = (1 - alpha) * start_joints_for_interp + alpha * target_q_for_mujoco_arm
                                    
                                    current_jaw_angle = 0.0
                                    if self.current_mujoco_joint_positions is not None:
                                        current_jaw_angle = self.current_mujoco_joint_positions[NUM_ARM_JOINTS_FOR_IK]
                                    
                                    full_goal_for_mujoco = np.concatenate((interpolated_arm_joints, [current_jaw_angle]))
                                    self.node.send_output("goal_joint_positions_to_mujoco", pa.array(full_goal_for_mujoco))
                                    time.sleep(STEP_DELAY)
                                print(f"INFO ({self.node_id_log_prefix}): Interpolation finished.")
                            else:
                                print(f"ERROR ({self.node_id_log_prefix}): DifferentialIK failed to find a solution.")
                        # ... (处理其他命令或错误格式) ...
                    except Exception as e:
                        print(f"ERROR ({self.node_id_log_prefix}): Error processing cartesian command: {e}")
                        import traceback
                        traceback.print_exc()
            # ... (处理 STOP 和 ERROR 事件) ...
# ... (main 函数基本不变, 但要确保 planner 节点能获取到关节配置文件路径) ...

# 在 main 函数中，需要确保 planner 能够获取到关节配置文件路径
# 例如，可以通过一个新的环境变量传递，或者修改 YAML
def main():
    parser = argparse.ArgumentParser(description="SO100 Path Planner using Pinocchio and PyRoboPlan DifferentialIK")
    parser.add_argument("--name", type=str, required=False, default="so100_planner_node") # 添加 default
    parser.add_argument("--urdf", type=str, required=False,
                        help="Path to the SO100 URDF file (alternative to URDF_PATH env var).")
    parser.add_argument("--base-link", type=str, # default 在类定义中处理或从env获取
                        required=False, help="Name of the base link in URDF (alternative to BASE_LINK_NAME env var).")
    parser.add_argument("--ee-link", type=str, # default 在类定义中处理或从env获取
                        required=False, help="Name of the end-effector link in URDF (alternative to EE_LINK_NAME env var).")
    parser.add_argument("--joint-config", type=str, required=False,
                        help="Path to the joint names config file (alternative to CONFIG_FOR_PLANNER_JOINT_NAMES env var).")

    args = parser.parse_args()

    # --- 从环境变量获取配置 ---
    urdf_path_from_env = os.getenv("URDF_PATH")
    base_link_from_env = os.getenv("BASE_LINK_NAME")     # <--- 补上
    ee_link_from_env = os.getenv("EE_LINK_NAME")         # <--- 补上
    joint_config_from_env = os.getenv("CONFIG_FOR_PLANNER_JOINT_NAMES")

    # 优先使用环境变量，否则使用命令行参数，最后使用类定义的默认值
    final_urdf_path = urdf_path_from_env if urdf_path_from_env is not None else args.urdf
    final_base_link = base_link_from_env if base_link_from_env is not None else args.base_link # <--- 补上
    final_ee_link = ee_link_from_env if ee_link_from_env is not None else args.ee_link     # <--- 补上
    final_joint_config_path = joint_config_from_env if joint_config_from_env is not None else args.joint_config

    if not final_urdf_path:
        raise ValueError("URDF_PATH must be set via YAML env or --urdf argument.")

    # 如果 base_link 或 ee_link 仍然是 None，Planner 类会使用其内部定义的 DEFAULT (或者我们在这里提供模块级的默认值)
    if final_base_link is None: final_base_link = BASE_LINK_DEFAULT # 使用模块级默认值
    if final_ee_link is None: final_ee_link = EE_LINK_DEFAULT     # 使用模块级默认值


    if not final_joint_config_path:
        print(f"WARNING (Planner Main): Joint config path not explicitly provided. Planner will try a default relative path if env var is not set in its __init__.")


    print(f"INFO (Planner Main): CWD: {os.getcwd()}")
    print(f"INFO (Planner Main): URDF: '{final_urdf_path}', EE: '{final_ee_link}'") # 现在 final_ee_link 应该已定义
    if final_joint_config_path:
         print(f"INFO (Planner Main): Joint Config for Planner: '{final_joint_config_path}'")

    planner = Planner(args.name, final_urdf_path, final_base_link, final_ee_link)
    planner.run()


if __name__ == "__main__":
    # 需要导入 json
    import json
    main()