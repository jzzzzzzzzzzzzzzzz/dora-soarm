import os
import argparse
import time
import json
import numpy as np # 添加 numpy
import pyarrow as pa
from dora import Node
import mujoco
import mujoco.viewer

class Client:
    def __init__(self, config: dict[str, any]):
        self.config = config
        self.node = Node(config["name"]) # 将 Node 初始化提前

        print(f"Loading MuJoCo model from: {config['scene']}")
        try:
            self.m = mujoco.MjModel.from_xml_path(filename=config["scene"])
        except Exception as e:
            print(f"Error loading MuJoCo model: {e}")
            raise
        self.data = mujoco.MjData(self.m)

        # 加载关节名称并获取索引
        self.joint_names = config["joints"].to_pylist()
        print(f"Tracking joints: {self.joint_names}")
        try:
            self.joint_qpos_indices = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names]
            # 获取每个关节的第一个 qpos 地址 (假设每个关节只有一个自由度)
            self.joint_qpos_addrs = [self.m.jnt_qposadr[i] for i in self.joint_qpos_indices]
            # 获取每个关节的第一个 dof 地址 (用于速度)
            self.joint_dof_addrs = [self.m.jnt_dofadr[i] for i in self.joint_qpos_indices]
            print(f"Joint qpos addresses: {self.joint_qpos_addrs}")
            print(f"Joint dof addresses: {self.joint_dof_addrs}")
        except ValueError as e:
            print(f"Error finding joint ID/address: {e}. Check joint names in config and model.")
            raise

    # ... (run 方法保持不变) ...

    def pull_position(self, node, metadata):
        # 从 data.qpos 中提取指定关节的位置
        current_positions = self.data.qpos[self.joint_qpos_addrs]
        # print(f"Current Positions: {current_positions}") # Debug print
        node.send_output("position", pa.array(current_positions), metadata)

    def pull_velocity(self, node, metadata):
        # 从 data.qvel 中提取指定关节的速度
        current_velocities = self.data.qvel[self.joint_dof_addrs]
        # print(f"Current Velocities: {current_velocities}") # Debug print
        node.send_output("velocity", pa.array(current_velocities), metadata)

    def pull_current(self, node, metadata):
         # 如果模型中有力传感器或需要估算，可以在这里实现
         # 暂时发送空值或零值
         current_effort = np.zeros(len(self.joint_names)) # Placeholder
         node.send_output("effort", pa.array(current_effort), metadata)

    # ... (write_goal_position 暂时保持原样或 pass) ...
    def write_goal_position(self, goal_position_with_joints):
         pass # 稍后在步骤 2 中实现

# ... (main 函数基本保持不变, 确保 bus 字典正确) ...
def main():
    parser = argparse.ArgumentParser(...) # 同前
    args = parser.parse_args()

    # --- 配置加载逻辑 ---
    scene_path = os.getenv("SCENE", args.scene)
    if not scene_path:
         raise ValueError("SCENE path is required via --scene or SCENE env var.")
    if not os.path.isabs(scene_path): # Dora CWD is the graph file location
        graph_dir = os.getcwd()
        scene_path = os.path.abspath(os.path.join(graph_dir, scene_path))

    config_path = os.getenv("CONFIG", args.config)
    if not config_path:
        raise ValueError("CONFIG path is required via --config or CONFIG env var.")
    if not os.path.isabs(config_path):
        graph_dir = os.getcwd()
        config_path = os.path.abspath(os.path.join(graph_dir, config_path))

    print(f"Using Scene: {scene_path}")
    print(f"Using Config: {config_path}")

    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene file not found: {scene_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as file:
        config_data = json.load(file)

    joint_names_list = config_data.get("joint_names", [])
    if not joint_names_list:
         raise ValueError("`joint_names` key not found or empty in config JSON.")

    bus = {
        "name": args.name,
        "scene": scene_path,
        "joints": pa.array(joint_names_list, pa.string()),
    }
    # --- 配置加载逻辑结束 ---

    print("Mujoco Client Configuration: ", bus, flush=True)
    client = Client(bus)
    client.run()

if __name__ == "__main__":
    main()