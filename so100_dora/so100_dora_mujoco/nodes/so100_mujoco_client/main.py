"""
Mujoco Client: This node is used to represent simulated robot, it can be used to read virtual positions,
or can be controlled
"""

import os
import argparse
import time
import json
import numpy as np 
import pyarrow as pa

from dora import Node

import mujoco
import mujoco.viewer

class Client:

    def __init__(self, config: dict[str, any]):
        print(f"DEBUG (Node {config.get('name', 'UNKNOWN_CLIENT')}): Client __init__ CALLED.")
        self.config = config
        self.node = Node(config["name"])

        print(f"DEBUG ({config['name']}): Current working directory in __init__: {os.getcwd()}")
        scene_file = config.get("scene")
        config_file_path_from_bus = config.get("config_file_path") # 从 bus 获取配置文件路径

        print(f"DEBUG ({config['name']}): Attempting to load scene from path: '{scene_file}'")
        if not scene_file or not os.path.exists(scene_file):
            print(f"CRITICAL ({config['name']}): Scene file NOT FOUND or not specified: '{scene_file}'")
            raise FileNotFoundError(f"Scene file not found: {scene_file}")
        
        self.m = mujoco.MjModel.from_xml_path(filename=scene_file)
        print(f"DEBUG ({config['name']}): mujoco.MjModel.from_xml_path for scene '{scene_file}' successful.")
        self.data = mujoco.MjData(self.m)
        print(f"DEBUG ({config['name']}): mujoco.MjData successful.")

        # 加载关节名称并获取索引
        # 如果 bus 中直接传递了 joints 列表:
        if "joints" in config and isinstance(config["joints"], pa.Array):
             self.joint_names = config["joints"].to_pylist()
        # 如果 bus 中传递的是配置文件路径，则从文件加载:
        elif config_file_path_from_bus:
            print(f"DEBUG ({config['name']}): Attempting to load joint config from path: '{config_file_path_from_bus}'")
            if not os.path.exists(config_file_path_from_bus):
                print(f"CRITICAL ({config['name']}): Joint config file NOT FOUND: '{config_file_path_from_bus}'")
                raise FileNotFoundError(f"Joint config file not found: {config_file_path_from_bus}")
            with open(config_file_path_from_bus) as file:
                config_data_json = json.load(file)
            self.joint_names = config_data_json.get("joint_names", [])
            if not self.joint_names:
                raise ValueError(f"`joint_names` key not found or empty in config file: {config_file_path_from_bus}")
        else:
            raise ValueError("Either 'joints' array or 'config_file_path' must be provided in the bus configuration for Client.")

        print(f"DEBUG ({config['name']}): Tracking joints: {self.joint_names}")
        try:
            self.joint_qpos_indices = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names]
            self.joint_qpos_addrs = [self.m.jnt_qposadr[i] for i in self.joint_qpos_indices]
            self.joint_dof_addrs = [self.m.jnt_dofadr[i] for i in self.joint_qpos_indices]
            print(f"DEBUG ({config['name']}): Joint qpos addresses: {self.joint_qpos_addrs}")
            print(f"DEBUG ({config['name']}): Joint dof addresses: {self.joint_dof_addrs}")
        except ValueError as e:
            print(f"ERROR ({config['name']}): Error finding joint ID/address: {e}. Check joint names in config and model.")
            raise
        print(f"DEBUG (Node {config.get('name', 'UNKNOWN_CLIENT')}): Client __init__ FINISHED.")


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
    
    def write_goal_position(self, goal_position_value, metadata):
        node_name_log = self.config.get('name', 'UNKNOWN_CLIENT')
        print(f"DEBUG (Node {node_name_log}): write_goal_position CALLED.")
        try:
            if not isinstance(goal_position_value, pa.FloatingPointArray):
                print(f"WARNING (Node {node_name_log}): Received unexpected type: {type(goal_position_value)}")
                return
            goal_positions = goal_position_value.to_numpy(zero_copy_only=False)
            print(f"INFO (Node {node_name_log}): Received goal position: {goal_positions} (metadata: {metadata})")
            if len(goal_positions) != len(self.joint_names):
                print(f"WARNING (Node {node_name_log}): Expected {len(self.joint_names)} joints, got {len(goal_positions)}")
                return
            self.data.qpos[self.joint_qpos_addrs] = goal_positions
            print(f"DEBUG (Node {node_name_log}): Applied qpos: {goal_positions}")
        except Exception as e:
            print(f"ERROR (Node {node_name_log}): Failed processing write_goal_position: {e}")

    def run(self):
        node_name_log = self.config.get('name', 'UNKNOWN_CLIENT')
        print(f"DEBUG (Node {node_name_log}): Client run() CALLED.")
        try:
            with mujoco.viewer.launch_passive(self.m, self.data) as viewer:
                print(f"INFO (Node {node_name_log}): MuJoCo viewer launched. Waiting for inputs...")
                for event in self.node:
                    event_type = event["type"]
                    if event_type == "INPUT":
                        event_id = event["id"]
                        if event_id == "tick":
                            if not viewer.is_running(): break
                            step_start = time.time()
                            mujoco.mj_step(self.m, self.data)
                            viewer.sync()
                            time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
                            if time_until_next_step > 0: time.sleep(time_until_next_step)
                        elif event_id == "pull_position": self.pull_position(self.node, event["metadata"])
                        elif event_id == "pull_velocity": self.pull_velocity(self.node, event["metadata"])
                        elif event_id == "pull_current": self.pull_current(self.node, event["metadata"])
                        elif event_id == "write_goal_position": self.write_goal_position(event["value"], event["metadata"])
                        elif event_id == "end": print(f"INFO (Node {node_name_log}): Received end signal."); break
                    elif event_type == "STOP": print(f"INFO (Node {node_name_log}): Received STOP signal."); break
                    elif event_type == "ERROR": print(f"ERROR (Node {node_name_log}): Received error: {event['error']}"); raise ValueError("STOPPING due to dataflow error.")
            print(f"INFO (Node {node_name_log}): Exiting run loop.")
        except Exception as e:
            print(f"CRITICAL ERROR in {node_name_log} Client run() method: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"DEBUG (Node {node_name_log}): Client run() FINISHED or EXITED.")

def main():
    print("DEBUG: so100_mujoco_client main() function CALLED.")

    parser = argparse.ArgumentParser(
        description="MujoCo Client Node"
    )
    parser.add_argument(
        "--name", type=str, required=False, default="so100_mujoco_client_node", # 和 YAML id 一致
        help="The name of the node in the dataflow.",
    )
    parser.add_argument(
        "--scene", type=str, required=False, help="The scene file of the MuJoCo simulation.",
    )
    parser.add_argument(
        "--config", type=str, help="The configuration file for the joints.", default=None
    )
    args = parser.parse_args()

    # 获取 SCENE 和 CONFIG 路径
    # 将 YAML 中 env 定义的 SCENE 和 CONFIG 设置为环境变量
    # argparse 的 default 只有在环境变量不存在时才生效
    scene_path_from_env = os.getenv("SCENE")
    config_path_from_env = os.getenv("CONFIG")

    # 如果环境变量没有设置，则使用命令行参数 (如果提供了)
    final_scene_path = scene_path_from_env if scene_path_from_env is not None else args.scene
    final_config_path = config_path_from_env if config_path_from_env is not None else args.config

    if not final_scene_path:
        raise ValueError("SCENE path is required either via YAML env or --scene argument.")
    if not final_config_path:
        raise ValueError("CONFIG path is required either via YAML env or --config argument.")
    
    print(f"DEBUG: Using Scene from env/arg: {final_scene_path}")
    print(f"DEBUG: Using Config from env/arg: {final_config_path}")
    print(f"DEBUG: Current working directory for main(): {os.getcwd()}")


    # 构建 bus 字典
    bus = {
        "name": args.name,
        "scene": final_scene_path,
        "config_file_path": final_config_path, # 将配置文件路径传递给 Client
    }
    # --- bus 字典创建结束 ---

    print(f"DEBUG: Mujoco Client Configuration (bus): {bus}", flush=True) # 确保 bus 已定义

    try:
        print(f"DEBUG: Attempting to create Client instance for node: {bus.get('name', 'UNKNOWN_BUS_NAME')}")
        client = Client(bus) 
        print("DEBUG: Client instance created. Calling client.run().")
        client.run()
        print("DEBUG: client.run() completed.")
    except Exception as e:
        print(f"CRITICAL ERROR in so100_mujoco_client main() after bus creation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("DEBUG: so100_mujoco_client main() function EXITED.")


if __name__ == "__main__":
    print(f"DEBUG: so100_mujoco_client script executing __main__ block.")
    main()
    print(f"DEBUG: so100_mujoco_client script __main__ block finished.")
