# nodes/manual_commander/main.py
import pyarrow as pa
import numpy as np
from dora import Node

# POSES 和 NUM_JOINTS 定义保持不变
POSES = {
    "home": np.array([0.0, -1.57079, 1.57079, 1.57079, -1.57079, 0.0]),
    "zero": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "pose1": np.array([0.5, -0.5, 0.8, 0.2, 0.0, 0.1]),
    "pose2": np.array([-0.5, -1.0, 1.2, -0.5, 0.5, 0.0]),
}
NUM_JOINTS = 6

def main():
    node = Node()
    node_id_log_prefix = "manual_commander"
    print(f"INFO (Node {node_id_log_prefix}): Manual Commander Ready. Waiting for commands from 'simple_text_commander'.")
    print(f"Available poses: {list(POSES.keys())}")

    for event in node:
        if event["type"] == "INPUT":
            event_id = event["id"]
            # --- 修改这里，监听新的输入 ID ---
            if event_id == "received_text_command": # 这个 ID 对应 YAML 中的输入名
            # --- 修改结束 ---
                try:
                    if len(event["value"]) > 0:
                       pose_name = event["value"][0].as_py().strip().lower() # 获取命令，并清理
                       if pose_name in POSES:
                           goal_joints = POSES[pose_name]
                           if len(goal_joints) == NUM_JOINTS:
                               print(f"INFO (Node {node_id_log_prefix}): Processing command '{pose_name}'. Sending goal pose: {goal_joints}")
                               node.send_output("goal_position", pa.array(goal_joints), event["metadata"])
                           else:
                               print(f"WARNING (Node {node_id_log_prefix}): Pose '{pose_name}' has wrong number of joints.")
                       # 你可以添加一个对 "__END__" 的处理，如果 simple_text_commander 发送它
                       # elif pose_name == "__end__":
                       #     print(f"INFO (Node {node_id_log_prefix}): Received end command.")
                       #     # 可能需要通知 mujoco_client 结束
                       #     break
                       else:
                           print(f"WARNING (Node {node_id_log_prefix}): Unknown pose name: '{pose_name}'. Available: {list(POSES.keys())}")
                    else:
                       print(f"WARNING (Node {node_id_log_prefix}): Received empty command input.")
                except Exception as e:
                    print(f"ERROR (Node {node_id_log_prefix}): Error processing command: {e}")

        elif event["type"] == "STOP":
            print(f"INFO (Node {node_id_log_prefix}): Received STOP signal.")
            break
        elif event["type"] == "ERROR":
            print(f"ERROR (Node {node_id_log_prefix}): Received error: {event['error']}")
            break
    print(f"INFO (Node {node_id_log_prefix}): Exiting.")

if __name__ == "__main__":
    main()