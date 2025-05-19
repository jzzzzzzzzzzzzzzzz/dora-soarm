# nodes/simple_text_commander/simple_text_commander/main.py
import pyarrow as pa
from dora import Node
import argparse
import os
import sys
import time

def clean_string(input_string:str):
    return input_string.encode('utf-8', 'replace').decode('utf-8')

def send_commands_loop(node: Node, node_id_log_prefix: str):
    print(f"INFO ({node_id_log_prefix}): Ready. Type a command (e.g., 'home', 'pose1', 'exit_commander') and press Enter.")
    while True:
        try:
            user_command_raw = input(f"{node_id_log_prefix}> ") # 直接从当前终端读取
            user_command_cleaned = user_command_raw.strip()

            if not user_command_cleaned:
                continue

            if user_command_cleaned.lower() == 'exit_commander':
                print(f"INFO ({node_id_log_prefix}): 'exit_commander' received. Shutting down this node.")
                # node.send_output("text_command", pa.array(["__COMMANDER_END__"])) # 可选
                break

            print(f"INFO ({node_id_log_prefix}): Sending command: '{user_command_cleaned}'")
            node.send_output("text_command", pa.array([clean_string(user_command_cleaned)]))

        except EOFError:
            print(f"INFO ({node_id_log_prefix}): EOF received (Ctrl+D). Shutting down node.")
            break
        except KeyboardInterrupt:
            print(f"INFO ({node_id_log_prefix}): Keyboard interrupt (Ctrl+C). Shutting down node.")
            break
        except Exception as e:
            print(f"ERROR ({node_id_log_prefix}): An error occurred in input loop: {e}")


def main():
    parser = argparse.ArgumentParser(description="Simple Text Command Sender Node (Standalone)")
    parser.add_argument(
        "--name", type=str, required=False, default="simple_text_commander_node", # 这个名字需要和 YAML 中引用它的 ID 一致
        help="The name of the node in the dataflow to connect to.",
    )
    args = parser.parse_args()

    node_name_to_connect = args.name # 这是它在 Dora 数据流图中的 ID
    node = None
    last_err_str = ""

    print(f"INFO (Standalone Commander): Attempting to connect to Dora dataflow as node ID '{node_name_to_connect}'...")

    # 尝试连接到 Dora Daemon
    while node is None:
        try:
            # 当作为独立进程运行时，Node() 会尝试连接到现有的数据流
            # 它需要知道自己在数据流图中的 ID 是什么，以便正确发送数据
            node = Node(node_name_to_connect)
            print(f"INFO (Standalone Commander): Successfully connected to Dora daemon as '{node_name_to_connect}'.")
        except RuntimeError as err:
            current_err_str = str(err)
            if current_err_str != last_err_str:
                print(f"ERROR (Standalone Commander): Failed to connect to Dora daemon for ID '{node_name_to_connect}': {current_err_str}. Retrying in 1s...")
                last_err_str = current_err_str
            time.sleep(1)
        except Exception as e:
            print(f"FATAL (Standalone Commander): Unexpected error during Node initialization for ID '{node_name_to_connect}': {e}. Exiting.")
            return

    try:
        send_commands_loop(node, node_name_to_connect) # 使用连接的节点名作为日志前缀
    except Exception as e:
        print(f"ERROR (Standalone Commander): Node failed catastrophically: {e}")
    finally:
        print(f"INFO (Standalone Commander): Node '{node_name_to_connect}' has finished.")


if __name__ == "__main__":
    main()