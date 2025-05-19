# nodes/so100_mujoco_client/so100_mujoco_client/main.py (临时测试版本)
print(f"DEBUG: SKELETON so100_mujoco_client/main.py SCRIPT STARTED at {__file__}")
import time
try:
    from dora import Node
    print("DEBUG: SKELETON - dora.Node imported successfully")
except ImportError as e:
    print(f"CRITICAL ERROR: SKELETON - Failed to import dora.Node: {e}")
    # 通常 Dora 会捕获这个，但以防万一
    raise SystemExit(f"CRITICAL ERROR: SKELETON - Failed to import dora.Node: {e}")

def main():
    node_name = "so100_mujoco_client_SKELETON" # 给一个明确的测试名
    print(f"DEBUG: SKELETON - main() function CALLED for node: {node_name}")
    try:
        node = Node(node_name)
        print(f"DEBUG: SKELETON - Node instance created for {node_name}.")

        running = True
        tick_count = 0
        for event in node: # 进入 Dora 事件循环
            print(f"DEBUG: SKELETON - Received event: {event}")
            event_type = event["type"]
            if event_type == "INPUT":
                event_id = event["id"]
                if event_id == "tick":
                    tick_count += 1
                    print(f"DEBUG: SKELETON - Tick {tick_count} received by {node_name}.")
                    if tick_count > 10: # 运行一段时间后自动停止
                        print("DEBUG: SKELETON - Reached 10 ticks, stopping.")
                        running = False
                        break
            elif event_type == "STOP":
                print(f"DEBUG: SKELETON - Received STOP signal from Dora for {node_name}.")
                running = False
                break
            elif event_type == "ERROR":
                print(f"DEBUG: SKELETON - Received ERROR signal from Dora for {node_name}: {event['error']}")
                running = False
                break
            
            if not running:
                break
        
        print(f"DEBUG: SKELETON - Event loop finished for {node_name}.")

    except Exception as e:
        print(f"CRITICAL ERROR in SKELETON main() for {node_name}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"DEBUG: SKELETON - main() function EXITED for {node_name}.")

if __name__ == "__main__":
    print(f"DEBUG: SKELETON script executing __main__ block.")
    main()
    print(f"DEBUG: SKELETON script __main__ block finished.")
else:
    print(f"DEBUG: SKELETON script imported, __name__ is {__name__}")