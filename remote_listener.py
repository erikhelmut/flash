import subprocess
import signal
import os
from pynput import keyboard
import argparse

RECORD_SCRIPT = "record_rosbag.bash"
task_name = ""
process = None

def start_recording():
    global process, task_name
    if process is None:
        print(f"\n🚀 STARTING RECORDING: {task_name}...")
        process = subprocess.Popen(["bash", RECORD_SCRIPT, task_name], preexec_fn=os.setsid)
    else:
        print("\n⚠️ Already recording.")

def stop_recording():
    global process
    if process is not None:
        print("\n🛑 STOPPING AND SAVING...")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process = None
        print("✅ Saved.")
    else:
        print("\nℹ️ Nothing is running.")

def on_press(key):
    try:
        if key == keyboard.Key.page_down or key == keyboard.Key.right:
            start_recording()
        elif key == keyboard.Key.page_up or key == keyboard.Key.left:
            stop_recording()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UMI Remote Listener for ROS Bag Recording")
    parser.add_argument("--task", type=str, required=True, help="Task name for organizing recordings")
    args = parser.parse_args()

    task_name = args.task

    print(f"🎮 UMI Remote Listener Active | Task: {task_name}")
    print("Buttons: NEXT/RIGHT = Start | BACK/LEFT = Stop")
    print("---------------------------------------------")

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()