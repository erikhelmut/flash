#!/usr/bin/env python3
import numpy as np
import cv2
import msgpack
import msgpack_numpy as m
m.patch()
import socket
import struct
from collections import defaultdict
import threading
import queue
import time

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
PC_IP = "10.42.0.100"        # Set to None or "" to accept from any IP
PORT = 5555
UDP_BUFFER_SIZE = 65535

# We only ever keep the very latest frame → zero lag, zero backlog
latest_frame_queue = queue.Queue(maxsize=5)

# ------------------------------------------------------------------
# UDP Receiver Thread (fast, robust, no crashes)
# ------------------------------------------------------------------
def receiver_thread():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
    sock.bind(("", PORT))
    print(f"UDP receiver listening on port {PORT}")

    # One buffer per sequence number
    buffers = defaultdict(lambda: {'total': 0, 'chunks': None, 'received': 0})

    while True:
        data, addr = sock.recvfrom(UDP_BUFFER_SIZE)

        if PC_IP and addr[0] != PC_IP:
            continue
        if len(data) < 12:
            continue

        seq, total, idx = struct.unpack(">QHH", data[:12])
        payload = data[12:]

        buf = buffers[seq]

        # First packet of this sequence → initialize
        if buf['total'] == 0:
            buf['total'] = total
            buf['chunks'] = [None] * total
            buf['received'] = 0

        if idx >= total:
            continue

        # Store only if missing (handles duplicates perfectly)
        if buf['chunks'][idx] is None:
            buf['chunks'][idx] = payload
            buf['received'] += 1

            if buf['received'] == total:
                # All chunks arrived → reconstruct
                try:
                    full_payload = b''.join(buf['chunks'])
                    events = msgpack.unpackb(full_payload, ext_hook=m.decode)

                    # Drop old frame if present → always show newest
                    try:
                        latest_frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    latest_frame_queue.put_nowait(events)

                except Exception as e:
                    print(f"Unpack error (seq {seq}): {e}")
                finally:
                    del buffers[seq]  # free memory immediately

# ------------------------------------------------------------------
# Visualization Thread – 100% vectorized, ultra fast
# ------------------------------------------------------------------
def visualization_thread():
    cv2.namedWindow("GenX320 Events", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("GenX320 Events", 960, 960)

    # Persistent image buffer
    img = np.zeros((320, 320), dtype=np.uint8)

    print("Visualization started – press 'q' to quit")

    while True:
        try:
            events = latest_frame_queue.get(timeout=0.01)
        except queue.Empty:
            time.sleep(0.001)
            continue

        # ───── THIS IS THE ONLY IMPORTANT LINE (vectorized magic) ─────
        img[:] = 0
        img[events["y"], events["x"]] = np.where(events["p"], 255, 128)
        # ─────────────────────────────────────────────────────────────────

        # Upscale and display
        display = cv2.resize(img, (960, 960), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("GenX320 Events", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Visualization stopped")

# ------------------------------------------------------------------
# Start everything
# ------------------------------------------------------------------
if __name__ == "__main__":
    threading.Thread(target=receiver_thread, daemon=True).start()
    visualization_thread()