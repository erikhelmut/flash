#!/usr/bin/env python3
import numpy as np
import cv2
import msgpack
import msgpack_numpy as m
m.patch()
import socket
import struct
from collections import defaultdict

PC_IP =  "10.42.0.100" #"130.83.164.80"  # Sender IP (optional: filter incoming)
PORT = 5555
UDP_BUFFER_SIZE = 65535  # Max UDP packet size

# Raw UDP receiver socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8*1024*1024)  # 8 MiB receive buffer for high rate
sock.bind(("", PORT))  # Bind to all interfaces on port 5555
print(f"Listening for raw UDP on port {PORT} (from {PC_IP} if specified)")

# Buffer for reassembly: slice_idx → {'total': int, 'chunks': list of bytes}
buffers = defaultdict(lambda: {'total': 0, 'chunks': []})

try:
    while True:
        data, addr = sock.recvfrom(UDP_BUFFER_SIZE)
        if PC_IP and addr[0] != PC_IP:
            continue  # Ignore non-matching senders

        # Parse 12-byte header: seq (u64) + total (u16) + idx (u16)
        if len(data) < 12:
            print(f"Invalid packet from {addr}: too short ({len(data)} bytes)")
            continue
        seq, total, idx = struct.unpack(">QHH", data[:12])
        payload = data[12:]

        buf = buffers[seq]
        if buf['total'] == 0:
            buf['total'] = total
            buf['chunks'] = [None] * total
        buf['chunks'][idx] = payload

        # Check if all chunks arrived
        if all(c is not None for c in buf['chunks']):
            full_payload = b''.join(buf['chunks'])
            try:
                events = msgpack.unpackb(full_payload, ext_hook=m.decode)
                print(f"Reassembled slice {seq}: {len(events)} events from {addr} ({total} packets)")
                # TODO: Process events here (e.g., visualize, save, analyze)
            except Exception as e:
                print(f"Failed to unpack slice {seq}: {e}")
            del buffers[seq]  # Free memory

        # Optional: Cleanup old incomplete buffers (e.g., after 10s timeout)
        # import time; now = time.time(); ... if now - buf.get('ts', now) > 10: del buffers[seq]

        # visualize the events as an image
        img = np.zeros((320, 320), dtype=np.uint8)
        for x, y, p in zip(events["x"], events["y"], events["p"]):
            img[y, x] = 255 if p else 128  # white for ON, gray for OFF

        # make big window
        img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Events", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


except KeyboardInterrupt:
    print("\nStopped by user")
finally:
    sock.close()