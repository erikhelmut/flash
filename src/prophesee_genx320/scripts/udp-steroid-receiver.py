#!/usr/bin/env python3
# receiver_fragmented.py
import socket
import struct
import numpy as np
import cv2

PORT = 5555
MAX_BUF = 70_000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32 * 1024 * 1024)
sock.bind(("", PORT))

# Temporary storage for incomplete slices
pending = {}   # key = seq → (num_events, frag_total, received fragments dict)

events_dtype = np.dtype([('x', np.uint16), ('y', np.uint16), ('p', np.uint8), ('t', np.uint32)])

print("Receiving fragmented events...")

while True:
    data, addr = sock.recvfrom(MAX_BUF)

    if len(data) < 12:
        continue

    seq, num_events, frag_idx, frag_total = struct.unpack("!IIHH", data[:12])
    payload = data[12:]

    # Init entry if first fragment
    if seq not in pending:
        pending[seq] = ({}, frag_total, num_events)

    fragments, expected, _ = pending[seq]
    fragments[frag_idx] = payload

    # When all fragments arrived → reconstruct
    if len(fragments) == expected:
        full_payload = b"".join(fragments[i] for i in range(expected))
        events = np.frombuffer(full_payload, dtype=events_dtype, count=num_events)

        print(f"Reconstructed slice {seq}: {num_events:,} events")

        # Clean up old entries older than 1000 to prevent memory leak
        old = [k for k in pending if k < seq - 1000]
        for k in old:
            del pending[k]

        del pending[seq]

        # ← put your processing here (visualisation, ML inference, saving, etc.)
        img = np.zeros((320, 320), dtype=np.uint8)
        for x, y, p in zip(events["x"], events["y"], events["p"]):
            img[y, x] = 255 if p else 128  # white for ON, gray for OFF

        # make big window
        img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Events", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break