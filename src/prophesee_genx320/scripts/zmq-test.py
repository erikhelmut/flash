#!/usr/bin/env python3
from logging import lastResort
import zmq
import msgpack
import msgpack_numpy as m
m.patch()
import numpy as np
import cv2
import time
from collections import deque

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.bind(f"tcp://*:{5555}")
socket.setsockopt(zmq.SUBSCRIBE, b"")  # subscribe to all

socket.setsockopt(zmq.CONFLATE, 1)   # if ZMQ ≥ 4.2

print("ZMQ SUB listening on :5555...")

msg_count = 0
start_time = time.monotonic()          # monotonic is better than time.time()
freq_history = deque(maxlen=30)        # for smooth display
while True:
    payload = socket.recv()  # ← auto-reassembled
    msg_count += 1
    events = msgpack.unpackb(payload, raw=False)
    #print(f"Received {len(events)} events "
          #f"| t ∈ [{events['t'].min()}, {events['t'].max()}] µs")
    #print(1 / ((events["t"].max() - events["t"].min()) / 1e6))
    
    # Update frequency every ~0.2–0.5 seconds (you can adjust)
    now = time.monotonic()
    elapsed = now - start_time
    
    if elapsed >= 0.25:  # update ~4 times per second
        current_freq = msg_count / elapsed
        freq_history.append(current_freq)
        
        # Smooth frequency (optional but looks much nicer)
        smooth_freq = sum(freq_history) / len(freq_history)
        
        print(f"\rPackage receive frequency: {smooth_freq:6.2f} Hz  "
                f"(instant: {current_freq:6.2f} Hz)  |  "
                f"{len(events):5} events this packet     ", end="")
        
        # Reset counters
        msg_count = 0
        start_time = now 
    
    
    # visualize the events as an image
    img = np.zeros((320, 320), dtype=np.uint8)
    for x, y, p in zip(events["x"], events["y"], events["p"]):
        img[y, x] = 255 if p else 128  # white for ON, gray for OFF
    
    # make big window
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Events", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
