#!/usr/bin/env python3
import socket

# ---- CONFIG ----
LOCAL_IP   = "0.0.0.0"   # listen on all interfaces
LOCAL_PORT = 5005
BUF_SIZE   = 4096       # large enough for typical messages
# ----------------

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
sock.bind((LOCAL_IP, LOCAL_PORT))
print(f"Listening for UDP packets on {LOCAL_IP}:{LOCAL_PORT} ...")

while True:
    data, addr = sock.recvfrom(BUF_SIZE)   # blocks until a packet arrives
    print(f"Received from {addr}: {data!r}")
    # If you want to stop on a special message:
    # if data == b"QUIT": break
