#!/usr/bin/env python3
import msgpack
import msgpack_numpy as m
import lz4.frame

m.patch()  # enable msgpack to handle numpy arrays
import numpy as np
import zmq
from metavision_sdk_stream import Camera, CameraStreamSlicer, SliceCondition

PC_IP = "130.83.164.80"
PORT = 5555
# accumulation time: X320 supports 10k fps = 100 us; 1 ms = 1000 us
ACCUMULATION_TIME_US = 1000 

# set up ZMQ publisher socket
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.setsockopt(zmq.SNDHWM, 1)  # keep only 1 message in queue
socket.connect(f"tcp://{PC_IP}:{PORT}")

print(f"Connected to tcp://{PC_IP}:{PORT}")

camera = Camera.from_first_available()  # initialize the camera
slice_condition = SliceCondition.make_n_us(ACCUMULATION_TIME_US)
slicer = CameraStreamSlicer(camera.move(), slice_condition=slice_condition)  # create a slicer to get slices of events

try:
    for slice in slicer:
        if slice.events.size > 0:
            raw_data = slice.events.tobytes()
            compressed_payload = lz4.frame.compress(raw_data, compression_level=0)
            socket.send(compressed_payload)
except KeyboardInterrupt:
    pass
