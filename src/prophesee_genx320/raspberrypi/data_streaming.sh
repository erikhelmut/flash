#!/usr/bin/env bash

cd rpi-sensor-drivers/

sudo dtoverlay genx320
sleep 1
sudo dtoverlay genx320,cam0
sleep 1

./rp5_setup_v4l.sh

cd ..
