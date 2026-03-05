docker run -it --privileged --runtime=nvidia --gpus=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $XAUTHORITY:/root/.Xauthority \
  -v /run/user/$(id -u)/pulse:/run/user/0/pulse \
  -v ~/.config/pulse/cookie:/root/.config/pulse/cookie \
  -v /dev:/dev \
  -e DISPLAY=$DISPLAY \
  -e PULSE_SERVER=unix:/run/user/0/pulse/native \
  --net host --rm --name flightmare flightmare
