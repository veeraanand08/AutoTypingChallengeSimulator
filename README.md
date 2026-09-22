# URC autonomous-typing challenge simulator

A ROS 2 (Jazzy) simulator of the URC autonomous-typing task. A five-joint,
velocity-controlled arm with a forearm camera must find a keyboard on an
ArUco-marked panel and type a launch key. The simulator publishes camera
images, joint states and TF; you publish joint velocities and presses; a
browser dashboard lets a person watch the run. It runs in Docker on an
Ubuntu 24.04 machine.

## Run Sim Steps

In the Terminal…

cd AutoTypingChallengeSim

docker compose down
docker compose up -d

docker compose exec dev bash

cd /ws
colcon build --symlink-install
source install/setup.bash

ros2 launch autotyper typist.launch.py

ros2 service call /sim/reset std_srvs/srv/Trigger
