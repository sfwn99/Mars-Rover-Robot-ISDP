# Integrated System Design Project - Mars Rover 

In this project, we are required to built a Mars Rover that supposed to replicate Mars Exploration Rover (MER) built by NASA. 
The robot must generally able to :
1. Maneuver around the gamefield and avoid obstacle
2. Detect objects ( Corn or Tomato )
3. Approach the objects and grip it
4. Deliver data from the robot such as battery level and status, and gripped objects counter to online platfrom (Blynk)
5. Able to remotely turn off and on the whole system

However, the robot built by our team does not meet the full requirement in the project. 
My team only able to achieve from number 1 to 4 from the list of mentioned tasks above.

I am sharing steps involved in building this rover to you, incase it will be useful for your own hobby or assignment in the future. 

# Github references

1. [Tensorflow Lite Object Detection on Android and Raspberry Pi by EdjeElectronics](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi)
2. [Object Tracking Ai Robot by jiteshaini](https://github.com/jiteshsaini/robotics-level-4/tree/main/earthrover/object_tracking)

Repositories listed above has help me to build the mars rover to meet the minimum requirement of the project.

# Softwares and Hardwares

The rover system is divided into few subsystems :

1. Object Detection
2. Autonomous Movement and Obstacle Avoidance
3. Communication
4. Gripper
5. Power Management

Listed below are hardwares used in each subsystem :

| Object Detection | Autonomous Movement | Communication | Gripper | 
| ---------------- | ------------------- | ------------- | ------- |
| Raspberry Pi 4   | 4 DC Motors         | Arduino Wemos | SG90    |
| Raspberry Pi Cam | 4 Wheel Chassis     | Current Meter | Gripper |
|                  | L298N Motor Driver  | Voltage Meter |         |







