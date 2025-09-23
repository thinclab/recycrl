# recycrl
Author: Jacob Kruse

Currently Managed By: Jacob Kruse (jacob-kruse@outlook.com).

This Repository contains the Global Policy, Neural Network, and Reinforcement Learning Code for the Recycling Project.

#### Table of Contents
[1) Install Anaconda](#1-Install-Anaconda)<br>
[2) Install ROS2 and create a ROS2 workspace](#2-Install-ROS2-and-create-a-ROS2-workspace)<br>
[3) Clone `recycrl` into your ROS2 Workspace](#3-Clone-recycrl-into-your-ROS2-Workspace)<br>
[4) Create a Python environment with the `environment.yml`](#4-Create-a-Python-environment-with-the-environmentyml)<br>
[5) Build the `recycrl` package](#5-Build-the-recycrl-package)<br>
[6) Install the Oak-D S2 Camera Driver](#6-Install-the-Oak-D-S2-Camera-Driver)<br>
[7) Install `kuka_kontrol` and KUKA Dependencies](#7-Install-kuka_kontrol-and-KUKA-Dependencies)<br>
[8) Run `recycrl`](#8-Run-recycrl)<br>

##### Note: The following instructions are written for Ubuntu 24.04, ROS2 Jazzy Jalisco

## 1) Install Anaconda

With the transition to virtual python environments in newer versions of Ubuntu, an Anaconda environment is used in this project. If you need to install Anaconda, check out the link below.<br>[Anaconda](https://www.anaconda.com/download)

After signing up, you should receive a download link for Anaconda. Download the Linux installer to your machine. Give execute permission to the installation script and execute the script using the following commands.

    cd ~/Downloads
    chmod +x Anaconda3-2024.10-1-Linux-x86_64.sh
    ./Anaconda3-2024.10-1-Linux-x86_64.sh

##### Note: The download location and file name might be different, replace the commands accordingly

Before the installation process completes, Anaconda's installer will ask if you would like Anaconda and the `conda` command to be initialized automatically in your shell by editing your `~/.bashrc`. It is recommended to answer "yes" for convenience. However, every time you open a new terminal, the Anaconda base environment will automatically activate. Use the following command to turn this off.

    conda config --set auto_activate_base false

## 2) Install ROS2 and create a ROS2 workspace

The instructions at the link below describe how to install ROS2 Jazzy Jalisco on your machine.<br>
[ROS2 Jazzy Jalisco](https://docs.ros.org/en/jazzy/Installation.html)

After installing ROS2, go to the following link to create a ROS2 Workspace. It is a good idea to have a dedicated workspace for this package that you can build independently within the Anaconda environment. Conventionally this workspace is named `ros2_ws`, but you can name this workspace whatever you like. In this tutorial, we will use `recycle_ws`, so make sure to change certain commands accordingly.<br>
[ROS2 Workspace](https://docs.ros.org/en/jazzy/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html)

## 3) Clone `recycrl` into your ROS2 Workspace

Use the following commands to clone this package into your `~/recycle_ws/src` folder and resolve dependencies.

    cd ~/recycle_ws/src
    git clone -b jazzy https://github.com/thinclab/recycrl
    cd ..
    rosdep install --from-paths src --ignore-src -r -y

##### Note: Do not use `colcon build` yet.

## 4) Create a Python environment with the `environment.yml`

At this time, this repository is still in development. There is currently no Python environment provided.

<!-- After you have cloned the `recycrl` package, use the following commands to go to the package's `/anaconda_env` directory and create the required Python environment specified in the `environment.yml`. Before doing so, feel free to change the name of environment at the top of the `environment.yml` file.

cd ~/recycle_ws/src/recycrl/anaconda_env
conda env create -f environment.yml

Use the next command to activate the environment after installation. Replace `ros2_env` in the next couple of commands if you changed your environment name.

conda activate ros2_env

Make sure to deactivate your Python environment whenever necessary using the following command.

conda deactivate

You can also add alias scripts into your `~/.bashrc` to easily activate and deactivate your environment. Use the commands below to do this. You can now use `py` and `de` in your terminal to activate or deactivate the environment, respectively.

echo 'alias py="conda activate ros2_env"' >> ~/.bashrc
echo 'alias de="conda deactivate"' >> ~/.bashrc
source ~/.bashrc

After setting up your environment, use the following commands to add the `activate_env_vars.sh` and `deactivate_env_vars.sh` scripts to your Anaconda environment's `/activate.d` and `/deactivate.d` folders. These scripts execute automatically when the environment is activated or deactivated. These scripts set the `$LD_LIBRARY_PATH` so that the Anaconda environment's library versions are prioritized in the terminal session when the Anaconda environment is activated. This helps avoid issues when building and executing the scripts.

cd ~/recycle_ws/src/recycrl/anaconda_env
mv activate_env_vars.sh ~/anaconda3/envs/ros2_env/etc/conda/activate.d/
mv deactivate_env_vars.sh ~/anaconda3/envs/ros2_env/etc/conda/deactivate.d/ -->

## 5) Build the `recycrl` package

With the previous steps complete, the package is ready to be built. Use the following commands to go to your workspace and build the package.

    cd ~/recycle_ws
    colcon build

Feel free to replace the `colcon build` above with the following to avoid compiling when changes are made to any Python files.

    colcon build --symlink-install

<!-- With the previous steps complete, the package is ready to be built. Use the following commands to go to your workspace, activate your environment, and build the package. It is important to activate your environment before building the first time.

cd ~/recycle_ws
conda activate ros2_env
colcon build

Feel free to replace the `colcon build` above with the following to avoid compiling when changes are made to any Python files.

colcon build --symlink-install

If you accidentally built the package without first activating the environment, you will get this error when you build inside the environment thereafter:

/usr/bin/cmake: /home/psuresh/anaconda3/envs/ros2_env/lib/libcurl.so.4: no version information available (required by /usr/bin/cmake)

This will not lead to any errors, but it can be annoying to see this. Remove the `/build` and `/install` folders in your `/recycle_ws` and build again in the Python environment to fix this.

cd ~/recycle_ws
rm -rf build install
conda activate ros2_env
colcon build

After the initial build in the environment, you can rebuild this package inside or outside of your Python environment without issue.  -->

Make sure to source your workspace. You can add the following lines to your `~/.bashrc` with the commands below so that this happens automatically.

    echo "source ~/recycle_ws/install/setup.bash" >> ~/.bashrc
    echo "source ~/recycle_ws/install/local_setup.bash" >> ~/.bashrc

## 6) Install the Oak-D S2 Camera Driver

Currently, this package implements the Oak-D S2 Camera. The DepthAI-ROS GitHub Repository and Documentation can be found at the links below. <br>
[Depth-AI ROS GitHub](https://github.com/luxonis/depthai-ros.git)<br>
[Depth-AI ROS Documentation](https://docs.luxonis.com/software/depthai/manual-install/)

You can download and install our fork of this driver to emulate our setup. The fork contains the locations of the camera in the current lab setup, a topic remapping for the robot state publisher to avoid conflicts, and `udev` rules to give the camera read/write permissions to your machine. You can also implement the original driver without our changes. Installation instructions for the Oak-D S2 Forked Driver can be found at the link below.<br>
[Depth-AI ROS Fork Installation](https://github.com/thinclab/depthai-ros?tab=readme-ov-file#Installation)

More information on the fork and what it includes can be found in the Setup section.<br>
[Depth-AI ROS Fork Setup](https://github.com/thinclab/depthai-ros?tab=readme-ov-file#Setup)

## 7) Install `kuka_kontrol` and KUKA Dependencies

This package also implements the KUKA LBR iisy r3760 Collaborative Robot. This robot needs to be controlled externally with ROS2 for the code in this repository. A custom package for this purpose has been developed and is called `kuka_kontrol`. Installation instructions for this package can be found at the link below.<br>
[`kuka_kontrol` Installation](https://github.com/thinclab/kuka_kontroltab=readme-ov-file#note-the-following-instructions-are-written-for-ubuntu-2404-ros2-jazzy-jalisco)

## 8) Run `recycrl`

After all of the previous steps have been completed, the package is ready to be utilized. First, the KUKA robot needs to be turned on and set up to allow for external execution. Startup the robot by clicking the power button on the KR C5 micro controller and sign in as an Admin to the SmartPad when it has finished loading. After signing in, make sure the robot is in AUT mode (check top center, There are two modes T1 and AUT) on the SmartPad. Then click on the user icon on the top right and click "Release SPOC" to allow other hosts to control the robot.

If the robot is setup correctly according to the steps above, you can now connect to the robot through ROS2. First, run the robot launch below which starts the required nodes for real-world KUKA operation.

    ros2 launch kuka_irl_project robot.launch.py

You can then configure and activate the robot using the commands below in a separate terminal. Make sure that the launch above is still active and running.

    ros2 lifecycle set robot_manager configure
    ros2 lifecycle set robot_manager activate

After running the "configure" command, you should see a message that says "Transitioning successful". If you do not, restart the robot launch and try again. If you still have an error, something may not be setup correctly. When you finally have a successful transition, you can then use the "activate" command. This should also return a "Transitioning successful", and you should hear a click from the robot which is the brakes releasing. At this point the computer has been connected to the robot and external motion planning is now possible.

If you are having trouble connecting to the robot, please see the `kuka_kontrol` repository for more detailed instructions.<br>
[`kuka_kontrol` Execution](https://github.com/thinclab/kuka_kontrol?tab=readme-ov-file#real-execution)

Then, start the Oak-D S2 Camera driver with the following launch command.

    ros2 launch depthai_ros_driver camera.launch.py

If it is successful, you should see a message like this,

    [component_container-3] [INFO] [1742215114.606896223] [oak]: Camera ready!

##### Note: At this point the ROS2 Nodes are still in development
