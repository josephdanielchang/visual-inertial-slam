# ECE276A Project 3 - Joseph Chang

## Visual-Intertial SLAM

### Directory Structure

ECE276A_PR2_Local/...<br />
├─── hw3_main.py<br />
├─── utils.py<br />
├─── plots/...<br />
├─── data/...<br />
│ ├─── 03.npz/...<br />
│ ├─── 10.npz/...<br />


### Main Code Files
* **hw3_main.py**: use to plot first lidar scan onto an occupancy grid map, uncomment plotting in mapping.py to use
* **utils.py**: main code to run particle filter slam
* **mapping.py**: given lidar scan, finds all occupied and free cells with bresenham2D, updates grid map
* **prediction.py**: calculates new vehicle pose for each particle given linear velocity and change in yaw
* **update.py**: updates particle weights based on which lidar and pose have highest correlation
* **resampling.py**: creates new set of particles
* **texture_map.py**: projects rgb image to map using disparity
* **pr2_utils.py**: methods for read_data_from_csv, mapCorrelation, bresenham2D, compute_stereo
* **fog_delta.npy**: fog data synced to encoder timestamps, sync code commented out in slam.py

### Main Code Files
* **hw3_main.py**: main code to run visual inertial slam through EKF Prediction, EKF Update
* **utils.py**: methods for load_data, visualize_trajectory_2d
* **10.npz**: synchronized imu data, stereo camera intrinsic and extrinsic calibration

### Other Folders
* **data**: contains 10.npz
* **plot**: save maps of trajectory and landmarks downsampled by various factors

### How to Run

Download modules in requirements.txt <br />
Run hw3_main.py <br />
Displays map after iterating through all landmarks

