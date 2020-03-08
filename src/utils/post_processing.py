from ppo_alogrithm import PPOLearner
import pandas as pd
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc


def save_training_rewards(learner: PPOLearner, path: str) -> None:
    # Save training rewards as csv
    pd.DataFrame(learner.mean_rewards).to_csv(f"{path}.csv")


def save_videos(learner: PPOLearner, path: str, n_videos: int = 1) -> None:
    # Save videos of agent as mp4

    robot = learner.environment.robot
    robot.cam.setup_camera(focus_pt=robot.arm.robot_base_pos, dist=3, yaw=55, pitch=-30, roll=0)
    for i in range(n_videos):
        file_path = f"{path}_{i+1}.mp4"
        output = VideoWriter(file_path, VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
        image = robot.cam.get_images(get_rgb=True, get_depth=False)[0]
        output.write(np.array(image))
        states, _, _ = learner.generate_trajectory()
        for state in states:
            print(robot.arm.set_ee_pose(state, ignore_phyics=True))
            image = robot.cam.get_images(get_rgb=True, get_depth=False)[0]
            output.write(np.array(image))
        output.release()



