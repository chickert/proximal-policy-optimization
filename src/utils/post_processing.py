from algorithm.ppo import PPOLearner
import pandas as pd
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc
from collections import Counter
import logging

logger = logging.getLogger(__name__)


def save_training_rewards(learner: PPOLearner, path: str) -> None:
    # Save training rewards as csv
    try:
        df = pd.read_csv(f"{path}.csv")
        df[learner.seed] = learner.mean_rewards
    except FileNotFoundError:
        df = pd.DataFrame(learner.mean_rewards, columns=[learner.seed])
        df.index.name = "iteration"
    df.to_csv(f"{path}.csv")


def save_video(learner: PPOLearner, path: str, use_argmax: bool = False) -> None:
    # Save videos of agent as mp4
    learner.environment.robot.cam.setup_camera(
        focus_pt=learner.environment.robot.arm.robot_base_pos,
        dist=3,
        yaw=55,
        pitch=-30,
        roll=0
    )
    file_path = f"{path}.mp4"
    output = VideoWriter(file_path, VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    image = learner.environment.robot.cam.get_images(get_rgb=True, get_depth=False)[0]
    output.write(np.array(image))
    learner.policy = learner.best_policy
    if use_argmax:
        _, actions, _, _ = learner.generate_argmax_trajectory()
    else:
        _, actions, _, _ = learner.generate_sample_trajectory()
    #logger.info(f"Action counts: {dict(Counter(map(tuple, actions)))}")
    learner.environment.reset()
    for action in actions:
        learner.environment.step(action)
        image = learner.environment.robot.cam.get_images(get_rgb=True, get_depth=False)[0]
        output.write(np.array(image))
    output.release()



