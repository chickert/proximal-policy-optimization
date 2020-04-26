
import numpy as np
import pybullet as p
from airobot import Robot
from airobot.utils.common import euler2quat
from airobot.utils.common import quat_multiply
from airobot.utils.common import rotvec2quat
from gym import spaces


class PusherSimulator:

    def __init__(self, action_repeat=10, render=False):
        self._action_repeat = action_repeat
        self.spec = None
        self.robot = Robot('ur5e_stick', pb=True, pb_cfg={'gui': render, 'realtime': False})
        self.ee_ori = [-np.sqrt(2) / 2, np.sqrt(2) / 2, 0, 0]
        self._action_bound = 1.0
        self._ee_pos_scale = 0.02
        self._ee_ori_scale = np.pi / 36.0
        self._action_high = np.array([self._action_bound] * 2)
        self.action_space = spaces.Box(low=-self._action_high, high=self._action_high, dtype=np.float32)
        self.goal = np.array([0.75, -0.3, 1.0])
        self.init = np.array([0.5, 0.3, 1.0])
        self.init_obj = np.array([0.5, 0.2, 1.0])
        self.push_len = 0.5
        self.push_ang_range = np.pi / 3
        self.robot.arm.reset()

        ori = euler2quat([0, 0, np.pi / 2])
        self.table_id = self.robot.pb_client.load_urdf("table/table.urdf", [.5, 0, 0.4], ori, scaling=0.9)
        self.obj_id = self.robot.pb_client.load_geom("cylinder", size=[0.08, 0.05], mass=1, base_pos=self.init_obj,
                                                     rgba=[1, 0, 0, 1])
        self.marker_id = self.robot.pb_client.load_geom("box", size=0.05, mass=1, base_pos=self.goal,
                                                        rgba=[0, 1, 0, 0.4])
        client_id = self.robot.pb_client.get_client_id()

        p.setCollisionFilterGroupMask(self.marker_id, -1, 0, 0, physicsClientId=client_id)
        p.setCollisionFilterPair(self.marker_id, self.table_id, -1, -1, 1, physicsClientId=client_id)

        self.reset()
        state_low = np.full(len(self.get_obs()), -np.inf)
        state_high = np.full(len(self.get_obs()), np.inf)
        self.observation_space = spaces.Box(state_low, state_high, dtype=np.float32)

    def place_marker(self):
        self.robot.pb_client.reset_body(self.marker_id, self.goal)

    def reset(self):
        push_ang = 1.5 * np.pi + (self.push_ang_range * np.random.random() - self.push_ang_range / 2.0)
        self.goal = self.init_obj + np.array([self.push_len * np.cos(push_ang), self.push_len * np.sin(push_ang), 0.0])
        self.place_marker()

        self.robot.arm.go_home(ignore_physics=True)
        jnt_pos = self.robot.arm.compute_ik(self.init)
        self.robot.arm.set_jpos(jnt_pos, ignore_physics=True)
        self.robot.pb_client.reset_body(self.obj_id, self.init_obj)

        self.ref_ee_ori = self.robot.arm.get_ee_pose()[1]
        self.gripper_ori = 0
        self.timestep = 0
        return self.get_obs()

    def step(self, action):
        self.apply_action(action)
        state = self.get_obs()
        self.timestep += 1
        done = (self.timestep >= 200)
        info = dict()
        reward, dist = self.compute_reward_push(state)
        if done:
            info['final_distance'] = dist
        return state, reward, done, info

    def compute_reward_push(self, state):
        c1 = 1.0
        c2 = 0.01
        c3 = 0.001
        reach_dist = np.linalg.norm(state[:3] - state[3:6])
        reach_rew = -reach_dist
        push_dist = np.linalg.norm(state[3:6] - state[6:9])
        push_rew = c1 * (self.push_len - push_dist) + c1 * (
                np.exp(-(push_dist ** 2) / c2) + np.exp(-(push_dist ** 2) / c3))
        push_rew = max(push_rew, 0)
        reward = reach_rew + push_rew
        return reward

    def get_obs(self):
        gripper_pos = self.robot.arm.get_ee_pose()[0]
        obj_pos = self.robot.pb_client.get_body_state(self.obj_id)[0]
        state = np.concatenate([gripper_pos, obj_pos, self.goal])
        return state

    def apply_action(self, action):
        if not isinstance(action, np.ndarray):
            action = np.array(action).flatten()
        if action.size != 2:
            raise ValueError('Action should be [d_x, d_y].')

        action = np.concatenate([action, np.array([0.])])
        pos, quat, rot_mat, euler = self.robot.arm.get_ee_pose()
        pos += action[:3] * self._ee_pos_scale

        rot_vec = np.array([0, 0, 1]) * self.gripper_ori
        rot_quat = rotvec2quat(rot_vec)
        ee_ori = quat_multiply(self.ref_ee_ori, rot_quat)
        jnt_pos = self.robot.arm.compute_ik(pos, ori=ee_ori)

        for step in range(self._action_repeat):
            self.robot.arm.set_jpos(jnt_pos)
            self.robot.pb_client.stepSimulation()

    def render(self, **kwargs):
        robot_base = self.robot.arm.robot_base_pos
        self.robot.cam.setup_camera(focus_pt=robot_base, dist=3, yaw=55, pitch=-30, roll=0)

        rgb, _ = self.robot.cam.get_images(get_rgb=True, get_depth=False)
        return rgb

    def close(self):
        return
