import sapien
import torch
import numpy as np
from typing import Dict, Any, Union, List
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.structs import Pose
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.sensors.camera import CameraConfig

# Set some constants
PAPER_SIZE = [0.15, 0.1, 0.001]     # half-sizes for x, y, z(m), the full size is 30cm * 20cm * 0.2cm
BOOK_SIZE = [0.10, 0.075, 0.01]     # half-sizes for x, y, z(m), the full size is 20cm * 15cm * 2cm
BOOK1_NOTMOVE_SIZE = [0.07, 0.01, 0.1]     # half-sizes for x, y, z(m), the full size is 14cm * 20cm * 2cm, the book is placed upright and not move
BOOK2_NOTMOVE_SIZE = [0.05, 0.01, 0.1]     # half-sizes for x, y, z(m), the full size is 10cm * 20cm * 2cm
BOOK3_NOTMOVE_SIZE = [0.09, 0.01, 0.15]      # half-sizes for x, y, z(m), the full size is 18c·m * 30cm * 2cm
SHELF_BOTTOM_SIZE = [0.10, 0.35, 0.01]  # half-sizes for x, y, z(m), the full size is 20cm * 70cm * 2cm
SHELF_BACK_SIZE = [0.01, 0.37, 0.075]     # half-sizes for x, y, z(m), the full size is 2cm * 74cm * 15cm    
SHELF_LEFT_SIZE = [0.10, 0.01, 0.075]     # half-sizes for x, y, z(m), the full size is 20cm * 2cm * 15cm
SHELF_RIGHT_SIZE = [0.10, 0.01, 0.075]    # half-sizes for x, y, z(m), the full size is 20cm * 2cm * 15cm
SEAL_SIZE = [0.01, 0.01, 0.02]      # half-sizes for x, y, z(m), the full size is 2cm * 2cm * 4cm
INK_PAD_RADIUS = 0.03 # Ink pad is a cylinder with radius 0.02m and height 0.02m
INK_PAD_HEIGHT = 0.01 # half-height of the ink pad
GOAL_MARKER_SIZE = [0.01, 0.01, 0.02] # half-sizes for x, y, z(m), the full size is 2cm * 2cm * 4cm

@register_env("SimpleOffice-v1", max_episode_steps=100)
class SimpleOfficeEnv(BaseEnv):
    """
    Task: 
    place the book on the paper in its proper position on the shelf,
    then pick up the seal from the ink pad and
    stamp it in the bottom right corner of the paper.
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Fetch, Panda]

    def __init__(
      self,
      *args,
      robot_uids="panda",
      num_envs=1,
      reconfiguration_freq=None,
      **kwargs,      
    ):
        # Set reconfiguration frequency - for single env, reconfigure every time
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0

        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()

        # To make code more readable, define a function to create boxes
        def create_box(name, half_size, color, is_static, init_pos, init_q):
            builder = self.scene.create_actor_builder()
            if name != "goal_marker" and name != "insert_target":  # Visual-only objects don't need collision
                builder.add_box_collision(half_size=half_size)
            builder.add_box_visual(
                half_size=half_size,
                material=sapien.render.RenderMaterial(base_color=color),
            )
            # Set initial position
            builder.initial_pose = sapien.Pose(p=[init_pos[0], init_pos[1], init_pos[2] + 0.5], q=init_q)
            # Build the object
            if is_static:
                return builder.build_static(name=name)
            else:
                return builder.build(name=name)
        
        # Create static paper on the table
        self.paper = create_box(
            name="Paper",
            half_size=PAPER_SIZE,
            color=[1, 1, 1, 1],  # white
            is_static=True,
            init_pos=[0, 0, PAPER_SIZE[2]],
            init_q=[1, 0, 0, 0]
        )

        # Create book on the table
        self.book = create_box(
            name="Book",
            half_size=BOOK_SIZE,
            color=[0.1, 0.1, 0.1, 1],  # black
            is_static=False,
            init_pos=[0, 0, BOOK_SIZE[2]],
            init_q=[1, 0, 0, 0]
        )
        # Create Shelf 
        self.shelf_bottom = create_box(
            name="ShelfBottom",
            half_size=SHELF_BOTTOM_SIZE,
            color=[0.8, 0.7, 0.55, 1],  # brown
            is_static=True,
            init_pos=[0.3, 0, SHELF_BOTTOM_SIZE[2]],
            init_q=[1, 0, 0, 0]
        )
        self.shelf_back = create_box(
            name="ShelfBack",
            half_size=SHELF_BACK_SIZE,
            color=[0.8, 0.7, 0.55, 1],  # brown
            is_static=True,
            init_pos=[0.3 + SHELF_BOTTOM_SIZE[0] + SHELF_BACK_SIZE[0], 0, SHELF_BACK_SIZE[2]],
            init_q=[1, 0, 0, 0]
        )
        self.shelf_left = create_box(
            name="ShelfLeft",
            half_size=SHELF_LEFT_SIZE,
            color=[0.8, 0.7, 0.55, 1],  # brown
            is_static=True,
            init_pos=[0.3, 0 + SHELF_BOTTOM_SIZE[1] + SHELF_LEFT_SIZE[1], SHELF_LEFT_SIZE[2]],
            init_q=[1, 0, 0, 0]
        )
        self.shelf_right = create_box(
            name="ShelfRight",
            half_size=SHELF_RIGHT_SIZE,
            color=[0.8, 0.7, 0.55, 1],  # brown
            is_static=True,
            init_pos=[0.3, 0 - SHELF_BOTTOM_SIZE[1] - SHELF_RIGHT_SIZE[1], SHELF_RIGHT_SIZE[2]],
            init_q=[1, 0, 0, 0]
        )

        # Create Book_not_move
        self.book1_not_move = create_box(
            name="Book1_not_move",
            half_size=BOOK1_NOTMOVE_SIZE,
            color=[0.4, 0.26, 0.13, 1],  # brown
            is_static=False,
            init_pos=[0.3, 0 + SHELF_BOTTOM_SIZE[1]/2, BOOK1_NOTMOVE_SIZE[2]],
            init_q=[1, 0, 0, 0]
        )
        self.book2_not_move = create_box(
            name="Book2_not_move",
            half_size=BOOK2_NOTMOVE_SIZE,
            color=[0.05, 0.17, 0.38, 1],  # blue
            is_static=False,
            init_pos=[0.3, 0, BOOK2_NOTMOVE_SIZE[2]],
            init_q=[1, 0, 0, 0]
        )
        self.book3_not_move = create_box(
            name="Book3_not_move",
            half_size=BOOK3_NOTMOVE_SIZE,
            color=[0.13, 0.33, 0.18, 1],  # green
            is_static=False,
            init_pos=[0.3, 0 - SHELF_BOTTOM_SIZE[1]/2, BOOK3_NOTMOVE_SIZE[2]],
            init_q=[1, 0, 0, 0]
        )
        # Create ink pad(cylinder)
        ink_pad_builder = self.scene.create_actor_builder()
        ink_pad_builder.add_cylinder_collision(radius=INK_PAD_RADIUS, half_length=INK_PAD_HEIGHT)
        ink_pad_builder.add_cylinder_visual(radius=INK_PAD_RADIUS, half_length=INK_PAD_HEIGHT, material=sapien.render.RenderMaterial(base_color=[0.2, 0.2, 0.2, 1])) #作用：
        ink_pad_builder.initial_pose = sapien.Pose(p=[0, 0, INK_PAD_HEIGHT + 0.5], q=[0.7071, 0, 0.7071, 0]) 
        self.ink_pad = ink_pad_builder.build_static(name="InkPad")

        self.seal = create_box(
            name="Seal",
            half_size=SEAL_SIZE,
            color=[1, 0, 0, 1],  # red
            is_static=False,
            init_pos=[0, 0, SEAL_SIZE[2]],
            init_q=[1, 0, 0, 0]
        )

        # Create goal marker
        self.goal_marker = create_box(
            name="goal_marker",
            half_size=GOAL_MARKER_SIZE,
            color=[0, 1, 0, 0.5],  # transparent green
            is_static=True,
            init_pos=[-PAPER_SIZE[0]/2, -PAPER_SIZE[1]/2, GOAL_MARKER_SIZE[2]],
            init_q=[1, 0, 0, 0]
        )
        self.insert_target = create_box(
            name="insert_target",
            half_size=BOOK_SIZE,
            color=[0, 1, 0, 0.5],  # transparent green
            is_static=True,
            init_pos=[0.3, 0, BOOK_SIZE[0]],
            init_q=[0.5, -0.5, 0.5, 0.5]
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.table_scene.initialize(env_idx)

        with torch.device(self.device):
            b = len(env_idx)

            # Helper function to set up object positions
            def setup_object(obj, pos, quat=[1,0,0,0], reset_velocity=True):
                pose = Pose.create_from_pq(p=pos, q=quat)
                obj.set_pose(pose)
                if reset_velocity:
                    obj.set_linear_velocity(torch.zeros((b, 3)))
                    obj.set_angular_velocity(torch.zeros((b, 3)))
                return pos  # Return position for later use
            
            # Reset paper position
            paper_pos = torch.zeros((b, 3))
            paper_pos[..., 0] = torch.rand((b,)) * 0.08 - 0.04    # Ensure that there is no collision with shelf
            paper_pos[..., 1] = torch.rand((b,)) * 0.3 - 0.15
            paper_pos[..., 2] = PAPER_SIZE[2]   # paper is on the table
            self.paper_position = setup_object(self.paper, paper_pos, reset_velocity=False)

            book_pos = torch.clone(paper_pos)
            book_pos[..., 2] = 2*PAPER_SIZE[2] + BOOK_SIZE[2]   # book is on the center of the paper
            self.book_position = setup_object(self.book, book_pos)

            # Shelf components
            shelf_positions = {
                "bottom": torch.tensor([0.3, 0, SHELF_BOTTOM_SIZE[2]]).expand(b, -1),
                "back": torch.tensor([0.3 + SHELF_BOTTOM_SIZE[0] + SHELF_BACK_SIZE[0], 0, SHELF_BACK_SIZE[2]]).expand(b, -1),
                "left": torch.tensor([0.3, SHELF_BOTTOM_SIZE[1] + SHELF_LEFT_SIZE[1], SHELF_LEFT_SIZE[2]]).expand(b, -1),
                "right": torch.tensor([0.3, -SHELF_BOTTOM_SIZE[1] - SHELF_RIGHT_SIZE[1], SHELF_RIGHT_SIZE[2]]).expand(b, -1)
            }

            self.shelf_bottom_position = setup_object(self.shelf_bottom, shelf_positions["bottom"], reset_velocity=False)
            self.shelf_back_position = setup_object(self.shelf_back, shelf_positions["back"], reset_velocity=False)
            self.shelf_left_position = setup_object(self.shelf_left, shelf_positions["left"], reset_velocity=False)
            self.shelf_right_position = setup_object(self.shelf_right, shelf_positions["right"], reset_velocity=False)

            # Reset book_not_move position
            book1_not_move_pos = torch.zeros((b, 3))
            book1_not_move_pos[..., 0] = 0.3
            book1_not_move_pos[..., 1] = torch.rand((b,)) * 0.1 - 0.25
            book1_not_move_pos[..., 2] = 2 * SHELF_BOTTOM_SIZE[2] + BOOK1_NOTMOVE_SIZE[2]
            self.book1_not_move_position = setup_object(self.book1_not_move, book1_not_move_pos)

            book2_not_move_pos = torch.zeros((b, 3))
            book2_not_move_pos[..., 0] = 0.3
            book2_not_move_pos[..., 1] = torch.rand((b,)) * 0.2 - 0.1
            book2_not_move_pos[..., 2] = 2 * SHELF_BOTTOM_SIZE[2] + BOOK2_NOTMOVE_SIZE[2]
            self.book2_not_move_position = setup_object(self.book2_not_move, book2_not_move_pos)

            book3_not_move_pos = torch.zeros((b, 3))
            book3_not_move_pos[..., 0] = 0.3
            book3_not_move_pos[..., 1] = torch.rand((b,)) * 0.1 + 0.15
            book3_not_move_pos[..., 2] = 2 * SHELF_BOTTOM_SIZE[2] + BOOK3_NOTMOVE_SIZE[2]
            self.book3_not_move_position = setup_object(self.book3_not_move, book3_not_move_pos)

            # Reset ink pad position
            ink_pad_pos = torch.zeros((b, 3))
            ink_pad_pos[..., 0] = torch.rand((b,)) * 0.1 - 0.35     # Ensure that there is no collision with paper
            ink_pad_pos[..., 1] = torch.rand((b,)) * 0.4 - 0.2
            ink_pad_pos[..., 2] = INK_PAD_HEIGHT
            self.ink_pad_position = setup_object(self.ink_pad, ink_pad_pos, quat=[0.7071, 0, 0.7071, 0],reset_velocity=False)

            # Reset seal position. Initially above the center of the ink pad.
            seal_pos = torch.zeros((b, 3))
            seal_pos[..., 0] = ink_pad_pos[...,0]
            seal_pos[..., 1] = ink_pad_pos[...,1]
            seal_pos[..., 2] = 2 * INK_PAD_HEIGHT + SEAL_SIZE[2]
            self.seal_position = setup_object(self.seal, seal_pos)

            # Reset goal marker position
            goal_marker_pos = torch.zeros((b, 3))
            goal_marker_pos[..., 0] = paper_pos[...,0] - PAPER_SIZE[0]/2
            goal_marker_pos[..., 1] = paper_pos[...,1] - PAPER_SIZE[1]/2
            goal_marker_pos[..., 2] = 2 * PAPER_SIZE[2] + GOAL_MARKER_SIZE[2]
            self.goal_marker_position = setup_object(self.goal_marker, goal_marker_pos, reset_velocity=False)

            # Determine where is the best place to insert the book
            distance0 = book1_not_move_pos[..., 1] - BOOK1_NOTMOVE_SIZE[1] - (-SHELF_BOTTOM_SIZE[1])
            distance1 = book2_not_move_pos[..., 1] - BOOK2_NOTMOVE_SIZE[1] - book1_not_move_pos[..., 1] - BOOK1_NOTMOVE_SIZE[1]
            distance2 = book3_not_move_pos[..., 1] - BOOK3_NOTMOVE_SIZE[1] - book2_not_move_pos[..., 1] - BOOK2_NOTMOVE_SIZE[1]
            distance3 = SHELF_BOTTOM_SIZE[1] - book3_not_move_pos[..., 1] - BOOK3_NOTMOVE_SIZE[1]
            distances = torch.stack([distance0, distance1, distance2, distance3], dim=-1)
            max_distance, max_idx = torch.max(distances, dim=-1)
            if max_idx == 0:
                insert_target_y = -SHELF_BOTTOM_SIZE[1] + distance0 / 2
            elif max_idx == 1:
                insert_target_y = book1_not_move_pos[..., 1] + BOOK1_NOTMOVE_SIZE[1] + distance1 / 2
            elif max_idx == 2:
                insert_target_y = book2_not_move_pos[..., 1] + BOOK2_NOTMOVE_SIZE[1] + distance2 / 2
            else:
                insert_target_y = SHELF_BOTTOM_SIZE[1] - distance3 / 2
            insert_target_pos = torch.zeros((b, 3))
            insert_target_pos[..., 0] = 0.3
            insert_target_pos[..., 1] = insert_target_y
            insert_target_pos[..., 2] = 2 * SHELF_BOTTOM_SIZE[2] + BOOK_SIZE[0]
            self.insert_target_position = setup_object(self.insert_target, insert_target_pos, quat=[0.5, -0.5, 0.5, 0.5], reset_velocity=False)

    def evaluate(self):
        """
        Determine success/failure of the task
        """

        with torch.device(self.device):
            # Check if the book is inserted in the proper position
            book_reach_target_p = (
                torch.linalg.norm(self.book.pose.p - self.insert_target.pose.p, dim=-1)
                < 0.005
            )

            # To check the quaternion angle difference between book and insert_target by quaternion dot product
            # Attention: it is not proper to use L2 distance to check the angle difference of quaternion
            q1 = self.book.pose.q
            q2 = self.insert_target.pose.q
            dot_product = torch.abs(torch.sum(q1 * q2, dim=-1))
            dot_product = torch.clamp(dot_product, min=0.0, max=1.0)
            angle_diff = 2 * torch.acos(dot_product)
            book_reach_target_q = angle_diff < 0.1  # Threshold in radians, 0.1 rad = 5.73 degree
        
            book_reach_target = book_reach_target_p & book_reach_target_q
            # Check if the seal is on the goal_marker
            seal_at_goal = (
                torch.linalg.norm(self.seal.pose.p - self.goal_marker_position, dim=-1)
                < 0.005
            )

            # Check if the books_not_move is moved 
            book1_not_move_distance = torch.linalg.norm(self.book1_not_move.pose.p - self.book1_not_move_position, dim=-1)
            book2_not_move_distance = torch.linalg.norm(self.book2_not_move.pose.p - self.book2_not_move_position, dim=-1)
            book3_not_move_distance = torch.linalg.norm(self.book3_not_move.pose.p - self.book3_not_move_position, dim=-1)
            move_threshold = 0.01  
            any_book_moved = (book1_not_move_distance > move_threshold) | \
                            (book2_not_move_distance > move_threshold) | \
                            (book3_not_move_distance > move_threshold)
            
            success = book_reach_target & seal_at_goal & (~any_book_moved)

            return {
                "success": success,
                "book_reach_target": book_reach_target,
                "seal_at_goal": seal_at_goal,
                "angle_diff": angle_diff,
                "any_book_moved": any_book_moved,
            }
        
    def _get_obs_extra(self, info: Dict):
        """
        Additional observation for solving the task
        """

        obs = dict(
            tcp_pose = self.agent.tcp.pose.raw_pose,
            insert_target_pose = self.insert_target.pose.raw_pose,
            goal_marker_pose = self.goal_marker.pose.raw_pose,
        )

        if self.obs_mode_struct.use_state:
            obs["seal_pos"] = self.seal.pose.raw_pose
            obs["seal_vel"] = self.seal.linear_velocity
            obs["book_pos"] = self.book.pose.raw_pose
            obs["book_linear_vel"] = self.book.linear_velocity
            obs["book_rotation_vel"] = self.book.angular_velocity

        return obs
    

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """
        Compute a dense reward signal to guide learning
        """

        with torch.device(self.device):
            reward = torch.zeros(self.num_envs, device=self.device)

            # Success reward
            success = info["success"]
            reward = torch.where(success, reward + 10.0, reward)

            # Reward for having book close to the target
            target_distance = torch.linalg.norm(self.book.pose.p - self.insert_target_position, dim=-1)
            reward += torch.exp(-10.0 * target_distance) * 1

            # Reward for having book rotated into proper position
            angle_diff = info["angle_diff"]
            reward += torch.exp(-10.0 * angle_diff) * 1

            # Book reach target reward
            book_reach_target = info["book_reach_target"]
            reward = torch.where(book_reach_target, reward + 1, reward)

            # Reward for having seal close to the goal
            seal_distance = torch.linalg.norm(self.seal.pose.p - self.goal_marker_position, dim=-1)
            reward += torch.exp(-10.0 * seal_distance) * 1

            # Seal at goal reward
            seal_at_goal = info["seal_at_goal"]
            reward = torch.where(seal_at_goal, reward + 1, reward)

            # Penality for moving the book_not_move. However, to avoid minus reward, we use the reward of 1.0.
            any_book_moved = info["any_book_moved"]
            reward = torch.where(any_book_moved, reward, reward+1)
            return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Normalize the dense reward"""
        max_reward = (
            16.0  # Maximum possible reward (success + all intermediate rewards)
        )
        return self.compute_dense_reward(obs, action, info) / max_reward
    
    @property
    def _default_sensor_configs(self):
        """Configure camera sensors for the environment"""
        # Top-down camera view
        top_camera = CameraConfig(
            "top_camera",
            pose=look_at(eye=[0, 0, 0.8], target=[0, 0, 0]),
            width=256,
            height=256,
            fov=np.pi / 3,
            near=0.01,
            far=100,
        )

        # Side view camera
        side_camera = CameraConfig(
            "side_camera",
            pose=look_at(eye=[1.0, -1.0, 1.0], target=[0, 0, 0.2]),
            width=256,
            height=256,
            fov=np.pi / 3,
            near=0.01,
            far=100,
        )
        return [top_camera, side_camera]

    @property
    def _default_human_render_camera_configs(self):
        """Configure camera for human viewing"""
        return CameraConfig(
            "render_camera",
            pose=look_at(eye=[1.0, -1.2, 1.2], target=[0, 0, 0.1]),
            width=1024,
            height=1024,
            fov=np.pi / 3,
            near=0.01,
            far=100,
        )