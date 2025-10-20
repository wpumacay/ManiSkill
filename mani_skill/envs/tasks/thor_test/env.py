from __future__ import annotations

from typing import Any

import torch
import sapien
from pathlib import Path

from mani_skill.agents.robots import GoogleRobot, UnitreeG1
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.registration import register_env


from sapien_thor.assets.loader import (
    ThorMjcfAssetActorLoader,
    ThorMjcfAssetArticulationLoader,
    ThorMjcfSceneLoader
)
from sapien_thor.core.types import RobotUidType


class IThorBasicEnv(BaseEnv):
    """Main interface for simple tasks that make use of THOR assets from the MuJoCo-THOR project"""

    SUPPORTED_ROBOTS = [
        GoogleRobot.uid,
        UnitreeG1.uid,
    ]

    agent: GoogleRobot | UnitreeG1

    def __init__(
        self,
        *args,
        robot_uids: RobotUidType = GoogleRobot.uid,
        **kwargs,
    ) -> None:
        self._mjcf_actor_loader = ThorMjcfAssetActorLoader[ManiSkillScene]()
        self._mjcf_articulation_loader = ThorMjcfAssetArticulationLoader[ManiSkillScene]()

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_scene(self, options: dict) -> None:
        self._mjcf_actor_loader.set_scene(self.scene)
        self._mjcf_articulation_loader.set_scene(self.scene)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return 0.0

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return 0.0


@register_env("ThorBasicPickTomato-v1")
class ThorBasicPickTomatoEnv(IThorBasicEnv):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

@register_env("ThorHouse-v1")
class IThorHouseEnv(BaseEnv):
    SUPPORTED_ROBOTS = [
        GoogleRobot.uid,
        UnitreeG1.uid,
    ]

    agent: GoogleRobot | UnitreeG1

    def __init__(
        self,
        *args,
        robot_uids: RobotUidType = GoogleRobot.uid,
        **kwargs,
    ) -> None:
        self.loader = ThorMjcfSceneLoader()

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-1, 0, 0]))

    def _load_scene(self, options: dict) -> None:
        self.loader.set_scene(self.scene)
        # TODO (sam): remove hardcoded asset path
        ROOT_DIR = Path("/home/robopc1/swang/sapien-thor")
        filepath = ROOT_DIR / "assets" / "scenes" / "ithor-bundled" / "FloorPlan1_physics_mesh.xml"
        actors, articulations = self.loader.load(filepath)
        
    def _load_lighting(self, options: dict) -> None:
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=False)
        self.scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=False)
        self.scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=False)
        self.scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=False)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return 0.0

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return 0.0

