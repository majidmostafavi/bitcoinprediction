import logging

from typing import List, NamedTuple

logger = logging.getLogger("mlagents.trainers")


class CameraResolution(NamedTuple):
    height: int
    width: int
    num_channels: int

    @property
    def gray_scale(self) -> bool:
        return self.num_channels == 1

    def __str__(self):
        return f"CameraResolution({self.height}, {self.width}, {self.num_channels})"


class BrainParameters:
    def __init__(
        self,
        brain_name: str,
        vector_observation_space_size: int,
        camera_resolutions: List[CameraResolution],
        vector_action_space_size: List[int],
        vector_action_descriptions: List[str],
        vector_action_space_type: int,
    ):
        """
        Contains all brain-specific parameters.
        """
        self.brain_name = brain_name
        self.vector_observation_space_size = vector_observation_space_size
        self.number_visual_observations = len(camera_resolutions)
        self.camera_resolutions = camera_resolutions
        self.vector_action_space_size = vector_action_space_size
        self.vector_action_descriptions = vector_action_descriptions
        self.vector_action_space_type = ["discrete", "continuous"][
            vector_action_space_type
        ]

    def __str__(self):
        return """Unity brain name: {}
        Number of Visual Observations (per agent): {}
        Camera Resolutions: {}
        Vector Observation space size (per agent): {}
        Vector Action space type: {}
        Vector Action space size (per agent): {}
        Vector Action descriptions: {}""".format(
            self.brain_name,
            str(self.number_visual_observations),
            str([str(cr) for cr in self.camera_resolutions]),
            str(self.vector_observation_space_size),
            self.vector_action_space_type,
            str(self.vector_action_space_size),
            ", ".join(self.vector_action_descriptions),
        )

