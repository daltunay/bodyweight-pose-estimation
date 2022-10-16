"""Class for joint angles to be processed"""

import json
import os
from typing import Optional

import mediapipe as mp


class AngleJoints:
    """Class for the three joints required to define an angle"""

    def __init__(
        self, first: str, mid: str, end: str, _name: Optional[str] = None
    ) -> None:
        self.joints = (first, mid, end)
        self.joints_idx = (
            mp.solutions.pose.PoseLandmark[first].value,
            mp.solutions.pose.PoseLandmark[mid].value,
            mp.solutions.pose.PoseLandmark[end].value,
        )
        self._name = _name

    def __str__(self):
        return f"""({self.joints[0].lower()},
                    {self.joints[1].lower()},
                    {self.joints[2].lower()})"""


with open(
    os.path.join(os.path.dirname(__file__), "joints.json"), "r", encoding="utf-8"
) as joints_json:
    joints_dict = json.load(joints_json)

    JOINTS = [
        AngleJoints(**angle_joints)
        for joint_name in joints_dict
        for angle_joints in joints_dict[joint_name]
    ]
