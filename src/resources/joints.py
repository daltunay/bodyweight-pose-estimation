"""Class for joint angles to be processed"""

import json

import mediapipe as mp


class AngleJoints:
    """Class for the three joints required to define an angle"""

    def __init__(self, first: str, mid: str, end: str) -> None:
        self.joints = (first, mid, end)
        self.joints_idx = (
            mp.solutions.pose.PoseLandmark[first].value,
            mp.solutions.pose.PoseLandmark[mid].value,
            mp.solutions.pose.PoseLandmark[end].value,
        )
        self.side = self.parse_side()

    def parse_side(self):
        side_counts = {
            "left": str(self).count("left"),
            "right": str(self).count("right"),
        }
        return max(side_counts, key=side_counts.get)

    def __str__(self):
        return f"({self.joints[0].lower()}, {self.joints[1].lower()}, {self.joints[2].lower()})"


import os

print(os.getcwd())

with open(
    r"C:\Users\daniel\Projects\bwest\src\resources\joints.json", "r"
) as json_joints:
    joints_list = json.load(json_joints)
    JOINTS = {
        "left": [
            AngleJoints(**angle_joints)
            for angle_joints in joints_list.get("left_side")
        ],
        "right": [
            AngleJoints(**angle_joints)
            for angle_joints in joints_list.get("right_side")
        ],
        "sym": [
            AngleJoints(**angle_joints)
            for angle_joints in joints_list.get("symmetrical")
        ],
    }
