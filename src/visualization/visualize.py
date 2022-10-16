"""Functions related to visualization of landmarks and angles"""

from typing import List, Tuple

import mediapipe as mp
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

sns.set_style("whitegrid")

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5


def plot_3d_landmarks(
    landmark_list: NormalizedLandmarkList,
    connections: List[Tuple[int, int]] = mp.solutions.pose.POSE_CONNECTIONS,
) -> go.Figure:
    """Function to plot landmarks in 3D

    Args:
        landmark_list (NormalizedLandmarkList): list of landmarks to plot
        connections (List[Tuple[int, int]]): connections between landmarks

    Returns:
        go.Figure: Interactive 3D figure
    """
    plotted_landmarks = {
        idx: (-landmark.z, landmark.x, -landmark.y)
        for idx, landmark in enumerate(landmark_list.landmark)
        if (
            not landmark.HasField("visibility")
            or landmark.visibility >= _VISIBILITY_THRESHOLD
        )
        and (
            not landmark.HasField("presence")
            or landmark.presence >= _PRESENCE_THRESHOLD
        )
    }

    out_cn = []
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
            landmark_pair = [
                plotted_landmarks[start_idx],
                plotted_landmarks[end_idx],
            ]
            out_cn.append(
                dict(
                    xs=[landmark_pair[0][0], landmark_pair[1][0]],
                    ys=[landmark_pair[0][1], landmark_pair[1][1]],
                    zs=[landmark_pair[0][2], landmark_pair[1][2]],
                )
            )
    cn2 = {"xs": [], "ys": [], "zs": []}
    for pair in out_cn:
        for k in pair.keys():
            cn2[k].append(pair[k][0])
            cn2[k].append(pair[k][1])
            cn2[k].append(None)

    df_plot = pd.DataFrame(plotted_landmarks).T.rename(columns={0: "z", 1: "x", 2: "y"})
    df_plot["lm"] = df_plot.index.map(
        lambda s: mp.solutions.pose.PoseLandmark(s).name
    ).values
    fig = (
        px.scatter_3d(
            df_plot,
            x="z",
            y="x",
            z="y",
            hover_name="lm",
        )
        .update_traces(marker={"color": "red"})
        .update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            scene={"camera": {"eye": {"x": 2.1, "y": 0, "z": 0}}},
        )
    )
    fig.add_traces(
        [
            go.Scatter3d(
                x=cn2["xs"],
                y=cn2["ys"],
                z=cn2["zs"],
                mode="lines",
                line={"color": "black", "width": 5},
                name="connections",
            )
        ]
    )

    return fig
