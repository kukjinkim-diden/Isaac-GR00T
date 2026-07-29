"""GR00T N1.7 embodiment config — DIDEN humanoid v1 RIGHT arm + DG-5F right hand.

For the single-arm quest_fixed_hand recordings (Push_Buttons_Center
v2.1_quest_fixed_hand). Modality layout (meta/modality.json):
    observation.state [53]:
        right_arm_joint_pos [0:7]    — 7-DOF right arm (rad)
        right_arm_joint_vel [7:14]   — joint velocities (rad/s)
        right_ee_pos        [14:17]  — EE position (m)
        finger_right        [17:37]  — Tesollo DG-5F, 20 joints (deg)
        fingertip_right     [37:52]  — fingertip FK (m)
        right_grasp_ratio   [52:53]
    action [28]:
        right_arm_joint_pos [0:7]    — absolute joint positions
        finger_right        [7:27]   — 20 joints (deg)
        right_grasp_ratio   [27:28]
"""

from __future__ import annotations
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


def build_diden_humanoid_right_arm_hand_config() -> "dict[str, ModalityConfig]":
    return {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_cam"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "right_arm_joint_pos",
                "right_arm_joint_vel",
                "finger_right",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),  # 16-step prediction horizon
            modality_keys=["right_arm_joint_pos", "finger_right"],
            action_configs=[
                # Arm joints: stored as absolute → delta computed at train time
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # Right finger joints
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    }


def register_diden_humanoid_right_arm_hand() -> None:
    register_modality_config(
        build_diden_humanoid_right_arm_hand_config(),
        EmbodimentTag.NEW_EMBODIMENT,
    )
    print("[GR00T] DIDEN humanoid v1 RIGHT arm + DG-5F hand registered as NEW_EMBODIMENT.")


register_diden_humanoid_right_arm_hand()
