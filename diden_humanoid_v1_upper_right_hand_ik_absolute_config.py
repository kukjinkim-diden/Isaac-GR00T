"""GR00T N1.7 embodiment config for DIDEN humanoid v1 upper-body with DG-5F hands.

Usage (finetuning script):
    from diden_vr.groot_embodiment_hand import register_diden_humanoid_right_hand
    register_diden_humanoid_right_hand()

Then pass ``EmbodimentTag.NEW_EMBODIMENT`` to the GR00T trainer.

Dataset modality layout (must match data_recorder._MODALITY_JSON with use_finger_data=True):
    observation.state [74]:
        joint_pos    [0:14]  — 7-DOF left arm + 7-DOF right arm (rad)
        joint_vel    [14:28] — joint velocities (rad/s)
        ee_pos       [28:34] — EE positions [Lx,Ly,Lz,Rx,Ry,Rz] (m)
        finger_left  [34:54] — Tesollo DG-5F left hand, 20 joints (deg)
        finger_right [54:74] — Tesollo DG-5F right hand, 20 joints (deg)
    action [54]:
        joint_pos    [0:14]  — absolute joint positions; delta computed at train time
        finger_left  [14:34] — left hand 20 joints (deg)
        finger_right [34:54] — right hand 20 joints (deg)
"""

from __future__ import annotations
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ActionType,
    ModalityConfig,
)


def build_diden_humanoid_hand_config() -> "dict[str, ModalityConfig]":
    """Return the ModalityConfig dict for DIDEN humanoid v1 + DG-5F hands."""
    return {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_cam",],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["right_arm_joint_pos", "right_arm_joint_vel", "right_ee_pos", "finger_right", "fingertip_right", "right_grasp_ratio"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),  # 16-step prediction horizon
            modality_keys=["right_arm_joint_pos",],
            action_configs=[
                # Arm joints: stored as absolute → delta computed at train time
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


def register_diden_humanoid_right_hand_ik_command_action() -> None:
    """Register DIDEN humanoid + DG-5F hand config under EmbodimentTag.NEW_EMBODIMENT."""
    register_modality_config(
        build_diden_humanoid_hand_config(),
        EmbodimentTag.NEW_EMBODIMENT,
    )
    print("[GR00T] DIDEN humanoid v1 upper-body + DG-5F hands registered as NEW_EMBODIMENTright_.")


register_diden_humanoid_right_hand_ik_command_action()
