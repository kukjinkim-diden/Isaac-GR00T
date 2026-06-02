"""GR00T N1.7 embodiment config for DIDEN humanoid v1 upper-body.

Usage (finetuning script):
    from diden_vr.groot_embodiment import register_diden_humanoid
    register_diden_humanoid()

Then pass ``EmbodimentTag.NEW_EMBODIMENT`` to the GR00T trainer.

Dataset modality layout (must match data_recorder._MODALITY_JSON):
    observation.state [34]:
        joint_pos [0:14]   — 7-DOF left arm + 7-DOF right arm (rad)
        joint_vel [14:28]  — joint velocities (rad/s)
        ee_pos    [28:34]  — EE positions [Lx,Ly,Lz,Rx,Ry,Rz] (m)
    action [14]:
        joint_pos [0:14]   — absolute joint positions; RELATIVE rep → delta at train time
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


def build_diden_humanoid_config() -> dict[str, ModalityConfig]:
    """Return the ModalityConfig dict for DIDEN humanoid v1 upper-body."""
    return {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_cam"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["joint_pos", "joint_vel", "ee_pos"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),  # 16-step prediction horizon
            modality_keys=["joint_pos"],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,  # absolute stored → delta computed at train time
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


def register_diden_humanoid() -> None:
    """Register DIDEN humanoid config under EmbodimentTag.NEW_EMBODIMENT."""
    register_modality_config(
        build_diden_humanoid_config(),
        EmbodimentTag.NEW_EMBODIMENT,
    )
    print("[GR00T] DIDEN humanoid v1 upper-body registered as NEW_EMBODIMENT.")


register_diden_humanoid()
