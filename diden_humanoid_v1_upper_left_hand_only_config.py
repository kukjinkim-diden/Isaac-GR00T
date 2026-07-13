"""GR00T N1.7 HAND-ONLY embodiment config for DIDEN humanoid v1 upper-body.

Companion to diden_humanoid_v1_upper_left_arm_config.py (arm-only). Splitting
arm and hand into two policies decouples the shared flow-matching action head:
when arm+finger are predicted jointly, the high-variance finger dims degrade the
arm's velocity field and the arm chatters at inference. Train this alongside the
arm-only config and compose them at eval (vla_runner --hand-model-path).

Usage (finetuning script):
    from diden_vr.groot_embodiment_hand import register_diden_humanoid_left_hand_only
    register_diden_humanoid_left_hand_only()

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
    ModalityConfig,
)


def build_diden_humanoid_hand_config() -> "dict[str, ModalityConfig]":
    """Return the HAND-ONLY ModalityConfig dict for DIDEN humanoid v1 + DG-5F hands."""
    return {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_cam",],
        ),
        # State is identical to the arm-only config so both policies consume the
        # same observation (vla_runner builds one obs and feeds both).
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["left_arm_joint_pos", "left_arm_joint_vel", "finger_left",],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),  # 16-step prediction horizon
            modality_keys=["finger_left",],  # hand only — no arm
            action_configs=[
                # Left finger joints: stored as absolute → delta computed at train time
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


def register_diden_humanoid_left_hand_only() -> None:
    """Register DIDEN humanoid hand-only config under EmbodimentTag.NEW_EMBODIMENT."""
    register_modality_config(
        build_diden_humanoid_hand_config(),
        EmbodimentTag.NEW_EMBODIMENT,
    )
    print("[GR00T] DIDEN humanoid v1 upper-body LEFT HAND-ONLY registered as NEW_EMBODIMENT.")


register_diden_humanoid_left_hand_only()
