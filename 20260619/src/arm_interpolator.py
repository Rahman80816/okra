from __future__ import annotations

from collections.abc import Mapping


class ArmInterpolator:
    def __init__(
        self,
        start_angles: Mapping[str, float],
        goal_angles: Mapping[str, float],
        duration_s: float,
        control_freq_hz: int = 250,
    ) -> None:
        """
        start_angles: dict[str, float] - start joint angles
        goal_angles: dict[str, float] - goal joint angles
        duration_s: movement duration in seconds
        control_freq_hz: control frequency
        """
        if duration_s < 0:
            raise ValueError("duration_s must be non-negative")
        if control_freq_hz <= 0:
            raise ValueError("control_freq_hz must be positive")

        start_keys = set(start_angles)
        goal_keys = set(goal_angles)
        if start_keys != goal_keys:
            missing_from_goal = sorted(start_keys - goal_keys)
            missing_from_start = sorted(goal_keys - start_keys)
            raise ValueError(
                "start_angles and goal_angles must have the same joints "
                f"(missing_from_goal={missing_from_goal}, "
                f"missing_from_start={missing_from_start})"
            )

        self._joint_names = tuple(start_angles.keys())
        self._start_angles = {name: float(start_angles[name]) for name in self._joint_names}
        self._goal_angles = {name: float(goal_angles[name]) for name in self._joint_names}
        self._total_steps = max(1, int(round(duration_s * control_freq_hz)))

    def get_target(self, step: int) -> dict[str, float]:
        """Return interpolated target angles for the given step."""
        if step <= 0:
            ratio = 0.0
        elif step >= self._total_steps:
            ratio = 1.0
        else:
            ratio = step / self._total_steps

        return {
            name: self._start_angles[name]
            + (self._goal_angles[name] - self._start_angles[name]) * ratio
            for name in self._joint_names
        }

    def is_done(self, step: int) -> bool:
        """Return True when interpolation is complete."""
        return step >= self._total_steps

    @property
    def total_steps(self) -> int:
        """Total interpolation steps."""
        return self._total_steps
