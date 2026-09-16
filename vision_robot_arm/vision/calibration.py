class PoseCalibration:
    def __init__(self) -> None:
        self._neutral_angles: dict[str, float] | None = None

    @property
    def calibrated(self) -> bool:
        return self._neutral_angles is not None

    def capture(self, angles: dict[str, float | None]) -> int:
        neutral = {
            name: value
            for name, value in angles.items()
            if value is not None
        }
        self._neutral_angles = neutral
        return len(neutral)

    def relative_angles(
        self,
        angles: dict[str, float | None],
    ) -> dict[str, float | None]:
        if self._neutral_angles is None:
            return {name: None for name in angles}

        relative: dict[str, float | None] = {}
        for name, value in angles.items():
            neutral_value = self._neutral_angles.get(name)
            if value is None or neutral_value is None:
                relative[name] = None
            else:
                relative[name] = value - neutral_value
        return relative
