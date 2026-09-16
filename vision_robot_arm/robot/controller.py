from typing import Protocol

from vision_robot_arm.core.pose_state import PoseState
from vision_robot_arm.robot.backend import RobotBackend
from vision_robot_arm.robot.mapping import RobotMapper
from vision_robot_arm.robot.targets import RobotState


class RobotController(Protocol):
    def update(self, state: PoseState) -> None:
        ...

    def reset(self) -> None:
        ...

    def robot_state(self) -> RobotState | None:
        ...

    def status_lines(self) -> list[str]:
        ...

    def close(self) -> None:
        ...


class NullRobotController:
    def update(self, state: PoseState) -> None:
        return

    def reset(self) -> None:
        return

    def robot_state(self) -> RobotState | None:
        return None

    def status_lines(self) -> list[str]:
        return []

    def close(self) -> None:
        return


class MappedRobotController:
    def __init__(self, mapper: RobotMapper, backend: RobotBackend) -> None:
        self._mapper = mapper
        self._backend = backend

    def update(self, state: PoseState) -> None:
        self._backend.send(self._mapper.map(state))

    def reset(self) -> None:
        self._mapper.reset()

    def robot_state(self) -> RobotState | None:
        return self._backend.robot_state()

    def status_lines(self) -> list[str]:
        return self._backend.status_lines()

    def close(self) -> None:
        self._backend.close()
