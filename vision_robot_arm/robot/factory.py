from vision_robot_arm.robot.backend import DebugBackend, RobotBackend
from vision_robot_arm.robot.config import BACKEND_DEBUG, BACKEND_NONE, RobotConfig
from vision_robot_arm.robot.controller import (
    MappedRobotController,
    NullRobotController,
    RobotController,
)
from vision_robot_arm.robot.mapping import RobotMapper


def create_robot_controller(config: RobotConfig) -> RobotController:
    if config.backend == BACKEND_NONE:
        return NullRobotController()
    return MappedRobotController(RobotMapper(config), create_robot_backend(config))


def create_robot_backend(config: RobotConfig) -> RobotBackend:
    if config.backend == BACKEND_DEBUG:
        return DebugBackend(print_interval=config.print_interval)
    raise SystemExit(f"Robot backend '{config.backend}' is not available yet.")
