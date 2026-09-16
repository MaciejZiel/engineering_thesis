from vision_robot_arm.robot.backend import DebugBackend, RobotBackend
from vision_robot_arm.robot.config import (
    BACKEND_DEBUG,
    BACKEND_NONE,
    BACKEND_SIM,
    RobotConfig,
)
from vision_robot_arm.robot.controller import (
    MappedRobotController,
    NullRobotController,
    RobotController,
)
from vision_robot_arm.robot.mapping import RobotMapper
from vision_robot_arm.robot.simulation import SimulationBackend


def create_robot_controller(config: RobotConfig) -> RobotController:
    if config.backend == BACKEND_NONE:
        return NullRobotController()
    return MappedRobotController(RobotMapper(config), create_robot_backend(config))


def create_robot_backend(config: RobotConfig) -> RobotBackend:
    if config.backend == BACKEND_DEBUG:
        return DebugBackend(print_interval=config.print_interval)
    if config.backend == BACKEND_SIM:
        return SimulationBackend(config)
    raise SystemExit(f"Robot backend '{config.backend}' is not available yet.")
