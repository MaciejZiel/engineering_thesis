from vision_robot_arm.robot.backend import DebugBackend, RobotBackend
from vision_robot_arm.robot.config import (
    BACKEND_DEBUG,
    BACKEND_NONE,
    BACKEND_SERIAL,
    BACKEND_SIM,
    BACKEND_UR,
    RobotConfig,
)
from vision_robot_arm.robot.controller import (
    MappedRobotController,
    NullRobotController,
    RobotController,
)
from vision_robot_arm.robot.mapping import RobotMapper
from vision_robot_arm.robot.serial_backend import SerialBackend
from vision_robot_arm.robot.simulation import SimulationBackend
from vision_robot_arm.robot.ur_backend import URBackend


def create_robot_controller(config: RobotConfig) -> RobotController:
    if config.backend == BACKEND_NONE:
        return NullRobotController()
    return MappedRobotController(RobotMapper(config), create_robot_backend(config))


def create_robot_backend(config: RobotConfig) -> RobotBackend:
    if config.backend == BACKEND_DEBUG:
        return DebugBackend(print_interval=config.print_interval)
    if config.backend == BACKEND_SIM:
        return SimulationBackend(config)
    if config.backend == BACKEND_UR:
        if not config.hosts:
            raise SystemExit(
                "--robot-right-host and/or --robot-left-host is required with --robot-backend ur"
            )
        return URBackend(config)
    if config.backend == BACKEND_SERIAL:
        if config.port is None:
            raise SystemExit("--robot-port is required with --robot-backend serial")
        return SerialBackend(
            port=config.port,
            baud_rate=config.baud_rate,
            send_interval=config.send_interval,
        )
    raise SystemExit(f"Unknown robot backend: {config.backend}")
