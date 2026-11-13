"""Shared pytest fixtures."""

from __future__ import annotations

import logging

import pytest

logging.getLogger("robosuite_logs").setLevel(logging.ERROR)


@pytest.fixture(scope="session")
def sim_env():
    """One real MuJoCo/robosuite environment shared by the integration tests."""
    from robotics.simulation.env import SimEnv

    env = SimEnv()
    yield env
    env.close()
