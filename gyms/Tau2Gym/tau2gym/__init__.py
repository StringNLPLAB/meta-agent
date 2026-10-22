"""Native τ³-bench text environment for the vendored verl tool interface.

The compatibility name is retained because τ³-bench itself still imports as
``tau2`` and the handover's rollout implementation uses ``Tau2Gym``.
"""

from .native import Tau2Env, create_environment

__all__ = ["Tau2Env", "create_environment"]
