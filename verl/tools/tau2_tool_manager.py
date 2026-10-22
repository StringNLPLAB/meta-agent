"""
Tau2ToolManager: Manages tau2-bench tools for UserRL training.

This module provides functionality to:
1. Load tools from tau2-bench environments
2. Convert tools to UserRL format
3. Generate tool schemas for different domains/tasks
"""

import logging
from typing import Dict, List, Optional, Any
import json

logger = logging.getLogger(__file__)

try:
    from tau2.registry import registry
    from tau2.environment.environment import Environment
    from tau2.environment.tool import Tool as Tau2Tool
    TAU2_AVAILABLE = True
except ImportError:
    TAU2_AVAILABLE = False
    registry = None
    Environment = None
    Tau2Tool = None


def get_tools_for_domain(domain: str, solo_mode: bool = False) -> List[Tau2Tool]:
    """
    Get all tools for a tau2-bench domain.

    Args:
        domain: Domain name (retail, airline, telecom)
        solo_mode: Whether to use solo mode (only supported by telecom)

    Returns:
        List of tau2-bench Tool instances
    """
    if not TAU2_AVAILABLE:
        raise ImportError("tau2-bench is not installed")

    # Get environment constructor
    if domain == "telecom":
        # Telecom has two policy types, default to manual
        env_constructor = registry.get_env_constructor(domain)
    else:
        env_constructor = registry.get_env_constructor(domain)

    # Create environment to get tools
    try:
        env = env_constructor(solo_mode=solo_mode)
    except TypeError:
        # Some domains don't support solo_mode parameter
        env = env_constructor()

    # Get tools
    tools = list(env.get_tools())
    # Native AgentGymEnv adds this stop action to the domain tools. Keep the
    # actor's tool schemas consistent with the actual native Gym agent.
    if not any(tool.name == "done" for tool in tools):
        from tau2.environment.tool import as_tool
        from tau2.gym.gym_agent import done
        tools.append(as_tool(done))
    return tools


def get_tool_schemas_for_domain(domain: str, solo_mode: bool = False) -> List[Dict[str, Any]]:
    """
    Get tool schemas (OpenAI format) for a domain.

    Args:
        domain: Domain name (retail, airline, telecom)
        solo_mode: Whether to use solo mode

    Returns:
        List of tool schemas in OpenAI format
    """
    tools = get_tools_for_domain(domain, solo_mode)
    schemas = [tool.openai_schema for tool in tools]
    return schemas


def create_userrl_tools_for_domain(
    domain: str,
    solo_mode: bool = False,
    include_send_message: bool = False
) -> List[Any]:
    """
    Create UserRL BaseTool instances for a domain.

    Args:
        domain: Domain name (retail, airline, telecom)
        solo_mode: Whether to use solo mode
        include_send_message: Whether to include send_message tool (deprecated, not needed)

    Returns:
        List of UserRL BaseTool instances

    Note:
        send_message tool is not needed because tau2-bench's parse_action_string()
        can handle plain text directly. Plain text messages are sent via env.step()
        without requiring a special tool.
    """
    from .tau2_tool_wrapper import convert_tau2_tool_to_userrl

    tools = get_tools_for_domain(domain, solo_mode)
    userrl_tools = [convert_tau2_tool_to_userrl(tool) for tool in tools]

    # Note: We don't add send_message tool because:
    # 1. tau2-bench's parse_action_string() handles plain text natively
    # 2. In evaluation, models generate plain text directly (not via a tool)
    # 3. Plain text is sent via env.step() which calls parse_action_string()

    return userrl_tools


def get_tool_schema_dict_for_domain(domain: str, solo_mode: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Get tool schemas as a dictionary (tool_name -> schema).

    Args:
        domain: Domain name
        solo_mode: Whether to use solo mode

    Returns:
        Dictionary mapping tool names to their schemas

    Note:
        Does not include send_message tool because tau2-bench handles plain text
        natively via parse_action_string().
    """
    tools = get_tools_for_domain(domain, solo_mode)
    schemas = {tool.name: tool.openai_schema for tool in tools}

    # Note: We don't add send_message tool schema because:
    # - tau2-bench's parse_action_string() handles plain text natively
    # - Models can generate plain text directly without a tool
    # - This matches the evaluation setup

    return schemas


def verify_tools_across_domains() -> Dict[str, Any]:
    """
    Verify that tools are correctly loaded for all domains.

    Returns:
        Dictionary with verification results for each domain
    """
    if not TAU2_AVAILABLE:
        return {"error": "tau2-bench not installed"}

    domains = ["retail", "airline", "telecom"]
    results = {}

    for domain in domains:
        try:
            tools = get_tools_for_domain(domain, solo_mode=False)
            tool_names = [tool.name for tool in tools]

            results[domain] = {
                "status": "success",
                "num_tools": len(tools),
                "tool_names": tool_names,
                "schemas": [tool.openai_schema for tool in tools]
            }

            logger.info(f"Domain {domain}: {len(tools)} tools loaded")
            for tool_name in tool_names:
                logger.info(f"  - {tool_name}")

        except Exception as e:
            results[domain] = {
                "status": "error",
                "error": str(e)
            }
            logger.error(f"Error loading tools for {domain}: {e}")

    return results
