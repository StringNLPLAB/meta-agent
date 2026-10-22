"""
Tau2ToolWrapper: Wrapper for tau2-bench tools to work with UserRL's BaseTool interface.

This module provides tools for interacting with tau2-bench environments using
direct tool calling (instead of interact_with_env wrapper).
"""

from typing import Any, Optional, Tuple
import json
import logging
from uuid import uuid4
from copy import deepcopy

from pydantic import ConfigDict

from .base_tool import BaseTool
from .schemas import OpenAIFunctionSchema, OpenAIFunctionToolSchema
from .env_manager import get_environment_manager

logger = logging.getLogger(__file__)


class NativeFunctionSchema(OpenAIFunctionSchema):
    """Keep complete native JSON Schema, including refs and array definitions."""

    model_config = ConfigDict(extra="allow")
    parameters: dict[str, Any]


class NativeToolSchema(OpenAIFunctionToolSchema):
    """Compatible typed function names with lossless schema serialization."""

    model_config = ConfigDict(extra="allow")
    function: NativeFunctionSchema

    def model_dump(self, *args, **kwargs):
        # Optional compatibility-model defaults are not native JSON Schema.
        # Preserve intentionally supplied null/default values inside parameters.
        kwargs.setdefault("exclude_unset", True)
        return super().model_dump(*args, **kwargs)

try:
    from tau2.environment.tool import Tool as Tau2Tool
    from tau2.data_model.message import ToolCall
    TAU2_AVAILABLE = True
except ImportError:
    TAU2_AVAILABLE = False
    Tau2Tool = None


class Tau2ToolWrapper(BaseTool):
    """Wrapper for tau2-bench tools to work with UserRL's BaseTool interface.

    This wrapper allows tau2-bench tools to be used directly in UserRL training,
    matching the evaluation setup where tools are called directly via OpenAI tool calling.

    The wrapper:
    1. Converts tau2-bench tool's openai_schema to UserRL's OpenAIFunctionToolSchema
    2. Executes tools by calling them through the tau2-bench environment
    3. Returns results in UserRL's expected format
    """

    def __init__(self, tau2_tool: Tau2Tool, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        Initialize the wrapper.

        Args:
            tau2_tool: The tau2-bench Tool instance
            config: Tool configuration (not used currently)
            tool_schema: UserRL's OpenAIFunctionToolSchema (converted from tau2_tool.openai_schema)
        """
        if not TAU2_AVAILABLE:
            raise ImportError("tau2-bench is not installed. Please install it first.")

        self.tau2_tool = tau2_tool
        self._env_manager = get_environment_manager()

        super().__init__(config, tool_schema)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Get the OpenAI tool schema."""
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """
        Create tool instance (for compatibility with BaseTool interface).

        For tau2 tools, we don't need to create instances separately.
        The instance_id corresponds to the request_id (conversation identifier).

        Note: The environment should already be created via _tau2_env_kwargs in rollout.
        This method is a no-op for tau2 tools.
        """
        if instance_id is None:
            instance_id = str(uuid4())
        # Environment creation is handled separately in rollout via _tau2_env_kwargs
        return instance_id

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        current_turns: Optional[int] = None,
        **kwargs
    ) -> Tuple[str, float, dict]:
        """
        Execute the tau2-bench tool.

        Args:
            instance_id: Request ID (conversation identifier)
            parameters: Tool parameters (from tool call)
            current_turns: Current turn number (not used currently)
            **kwargs: Additional arguments

        Returns:
            Tuple of (response_text, step_reward, metrics)
        """
        # Get the tau2-bench environment
        tau2_env = self._env_manager.get_environment(instance_id)
        if tau2_env is None:
            raise ValueError(f"Tau2 environment for {instance_id} not found. Call create() first.")

        # Get the underlying tau2-bench AgentGymEnv
        if not hasattr(tau2_env, 'tau2_env') or tau2_env.tau2_env is None:
            raise ValueError(f"Tau2-bench environment not initialized for {instance_id}")

        tau2_gym_env = tau2_env.tau2_env

        # Execute the tool through tau2-bench environment
        # tau2-bench's AgentGymEnv.step() accepts action strings that are parsed by parse_action_string()
        # We need to format the tool call as a JSON string and pass it to env.step()

        try:
            # Check if episode is complete before executing tool
            if hasattr(tau2_env, 'episode_complete') and tau2_env.episode_complete:
                logger.warning(
                    f"[TAU2_TOOL_WRAPPER] Request {instance_id}: "
                    f"Episode is already complete, cannot execute tool '{self.tau2_tool.name}'"
                )
                return "Episode is complete. Cannot execute tool.", 0.0, {
                    "tool_name": self.tau2_tool.name,
                    "episode_complete": True,
                }

            # Format tool call as JSON string (tau2-bench's expected format)
            tool_call_json = {
                "name": self.tau2_tool.name,
                "arguments": parameters
            }
            action_string = json.dumps(tool_call_json)

            # Execute tool by calling env.step() with the formatted action string
            # This will be parsed by parse_action_string() and executed by the orchestrator
            observation, reward, terminated, truncated, info = await tau2_env.step_async(action_string)

            # Format response
            if isinstance(observation, dict):
                feedback = observation.get("feedback", "")
            else:
                feedback = str(observation)

            # tau2-bench returns tool execution results in the observation
            # The observation contains the tool response and any subsequent user messages
            response_text = feedback

            # Get step reward (tau2-bench gives sparse rewards, usually 0 during execution)
            step_reward = float(reward)

            # Return metrics
            metrics = {
                "tool_name": self.tau2_tool.name,
                "terminated": terminated,
                "truncated": truncated,
            }

            return response_text, step_reward, metrics

        except Exception as e:
            logger.error(f"Error executing tau2 tool {self.tau2_tool.name}: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError("Native tau3 tool/environment execution failed") from e

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """
        Calculate final reward (not used for individual tools).

        Tau2-bench calculates reward at episode end based on task completion,
        not per-tool execution.
        """
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release tool instance (no-op for tau2 tools)."""
        pass


class SendMessageTool(BaseTool):
    """
    Special tool for sending plain text messages in tau2-bench.

    Tau2-bench allows agents to send messages directly to users (not via tool calls).
    This tool wraps that functionality to work with UserRL's tool calling system.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize the send_message tool."""
        super().__init__(config, tool_schema)
        self._env_manager = get_environment_manager()

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create tool instance."""
        if instance_id is None:
            instance_id = str(uuid4())
        return instance_id

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        current_turns: Optional[int] = None,
        **kwargs
    ) -> Tuple[str, float, dict]:
        """
        Send a message to the user.

        Args:
            instance_id: Request ID (conversation identifier)
            parameters: Must contain "message" field with the message text
            current_turns: Current turn number
            **kwargs: Additional arguments

        Returns:
            Tuple of (response_text, step_reward, metrics)
        """
        message_content = parameters.get("message", "")
        if not message_content:
            return "Error: message parameter is required", 0.0, {}

        # Get the tau2-bench environment
        tau2_env = self._env_manager.get_environment(instance_id)
        if tau2_env is None:
            raise ValueError(f"Tau2 environment for {instance_id} not found")

        tau2_gym_env = tau2_env.tau2_env
        if tau2_gym_env is None:
            raise ValueError(f"Tau2-bench environment not initialized")

        # Send message by calling env.step() with plain text
        # tau2-bench's parse_action_string handles plain text messages
        try:
            observation, reward, terminated, truncated, info = await tau2_env.step_async(message_content)

            # Format response
            if isinstance(observation, dict):
                feedback = observation.get("feedback", "")
            else:
                feedback = str(observation)

            response_text = feedback
            step_reward = float(reward)

            metrics = {
                "tool_name": "send_message",
                "terminated": terminated,
                "truncated": truncated,
            }

            return response_text, step_reward, metrics

        except Exception as e:
            logger.error(f"Error sending message: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError("Native tau3 user simulation failed") from e

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate reward (not used for send_message)."""
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release tool instance (no-op)."""
        pass


def convert_tau2_tool_to_userrl(
    tau2_tool: Tau2Tool,
    config: Optional[dict] = None
) -> BaseTool:
    """
    Convert a tau2-bench tool to UserRL BaseTool.

    Args:
        tau2_tool: The tau2-bench Tool instance
        config: Optional tool configuration

    Returns:
        UserRL BaseTool instance (Tau2ToolWrapper)
    """
    if not TAU2_AVAILABLE:
        raise ImportError("tau2-bench is not installed")

    # Convert tau2 tool's openai_schema to UserRL's OpenAIFunctionToolSchema
    tau2_schema = deepcopy(tau2_tool.openai_schema)
    userrl_schema = NativeToolSchema.model_validate(tau2_schema)

    # Create wrapper
    return Tau2ToolWrapper(
        tau2_tool=tau2_tool,
        config=config or {},
        tool_schema=userrl_schema
    )


def create_send_message_tool() -> BaseTool:
    """
    Create the send_message tool for tau2-bench.

    Returns:
        SendMessageTool instance
    """
    from .schemas import OpenAIFunctionToolSchema

    schema = OpenAIFunctionToolSchema(
        type="function",
        function={
            "name": "send_message",
            "description": "Send a message to the user in the conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message content to send to the user."
                    }
                },
                "required": ["message"]
            }
        }
    )

    return SendMessageTool(config={}, tool_schema=schema)
