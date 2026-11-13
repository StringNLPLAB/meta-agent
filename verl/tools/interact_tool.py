from typing import Any, Optional, Tuple
import asyncio
import json
import os
import re

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema
from .env_manager import get_environment_manager

class InteractTool(BaseTool):
    """A tool for interacting with environments across multi-turn conversations.

    - `create`: create environment for a conversation (request_id)
    - `execute`: interact with the persistent environment
    - `calc_reward`: calculate reward from environment state
    - `release`: clean up environment and conversation state
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._conversation_data = {}  # request_id -> conversation state
        self._env_manager = get_environment_manager()
        # Control turn-by-turn logging (default: False to reduce noise during training)
        self.verbose_turns = self.config.get("verbose_turns", False)
        self.disable_search = bool(self.config.get("disable_search", False))
        self.env_step_timeout = float(
            os.getenv("USER_ENV_STEP_TIMEOUT", self.config.get("env_step_timeout", 90.0))
        )

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: str, env_name: Optional[str] = None, max_turns: int = 15, **kwargs) -> str:
        """Create environment and initialize conversation state.

        Args:
            instance_id: Request ID for the conversation (serves as conversation identifier)
            env_name: Type of environment to create
            max_turns: Maximum number of interaction turns
            **kwargs: Environment-specific configuration

        Returns:
            instance_id (request_id)
        """
        if instance_id in self._conversation_data:
            # Suppress noisy duplicate-create logs
            return instance_id

        # Create environment through environment manager
        if env_name:
            kwargs["max_turns"] = max_turns
            self._env_manager.create_environment(instance_id, env_name, **kwargs)

        # Initialize conversation state (separate from environment)
        self._conversation_data[instance_id] = {
            "history": [],
            "reward": 0.0,
            "ground_truth": kwargs.get("ground_truth"),
            "env_name": env_name,
        }

        # Suppress verbose creation log to reduce noise during debugging
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], current_turns, **kwargs) -> Tuple[str, float, dict]:
        """Execute action in the persistent environment.

        Args:
            instance_id: Request ID (conversation identifier)
            parameters: Action parameters (choice, content)

        Returns:
            (response_text, step_reward, is_terminated)
        """

        if instance_id not in self._conversation_data:
            raise ValueError(f"Conversation {instance_id} not found. Call create() first.")

        # Get persistent environment
        env = self._env_manager.get_environment(instance_id)
        if env is None:
            raise ValueError(f"Environment for conversation {instance_id} not found")

        # Get environment name from conversation state
        conversation_state = self._conversation_data[instance_id]
        current_env_name = conversation_state.get("env_name", "")

        # Parse action parameters
        choice = str(parameters.get("choice", ""))
        raw_content = parameters.get("content", "")
        if isinstance(raw_content, (dict, list)):
            content = json.dumps(raw_content, separators=(",", ":"))
        else:
            content = str(raw_content)

        def _maybe_extract_interact_content(raw: str) -> str:
            """
            ColBench training SHOULD be pure-text (like eval/colbench/run_simulate_api.py),
            but some runs still produce tool-call wrappers in the model output, e.g.:
              - <tool_call>{"name":"interact_with_env","arguments":{"choice":"action","content":"..."}}</tool_call>
              - <|functions.interact_with_env |{"choice":"action","content":"..."}|>
            For ColBench direct-text stepping, strip wrappers and keep the inner `content` when possible.
            """
            if not raw:
                return raw

            # Case 1: XML-like <tool_call> ... </tool_call>
            if "<tool_call>" in raw and "</tool_call>" in raw:
                m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", raw, flags=re.DOTALL)
                if m:
                    try:
                        payload = json.loads(m.group(1))
                        # Expected: {"name":"interact_with_env","arguments":{"choice":"action","content":"..."}}
                        args = payload.get("arguments", {}) if isinstance(payload, dict) else {}
                        inner = args.get("content")
                        if isinstance(inner, str) and inner.strip():
                            return inner.strip()
                    except Exception:
                        pass

            # Case 2: SGLang tool-call marker: <|functions.interact_with_env |{...}|>
            # Extract the first balanced JSON object after the marker.
            marker = "<|functions.interact_with_env"
            if marker in raw:
                start_match = re.search(r"<\|functions\.interact_with_env\s*\|\s*(\{)", raw)
                if start_match:
                    start_pos = start_match.start(1)
                    brace_count = 0
                    i = start_pos
                    while i < len(raw):
                        if raw[i] == "{":
                            brace_count += 1
                        elif raw[i] == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                json_str = raw[start_pos : i + 1]
                                try:
                                    payload = json.loads(json_str)
                                    inner = payload.get("content") if isinstance(payload, dict) else None
                                    if isinstance(inner, str) and inner.strip():
                                        return inner.strip()
                                except Exception:
                                    pass
                                break
                        i += 1

            return raw.strip()

        # ========== COLBENCH & TAU2GYM SPECIAL HANDLING: Direct text interaction (like sweet_rl) ==========
        # For ColBench and Tau2Gym, use content directly without any prefix formatting
        # This matches sweet_rl's behavior where agent response is pure text
        # Tau2Gym uses tau2-bench's parse_action_string which handles JSON/functional/plain text automatically
        if current_env_name == "ColBenchCodeEnv":
            # ColBench: Use content directly as action (no [action] prefix, no tool call format)
            # This matches sweet_rl's step(response) where response is pure text
            formatted_action = _maybe_extract_interact_content(content)
        elif current_env_name == "Tau2Gym":
            # Tau2Gym: keep the tool schema's own `choice` enum. Passing only
            # `content` threw it away, so every action reached the env as prose
            # and Tau2Env.parse_action_string could only classify it as a
            # message. The agent then never executed a single tool -- it
            # hallucinated the results instead ("I have located your user
            # profile, your user ID is ...") -- and with no tool calls the
            # action-based reward was 0 on every episode of the run.
            body = content.strip()
            if choice in ("message", "tool_call", "done"):
                formatted_action = f"[{choice}] {body}"
            else:
                formatted_action = body
        else:
            # Other environments: Use tool call format with prefixes
            if choice == "action" and not content.startswith("[action]"):
                formatted_action = "[action] " + content
            elif choice == "answer" and not content.startswith("[answer]"):
                formatted_action = "[answer] " + content
            elif choice == "search" and not content.startswith("[search]"):
                formatted_action = "[search] " + content
            elif choice == "finish":
                formatted_action = "[finish]"
            else:
                formatted_action = content
        # ========== END COLBENCH SPECIAL HANDLING ==========

        if self.disable_search and choice == "search":
            # The no-search benchmark variant removes this action entirely. End
            # malformed calls without sending them to an external search backend.
            observation = {"feedback": "Search is disabled for this rollout."}
            reward, terminated, truncated, info = 0.0, True, False, {"search_disabled": True}
        else:
            try:
                # Add timeout to prevent hanging for too long
                # Environment steps can fan out into multiple user-sim calls.
                observation, reward, terminated, truncated, info = await asyncio.wait_for(
                    env.step_async(formatted_action),
                    timeout=self.env_step_timeout
                )
            except asyncio.TimeoutError:
                print(
                    f"Environment step timed out for {instance_id} after "
                    f"{self.env_step_timeout:g}s; returning terminal fallback"
                )
                observation = {"feedback": "Environment operation timed out"}
                reward, terminated, truncated, info = 0.0, True, False, {"timeout": True}
            except Exception as e:
                print(f"Environment step failed for {instance_id}: {e}")
                # Return safe fallback values
                observation = {"feedback": f"Error: {str(e)}"}
                reward, terminated, truncated, info = 0.0, True, False, {}

        # Update conversation state (conversation_state and current_env_name already retrieved above)
        conversation_state["reward"] = reward
        conversation_state["history"].append({
            "choice": choice,
            "content": content,
            "observation": observation,
            "reward": reward,
            "info": info
        })

        # Format response
        feedback = observation.get("feedback", "") if isinstance(observation, dict) else str(observation)
        response_text = f"{feedback}\nReward: {reward}"

        is_done = terminated or truncated
        # Only print turn-by-turn logs if verbose_turns is enabled
        if self.verbose_turns:
            print(f"Turn {current_turns}: Executed {choice} in conversation {instance_id} (Env: {current_env_name}), action: {formatted_action}, feedback: {feedback}, reward: {reward}, done: {is_done}")

        # Calculate where pure observation (feedback) appears in response_text
        # This provides character-level offsets that rollout can use to find exact token boundaries
        obs_char_start = response_text.find(feedback)
        obs_char_end = obs_char_start + len(feedback) if obs_char_start != -1 else len(response_text)

        # Return offset information for accurate boundary recording
        metrics = {
            "pure_observation": feedback,
            "obs_char_start": obs_char_start,  # Where observation starts in response_text (character-level)
            "obs_char_end": obs_char_end,      # Where observation ends in response_text (character-level)
        }

        return response_text, reward, is_done, choice, content, metrics

    def _run_env_in_process(self, env, formatted_action):
        """Run environment step in separate process to isolate from NCCL context."""
        try:
            # Use synchronous step since we're in a separate process
            if hasattr(env, 'step'):
                return env.step(formatted_action)
            else:
                # If only async available, run in new event loop
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(env.step_async(formatted_action))
        except Exception as e:
            return {"feedback": f"Process error: {e}"}, 0.0, True, False, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate final reward for the conversation.

        Args:
            instance_id: Request ID (conversation identifier)

        Returns:
            Final conversation reward
        """
        if instance_id not in self._conversation_data:
            print(f"!!!!!!!! Conversation {instance_id} not found for reward calculation !!!!!!!!")
            return 0.0

        conversation_state = self._conversation_data[instance_id]
        env_name = conversation_state.get("env_name", "")

        # For ColBenchCodeEnv, use environment's reward calculation method
        if env_name == "ColBenchCodeEnv":
            env = self._env_manager.get_environment(instance_id)
            if env is None:
                print(f"[WARNING] ColBench env not found for instance {instance_id}")
                return 0.0
            if not hasattr(env, 'calculate_reward'):
                print(f"[WARNING] ColBench env {instance_id} does not have calculate_reward method")
                return 0.0
            try:
                # Check episode state before calculating reward
                episode_complete = getattr(env, 'episode_complete', False)
                agent_answer = getattr(env, 'agent_answer', 'No answer')
                step_count = getattr(env, 'step_count', 0)
                dialogue_history = getattr(env, 'dialogue_history', [])

                # Get last few messages for debugging
                last_messages = dialogue_history[-4:] if len(dialogue_history) >= 4 else dialogue_history
                last_messages_str = "\n".join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')[:100]}" for msg in last_messages])

                # print(f"[DEBUG] ColBench calc_reward for {instance_id}: episode_complete={episode_complete}, step_count={step_count}, agent_answer_length={len(agent_answer) if agent_answer else 0}, agent_answer_preview={agent_answer[:100] if agent_answer else 'None'}...")
                # print(f"[DEBUG] ColBench dialogue history (last 4 messages):\n{last_messages_str}")

                reward = env.calculate_reward()

                # Check state after reward calculation
                episode_complete_after = getattr(env, 'episode_complete', False)
                agent_answer_after = getattr(env, 'agent_answer', 'No answer')
                # print(f"[DEBUG] ColBench reward calculated: {reward} for instance {instance_id} (episode_complete changed: {episode_complete} -> {episode_complete_after}, agent_answer_length: {len(agent_answer) if agent_answer else 0} -> {len(agent_answer_after) if agent_answer_after else 0})")
                if self.verbose_turns:
                    print(f"ColBench reward calculated: {reward} (agent_answer length: {len(env.agent_answer) if hasattr(env, 'agent_answer') else 0})")
                return float(reward)
            except Exception as e:
                print(f"Error calculating ColBench reward for {instance_id}: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to history-based reward
                pass

        # For other environments, return the highest reward achieved during the conversation
        if conversation_state["history"]:
            max_reward = max(step["reward"] for step in conversation_state["history"])
            return max_reward

        return conversation_state["reward"]

    async def get_evaluation_metrics(self, instance_id: str, **kwargs) -> dict[str, float]:
        """Return paper-compatible evaluation metrics without changing training reward semantics."""
        state = self._conversation_data.get(instance_id)
        if state is None:
            return {}

        history = state.get("history", [])
        step_rewards = [float(step.get("reward", 0.0) or 0.0) for step in history]
        step_reward_sum = float(sum(step_rewards))
        step_reward_max = float(max(step_rewards, default=0.0))
        answer_rewards = [
            float(step.get("reward", 0.0) or 0.0)
            for step in history
            if step.get("choice") == "answer"
        ]
        final_answer_score = float(answer_rewards[-1]) if answer_rewards else 0.0
        env_name = state.get("env_name", "")
        env = self._env_manager.get_environment(instance_id)

        # UserRL evaluates TravelGym and TauGym by their final correctness. Every
        # other Gym reports cumulative incremental progress as the sum of its turn
        # rewards, including FunctionGym, TelepathyGym, and SearchGym.
        paper_score = step_reward_sum
        if env_name == "TravelGym":
            paper_score = self._travel_choice_correctness(env, history)
        elif env_name in {"TauGym", "Tau2Gym"}:
            paper_score = self._tau_task_completion(env)

        return {
            "interact_with_env_eval_score": float(paper_score),
            "interact_with_env_step_reward_sum": step_reward_sum,
            "interact_with_env_step_reward_max": step_reward_max,
            "interact_with_env_final_answer_score": final_answer_score,
            "interact_with_env_tool_call_count": float(len(history)),
        }

    @staticmethod
    def _travel_choice_correctness(env: Any, history: list[dict[str, Any]]) -> float:
        """Compute the UserBench score from the best submitted choice per aspect."""
        if env is None or not getattr(env, "current_task", None):
            return 0.0

        task = env.current_task
        dimensions = task.get("dimensions", [])
        if not dimensions:
            return 0.0

        # Match UserRL's eval/analyze.py: one answer per travel aspect counts,
        # using the highest recorded environment reward for that aspect. This
        # preserves partial credit for a correct but non-best option (0.8).
        reward_by_aspect: dict[str, float] = {}
        for step in history:
            content = str(step.get("content", "")).strip()
            if step.get("choice") != "answer" or not content:
                continue
            match = re.search(r"\b([ACFHR])\d+\b", content)
            if match is None:
                continue
            aspect = match.group(1)
            reward_by_aspect[aspect] = max(
                reward_by_aspect.get(aspect, 0.0),
                float(step.get("reward", 0.0) or 0.0),
            )

        return float(sum(reward_by_aspect.values()) / len(dimensions))

    @staticmethod
    def _tau_task_completion(env: Any) -> float:
        """Use TauBench's terminal task reward rather than an earlier intermediate reward."""
        if env is None:
            return 0.0
        last_response = getattr(env, "last_tau_response", None)
        if last_response is None:
            return 0.0
        return float(getattr(last_response, "reward", 0.0) or 0.0)

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up conversation and environment.

        Args:
            instance_id: Request ID (conversation identifier)
        """
        # Clean up conversation state
        if instance_id in self._conversation_data:
            del self._conversation_data[instance_id]

        # Clean up environment through manager
        self._env_manager.release_environment(instance_id)
