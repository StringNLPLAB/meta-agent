"""
Environment Manager for Multi-turn Interactions
Handles environment lifecycle and persistence across conversation turns.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class RepetitionPricingWrapper:
    """Optionally charge a bounded penalty for repeated environment feedback.

    Only configured action channels are charged. Short replies and excluded
    gyms are ignored; a zero penalty leaves rewards unchanged. The optional
    symbolic_layer helper provides live settings when installed.
    """

    def __init__(self, env: Any, env_name: str):
        self.env = env
        self.env_name = env_name
        self._recent: List[str] = []
        self._n_priced = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def _knobs(self):
        try:
            from symbolic_layer.env_policy import repetition_knobs
            return repetition_knobs()
        except Exception:  # noqa: BLE001
            return {"penalty": 0.0, "window": 4, "min_chars": 40, "exclude_gyms": []}

    def _price(self, observation, reward, action_input=None):
        cfg = self._knobs()
        penalty = float(cfg.get("penalty", 0.0) or 0.0)
        if penalty <= 0.0 or self.env_name in (cfg.get("exclude_gyms") or []):
            return reward
        choices = cfg.get("charge_choices") or ["action"]
        act = str(action_input or "").strip()
        if not any(act.startswith(f"[{c}]") for c in choices):
            return reward
        if isinstance(observation, dict):
            text = observation.get("feedback", "")
            if not isinstance(text, str):
                text = str(text)
        elif isinstance(observation, str):
            text = observation
        else:
            text = str(observation)
        text = text.strip()
        if len(text) < int(cfg.get("min_chars", 40)):
            return reward
        window = max(int(cfg.get("window", 4)), 1)
        repeated = text in self._recent[-window:]
        self._recent.append(text)
        if len(self._recent) > 64:
            del self._recent[:-64]
        if not repeated:
            return reward
        self._n_priced += 1
        try:
            return reward - penalty
        except Exception:  # noqa: BLE001
            return reward

    def step(self, action_input: str):
        out = self.env.step(action_input)
        return self._apply(out, action_input)

    async def step_async(self, action_input: str):
        if hasattr(self.env, "step_async"):
            out = await self.env.step_async(action_input)
        else:
            out = self.env.step(action_input)
        return self._apply(out, action_input)

    def _apply(self, out, action_input=None):
        # Gym step returns (observation, reward, terminated, truncated, info).
        # Anything else is passed through untouched rather than guessed at.
        if not (isinstance(out, tuple) and len(out) == 5):
            return out
        observation, reward, terminated, truncated, info = out
        return observation, self._price(observation, reward, action_input), terminated, truncated, info


class TemplateUserSimWrapper:
    """Deterministic replacement for LLM-backed user/judge turns.

    The wrapper keeps the underlying Gym for structured search/tool execution
    and final environment bookkeeping, but intercepts open-ended user-sim turns
    that otherwise require a chat model. This is intentionally conservative:
    unsupported actions fall back to the wrapped environment.
    """

    def __init__(self, env: Any, env_name: str):
        self.env = env
        self.env_name = env_name

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def step(self, action_input: str):
        action = str(action_input).strip()
        if self._should_template(action):
            return self._template_step(action)
        return self.env.step(action_input)

    async def step_async(self, action_input: str):
        action = str(action_input).strip()
        if self._should_template(action):
            return self._template_step(action)
        if hasattr(self.env, "step_async"):
            return await self.env.step_async(action_input)
        return self.env.step(action_input)

    def _should_template(self, action: str) -> bool:
        if action.startswith("[search]"):
            return self.env_name == "TravelGym"
        if action.startswith("[action]"):
            return self.env_name in {"TravelGym", "TauGym", "PersuadeGym", "TurtleGym"}
        if action.startswith("[answer]"):
            return self.env_name in {"PersuadeGym", "TurtleGym"}
        return False

    def _template_step(self, action: str):
        if self.env_name == "TravelGym":
            if action.startswith("[search]"):
                return self._travel_search_step(action)
            return self._travel_step(action)
        if self.env_name == "TauGym":
            return self._tau_step(action)
        if self.env_name == "PersuadeGym":
            return self._persuade_step(action)
        if self.env_name == "TurtleGym":
            return self._turtle_step(action)
        return self.env.step(action)

    def _bump_step(self, action: str) -> None:
        if getattr(self.env, "episode_complete", False):
            raise ValueError("Episode is complete. Call reset() to start a new episode.")
        self.env.step_count = getattr(self.env, "step_count", 0) + 1
        if hasattr(self.env, "action_history"):
            self.env.action_history.append(action)

    def _finish_common(self, reward: float, terminated: bool = False):
        truncated = getattr(self.env, "step_count", 0) >= getattr(self.env.config, "max_steps", 10**9)
        if terminated or truncated:
            self.env.episode_complete = True
        self.env.total_reward = getattr(self.env, "total_reward", 0.0) + float(reward)
        return truncated

    def _travel_step(self, action: str):
        self._bump_step(action)
        utterance = action[len("[action]") :].strip()
        pref = self._select_travel_preference(utterance)
        elicited_ids = []
        if pref is not None:
            pref["elicited"] = True
            elicited_ids = [pref["id"]]
            self.env.elicited_preferences.append(pref)
            self.env.remaining_preferences = [
                item for item in self.env.remaining_preferences if item.get("id") != pref["id"]
            ]
            self.env.state_list["active_elicited_preferences"] += 1
            self.env.state_list["nonpreference_times"] = 0
            feedback = pref.get("implicit_elicitation") or f"I prefer {pref.get('preference', 'that option')}."
            reward = float(self.env.config.preference_correct_reward)
            note = f"Template user revealed preference {pref['id']}."
        else:
            self.env.state_list["nonpreference_times"] += 1
            feedback = (
                "I do not have a more specific preference for that. Please continue with the "
                "information already provided or ask about another concrete travel aspect."
            )
            reward = 0.0
            note = "Template user had no matching hidden preference."

        reward = reward * self.env.config.reward_scale - self.env.config.step_penalty
        if self.env.config.normalize_rewards:
            reward = max(0.0, min(1.0, reward))
        if hasattr(self.env, "conversation_history"):
            self.env.conversation_history.append({"role": "agent", "content": utterance, "note": note})
            self.env.conversation_history.append({"role": "user", "content": feedback})
        terminated = self.env.state_list["remaining_best_options"] == []
        if self.env.config.one_choice_per_aspect and len(self.env.current_task["dimensions"]) == len(self.env.state_list["choice_initials"]):
            terminated = True
        truncated = self._finish_common(reward, terminated)
        observation = {
            "task_description": self.env.current_task.get("scenario", ""),
            "goal": "Elicit travel preferences and provide appropriate recommendations.",
            "feedback": feedback,
            "step_count": self.env.step_count,
            "episode_complete": self.env.episode_complete,
            "total_preferences": len(self.env.remaining_preferences) + len(self.env.elicited_preferences),
            "remaining_preferences": len(self.env.remaining_preferences),
            "elicitation_ratio": self.env._calculate_elicitation_ratio(),
            "active_elicited_preferences": self.env.state_list["active_elicited_preferences"],
            "passive_elicited_preferences": self.env.state_list["passive_elicited_preferences"],
            "last_reward": reward,
        }
        info = {
            "template_user_sim": True,
            "elicited_preference_ids": elicited_ids,
            "action_history": self.env.action_history.copy(),
            "conversation_history": getattr(self.env, "conversation_history", []).copy(),
            "total_reward": self.env.total_reward,
        }
        return observation, reward, terminated, truncated, info

    def _travel_search_step(self, action: str):
        self._bump_step(action)
        query = action[len("[search]") :].strip()
        self.env.state_list["search_times"] += 1
        aspect = self._select_travel_search_aspect(query)
        if aspect is None:
            feedback = (
                "There's something wrong within your searching request, or the arguments in your "
                "request are not accurate. Please make one search request for one travel aspect "
                "using the task's concrete city/date/origin/destination arguments."
            )
            reward = 0.0
        elif aspect not in self.env.state_list["search_arguments"]:
            feedback = (
                f"You have already got the search results for <{aspect}> in previous search attempts. "
                "Please directly refer to the previous search results."
            )
            reward = 0.0
        else:
            self.env.state_list["search_arguments"].remove(aspect)
            response_options = self.env.current_task["all_options"][aspect]
            schema = self._travel_option_schema(aspect)
            feedback = f"You have provided the correct search request arguments.\n\n{schema}\n\nHere are all the options for <{aspect}>:\n"
            for option in response_options:
                feedback += f"{self._json_dumps(option)}\n"
            feedback = feedback.strip()
            reward = float(self.env.config.search_correct_reward)

        if hasattr(self.env, "conversation_history"):
            self.env.conversation_history.append({"role": "agent", "content": query})
            db_content = feedback.split("\n")[0] + (" ... (skip detailed results here) ..." if "Here are all the options" in feedback else "")
            self.env.conversation_history.append({"role": "database", "content": db_content})

        reward = reward * self.env.config.reward_scale - self.env.config.step_penalty
        if self.env.config.normalize_rewards:
            reward = max(0.0, min(1.0, reward))
        terminated = self.env.state_list["remaining_best_options"] == []
        if self.env.config.one_choice_per_aspect and len(self.env.current_task["dimensions"]) == len(self.env.state_list["choice_initials"]):
            terminated = True
        truncated = self._finish_common(reward, terminated)
        observation = {
            "task_description": self.env.current_task.get("scenario", ""),
            "goal": "Elicit travel preferences and provide appropriate recommendations.",
            "feedback": feedback,
            "step_count": self.env.step_count,
            "episode_complete": self.env.episode_complete,
            "total_preferences": len(self.env.remaining_preferences) + len(self.env.elicited_preferences),
            "remaining_preferences": len(self.env.remaining_preferences),
            "elicitation_ratio": self.env._calculate_elicitation_ratio(),
            "active_elicited_preferences": self.env.state_list["active_elicited_preferences"],
            "passive_elicited_preferences": self.env.state_list["passive_elicited_preferences"],
            "last_reward": reward,
        }
        info = {
            "template_user_sim": True,
            "template_search_aspect": aspect,
            "action_history": self.env.action_history.copy(),
            "conversation_history": getattr(self.env, "conversation_history", []).copy(),
            "total_reward": self.env.total_reward,
        }
        return observation, reward, terminated, truncated, info

    def _select_travel_preference(self, utterance: str):
        available = list(getattr(self.env, "remaining_preferences", []) or [])
        if not available:
            return None
        text = utterance.lower()
        for pref in available:
            fields = " ".join(
                str(pref.get(key, "")) for key in ("aspect", "subcategory", "dimension", "preference")
            ).lower()
            if any(token and token in text for token in re.split(r"[^a-z0-9]+", fields) if len(token) >= 4):
                return pref
        return available[0]

    def _select_travel_search_aspect(self, query: str):
        text = query.lower().replace("-", "_")
        aliases = {
            "flight": ("flight", "airline", "plane"),
            "hotel": ("hotel",),
            "apartment": ("apartment", "rental apartment", "lodging"),
            "rental_car": ("rental_car", "rental car", "car rental", "car"),
            "restaurant": ("restaurant", "dining", "food"),
        }
        dimensions = list(self.env.current_task.get("dimensions", []))
        matches = []
        for aspect in dimensions:
            for alias in aliases.get(aspect, (aspect,)):
                if alias in text:
                    matches.append(aspect)
                    break
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return None
        for aspect, args in self.env.current_task.get("arguments", {}).items():
            if aspect not in dimensions:
                continue
            values = [str(value).lower() for value in args.values()]
            if values and all(value in text for value in values if value):
                return aspect
        return None

    def _travel_option_schema(self, aspect: str) -> str:
        try:
            from travelgym.env.prompt_async import OPTION_SCHEMAS
        except Exception:
            try:
                from travelgym.env.prompts import OPTION_SCHEMAS
            except Exception:
                OPTION_SCHEMAS = {}
        return str(OPTION_SCHEMAS.get(aspect, "Each option includes an id and task-specific fields.")).strip()

    def _json_dumps(self, value: Any) -> str:
        import json

        return json.dumps(value, ensure_ascii=False)

    def _tau_step(self, action: str):
        self._bump_step(action)
        content = action[len("[action]") :].strip()
        feedback = (
            "User says: I do not have additional information beyond my original request. "
            "Please use the available tools and proceed with the task."
        )
        reward = 0.0
        terminated = False
        truncated = self._finish_common(reward, terminated)
        observation = {
            "instruction": self.env.current_task.instruction,
            "feedback": feedback,
            "step_count": self.env.step_count,
            "episode_complete": self.env.episode_complete,
            "tools_info": self.env._format_tools_info(),
        }
        info = {
            "template_user_sim": True,
            "raw_action": action,
            "task_id": self.env.config.data_source,
            "task_category": self.env.config.task_category,
            "action_history": self.env.action_history.copy(),
            "total_reward": self.env.total_reward,
            "tau_response": self.env.last_tau_response.model_dump() if self.env.last_tau_response else None,
            "template_user_message": feedback,
            "agent_message": content,
        }
        return observation, reward, terminated, truncated, info

    def _persuade_step(self, action: str):
        self._bump_step(action)
        content = action.split("]", 1)[-1].strip()
        old_score = self._stance_score(getattr(self.env, "current_stance", "Strongly Agree"))
        if action.startswith("[answer]"):
            target_score = self._persuasion_text_score(content)
        else:
            target_score = min(1.0, old_score + self._persuasion_text_score(content) * 0.25)
        reward = max(0.0, target_score - old_score)
        self.env.current_stance = self._score_to_stance(target_score)
        self.env.best_score = max(getattr(self.env, "best_score", 0.0), target_score)
        feedback = self._persuasion_feedback(target_score)
        if hasattr(self.env, "conversation_history"):
            self.env.conversation_history.append((content, feedback))
        terminated = action.startswith("[answer]") and target_score >= 0.8
        truncated = self._finish_common(reward, terminated)
        observation = {
            "statement_title": self.env.current_statement.get("claim", "")[:100],
            "statement_text": self.env.current_statement["claim"],
            "goal": "Try to persuade me to change my position on this statement.",
            "feedback": feedback,
            "step_count": self.env.step_count,
            "episode_complete": self.env.episode_complete,
            "current_score": target_score,
            "best_score": self.env.best_score,
            "current_stance": self.env.current_stance,
        }
        info = {
            "template_user_sim": True,
            "raw_action": action,
            "action_history": self.env.action_history.copy(),
            "conversation_history": self.env.conversation_history.copy(),
            "current_stance": self.env.current_stance,
            "total_reward": self.env.total_reward,
        }
        return observation, reward, terminated, truncated, info

    def _turtle_step(self, action: str):
        self._bump_step(action)
        content = action.split("]", 1)[-1].strip()
        if action.startswith("[answer]"):
            score = self._turtle_answer_score(content)
            reward = max(0.0, score - getattr(self.env, "best_score", 0.0))
            self.env.best_score = max(getattr(self.env, "best_score", 0.0), score)
            feedback = self._turtle_answer_feedback(score)
            terminated = score >= self.env.config.success_threshold
            json_response = {"template_score": score}
        else:
            score = 0.0
            reward = 0.0
            feedback = self._turtle_question_feedback(content)
            terminated = False
            json_response = {"template_feedback": feedback}
        if hasattr(self.env, "judge_history"):
            self.env.judge_history.append(json_response)
        truncated = self._finish_common(reward, terminated)
        observation = {
            "story_title": self.env.current_story["title"],
            "story_text": self.env.current_story["description"],
            "goal": self.env.current_story["goal"],
            "feedback": feedback,
            "step_count": self.env.step_count,
            "episode_complete": self.env.episode_complete,
            "current_score": score,
            "best_score": self.env.best_score,
        }
        info = {
            "template_user_sim": True,
            "raw_action": action,
            "judge_success": True,
            "current_score": score,
            "best_score": self.env.best_score,
            "total_reward": self.env.total_reward,
            "action_history": self.env.action_history.copy(),
            "judge_history": self.env.judge_history.copy(),
        }
        return observation, reward, terminated, truncated, info

    def _persuasion_text_score(self, text: str) -> float:
        tokens = self._tokens(text)
        claim_tokens = self._tokens(self.env.current_statement.get("claim", ""))
        argument_tokens = self._tokens(self.env.current_statement.get("argument", ""))
        overlap = len(tokens & (claim_tokens | argument_tokens))
        structure_bonus = sum(word in tokens for word in {"because", "therefore", "evidence", "cost", "risk", "harm", "benefit"})
        return min(1.0, 0.15 + 0.08 * overlap + 0.08 * structure_bonus + min(len(tokens), 80) / 160.0)

    def _persuasion_feedback(self, score: float) -> str:
        if score >= 0.8:
            return "Your argument addresses my concern clearly. I am willing to change my position."
        if score >= 0.5:
            return "That is somewhat persuasive, but I still need a clearer reason or evidence."
        return "I understand your point, but it has not changed my view yet."

    def _stance_score(self, stance: str) -> float:
        mapping = {
            "strongly agree": 0.0,
            "agree": 0.25,
            "neutral": 0.5,
            "disagree": 0.75,
            "strongly disagree": 1.0,
        }
        return mapping.get(str(stance).lower(), 0.0)

    def _score_to_stance(self, score: float) -> str:
        if score >= 0.875:
            return "Strongly Disagree"
        if score >= 0.625:
            return "Disagree"
        if score >= 0.375:
            return "Neutral"
        if score >= 0.125:
            return "Agree"
        return "Strongly Agree"

    def _turtle_question_feedback(self, text: str) -> str:
        target = " ".join(
            str(self.env.current_story.get(key, ""))
            for key in ("ground_truth", "twist", "solution", "explanation")
        )
        if not target.strip():
            target = str(self.env.current_story.get("description", ""))
        overlap = len(self._tokens(text) & self._tokens(target))
        if overlap >= 2:
            return "Yes, that is related to the hidden explanation."
        if overlap == 1:
            return "Partly related, but it is not the main point."
        return "No, that is not central to the hidden explanation."

    def _turtle_answer_score(self, text: str) -> float:
        criteria = self.env.current_story.get("evaluation_criteria", []) or []
        answer_tokens = self._tokens(text)
        if not criteria:
            target = str(self.env.current_story.get("ground_truth", ""))
            target_tokens = self._tokens(target)
            return min(1.0, len(answer_tokens & target_tokens) / max(1, len(target_tokens) * 0.5))
        covered = 0
        for criterion in criteria:
            criterion_tokens = self._tokens(str(criterion))
            if len(answer_tokens & criterion_tokens) >= max(1, min(3, len(criterion_tokens) // 4)):
                covered += 1
        return covered / max(1, len(criteria))

    def _turtle_answer_feedback(self, score: float) -> str:
        if score >= 1.0:
            return "Your answer covers the hidden explanation."
        if score > 0.0:
            return "Your answer covers part of the hidden explanation, but some key points are missing."
        return "Your answer does not cover the hidden explanation yet."

    def _tokens(self, text: str) -> set[str]:
        stop = {"the", "and", "that", "this", "with", "from", "have", "has", "are", "was", "were", "for", "not"}
        return {tok for tok in re.findall(r"[a-z0-9]+", str(text).lower()) if len(tok) >= 3 and tok not in stop}


class EnvironmentManager:
    """Manages environment instances for multi-turn conversations."""

    def __init__(self):
        self._environments: Dict[str, Any] = {}  # request_id -> environment
        self._env_configs: Dict[str, Dict] = {}   # request_id -> config

    def _apply_difficulty_overrides(self, env_config: Any, env_name: str) -> None:
        """Apply hot-reloadable per-gym difficulty overrides.

        Difficulty fields (how often the search backend fails on purpose, how
        long the user waits before volunteering, how many decoys exist) used to
        be reachable only by restarting. This lets the meta-agent raise them on
        a gym the policy has grown comfortable with, mid-run.

        Absent config = untouched defaults, so this is a no-op until someone
        writes difficulty.json through the gate.
        """
        try:
            from symbolic_layer.difficulty import apply_to_config
        except Exception:  # noqa: BLE001
            return
        try:
            applied = apply_to_config(env_config, env_name)
            if applied:
                logger.info(f"[difficulty] {env_name}: {applied}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[difficulty] override failed for {env_name}: {e}")

    def _apply_llm_env_overrides(self, env_config: Any, **kwargs) -> None:
        """Apply shared OpenAI-compatible settings to Gym configs."""
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        base_url = kwargs.get("base_url") or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
        timeout = kwargs.get("timeout") or os.getenv("USER_GYM_TIMEOUT") or os.getenv("USER_MODEL_TIMEOUT")
        max_tokens = kwargs.get("max_tokens") or os.getenv("USER_GYM_MAX_TOKENS")
        temperature = kwargs.get("temperature") or os.getenv("USER_GYM_TEMPERATURE")

        if api_key and hasattr(env_config, "api_key"):
            env_config.api_key = api_key
        if base_url and hasattr(env_config, "base_url"):
            env_config.base_url = base_url
        if timeout and hasattr(env_config, "timeout"):
            env_config.timeout = float(timeout)
        if max_tokens and hasattr(env_config, "max_tokens"):
            env_config.max_tokens = int(max_tokens)
        if temperature and hasattr(env_config, "temperature"):
            env_config.temperature = float(temperature)

    def create_environment(self, request_id: str, env_name: str, **kwargs) -> str:
        """Create and store environment for a conversation.

        Args:
            request_id: Unique identifier for the conversation
            env_name: Type of environment (TurtleGym, TelepathyGym, etc.)
            **kwargs: Environment-specific configuration

        Returns:
            request_id for the created environment
        """
        if request_id in self._environments:
            logger.warning(f"Environment for request_id {request_id} already exists")
            return request_id

        env = None

        if env_name == "TurtleGym":
            env = self._create_turtlegym_environment(**kwargs)
        elif env_name == "TelepathyGym":
            env = self._create_telepathygym_environment(**kwargs)
        elif env_name == "PersuadeGym":
            env = self._create_persuadegym_environment(**kwargs)
        elif env_name == "IntentionGym":
            env = self._create_intentiongym_environment(**kwargs)
        elif env_name == "TravelGym":
            env = self._create_travelgym_environment(**kwargs)
        elif env_name == "SearchGym":
            env = self._create_searchgym_environment(**kwargs)
        elif env_name == "TauGym":
            env = self._create_taugym_environment(**kwargs)
        elif env_name == "FunctionGym":
            env = self._create_functiongym_environment(**kwargs)
        elif env_name == "Tau2Gym":
            env, initial_observation = self._create_tau2gym_environment(**kwargs)
            self._environments[request_id] = env
            self._env_configs[request_id] = {
                "env_name": env_name,
                "kwargs": kwargs
            }
            logger.info(f"Created {env_name} environment for request {request_id}")
            return initial_observation  # Return initial observation for Tau2Gym
        elif env_name == "ColBenchCodeEnv":
            env = self._create_colbenchcodeenv_environment(**kwargs)
        else:
            raise ValueError(f"Unknown environment type: {env_name}")

        if os.getenv("USERRL_TEMPLATE_USER_SIM", "0") == "1":
            env = TemplateUserSimWrapper(env, env_name)
        # Outermost so it sees whatever reply actually reached the policy,
        # including the templated ones. No-op while repetition.penalty is 0.0.
        env = RepetitionPricingWrapper(env, env_name)

        self._environments[request_id] = env
        self._env_configs[request_id] = {
            "env_name": env_name,
            "kwargs": kwargs
        }

        logger.info(f"Created {env_name} environment for request {request_id}")
        return request_id

    def get_environment(self, request_id: str) -> Optional[Any]:
        """Get environment for a conversation."""
        return self._environments.get(request_id)

    def release_environment(self, request_id: str) -> None:
        """Clean up environment for a conversation."""
        if request_id in self._environments:
            # Cleanup environment if it has cleanup methods
            env = self._environments[request_id]
            if hasattr(env, 'close'):
                env.close()

            del self._environments[request_id]
            del self._env_configs[request_id]
            logger.info(f"Released environment for request {request_id}")

    def _create_turtlegym_environment(self, **kwargs):
        """Create TurtleGym environment."""
        import turtlegym

        env_config = turtlegym.get_default_config()

        # Configure from kwargs
        title = kwargs.get("title")
        max_turns = kwargs.get("max_turns", 15)
        model_name = kwargs.get("model_name", "gpt-4o-mini")

        env_config.max_steps = max_turns
        env_config.success_threshold = 1.0
        env_config.data_mode = "single"
        env_config.data_source = title
        env_config.model_name = model_name
        self._apply_llm_env_overrides(env_config, **kwargs)
        self._apply_difficulty_overrides(env_config, "TurtleGym")

        env = turtlegym.StoryEnv(config=env_config)
        env.reset()

        return env

    def _create_telepathygym_environment(self, **kwargs):
        """Create TelepathyGym environment."""
        import telepathygym

        env_config = telepathygym.get_default_config()

        # Configure from kwargs
        title = kwargs.get("title")
        max_turns = kwargs.get("max_turns", 15)
        model_name = kwargs.get("model_name", "gpt-4o-mini")

        env_config.max_steps = max_turns
        env_config.data_mode = "single"
        env_config.data_source = title
        env_config.model_name = model_name
        self._apply_llm_env_overrides(env_config, **kwargs)
        self._apply_difficulty_overrides(env_config, "TelepathyGym")

        env = telepathygym.TelepathyEnv(config=env_config)
        env.reset()

        return env

    def _create_persuadegym_environment(self, **kwargs):
        """Create PersuadeGym environment."""
        import persuadegym

        env_config = persuadegym.get_default_config()

        # Configure from kwargs
        max_turns = kwargs.get("max_turns", 15)
        model_name = kwargs.get("model_name", "gpt-4o-mini")
        id = kwargs.get("id")

        env_config.max_steps = max_turns
        env_config.data_mode = "single"
        env_config.data_source = id
        env_config.model_name = model_name
        self._apply_llm_env_overrides(env_config, **kwargs)
        self._apply_difficulty_overrides(env_config, "PersuadeGym")

        env = persuadegym.PersuadeEnv(config=env_config)
        env.reset()

        return env

    def _create_intentiongym_environment(self, **kwargs):
        """Create IntentionGym environment."""
        import intentiongym

        env_config = intentiongym.get_default_config()

        # Configure from kwargs
        max_turns = kwargs.get("max_turns", 15)
        model_name = kwargs.get("model_name", "gpt-4o-mini")
        id = kwargs.get("id")

        env_config.max_steps = max_turns
        env_config.data_mode = "single"
        env_config.data_source = id
        env_config.model_name = model_name
        self._apply_llm_env_overrides(env_config, **kwargs)
        self._apply_difficulty_overrides(env_config, "IntentionGym")

        env = intentiongym.IntentionEnv(config=env_config)
        env.reset()

        return env

    def _create_travelgym_environment(self, **kwargs):
        """Create TravelGym environment."""
        import travelgym

        env_config = travelgym.get_default_config()

        # Configure from kwargs
        max_turns = kwargs.get("max_turns", 20)
        model_name = kwargs.get("model_name", "gpt-4o")
        id = kwargs.get("id")

        env_config.max_steps = max_turns
        env_config.data_mode = "single"
        env_config.data_source = id
        env_config.model_name = model_name
        env_config.one_choice_per_aspect = True
        self._apply_llm_env_overrides(env_config, **kwargs)
        self._apply_difficulty_overrides(env_config, "TravelGym")

        env_config.search_correct_reward = float(os.environ.get("TRAVELGYM_SEARCH_CORRECT_REWARD", "0.0"))
        env_config.preference_correct_reward = 0.8

        # Allow per-episode override of the simulated search failure interval
        # (used by epoch-aware curriculum schedules injected by the rollout).
        if "search_failure_interval" in kwargs:
            env_config.search_failure_interval = int(kwargs["search_failure_interval"])

        env = travelgym.TravelEnv(config=env_config)
        env.reset()

        return env

    def _create_searchgym_environment(self, **kwargs):
        """Create SearchGym environment."""
        import searchgym

        env_config = searchgym.get_default_config()
        model_name = kwargs.get("model_name", "gpt-4o")

        # Configure from kwargs
        max_turns = kwargs.get("max_turns", 20)
        id = kwargs.get("id")

        env_config.max_steps = max_turns
        env_config.data_mode = "single"
        env_config.data_source = id
        env_config.eval_method = "llm"
        env_config.model_name = model_name
        self._apply_llm_env_overrides(env_config, **kwargs)
        self._apply_difficulty_overrides(env_config, "SearchGym")

        env = searchgym.SearchEnv(config=env_config)
        env.reset()

        return env

    def _create_taugym_environment(self, **kwargs):
        """Create TauGym environment."""
        import taugym

        env_config = taugym.get_default_config()

        # Configure from kwargs
        max_turns = kwargs.get("max_turns", 20)
        model_name = kwargs.get("model_name", "gpt-4o")
        id = kwargs.get("id")

        env_config.max_steps = max_turns
        env_config.data_mode = "single"
        env_config.data_source = id
        env_config.user_model = model_name
        # Optional flow-based user sim (symbolic flow tracking + LLM rendering).
        env_config.user_strategy = os.environ.get("TAU_USER_STRATEGY", env_config.user_strategy)
        self._apply_llm_env_overrides(env_config, **kwargs)
        self._apply_difficulty_overrides(env_config, "TauGym")

        if "retail" in id:
            env_config.task_category = "retail"
        else:
            env_config.task_category = "airline"
        if "train" in id:
            env_config.task_split = "train"
        else:
            env_config.task_split = "test"

        last_error = None
        for attempt in range(2):
            env = taugym.TauEnv(config=env_config)
            try:
                env.reset()
                return env
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "TauGym reset failed for id=%s category=%s split=%s on attempt %s/2: %s",
                    id,
                    env_config.task_category,
                    env_config.task_split,
                    attempt + 1,
                    exc,
                )

        raise RuntimeError(f"TauGym reset failed after retries for id={id}: {last_error}")

    def _create_functiongym_environment(self, **kwargs):
        """Create FunctionGym environment."""
        import functiongym

        env_config = functiongym.get_default_config()

        # Configure from kwargs
        max_turns = kwargs.get("max_turns", 20)
        id = kwargs.get("id")

        env_config.max_steps = max_turns
        env_config.data_mode = "single"
        env_config.data_source = id

        # FunctionGym was the only gym missing this call, while
        # symbolic_layer/difficulty.py's KNOWN_FIELDS whitelists
        # function.max_repeated_probes (default 3) and function.max_steps
        # (default 20). So the gate would ACCEPT a write to difficulty.function
        # and it would silently do nothing -- the exact failure the gate exists
        # to prevent, in the direction that is hardest to notice, because a
        # write that is accepted and has no effect looks like "the knob does
        # not work" rather than "the knob is not connected".
        self._apply_difficulty_overrides(env_config, "FunctionGym")

        env = functiongym.FunctionEnv(config=env_config)
        env.reset()

        return env

    def _create_tau2gym_environment(self, **kwargs):
        """Create the bundled native τ³ training environment.

        The adapter validates the official training split and receives user
        simulator settings explicitly. Hidden reference actions are available
        only to the benchmark's evaluator, never to a scripted simulator.
        """
        import tau2gym
        return tau2gym.create_environment(**kwargs)

    def _create_colbenchcodeenv_environment(self, **kwargs):
        """Create ColBenchCodeEnv environment."""
        import os
        import colbenchgym

        env_config = colbenchgym.get_code_config()

        # Configure from kwargs or environment variables
        max_turns = kwargs.get("max_turns", 10)

        # Model name: prefer MULTITURN_MODEL_NAME (unified), then COLBENCH_ENV_MODEL_NAME (backward compat)
        model_name = (kwargs.get("model_name") or
                     os.getenv("MULTITURN_MODEL_NAME") or
                     os.getenv("COLBENCH_ENV_MODEL_NAME", ""))

        # Base URL: prefer OPENAI_BASE_URL (unified), then OPENAI_API_BASE (alias), then COLBENCH_ENV_BASE_URL (backward compat)
        base_url = (kwargs.get("base_url") or
                   os.getenv("OPENAI_BASE_URL") or
                   os.getenv("OPENAI_API_BASE") or  # Support both OPENAI_BASE_URL and OPENAI_API_BASE
                   os.getenv("COLBENCH_ENV_BASE_URL", ""))

        if base_url:
            # Use provided base_url directly
            env_config.env_base_url = base_url
            # Extract hostname and port for backward compatibility (optional)
            if "://" in base_url:
                # Parse URL to extract hostname and port
                from urllib.parse import urlparse
                parsed = urlparse(base_url)
                env_config.env_hostname = parsed.hostname or "localhost"
                env_config.env_port = parsed.port or 8000
        else:
            # Fallback to hostname:port format
            hostname = kwargs.get("hostname") or os.getenv("COLBENCH_ENV_HOSTNAME", "localhost")
            port = kwargs.get("port") or int(os.getenv("COLBENCH_ENV_PORT", "8000"))
            env_config.env_hostname = hostname
            env_config.env_port = port

        # API key: prefer OPENAI_API_KEY (unified), then COLBENCH_ENV_API_KEY (backward compat)
        api_key = (kwargs.get("api_key") or
                  os.getenv("OPENAI_API_KEY") or
                  os.getenv("COLBENCH_ENV_API_KEY", ""))
        env_config.env_api_key = api_key

        # Get task information
        # Task can be passed directly as a dict, or as individual fields
        task = kwargs.get("task", {})
        if isinstance(task, dict):
            problem_description = task.get("problem_description", kwargs.get("problem_description", ""))
            ground_truth = task.get("ground_truth", kwargs.get("ground_truth", ""))
            test_cases = task.get("test_cases", kwargs.get("test_cases", {}))
        else:
            problem_description = kwargs.get("problem_description", "")
            ground_truth = kwargs.get("ground_truth", "")
            test_cases = kwargs.get("test_cases", {})

        task_id = kwargs.get("id")
        category = kwargs.get("category")

        # Configure environment
        env_config.max_steps = max_turns
        env_config.env_model_name = model_name
        env_config.verbose = kwargs.get("verbose", False)

        # ColBench was the one branch left off the hot difficulty layer, so the
        # meta-agent had no environment-side lever on this run at all.
        self._apply_difficulty_overrides(env_config, "ColBenchCodeEnv")

        # Create task dict for environment (include test_cases if available)
        task = {
            "problem_description": problem_description,
            "ground_truth": ground_truth,
            "test_cases": test_cases
        }

        # Create environment
        env = colbenchgym.ColBenchCodeEnv(
            config=env_config,
            task=task,
            category=category,
            id=task_id
        )

        # Reset environment with task
        env.reset(options={"task": task})

        return env


# Global environment manager instance
_env_manager = EnvironmentManager()


def get_environment_manager() -> EnvironmentManager:
    """Get the global environment manager instance."""
    return _env_manager
