# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pickle
from typing import Any, List, Optional

import numpy as np
import torch
import torch.distributed as dist


def _is_picklable(obj: Any) -> bool:
    """Check if an object can be pickled."""
    try:
        pickle.dumps(obj)
        return True
    except (pickle.PicklingError, AttributeError, TypeError):
        return False


def _clean_dict_recursively(d: dict, max_depth: int = 5) -> dict:
    """
    Recursively clean a dictionary, removing unpicklable values.
    """
    if max_depth <= 0:
        return {}

    cleaned = {}
    for key, value in d.items():
        # Skip known problematic keys
        if key == "_dynamic_tool_map":
            continue

        # Check if value is picklable
        if _is_picklable(value):
            cleaned[key] = value
        elif isinstance(value, dict):
            # Try to clean nested dict
            cleaned_nested = _clean_dict_recursively(value, max_depth - 1)
            if cleaned_nested:
                cleaned[key] = cleaned_nested
        elif isinstance(value, (list, tuple)):
            # Try to clean list/tuple
            try:
                cleaned_list = []
                for item in value:
                    if _is_picklable(item):
                        cleaned_list.append(item)
                    elif isinstance(item, dict):
                        cleaned_item = _clean_dict_recursively(item, max_depth - 1)
                        if cleaned_item:
                            cleaned_list.append(cleaned_item)
                if cleaned_list:
                    cleaned[key] = tuple(cleaned_list) if isinstance(value, tuple) else cleaned_list
            except Exception:
                pass
        # Otherwise skip this unpicklable value
    return cleaned


def _clean_tools_kwargs_for_broadcast(obj: Any) -> Any:
    """
    Clean AsyncRolloutRequest objects before broadcasting.
    This removes unpicklable objects (like tool instances, environment objects with thread locks)
    that are not needed after broadcasting.

    Fields cleaned:
    - tools_kwargs: Not used after broadcast, may contain tool/environment objects with locks
    - metrics: May contain unpicklable objects, but we try to preserve serializable values
    - Any other attributes that might contain unpicklable objects

    Note: This function modifies the object in-place for efficiency, but only affects
    fields that are not used after broadcast.
    """
    # For lists, recursively clean each item
    if isinstance(obj, list):
        for item in obj:
            _clean_tools_kwargs_for_broadcast(item)
        return obj

    # Check if this is an AsyncRolloutRequest object
    if hasattr(obj, 'tools_kwargs') and hasattr(obj, 'request_id'):
        # This looks like an AsyncRolloutRequest
        # Clean tools_kwargs by removing unpicklable entries
        # tools_kwargs is not used after broadcast, so we can be aggressive
        if obj.tools_kwargs:
            obj.tools_kwargs = _clean_dict_recursively(obj.tools_kwargs)

        # Also clean metrics if it exists (may contain unpicklable objects)
        if hasattr(obj, 'metrics') and obj.metrics:
            try:
                # Try to clean metrics, but be more conservative
                cleaned_metrics = {}
                for key, value in obj.metrics.items():
                    if isinstance(value, list):
                        cleaned_list = []
                        for item in value:
                            if _is_picklable(item):
                                cleaned_list.append(item)
                        if cleaned_list:
                            cleaned_metrics[key] = cleaned_list
                    elif _is_picklable(value):
                        cleaned_metrics[key] = value
                obj.metrics = cleaned_metrics
            except Exception:
                # If cleaning fails, set to empty dict
                obj.metrics = {}

        # Check for any other attributes that might contain unpicklable objects
        # This is a safety measure to catch any unexpected unpicklable attributes
        if hasattr(obj, '__dict__'):
            for attr_name, attr_value in list(obj.__dict__.items()):
                # Skip standard fields that are known to be safe
                safe_attrs = {
                    'batch_data_id', 'rollout_offset', 'request_id', 'state',
                    'messages', 'tool_schemas', 'input_ids', 'prompt_ids', 'response_ids',
                    'attention_mask', 'prompt_attention_mask', 'response_attention_mask',
                    'position_ids', 'prompt_position_ids', 'response_position_ids',
                    'loss_mask', 'prompt_loss_mask', 'response_loss_mask',
                    'reward_scores', 'max_prompt_len', 'max_response_len', 'max_model_len',
                    'search_action_count', 'turn_boundaries', 'conversation_histories',
                    'use_inference_chat_template', 'enable_tokenization_sanity_check',
                    'generation_prompt_ids', 'base_conv_wo_gen_prompt_end_pos',
                    'base_conv_with_gen_prompt_end_pos', 'tools_kwargs', 'metrics'
                }
                if attr_name not in safe_attrs and not _is_picklable(attr_value):
                    # Remove unpicklable attributes
                    try:
                        delattr(obj, attr_name)
                    except Exception:
                        pass

    # For other types, return as-is
    return obj


def _clean_unpicklable_objects(obj: Any) -> Any:
    """
    Recursively clean objects that cannot be pickled, particularly dynamically
    created Pydantic model classes and instances.

    This function handles:
    - Dynamically created Pydantic model classes (e.g., tau2.environment.tool.parameters)
    - Pydantic model instances (converts to dict)
    - Tool objects that contain unpicklable references
    - Other unpicklable types
    """
    if obj is None:
        return None

    # Handle classes (type objects)
    if isinstance(obj, type):
        try:
            # Try to pickle the class to see if it's picklable
            pickle.dumps(obj)
            return obj
        except (pickle.PicklingError, AttributeError, TypeError):
            # If it's a dynamically created class, check if it's from tau2.environment.tool
            module_name = getattr(obj, '__module__', None)
            class_name = getattr(obj, '__name__', None)
            if module_name and 'tau2.environment.tool' in module_name:
                # Return a placeholder for dynamically created classes
                return f"<unpicklable_class:{module_name}.{class_name}>"
            # For other unpicklable classes, try to get their name
            return f"<unpicklable_class:{module_name}.{class_name}>" if module_name else f"<unpicklable_class:{class_name}>"

    # Handle Pydantic BaseModel instances
    try:
        from pydantic import BaseModel
        if isinstance(obj, BaseModel):
            # Convert to dict and recursively clean
            return _clean_unpicklable_objects(obj.model_dump() if hasattr(obj, 'model_dump') else obj.dict())
    except ImportError:
        pass

    # Handle lists
    if isinstance(obj, list):
        return [_clean_unpicklable_objects(item) for item in obj]

    # Handle dicts
    if isinstance(obj, dict):
        return {key: _clean_unpicklable_objects(value) for key, value in obj.items()}

    # Handle tuples
    if isinstance(obj, tuple):
        return tuple(_clean_unpicklable_objects(item) for item in obj)

    # Handle sets
    if isinstance(obj, set):
        return {_clean_unpicklable_objects(item) for item in obj}

    # Handle common types that are always picklable
    if isinstance(obj, (int, float, str, bool, bytes)):
        return obj

    # Handle torch tensors (they should be picklable, but check anyway)
    if isinstance(obj, torch.Tensor):
        return obj

    # For other types, try to pickle to check if picklable
    try:
        pickle.dumps(obj)
        return obj
    except (pickle.PicklingError, AttributeError, TypeError) as e:
        # If not picklable, try to convert to dict or string
        # Check if it's a tool object or has special handling
        if hasattr(obj, '__dict__'):
            try:
                # Try to convert to dict
                obj_dict = obj.__dict__.copy()
                cleaned_dict = _clean_unpicklable_objects(obj_dict)
                return cleaned_dict
            except Exception:
                pass

        # Last resort: convert to string representation
        obj_type = type(obj)
        module_name = getattr(obj_type, '__module__', None)
        class_name = getattr(obj_type, '__name__', None)
        if module_name:
            return f"<unpicklable_object:{module_name}.{class_name}>"
        else:
            return f"<unpicklable_object:{class_name}>"


def broadcast_pyobj(
    data: List[Any],
    rank: int,
    dist_group: Optional[torch.distributed.ProcessGroup] = None,
    src: int = 0,
    force_cpu_device: bool = False,
):
    """from https://github.com/sgl-project/sglang/blob/844e2f227ab0cce6ef818a719170ce37b9eb1e1b/python/sglang/srt/utils.py#L905

    Broadcast inputs from src rank to all other ranks with torch.dist backend.
    The `rank` here refer to the source rank on global process group (regardless
    of dist_group argument).
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not force_cpu_device else "cpu"
    )

    if rank == src:
        if len(data) == 0:
            tensor_size = torch.tensor([0], dtype=torch.long, device=device)
            dist.broadcast(tensor_size, src=src, group=dist_group)
        else:
            # Clean tools_kwargs before serialization to remove unpicklable objects
            # This is safe because tools_kwargs is not used after broadcast
            cleaned_data = _clean_tools_kwargs_for_broadcast(data)

            # Try to serialize with standard pickle
            try:
                serialized_data = pickle.dumps(cleaned_data)
                size = len(serialized_data)
            except (pickle.PicklingError, AttributeError, TypeError) as e:
                # If still fails, use more aggressive cleaning
                cleaned_data = _clean_unpicklable_objects(cleaned_data)
                serialized_data = pickle.dumps(cleaned_data)
                size = len(serialized_data)

            tensor_data = torch.ByteTensor(
                np.frombuffer(serialized_data, dtype=np.uint8)
            ).to(device)
            tensor_size = torch.tensor([size], dtype=torch.long, device=device)

            dist.broadcast(tensor_size, src=src, group=dist_group)
            dist.broadcast(tensor_data, src=src, group=dist_group)
        return data
    else:
        tensor_size = torch.tensor([0], dtype=torch.long, device=device)
        dist.broadcast(tensor_size, src=src, group=dist_group)
        size = tensor_size.item()

        if size == 0:
            return []

        tensor_data = torch.empty(size, dtype=torch.uint8, device=device)
        dist.broadcast(tensor_data, src=src, group=dist_group)

        serialized_data = bytes(tensor_data.cpu().numpy())
        data = pickle.loads(serialized_data)
        return data
