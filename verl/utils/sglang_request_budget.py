"""Respect the live SGLang scheduler's input cap before submitting a turn."""


def generation_budget(*, context_length, input_length, response_length, scheduler_info, turn_limit=None):
    limit = scheduler_info.get('max_req_input_len') if isinstance(scheduler_info, dict) else None
    if type(limit) is not int or limit <= 0:
        raise ValueError('SGLang did not report a positive max_req_input_len')
    if input_length > limit:
        return 0
    remaining = context_length - input_length - 1
    if remaining <= 0:
        return 0
    budget = min(response_length, remaining)
    if turn_limit is not None:
        if type(turn_limit) is not int or turn_limit <= 0:
            raise ValueError('Per-turn generation limit must be a positive integer')
        budget = min(budget, turn_limit)
    return budget
