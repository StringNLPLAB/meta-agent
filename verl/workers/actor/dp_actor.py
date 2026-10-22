# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    _GRAD_NAN_DIAG_BUDGET = int(os.environ.get("USERRL_GRAD_NAN_DIAG_BUDGET", "40"))

    _NAN_PROBE = os.environ.get("USERRL_NAN_PROBE", "1") != "0"
    _NAN_PROBE_REPORTS = int(os.environ.get("USERRL_NAN_PROBE_REPORTS", "20"))
    # Survey cadence: print the policy/env/pad breakdown every Nth mini-batch
    # even when nothing went wrong. 0 disables.
    _NAN_PROBE_SURVEY_EVERY = int(os.environ.get("USERRL_NAN_PROBE_SURVEY_EVERY", "4"))
    _NAN_PROBE_SURVEY = _NAN_PROBE_SURVEY_EVERY

    def _nan_probe_param(self):
        """Smallest parameter that carries a gradient -- cheapest finiteness probe.

        Cached after the first call. Returns None if nothing has a grad yet.
        """
        prm = getattr(self, "_nan_probe_param_cache", None)
        if prm is not None:
            return prm
        best = None
        for _, q in self.actor_module.named_parameters():
            if q.grad is None:
                continue
            if best is None or q.grad.numel() < best.grad.numel():
                best = q
        if best is not None:
            self._nan_probe_param_cache = best
        return best

    def _diagnose_nonfinite_grads(self, max_report: int = 4):
        """Name the parameters whose gradients are non-finite.

        Runs ONLY on the failure path, so it costs nothing in normal operation.

        FSDP1 here is built with ``use_orig_params=False``, so ``named_parameters``
        yields one flat parameter per wrapped unit (one transformer layer). The
        module path already gives the layer; ``_param_infos``/``_numels`` are then
        used, best-effort, to map the offending offsets back to the original
        parameter names. Those two attributes are private to torch, so every use
        of them is wrapped -- a diagnostic must never be able to kill the run it
        is diagnosing.
        """
        hits = []
        try:
            for name, prm in self.actor_module.named_parameters():
                g = prm.grad
                if g is None:
                    continue
                bad = ~torch.isfinite(g)
                n_bad = int(bad.sum().item())
                if n_bad == 0:
                    continue
                entry = {
                    "flat_param": name,
                    "numel": int(g.numel()),
                    "n_nonfinite": n_bad,
                    "n_nan": int(torch.isnan(g).sum().item()),
                }
                try:
                    # flatten FIRST: nonzero() on a >1-D tensor returns index
                    # *tuples*, and flattening those yields coordinates, not
                    # offsets -- silently, and in the wrong order.
                    idx = bad.reshape(-1).nonzero(as_tuple=False).reshape(-1)
                    lo, hi = int(idx[0].item()), int(idx[-1].item())
                    entry["bad_offset_range"] = (lo, hi)
                    infos = getattr(prm, "_param_infos", None)
                    numels = getattr(prm, "_numels", None)
                    if infos is not None and numels is not None:
                        off, orig = 0, []
                        for info, n in zip(infos, numels):
                            n = int(n)
                            if off <= hi and lo < off + n:
                                orig.append(f"{getattr(info, 'module_name', '?')}.{getattr(info, 'param_name', '?')}")
                            off += n
                        entry["orig_params"] = orig[:6]
                except Exception as e:  # private torch internals moved
                    entry["map_error"] = repr(e)
                hits.append(entry)
                if len(hits) >= max_report:
                    break
        except Exception as e:
            return [{"diag_error": repr(e)}]
        return hits

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            self._optim_skipped = 1.0
            rank = torch.distributed.get_rank()
            # Gradients are sharded, so each rank sees a different slice: let
            # every rank report. The FSDP all-reduce of the total norm means
            # the *norm* is nan everywhere, but the nan *entries* need not be.
            if DataParallelPPOActor._GRAD_NAN_DIAG_BUDGET > 0:
                DataParallelPPOActor._GRAD_NAN_DIAG_BUDGET -= 1
                for h in self._diagnose_nonfinite_grads():
                    print(f"WARN: rank {rank} grad_nan_detail {h}")
            print(f"WARN: rank {rank} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self._optim_skipped = 0.0
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)

        else:
            micro_batches = batch.split(micro_batch_size)
            # logger.warning(f"DEBUG: batch: {batch}")
            # logger.warning(f"DEBUG: micro_batches: {micro_batches}")
            valid_tokens_per_sample = batch["attention_mask"].sum(dim=1)
            # logger.warning(f"DEBUG: valid_tokens_per_sample: {valid_tokens_per_sample}")
            # logger.warning(f"DEBUG: total valid tokens: {valid_tokens_per_sample.sum().item()}")
        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():

                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if calculate_entropy:
                entropys = entropys[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)
        # Optional per-step scheduled clip_ratio_high injected by the trainer
        # (clip-high scheduler). Falls back to the static config when absent.
        scheduled_clip_high = data.meta_info.get("clip_ratio_high", None)
        # Same deal for the hot-reloadable entropy bonus. Must be read HERE:
        # inside the micro-batch loop below `data` is rebound to a TensorDict,
        # which has no meta_info.
        scheduled_entropy_coeff = data.meta_info.get("entropy_coeff", None)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                # One 0-dim GPU tensor per micro-batch; stacked and synced ONCE
                # at the end of the mini-batch, so the probe costs a single
                # device->host transfer per mini-batch rather than one per
                # micro-batch. The flag is cumulative (the grad buffer keeps
                # accumulating), so the FIRST True is the culprit.
                probe_flags, probe_rows = [], []

                for mb_idx, data in enumerate(micro_batches):
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = (
                        scheduled_clip_high
                        if scheduled_clip_high is not None
                        else (self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio)
                    )
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    clip_zero_out_of_range = self.config.get("clip_zero_out_of_range", False)
                    # Hot-reloadable entropy bonus, resolved once per step by
                    # the trainer so all workers agree. Absent -> static config,
                    # zero behavior change.
                    entropy_coeff = (
                        scheduled_entropy_coeff
                        if scheduled_entropy_coeff is not None
                        else self.config.entropy_coeff
                    )
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, offpolicy_seq_mask_frac = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        clip_zero_out_of_range=clip_zero_out_of_range,
                        loss_agg_mode=loss_agg_mode,
                        offpolicy_seq_mask_delta=self.config.get("offpolicy_seq_mask_delta", None),
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    if DataParallelPPOActor._NAN_PROBE:
                        _p = self._nan_probe_param()
                        if _p is not None and _p.grad is not None:
                            probe_flags.append((~torch.isfinite(_p.grad)).any())
                            # Cheap per-micro-batch identity, kept on device and
                            # synced with the flags. These are the quantities a
                            # nan-producing backward would plausibly hinge on.
                            # Split |log_prob| by WHICH KIND of position it sits
                            # on. The three kinds are disjoint and cover the
                            # response window:
                            #   policy : attention_mask=1, loss_mask=1  -- tokens the
                            #            policy generated; these carry the loss.
                            #   env    : attention_mask=1, loss_mask=0  -- tool output
                            #            and user-simulator text. Real tokens, attended
                            #            and scored, but NOT the policy's own writing,
                            #            so the policy may score them at ~1e-39.
                            #   pad    : attention_mask=0 -- pad_input() scatters into a
                            #            ZERO tensor, so these must read exactly 0.
                            # kl_penalty runs on all three; only `policy` reaches the
                            # loss. If the extreme values live on `env`, the loss mask
                            # is not protecting the backward -- which is the whole point.
                            _lp = log_prob.detach().abs().float()
                            _ra = attention_mask[:, -response_length:].float()
                            _pm = response_mask.float()
                            _em = _ra * (1.0 - _pm)
                            probe_rows.append(torch.stack([
                                response_mask.sum().float(),
                                _lp.max(),
                                (_lp * _pm).max(),
                                (_lp * _em).max(),
                                (_lp * (1.0 - _ra)).max(),
                                _em.sum(),
                                old_log_prob.detach().abs().max().float(),
                                advantages.detach().abs().max().float(),
                                pg_loss.detach().float(),
                            ]))

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        "actor/offpolicy_seq_mask_frac": offpolicy_seq_mask_frac.detach().item(),
                        # mean token log-likelihood under current vs rollout policy;
                        # useful to spot log-likelihood collapse (drifts toward 0)
                        "actor/log_prob_mean": verl_F.masked_mean(log_prob, response_mask).detach().item(),
                        "actor/old_log_prob_mean": verl_F.masked_mean(old_log_prob, response_mask).detach().item(),
                        "actor/clip_ratio_high": float(clip_ratio_high),
                        # The coeff actually in effect this step, hot value
                        # included -- a knob you cannot read back is a knob
                        # whose predictions you cannot verify.
                        "actor/entropy_coeff": float(entropy_coeff),
                    }
                    append_to_dict(metrics, data)

                if DataParallelPPOActor._NAN_PROBE and probe_flags:
                    try:
                        flags = torch.stack(probe_flags).cpu().tolist()
                        rank = torch.distributed.get_rank()

                        def _fmt(tag, i, r):
                            return (
                                f"WARN: rank {rank} nan_probe "
                                f"{tag}={i}/{len(flags)} n_policy_tok={r[0]:.0f} "
                                f"n_env_tok={r[5]:.0f} "
                                f"maxlp_all={r[1]:.4g} maxlp_policy={r[2]:.4g} "
                                f"maxlp_env={r[3]:.4g} maxlp_pad={r[4]:.4g} "
                                f"max_abs_old_lp={r[6]:.4g} max_abs_adv={r[7]:.4g} "
                                f"pg_loss={r[8]:.6g}"
                            )

                        if any(flags):
                            if DataParallelPPOActor._NAN_PROBE_REPORTS > 0:
                                DataParallelPPOActor._NAN_PROBE_REPORTS -= 1
                                first = flags.index(True)
                                rows = torch.stack(probe_rows).cpu().tolist()
                                print(_fmt("first_bad_micro_batch", first, rows[first]))
                                # The neighbour gives a clean/dirty contrast
                                # without dumping every row.
                                if first > 0:
                                    print(_fmt("prev_clean_micro_batch", first - 1, rows[first - 1]))
                        else:
                            # No nan this mini-batch -- still report the log_prob
                            # breakdown periodically. "Which positions carry the
                            # extreme log-probs" is a question about the DATA, not
                            # about the failure, so gating it on a nan was a design
                            # error: the B13 fix removes the nan and would then
                            # also remove the measurement.
                            DataParallelPPOActor._NAN_PROBE_SURVEY -= 1
                            if DataParallelPPOActor._NAN_PROBE_SURVEY <= 0:
                                DataParallelPPOActor._NAN_PROBE_SURVEY = (
                                    DataParallelPPOActor._NAN_PROBE_SURVEY_EVERY
                                )
                                rows = torch.stack(probe_rows).cpu().tolist()
                                # the micro-batch with the largest env-side value
                                w = max(range(len(rows)), key=lambda i: rows[i][3])
                                print(_fmt("survey_worst_env_micro_batch", w, rows[w]))
                    except Exception as e:  # noqa: BLE001
                        DataParallelPPOActor._NAN_PROBE = False
                        print(f"WARN: nan_probe disabled after {e!r}")

                grad_norm = self._optimizer_step()
                # Mean over mini-batches == fraction of optimizer steps skipped
                # this training step. 0.0 healthy, 1.0 nothing was applied.
                data = {
                    "actor/grad_norm": grad_norm.detach().item(),
                    "actor/optim_skipped": float(getattr(self, "_optim_skipped", 0.0)),
                }
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
