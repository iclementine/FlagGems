import logging
from enum import IntEnum

import torch
import triton
import triton.language as tl

from ..utils import libentry
from .sum import sum, sum_dim


class Reduction(IntEnum):
    NONE = 0
    MEAN = 1
    SUM = 2


def heur_block_n(args):
    return triton.next_power_of_2(args["N"])


def heur_num_warps(args):
    if args["N"] <= 1024:
        return 4
    elif args["N"] <= 2048:
        return 8
    else:
        return 16


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1}, num_stages=4),
        triton.Config({"BLOCK_M": 1}, num_stages=5),
        triton.Config({"BLOCK_M": 2}, num_stages=4),
        triton.Config({"BLOCK_M": 2}, num_stages=5),
        triton.Config({"BLOCK_M": 4}, num_stages=4),
        triton.Config({"BLOCK_M": 4}, num_stages=5),
        triton.Config({"BLOCK_M": 8}, num_stages=4),
        triton.Config({"BLOCK_M": 8}, num_stages=5),
    ],
    key=[
        "M",
        "N",
    ],
)
@triton.heuristics(
    {
        "BLOCK_N": heur_block_n,
        "num_warps": heur_num_warps,
    }
)
@triton.jit(do_not_specialize=["mean_num"])
def log_softmax_and_mul_kernel(
    output_ptr,
    input_ptr,
    target_ptr,
    mean_num,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)
    offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
    mask = m_offset[:, None] < M and n_offset[None, :] < N
    input_ptrs = input_ptr + offset
    inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
    row_minus_max = inp - tl.max(inp, axis=1)[:, None]
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=1)[:, None]
    softmax_output = tl.log(numerator / denominator)
    target = tl.load(target_ptr + offset, mask=mask, other=0.0)
    out = softmax_output * target / (mean_num)
    output_ptrs = output_ptr + offset
    tl.store(output_ptrs, out, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1}, num_stages=4),
        triton.Config({"BLOCK_M": 1}, num_stages=5),
        triton.Config({"BLOCK_M": 2}, num_stages=4),
        triton.Config({"BLOCK_M": 2}, num_stages=5),
        triton.Config({"BLOCK_M": 4}, num_stages=4),
        triton.Config({"BLOCK_M": 4}, num_stages=5),
        triton.Config({"BLOCK_M": 8}, num_stages=4),
        triton.Config({"BLOCK_M": 8}, num_stages=5),
    ],
    key=[
        "M",
        "N",
    ],
)
@triton.heuristics(
    {
        "BLOCK_N": heur_block_n,
        "num_warps": heur_num_warps,
    }
)
@triton.jit(do_not_specialize=["mean_num"])
def softmax_and_sub_kernel(
    output_ptr,
    input_ptr,
    target_ptr,
    out_grad,
    mean_num,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)
    offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
    mask = m_offset[:, None] < M and n_offset[None, :] < N
    input_ptrs = input_ptr + offset
    inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
    row_minus_max = inp - tl.max(inp, axis=1)[:, None]
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=1)[:, None]
    # todo: reduce unnecessary calculations through mask operations to improve performance
    softmax_output = numerator / denominator
    target_ptrs = target_ptr + offset
    target = tl.load(target_ptrs, mask=mask, other=0.0)
    out_grad_ptr = out_grad + m_offset[:, None] * K + pid_k
    out_grad_value = tl.load(out_grad_ptr)
    # backward formula derivation for value of ingnore index
    target_sum = tl.sum(target, axis=1)[:, None]
    out = out_grad_value * (target_sum * softmax_output - target) / mean_num

    output_ptrs = output_ptr + offset
    tl.store(output_ptrs, out, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1}, num_stages=4),
        triton.Config({"BLOCK_M": 1}, num_stages=5),
        triton.Config({"BLOCK_M": 2}, num_stages=4),
        triton.Config({"BLOCK_M": 2}, num_stages=5),
        triton.Config({"BLOCK_M": 4}, num_stages=4),
        triton.Config({"BLOCK_M": 4}, num_stages=5),
        triton.Config({"BLOCK_M": 8}, num_stages=4),
        triton.Config({"BLOCK_M": 8}, num_stages=5),
    ],
    key=[
        "M",
        "N",
    ],
)
@triton.heuristics(
    {
        "BLOCK_N": heur_block_n,
        "num_warps": heur_num_warps,
    }
)
@triton.jit(do_not_specialize=["mean_num"])
def softmax_and_sub_reduce_kernel(
    output_ptr,
    input_ptr,
    target_ptr,
    out_grad,
    mean_num,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)
    offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
    mask = m_offset[:, None] < M and n_offset[None, :] < N
    input_ptrs = input_ptr + offset
    inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
    row_minus_max = inp - tl.max(inp, axis=1)[:, None]
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=1)[:, None]
    # todo: reduce unnecessary calculations through mask operations to improve performance
    softmax_output = numerator / denominator
    target_ptrs = target_ptr + offset
    target = tl.load(target_ptrs, mask=mask, other=0.0)
    # backward formula derivation for value of ingnore index
    target_sum = tl.sum(target, axis=1)[:, None]
    out_grad_value = tl.load(out_grad)
    out = out_grad_value * (target_sum * softmax_output - target) / mean_num

    output_ptrs = output_ptr + offset
    tl.store(output_ptrs, out, mask=mask)


class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target, weight, reduction, ignore_index, label_smoothing):
        logging.debug("GEMS CrossEntropyLoss")
        assert reduction in Reduction._value2member_map_, "Invalid reduction"
        assert isinstance(input, torch.Tensor), "input is not a tensor"

        if input.ndim >= 2:
            dim = 1
        else:
            dim = 0

        if reduction != Reduction.MEAN.value:
            mean_num = -1
        else:
            # get all ingore count of target
            ignore_count = (target == ignore_index).to(torch.int64).sum().item()

            mean_num = -target.numel() + ignore_count

        # special mean_num is 0, return 0
        if mean_num == 0:
            ctx.shape = input.shape
            ctx.mean_num = mean_num
            ctx.dim = dim
            ctx.input_dtype = input.dtype
            ctx.input_device = input.device
            if dim == 0:
                return torch.tensor(0, device=input.device, dtype=input.dtype)
            else:
                return torch.tensor(
                    float("nan"), device=input.device, dtype=input.dtype
                )

        shape = list(input.shape)
        shape[dim] = 1

        # action for target value equals ignore index, out of [0,C)
        # 1 to delete target negetive value and set 0 to make sure scatter is OK
        target_tmp = target
        target = torch.where(target == ignore_index, 0, target)
        target = torch.zeros_like(input).scatter(dim, target.view(shape), 1)

        # 2 set ignore index of target value 0
        target_tmp = target_tmp.unsqueeze(dim)
        target_tmp = target_tmp.expand(input.shape).contiguous()
        target = torch.where(target_tmp == ignore_index, 0, target)

        M = 1
        N = input.shape[dim]
        for i in range(dim):
            M *= input.shape[i]
        inp = input.contiguous()
        out = torch.empty_like(inp, dtype=inp.dtype)
        K = inp.numel() // M // N

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            K,
        )
        with torch.cuda.device(inp.device):
            log_softmax_and_mul_kernel[grid](
                out,
                inp,
                target,
                mean_num,
                M,
                N,
                K,
            )
        if reduction != Reduction.NONE.value:
            out_result = sum(out)
        else:
            out_result = sum_dim(out, dim=[dim])

        ctx.save_for_backward(input, target)
        ctx.dim = dim
        ctx.mean_num = -mean_num
        ctx.reduction = reduction
        return out_result

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS CrossEntropyLoss VJP")
        mean_num = ctx.mean_num
        dim = ctx.dim
        if mean_num == 0:
            input_shape = ctx.shape
            input_dtype = ctx.input_dtype
            input_device = ctx.input_device
            if dim == 0:
                return (
                    torch.zeros(input_shape, dtype=input_dtype, device=input_device),
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            else:
                return (
                    torch.tensor(float("nan"), dtype=input_dtype, device=input_device),
                    None,
                    None,
                    None,
                    None,
                    None,
                )
        input, target = ctx.saved_tensors
        reduction = ctx.reduction
        M = 1
        N = input.shape[dim]
        for i in range(dim):
            M *= input.shape[i]
        inp = input.contiguous()
        out = torch.empty_like(inp, dtype=input.dtype)
        K = inp.numel() // M // N

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            K,
        )
        with torch.cuda.device(inp.device):
            if reduction != Reduction.NONE.value:
                softmax_and_sub_reduce_kernel[grid](
                    out,
                    inp,
                    target,
                    out_grad,
                    mean_num,
                    M,
                    N,
                    K,
                )
            else:
                softmax_and_sub_kernel[grid](
                    out,
                    inp,
                    target,
                    out_grad,
                    mean_num,
                    M,
                    N,
                    K,
                )
        return out, None, None, None, None, None


def cross_entropy_loss(
    input, target, weight=None, reduction=1, ignore_index=-100, label_smoothing=0.0
):
    return CrossEntropyLoss.apply(
        input, target, weight, reduction, ignore_index, label_smoothing
    )
