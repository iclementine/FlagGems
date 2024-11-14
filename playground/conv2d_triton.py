# To write a convolution in triton
# strided layout - copy it as c-contiguous or nhwc-contiguous
# stride
# dilation
# padding

import torch
import triton
from triton import language as tl


def empty_keep_stride_order(
    shape: torch.Size, x: torch.Tensor, dtype=None, layout=None, device=None
):
    """Create an empyu tensor with specified shape and keep the stride order of another tensor"""
    shape = torch.Size(shape)
    ndim = x.ndim
    if len(shape) != ndim:
        raise ValueError("Rank mismatch!")

    if torch.Size(shape) == x.shape:
        return torch.empty_like(x)

    strides = x.stride()
    stride_order = sorted(list(range(ndim)), key=lambda i: strides[i], reverse=True)
    out = torch.empty_permuted(
        shape,
        stride_order,
        dtype=dtype or x.dtype,
        device=device or x.device,
        layout=layout or x.layout,
    )
    return out


def conv2d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    dilation: int,
    padding: int,
) -> int:
    """
    Determines the output size of a 2D convolution operation.

    Args:
        in_size: Input size.
        kernel_size: Kernel size.
        stride: Stride.
        padding: Padding.

    Returns:
        Output size of 2D convolution.
    """
    receptive_field = 1 + (kernel_size - 1) * dilation
    return (in_size + 2 * padding - receptive_field) // stride + 1


@triton.jit
def conv2d_forward_kernel(
    # data pointers
    in_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    # sizes
    batch_size,
    c_in,
    h_in,
    w_in,
    c_out,
    h_out,
    w_out,
    kh,
    kw,
    # strides
    input_n_stride,
    input_c_stride,
    input_h_stride,
    input_w_stride,
    weight_c_out_stride,
    weight_c_in_stride,
    weight_h_stride,
    weight_w_stride,
    bias_c_out_stride,
    output_n_stride,
    output_c_stride,
    output_h_stride,
    output_w_stride,
    # conv parameters
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    group,
    # tile size
    TILE_H: tl.constexpr,
    TILE_W: tl.constexpr,
    TILE_C_IN: tl.constexpr,
    TILE_C_OUT: tl.constexpr,
    TILE_KH: tl.constexpr,
    TILE_KW: tl.constexpr,
):
    # implement direct conv
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_bg = tl.program_id(2)

    pid_g = pid_bg % group
    pid_b = pid_bg // group

    in_channels_per_group = c_in // group
    out_channels_per_group = c_out // group

    # ------------------ offstes & mask for output tensor ------------------
    out_index_h = pid_h * TILE_H + tl.arange(0, TILE_H)
    out_index_w = pid_w * TILE_W + tl.arange(0, TILE_W)
    out_index_c = pid_g * out_channels_per_group + tl.arange(0, TILE_C_OUT)
    out_index_b = pid_b

    mask_out_h = out_index_h < h_out
    mask_out_w = out_index_w < w_out
    mask_out_c = out_index_c < ((pid_g + 1) * out_channels_per_group)

    out_ptrs = (
        out_ptr
        + out_index_b * output_n_stride
        + out_index_h[:, None, None] * output_h_stride
        + out_index_w[None, :, None] * output_w_stride
        + out_index_c[None, None, :] * output_c_stride
    )
    out_ptrs = tl.reshape(out_ptrs, [TILE_H * TILE_W, TILE_C_OUT])

    out_mask = (
        mask_out_h[:, None, None]
        & mask_out_w[None, :, None]
        & mask_out_c[None, None, :]
    )
    out_mask = tl.reshape(out_mask, [TILE_H * TILE_W, TILE_C_OUT])

    # ------------------ offstes & mask for input tensor ------------------
    in_start_h = out_index_h * stride_h - padding_h
    in_start_w = out_index_w * stride_w - padding_w

    in_offsets_h = tl.arange(0, TILE_KH) * dilation_h
    in_offsets_w = tl.arange(0, TILE_KW) * dilation_w
    in_index_c = pid_g * in_channels_per_group + tl.arange(0, TILE_C_IN)
    in_index_b = pid_b
    in_index_h = in_start_h[:, None] + in_offsets_h[None, :]
    in_index_w = in_start_w[:, None] + in_offsets_w[None, :]

    mask_in_h = (0 <= in_index_h) & (in_index_h < h_in)  # [TILE_H, TILE_KH]
    mask_in_w = (0 <= in_index_w) & (in_index_w < w_in)  # [TILE_W, TILE_KW]
    mask_in_c = in_index_c < ((pid_g + 1) * in_channels_per_group)

    in_ptrs = (
        in_ptr
        + in_index_b * input_n_stride
        + in_index_c[None, None, :, None, None] * input_c_stride
        + in_index_h[:, None, None, :, None] * input_h_stride
        + in_index_w[None, :, None, None, :] * input_w_stride
    )
    in_ptrs = tl.reshape(in_ptrs, [TILE_H * TILE_W, TILE_C_IN * TILE_KH * TILE_KW])

    in_mask = (
        mask_in_c[None, None, :, None, None]
        & mask_in_h[:, None, None, :, None]
        & mask_in_w[None, :, None, None, :]
    )  # [TILE_H, TILE_W, TILE_C_IN, TILE_KH, TILE_KW]
    in_mask = tl.reshape(in_mask, [TILE_H * TILE_W, TILE_C_IN * TILE_KH * TILE_KW])

    # ------------------ offstes & mask for weight tensor ------------------
    w_index_c_out = pid_g * out_channels_per_group + tl.arange(0, TILE_C_OUT)
    w_index_c_in = tl.arange(0, TILE_C_IN)
    w_index_kh = tl.arange(0, TILE_KH)
    w_index_kw = tl.arange(0, TILE_KW)
    w_ptrs = (
        w_ptr
        + w_index_c_out[:, None, None, None] * weight_c_out_stride
        + w_index_c_in[None, :, None, None] * weight_c_in_stride
        + w_index_kh[None, None, :, None] * weight_h_stride
        + w_index_kw[None, None, None, :] * weight_w_stride
    )
    w_ptrs = tl.reshape(w_ptrs, [TILE_C_OUT, TILE_C_IN * TILE_KH * TILE_KW])

    w_mask_c_out = w_index_c_out < ((pid_g + 1) * out_channels_per_group)
    w_mask_c_in = w_index_c_in < in_channels_per_group
    w_mask_kh = w_index_kh < kh
    w_mask_kw = w_index_kw < kw
    w_mask = (
        w_mask_c_out[:, None, None, None]
        & w_mask_c_in[None, :, None, None]
        & w_mask_kh[None, None, :, None]
        & w_mask_kw[None, None, None, :]
    )
    w_mask = tl.reshape(w_mask, [TILE_C_OUT, TILE_C_IN * TILE_KH * TILE_KW])

    # ------------------ offstes & mask for bias tensor ------------------
    bias_index_c_out = pid_g * out_channels_per_group + tl.arange(0, TILE_C_OUT)
    b_mask = bias_index_c_out < ((pid_g + 1) * out_channels_per_group)
    b_ptrs = b_ptr + bias_index_c_out * bias_c_out_stride

    # ------------------ compute ------------------
    inputs = tl.load(in_ptrs, mask=in_mask)
    weights = tl.load(w_ptrs, mask=w_mask)
    output = tl.load(b_ptrs, mask=b_mask)[None, :]
    output += tl.dot(inputs, tl.trans(weights), allow_tf32=False)

    tl.store(out_ptrs, output, mask=out_mask)


class Conv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups):
        assert weight.ndim == 4, "Weights must be 4D, received shape {weight.shape}"
        assert (
            bias is None or bias.ndim == 1
        ), "Bias must be 1D, received shape {bias.shape}"

        assert (
            input.shape[1] == groups * weight.shape[1]
        ), "Incompatible input ({input.shape}) and weights ({weight.shape}) shape with {groups} groups"
        assert (
            bias is None or weight.shape[0] == bias.shape[0]
        ), "Incompatible weights ({weight.shape}) and bias ({bias.shape}) shape"

        if isinstance(stride, (list, tuple)):
            stride_height, stride_width = stride
        else:
            stride_height = stride_width = stride

        if isinstance(dilation, (list, tuple)):
            dilation_height, dilation_width = dilation
        else:
            dilation_height = dilation_width = dilation

        if isinstance(padding, (list, tuple)):
            padding_height, padding_width = padding
        else:
            padding_height = padding_width = padding

        batch_size, c_in, in_height, in_width = input.shape
        c_out, _, kernel_height, kernel_width = weight.shape
        in_channels_per_group = c_in // groups
        out_channels_per_group = c_out // groups
        out_height = conv2d_output_size(
            in_height, kernel_height, stride_height, dilation_height, padding_height
        )
        out_width = conv2d_output_size(
            in_width, kernel_width, stride_width, dilation_width, padding_width
        )

        # stride order, we should copy stride order from the input to ensure that
        # nhwc, nchw stride order propagates
        output_dtype = input.dtype
        output_shape = (batch_size, c_out, out_height, out_width)
        output = empty_keep_stride_order(output_shape, input, dtype=output_dtype)

        # BLOCK_NI_HO_WO along the in_n, out_height, and out_width dimensions,
        # BLOCK_CO along the out_c,
        # one group per cat
        TILE_H = 1
        TILE_W = 16
        TILE_KH = triton.next_power_of_2(kernel_height)
        TILE_KW = triton.next_power_of_2(kernel_width)
        TILE_C_IN = triton.next_power_of_2(in_channels_per_group)
        TILE_C_OUT = max(16, triton.next_power_of_2(out_channels_per_group))

        grid = (
            triton.cdiv(out_width, TILE_W),
            triton.cdiv(out_height, TILE_H),
            groups * batch_size,
        )
        conv2d_forward_kernel[grid](
            # data pointers
            input,
            weight,
            bias,
            output,
            # sizes
            batch_size,
            c_in,
            in_height,
            in_width,
            c_out,
            out_height,
            out_width,
            kernel_height,
            kernel_width,
            # strides
            *input.stride(),
            *weight.stride(),
            *bias.stride(),
            *output.stride(),
            # conv parameters
            stride_height,
            stride_width,
            padding_height,
            padding_width,
            dilation_height,
            dilation_width,
            groups,
            # tile size
            TILE_H,
            TILE_W,
            TILE_C_IN,
            TILE_C_OUT,
            TILE_KH,
            TILE_KW,
        )

        requires_grad = (
            input.requires_grad
            or weight.requires_grad
            or (bias is not None and bias.requires_grad)
        )

        ctx.stride = (stride_height, stride_width)
        ctx.padding = (padding_height, padding_width)
        ctx.groups = groups
        ctx.bias_requires_grad = False if bias is None else bias.requires_grad
        ctx.output_dtype = output_dtype
        if requires_grad:
            ctx.save_for_backward(input, weight)

        return output


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return Conv2d.apply(input, weight, bias, stride, padding, dilation, groups)


def test():
    N = 10
    C_IN = 4
    C_OUT = 4
    GROUPS = 4
    C_IN_PER_GROUP = C_IN // GROUPS
    # C_OUT_PER_GROUP = C_OUT // GROUPS
    KH = 3
    KW = 3
    H = 1000
    W = 2000
    STRIDE = 1
    PADDING = 1
    DILATION = 1
    x = torch.randn((N, H, W, C_IN), dtype=torch.float32, device="cuda").permute(
        0, 3, 1, 2
    )
    w = torch.randn((C_OUT, C_IN_PER_GROUP, KH, KW), dtype=torch.float32, device="cuda")
    b = torch.randn((C_OUT,), dtype=torch.float32, device="cuda")
    out = conv2d(
        x, w, b, stride=STRIDE, padding=PADDING, dilation=DILATION, groups=GROUPS
    )
    out_ref = torch.conv2d(
        x, w, b, stride=STRIDE, padding=PADDING, dilation=DILATION, groups=GROUPS
    )
    print(out.shape, out.stride())
    print(out_ref.shape, out_ref.stride())

    print(out[-1, -1, :10, :6])
    print(out_ref[-1, -1, :10, :6])
    torch.testing.assert_close(out, out_ref)
    # print((out - out_ref)[-1, -1, :8, :6])


def benchmark():
    N = 10
    C_IN = 16
    C_OUT = 16
    GROUPS = 1
    C_IN_PER_GROUP = C_IN // GROUPS
    KH = 3
    KW = 3
    H = 1000
    W = 1000
    STRIDE = 1
    PADDING = 0
    DILATION = 1
    x = torch.randn((N, H, W, C_IN), dtype=torch.float16, device="cuda").permute(
        0, 3, 1, 2
    )
    w = torch.randn((C_OUT, C_IN_PER_GROUP, KH, KW), dtype=torch.float16, device="cuda")
    b = torch.randn((C_OUT,), dtype=torch.float16, device="cuda")

    def f1():
        return conv2d(
            x, w, b, stride=STRIDE, padding=PADDING, dilation=DILATION, groups=GROUPS
        )

    def f2():
        return torch.conv2d(
            x, w, b, stride=STRIDE, padding=PADDING, dilation=DILATION, groups=GROUPS
        )

    def _f3(x, w, b):
        return torch.conv2d(
            x, w, b, stride=STRIDE, padding=PADDING, dilation=DILATION, groups=GROUPS
        )

    __f3 = torch.compile(_f3)

    def f3():
        return __f3(x, w, b)

    t1 = triton.testing.do_bench(f1, return_mode="median")
    t2 = triton.testing.do_bench(f2, return_mode="median")
    t3 = triton.testing.do_bench(f3, return_mode="median")
    print(f"triton: {t1} ms")
    print(f"aten: {t2} ms")
    print(f"compile: {t3} ms")


if __name__ == "__main__":
    # test()
    benchmark()
