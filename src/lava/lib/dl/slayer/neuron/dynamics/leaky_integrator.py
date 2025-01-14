# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Leaky Integrator dynamics."""

import os
import torch
from torch.utils.cpp_extension import load

from ...utils.int_utils import right_shift_to_zero, Q2Zero
from ...utils.utils import staticproperty
from ... import jitconfig

import pdb


class Accelerated:
    """ """
    # This provides a static handle to jit cuda accelerated module
    # They are callable as
    # Accelerated.leaky_integrator.dynamics(...) -> handles autograd
    # Accelerated.leaky_integrator.fwd(...)
    # Accelerated.leaky_integrator.bwd(...)
    module = None

    @staticproperty
    def leaky_integrator():
        if Accelerated.module is None:
            if jitconfig.VERBOSE is True:
                print(
                    'Leaky Integrator accelerated module does not exist. '
                    'Initializing with JIT compilation.'
                )
            if not torch.cuda.is_available():
                raise Exception(
                    'CUDA acceleration of Leaky Integrator failed. '
                    'CUDA is not available in the system.'
                )
            if jitconfig.TORCH_CUDA_ARCH_LIST is not None:
                os.environ['TORCH_CUDA_ARCH_LIST'] = \
                    jitconfig.TORCH_CUDA_ARCH_LIST
            Accelerated.module = load(
                name='dynamics',
                sources=[
                    os.path.dirname(os.path.abspath(__file__))
                    + '/leaky_integrator.cu'
                ],
            )
        return Accelerated.module


def dynamics(input, decay, state, w_scale, threshold=None, debug=False, quantize=False):
    """Leaky integrator dynamics. It automatically switches
    between CUDA and CPU implementation depending on available hardware.

    .. math::
        y[t] &= (1 - \\alpha)\\,y[t-1] + x[t] \\

        s[t] &= y[t] \\geq \\vartheta \\

        y[t] &= y[t]\\,(1-s[t])

    Parameters
    ----------
    input : torch tensor
        input tensor.
    decay : torch tensor
        decay tensor. Note: it is unscaled integer value here.
    state : torch tensor
        dynamics state.
    w_scale : int
        parameter scaling for integer calculations.
    threshold : float or None, optional
        threshold for spiking dynamics. Default is None.
    debug : bool, optional
        enable/disable debug mode. Default is False.

    Returns
    -------
    torch tensor
        leaky integrator output.

    Note
    ----
    When threshold is not supplied, no spike is generated.
    """
    
    debug=True
    
    if threshold is None:
        threshold = -1  # -1 means no reset mechanism
    _LIDynamics.DEBUG = debug

    if torch.numel(state) == 1:
        state = state * torch.ones(input.shape[:-1]).to(input.device)

    if input.is_cuda is False or debug is True:
        output = _LIDynamics.apply(input, decay, state, threshold, w_scale, quantize)
    else:
        output = Accelerated.leaky_integrator.dynamics(
            input.contiguous(), decay.contiguous(), state.contiguous(),
            threshold, w_scale
        )

    return output


def persistent_state(state, spike):
    """Handles persistent state changes due to spike in the last time bin.

    Parameters
    ----------
    state : torch tensor
        state in last time bin.
    spike : torch tensor
        spike state in last time bin.

    Returns
    -------
    torch tensor
        persistent state to store for next time.
    """
    spike = (spike > 0).to(state.dtype)
    return state * (1 - spike)


class _LIDynamics(torch.autograd.Function):
    """ """
    DEBUG = False

    @staticmethod
    def forward(ctx, input, decay, state, threshold, w_scale, quantize):
        """ """
        output = _li_dynamics_fwd(
            input, decay, state, threshold,
            w_scale, dtype=torch.int64, quantize=quantize
        )

#         if _LIDynamics.DEBUG is True and input.is_cuda is True:
#             _output, *_ = Accelerated.leaky_integrator.fwd(
#                 input, decay, state, threshold, w_scale
#             )
#             # print('Fwd Checking')
#             for i in range(output.shape[1]):
#                 if torch.norm(output[0, i] - _output[0, i]) > 1e-6:
#                     print('output:', i, torch.norm(
#                         output[0, i] - _output[0, i]
#                     ))
#                     print(output[0, i, :50] * w_scale)
#                     print(_output[0, i, :50] * w_scale)
#                     print(
#                         torch.norm(output - _output),
#                         output[output != _output] * w_scale,
#                         _output[output != _output] * w_scale
#                     )

#                     # this is due to int32 value underflow
#                     if (output[output != _output] * w_scale).min() > -524287:
#                         raise Exception
#                     raise Exception

        ctx.save_for_backward(output, decay)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """ """
        output, decay = ctx.saved_tensors

        grad_input, grad_decay = _li_dynamics_bwd(grad_output, output, decay)

#         if _LIDynamics.DEBUG is True and grad_output.is_cuda is True:
#             _grad_input, _grad_decay = Accelerated.leaky_integrator.bwd(
#                 grad_output, output, decay
#             )

#             # print('Bwd Checking')
#             for i in range(grad_input.shape[1]):
#                 diff = torch.abs(grad_input[0, i] - _grad_input[0, i])
#                 rel_diff = diff / grad_input[0, i].max()
#                 if torch.norm(
#                     grad_input[0, i] - _grad_input[0, i]
#                 ) / torch.numel(grad_input[0, i]) > 1e-6:
#                     print('grad_input:', i, torch.norm(
#                         grad_input[0, i] - _grad_input[0, i]
#                     ))
#                     print(grad_input[0, i, -50:])
#                     print(_grad_input[0, i, -50:])
#                     ind = torch.max(rel_diff, dim=-1)[1].item()
#                     print(ind, rel_diff.shape)
#                     print(rel_diff.mean(), rel_diff.max(), rel_diff[ind])
#                     print(
#                         grad_input[0, i, ind:ind + 50],
#                         _grad_input[0, i, ind:ind + 50],
#                         rel_diff[ind:ind + 50],
#                     )
#                     raise Exception
#             if torch.norm(
#                 grad_decay - _grad_decay
#             ) / torch.numel(grad_decay) > 1e-2:
#                 print('grad_decay:', i, torch.norm(grad_decay - _grad_decay))
#                 print(grad_decay[:50])
#                 print(_grad_decay[:50])
#                 print(
#                     torch.norm(grad_decay - _grad_decay),
#                     grad_decay[grad_decay != _grad_decay],
#                     _grad_decay[grad_decay != _grad_decay],
#                 )
#                 raise Exception

        return grad_input, grad_decay, None, None, None, None

# the original version, should be equivalent to cuda dynamics
def _li_dynamics_fwd_og(
    input, decay, state, threshold, w_scale, dtype=torch.int32, quantize=False
):
    """ """
    output_old = (state * w_scale).clone().detach().to(dtype).to(input.device)
    decay_int = 4096 - decay.clone().detach().to(dtype).to(input.device)
    output = torch.zeros_like(input)

    threshold *= w_scale

    for n in range(input.shape[-1]):
        output_new = right_shift_to_zero(output_old * decay_int, 12)+ (w_scale * input[..., n]).to(dtype)
        #output_new = Q2Zero.apply(output_old * decay_int, quantize) + (w_scale * input[..., n]).to(dtype) #right_shift_to_zero(output_old * decay_int, 12) + \
            
        if threshold > 0:
            # spike_new = (output_new >= threshold)
            # output_old = output_new * (spike_new < 0.5)
            output_old =torch.where(output_new>=threshold, torch.tensor(0,dtype=output_new.dtype).to(input.device), output_new)
        else:
            output_old = output_new

        output[..., n] = output_new / w_scale

    return output

# my customized version for quantized/non-quantized meta-learning
def _li_dynamics_fwd(
    input, decay, state, threshold, w_scale, dtype=torch.int32, quantize=False
):
    """ """
    # SETTING IT TO TRUE TO RUN THIS MODEL AND SEE HOW IT WORKS
    #print("python LI dynamics")
    if quantize:
        w_scale=4096
        output_old = (state*w_scale).clone().detach().to(dtype).to(input.device) #.to(dtype) is int64 so quantization is done here... removing
        decay_int = (w_scale) - (decay.clone().detach()).to(dtype).to(input.device) # replacing 1<<12 with 4096
    else:
        w_scale = 1 # 4096
        output_old = (state * w_scale).clone().detach().to(input.device) #.to(dtype) is int64 so quantization is done here... removing
        decay_int = 1-(decay.clone().detach()/4096).to(input.device)#(w_scale) - (decay.clone().detach()/4096).to(input.device) # replacing 1<<12 with 4096
        
    
    output = torch.zeros_like(input)

    threshold *= w_scale

    for n in range(input.shape[-1]):
        #pdb.set_trace()
        # since 12 is hardcoded will be shifted by 12 always
        # Q2Zero is apparently autograd compliant version of right_shift_to_zero...
        # print("li_fwd")
        # pdb.set_trace()
        # .to(dtype) makes it int!!! so it's quantizing Only use if quantizing
        output_new = Q2Zero.apply(output_old * decay_int, quantize)+ (w_scale * input[..., n]).to(dtype) #(output_old * (1-decay))+ (input[..., n]).to(dtype) #Q2Zero.apply(output_old * decay_int)+ (w_scale * input[..., n]).to(dtype) #right_shift_to_zero(output_old * decay_int, 12) + (w_scale * input[..., n]).to(dtype)
        if threshold > 0:
            #pdb.set_trace()
            #output_old =torch.where(output_new>=threshold, torch.tensor(0,dtype=output_new.dtype).to(input.device), output_new) # what cuda does but pytorch
            # this part makes no sense and is not in the cuda... maybe I should copy the cuda here and see if better?
            # or maybe this is some kind of soft reset but 
            spike_new = (output_new >= threshold)
            output_old = output_new * (spike_new < 0.5)
        else:
            output_old = output_new

        output[..., n] = output_new / w_scale

    return output


def _li_dynamics_bwd(grad_output, output, decay): # should be second order diff?
    """ """
    grad_input = torch.zeros_like(grad_output)
    decay = 1 - decay / (4096)

    num_steps = grad_output.shape[-1]

    grad_input[..., num_steps - 1] = grad_output[..., num_steps - 1]

    for n in range(num_steps - 1)[::-1]:
        grad_input[..., n] = decay * grad_input[..., n + 1] \
            + grad_output[..., n]

    grad_decay = grad_input[..., 1:] * output[..., :-1]

    if torch.numel(decay) == 1:  # shared parameters
        grad_decay = torch.sum(grad_decay.flatten(), dim=0, keepdim=True)
    else:
        grad_decay = torch.sum(grad_decay, dim=[0, -1])
        if len(grad_decay.shape) != 1:
            grad_decay = torch.sum(
                grad_decay.reshape(grad_decay.shape[0], -1),
                dim=1
            )

    return grad_input, grad_decay
