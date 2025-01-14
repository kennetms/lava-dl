# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Composable blocks processes in Lava."""

from typing import Type, Union
import numpy as np
import h5py
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.proc.dense.process import Dense as DenseSynapse
from lava.proc.dense.process import LearningDense as LearningDenseSynapse
from lava.proc.conv.process import Conv as ConvSynapse

from lava.lib.dl.netx.blocks.lava_utils import RSTDPLIF


class AbstractBlock(AbstractProcess):
    """Abstract block definition.

    Parameters
    ----------
    shape : tuple or list
        shape of the block output in (x, y, z) or WHC format.
    neuron_params : dict, optional
        dictionary of neuron parameters. Defaults to None.
    input_message_bits : int, optional
        number of message bits in input spike. Defaults to 0 meaning unary
        spike.
    """

    def __init__(self, **kwargs: Union[dict, tuple, list, int, bool]) -> None:
        super().__init__(**kwargs)
        self.shape: tuple = kwargs.pop('shape')
        self.neuron_params = kwargs.pop('neuron_params', None)
        self.neuron_process = None
        if self.neuron_params is not None:
            self.neuron_process = self.neuron_params.pop('neuron_proc', None)
        self.bias_key = self.neuron_params.pop('bias_key', 'bias')
        self.neuron = None
        self.input_message_bits = kwargs.pop('input_message_bits', False)
        self.output_message_bits = 0
        if 'num_message_bits' in self.neuron_params.keys():
            num_msg_bits = self.neuron_params.pop('num_message_bits')
            self.output_message_bits = num_msg_bits

    def _clean(self) -> None:
        del self.neuron_params
        del self.neuron_process

    def _neuron(self, bias: np.ndarray) -> Type[AbstractProcess]:
        if bias is None:
            return self.neuron_process(shape=self.shape, **self.neuron_params)
        else:
            if len(self.shape) == 1:  # scalar bias value
                bias_mant = bias
            elif list(bias.shape) == list(self.shape):
                # neuron wise bias value
                bias_mant = bias
            elif self.shape[-1] == bias.shape[0]:  # channel wise bias value
                bias_mant = np.zeros(self.shape)
                bias_mant[:, :] = bias
            else:
                raise RuntimeError(
                    f'Bias of shape {bias.shape} could not be broadcast to '
                    f'neuron of shape {self.shape}.'
                )
            neuron_params = {'shape': self.shape, **self.neuron_params}
            neuron_params[self.bias_key] = bias_mant.astype(np.int32)
            return self.neuron_process(**neuron_params)

    @property
    def block(self) -> str:
        return str(self.__class__.__name__)


class Input(AbstractBlock):
    """Input layer block.

    Parameters
    ----------
    shape : tuple or list
        shape of the layer block in (x, y, z)/WHC format.
    neuron_params : dict, optional
        dictionary of neuron parameters. Defaults to None.
    transform : fx pointer or lambda
        input transform to be applied. Defualts to ``lambda x: x``.
    bias : np.ndarray or None
        bias of input neuron. None means no bias. Defaults to None.
    input_message_bits : int, optional
        number of message bits in input spike. Defaults to 0 meaning unary
        spike.
    """

    def __init__(self, **kwargs: Union[dict, tuple, list, int, bool]) -> None:
        super().__init__(**kwargs)
        self.transform = kwargs.pop('transform', lambda x: x)

        self.neuron = self._neuron(kwargs.pop('bias', None))

        # The input must be handled through neuron's bias.
        self.inp = InPort(shape=self.neuron.a_in.shape)
        self.out = OutPort(shape=self.neuron.s_out.shape)
        self.neuron.s_out.connect(self.out)

        self._clean()

    def export_hdf5(self, handle: Union[h5py.File, h5py.Group]) -> None:
        raise NotImplementedError


class Dense(AbstractBlock):
    """Dense layer block.

    Parameters
    ----------
    shape : tuple or list
        shape of the layer block in (x, y, z)/WHC format.
    neuron_params : dict, optional
        dictionary of neuron parameters. Defaults to None.
    weight : np.ndarray
        synaptic weight.
    bias : np.ndarray or None
        bias of neuron. None means no bias. Defaults to None.
    has_graded_input : dict
        flag for graded spikes at input. Defaults to False.
    num_weight_bits : int
        number of weight bits. Defaults to 8.
    weight_exponent : int
        weight exponent value. Defaults to 0.
    input_message_bits : int, optional
        number of message bits in input spike. Defaults to 0 meaning unary
        spike.
    """

    def __init__(self, **kwargs: Union[dict, tuple, list, int, bool]) -> None:
        super().__init__(**kwargs)

        weight = kwargs.pop('weight')
        num_weight_bits = kwargs.pop('num_weight_bits', 8)
        weight_exponent = kwargs.pop('weight_exponent', 0)

        self.synapse = DenseSynapse(
            weights=weight,
            weight_exp=weight_exponent,
            num_weight_bits=num_weight_bits,
            num_message_bits=self.input_message_bits,
        )

        if self.shape != self.synapse.a_out.shape:
            raise RuntimeError(
                f'Expected synapse output shape to be {self.shape[-1]}, '
                f'found {self.synapse.a_out.shape}.'
            )

        self.neuron = self._neuron(kwargs.pop('bias', None))

        self.inp = InPort(shape=self.synapse.s_in.shape)
        self.out = OutPort(shape=self.neuron.s_out.shape)
        self.inp.connect(self.synapse.s_in)
        self.synapse.a_out.connect(self.neuron.a_in)
        self.neuron.s_out.connect(self.out)

        self._clean()

    def export_hdf5(self, handle: Union[h5py.File, h5py.Group]) -> None:
        raise NotImplementedError

class LearningDense(AbstractBlock):
    """Learning Dense layer block.

    Parameters
    ----------
    shape : tuple or list
        shape of the layer block in (x, y, z)/WHC format.
    neuron_params : dict, optional
        dictionary of neuron parameters. Defaults to None.
    weight : np.ndarray
        synaptic weight.
    bias : np.ndarray or None
        bias of neuron. None means no bias. Defaults to None.
    has_graded_input : dict
        flag for graded spikes at input. Defaults to False.
    num_weight_bits : int
        number of weight bits. Defaults to 8.
    weight_exponent : int
        weight exponent value. Defaults to 0.
    input_message_bits : int, optional
        number of message bits in input spike. Defaults to 0 meaning unary
        spike.
    """

    def __init__(self, **kwargs: Union[dict, tuple, list, int, bool]) -> None:
        super().__init__(**kwargs)

        weight = kwargs.pop('weight')
        num_weight_bits = kwargs.pop('num_weight_bits', 7)
        weight_exponent = kwargs.pop('weight_exponent', -1)
        learning_rule = kwargs.pop('learning_rule')

        self.synapse_learning = LearningDenseSynapse(
            weights=weight,
            weight_exp=weight_exponent,
            num_weight_bits=num_weight_bits,
            num_message_bits=self.input_message_bits,
            learning_rule=learning_rule
        )

        self.third_factor_syn = DenseSynapse(weights=np.eye(self.shape[0]), num_message_bits=24)

        if self.shape != self.synapse_learning.a_out.shape:
            raise RuntimeError(
                f'Expected synapse output shape to be {self.shape[-1]}, '
                f'found {self.synapse_learning.a_out.shape}.'
            )

        #self.neuron = self._neuron(kwargs.pop('bias', None))
        self.neuron = RSTDPLIF(shape=self.shape, **self.neuron_params, learning_rule=learning_rule)

        self.inp = InPort(shape=self.synapse_learning.s_in.shape)
        self.out = OutPort(shape=self.neuron.s_out.shape)
        self.inp.connect(self.synapse_learning.s_in)
        self.synapse_learning.a_out.connect(self.neuron.a_in)
        self.neuron.s_out.connect(self.out)

        self.third_factor_syn.a_out.connect(self.neuron.a_third_factor_in)

        # connect traces
        self.neuron.s_out_bap.connect(self.synapse_learning.s_in_bap)

        self.neuron.s_out_y1.connect(self.synapse_learning.s_in_y1)
        self.neuron.s_out_y2.connect(self.synapse_learning.s_in_y2)
        self.neuron.s_out_y3.connect(self.synapse_learning.s_in_y3)

        self._clean()

    def export_hdf5(self, handle: Union[h5py.File, h5py.Group]) -> None:
        raise NotImplementedError

class Conv(AbstractBlock):
    """Conv layer block.

    Parameters
    ----------
    shape : tuple or list
        shape of the layer block in (x, y, z)/WHC format.
    input_shape : tuple or list
        shape of input layer in (x, y, z)/WHC format.
    neuron_params : dict, optional
        dictionary of neuron parameters. Defaults to None.
    weight : np.ndarray
        kernel weight.
    bias : np.ndarray or None
        bias of neuron. None means no bias. Defaults to None.
    stride : int or tuple of ints, optional
        convolution stride. Defaults to 1.
    padding : int or tuple of ints, optional
        convolution padding. Defaults to 0.
    dilation : int or tuple of ints, optional
        convolution dilation. Defaults to 1.
    groups : int
        convolution groups. Defaults to 1.
    input_message_bits : int, optional
        number of message bits in input spike. Defaults to 0 meaning unary
        spike.
    """

    def __init__(self, **kwargs: Union[dict, tuple, list, int, bool]) -> None:
        super().__init__(**kwargs)
        weight = kwargs.pop('weight')

        self.synapse = ConvSynapse(
            input_shape=kwargs.pop('input_shape'),
            weight=weight,
            stride=kwargs.pop('stride', 1),
            padding=kwargs.pop('padding', 0),
            dilation=kwargs.pop('dilation', 1),
            groups=kwargs.pop('groups', 1),
            num_message_bits=self.input_message_bits,
        )

        if list(self.shape) != list(self.synapse.output_shape):
            raise RuntimeError(
                f'Expected synapse output shape to be {self.shape}, '
                f'found {self.synapse.output_shape}.'
            )

        self.neuron = self._neuron(kwargs.pop('bias', None))

        self.inp = InPort(shape=self.synapse.s_in.shape)
        self.out = OutPort(shape=self.neuron.s_out.shape)
        self.inp.connect(self.synapse.s_in)
        self.synapse.a_out.connect(self.neuron.a_in)
        self.neuron.s_out.connect(self.out)

        self._clean()

    def export_hdf5(self, handle: Union[h5py.File, h5py.Group]) -> None:
        raise NotImplementedError
