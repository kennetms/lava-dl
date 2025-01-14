# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU

from lava.lib.dl.netx.blocks.process import Input, Dense, Conv


@requires(CPU)
@tag('fixed_pt')
class AbstractPyBlockModel(AbstractSubProcessModel):
    """Abstract Block model. A block typically encapsulates at least a
    synapse and a neuron in a layer. It could also include recurrent
    connection as well as residual connection. A minimal example of a
    block is a feedforward layer."""
    def __init__(self, proc: AbstractProcess) -> None:
        if proc.input_message_bits > 0:
            self.inp: PyInPort = LavaPyType(np.ndarray,
                                            np.int32,
                                            precision=proc.input_message_bits)
        else:
            self.inp: PyInPort = LavaPyType(np.ndarray,
                                            np.int8,
                                            precision=1)

        if proc.output_message_bits > 0:
            self.out: PyOutPort = LavaPyType(np.ndarray,
                                             np.int32,
                                             precision=proc.output_message_bits)
        else:
            self.out: PyOutPort = LavaPyType(np.ndarray,
                                             np.int8,
                                             precision=1)

@requires(CPU)
@tag('floating_pt')
class AbstractPyBlockModelFloat(AbstractSubProcessModel):
    """Abstract Block model. A block typically encapsulates at least a
    synapse and a neuron in a layer. It could also include recurrent
    connection as well as residual connection. A minimal example of a
    block is a feedforward layer."""
    def __init__(self, proc: AbstractProcess) -> None:
        if proc.input_message_bits > 0:
            self.inp: PyInPort = LavaPyType(np.ndarray,
                                            float)
        else:
            self.inp: PyInPort = LavaPyType(np.ndarray,
                                            float)

        if proc.output_message_bits > 0:
            self.out: PyOutPort = LavaPyType(np.ndarray,
                                             float)
        else:
            self.out: PyOutPort = LavaPyType(np.ndarray,
                                             float)

@implements(proc=Input, protocol=LoihiProtocol)
@tag('fixed_pt')
class PyInputModel(AbstractPyBlockModel):
    def __init__(self, proc: AbstractProcess) -> None:
        super().__init__(proc)


@implements(proc=Dense, protocol=LoihiProtocol)
@tag('fixed_pt')
class PyDenseModel(AbstractPyBlockModel):
    def __init__(self, proc: AbstractProcess) -> None:
        super().__init__(proc)


@implements(proc=Conv, protocol=LoihiProtocol)
@tag('fixed_pt')
class PyConvModel(AbstractPyBlockModel):
    def __init__(self, proc: AbstractProcess) -> None:
        super().__init__(proc)

@implements(proc=Input, protocol=LoihiProtocol)
@tag('floating_pt')
class PyInputModelFloat(AbstractPyBlockModelFloat):
    def __init__(self, proc: AbstractProcess) -> None:
        super().__init__(proc)


@implements(proc=Dense, protocol=LoihiProtocol)
@tag('floating_pt')
class PyDenseModelFloat(AbstractPyBlockModelFloat):
    def __init__(self, proc: AbstractProcess) -> None:
        super().__init__(proc)


@implements(proc=Conv, protocol=LoihiProtocol)
@tag('floating_pt')
class PyConvModel(AbstractPyBlockModelFloat):
    def __init__(self, proc: AbstractProcess) -> None:
        super().__init__(proc)