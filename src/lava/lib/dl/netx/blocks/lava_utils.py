# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import matplotlib.pyplot as plt
import typing as ty
import numpy as np

from lava.proc.lif.process import LIF, AbstractLIF, LogConfig, LearningLIF, LIFRefractory
from lava.proc.io.source import RingBuffer
from lava.proc.dense.process import LearningDense, Dense
from lava.magma.core.process.neuron import LearningNeuronProcess
from lava.proc.learning_rules.r_stdp_learning_rule import RewardModulatedSTDP
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.model.py.neuron import (
    LearningNeuronModelFloat, LearningNeuronModelFixed
)
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.proc.lif.models import (
    AbstractPyLifModelFloat, AbstractPyLifModelFixed
)


class RSTDPLIF(LearningLIF):
    pass


@implements(proc=RSTDPLIF, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class RSTDPLIFModelFloat(LearningNeuronModelFloat, AbstractPyLifModelFloat):
    """Implementation of Leaky-Integrate-and-Fire neural
    process in floating point precision with learning enabled
    to do R-STDP.
    """

    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    vth: float = LavaPyType(float, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.s_out_buff = np.zeros(proc_params["shape"])

    def spiking_activation(self):
        """Spiking activation function for Learning LIF."""
        return self.v > self.vth

    def calculate_third_factor_trace(self, s_graded_in: float) -> float:
        """Generate's a third factor reward traces based on
        graded input spikes to the Learning LIF process.

        Currently, the third factor resembles the input graded spike.
        """
        return s_graded_in

    def compute_post_synaptic_trace(self, s_out_buff):
        """Compute post-synaptic trace values for this time step.

        Parameters
        ----------
        s_out_buff : ndarray
            Spikes array.

        Returns
        ----------
        result : ndarray
            Computed post synaptic trace values.
        """
        y1_tau = self._learning_rule.post_trace_decay_tau
        y1_impulse = self._learning_rule.post_trace_kernel_magnitude

        return self.y1 * np.exp(-1 / y1_tau) + y1_impulse * s_out_buff

    def run_spk(self) -> None:
        """Calculates the third factor trace and sends it to the
        Dense process for learning.
        s_out_y1: sends the post-synaptic spike times.
        s_out_y2: sends the graded third-factor reward signal.
        """

        self.y1 = self.compute_post_synaptic_trace(self.s_out_buff)

        super().run_spk()

        a_graded_in = self.a_third_factor_in.recv()

        self.y2 = self.calculate_third_factor_trace(a_graded_in)

        self.s_out_bap.send(self.s_out_buff)
        self.s_out_y1.send(self.y1)
        self.s_out_y2.send(self.y2)
        self.s_out_y3.send(self.y3)


@implements(proc=RSTDPLIF, protocol=LoihiProtocol)
@requires(CPU)
@tag("bit_accurate_loihi", "fixed_pt")
class RSTDPLIFBitAcc(LearningNeuronModelFixed, AbstractPyLifModelFixed):
    """Implementation of RSTDP Leaky-Integrate-and-Fire neural
    process bit-accurate with Loihi's hardware LIF dynamics,
    which means, it mimics Loihi behaviour bit-by-bit.

    Currently missing features (compared to Loihi 1 hardware):

    - refractory period after spiking
    - axonal delays

    Precisions of state variables

    - du: unsigned 12-bit integer (0 to 4095)
    - dv: unsigned 12-bit integer (0 to 4095)
    - bias_mant: signed 13-bit integer (-4096 to 4095). Mantissa part of neuron
      bias.
    - bias_exp: unsigned 3-bit integer (0 to 7). Exponent part of neuron bias.
    - vth: unsigned 17-bit integer (0 to 131071).

    """

    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    vth: int = LavaPyType(int, np.int32, precision=17)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.effective_vth = 0
        self.s_out_buff = np.zeros(proc_params["shape"])
        self.refractory_period = 1
        self.refractory_period_end = np.zeros(proc_params["shape"])

    def scale_threshold(self):
        """Scale threshold according to the way Loihi hardware scales it. In
        Loihi hardware, threshold is left-shifted by 6-bits to MSB-align it
        with other state variables of higher precision.
        """
        self.effective_vth = np.left_shift(self.vth, self.vth_shift)
        self.isthrscaled = True

    def spiking_activation(self):
        """Spike when voltage exceeds threshold."""
        return self.v > self.effective_vth

    # def subthr_dynamics(self, activation_in: np.ndarray):
    #     """Sub-threshold dynamics of current and voltage variables for
    #     all refractory LIF models. This is where the 'leaky integration'
    #     happens.
    #     """
    #     # Update current
    #     # --------------
    #     decay_const_u = self.du + self.ds_offset
    #     # Below, u is promoted to int64 to avoid overflow of the product
    #     # between u and decay constant beyond int32. Subsequent right shift by
    #     # 12 brings us back within 24-bits (and hence, within 32-bits)
    #     decayed_curr = np.int64(self.u) * (self.decay_unity - decay_const_u)
    #     decayed_curr = np.sign(decayed_curr) * np.right_shift(
    #         np.abs(decayed_curr), self.decay_shift
    #     )
    #     decayed_curr = np.int32(decayed_curr)
    #     # Hardware left-shifts synaptic input for MSB alignment
    #     activation_in = np.left_shift(activation_in, self.act_shift)
    #     # Add synptic input to decayed current
    #     decayed_curr += activation_in
    #     # Check if value of current is within bounds of 24-bit. Overflows are
    #     # handled by wrapping around modulo 2 ** 23. E.g., (2 ** 23) + k
    #     # becomes k and -(2**23 + k) becomes -k
    #     wrapped_curr = np.where(
    #         decayed_curr > self.max_uv_val,
    #         decayed_curr - 2 * self.max_uv_val,
    #         decayed_curr,
    #     )
    #     wrapped_curr = np.where(
    #         wrapped_curr <= -self.max_uv_val,
    #         decayed_curr + 2 * self.max_uv_val,
    #         wrapped_curr,
    #     )
    #     self.u[:] = wrapped_curr
    #
    #     # refractory
    #     non_refractory = self.refractory_period_end < self.time_step
    #
    #     # Update voltage
    #     # --------------
    #     decay_const_v = self.dv + self.dm_offset
    #
    #     neg_voltage_limit = -np.int32(self.max_uv_val) + 1
    #     pos_voltage_limit = np.int32(self.max_uv_val) - 1
    #     # Decaying voltage similar to current. See the comment above to
    #     # understand the need for each of the operations below.
    #     decayed_volt = np.int64(self.v) * (self.decay_unity - decay_const_v)
    #     decayed_volt = np.sign(decayed_volt) * np.right_shift(
    #         np.abs(decayed_volt), self.decay_shift
    #     )
    #     decayed_volt = np.int32(decayed_volt)
    #     updated_volt = decayed_volt + self.u + self.effective_bias
    #     self.v[non_refractory] = np.clip(updated_volt[non_refractory], neg_voltage_limit, pos_voltage_limit)

    def process_spikes(self, spike_vector: np.ndarray):
        self.refractory_period_end[spike_vector] = (self.time_step
                                                    + self.refractory_period)
        super().reset_voltage(spike_vector)
    def calculate_third_factor_trace(self, s_graded_in: float) -> float:
        """Generate's a third factor reward traces based on
        graded input spikes to the Learning LIF process.

        Currently, the third factor resembles the input graded spike.
        """
        return s_graded_in

    def compute_post_synaptic_trace(self, s_out_buff):
        """Compute post-synaptic trace values for this time step.

        Parameters
        ----------
        s_out_buff : ndarray
            Spikes array.

        Returns
        ----------
        result : ndarray
            Computed post synaptic trace values.
        """
        y1_tau = self._learning_rule.post_trace_decay_tau
        y1_impulse = self._learning_rule.post_trace_kernel_magnitude

        return np.floor(self.y1 * np.exp(-1 / y1_tau) + y1_impulse * s_out_buff)

    def run_spk(self) -> None:
        """Calculates the third factor trace and sends it to the
        Dense process for learning.
        s_out_y1: sends the post-synaptic spike times.
        s_out_y2: sends the graded third-factor reward signal.
        """

        self.y1 = self.compute_post_synaptic_trace(self.s_out_buff)

        #super().run_spk()
        # Receive synaptic input
        a_in_data = self.a_in.recv()

        self.subthr_dynamics(activation_in=a_in_data)

        s_out = self.spiking_activation()

        # Reset voltage of spiked neurons to 0
        self.process_spikes(spike_vector=s_out)
        self.s_out.send(s_out)

        a_graded_in = self.a_third_factor_in.recv()

        self.y2 = self.calculate_third_factor_trace(a_graded_in)

        self.s_out_bap.send(self.s_out_buff)
        self.s_out_y1.send(self.y1)
        self.s_out_y2.send(self.y2)
        self.s_out_y3.send(self.y3)


def generate_post_spikes(pre_spike_times,
                         num_steps, spike_prob_post):
    """generates specific post synaptic spikes to
    demonstrate potentiation and depression.
    """
    pre_synaptic_spikes = np.where(pre_spike_times == 1)[1]

    spike_raster_post = np.zeros((len(spike_prob_post), num_steps))

    for ts in range(num_steps):
        for pre_ts in pre_synaptic_spikes:
            if ts in range(pre_ts, pre_ts + 20):
                if np.random.rand(1) < spike_prob_post[0]:
                    spike_raster_post[0][ts] = 1

    for ts in range(num_steps):
        for pre_ts in pre_synaptic_spikes:
            if ts in range(pre_ts - 12, pre_ts - 2):
                if np.random.rand(1) < spike_prob_post[1]:
                    spike_raster_post[1][ts] = 1

    return spike_raster_post


def plot_spikes(spikes, figsize, legend, colors, title, num_steps):
    offsets = list(range(1, len(spikes) + 1))
    num_x_ticks = np.arange(0, num_steps + 1, 25)

    plt.figure(figsize=figsize)

    spikes_plot = plt.eventplot(positions=spikes,
                                lineoffsets=offsets,
                                linelength=0.9,
                                colors=colors)

    plt.title(title)
    plt.xlabel("Time steps")
    plt.ylabel("Neurons")

    plt.xticks(num_x_ticks)
    plt.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    plt.grid(which='major', color='lightgray', linewidth=0.8)
    plt.minorticks_on()

    plt.yticks(ticks=offsets, labels=legend)

    plt.show()


def plot_time_series(time, time_series, ylabel, title, figsize, color):
    plt.figure(figsize=figsize)
    plt.step(time, time_series, color=color)

    plt.title(title)
    plt.xlabel("Time steps")
    plt.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    plt.grid(which='major', color='lightgray', linewidth=0.8)
    plt.minorticks_on()

    plt.ylabel(ylabel)

    plt.show()


def plot_time_series_subplots(time, time_series_y1, time_series_y2, ylabel, title, figsize, color, legend,
                              leg_loc="upper left"):
    plt.figure(figsize=figsize)

    plt.step(time, time_series_y1, label=legend[0], color=color[0])
    plt.step(time, time_series_y2, label=legend[1], color=color[1])

    plt.title(title)
    plt.xlabel("Time steps")
    plt.ylabel(ylabel)
    plt.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    plt.grid(which='major', color='lightgray', linewidth=0.8)
    plt.minorticks_on()
    plt.xlim(0, len(time_series_y1))

    plt.legend(loc=leg_loc)

    plt.show()


def plot_spikes_time_series(time, time_series, spikes, figsize, legend, colors, title, num_steps):
    offsets = list(range(1, len(spikes) + 1))
    num_x_ticks = np.arange(0, num_steps + 1, 25)

    plt.figure(figsize=figsize)

    plt.subplot(211)
    plt.eventplot(positions=spikes,
                  lineoffsets=offsets,
                  linelength=0.9,
                  colors=colors)

    plt.title("Spike Arrival")
    plt.xlabel("Time steps")

    plt.xticks(num_x_ticks)
    plt.xlim(0, num_steps)
    plt.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    plt.grid(which='major', color='lightgray', linewidth=0.8)
    plt.minorticks_on()

    plt.yticks(ticks=offsets, labels=legend)
    plt.tight_layout(pad=3.0)

    plt.subplot(212)
    plt.step(time, time_series, color=colors)

    plt.title(title[0])
    plt.xlabel("Time steps")
    plt.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    plt.grid(which='major', color='lightgray', linewidth=0.8)
    plt.minorticks_on()
    plt.margins(x=0)

    plt.ylabel("Trace Value")

    plt.show()
