# ---------------------------------
# Copyright 2024 KULeuven
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
# Author: Ryan Antonio (ryan.antonio@esat.kuleuven.be)
#
# Description:
# This test is a complete set where the TCDM subsystem,
# SNAX streamer, and a dummy accelerator are connected together.
# The dummy accelerator is a simple multiply streamer.
#
# Sequence of tests:
# 1. Load data through DMA
# 2. Set the streamer CSRs
# 3. Check the output of the streamer
# ---------------------------------

import cocotb
from cocotb.triggers import RisingEdge, Timer, with_timeout
from cocotb.clock import Clock
from cocotb_test.simulator import run
import snax_util
import os
import subprocess
from decimal import Decimal

from tests.cocotb.test_tcdm_subsys import MAX_VAL


# Configurable testing parameters
# In the default value below, the number
# of tests fills the entire memory
NARROW_DATA_WIDTH = 64
WIDE_DATA_WIDTH = 512
TCDM_DEPTH = 64
NR_BANKS = 32
SPATPAR = 4
BANK_INCREMENT = int(NARROW_DATA_WIDTH / 8)
WIDE_BANK_INCREMENT = int(WIDE_DATA_WIDTH / 8)
WIDE_NARROW_RATIO = int(WIDE_DATA_WIDTH / NARROW_DATA_WIDTH)
NUM_NARROW_TESTS = TCDM_DEPTH * NR_BANKS
NUM_WIDE_TESTS = int(NUM_NARROW_TESTS / 8)
MIN_VAL = 0
MAX_NARROW_VAL = 2**NARROW_DATA_WIDTH
MAX_WIDE_VAL = 2**WIDE_DATA_WIDTH

# DON'T TOUCH ME PLEASE
# CSR parameters from the default
# Configuration found under util/cfg/streamer_cfg.hjson
# Also some pre-computed that are fixed

CSR_ALU_CONFIG = 0
CSR_ALU_GPP_1 = 1
CSR_ALU_GPP_2 = 2
CSR_ALU_GPP_3 = 3
CSR_ALU_GPP_4 = 4
CSR_ALU_GPP_5 = 5
CSR_ALU_GPP_6 = 6
CSR_ALU_GPP_7 = 7

# This STREAMER_OFFSET is the offset
# For the address registers
STREAMER_OFFSET = 8

CSR_LOOP_COUNT_0 = 0 + STREAMER_OFFSET
CSR_TEMPORAL_STRIDE_0 = 1 + STREAMER_OFFSET
CSR_TEMPORAL_STRIDE_1 = 2 + STREAMER_OFFSET
CSR_TEMPORAL_STRIDE_2 = 3 + STREAMER_OFFSET
CSR_SPATIAL_STRIDE_0 = 4 + STREAMER_OFFSET
CSR_SPATIAL_STRIDE_1 = 5 + STREAMER_OFFSET
CSR_SPATIAL_STRIDE_2 = 6 + STREAMER_OFFSET
CSR_BASE_PTR_0 = 7 + STREAMER_OFFSET
CSR_BASE_PTR_1 = 8 + STREAMER_OFFSET
CSR_BASE_PTR_2 = 9 + STREAMER_OFFSET
CSR_START_STREAMER = 10 + STREAMER_OFFSET


@cocotb.test()
async def stream_alu_dut(dut):
    # Value configurations you can set
    # For exploration and testing
    # These values go into the respective
    # CSR register addresses above

    # ACLU_CONFIG has the following:
    # 0 - addition
    # 1 - subtraction
    # 2 - multiplication
    # 3 - XOR
    ALU_CONFIG = 1
    ALU_GPP_1 = 123
    ALU_GPP_2 = 456
    ALU_GPP_3 = 789
    ALU_GPP_4 = 910
    ALU_GPP_5 = 101
    ALU_GPP_6 = 121
    ALU_GPP_7 = 368

    # These ones go into the
    # streamer registers
    LOOP_COUNT_0 = 100
    TEMPORAL_STRIDE_0 = 64
    TEMPORAL_STRIDE_1 = 64
    TEMPORAL_STRIDE_2 = 64
    SPATIAL_STRIDE_0 = 8
    SPATIAL_STRIDE_1 = 8
    SPATIAL_STRIDE_2 = 8
    BASE_PTR_0 = 0
    BASE_PTR_1 = 32
    BASE_PTR_2 = 64

    # Start clock
    clock = Clock(dut.clk_i, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Reset dut
    await snax_util.reset_dut(dut)

    # Always active assuming core can
    # contiuously read data from streamer CSR

    # Let simulation time run
    for i in range(10):
        await snax_util.clock_and_wait(dut)


# Main test run
def test_streamer_gemm(simulator, waves):
    repo_path = os.getcwd()
    tests_path = repo_path + "/tests/cocotb/"

    # Make sure to generate the testbench
    # And all necessary files to make it work
    stream_gemm_tb_file = repo_path + "/tests/tb/tb_streamer_gemm.sv"
    if not os.path.exists(stream_gemm_tb_file):
        subprocess.run(["make", stream_gemm_tb_file])

    # Extract TCDM components
    tcdm_includes, tcdm_verilog_sources = snax_util.extract_tcdm_list()

    # Extract resources for simple mul
    streamer_gemm_sources = [
        repo_path + "/rtl/streamer-gemm/BareBlockGemmTop.sv",
        repo_path + "/rtl/streamer-gemm/streamer_for_gemm_wrapper.sv",
        repo_path + "/rtl/streamer-gemm/streamer_gemm_wrapper.sv",
        repo_path + "/rtl/streamer-gemm/StreamerTop.sv",
    ]

    rtl_util_sources = [
        repo_path + "/rtl/rtl-util/csr_mux_demux.sv",
    ]

    tb_verilog_source = [
        stream_gemm_tb_file,
    ]

    verilog_sources = (
        tcdm_verilog_sources
        + rtl_util_sources
        + streamer_gemm_sources
        + tb_verilog_source
    )

    defines = []
    includes = [] + tcdm_includes

    toplevel = "tb_streamer_gemm"

    module = "test_streamer_gemm"

    sim_build = tests_path + "/sim_build/{}/".format(toplevel)

    if simulator == "verilator":
        compile_args = [
            "-Wno-LITENDIAN",
            "-Wno-WIDTH",
            "-Wno-CASEINCOMPLETE",
            "-Wno-BLKANDNBLK",
            "-Wno-CMPCONST",
            "-Wno-WIDTHCONCAT",
            "-Wno-UNSIGNED",
            "-Wno-UNOPTFLAT",
            "-Wno-TIMESCALEMOD",
            "-Wno-fatal",
            "--no-timing",
            "--trace",
            "--trace-structs",
        ]
        timescale = None
    else:
        compile_args = None
        timescale = "1ns/1ps"

    run(
        verilog_sources=verilog_sources,
        includes=includes,
        toplevel=toplevel,
        defines=defines,
        module=module,
        simulator=simulator,
        sim_build=sim_build,
        compile_args=compile_args,
        waves=waves,
        timescale=timescale,
    )
