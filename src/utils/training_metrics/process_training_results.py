# Copyright 2024 Google LLC
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

""" Tool to calculate training metrics for common LLM models"""

import argparse
import json

from src.data_defs import MODEL_FLOPS_PER_SAMPLE, MAX_TFLOPS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="DLLogger file")
    parser.add_argument(
        "--model_flops",
        type=float,
        required=False,
        help="Model flops fw + bw per 1 sample. If not provided will use defaults values in code",
    )
    parser.add_argument(
        "--max_flops",
        type=float,
        required=False,
        help="Max theoretical TFLOPS. If not provided, default values will be used for the accelerator type on bf16 precision",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Global batch size used during training.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=list(MODEL_FLOPS_PER_SAMPLE.keys()),
        help="Type of model",
    )
    parser.add_argument(
        "--num_accelerators",
        type=int,
        required=True,
        help="Number of GPUs/TPUs used for training",
    )
    parser.add_argument(
        "--accelerator_type",
        type=str,
        choices=list(MAX_TFLOPS.keys()),
        help="Number of GPUs used for training",
    )
    parser.add_argument(
        "--start_step",
        type=int,
        required=False,
        default=10,
        help="Start step to compute the training step time",
    )
    parser.add_argument(
        "--end_step",
        type=int,
        required=False,
        default=30,
        help="Start step to compute the training step time",
    )

    return parser.parse_args()


def compute_mfu(
    step_time: float,
    max_tflops: str,
    num_accelerators: int,
    model_flops_per_sample: float,
    batch_size: int,
) -> float:
    """Computes the MFU

    Args:
        step_time (float): forward + backward step time in seconds
        max_tflops (str): Max theoretical TFLOPs supported by the accelerator used
        num_accelerators (int): Number of accelerators used during the training process
        model_flops_per_sample (float): Number of FLOPS for a single sample training step
        batch_size (int): Global batch size used during training

    Returns:
        float: Returns the Model FLOPS Utilization MFU
    """
    tflops_per_accelerator = (
        model_flops_per_sample * batch_size / step_time / num_accelerators / 1e12
    )
    mfu = tflops_per_accelerator / max_tflops

    print(f"Average step time: {step_time:.6f}")
    print(f"TFLOPS/Accelerator {tflops_per_accelerator:.4f}")
    print(f"MFU:{mfu:.4f}")
    return mfu


def get_average_step_time(file: str, start_step: int, end_step: int) -> float:
    """Computes the average step time from a dllogger json file
    between the step start_step and end_step, both included.

    Args:
        file (str): path to the dllogger file to use

    Returns:
        float: average step time between the steps start_step and end_step
    """

    with open(file, "r", encoding="utf-8") as f:
        data = f.readlines()

    datajson = [json.loads(line[4:]) for line in data]
    time_step_accumulator = 0
    num_steps = 0
    for line in datajson:
        if line.get("step") != "PARAMETER":
            step = line.get("step")
            if step >= start_step and step <= end_step:
                time_step_accumulator += line["data"].get("train_step_timing in s")
                num_steps += 1
    if num_steps == 0:
        raise ValueError(
            "Make sure your dllogger.json file contains steps in the range of --start_step and --end_step"
        )
    return time_step_accumulator / num_steps


def main(args):
    """Main processing"""
    if args.model_type is None and args.model_flops is None:
        print("Either the --model_type or --model_flops is needed")
        return
    if args.accelerator_type is None and args.max_flops is None:
        print("Either the --accelerator_type or --max_flops is needed")
        return

    model_flops_per_sample = (
        args.model_flops
        if args.model_flops
        else MODEL_FLOPS_PER_SAMPLE[args.model_type]
    )
    max_tflops = args.max_flops if args.max_flops else MAX_TFLOPS[args.accelerator_type]
    average_step_time = get_average_step_time(
        args.file, start_step=args.start_step, end_step=args.end_step
    )
    compute_mfu(
        step_time=average_step_time,
        max_tflops=max_tflops,
        num_accelerators=args.num_accelerators,
        model_flops_per_sample=model_flops_per_sample,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
