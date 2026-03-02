# Copyright 2025 Google LLC
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


"""NeMo 2 log parser for checkpointing metrics."""

import re

import log_patterns
import utils
from log_parser import LogParser, register_parser


# NeMo 2 checkpoint start pattern.
_CHECKPOINT_WRITE_START = re.compile(
    r"Global Checkpoint Save : Rank: \d+ : Iteration: (\d+)"
    r" : Start time: ([\d.]+)s : Save duration: ([\d.]+)s"
)

# NeMo 2 checkpoint end pattern.
_CHECKPOINT_WRITE_END = re.compile(
    r"Async checkpoint save for step (\d+) .* finalized successfully"
)


@register_parser
class Nemo2Parser(LogParser):
  """Parser for NeMo 2 checkpoint log format."""

  @property
  def name(self) -> str:
    return "nemo2"

  @property
  def log_file_pattern(self) -> str:
    return log_patterns.NEMO_LOG_FILE_NAME

  @property
  def checkpoint_start_pattern(self) -> re.Pattern:
    return _CHECKPOINT_WRITE_START

  @property
  def checkpoint_end_pattern(self) -> re.Pattern:
    return _CHECKPOINT_WRITE_END

  def extract_step_from_start(self, match: re.Match) -> str:
    return match.group(1)

  def extract_start_time(self, match: re.Match, line: str) -> float:
    # NeMo 2 includes the epoch start time in the log message itself.
    return float(match.group(2))

  def extract_step_from_end(self, match: re.Match) -> str:
    # NeMo 2 reports step = iteration + 1 in the end message.
    return str(int(match.group(1)) - 1)

  def extract_end_time(self, match: re.Match, line: str) -> float:
    return utils.parse_nemo_timestamp(line)
