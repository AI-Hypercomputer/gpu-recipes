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


"""Abstract base class and registry for framework-specific log parsers."""

import abc
import re


class LogParser(abc.ABC):
  """Strategy for framework-specific log parsing differences.

  Each subclass encapsulates the regex patterns and extraction logic that
  differ between NeMo versions (or other frameworks). The shared
  file-processing loop in calculate_checkpoint_metrics.py calls these
  methods instead of hardcoding behavior.
  """

  @property
  @abc.abstractmethod
  def name(self) -> str:
    """Unique identifier for this parser (e.g. 'nemo1', 'nemo2')."""

  @property
  @abc.abstractmethod
  def log_file_pattern(self) -> str:
    """Regex pattern to match valid log file paths."""

  @property
  @abc.abstractmethod
  def checkpoint_start_pattern(self) -> re.Pattern:
    """Compiled regex for checkpoint write start log lines."""

  @property
  @abc.abstractmethod
  def checkpoint_end_pattern(self) -> re.Pattern:
    """Compiled regex for checkpoint write end log lines."""

  @abc.abstractmethod
  def extract_step_from_start(self, match: re.Match) -> str:
    """Extract the step number from a checkpoint start match."""

  @abc.abstractmethod
  def extract_start_time(self, match: re.Match, line: str) -> float:
    """Extract the start time (epoch seconds) from a checkpoint start match."""

  @abc.abstractmethod
  def extract_step_from_end(self, match: re.Match) -> str:
    """Extract the step number from a checkpoint end match.

    Some frameworks adjust the step (e.g. NeMo 2 reports step = iteration + 1
    in the end message, so we subtract 1).
    """

  @abc.abstractmethod
  def extract_end_time(self, match: re.Match, line: str) -> float:
    """Extract the end time (epoch seconds) from a checkpoint end match."""

  def validate_filename(self, file_path: str) -> bool:
    """Check whether a file path matches this parser's log file pattern."""
    return re.search(self.log_file_pattern, file_path) is not None


# ---- Parser Registry ----

_PARSERS: dict[str, LogParser] = {}


def register_parser(cls):
  """Class decorator that registers a LogParser subclass."""
  instance = cls()
  _PARSERS[instance.name] = instance
  return cls


def get_parser(name: str) -> LogParser:
  """Retrieve a registered parser by name."""
  if name not in _PARSERS:
    raise ValueError(
        f"Unknown log format: '{name}'. "
        f"Available formats: {available_parsers()}"
    )
  return _PARSERS[name]


def available_parsers() -> list[str]:
  """Return a list of registered parser names."""
  return list(_PARSERS.keys())


def detect_format_from_line(line: str):
  """Try all registered parsers' start patterns against a line.

  Args:
      line: A log line to test.

  Returns:
      A (parser, match) tuple if a parser's start pattern matches,
      or (None, None) if no parser matches.
  """
  for parser in _PARSERS.values():
    match = parser.checkpoint_start_pattern.search(line)
    if match:
      return parser, match
  return None, None


def default_filename_validator(file_path: str) -> bool:
  """Validate filenames using any registered parser's log_file_pattern.

  Used in auto-detection mode before a parser is selected.
  """
  return any(p.validate_filename(file_path) for p in _PARSERS.values())
