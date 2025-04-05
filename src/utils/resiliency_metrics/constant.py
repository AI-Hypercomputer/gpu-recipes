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

"""Goodput Measure related constant variables"""

USER_SCHEDULED = "user_scheduled"
USER_TERMINATED = "user_terminated"
JOB_STARTED = "job_started"
JOB_TERMINATED = "job_terminated"
CHECKPOINT_LOADED = "checkpoint_loaded"
CHECKPOINT_SAVED = "checkpoint_saved"


EVENT_TYPE_ORDER = {
    USER_SCHEDULED: 0,
    JOB_STARTED: 1,
    CHECKPOINT_LOADED: 2,
    CHECKPOINT_SAVED: 3,
    USER_TERMINATED: 4,
    JOB_TERMINATED: 5,
}
