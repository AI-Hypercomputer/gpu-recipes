#!/bin/bash

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


[ $# -eq 0 ] && {
    echo "Error: No prompt provided."
    echo "Usage: $0 \"Your prompt here\""
    exit 1
}

start_time=$(date +%s.%N)
temp_file="/tmp/temp_response.txt"

# format JSON payload to send to the model with streaming enabled
json_payload=$(jq -n \
    --arg prompt "$1" \
    '{
        model: "default",
        messages: [
            {role: "system", content: "You are a helpful AI assistant"},
            {role: "user", content: $prompt}
        ],
        temperature: 0.6,
        top_p: 0.95,
        max_tokens: 2048,
        stream: true
    }')

echo "Streaming response:"
echo "----------------"

# Send the request to the model and stream the response
curl -sN "http://localhost:30000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$json_payload" | while IFS= read -r line; do
    [[ -z $line ]] && continue

    line=${line#data: }
    [[ $line == "[DONE]" ]] && continue

    content=$(jq -r '.choices[0].delta.content // empty' <<< "$line")
    [[ -n $content ]] && {
        echo -n "$content"
        echo -n "$content" >> "$temp_file"
    }
done

echo -e "\n\n----------------"

[[ ! -s $temp_file ]] && {
    echo "Error: No response received from the API or an error occurred during streaming." >&2
    rm -f "$temp_file"
    exit 1
}

# Parse the response and extract the reasoning and final answer
full_content=$(<"$temp_file")

[[ $full_content =~ \<think\>([[:print:][:space:]]*)\</think\> ]] && \
    reasoning="${BASH_REMATCH[1]}" || reasoning=""

final_answer=$(sed 's/.*<\/think>//; s/^[[:space:]]*//; s/[[:space:]]*$//' <<< "$full_content")

execution_time=$(bc <<< "$(date +%s.%N) - $start_time")

echo -e "\nParsed Results:"
echo "----------------"
echo -e "Reasoning:\n$reasoning"
echo -e "\nFinal Answer:\n$final_answer"
echo -e "\nExecution time: $execution_time seconds"

rm "$temp_file"