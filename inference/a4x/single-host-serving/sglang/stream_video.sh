#!/bin/bash
# stream_video.sh - Utility to poll and download generated video

[ $# -eq 0 ] && { echo "Usage: $0 \"Your prompt\""; exit 1; }

PROMPT="$1"
API_URL="http://localhost:8000/v1/videos"

echo "Submitting Video Job..."

RESPONSE=$(curl -s -X POST "$API_URL" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"Wan-AI/Wan2.2-T2V-A14B-Diffusers\",
        \"prompt\": \"$PROMPT\",
        \"num_frames\": 81,
        \"fps\": 16
    }")

JOB_ID=$(echo "$RESPONSE" | jq -r '.id')
echo "Job Submitted! ID: $JOB_ID"

while true; do
    STATUS_REPLY=$(curl -s "$API_URL/$JOB_ID")
    STATUS=$(echo "$STATUS_REPLY" | jq -r '.status')
    
    if [ "$STATUS" == "completed" ]; then
        # Fetch the internal path and notify user
        FILE_PATH=$(echo "$STATUS_REPLY" | jq -r '.file_path')
        echo -e "\nSuccess! Video generated at: $FILE_PATH"
        echo "To download: kubectl cp <POD_NAME>:$FILE_PATH ./output.mp4"
        break
    elif [ "$STATUS" == "failed" ]; then
        echo "Error: $(echo "$STATUS_REPLY" | jq -r '.error')"
        exit 1
    else
        echo -n "."
        sleep 10
    fi
done
