#!/bin/bash
# stream_video.sh - Optimized for Wan2.2 on Blackwell

[ $# -eq 0 ] && {
    echo "Usage: $0 \"Your video prompt here\""
    exit 1
}

PROMPT="$1"
MODEL_NAME="Wan-AI/Wan2.2-T2V-A14B-Diffusers"
API_URL="http://localhost:8000/v1/videos"

echo "Submitting Video Job for: $PROMPT"

# 1. Submit the Job
RESPONSE=$(curl -s -X POST "$API_URL" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL_NAME\",
        \"prompt\": \"$PROMPT\",
        \"size\": \"1280x720\",
        \"num_frames\": 81
    }")

JOB_ID=$(echo "$RESPONSE" | jq -r '.id')

if [ "$JOB_ID" == "null" ]; then
    echo "Error submitting job: $RESPONSE"
    exit 1
fi

echo "Job Submitted! ID: $JOB_ID"
echo "Waiting for Blackwell to generate video..."

# 2. Poll for Completion
while true; do
    STATUS_REPLY=$(curl -s "$API_URL/$JOB_ID")
    STATUS=$(echo "$STATUS_REPLY" | jq -r '.status')

    if [ "$STATUS" == "completed" ]; then
        VIDEO_URL=$(echo "$STATUS_REPLY" | jq -r '.url')
        echo -e "\n----------------"
        echo "Success! Video URL: $VIDEO_URL"
        break
    elif [ "$STATUS" == "failed" ]; then
        echo "Generation failed!"
        echo "$STATUS_REPLY" | jq .
        exit 1
    else
        echo -n "."
        sleep 5
    fi
done
