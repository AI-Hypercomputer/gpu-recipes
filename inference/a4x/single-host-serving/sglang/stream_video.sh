#!/bin/bash

[ $# -eq 0 ] && { echo "Usage: $0 \"Your prompt\""; exit 1; }

PROMPT="$1"

POD_NAME=$(kubectl get pods --no-headers -o custom-columns=":metadata.name" | grep "${USER}-serving-wan2-2-model" | head -n 1)

if [ -z "$POD_NAME" ]; then
    echo "Error: Could not find a running Wan2.2 pod."
    echo "Please ensure your deployment is active."
    exit 1
fi

echo "Using Pod: $POD_NAME"
echo "Submitting Video Job..."

RESPONSE=$(kubectl exec "$POD_NAME" -- curl -s -X POST "http://localhost:8000/v1/videos" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"Wan-AI/Wan2.2-T2V-A14B-Diffusers\",
        \"prompt\": \"$PROMPT\",
        \"num_frames\": 81,
        \"fps\": 16
    }")

JOB_ID=$(echo "$RESPONSE" | jq -r '.id')

if [ "$JOB_ID" == "null" ] || [ -z "$JOB_ID" ]; then
    echo "Error: Failed to get Job ID. Response: $RESPONSE"
    exit 1
fi

echo "Job Submitted! ID: $JOB_ID"

echo -n "Rendering Video..."
while true; do
    # Check status inside the pod
    STATUS_REPLY=$(kubectl exec "$POD_NAME" -- curl -s "http://localhost:8000/v1/videos/$JOB_ID")
    STATUS=$(echo "$STATUS_REPLY" | jq -r '.status')
    PROGRESS=$(echo "$STATUS_REPLY" | jq -r '.progress')

    if [ "$STATUS" == "completed" ]; then
        FILE_PATH=$(echo "$STATUS_REPLY" | jq -r '.file_path')
        echo -e "\nSuccess! Video generated at: $FILE_PATH"
        echo "To download run: kubectl cp $POD_NAME:$FILE_PATH ./output.mp4"
        break
    elif [ "$STATUS" == "failed" ]; then
        ERROR_MSG=$(echo "$STATUS_REPLY" | jq -r '.error')
        echo -e "\nError during generation: $ERROR_MSG"
        exit 1
    else
        # Print progress percentage if available, otherwise dots
        if [ "$PROGRESS" != "null" ] && [ "$PROGRESS" != "0" ]; then
            echo -ne "\rRendering Video... $PROGRESS%"
        else
            echo -n "."
        fi
        sleep 10
    fi
done
