#!/bin/bash

# Script to launch MMPose inferencer demo with optional instance shutdown
# Usage: ./launch_inferencer.sh --json-file <path> --limit <number> [--shutdown]

set -e  # Exit on any error

# Default values
JSON_FILE=""
LIMIT=""
SHUTDOWN=false

# Function to display usage
usage() {
    echo "Usage: $0 --json-file <path> --limit <number> [--shutdown]"
    echo ""
    echo "Options:"
    echo "  --json-file <path>    Path to the JSON file containing video data"
    echo "  --limit <number>      Maximum number of videos to process"
    echo "  --shutdown           Shutdown the instance after processing (optional)"
    echo "  -h, --help           Display this help message"
    echo ""
    echo "Example:"
    echo "  $0 --json-file s3://bucket/data.json --limit 100 --shutdown"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --json-file)
            JSON_FILE="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --shutdown)
            SHUTDOWN=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$JSON_FILE" ]]; then
    echo "Error: --json-file is required"
    usage
fi

if [[ -z "$LIMIT" ]]; then
    echo "Error: --limit is required"
    usage
fi

# Validate limit is a number
if ! [[ "$LIMIT" =~ ^[0-9]+$ ]]; then
    echo "Error: --limit must be a positive integer"
    exit 1
fi

echo "=========================================="
echo "MMPose Inferencer Demo Launcher"
echo "=========================================="
echo "JSON File: $JSON_FILE"
echo "Limit: $LIMIT"
echo "Shutdown after completion: $SHUTDOWN"
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script located at: $SCRIPT_DIR"

# Change to the script directory (project root)
cd "$SCRIPT_DIR"
echo "Changed to project directory: $(pwd)"

# Check if .venv exists in the project directory
if [[ ! -d ".venv" ]]; then
    echo "Error: .venv directory not found in project directory: $SCRIPT_DIR"
    echo "Please ensure .venv exists in the project root"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if the inferencer demo exists
if [[ ! -f "demo/inferencer_demo.py" ]]; then
    echo "Error: demo/inferencer_demo.py not found"
    echo "Please ensure you're running this script from the project root"
    exit 1
fi

# Record start time
START_TIME=$(date)
echo "Starting processing at: $START_TIME"

# Run the inferencer demo
echo "Launching MMPose inferencer demo..."
python demo/inferencer_demo.py --json-file "$JSON_FILE" --limit "$LIMIT"

# Record end time
END_TIME=$(date)
echo ""
echo "=========================================="
echo "Processing completed successfully!"
echo "Start time: $START_TIME"
echo "End time: $END_TIME"
echo "=========================================="

# Deactivate virtual environment
deactivate

# Shutdown instance if requested
if [[ "$SHUTDOWN" == true ]]; then
    echo ""
    echo "Shutdown requested. Shutting down instance in 30 seconds..."
    echo "Press Ctrl+C to cancel shutdown."
    
    # Give user a chance to cancel
    for i in {30..1}; do
        echo "Shutting down in $i seconds..."
        sleep 1
    done
    
    echo "Shutting down now..."
    sudo shutdown -h now
else
    echo ""
    echo "Processing complete. Instance will remain running."
fi 