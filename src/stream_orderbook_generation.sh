#!/bin/bash

# Get the absolute path to the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Project directory: ${PROJECT_DIR}"

# Define the stock combinations to use
declare -A STOCK_COMBINATIONS=(
    ["combo1"]="WMT, XOM"
    ["combo6"]="BA,DIS"
    ["combo7"]="PG, RTX" 
    ["combo8"]="JPM, KO"
    ["combo9"]="WMT, XOM"
    # Uncomment these for additional combinations if needed
    # ["combo6"]="MSFT,AXP"
    # ["combo7"]="MSFT,GS"
    # ["combo8"]="JPM,IBM"
    # ["combo9"]="MSFT,BA"
    # ["combo10"]="NKE,IBM"
)

# Define parameters
NUM_STOCKS=2
NUM_ORDERS=5000
SEED=1  # Fixed seed for all combinations

# Create output directory if it doesn't exist
mkdir -p "${PROJECT_DIR}/data/generated_orderbooks"

# Find the generate_order_book.py script
SCRIPT_PATH="${PROJECT_DIR}/generate_order_book.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    # Try looking in the src directory
    SCRIPT_PATH="${PROJECT_DIR}/src/generate_order_book.py"
    if [ ! -f "$SCRIPT_PATH" ]; then
        echo "Error: Cannot find generate_order_book.py script"
        echo "Searched at:"
        echo "  ${PROJECT_DIR}/generate_order_book.py"
        echo "  ${PROJECT_DIR}/src/generate_order_book.py"
        exit 1
    fi
fi

echo "Found script at: ${SCRIPT_PATH}"

# Define the tmux sessions and their commands
declare -A sessions

# Build commands for each combination with a single seed
for combo_name in "${!STOCK_COMBINATIONS[@]}"; do
    stocks="${STOCK_COMBINATIONS[$combo_name]}"
    
    session_name="${combo_name}"
    output_dir="${PROJECT_DIR}/data/generated_orderbooks/${session_name}"
    
    # Ensure output directory exists
    mkdir -p "$output_dir"
    
    # Create the command to use generate_order_book.py with absolute path
    cmd="python ${SCRIPT_PATH} \
        --num_stocks=${NUM_STOCKS} \
        --num_orders=${NUM_ORDERS} \
        --stock_combo=\"${stocks}\" \
        --seed=${SEED} \
        --output_dir=\"${output_dir}\""
        
    sessions["${session_name}"]="$cmd"
done

echo "Will create ${#sessions[@]} tmux sessions for order book generation"

# Loop through the sessions array and create tmux sessions
for session in "${!sessions[@]}"; do
    # Check if the session already exists
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "Session $session already exists. Killing it."
        tmux kill-session -t "$session"
    fi
    
    # Create a new detached tmux session
    tmux new-session -d -s "$session"
    
    # Send the command to the session
    tmux send-keys -t "$session" "${sessions[$session]}" Enter
    
    echo "Started tmux session: $session"
    
    # Optional: Sleep briefly to prevent overloading
    sleep 0.2
done

echo "All order book generation processes started in tmux sessions."
echo ""
echo "To check sessions status: tmux list-sessions"
echo "To attach to a session: tmux attach-session -t SESSION_NAME"
echo "To detach from a session: Press Ctrl+B, then D"
echo "To kill all sessions when done: tmux kill-server" 