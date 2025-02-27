#!/bin/bash

# Define the names of the sessions and the commands to run
declare -A sessions
sessions=(
    ["session2"]="python combo_stock_frontier_data_preprocessor.py --num_stocks=2"
    ["session3"]="python combo_stock_frontier_data_preprocessor.py --num_stocks=2"
    ["session4"]="python combo_stock_frontier_data_preprocessor.py --num_stocks=2"
    ["session5"]="python combo_stock_frontier_data_preprocessor.py --num_stocks=2"
    ["session6"]="python combo_stock_frontier_data_preprocessor.py --num_stocks=2"
    ["session7"]="python combo_stock_frontier_data_preprocessor.py --num_stocks=2"
    ["session8"]="python combo_stock_frontier_data_preprocessor.py --num_stocks=2"
    ["session9"]="python combo_stock_frontier_data_preprocessor.py --num_stocks=2"
    ["session10"]="python combo_stock_frontier_data_preprocessor.py --num_stocks=2"
    ["session11"]="python combo_stock_frontier_data_preprocessor.py --num_stocks=2"
    ["session12"]="python combo_stock_frontier_data_preprocessor.py --num_stocks=2"
    ["session13"]="python combo_stock_frontier_data_preprocessor.py --num_stocks=2"
    ["session14"]="python combo_stock_frontier_data_preprocessor.py --num_stocks=2"
    ["session15"]="python combo_stock_frontier_data_preprocessor.py --num_stocks=2"
)

# Loop through the sessions array
for session in "${!sessions[@]}"; do
    # Create a new detached tmux session
    tmux new-session -d -s "$session"
    
    # Send the command to the session
    tmux send-keys -t "$session" "${sessions[$session]}" Enter
done

# # Optionally, attach to the first session
# tmux attach-session -t session1