#!/bin/bash

# Define the combinations and seeds to generate
declare -A sessions
sessions=(
    ["aapl_msft_1"]="python generate_order_book.py --num_stocks=2 --num_orders=5000 --stock_combo=AAPL,MSFT --seed=1"
    ["aapl_msft_2"]="python generate_order_book.py --num_stocks=2 --num_orders=5000 --stock_combo=AAPL,MSFT --seed=2"
    ["gs_jpm_1"]="python generate_order_book.py --num_stocks=2 --num_orders=5000 --stock_combo=GS,JPM --seed=1"
    ["gs_jpm_2"]="python generate_order_book.py --num_stocks=2 --num_orders=5000 --stock_combo=GS,JPM --seed=2"
    ["ibm_nke_1"]="python generate_order_book.py --num_stocks=2 --num_orders=5000 --stock_combo=IBM,NKE --seed=1"
    ["ibm_nke_2"]="python generate_order_book.py --num_stocks=2 --num_orders=5000 --stock_combo=IBM,NKE --seed=2"
    ["dis_ko_1"]="python generate_order_book.py --num_stocks=2 --num_orders=5000 --stock_combo=DIS,KO --seed=1"
    ["dis_ko_2"]="python generate_order_book.py --num_stocks=2 --num_orders=5000 --stock_combo=DIS,KO --seed=2"
    ["wmt_hd_1"]="python generate_order_book.py --num_stocks=2 --num_orders=5000 --stock_combo=WMT,HD --seed=1"
    ["wmt_hd_2"]="python generate_order_book.py --num_stocks=2 --num_orders=5000 --stock_combo=WMT,HD --seed=2"
)

# Create output directory
mkdir -p data/orderbooks

# Loop through the sessions array and create tmux sessions
for session in "${!sessions[@]}"; do
    # Create a new detached tmux session
    tmux new-session -d -s "$session"
    
    # Send the command to the session
    tmux send-keys -t "$session" "${sessions[$session]}" Enter
    
    echo "Started session: $session"
    
    # Sleep briefly to prevent tmux from getting overwhelmed
    sleep 0.5
done

echo "All tmux sessions started. Use 'tmux ls' to see running sessions."
echo "To attach to a session: tmux attach-session -t SESSION_NAME"
echo "To kill a session: tmux kill-session -t SESSION_NAME" 