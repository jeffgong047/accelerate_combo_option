#!/bin/bash

# Exit on error
set -e

# Define the stock combinations to use
declare -A STOCK_COMBINATIONS=(
    ["combo1"]="AAPL,MSFT"
    ["combo2"]="GS,JPM"
    ["combo3"]="IBM,NKE" 
    ["combo4"]="DIS,KO"
    ["combo5"]="WMT,HD"
)

# Define other interesting combinations (uncomment if needed)
# declare -A MORE_COMBINATIONS=(
#     ["combo6"]="MSFT,AXP"
#     ["combo7"]="MSFT,GS"
#     ["combo8"]="JPM,IBM"
#     ["combo9"]="MSFT,BA"
#     ["combo10"]="NKE,IBM"
# )

# Define parameters
NUM_STOCKS=2
NUM_ORDERS=5000
OUTPUT_DIR="data/orderbooks"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Function to generate an order book with specified parameters
generate_orderbook() {
    local combo_name=$1
    local stocks=$2
    local seed=$3
    local output_subdir="${OUTPUT_DIR}/${combo_name}_seed${seed}"
    
    echo "Generating order book for ${combo_name} (${stocks}) with seed ${seed}"
    
    # Create subdirectory for this combination
    mkdir -p ${output_subdir}
    
    # Run the Python script with the appropriate arguments
    python generate_order_book.py \
        --num_stocks=${NUM_STOCKS} \
        --num_orders=${NUM_ORDERS} \
        --stock_combo="${stocks}" \
        --seed=${seed} \
        --output_dir="${output_subdir}"
}

# Run for different combinations in parallel
echo "Starting order book generation with ${#STOCK_COMBINATIONS[@]} combinations..."

# Launch parallel processes for different stock combinations
for combo_name in "${!STOCK_COMBINATIONS[@]}"; do
    # Generate with different seeds for each combination
    for seed in {1..3}; do
        # Launch each process in the background
        generate_orderbook "${combo_name}" "${STOCK_COMBINATIONS[$combo_name]}" ${seed} &
        
        # Limit to max 5 parallel processes
        if [[ $(jobs -r | wc -l) -ge 5 ]]; then
            # Wait for one job to finish before starting another
            wait -n
        fi
    done
done

# Wait for all remaining background processes to complete
wait

echo "All order book generation processes completed!"
echo "Generated order books are available in: ${OUTPUT_DIR}" 