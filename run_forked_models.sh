#!/bin/bash

# Script to run the main_combo_forked.py with different offset_type parameters
# This is useful for training models on merged data from all stock combinations
# Only processes data with noise level 0.0625 (2^-4)

# Set common parameters
DATA_DIR="/common/home/hg343/Research/accelerate_combo_option/data/frontier_labels"
MODEL_DIR="/common/home/hg343/Research/accelerate_combo_option/models"
SEED=1
HIDDEN_SIZE=64
NUM_LAYERS=4
ITERATIONS=20
BATCH_SIZE=32

# Create model directory if it doesn't exist
mkdir -p $MODEL_DIR

# Function to print section header
print_header() {
    echo "==========================================="
    echo "$1"
    echo "==========================================="
}

# Function to run the script with specific parameters
run_model() {
    local offset_type=$1
    local test_dqn=$2
    local reward_type=${3:-profit_with_penalty}
    
    print_header "Running model with offset_type=$offset_type, test_dqn=$test_dqn, reward_type=$reward_type"
    echo "Training on merged data from all stock combinations with offset=$offset_type (noise=0.0625)"
    
    cmd="python src/main_combo_forked.py \
        --data_dir $DATA_DIR \
        --model_dir $MODEL_DIR \
        --seed $SEED \
        --hidden_size $HIDDEN_SIZE \
        --num_layers $NUM_LAYERS \
        --iterations $ITERATIONS \
        --batch_size $BATCH_SIZE \
        --offset_type $offset_type"
    
    if [ "$test_dqn" = true ]; then
        cmd="$cmd --test_dqn --reward_type $reward_type"
    fi
    
    echo "Running command: $cmd"
    eval $cmd
    
    echo "Completed training on merged data with offset_type=$offset_type and noise=0.0625"
    echo
}

# First test the data loading with the test script
print_header "Testing data loading for offset_type=0 with noise=0.0625"
python src/test_load_data.py --offset_type 0 --noise_level 0.0625

print_header "Testing data loading for offset_type=1 with noise=0.0625"
python src/test_load_data.py --offset_type 1 --noise_level 0.0625

# Train models for each offset type
print_header "TRAINING MODELS ON MERGED DATA FROM ALL COMBINATIONS (NOISE=0.0625)"

run_model 0 false  # Train model for offset_type=0, no DQN testing
run_model 1 false  # Train model for offset_type=1, no DQN testing

# Optionally, run with DQN testing
if [ "$1" = "--with-dqn" ]; then
    print_header "RUNNING DQN TESTS ON MERGED DATA (NOISE=0.0625)"
    
    run_model 0 true profit_with_penalty  # Test DQN with offset_type=0
    run_model 1 true profit_with_penalty  # Test DQN with offset_type=1
    run_model 0 true profit_minus_liability  # Test DQN with different reward type
    run_model 1 true profit_minus_liability  # Test DQN with different reward type
fi

print_header "All model training completed"
echo "Models are saved in: $MODEL_DIR"
echo "These models were trained on merged data from all stock combinations with noise level 0.0625"
ls -l $MODEL_DIR 