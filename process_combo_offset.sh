#!/bin/bash

# This script processes a single stock combination with a specific offset setting
# Usage: process_combo_offset.sh <STOCK1> <STOCK2> <SEED> <DEBUG_MODE> <OUTPUT_DIR> <NOISE> <NUM_STOCKS> <MARKET_SIZE> <NUM_ORDERS> <OFFSET>

STOCK1="$1"
STOCK2="$2"
SEED="$3"
DEBUG_MODE="$4"
OUTPUT_DIR="$5"
NOISE="$6"
NUM_STOCKS="$7"
MARKET_SIZE="$8"
NUM_ORDERS="$9"
OFFSET="${10}"

COMBO="${STOCK1}_${STOCK2}"
LOG_DIR="$OUTPUT_DIR/logs"

# Set up offset name for display and directories
if [ "$OFFSET" = "True" ]; then
  OFFSET_NAME="with_offset"
  OFFSET_DIR="offset1"
else
  OFFSET_NAME="no_offset"
  OFFSET_DIR="offset0"
fi

echo "Processing stock combination: $STOCK1 and $STOCK2 with offset=$OFFSET"

# Check if orderbook file exists before proceeding
ORDERBOOK_FILE="/common/home/hg343/Research/accelerate_combo_option/data/generated_orderbooks/combo2/STOCK_${NUM_STOCKS}_SEED_${SEED}_book_${COMBO}.npy"
ALTERNATIVE_PATH="combinatorial/book/STOCK_${NUM_STOCKS}_SEED_${SEED}_book_${COMBO}.npy"

if [ ! -f "$ORDERBOOK_FILE" ] && [ ! -f "$ALTERNATIVE_PATH" ]; then
  echo "ERROR: Orderbook file not found for combination ${COMBO} with seed ${SEED}"
  echo "Looked in paths:"
  echo "  - $ORDERBOOK_FILE"
  echo "  - $ALTERNATIVE_PATH"
  echo "Skipping processing for this combination."
  
  # Create the log directory and write error to log
  mkdir -p "$LOG_DIR"
  LOG_FILE="$LOG_DIR/${COMBO}_seed${SEED}_${OFFSET_DIR}.log"
  
  echo "ERROR: Orderbook file not found for combination ${COMBO} with seed ${SEED}" > "$LOG_FILE"
  echo "Looked in paths:" >> "$LOG_FILE"
  echo "  - $ORDERBOOK_FILE" >> "$LOG_FILE"
  echo "  - $ALTERNATIVE_PATH" >> "$LOG_FILE"
  echo "Skipping processing for this combination." >> "$LOG_FILE"
  
  exit 1
fi

# Process with the specified offset
LOG_FILE="$LOG_DIR/${COMBO}_seed${SEED}_${OFFSET_DIR}.log"
OUTPUT_PATH="$OUTPUT_DIR/${COMBO}/seed${SEED}/${OFFSET_DIR}"
mkdir -p $OUTPUT_PATH

echo "Found orderbook file. Starting processing."
echo "Log file: $LOG_FILE"
echo "Output path: $OUTPUT_PATH"

# Set up debug flag if debug mode is enabled
DEBUG_FLAG=""
if [ "$DEBUG_MODE" = "1" ]; then
  DEBUG_FLAG="--debug"
  echo "Debug mode enabled - will use breakpoints for debugging"
fi

# Set up verbose flag for better logging
VERBOSE_FLAG="--verbose"

# Run the Python script with proper error handling
python src/combo_stock_frontier_data_preprocessor_forked.py \
  --num_stocks $NUM_STOCKS \
  --market_size $MARKET_SIZE \
  --offset $OFFSET \
  --noise $NOISE \
  --num_orders $NUM_ORDERS \
  --stock_combo "$STOCK1,$STOCK2" \
  --seed $SEED \
  --output_dir "$OUTPUT_PATH" \
  $DEBUG_FLAG \
  $VERBOSE_FLAG > "$LOG_FILE" 2>&1

# Check if the Python script exited with an error
if [ $? -ne 0 ]; then
  echo "ERROR: Processing failed for $COMBO with offset=$OFFSET. Check log file: $LOG_FILE"
  # Append a note to the log
  echo "--------------------------------------------------------------------------------" >> "$LOG_FILE"
  echo "PROCESSING FAILED" >> "$LOG_FILE"
  echo "If this is due to a missing orderbook file, make sure to check the following paths:" >> "$LOG_FILE"
  echo "  - $ORDERBOOK_FILE" >> "$LOG_FILE"
  echo "  - $ALTERNATIVE_PATH" >> "$LOG_FILE"
  exit 1
else
  echo "Completed processing for $COMBO with offset=$OFFSET"
fi
