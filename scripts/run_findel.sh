#!/bin/bash
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # no color

OUTPUT_DIR="./output"
mkdir -p $OUTPUT_DIR

echo -e "${YELLOW}Starting findel${NC}"
echo "======================================"

echo -e "${YELLOW}[+] Installing package in development mode${NC}"
pip install -e . || { echo -e "${RED}Failed to install package${NC}"; exit 1; }
echo -e "${GREEN}Package installed successfully${NC}"
echo

echo -e "${YELLOW}[+] Generating test data${NC}"
python scripts/generate_test_data.py --output_path data/processed/test_data.csv --data_type synthetic --n_samples 1000 || { echo -e "${RED}Failed to generate test data${NC}"; exit 1; }
echo -e "${GREEN}Test data generated successfully${NC}"
echo

echo -e "${YELLOW}[+] Training the model${NC}"
python scripts/train_model.py \
    --data_path data/processed/test_data.csv \
    --target_column returns \
    --sequence_length 60 \
    --model_type transformer \
    --hidden_dim 64 \
    --num_layers 2 \
    --loss_type financial \
    --batch_size 32 \
    --epochs 10 \
    --lr 0.001 \
    --output_dir $OUTPUT_DIR \
    --model_name test_model || { echo -e "${RED}Failed to train model${NC}"; exit 1; }
echo -e "${GREEN}Model trained successfully${NC}"
echo

echo -e "${YELLOW}[+] Evaluating the model${NC}"
python scripts/evaluate_model.py \
    --data_path data/processed/test_data.csv \
    --target_column returns \
    --sequence_length 60 \
    --model_type transformer \
    --hidden_dim 64 \
    --num_layers 2 \
    --model_path $OUTPUT_DIR/test_model.pth \
    --output_dir $OUTPUT_DIR || { echo -e "${RED}Failed to evaluate model${NC}"; exit 1; }
echo -e "${GREEN}Model evaluated successfully${NC}"
echo

echo -e "${YELLOW}[+] Training a different model type (GRU)${NC}"
python scripts/train_model.py \
    --data_path data/processed/test_data.csv \
    --target_column returns \
    --sequence_length 60 \
    --model_type gru \
    --hidden_dim 64 \
    --num_layers 2 \
    --loss_type sharpe \
    --batch_size 32 \
    --epochs 5 \
    --lr 0.001 \
    --output_dir $OUTPUT_DIR \
    --model_name gru_model || { echo -e "${RED}Failed to train GRU model${NC}"; exit 1; }
echo -e "${GREEN}GRU model trained successfully${NC}"
echo

echo -e "${YELLOW}[+] Evaluating the GRU model${NC}"
python scripts/evaluate_model.py \
    --data_path data/processed/test_data.csv \
    --target_column returns \
    --sequence_length 60 \
    --model_type gru \
    --hidden_dim 64 \
    --num_layers 2 \
    --model_path $OUTPUT_DIR/gru_model.pth \
    --output_dir $OUTPUT_DIR || { echo -e "${RED}Failed to evaluate GRU model${NC}"; exit 1; }
echo -e "${GREEN}GRU model evaluated successfully${NC}"
echo

echo -e "${YELLOW}[+] Generating real data (optional)${NC}"
python scripts/generate_test_data.py \
    --output_path data/processed/real_data.csv \
    --data_type real \
    --ticker SPY \
    --start_date 2020-01-01 \
    --end_date 2023-01-01 || { echo -e "${RED}Failed to generate real data${NC}"; echo -e "${YELLOW}Skipping real data test${NC}"; }

if [ -f "data/processed/real_data.csv" ]; then
    echo -e "${GREEN}Real data generated successfully${NC}"
    echo
    
    echo -e "${YELLOW}[+] Training model on real data${NC}"
    python scripts/train_model.py \
        --data_path data/processed/real_data.csv \
        --target_column returns \
        --sequence_length 60 \
        --model_type multitask \
        --hidden_dim 128 \
        --num_layers 2 \
        --loss_type financial \
        --batch_size 32 \
        --epochs 5 \
        --lr 0.001 \
        --output_dir $OUTPUT_DIR \
        --model_name real_model || { echo -e "${RED}Failed to train model on real data${NC}"; }
    
    if [ -f "$OUTPUT_DIR/real_model.pth" ]; then
        echo -e "${GREEN}Model trained on real data successfully${NC}"
        echo
        
        echo -e "${YELLOW}[+] Evaluating model on real data${NC}"
        python scripts/evaluate_model.py \
            --data_path data/processed/real_data.csv \
            --target_column returns \
            --sequence_length 60 \
            --model_type multitask \
            --hidden_dim 128 \
            --num_layers 2 \
            --model_path $OUTPUT_DIR/real_model.pth \
            --output_dir $OUTPUT_DIR || { echo -e "${RED}Failed to evaluate model on real data${NC}"; }
        echo -e "${GREEN}Model evaluated on real data successfully${NC}"
        echo
    fi
fi

echo "Saved results and models to the $OUTPUT_DIR directory"