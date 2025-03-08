# scripts

This directory contains scripts for training, evaluating, and testing the financial deep learning models.

- `train_model.py`: Train a financial deep learning model
- `evaluate_model.py`: Evaluate a trained model on test data
- `generate_data.py`: Generate synthetic or real financial data for testing
- `run_findel.sh`: Run a complete integration test of the entire pipeline

## Testing the entire pipeline

Simply run `run_findel.sh` to test the system as a whole, it:

1. Installs the package in development mode
2. Generates synthetic financial data
3. Trains a Transformer model on the synthetic data
4. Evaluates the Transformer model
5. Trains a GRU model on the synthetic data
6. Evaluates the GRU model
7. (Optional) Downloads real financial data
8. (Optional) Trains a MultiTask model on the real data
9. (Optional) Evaluates the MultiTask model on the real data

## Running individual scripts

### Data generation

```bash
# generate synthetic data
python scripts/generate_data.py --output_path data/processed/test_data.csv --data_type synthetic --n_samples 1000

# download real data
python scripts/generate_data.py --output_path data/processed/real_data.csv --data_type real --ticker SPY --start_date 2020-01-01 --end_date 2023-01-01
```

### Training a model

```bash
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
    --output_dir ./output \
    --model_name test_model
```

### Evaluating a model

```bash
python scripts/evaluate_model.py \
    --data_path data/processed/test_data.csv \
    --target_column returns \
    --sequence_length 60 \
    --model_type transformer \
    --hidden_dim 64 \
    --num_layers 2 \
    --model_path ./output/test_model.pth \
    --output_dir ./output
```