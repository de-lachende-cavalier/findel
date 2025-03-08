# findel

An exploration of finance-specific deep learning approaches inspired by methodologies from physics-based machine learning.

More specifically, we aim to develop novel neural network architectures and training methodologies that encode financial domain knowledge and constraints directly into the learning process, similar to how physics-based deep learning incorporates physical laws and constraints.

For our experiments, we use financial time series data and develop specialized models that respect the unique characteristics of financial markets, such as non-stationarity, regime changes, and market microstructure.

Key areas of focus include:

1. **Financial Inductive Biases**: Encoding financial domain knowledge directly into neural network architectures
2. **Market-Aware Loss Functions**: Developing specialized loss functions that capture financial risk metrics and market behavior
3. **Regime-Aware Models**: Creating models that can identify and adapt to different market regimes
4. **Regulatory and Constraint Satisfaction**: Building models that inherently respect financial regulations and constraints

## Running the code

The code uses `python 3.11`. To run it, do the following:

1. (Optional) Set up a virtual environment;
2. Install all the requirements (i.e., `pip install -r requirements.txt`);
3. Run the scripts in the `scripts/run_findel.sh`.

    3. (Note that this will run the entire pipeline, check the README in the `scripts/` directory if you want to run individual scripts)


## Project structure

```
findel/
├── data/                  # data storage and processing
├── models/                # finance-specific neural network models
├── losses/                # specialized financial loss functions
├── utils/                 # utility functions
├── experiments/           # experiment configurations
├── scripts/               # training and evaluation scripts
├── tests/                 # unit tests
```
