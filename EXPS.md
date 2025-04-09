# Scaling Experiments

## Overall

The goal of this paper and regime of experiments is to determine the scaling laws for ViT models for weather prediction. An analogous paper to this would be "Training Compute-Optimal Large Language Models" from Google, though we are trying to investigate whether these same laws hold true for ViT models in the specific domain of weather prediction. To do this, we will be measuring FLOPs across all runs while evaluating various hyperparameters detailed below.

## Specifics

### 1. Model Scaling Experiments

- **Parameter Count Scaling**: Increase the number of transformer layers and attention heads systematically (e.g., 2x, 4x, 8x). Measure performance vs. computational cost to identify scaling patterns and diminishing returns.
- **Embedding Size Scaling**: Experiment with different embedding sizes, as this directly impacts memory usage and computational overhead. Measure quality vs. memory usage and latency.
- **Sequence Length Scaling**: Increase input sequence lengths (e.g., days, weeks, months) and measure performance and inference time. This explores temporal resolution's impact on scaling.
- **Multi-Resolution Scaling**: Test performance with weather data at different spatial resolutions (e.g., regional vs. global). Measure quality and computational needs.
- **Note**: Hold architecture constant across these experiments.

### 2. Data Scaling Experiments

- **Dataset Size Scaling**: Use subsets of data (e.g., 10%, 50%, 100%) to assess performance and compute needs. This will help determine the model's dependence on large datasets.
- **Temporal Coverage Scaling**: Vary historical coverage (e.g., 1 year, 5 years, 10 years) to study its impact on predictive quality and scaling efficiency.

### 3. Compute Resource Scaling Experiments

- **GPU Count Scaling**: Run the model on varying numbers of GPUs (e.g., 4, 8, 16 GPUs) to study parallelization efficiency. Track throughput, GPU utilization, and scaling efficiency.
- **Memory Optimization Experiments**: Test with and without memory-efficient operations like FlashAttention, gradient checkpointing, and mixed precision. This helps determine memory optimizations for large models on HPC.
- **Node Scaling Efficiency**: Increase the number of HPC nodes incrementally to identify optimal scaling and assess inter-node communication costs and throughput bottlenecks.
- **Memory Bandwidth Utilization**: Measure and analyze memory bandwidth usage across different model configurations.
- **Inter-node Communication Patterns**: Study the communication patterns between nodes during distributed training.
- **I/O Scaling and Bottlenecks**: Identify and address I/O bottlenecks as scale increases.
- **Storage Requirements**: Evaluate storage needs for checkpoints and intermediate results at different scales.
- **Note**: Hold architecture constant across these experiments.

### 4. Training & Inference Efficiency Experiments

- **Precision Scaling**: Run experiments with different precisions (FP32, FP16, BF16) to balance computational cost and model accuracy.
- **Batch Size Scaling**: Experiment with batch sizes across a range (small to large) to find the optimal balance for efficiency and quality.
- **Pipeline Parallelism**: Implement and test pipeline parallelism to understand how the model benefits from dividing layers across hardware for better latency and utilization.

### 5. Evaluation and Robustness Experiments

- **Prediction Quality vs. Complexity**: Plot prediction quality (e.g., MAE, MSE) against model complexity to establish the scaling relationship for optimal accuracy.
- **Overfitting/Underfitting Check**: Assess the model's generalization with varying sizes and complexity to establish robustness in predictions.