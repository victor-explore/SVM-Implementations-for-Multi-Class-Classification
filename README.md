# SVM Implementation for Multi-Class Classification

This repository contains implementations of Support Vector Machine (SVM) algorithms for both binary and multi-class classification problems. The implementations include variations with and without slack formulation, and explore different kernel functions.

## Implementations

### Binary Classification
1. SVM without slack formulation
2. SVM with slack formulation 
3. Different kernel implementations:
   - Linear kernel
   - Polynomial kernel

### Multi-Class Classification 
1. SVM without slack formulation
2. SVM with slack formulation
3. Kernel implementations:
   - Linear kernel
   - Polynomial kernel
   - Radial Basis Function (RBF) kernel

## Features

- Data preprocessing including normalization
- Train/test split functionality
- Multiple kernel implementations
- Support for both binary and multi-class classification
- Optimization using CVXPY and SciPy
- Accuracy evaluation metrics

## Dependencies

- NumPy
- CVXPY 
- SciPy
- Google Colab (optional, for notebook execution)

## Usage

The code is structured in sections:

```python
# Load and prepare data
data = np.loadtxt(file_path, delimiter='\t', skiprows=1)

# Split data into training and testing sets
split_ratio = 0.8
split_index = int(data.shape[0] * split_ratio)

# Normalize features
mean = np.mean(features, axis=0)
std = np.std(features, axis=0)
normalized_features = (features - mean) / std

# Train SVM models with different configurations
```

## Implementation Details

### Binary Classification
- Implements both slack and non-slack formulations
- Uses CVXPY for optimization
- Supports multiple kernel functions

### Multi-Class Classification
- One-vs-rest strategy
- Support for different kernel functions
- Includes slack variables for soft margin classification

### Kernel Functions
- Linear: `K(x,y) = x^T y`
- Polynomial: `K(x,y) = (γ<x,y> + coef0)^degree`
- RBF: `K(x,y) = exp(-γ||x-y||^2)`

## Performance

The implementation includes accuracy calculations for both training and test sets. Results can be evaluated using:

```python
accuracy = np.mean(predictions == test_labels)
print(f"Accuracy on the test data: {accuracy:.4f}")
```

## Contributing

Feel free to open issues or submit pull requests for improvements.
