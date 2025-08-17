# OLA: REGRESSION ONLINE ALGORITHMS FROM SOLMA

This repository, **OLA (Online Learning Algorithms)**, contains Python implementations of online regression algorithms inspired by the **SOLMA** library. These tools are specifically designed for efficient processing of sequential data, enabling real-time predictions and adaptive learning in dynamic environments.

The online regression algorithms within this library have been specifically implemented and validated on datasets from the **Energy Technology Institute (ETI)**, showcasing their practical utility in real-world energy-related forecasting applications. Additionally, the `ralgo` folder within this repository contains implementations tested with various datasets sourced from the **UCI Machine Learning Repository**. A dedicated study on the performance of a key online learning algorithm under various delayed and batched feedback protocols is also available as `delayed_feedback.py` directly within the `onlinereg` folder.

---

## Getting Started

To get started with the implementations, follow these general steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/w-jamil/ola.git
    cd ola
    ```
2.  **Navigate to the specific algorithm's directory** (e.g., `onlinereg`, or `onlinereg/ralgo`).
3.  **Install common and specific dependencies** as listed below.

### General Dependencies

All implementations require `Python 3.x` and core libraries like `numpy` and `argparse`.
For visualization and metrics, `matplotlib.pyplot` and `sklearn.metrics` are also frequently used.

You can install these common dependencies using pip:
```bash
pip install numpy argparse matplotlib scikit-learn pandas scipy seaborn
```

---

## Algorithms Overview

Each algorithm is encapsulated within its own subdirectory. Detailed instructions and descriptions are provided below.

### 1. REG: Regression Learning

This module offers implementations for online regression, focusing on algorithms that excel in sequential prediction tasks. These methods are designed to perform comparably to an optimal forecaster in hindsight, making them highly suitable for dynamic environments where data arrives continuously.

-   **Description:** Online regression algorithms designed to perform not much worse than the true forecaster in hindsight. These algorithms have been specifically implemented and tested on data from the Energy Technology Institute (ETI). The `ralgo` folder contains additional algorithm implementations alongside datasets sourced from the UCI Machine Learning Repository. Furthermore, the `delayed_feedback.py` script within `onlinereg` features studies on the behavior and performance of a best-performing online learning algorithm under various delayed and batched feedback scenarios.

-   **How to Run:**
    1.  Navigate to the `onlinereg` directory:
        ```bash
        cd onlinereg
        ```
    2.  Execute the desired script (replace `algoname.py` with the actual script name, e.g., `coirr.py` or `delayed_feedback.py`):
        ```bash
        python algoname.py
        # Example for the delayed feedback study:
        # python delayed_feedback.py
        ```
    3.  **Customizing Data Input:** The scripts are designed to be easily adaptable for custom datasets. Ensure your data file adheres to the `N x D` array format (N observations, D dimensions), where the last column represents the target label `y`.
        ```python
        # Lines to review and potentially modify in your chosen regression script (e.g., coirr.py)
        array = np.loadtxt(args.data_file, delimiter=' ') # Adjust delimiter if needed
        n = len(array[:-1]) # D dimensions
        model = COIRR(args.tuning_parameter, n) # Or other specific algorithm class
        for i, a in enumerate(array):
          x, y = a[:-1], a[-1] # x are features, y is label
          new_y = model.predict(x)
        ```

-   **Dependencies:**
    -   `numpy`
    -   `argparse`
    -   `matplotlib.pyplot` (for visualization)
    -   `sklearn.metrics` (for performance evaluation)
    -   `pandas` (for data processing, particularly in `delayed_feedback.py`)

-   **Inputs:**
    -   The original dataset as an array with size `N x D`, where `D` is the number of dimensions in the original space and `N` is the number of observations. The last column is assumed to be the label. This includes datasets from the UCI Machine Learning Repository found in the `ralgo` folder.
    -   Hyperparameters (optional) can be initialized via `argparse`.

-   **Outputs:**
    -   Predicted label for each observation.

---

## Contributing

Contributions are welcome! If you encounter any bugs, have feature requests, or wish to contribute code, please feel free to open an issue or submit a pull request on the GitHub repository.

