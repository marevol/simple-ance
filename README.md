# Simple ANCE

This project implements a simplified version of **ANCE (Approximate Nearest Neighbor Negative Contrastive Estimation)**, leveraging hard negative sampling from an Approximate Nearest Neighbor (ANN) index during training. The model is designed for efficient retrieval tasks using the **Amazon ESCI dataset**.

## Features

- **Training with Hard Negatives**: Train the ANCE model using hard negatives sampled from an ANN index built with FAISS.
- **ANN Indexing**: Utilize FAISS for efficient ANN indexing to retrieve hard negatives during training.
- **Efficient Retrieval**: Encode and index large document corpora for efficient information retrieval tasks.

## Installation

### Requirements

- Python 3.10+
- [Poetry](https://python-poetry.org/)
- PyTorch
- Transformers (Hugging Face)
- FAISS (Facebook AI Similarity Search)
- Pandas
- NumPy

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/marevol/simple-ance.git
   cd simple-ance
   ```

2. **Install dependencies using Poetry**:
   ```bash
   poetry install
   ```

   This will create a virtual environment and install all the necessary dependencies listed in `pyproject.toml`.

3. **Activate the virtual environment**:
   ```bash
   poetry shell
   ```

4. **Install FAISS**:

   - For CPU-only machines:
     ```bash
     pip install faiss-cpu
     ```
   - For GPU support (ensure compatibility with your CUDA version):
     ```bash
     pip install faiss-gpu
     ```

## Data Preparation

This project relies on the **Amazon ESCI dataset** for training the model. You need to download the dataset and place it in the correct directory.

1. **Download the dataset**:
   - Obtain the `shopping_queries_dataset_products.parquet` and `shopping_queries_dataset_examples.parquet` files from the [Amazon ESCI dataset](https://github.com/amazon-science/esci-data).

2. **Place the downloaded files** in the `downloads` directory within your project folder:
   ```bash
   ./downloads/shopping_queries_dataset_products.parquet
   ./downloads/shopping_queries_dataset_examples.parquet
   ```

3. **Verify data paths**:
   - The `main.py` script is set to load the dataset from the `downloads` directory by default. If you place the files elsewhere, modify the paths in the script accordingly.

## Usage

### Running the Training Script

The `main.py` script demonstrates how to use the **Amazon ESCI dataset** to train the ANCE model and evaluate its performance.

To run the training and evaluation:

```bash
poetry run python main.py
```

This script performs the following steps:

1. **Data Loading**: Loads product titles and queries from the Amazon ESCI dataset.
2. **Model Initialization**: Initializes the ANCE model using a pre-trained language model (e.g., `bert-base-uncased`).
3. **Corpus Encoding**: Encodes the document corpus to build the initial ANN index using FAISS.
4. **Training**: Trains the ANCE model using hard negatives sampled from the ANN index.
5. **Index Refreshing**: Periodically updates the ANN index during training to reflect the model's updated embeddings.
6. **Evaluation**: Evaluates the trained model on a test set and outputs performance metrics.

You can modify the script or dataset paths as needed.

### File Structure

- `main.py`: The main entry point for training and evaluating the ANCE model with the Amazon ESCI dataset.
- `simple_ance/model.py`: Defines the `SimpleANCE` model architecture.
- `simple_ance/train.py`: Handles the training process, including hard negative sampling and index refreshing.
- `simple_ance/ann_index.py`: Contains functions for building and updating the ANN index using FAISS.
- `simple_ance/evaluate.py`: Contains functions for evaluating the model's performance.
- `simple_ance/`: Other utility modules used in the project.

### Output

Upon completion of the script:

1. **Model Saving**: A trained model will be saved in the `ance_model` directory.
2. **Logging**: Training and evaluation logs, including loss and accuracy metrics, will be saved in `train_ance.log`.
3. **Console Output**: Key performance metrics and progress information will be printed to the console.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
