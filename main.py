import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from simple_ance.ann_index import encode_corpus
from simple_ance.evaluate import evaluate
from simple_ance.model import SimpleANCE
from simple_ance.train import train_with_hard_negatives


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("train_ance.log")],
    )


def drop_insufficient_data(df):
    id_df = df[["query_id", "exact"]]
    id_df.loc[:, ["total"]] = 1
    id_df = id_df.groupby("query_id").sum().reset_index()
    id_df = id_df[id_df.exact > 0]
    return pd.merge(id_df[["query_id"]], df, how="left", on="query_id")


def load_data():
    product_df = pd.read_parquet("downloads/shopping_queries_dataset_products.parquet")
    example_df = pd.read_parquet("downloads/shopping_queries_dataset_examples.parquet")
    df = pd.merge(
        example_df[["example_id", "query_id", "product_id", "query", "esci_label", "split"]],
        product_df[["product_id", "product_title"]],
        how="left",
        on="product_id",
    )[["example_id", "query_id", "product_id", "query", "product_title", "esci_label", "split"]]
    df["exact"] = df.esci_label.apply(lambda x: 1 if x == "E" else 0)
    train_df = drop_insufficient_data(
        df[df.split == "train"][["example_id", "query_id", "product_id", "query", "product_title", "exact"]]
    )
    test_df = drop_insufficient_data(
        df[df.split == "test"][["example_id", "query_id", "product_id", "query", "product_title", "exact"]]
    )
    return train_df, test_df


class QueryDocumentDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        """
        クエリとポジティブドキュメントのペアのデータセット。

        Args:
            df (DataFrame): クエリとドキュメント情報を含むDataFrame。
            tokenizer (AutoTokenizer): クエリとドキュメントをエンコードするためのトークナイザ。
            max_length (int): トークナイズの最大長。
            size (int): サブセットのサイズ（>0の場合、データセットサイズを制限）。
        """
        self.df = df[df["exact"] == 1]  # ポジティブサンプルのみを使用
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        query = row["query"]
        positive_doc = row["product_title"]
        positive_doc_id = row["product_id"]

        query_encoding = self.tokenizer(
            query, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        positive_encoding = self.tokenizer(
            positive_doc, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )

        return {
            "query_input_ids": query_encoding["input_ids"].squeeze(0),
            "query_attention_mask": query_encoding["attention_mask"].squeeze(0),
            "positive_input_ids": positive_encoding["input_ids"].squeeze(0),
            "positive_attention_mask": positive_encoding["attention_mask"].squeeze(0),
            "positive_doc_id": positive_doc_id,
        }


class CorpusDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        """
        コーパスドキュメントのデータセット。

        Args:
            df (DataFrame): ドキュメント情報を含むDataFrame。
            tokenizer (AutoTokenizer): ドキュメントをエンコードするためのトークナイザ。
            max_length (int): トークナイズの最大長。
        """
        self.df = df.drop_duplicates(subset="product_id")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        doc_text = row["product_title"]
        doc_id = row["product_id"]

        encoding = self.tokenizer(
            doc_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "doc_id": doc_id,
        }


def save_model(logger, model, optimizer=None, save_directory="ance_model"):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    model.model.save_pretrained(save_directory)
    model.tokenizer.save_pretrained(save_directory)

    if optimizer:
        optimizer_path = os.path.join(save_directory, "optimizer.pt")
        torch.save(optimizer.state_dict(), optimizer_path)

    logger.info(f"Model and optimizer saved to {save_directory}")


def load_model(save_directory="ance_model"):
    model = SimpleANCE(model_name=save_directory)
    print(f"Model loaded from {save_directory}")
    return model


def train(logger, train_df, test_df=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Initializing tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = SimpleANCE(model_name="bert-base-uncased").to(device)

    logger.info("Preparing dataset and dataloader...")
    train_dataset = QueryDocumentDataset(train_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # コーパスデータセットの準備
    corpus_dataset = CorpusDataset(train_df, tokenizer)
    corpus_loader = DataLoader(corpus_dataset, batch_size=32, shuffle=False)

    # コーパスをエンコードして初期のANNインデックスを構築
    logger.info("Encoding corpus to build initial ANN index...")
    corpus_embeddings, doc_ids = encode_corpus(model, corpus_loader, device=device)
    corpus_embeddings = np.array(corpus_embeddings).astype("float32")  # エンベディングをfloat32に変換

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    logger.info("Starting training with hard negatives from ANN index...")
    train_with_hard_negatives(
        model, train_loader, corpus_loader, corpus_embeddings, doc_ids, optimizer, num_epochs=2, device=device
    )
    logger.info("Training completed.")

    save_model(logger, model, optimizer)

    if test_df is not None:
        test_dataset = QueryDocumentDataset(test_df, tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        logger.info("Evaluating the model on the test set...")
        evaluate(model, test_loader, device=device)


if __name__ == "__main__":
    setup_logger()
    logger = logging.getLogger(__name__)

    logger.info("Loading data from Amazon ESCI dataset...")
    train_df, test_df = load_data()
    # train_df = train_df.head(10000)
    # test_df = test_df.head(1000)
    logger.info(f"Train data: {len(train_df)}, Test data: {len(test_df)}")

    logger.info("Starting ANCE training with Amazon ESCI dataset...")
    train(logger, train_df, test_df)
