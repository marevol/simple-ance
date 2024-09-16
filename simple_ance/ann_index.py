import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader


def build_ann_index(embeddings, dimension):
    """
    FAISSを使用してANNインデックスを構築します。

    Args:
        embeddings (np.ndarray): ドキュメントのエンベディング。
        dimension (int): エンベディングの次元数。

    Returns:
        faiss.IndexFlatL2: FAISSインデックス。
    """
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def encode_corpus(model, dataloader, device="cpu"):
    """
    モデルを使用してコーパス全体をエンコードします。
    モデル内でエンベディングは正規化されていることに注意。

    Args:
        model (SimpleANCE): エンコードに使用するモデル。
        dataloader (DataLoader): コーパスのデータローダー。
        device (torch.device): エンコードを実行するデバイス。

    Returns:
        np.ndarray: エンコードされたエンベディング。
        list: 対応するドキュメントID。
    """
    model.eval()
    embeddings = []
    doc_ids = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask)
            embeddings.append(outputs.cpu().numpy())
            doc_ids.extend([str(doc_id) for doc_id in batch["doc_id"]])

    embeddings = np.vstack(embeddings)
    return embeddings, doc_ids
