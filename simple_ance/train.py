import logging

import faiss
import numpy as np
import torch
import torch.nn.functional as F

from simple_ance.ann_index import build_ann_index, encode_corpus


def ance_ranking_loss(query_embeddings, positive_embeddings, negative_embeddings):
    """
    ハードネガティブを使用してANCEのランキング損失を計算します。

    Args:
        query_embeddings (Tensor): クエリエンベディング。 (batch_size, dim)
        positive_embeddings (Tensor): ポジティブドキュメントのエンベディング。 (batch_size, dim)
        negative_embeddings (Tensor): ネガティブドキュメントのエンベディング。 (batch_size, num_negatives, dim)

    Returns:
        Tensor: ランキング損失。
    """
    # ポジティブエンベディングを (batch_size, 1, dim) に変形
    positive_embeddings = positive_embeddings.unsqueeze(1)  # (batch_size, 1, dim)

    # ポジティブとネガティブのエンベディングを結合
    all_embeddings = torch.cat([positive_embeddings, negative_embeddings], dim=1)  # (batch_size, num_negatives+1, dim)

    # クエリエンベディングを (batch_size, 1, dim) に変形
    query_embeddings = query_embeddings.unsqueeze(1)  # (batch_size, 1, dim)

    # 類似度スコアを計算
    logits = torch.bmm(all_embeddings, query_embeddings.transpose(1, 2)).squeeze(2)  # (batch_size, num_negatives+1)

    # ラベル：ポジティブドキュメントは位置0
    labels = torch.zeros(query_embeddings.size(0), dtype=torch.long).to(query_embeddings.device)
    loss = F.cross_entropy(logits, labels)
    return loss


def train_with_hard_negatives(
    model,
    train_loader,
    corpus_loader,
    corpus_embeddings,
    doc_ids,
    optimizer,
    num_epochs=3,
    device="cpu",
    index_refresh_interval=1000,
    num_negatives=4,
):
    """
    ANNインデックスからのハードネガティブを使用してANCEモデルをトレーニングします。

    Args:
        model (SimpleANCE): ANCEモデル。
        train_loader (DataLoader): クエリとポジティブドキュメントのデータローダー。
        corpus_embeddings (np.ndarray): 事前に計算されたコーパスのエンベディング。
        doc_ids (list): コーパスのエンベディングに対応するドキュメントID。
        optimizer (torch.optim.Optimizer): トレーニング用のオプティマイザ。
        num_epochs (int): エポック数。
        device (torch.device): トレーニングに使用するデバイス。
        index_refresh_interval (int): ANNインデックスを更新する間隔（ステップ数）。
        num_negatives (int): サンプリングするハードネガティブの数。

    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    model.train()

    dimension = corpus_embeddings.shape[1]
    index = build_ann_index(corpus_embeddings, dimension)

    step = 0
    for epoch in range(num_epochs):
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            query_input_ids = batch["query_input_ids"].to(device)
            query_attention_mask = batch["query_attention_mask"].to(device)
            positive_input_ids = batch["positive_input_ids"].to(device)
            positive_attention_mask = batch["positive_attention_mask"].to(device)
            positive_doc_ids = [str(doc_id) for doc_id in batch["positive_doc_id"]]

            # クエリとポジティブドキュメントをエンコード
            query_embeddings = model(query_input_ids, query_attention_mask)  # (batch_size, dim)
            positive_embeddings = model(positive_input_ids, positive_attention_mask)  # (batch_size, dim)

            # クエリエンベディングを NumPy 配列に変換（計算グラフから切り離す）
            query_embeddings_np = query_embeddings.cpu().detach().numpy()

            # ANNインデックスからハードネガティブをサンプリング
            _, hard_negative_indices = index.search(query_embeddings_np, num_negatives + 1)
            hard_negative_embeddings = []

            for i, indices in enumerate(hard_negative_indices):
                # ポジティブドキュメントを除外
                pos_doc_id = positive_doc_ids[i]
                indices = [idx for idx in indices if doc_ids[idx] != pos_doc_id][:num_negatives]
                if len(indices) < num_negatives:
                    # ネガティブが不足している場合はランダムに補完
                    additional_idx = np.random.choice(len(corpus_embeddings))
                    if doc_ids[additional_idx] != pos_doc_id:
                        indices.append(additional_idx)
                negatives = torch.tensor(corpus_embeddings[indices]).to(device)
                hard_negative_embeddings.append(negatives)

            hard_negative_embeddings = torch.stack(hard_negative_embeddings)  # (batch_size, num_negatives, dim)

            loss = ance_ranking_loss(query_embeddings, positive_embeddings, hard_negative_embeddings)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1

            # 定期的にANNインデックスを更新
            if step % index_refresh_interval == 0:
                logger.info("Refreshing ANN index...")
                corpus_embeddings, _ = encode_corpus(model, corpus_loader, device=device)
                corpus_embeddings = np.array(corpus_embeddings).astype("float32")
                index = build_ann_index(corpus_embeddings, dimension)

            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item()}")

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")
