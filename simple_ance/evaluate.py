import logging

import torch
import torch.nn.functional as F


def evaluate(model, dataloader, device="cpu"):
    """
    ANCEモデルを評価データセットで評価します。

    Args:
        model (SimpleANCE): 評価するモデル。
        dataloader (DataLoader): テストサンプルを含むデータローダー。
        device (torch.device): 評価を実行するデバイス。

    Returns:
        float: 評価セットでの平均損失。
        float: ランキングに基づくモデルの精度。
    """
    logger = logging.getLogger(__name__)
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            query_input_ids = batch["query_input_ids"].to(device)
            query_attention_mask = batch["query_attention_mask"].to(device)
            positive_input_ids = batch["positive_input_ids"].to(device)
            positive_attention_mask = batch["positive_attention_mask"].to(device)

            query_embeddings = model(query_input_ids, query_attention_mask)
            positive_embeddings = model(positive_input_ids, positive_attention_mask)

            # エンベディングを正規化
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
            positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)

            # 類似度スコアを計算
            logits = torch.matmul(query_embeddings, positive_embeddings.t())  # (batch_size, batch_size)

            labels = torch.arange(logits.size(0)).to(query_embeddings.device)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    logger.info(f"Evaluation - Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy
