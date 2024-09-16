import torch
from transformers import AutoModel, AutoTokenizer


class SimpleANCE(torch.nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(SimpleANCE, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # [CLS]トークンの出力を使用
        cls_output = outputs.last_hidden_state[:, 0, :]
        # エンベディングを正規化
        cls_output = torch.nn.functional.normalize(cls_output, p=2, dim=1)
        return cls_output
