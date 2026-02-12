import torch
from transformers import AutoTokenizer
import sys
sys.path.append('.')
from legato.models.modeling_legato import LegatoModel

# 加载模型和分词器
model_path = "../legato-model"
print("加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("加载模型...")
model = LegatoModel.from_pretrained(model_path, load_pretrained_encoder=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print("使用设备:", device)

# 准备输入（一个起始提示，ABC 格式开头）
prompt = "X:1\nT:Test\nM:4/4\nL:1/8\nK:C\n"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 生成
print("生成中...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=100,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

# 解码输出
generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n=== 生成结果 ===")
print(generated)
