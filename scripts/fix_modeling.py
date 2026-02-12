import re

file_path = 'legato/models/modeling_legato.py'
with open(file_path, 'r') as f:
    content = f.read()

# 1. 将所有 self.model.vision_model 改为 self.vision_model
content = re.sub(r'self\.model\.vision_model', 'self.vision_model', content)

# 2. 替换 __init__ 为正确版本
correct_init = '''    def __init__(
        self,
        config : LegatoConfig,
        load_pretrained_encoder: bool = True
    ):
        super().__init__(config)
        encoder_ref = getattr(config, 'encoder_pretrained_model_name_or_path', None)
        if encoder_ref is not None:
            if load_pretrained_encoder:
                logger.info(f"Loading vision encoder from {encoder_ref}")
                self.vision_model = MllamaVisionModel.from_pretrained(encoder_ref)
                for param in self.vision_model.parameters():
                    param.requires_grad = False
            else:
                self.vision_model = None
        elif load_pretrained_encoder:
            raise ValueError(
                "The configuration does not specify 'encoder_pretrained_model_name_or_path'. "
                "Set load_pretrained_encoder to False to skip loading the encoder."
            )
        else:
            self.vision_model = None
'''

import re
pattern = r'def __init__\(.*?\):.*?(?=    @classmethod|\Z)'
content = re.sub(pattern, correct_init, content, flags=re.DOTALL)

# 3. 移除 from_pretrained 中的冲突检查
content = re.sub(
    r'if "load_pretrained_encoder" in kwargs:.*?raise ValueError\([^)]*\)',
    '',
    content,
    flags=re.DOTALL
)

with open(file_path, 'w') as f:
    f.write(content)

print("✅ modeling_legato.py 已完全修复")