import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. 모델 및 토크나이저 설정 (Llama-3-8B 4bit 양자화 버전)
max_seq_length = 1024  # RTX 3060 12GB에 최적화된 길이
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)

# 2. LoRA 어댑터 설정 (학습할 파라미터 지정)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", # VRAM 절약 핵심 설정
)

# 3. KoAlpaca 데이터셋 포맷팅
alpaca_prompt = """아래는 작업을 설명하는 지시 사항입니다. 요청을 적절히 완료하는 응답을 작성하세요.

### 지시:
{}

### 응답:
{}"""

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs      = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        # 지시와 응답을 합치고 문장 끝에 EOS 토큰을 붙입니다.
        text = alpaca_prompt.format(instruction, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

# 데이터셋 로드 및 매핑
dataset = load_dataset("beomi/KoAlpaca-v1.1", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True)

# 4. 학습 인자 설정 (RTX 3060 12GB 최적화 세팅)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 2,     # VRAM 12GB 고려
        gradient_accumulation_steps = 4,     # 실제 배치 사이즈는 2*4=8 효과
        warmup_steps = 10,
        max_steps = 100,                     # 테스트용 (전체 학습시 num_train_epochs=1 권장)
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "paged_adamw_8bit",          # 메모리 부족 방지 옵티마이저
        weight_decay = 0.01,
        output_dir = "outputs",
    ),
)

# 5. 학습 시작
trainer.train()

# 6. 학습된 모델 저장
model.save_pretrained("koalpaca_lora_model")
tokenizer.save_pretrained("koalpaca_lora_model")