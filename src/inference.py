from unsloth import FastLanguageModel

# 저장된 모델 불러오기
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "koalpaca_lora_model",
    max_seq_length = 1024,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # 추론 모드 전환

# 대화 실행
question = "한국의 수도는 어디인가요?"
inputs = tokenizer(
    [alpaca_prompt.format(question, "")], 
    return_tensors = "pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 128)
response = tokenizer.batch_decode(outputs)[0]
print(response.split("### 응답:")[1].strip())