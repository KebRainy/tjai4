from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", cache_dir=".", trust_remote_code=True)
# model = AutoModel.from_pretrained("THUDM/chatglm-6b", cache_dir=".", trust_remote_code=True).quantize(8).half().cuda()
model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4",trust_remote_code=True).float()
model = model.eval()

with open('questions.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

statements = [line.strip() for line in lines]

for q in statements:
    print("User >>> ", q)
    print()
    response, history = model.chat(tokenizer, q, history=[])
    print("Assistant >>> ")
    print(response)
    print()