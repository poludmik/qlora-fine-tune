import torch
from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

merged_model_path = "merged_models"
# merged_model_path = "codellama/CodeLlama-7b-Instruct-hf"

print(f"Starting to load the model  from '{merged_model_path}/' into memory")

m = AutoModelForCausalLM.from_pretrained(
    merged_model_path,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map={"":0}
)

# for name, mod in m.named_modules(): 
#     print(name)

total_params = sum(p.numel() for p in m.parameters())
print(f'Total number of model parameters: {total_params}')



# tok = AutoTokenizer.from_pretrained(merged_model_path)
# # tok.bos_token_id = 1
# # stop_token_ids = [0]

# print(f"Successfully loaded the model {merged_model_path} into memory")

# # user = """Pandas dataframe 'df' is already initialized and filled with data.
# #             1. Select 5 maximal values from column 'GDP'.\n2. Multiple these values by 2.\n3. Print the resulting column.
# #             """

# user = "Pandas dataframe 'df' is already initialized and pandas is imported as pd.\n1. Find the second minimal value in column 'Temperature (C)'.\n2. Subtract 30.0 from this value.\n3. Print the result."

# # prompt = f"<s>[INST] {user.strip()} [/INST]"
# # inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")

# system = "Provide answers in Python"
# prompt = f"<s>[INST] <<SYS>>\\n{system}\\n<</SYS>>\\n\\n{user}[/INST]"
# inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")

# output = m.generate(input_ids=inputs.input_ids, max_new_tokens=200)

# output = output[0].to("cpu")
# print(tok.decode(output))