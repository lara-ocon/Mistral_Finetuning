import json
from sklearn.model_selection import train_test_split

# Load the JSON data
json_file_path = 'diseases.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Split the data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save the training and test data to JSON files
train_json_file_path = 'train_data.json'
test_json_file_path = 'test_data.json'
with open(train_json_file_path, 'w') as file:
    json.dump(train_data, file)
with open(test_json_file_path, 'w') as file:
    json.dump(test_data, file)

def create_prompt(sample):
    bos_token = "<s>"
    system_message = sample['Instruction']
    input = sample['Input']
    response = sample['Response']
    eos_token = "</s>"

    full_prompt = ""
    full_prompt += bos_token
    full_prompt += "\n### Instruction:"
    full_prompt += "\n" + system_message
    full_prompt += "\n\n### Input:"
    full_prompt += "\n" + input
    full_prompt += "\n\n### Response:"
    full_prompt += "\n" + response
    full_prompt += eos_token

    return full_prompt

print('ejemplo de prompt: ')
print(create_prompt(train_data[0]))


print('\nProcedemos a cargar el modelo de MIstral 7B normal')

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    device_map='auto',
    quantization_config=nf4_config,
    use_cache=False
)

print('modelo cargado')
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print('tokenizer creado')


def generate_response(prompt, model):
  encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
  model_inputs = encoded_input.to('cuda')

  generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)

  decoded_output = tokenizer.batch_decode(generated_ids)

  return decoded_output[0].replace(prompt, "")



prompt="### Instruction:\nUse the provided input to create an instruction that could have been used to generate the response with an LLM.### Input:\nThere are more than 12,000 species of grass. The most common is Kentucky Bluegrass, because it grows quickly, easily, and is soft to the touch. Rygrass is shiny and bright green colored. Fescues are dark green and shiny. Bermuda grass is harder but can grow in drier soil.\n\n### Response:"
print(generate_response(prompt, model))


print('Comenzamos a entrenar: ')

from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)
# we need to prepare the model to be trained in 4bit so we 
# will use the prepare_model_for_kbit_training function 
# from peft
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

print('modelo preparado')

from transformers import TrainingArguments

args = TrainingArguments(
  output_dir = "mistral_instruct_generation",
  #num_train_epochs=5,
  max_steps = 100, # comment out this line if you want to train in epochs
  per_device_train_batch_size = 4,
  warmup_steps = 0.03,
  logging_steps=10,
  save_strategy="epoch",
  #evaluation_strategy="epoch",
  evaluation_strategy="steps",
  eval_steps=20, # comment out this line if you want to evaluate at the end of each epoch
  learning_rate=2e-4,
  bf16=True,
  lr_scheduler_type='constant',
)

print('training arguments creadossss')

from trl import SFTTrainer

max_seq_length = 2048

trainer = SFTTrainer(
  model=model,
  peft_config=peft_config,
  max_seq_length=max_seq_length,
  tokenizer=tokenizer,
  packing=True,
  formatting_func=create_prompt, # this will aplly the create_prompt mapping to all training and test dataset
  args=args,
  train_dataset=train_data,
  eval_dataset=test_data
)


print('trainer creadoooo')

print('comenzamos a entrenar')

trainer.train()

print('guardamos el modelo')

trainer.save_model("mistral_instruct_generation")

print('probamos el modelo')
merged_model = model.merge_and_unload()

def generate_response(prompt, model):
  encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
  model_inputs = encoded_input.to('cuda')

  generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)

  decoded_output = tokenizer.batch_decode(generated_ids)

  return decoded_output[0]


print(generate_response("### Instruction:\nUse the provided input to create an instruction that could have been used to generate the response with an LLM.### Input:\nThe first thing to know is that guacamole is a popular dip made from avocados, tomatoes, onions, and spices. It originated in Mexico, and is generally eaten as an appetizer or snack. Here are some simple steps: Choose 2 ripe avocados, about 2 cups Mash avocados in a large bowl using a fork or potato masher Add in 1-2 chopped tomatoes, salt, pepper, 1-2 garlic cloves, minced, 1-2 teaspoons fresh lime juice, and 1⁄4-1⁄2 cup chopped cilantro (optional). Let sit for about 10 minutes Taste, and add more salt, pepper, cilantro, or lime juice if needed Guacamole is usually served with tortilla chips. There are many variations, such as adding sour cream, diced vegetables, or more spicy hot peppers.\n\n### Response:", merged_model))
