# cargamos el modelo al que hemos hecho finetuning
from transformers import AutoModelForCausalLM

model_path = "mistral_instruct_generation/checkpoint-24"
model = AutoModelForCausalLM.from_pretrained(model_path)
print('modelo cargado!\n')

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path)
print('tokenizer creado')
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))


def getResponse(p):
    print('I received question', p)
    inputs = tokenizer(p, return_tensors='pt', padding='longest',truncation=False )
    attention_mask = inputs['attention_mask']

    print('generating response')
    output_sequences = model.generate(
        input_ids = inputs["input_ids"],
        attention_mask=attention_mask, 
        pad_token_id =tokenizer.pad_token_id,
        max_new_tokens = 500
    )
    generated_responses = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    print('Answer ready:')
    print(generated_responses)
    print()
    print('Answer:', generated_responses[0], '\n')
    print('thats alllllll')
    return generated_responses[0]
