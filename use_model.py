# cargamos el modelo al que hemos hecho finetuning
from transformers import AutoModelForCausalLM

model_path = "mistral_instruct_generation/checkpoint-72"
model = AutoModelForCausalLM.from_pretrained(model_path)
print('modelo cargado!\n')

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path)
print('tokenizer creado')

preguntas = ["I have gained weight and feel weak. I also have purple stretch marks, hair loss and hormone changes. What disease do I Have?"]
"""
preguntas = ["What are the different types of grass?",
            "How can I cook food while camping?",
            "What are some fun scenarios my kids can play with their Barbies?",
            "How many titles have Liverpool won?",
            "Was She Couldn't Say No movie released?",
            "What are the antidepressants I can consider?"]

            Signs and symptoms of adrenal cancer include: Weight gain Muscle weakness 
            Pink or purple stretch marks on the skin Hormone changes in women that might 
            cause excess facial hair, hair loss on the head and irregular periods Hormone 
            changes in men that might cause enlarged breast tissue and shrinking testicles 
            Nausea Vomiting Abdominal bloating Back pain Fever Loss of appetite Loss of weight without trying
"""
for p in preguntas:
    inputs = tokenizer(p, return_tensors='pt', padding='longest',truncation=False )
    attention_mask = inputs['attention_mask']
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    output_sequences = model.generate(
        input_ids = inputs["input_ids"],
        attention_mask=attention_mask, 
        pad_token_id =tokenizer.pad_token_id,
        max_new_tokens = 100
    )
    generated_responses = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    print(generated_responses[0])
    print()