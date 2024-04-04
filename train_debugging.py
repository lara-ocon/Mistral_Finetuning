from datasets import load_dataset
import pandas as pd
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# Función para formatear el texto
def text_formatting(data):
    if data['input']:
        text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{data["instruction"]} \n\n### Input:\n{data["input"]}\n\n### Response:\n{data["output"]}"""
    else:
        text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{data["instruction"]}\n\n### Response:\n{data["output"]}""" 
    return text

# Cargar y preparar el dataset
def prepare_dataset():
    train = load_dataset("tatsu-lab/alpaca", split='train[:10%]')
    train = pd.DataFrame(train)
    train['text'] = train.apply(text_formatting, axis=1)
    train.to_csv('data/mini_train.csv', index=False)


# Añadido
def preprocess_function(examples, tokenizer):
    instructions = examples['instruction']
    outputs = examples['output']
    tokenized_inputs = tokenizer(instructions, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(outputs, padding='max_length', truncation=True, max_length=512, return_tensors="pt")["input_ids"]
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def calculate_loss_for_debugging(model, tokenizer, dataset):
    # Tomar una muestra del dataset para depuración
    sample = dataset.select(range(5))  # Ajusta el rango según sea necesario
    sample = sample.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
    sample.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Calcular la pérdida para un pequeño subconjunto
    for i in range(len(sample)):
        inputs = {k: sample[k][i].unsqueeze(0) for k in ['input_ids', 'attention_mask', 'labels']}
        outputs = model(**inputs)
        loss = outputs.loss
        print(f"Loss for example {i}: {loss.item()}")


# Función principal para el entrenamiento
def main():
    #prepare_dataset()
    print('Dataset prepared')
    # Configuración de entrenamiento
    model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print('tokenizer creado')
    print()
    # añadido:
    tokenizer.pad_token = tokenizer.eos_token

    print('tokenizer pad token añadido')
    print()
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print('model and tokenizer created')
    train_dataset = load_dataset('csv', data_files='data/mini_train.csv', split='train')
    print('dataset lodades')
    #train_dataset = train_dataset.map(lambda examples: tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512), batched=True)
    # modificado
    train_dataset = train_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    print('Calculating loss for debugging purposes...')
    calculate_loss_for_debugging(model, tokenizer, train_dataset)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=4,
        per_device_train_batch_size=1,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        learning_rate=2e-4,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    print('trainer created')
    # Iniciar el entrenamiento
    trainer.train()
    print('model trained')

    # Guardar el modelo y el tokenizador
    model.save_pretrained('./my_trained_model')
    tokenizer.save_pretrained('./my_trained_model')
    print('model and tokenizer saved')
if __name__ == "__main__":
    main()
