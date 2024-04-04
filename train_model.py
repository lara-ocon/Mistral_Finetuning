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
    train.to_csv('data/train.csv', index=False)

# Función principal para el entrenamiento
def main():
    prepare_dataset()

    # Configuración de entrenamiento
    model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    train_dataset = load_dataset('csv', data_files='data/train.csv', split='train')
    train_dataset = train_dataset.map(lambda examples: tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512), batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

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

    # Iniciar el entrenamiento
    trainer.train()

    # Guardar el modelo y el tokenizador
    model.save_pretrained('./my_trained_model')
    tokenizer.save_pretrained('./my_trained_model')

if __name__ == "__main__":
    main()
