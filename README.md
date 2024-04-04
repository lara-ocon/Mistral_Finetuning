conda create --name finetuning_env python=3.8
conda activate finetuning_env

pip install -r requirements.txt


python train_model.py
