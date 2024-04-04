1. Instalar visual studio code y miniconda
2. Crear un entorno usando miniconda:

conda create --name finetuning_env python=3.8
conda activate finetuning_env

3. Instalarse pytorch con el comando de la pagina:
https://pytorch.org/get-started/locally/

En nuestro caso:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

4. Instalar las dependencias:

pip install -r requirements.txt

5. Hacer finetuning con el comando:

python train.py

6. Se irán guardando checkpoints del modelo, podemos probarlos modificando el checkpoint en usemodel.py, y modificar las preguntas en ese mismo archivo. Para ejecutarlo:

python use_model.py
