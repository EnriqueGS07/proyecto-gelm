# Generador de Código Terraform

Proyecto en Python que utiliza LangChain y modelos de Hugging Face para generar código Terraform a partir de descripciones en lenguaje natural.

## Instalación

1. Crear entorno virtual (recomendado):

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

Si hay conflictos, instalar manualmente:

```bash
pip install python-dotenv sentencepiece "protobuf>=3.20.0,<5.0.0"
pip install "huggingface-hub>=0.16.0" "transformers>=4.30.0" "datasets>=2.12.0" "accelerate>=0.20.0"
pip install "langchain>=0.1.0" "langchain-community>=0.0.10" "langchain-core>=0.1.0"
pip install torch
```

## Uso

### Modo Interactivo

```bash
python main.py
```

Para usar modelo entrenado:

```bash
python main.py --model_path models/terraform_generator
```

### Desde Línea de Comandos

```bash
python generate_terraform.py --prompt "Crear un bucket S3"
```

Guardar en archivo:

```bash
python generate_terraform.py --prompt "Crear un bucket S3" --output s3.tf
```

### Desde Python

```python
from terraform_generator import TerraformGenerator

generator = TerraformGenerator()
generator.load_model()
codigo = generator.generate("Crear un bucket S3")
print(codigo)
```

## Entrenamiento

1. Preparar datos en `training_data/` con formato JSON:

```json
[
  {
    "description": "Crear un bucket S3",
    "terraform_code": "resource \"aws_s3_bucket\" \"example\" {\n  bucket = \"my-bucket\"\n}"
  }
]
```

2. Entrenar el modelo:

```bash
python train_model.py --data_dir training_data/ --output_dir models/terraform_generator
```

Opciones avanzadas:

```bash
python train_model.py \
  --data_dir training_data/ \
  --output_dir models/terraform_generator \
  --epochs 5 \
  --batch_size 4 \
  --learning_rate 5e-5
```

3. Usar modelo entrenado:

```bash
python main.py --model_path models/terraform_generator
```

## Notas

- El modelo pre-entrenado puede generar código de baja calidad. Se recomienda entrenar el modelo con datos específicos.
- El entrenamiento requiere tiempo y recursos computacionales (1-5 horas dependiendo del número de ejemplos).
- Los modelos se descargan automáticamente la primera vez que se usan.
