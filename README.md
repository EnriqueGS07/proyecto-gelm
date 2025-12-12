# Generador de Código Terraform con LangChain y Hugging Face

Proyecto en Python que utiliza LangChain y modelos de Hugging Face para generar código Terraform a partir de descripciones en lenguaje natural. Incluye capacidad de entrenar modelos personalizados con tus propios datos.

## 📋 Tabla de Contenidos

1. [Instalación](#instalación)
2. [Uso Rápido](#uso-rápido)
3. [Entrenamiento del Modelo](#entrenamiento-del-modelo)
4. [Estructura del Proyecto](#estructura-del-proyecto)
5. [Solución de Problemas](#solución-de-problemas)

---

## 🚀 Instalación

### Opción 1: Instalación Automática (Recomendada)

```bash
python install_dependencies.py
```

Este script instala las dependencias en el orden correcto y maneja conflictos automáticamente.

### Opción 2: Instalación Manual

1. **Dependencias básicas:**

```bash
pip install python-dotenv sentencepiece "protobuf>=3.20.0,<5.0.0"
```

2. **Ecosistema Hugging Face:**

```bash
pip install "huggingface-hub>=0.16.0" "transformers>=4.30.0" "datasets>=2.12.0" "accelerate>=0.20.0"
```

3. **LangChain:**

```bash
pip install "langchain>=0.1.0" "langchain-community>=0.0.10" "langchain-core>=0.1.0"
```

4. **PyTorch (según tu sistema):**

```bash
# Para CPU
pip install torch

# Para GPU NVIDIA (CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Para GPU NVIDIA (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Verificar Instalación

```bash
python -c "from terraform_generator import TerraformGenerator; print('✓ Instalación correcta')"
```

---

## ⚡ Uso Rápido

### 1. Probar con Modelo Pre-entrenado

**⚠️ Importante:** El modelo pre-entrenado NO está especializado en Terraform y puede generar código de baja calidad o basura. Para mejores resultados, [entrena el modelo](#entrenamiento-del-modelo).

#### Modo Interactivo:

```bash
python main.py
```

#### Desde línea de comandos:

```bash
python generate_terraform.py --prompt "Crear un bucket S3 con versionado habilitado"
```

#### Desde Python:

```python
from terraform_generator import TerraformGenerator

generator = TerraformGenerator()
codigo = generator.generate("Crear un bucket S3")
print(codigo)
```

### 2. Usar Modelo Entrenado (Recomendado)

```bash
python main.py --model_path models/terraform_generator
```

---

## 🎓 Entrenamiento del Modelo

### ¿Por qué entrenar?

| Aspecto         | Modelo Pre-entrenado | Modelo Entrenado          |
| --------------- | -------------------- | ------------------------- |
| Calidad         | ⚠️ Variable/Basura   | ✅ Código válido          |
| Especialización | ❌ No                | ✅ Sí, en Terraform       |
| Listo para usar | ✅ Inmediato         | ⏱️ Requiere entrenamiento |

**El modelo pre-entrenado puede generar código basura.** Para obtener código Terraform de calidad, **debes entrenar el modelo** con tus datos.

### Paso 1: Preparar Datos de Entrenamiento

Los datos deben estar en `training_data/` con formato JSON:

```json
[
  {
    "description": "Crear un bucket S3 con versionado habilitado",
    "terraform_code": "resource \"aws_s3_bucket\" \"example\" {\n  bucket = \"my-bucket\"\n  versioning {\n    enabled = true\n  }\n}"
  }
]
```

**Ya tienes ejemplos en:** `training_data/example_training_data.json`

### Paso 2: Entrenar el Modelo

```bash
python train_model.py --data_dir training_data/ --output_dir models/terraform_generator
```

**Opciones avanzadas:**

```bash
python train_model.py \
  --data_dir training_data/ \
  --output_dir models/terraform_generator \
  --epochs 5 \
  --batch_size 4 \
  --learning_rate 5e-5
```

**Tiempo estimado:**

- 5-10 ejemplos: 15-30 minutos
- 50+ ejemplos: 1-3 horas o más
- Requiere CPU o GPU

### Paso 3: Usar el Modelo Entrenado

```bash
# Modo interactivo
python main.py --model_path models/terraform_generator

# Desde línea de comandos
python generate_terraform.py \
  --prompt "Crear una instancia EC2" \
  --model_path models/terraform_generator \
  --output mi_codigo.tf
```

### Verificar si Tienes Modelo Entrenado

```bash
dir models\terraform_generator
```

Si la carpeta está vacía o no existe, no tienes un modelo entrenado aún.

---

## 📁 Estructura del Proyecto

```
.
├── main.py                      # Script interactivo principal
├── generate_terraform.py        # Generación desde línea de comandos
├── train_model.py              # Script de entrenamiento
├── terraform_generator.py      # Módulo principal con la lógica
├── install_dependencies.py     # Instalación automática
├── requirements.txt            # Dependencias
├── training_data/             # Datos de entrenamiento
│   └── example_training_data.json
└── models/                     # Modelos entrenados (se crea al entrenar)
```

---

## 🔧 Solución de Problemas

### Error: "Código basura generado"

**Problema:** El modelo pre-entrenado genera código inválido como:

```
HAIL: dongeran la colonlas s3 en terraform """ # TODO: Fix this...
```

**Solución:** Esto es normal con modelos pre-entrenados. **Entrena el modelo** con tus datos:

```bash
python train_model.py --data_dir training_data/ --output_dir models/terraform_generator
```

### Error: "Conflictos de dependencias"

**Solución:** Usa el script de instalación automática:

```bash
python install_dependencies.py
```

O instala manualmente en el orden correcto (ver [Instalación](#instalación)).

### Error: "Modelo no encontrado"

- Los modelos se descargan automáticamente la primera vez
- Verifica tu conexión a internet
- Algunos modelos pueden requerir autenticación en Hugging Face

### Error: "Out of memory"

- Usa un modelo más pequeño: `--model_name gpt2`
- Reduce batch_size: `--batch_size 2`
- Cierra otras aplicaciones

### Generación lenta

- El primer uso es más lento (descarga del modelo)
- Considera usar GPU si está disponible
- Los modelos entrenados son más rápidos

### Warnings sobre bitsandbytes/torch

Son normales si no tienes GPU. No afectan la funcionalidad, el proyecto funciona en CPU.

---

## 📝 Ejemplos Prácticos

### Generar y Guardar Código

```bash
python generate_terraform.py \
  --prompt "Crear un bucket S3 con encriptación" \
  --output s3_bucket.tf
```

### Generar con Contexto

```bash
python generate_terraform.py \
  --prompt "Crear un grupo de seguridad" \
  --context "Para una instancia EC2 en la VPC vpc-12345"
```

---

## 🎯 Modelos Soportados

Por defecto se usa:

- `microsoft/CodeGPT-small-py` (pre-entrenado)
- `gpt2` (para pruebas rápidas)

Puedes especificar otros modelos:

```bash
python main.py --model_name "Salesforce/codegen-350M-mono"
```

---

## ⚙️ Configuración Avanzada

### Variables de Entorno

Crear archivo `.env`:

```
HUGGINGFACE_API_TOKEN=tu_token_aqui
```

### Personalizar Parámetros

En `terraform_generator.py` puedes ajustar:

- `temperature`: Controla la creatividad (0.1-1.0)
- `max_length`: Longitud máxima de generación
- `device`: CPU o CUDA

---

## 📚 Próximos Pasos

1. ✅ **Probar** con modelo pre-entrenado: `python main.py`
2. 📊 **Agregar más datos** de entrenamiento en `training_data/`
3. 🎓 **Entrenar el modelo**: `python train_model.py --data_dir training_data/ --output_dir models/terraform_generator`
4. 🚀 **Usar modelo entrenado**: `python main.py --model_path models/terraform_generator`

---

## ⚠️ Notas Importantes

- **Primera ejecución:** Los modelos se descargan automáticamente (puede tomar varios minutos)
- **Recursos:** El entrenamiento requiere recursos computacionales significativos
- **GPU:** Opcional pero recomendado para entrenamiento y modelos grandes
- **Datos:** Mientras más ejemplos de entrenamiento tengas, mejor será el modelo
- **Calidad:** El modelo pre-entrenado nunca será tan bueno como uno entrenado específicamente para Terraform

---

## 📄 Licencia

Este proyecto es de código abierto. Siéntete libre de usarlo y modificarlo según tus necesidades.

---

**¿Necesitas ayuda?** Revisa la sección [Solución de Problemas](#solución-de-problemas) o consulta los comentarios en el código.
