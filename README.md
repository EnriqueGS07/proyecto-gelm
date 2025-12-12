# Generador de Código Terraform con LangChain y Hugging Face

Proyecto en Python que utiliza LangChain y modelos de Hugging Face para generar código Terraform válido a partir de descripciones en lenguaje natural. Incluye capacidad de entrenar modelos personalizados con tus propios datos.

## 📋 Tabla de Contenidos

1. [Características](#características)
2. [Instalación](#instalación)
3. [Uso Rápido](#uso-rápido)
4. [Entrenamiento del Modelo](#entrenamiento-del-modelo)
5. [Arquitectura y Código](#arquitectura-y-código)
6. [Estructura del Proyecto](#estructura-del-proyecto)
7. [Solución de Problemas](#solución-de-problemas)
8. [Ejemplos Prácticos](#ejemplos-prácticos)

---

## ✨ Características

- **Generación de código Terraform**: Convierte descripciones en lenguaje natural a código Terraform válido
- **Modelos pre-entrenados**: Funciona inmediatamente con modelos de Hugging Face
- **Fine-tuning personalizado**: Entrena modelos con tus propios datos de Terraform
- **Validación inteligente**: Filtra y valida código generado automáticamente
- **Interfaz flexible**: Modo interactivo, CLI y programático
- **Soporte multi-recurso**: S3, EC2, EBS, Internet Gateway, KMS, y más

---

## 🚀 Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Conexión a internet (para descargar modelos la primera vez)
- Opcional: GPU NVIDIA con CUDA (acelera entrenamiento y generación)

### Instalación Manual

1. **Clonar o descargar el proyecto**

2. **Crear entorno virtual (recomendado):**

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Instalar dependencias básicas:**

   ```bash
   pip install python-dotenv sentencepiece "protobuf>=3.20.0,<5.0.0"
   ```

4. **Instalar ecosistema Hugging Face:**

   ```bash
   pip install "huggingface-hub>=0.16.0" "transformers>=4.30.0" "datasets>=2.12.0" "accelerate>=0.20.0"
   ```

5. **Instalar LangChain:**

   ```bash
   pip install "langchain>=0.1.0" "langchain-community>=0.0.10" "langchain-core>=0.1.0"
   ```

6. **Instalar PyTorch (según tu sistema):**

   ```bash
   # Para CPU
   pip install torch

   # Para GPU NVIDIA (CUDA 11.8)
   pip install torch --index-url https://download.pytorch.org/whl/cu118

   # Para GPU NVIDIA (CUDA 12.1)
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

### Instalación Rápida (requirements.txt)

```bash
pip install -r requirements.txt
```

**Nota:** Si hay conflictos de dependencias, instala manualmente en el orden mostrado arriba.

### Verificar Instalación

```bash
python -c "from terraform_generator import TerraformGenerator; print('✓ Instalación correcta')"
```

---

## ⚡ Uso Rápido

### 1. Probar con Modelo Pre-entrenado

**⚠️ Importante:** El modelo pre-entrenado NO está especializado en Terraform y puede generar código de baja calidad o basura. Para mejores resultados, [entrena el modelo](#entrenamiento-del-modelo).

#### Modo Interactivo (Recomendado para principiantes):

```bash
python main.py
```

Luego ingresa descripciones como:

- "Crear un bucket S3 con versionado habilitado"
- "Crear una instancia EC2 tipo t2.micro"
- "Crear un volumen EBS de 20GB"

#### Desde línea de comandos:

```bash
python generate_terraform.py --prompt "Crear un bucket S3 con versionado habilitado"
```

#### Guardar código en archivo:

```bash
python generate_terraform.py \
  --prompt "Crear un bucket S3" \
  --output s3_bucket.tf
```

#### Desde Python (programático):

```python
from terraform_generator import TerraformGenerator

# Inicializar generador
generator = TerraformGenerator()

# Cargar modelo (se descarga automáticamente la primera vez)
generator.load_model()

# Generar código
codigo = generator.generate("Crear un bucket S3 con versionado")
print(codigo)
```

### 2. Usar Modelo Entrenado (Recomendado)

```bash
# Modo interactivo con modelo entrenado
python main.py --model_path models/terraform_generator

# Desde línea de comandos
python generate_terraform.py \
  --prompt "Crear una instancia EC2" \
  --model_path models/terraform_generator \
  --output ec2.tf
```

---

## 🎓 Entrenamiento del Modelo

### ¿Por qué entrenar?

| Aspecto         | Modelo Pre-entrenado | Modelo Entrenado          |
| --------------- | -------------------- | ------------------------- |
| Calidad         | ⚠️ Variable/Basura   | ✅ Código válido          |
| Especialización | ❌ No                | ✅ Sí, en Terraform       |
| Listo para usar | ✅ Inmediato         | ⏱️ Requiere entrenamiento |
| Tiempo          | 0 minutos            | 1-5 horas                 |

**El modelo pre-entrenado puede generar código basura.** Para obtener código Terraform de calidad, **debes entrenar el modelo** con tus datos.

### Paso 1: Preparar Datos de Entrenamiento

Los datos deben estar en `training_data/` con formato JSON:

```json
[
  {
    "description": "Crear un bucket S3 con versionado habilitado",
    "terraform_code": "resource \"aws_s3_bucket\" \"example\" {\n  bucket = \"my-bucket\"\n  versioning {\n    enabled = true\n  }\n}"
  },
  {
    "description": "Crear una instancia EC2 tipo t2.micro",
    "terraform_code": "resource \"aws_instance\" \"example\" {\n  ami           = \"ami-0c55b159cbfafe1f0\"\n  instance_type = \"t2.micro\"\n  tags = {\n    Name = \"example-instance\"\n  }\n}"
  }
]
```

**Ya tienes ejemplos en:**

- `training_data/example_training_data.json` - 5 ejemplos básicos
- `training_data/expanded_training_data.json` - 56 ejemplos adicionales
- `training_data/additional_examples.json` - 93 ejemplos más
- `training_data/internet_gateway_examples.json` - 3 ejemplos de Internet Gateway
- `training_data/kms_examples.json` - 6 ejemplos de KMS

**Total: 163 ejemplos listos para usar**

### Paso 2: Entrenar el Modelo

#### Entrenamiento básico:

```bash
python train_model.py --data_dir training_data/ --output_dir models/terraform_generator
```

#### Entrenamiento con opciones avanzadas:

```bash
python train_model.py \
  --data_dir training_data/ \
  --output_dir models/terraform_generator \
  --epochs 5 \
  --batch_size 4 \
  --learning_rate 5e-5
```

#### Parámetros de entrenamiento:

- `--data_dir`: Directorio con archivos de entrenamiento (default: `training_data`)
- `--output_dir`: Donde guardar el modelo entrenado (default: `models/terraform_generator`)
- `--epochs`: Número de épocas (default: 5, recomendado: 3-10)
- `--batch_size`: Tamaño del batch (default: 4, ajustar según RAM/GPU)
- `--learning_rate`: Tasa de aprendizaje (default: 5e-5, típico: 1e-5 a 5e-5)
- `--model_name`: Modelo base a usar (default: `microsoft/CodeGPT-small-py`)

**Tiempo estimado:**

- 5-10 ejemplos: 15-30 minutos (CPU)
- 50+ ejemplos: 1-3 horas (CPU) o 30-60 minutos (GPU)
- 100+ ejemplos: 3-5 horas (CPU) o 1-2 horas (GPU)

### Paso 3: Usar el Modelo Entrenado

```bash
# Modo interactivo
python main.py --model_path models/terraform_generator

# Desde línea de comandos
python generate_terraform.py \
  --prompt "Crear una instancia EC2" \
  --model_path models/terraform_generator \
  --output ec2.tf
```

### Verificar si Tienes Modelo Entrenado

```bash
# Windows
dir models\terraform_generator

# Linux/Mac
ls models/terraform_generator
```

Si la carpeta está vacía o no existe, no tienes un modelo entrenado aún.

---

## 🏗️ Arquitectura y Código

### Componentes Principales

#### 1. `TerraformGenerator` (terraform_generator.py)

Clase principal que encapsula toda la lógica de generación:

```python
class TerraformGenerator:
    """
    Clase principal para generar código Terraform usando modelos de Hugging Face.

    Métodos principales:
    - __init__(): Inicializa el generador con configuración
    - load_model(): Carga modelo pre-entrenado desde Hugging Face
    - load_from_local(): Carga modelo entrenado desde disco
    - train(): Entrena el modelo con datos personalizados
    - generate(): Genera código Terraform desde descripción
    - generate_with_context(): Genera código con contexto adicional
    """
```

**Flujo de generación:**

1. **Carga del modelo**: Descarga o carga desde caché/disco
2. **Formateo del prompt**: Añade ejemplos few-shot y la descripción del usuario
3. **Generación**: El modelo genera código usando el pipeline de Hugging Face
4. **Limpieza**: Extrae solo el código Terraform válido
5. **Validación**: Filtra código inválido y atributos incorrectos
6. **Retorno**: Devuelve código limpio y válido

**Sistema de validación:**

El método `generate()` incluye validación en múltiples niveles:

1. **Filtrado de patrones inválidos:**

   - Recursos inexistentes: `aws_ebs_instance`, `aws_s3_instance`
   - Atributos inválidos: `volume_name`, `instance_name`
   - Código basura: `TODO:`, `github.com`, `dongeran`
   - Markdown residual: ` ```terraform`, ` ``` `

2. **Validación por tipo de recurso:**

   - `aws_internet_gateway`: Solo permite `vpc_id`, `tags`
   - `aws_s3_bucket`: Solo permite `bucket`, `tags`, `versioning`, etc.
   - `aws_instance`: Solo permite `ami`, `instance_type`, `tags`, etc.
   - `aws_ebs_volume`: Solo permite `availability_zone`, `size`, `type`, etc.
   - `aws_kms_key`: Solo permite `description`, `deletion_window_in_days`, etc.

3. **Validación estructural:**
   - Verifica que haya un recurso válido
   - Verifica que las llaves `{}` estén balanceadas
   - Extrae solo el primer recurso válido encontrado

#### 2. Scripts de Interfaz

**`main.py`**: Interfaz interactiva

- Bucle continuo para múltiples generaciones
- Opción de guardar código en archivos
- Manejo de errores amigable

**`generate_terraform.py`**: Interfaz CLI

- Ideal para automatización y scripts
- Soporta guardado directo en archivos
- Permite contexto adicional

**`train_model.py`**: Script de entrenamiento

- Carga datos desde múltiples archivos JSON/TXT
- Configura y ejecuta el entrenamiento
- Guarda checkpoints y modelo final

### Flujo de Entrenamiento

```
1. Cargar datos de entrenamiento
   ↓
2. Formatear ejemplos con prompt template
   ↓
3. Tokenizar datos usando el tokenizador del modelo
   ↓
4. Configurar TrainingArguments
   ↓
5. Crear Trainer con modelo, datos y collator
   ↓
6. Ejecutar trainer.train()
   ↓
7. Guardar modelo y tokenizador
```

### Prompt Engineering

El prompt template incluye:

1. **Instrucciones claras**: "Genera código Terraform válido y completo para AWS"
2. **Ejemplos few-shot**: 3 ejemplos de S3, EC2 y EBS
3. **Advertencias**: Lista de recursos y atributos inválidos
4. **Formato**: Estructura consistente para el modelo

---

## 📁 Estructura del Proyecto

```
.
├── main.py                      # Script interactivo principal
├── generate_terraform.py        # Generación desde línea de comandos
├── train_model.py              # Script de entrenamiento
├── terraform_generator.py      # Módulo principal con la lógica
├── requirements.txt            # Dependencias del proyecto
├── README.md                   # Esta documentación
│
├── training_data/              # Datos de entrenamiento
│   ├── example_training_data.json      # 5 ejemplos básicos
│   ├── expanded_training_data.json     # 56 ejemplos adicionales
│   ├── additional_examples.json        # 93 ejemplos más
│   ├── internet_gateway_examples.json  # 3 ejemplos de Internet Gateway
│   └── kms_examples.json               # 6 ejemplos de KMS
│
└── models/                     # Modelos entrenados (se crea al entrenar)
    └── terraform_generator/    # Modelo entrenado
        ├── model.safetensors   # Pesos del modelo
        ├── config.json         # Configuración
        ├── tokenizer.json      # Tokenizador
        └── ...                 # Otros archivos necesarios
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

**Solución:** Instala manualmente en el orden correcto (ver [Instalación](#instalación)) o usa versiones compatibles:

```bash
pip install --upgrade pip
pip install "langchain>=0.1.0" "langchain-community>=0.0.10" "langchain-core>=0.1.0"
```

### Error: "Modelo no encontrado"

- Los modelos se descargan automáticamente la primera vez
- Verifica tu conexión a internet
- Algunos modelos pueden requerir autenticación en Hugging Face
- Verifica que el nombre del modelo sea correcto

### Error: "Out of memory"

- Usa un modelo más pequeño: `--model_name gpt2`
- Reduce batch_size: `--batch_size 2`
- Cierra otras aplicaciones
- Usa CPU si no tienes suficiente VRAM en GPU

### Generación lenta

- El primer uso es más lento (descarga del modelo)
- Considera usar GPU si está disponible
- Los modelos entrenados son más rápidos que los pre-entrenados
- Reduce `max_length` si no necesitas código muy largo

### Warnings sobre bitsandbytes/torch

Son normales si no tienes GPU. No afectan la funcionalidad, el proyecto funciona perfectamente en CPU.

### Error: "TrainingArguments.**init**() got an unexpected keyword argument 'evaluation_strategy'"

**Solución:** Actualiza transformers:

```bash
pip install --upgrade transformers
```

El código ya incluye un fallback para versiones antiguas, pero es mejor actualizar.

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

### Entrenar con Parámetros Personalizados

```bash
python train_model.py \
  --data_dir training_data/ \
  --output_dir models/terraform_generator \
  --epochs 10 \
  --batch_size 8 \
  --learning_rate 3e-5
```

### Uso Programático

```python
from terraform_generator import TerraformGenerator

# Inicializar
generator = TerraformGenerator(
    model_name="microsoft/CodeGPT-small-py",
    temperature=0.3,
    max_length=512
)

# Cargar modelo entrenado
generator.load_from_local("models/terraform_generator")

# Generar código
codigo = generator.generate("Crear un bucket S3 con versionado")

# Generar con contexto
codigo = generator.generate_with_context(
    "Crear un grupo de seguridad",
    "Para una instancia EC2 en la VPC vpc-12345"
)

print(codigo)
```

---

## 🎯 Modelos Soportados

Por defecto se usa:

- `microsoft/CodeGPT-small-py` (pre-entrenado, especializado en Python pero adaptable)
- `gpt2` (para pruebas rápidas, más pequeño)

Puedes especificar otros modelos:

```bash
python main.py --model_name "Salesforce/codegen-350M-mono"
python train_model.py --model_name "gpt2" --data_dir training_data/
```

**Recomendaciones:**

- Para entrenamiento: Usa modelos pequeños como `gpt2` o `microsoft/CodeGPT-small-py`
- Para producción: Entrena primero con tus datos, luego usa el modelo entrenado
- Para pruebas: `gpt2` es rápido pero menos preciso

---

## ⚙️ Configuración Avanzada

### Variables de Entorno

Crear archivo `.env`:

```
HUGGINGFACE_API_TOKEN=tu_token_aqui
```

### Personalizar Parámetros en Código

En `terraform_generator.py` puedes ajustar:

```python
generator = TerraformGenerator(
    model_name="microsoft/CodeGPT-small-py",
    device="auto",           # "cpu", "cuda", o "auto"
    max_length=512,           # Longitud máxima de secuencia
    temperature=0.7           # Creatividad (0.1-1.0)
)
```

### Ajustar Validación

En el método `generate()` de `terraform_generator.py`, puedes modificar:

- `invalid_patterns`: Lista de patrones a filtrar
- `resource_attr_validation`: Validación por tipo de recurso
- Lógica de extracción de código válido

---

## 📚 Próximos Pasos

1. ✅ **Probar** con modelo pre-entrenado: `python main.py`
2. 📊 **Agregar más datos** de entrenamiento en `training_data/`
3. 🎓 **Entrenar el modelo**: `python train_model.py --data_dir training_data/ --output_dir models/terraform_generator`
4. 🚀 **Usar modelo entrenado**: `python main.py --model_path models/terraform_generator`
5. 🔧 **Personalizar**: Ajusta parámetros según tus necesidades

---

## ⚠️ Notas Importantes

- **Primera ejecución:** Los modelos se descargan automáticamente (puede tomar varios minutos)
- **Recursos:** El entrenamiento requiere recursos computacionales significativos
- **GPU:** Opcional pero recomendado para entrenamiento y modelos grandes
- **Datos:** Mientras más ejemplos de entrenamiento tengas, mejor será el modelo
- **Calidad:** El modelo pre-entrenado nunca será tan bueno como uno entrenado específicamente para Terraform
- **Validación:** El sistema de validación es agresivo para asegurar código válido, pero puede rechazar código válido en casos edge

---

## 📄 Licencia

Este proyecto es de código abierto. Siéntete libre de usarlo y modificarlo según tus necesidades.

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Algunas ideas:

- Agregar más ejemplos de entrenamiento
- Mejorar la validación de código
- Agregar soporte para más recursos de AWS
- Optimizar el rendimiento

---

**¿Necesitas ayuda?** Revisa la sección [Solución de Problemas](#solución-de-problemas) o consulta los comentarios detallados en el código fuente.
