"""
Módulo principal para generar código Terraform usando LangChain y Hugging Face.

Este módulo proporciona la clase TerraformGenerator que permite:
- Cargar modelos pre-entrenados o entrenados de Hugging Face
- Generar código Terraform a partir de descripciones en lenguaje natural
- Entrenar modelos personalizados con datos específicos de Terraform
- Validar y filtrar código generado para asegurar calidad

Autor: Proyecto GELM
Fecha: 2024
"""

import os
from typing import List, Optional, Dict

# ============================================================================
# IMPORTACIONES DE LANCHAIN
# ============================================================================
# LangChain ha cambiado su estructura en versiones recientes.
# Intentamos importar desde las nuevas ubicaciones primero, con fallbacks
# para mantener compatibilidad con versiones antiguas.
# ============================================================================
try:
    from langchain_community.llms import HuggingFacePipeline
except ImportError:
    try:
        from langchain.llms import HuggingFacePipeline
    except ImportError:
        raise ImportError("Necesitas instalar langchain-community: pip install langchain-community")

try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    try:
        from langchain.prompts import PromptTemplate
    except ImportError:
        raise ImportError("Necesitas instalar langchain-core: pip install langchain-core")

# En LangChain 1.x, LLMChain fue reemplazado por la API Runnable
# Usaremos PromptTemplate + LLM directamente con el operador |
try:
    # Intentar usar la nueva API de LangChain 1.x
    from langchain_core.runnables import RunnablePassthrough
    USE_RUNNABLE = True
except ImportError:
    USE_RUNNABLE = False
    # Fallback: intentar importar LLMChain de versiones antiguas
    try:
        from langchain.chains import LLMChain
        USE_RUNNABLE = False
    except ImportError:
        try:
            from langchain_community.chains import LLMChain
            USE_RUNNABLE = False
        except ImportError:
            LLMChain = None
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class TerraformGenerator:
    """
    Clase principal para generar código Terraform usando modelos de Hugging Face.
    
    Esta clase encapsula toda la lógica para:
    - Cargar modelos de lenguaje pre-entrenados o entrenados
    - Generar código Terraform válido desde descripciones en lenguaje natural
    - Entrenar modelos con datos personalizados de Terraform
    - Validar y limpiar el código generado
    
    Ejemplo de uso básico:
        >>> generator = TerraformGenerator()
        >>> generator.load_model()
        >>> codigo = generator.generate("Crear un bucket S3")
        >>> print(codigo)
    
    Ejemplo con modelo entrenado:
        >>> generator = TerraformGenerator()
        >>> generator.load_from_local("models/terraform_generator")
        >>> codigo = generator.generate("Crear una instancia EC2")
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/CodeGPT-small-py",
        device: str = "auto",
        max_length: int = 512,
        temperature: float = 0.7
    ):
        """
        Inicializa el generador de Terraform.
        
        Args:
            model_name (str): Nombre del modelo de Hugging Face a usar.
                Por defecto: "microsoft/CodeGPT-small-py"
                Otros modelos recomendados: "gpt2", "Salesforce/codegen-350M-mono"
            device (str): Dispositivo a usar para la inferencia.
                - "auto": Detecta automáticamente (CUDA si está disponible, sino CPU)
                - "cpu": Fuerza uso de CPU
                - "cuda": Fuerza uso de GPU (requiere PyTorch con CUDA)
            max_length (int): Longitud máxima de la secuencia generada en tokens.
                Valores más altos permiten código más largo pero consumen más memoria.
                Por defecto: 512
            temperature (float): Temperatura para la generación (0.0 - 1.0).
                - Valores bajos (0.1-0.3): Más determinístico, código más predecible
                - Valores altos (0.7-1.0): Más creativo pero menos consistente
                Por defecto: 0.7 (se ajusta automáticamente a 0.3 para modelos pre-entrenados)
        
        Atributos:
            model: Modelo de lenguaje cargado (AutoModelForCausalLM)
            tokenizer: Tokenizador asociado al modelo
            llm: Pipeline de LangChain para generación
            chain: Cadena de procesamiento de LangChain (prompt + modelo)
        """
        self.model_name = model_name
        if device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.model = None
        self.tokenizer = None
        self.llm = None
        self.chain = None
        
    def load_model(self):
        """
        Carga el modelo y tokenizador de Hugging Face.
        
        Este método:
        1. Descarga el modelo desde Hugging Face si no está en caché
        2. Configura el tokenizador con el token de padding apropiado
        3. Crea un pipeline de generación de texto
        4. Integra el pipeline con LangChain
        5. Configura el prompt template con ejemplos few-shot
        
        El prompt template incluye ejemplos de:
        - Buckets S3 con versionado
        - Instancias EC2
        - Volúmenes EBS
        
        Raises:
            ImportError: Si faltan dependencias necesarias
            OSError: Si no se puede descargar el modelo (sin conexión)
            Exception: Si falla la carga, intenta con modelo alternativo (gpt2)
        
        Nota:
            La primera vez que se ejecuta, el modelo se descarga automáticamente
            desde Hugging Face. Esto puede tomar varios minutos dependiendo
            del tamaño del modelo y la velocidad de conexión.
        """
        print(f"Cargando modelo {self.model_name}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if TORCH_AVAILABLE:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name
                )
            
            # Configurar pad_token si no existe
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Crear pipeline de Hugging Face
            # Reducir temperatura para modelos pre-entrenados (más determinístico)
            effective_temp = 0.3 if self.model_name == "microsoft/CodeGPT-small-py" or "gpt2" in self.model_name.lower() else self.temperature
            
            pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                max_new_tokens=300,  # Limitar tokens generados
                max_length=self.max_length,
                temperature=effective_temp,
                do_sample=True,
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2  # Reducir repeticiones
            )
            
            # Crear LLM de LangChain
            # Intentar usar la nueva versión si está disponible
            try:
                from langchain_huggingface import HuggingFacePipeline as NewHuggingFacePipeline
                self.llm = NewHuggingFacePipeline(pipeline=pipe)
            except ImportError:
                # Fallback a la versión antigua (deprecada pero funcional)
                import warnings
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                self.llm = HuggingFacePipeline(pipeline=pipe)
            
            # Crear prompt template para Terraform con ejemplos mejorados
            prompt_template = PromptTemplate(
                input_variables=["description"],
                template="""Genera código Terraform válido y completo para AWS. Usa solo recursos y atributos válidos de Terraform.

Ejemplo 1 - Bucket S3:
Descripción: Crear un bucket S3 con versionado habilitado
Código Terraform:
resource "aws_s3_bucket" "example" {{
  bucket = "my-example-bucket"
  
  versioning {{
    enabled = true
  }}
}}

Ejemplo 2 - Instancia EC2:
Descripción: Crear una instancia EC2 tipo t2.micro
Código Terraform:
resource "aws_instance" "example" {{
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"
  
  tags = {{
    Name = "example-instance"
  }}
}}

Ejemplo 3 - Volumen EBS:
Descripción: Crear un volumen EBS
Código Terraform:
resource "aws_ebs_volume" "example" {{
  availability_zone = "us-east-1a"
  size              = 20
  type              = "gp3"
  
  tags = {{
    Name = "example-volume"
  }}
}}

IMPORTANTE: 
- Usa solo recursos válidos: aws_s3_bucket, aws_instance, aws_ebs_volume, etc.
- NO uses recursos inválidos como aws_ebs_instance o aws_s3_instance
- Usa solo atributos válidos para cada recurso
- Genera código completo y válido

Descripción: {description}
Código Terraform:
"""
            )
            
            # Crear cadena de LangChain
            # En LangChain 1.x, usamos el operador | en lugar de LLMChain
            if USE_RUNNABLE:
                # Nueva API: prompt | llm
                self.chain = prompt_template | self.llm
            else:
                # API antigua: LLMChain
                if LLMChain is not None:
                    self.chain = LLMChain(llm=self.llm, prompt=prompt_template)
                else:
                    # Fallback: usar prompt y llm directamente
                    self.chain = {"prompt": prompt_template, "llm": self.llm}
            
            print("Modelo cargado exitosamente")
            
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            print("Intentando con un modelo alternativo...")
            # Fallback a un modelo más pequeño
            self.model_name = "gpt2"
            self.load_model()
    
    def load_from_local(self, model_path: str):
        """
        Carga un modelo entrenado desde un directorio local.
        
        Este método es útil para cargar modelos que fueron entrenados
        previamente con el método train() y guardados en el sistema de archivos.
        
        Args:
            model_path (str): Ruta al directorio que contiene el modelo entrenado.
                Debe contener:
                - model.safetensors o pytorch_model.bin (pesos del modelo)
                - config.json (configuración del modelo)
                - tokenizer.json y archivos relacionados (tokenizador)
                - generation_config.json (configuración de generación)
        
        Ejemplo:
            >>> generator = TerraformGenerator()
            >>> generator.load_from_local("models/terraform_generator")
            >>> codigo = generator.generate("Crear un bucket S3")
        
        Raises:
            FileNotFoundError: Si el directorio no existe
            OSError: Si los archivos del modelo están corruptos o incompletos
        """
        print(f"Cargando modelo desde {model_path}...")
        self.model_name = model_path
        self.load_model()
    
    def train(
        self,
        training_data: List[Dict[str, str]],
        output_dir: str,
        num_epochs: int = 5,
        batch_size: int = 4,
        learning_rate: float = 5e-5
    ):
        """
        Entrena el modelo con datos personalizados de Terraform.
        
        Este método realiza fine-tuning del modelo base con ejemplos específicos
        de código Terraform. El proceso incluye:
        1. Formateo de los datos de entrenamiento
        2. Tokenización de los ejemplos
        3. Configuración de los argumentos de entrenamiento
        4. Entrenamiento del modelo usando el Trainer de Hugging Face
        5. Guardado del modelo entrenado y tokenizador
        
        Args:
            training_data (List[Dict[str, str]]): Lista de ejemplos de entrenamiento.
                Cada diccionario debe tener:
                - "description": Descripción en lenguaje natural del recurso
                - "terraform_code": Código Terraform válido correspondiente
                
                Ejemplo:
                    [
                        {
                            "description": "Crear un bucket S3 con versionado",
                            "terraform_code": "resource \"aws_s3_bucket\" \"example\" {\n  bucket = \"my-bucket\"\n  versioning {\n    enabled = true\n  }\n}"
                        }
                    ]
            
            output_dir (str): Directorio donde se guardará el modelo entrenado.
                Se creará automáticamente si no existe.
                El modelo se guardará con todos los archivos necesarios:
                - model.safetensors (pesos del modelo)
                - config.json, tokenizer.json, etc.
            
            num_epochs (int): Número de épocas de entrenamiento.
                - Más épocas = mejor aprendizaje pero más tiempo
                - Recomendado: 3-10 épocas dependiendo del tamaño del dataset
                - Por defecto: 5
            
            batch_size (int): Tamaño del batch para entrenamiento.
                - Valores más altos = más rápido pero más memoria
                - Ajustar según RAM/GPU disponible
                - Por defecto: 4
            
            learning_rate (float): Tasa de aprendizaje para el optimizador.
                - Valores típicos: 1e-5 a 5e-5 para fine-tuning
                - Valores muy altos pueden causar inestabilidad
                - Por defecto: 5e-5
        
        Proceso de entrenamiento:
            - Los datos se formatean con un prompt que incluye la descripción
            - Se tokenizan usando el tokenizador del modelo
            - Se entrena usando Language Modeling (no MLM)
            - Se guardan checkpoints al final de cada época
            - El modelo final se guarda en output_dir
        
        Tiempo estimado:
            - 5-10 ejemplos: 15-30 minutos (CPU)
            - 50+ ejemplos: 1-3 horas (CPU) o 30-60 minutos (GPU)
            - 100+ ejemplos: 3-5 horas (CPU) o 1-2 horas (GPU)
        
        Raises:
            ValueError: Si training_data está vacío o tiene formato incorrecto
            RuntimeError: Si hay problemas durante el entrenamiento
            OSError: Si no se puede escribir en output_dir
        
        Nota:
            El entrenamiento requiere recursos computacionales significativos.
            Se recomienda usar GPU si está disponible para acelerar el proceso.
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        print("Preparando datos de entrenamiento...")
        
        # Preparar los datos en formato de texto
        def format_training_example(example):
            text = f"""Genera código Terraform para la siguiente descripción:

Descripción: {example['description']}

Código Terraform:
```terraform
{example['terraform_code']}
```"""
            return {"text": text}
        
        # Crear dataset
        dataset = Dataset.from_list(training_data)
        dataset = dataset.map(format_training_example)
        
        # Asegurar que el dataset tenga la columna 'text'
        if "text" not in dataset.column_names:
            raise ValueError("El dataset debe tener una columna 'text' después del formateo")
        
        # Tokenizar el dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors=None  # No retornar tensores, solo listas
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=dataset.column_names  # Remover columnas originales, solo mantener las tokenizadas
        )
        
        # Configurar argumentos de entrenamiento
        # Intentar usar eval_strategy (versiones nuevas) o evaluation_strategy (versiones antiguas)
        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                logging_dir=f"{output_dir}/logs",
                save_strategy="epoch",
                eval_strategy="no",  # Versiones nuevas de transformers
                push_to_hub=False,
                remove_unused_columns=False,
                report_to="none"
            )
        except TypeError:
            # Fallback para versiones antiguas
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                logging_dir=f"{output_dir}/logs",
                save_strategy="epoch",
                push_to_hub=False,
                remove_unused_columns=False,
                report_to="none"
            )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Crear trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator
        )
        
        print("Iniciando entrenamiento...")
        trainer.train()
        
        print(f"Guardando modelo en {output_dir}...")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print("Entrenamiento completado")
    
    def generate(self, description: str, max_new_tokens: int = 256) -> str:
        """
        Genera código Terraform válido a partir de una descripción en lenguaje natural.
        
        Este método es el corazón del generador. Realiza los siguientes pasos:
        1. Carga el modelo si no está cargado
        2. Formatea el prompt con la descripción del usuario
        3. Genera código usando el modelo de lenguaje
        4. Limpia y extrae el código Terraform válido
        5. Valida el código contra patrones conocidos de errores
        6. Filtra atributos inválidos según el tipo de recurso
        
        Args:
            description (str): Descripción en lenguaje natural de lo que se quiere crear.
                Ejemplos:
                - "Crear un bucket S3 con versionado habilitado"
                - "Crear una instancia EC2 tipo t2.micro"
                - "Crear un volumen EBS de 20GB"
                - "Crear un Internet Gateway para una VPC"
            
            max_new_tokens (int): Número máximo de tokens a generar.
                - Más tokens = código más largo pero más lento
                - Por defecto: 256 (suficiente para la mayoría de recursos)
        
        Returns:
            str: Código Terraform válido y completo.
                Si hay errores, retorna un mensaje de error como comentario.
        
        Proceso de validación:
            El método valida el código generado de múltiples formas:
            
            1. **Filtrado de patrones inválidos:**
               - Recursos que no existen: aws_ebs_instance, aws_s3_instance
               - Atributos inválidos: volume_name, instance_name
               - Código basura: TODO:, github.com, dongeran
               - Markdown residual: ```terraform, ```
            
            2. **Validación por tipo de recurso:**
               - aws_internet_gateway: Solo permite vpc_id, tags
               - aws_s3_bucket: Solo permite bucket, tags, versioning, etc.
               - aws_instance: Solo permite ami, instance_type, tags, etc.
               - aws_ebs_volume: Solo permite availability_zone, size, type, etc.
               - aws_kms_key: Solo permite description, deletion_window_in_days, etc.
            
            3. **Validación estructural:**
               - Verifica que haya un recurso válido
               - Verifica que las llaves {} estén balanceadas
               - Extrae solo el primer recurso válido encontrado
        
        Ejemplo:
            >>> generator = TerraformGenerator()
            >>> generator.load_model()
            >>> codigo = generator.generate("Crear un bucket S3")
            >>> print(codigo)
            resource "aws_s3_bucket" "example" {
              bucket = "my-bucket"
            }
        
        Raises:
            Exception: Si hay errores durante la generación, retorna mensaje de error
                en lugar de lanzar excepción (para mejor UX)
        
        Nota:
            - El modelo pre-entrenado puede generar código de baja calidad
            - Se recomienda entrenar el modelo con datos específicos para mejores resultados
            - La validación es agresiva para asegurar código válido, pero puede
              rechazar código válido en casos edge
        """
        if self.chain is None:
            self.load_model()
        
        try:
            # Manejar diferentes APIs de LangChain
            if USE_RUNNABLE:
                # Nueva API de LangChain 1.x
                result = self.chain.invoke({"description": description})
            elif hasattr(self.chain, 'run'):
                # API antigua con LLMChain
                result = self.chain.run(description=description)
            else:
                # Fallback: usar prompt y llm directamente
                prompt_text = self.chain["prompt"].format(description=description)
                result = self.chain["llm"].invoke(prompt_text)
            
            # ========================================================================
            # LIMPIEZA Y EXTRACCIÓN DE CÓDIGO TERRAFORM
            # ========================================================================
            # El modelo puede generar texto adicional además del código Terraform.
            # Este proceso extrae solo el código válido y lo limpia.
            # ========================================================================
            
            # Convertir resultado a string si es un diccionario
            if isinstance(result, dict):
                result = result.get("text", str(result))
            result = str(result).strip()
            
            # Remover markdown code blocks si están presentes
            # El modelo a veces envuelve el código en bloques markdown
            if result.startswith("```terraform"):
                result = result[12:].strip()
            if result.startswith("```"):
                result = result[3:].strip()
            if result.endswith("```"):
                result = result[:-3].strip()
            
            # ========================================================================
            # EXTRACCIÓN DEL PRIMER RECURSO VÁLIDO
            # ========================================================================
            # El modelo puede generar múltiples recursos o texto adicional.
            # Extraemos solo el primer recurso Terraform válido encontrado.
            # ========================================================================
            lines = result.split('\n')
            terraform_lines = []
            in_resource = False
            brace_count = 0  # Contador de llaves para detectar fin de bloque
            found_resource = False
            
            for line in lines:
                line_stripped = line.strip()
                
                # Detectar inicio de un bloque "resource"
                # Ejemplo: resource "aws_s3_bucket" "example" {
                if 'resource "' in line and not found_resource:
                    in_resource = True
                    found_resource = True
                    # Contar llaves en esta línea (puede tener { y })
                    brace_count = line.count('{') - line.count('}')
                    terraform_lines.append(line)
                    continue
                
                # Si estamos dentro de un bloque resource
                if in_resource:
                    # Actualizar contador de llaves
                    brace_count += line.count('{') - line.count('}')
                    terraform_lines.append(line)
                    
                    # Si cerramos todos los bloques (brace_count <= 0), terminamos
                    if brace_count <= 0:
                        break
                    
                    # Detectar código inválido y detener inmediatamente
                    # Estos son recursos que no existen en Terraform
                    if any(invalid in line.lower() for invalid in ['aws_ebs_instance', 'aws_s3_instance', '```terraform']):
                        # Remover esta línea inválida y todo lo que sigue
                        terraform_lines = terraform_lines[:-1]
                        break
                
                # Si encontramos otro resource después del primero, detener
                # Solo queremos el primer recurso válido
                elif 'resource "' in line and found_resource:
                    break
            
            # Si encontramos código válido, usarlo
            if terraform_lines and found_resource:
                result = '\n'.join(terraform_lines)
                # Asegurar que termine correctamente
                if brace_count > 0:
                    result += '\n' + '}' * brace_count
            else:
                # Si no hay código válido, intentar extraer cualquier resource
                for line in lines:
                    if 'resource "' in line or (terraform_lines and brace_count > 0):
                        if 'resource "' in line:
                            terraform_lines = [line]
                            brace_count = line.count('{') - line.count('}')
                        else:
                            terraform_lines.append(line)
                            brace_count += line.count('{') - line.count('}')
                            if brace_count <= 0:
                                break
                
                if terraform_lines:
                    result = '\n'.join(terraform_lines)
                else:
                    result = "# Error: No se pudo generar código Terraform válido.\n# El modelo necesita más entrenamiento o la descripción no es clara."
            
            # ========================================================================
            # VALIDACIÓN DE CÓDIGO TERRAFORM
            # ========================================================================
            # Validamos el código generado en múltiples niveles para asegurar
            # que sea válido y no contenga errores comunes.
            # ========================================================================
            
            # Patrones inválidos que indican código basura o errores
            invalid_patterns = [
                'aws_ebs_instance',  # No existe en Terraform, debe ser aws_instance
                'aws_s3_instance',   # No existe en Terraform
                'volume_name',       # Atributo inválido en aws_ebs_volume
                'aws_volumes[',      # Sintaxis inválida (no es sintaxis de Terraform)
                'instance_name',     # Atributo inválido (mezcla de recursos)
                'TODO:',             # Comentarios de desarrollo que no deben estar
                'github.com',         # URLs que indican código copiado o basura
                'dongeran',          # Palabra basura común en modelos pre-entrenados
                '```terraform',      # Markdown no debe estar en el código final
                '```'                # Markdown residual
            ]
            
            # ========================================================================
            # VALIDACIÓN DE ATRIBUTOS POR TIPO DE RECURSO
            # ========================================================================
            # Cada recurso de Terraform tiene atributos específicos válidos.
            # Esta validación detecta cuando el modelo mezcla atributos de diferentes
            # recursos (error común en modelos no entrenados).
            # ========================================================================
            resource_attr_validation = {
                'aws_internet_gateway': {
                    'valid': ['vpc_id', 'tags'],  # Solo estos atributos son válidos
                    'invalid': ['key', 'source', 'instance_name', 'target_group_arns', 
                               'bucket', 'ami', 'instance_type', 'availability_zone', 
                               'size', 'type', 'kms_id', 'kms_client_side_encryption']
                },
                'aws_s3_bucket': {
                    'valid': ['bucket', 'tags', 'versioning', 'server_side_encryption_configuration'],
                    'invalid': ['vpc_id', 'instance_type', 'ami', 'availability_zone', 'target_group_arns', 'kms_id', 'key', 'source']
                },
                'aws_instance': {
                    'valid': ['ami', 'instance_type', 'tags', 'key_name', 'security_groups'],
                    'invalid': ['bucket', 'vpc_id', 'key', 'source', 'target_group_arns', 'kms_id', 'kms_client_side_encryption']
                },
                'aws_ebs_volume': {
                    'valid': ['availability_zone', 'size', 'type', 'encrypted', 'tags'],
                    'invalid': ['volume_name', 'instance_name', 'bucket', 'vpc_id', 'target_group_arns', 'key', 'source', 'kms_id']
                },
                'aws_kms_key': {
                    'valid': ['description', 'deletion_window_in_days', 'enable_key_rotation', 'policy', 'tags', 'key_usage'],
                    'invalid': ['kms_id', 'key', 'source', 'kms_client_side_encryption', 'bucket', 'vpc_id', 'instance_type', 'ami', 'availability_zone']
                }
            }
            
            # Detectar si el código contiene patrones inválidos
            result_lower = result.lower()  # Convertir a minúsculas para comparación
            has_invalid = any(pattern in result_lower for pattern in invalid_patterns)
            
            # Validar atributos específicos por tipo de recurso
            # Esto detecta cuando el modelo mezcla atributos de diferentes recursos
            for resource_type, validation in resource_attr_validation.items():
                if f'resource "{resource_type}"' in result_lower:
                    # Verificar si tiene atributos inválidos para este tipo de recurso
                    for invalid_attr in validation['invalid']:
                        # Buscar atributo inválido (con o sin espacios alrededor del =)
                        # Ejemplo: "key = " o "key="
                        if f'{invalid_attr} =' in result_lower or f'{invalid_attr}=' in result_lower:
                            has_invalid = True
                            break
                    if has_invalid:
                        break
            
            # Validar estructura básica de Terraform
            # 1. Debe tener un recurso válido (resource "tipo" "nombre")
            has_valid_resource = 'resource "' in result and '"' in result.split('resource "')[1] if 'resource "' in result else False
            # 2. Las llaves deben estar balanceadas ({ y })
            has_valid_syntax = result.count('{') == result.count('}') if result else False
            
            # Si hay código inválido o estructura incorrecta, intentar limpiar
            if has_invalid or not has_valid_resource or not has_valid_syntax:
                # ====================================================================
                # INTENTO DE LIMPIEZA Y EXTRACCIÓN DE CÓDIGO VÁLIDO
                # ====================================================================
                # Si el código tiene problemas, intentamos extraer solo las partes
                # válidas filtrando líneas problemáticas.
                # ====================================================================
                lines = result.split('\n')
                clean_lines = []
                in_valid_resource = False
                brace_count = 0
                
                for line in lines:
                    line_lower = line.lower()
                    # Saltar líneas con patrones inválidos (código basura)
                    if any(pattern in line_lower for pattern in invalid_patterns):
                        continue
                    
                    # Detectar inicio de resource válido (no aws_ebs_instance, etc.)
                    if 'resource "' in line and not any(inv in line_lower for inv in ['aws_ebs_instance', 'aws_s3_instance']):
                        in_valid_resource = True
                        brace_count = line.count('{') - line.count('}')
                        clean_lines.append(line)
                        continue
                    
                    # Si estamos dentro de un resource válido, agregar línea
                    if in_valid_resource:
                        brace_count += line.count('{') - line.count('}')
                        clean_lines.append(line)
                        
                        # Si cerramos todos los bloques, terminamos
                        if brace_count <= 0:
                            break
                
                # Si logramos extraer código válido con llaves balanceadas, usarlo
                if clean_lines and brace_count == 0:
                    result = '\n'.join(clean_lines)
                else:
                    # Si no pudimos limpiar el código, retornar mensaje de error
                    result = "# Error: El modelo generó código inválido.\n# Por favor, entrena el modelo con más datos:\n# python train_model.py --data_dir training_data/ --output_dir models/terraform_generator --epochs 5"
            
            return result.strip()
        except Exception as e:
            print(f"Error al generar código: {e}")
            return f"Error: {str(e)}"
    
    def generate_with_context(self, description: str, context: Optional[str] = None) -> str:
        """
        Genera código Terraform con contexto adicional sobre recursos existentes.
        
        Este método es útil cuando necesitas generar código que referencia
        otros recursos de Terraform que ya existen o se crearán en el mismo
        archivo de configuración.
        
        Args:
            description (str): Descripción en lenguaje natural de lo que se quiere crear.
                Ejemplo: "Crear un grupo de seguridad"
            
            context (Optional[str]): Contexto adicional sobre el entorno o recursos existentes.
                Puede incluir:
                - Información sobre VPCs existentes
                - IDs de recursos que se referenciarán
                - Configuraciones del entorno
                
                Ejemplo: "Para una instancia EC2 en la VPC vpc-12345"
        
        Returns:
            str: Código Terraform generado que considera el contexto proporcionado.
        
        Ejemplo:
            >>> generator = TerraformGenerator()
            >>> generator.load_model()
            >>> codigo = generator.generate_with_context(
            ...     "Crear un grupo de seguridad",
            ...     "Para una instancia EC2 en la VPC vpc-12345"
            ... )
        
        Nota:
            El contexto se concatena con la descripción antes de generar.
            El modelo puede o no usar el contexto dependiendo de su entrenamiento.
        """
        if context:
            full_description = f"{description}\n\nContexto: {context}"
        else:
            full_description = description
        
        return self.generate(full_description)

