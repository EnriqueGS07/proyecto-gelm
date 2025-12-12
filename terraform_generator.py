"""
Módulo principal para generar código Terraform usando LangChain y Hugging Face
"""
import os
from typing import List, Optional, Dict

# Importaciones de LangChain (versión actualizada)
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
    """Clase principal para generar código Terraform usando modelos de Hugging Face"""
    
    def __init__(
        self,
        model_name: str = "microsoft/CodeGPT-small-py",
        device: str = "auto",
        max_length: int = 512,
        temperature: float = 0.7
    ):
        """
        Inicializa el generador de Terraform
        
        Args:
            model_name: Nombre del modelo de Hugging Face a usar
            device: Dispositivo a usar ('cpu', 'cuda', o 'auto')
            max_length: Longitud máxima de la secuencia generada
            temperature: Temperatura para la generación (controla la creatividad)
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
        """Carga el modelo y tokenizador de Hugging Face"""
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
            
            # Crear prompt template para Terraform con ejemplos (few-shot learning)
            prompt_template = PromptTemplate(
                input_variables=["description"],
                template="""Genera código Terraform válido y completo para AWS.

Ejemplo 1:
Descripción: Crear un bucket S3 con versionado habilitado
Código Terraform:
resource "aws_s3_bucket" "example" {{
  bucket = "my-example-bucket"
  
  versioning {{
    enabled = true
  }}
}}

Ejemplo 2:
Descripción: Crear una instancia EC2 tipo t2.micro
Código Terraform:
resource "aws_instance" "example" {{
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"
  
  tags = {{
    Name = "example-instance"
  }}
}}

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
        """Carga un modelo entrenado desde un directorio local"""
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
        Entrena el modelo con datos personalizados
        
        Args:
            training_data: Lista de diccionarios con 'description' y 'terraform_code'
            output_dir: Directorio donde guardar el modelo entrenado
            num_epochs: Número de épocas de entrenamiento
            batch_size: Tamaño del batch
            learning_rate: Tasa de aprendizaje
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
        Genera código Terraform a partir de una descripción
        
        Args:
            description: Descripción de lo que se quiere crear en Terraform
            max_new_tokens: Número máximo de tokens a generar
            
        Returns:
            Código Terraform generado
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
            
            # Limpiar el resultado
            if isinstance(result, dict):
                result = result.get("text", str(result))
            result = str(result).strip()
            
            # Remover markdown code blocks si están presentes
            if result.startswith("```terraform"):
                result = result[12:].strip()
            if result.startswith("```"):
                result = result[3:].strip()
            if result.endswith("```"):
                result = result[:-3].strip()
            
            # Detectar y extraer solo el primer bloque de código Terraform válido
            lines = result.split('\n')
            terraform_lines = []
            in_resource = False
            brace_count = 0
            found_resource = False
            
            for line in lines:
                line_stripped = line.strip()
                
                # Detectar inicio de resource
                if 'resource "' in line and not found_resource:
                    in_resource = True
                    found_resource = True
                    brace_count = line.count('{') - line.count('}')
                    terraform_lines.append(line)
                    continue
                
                # Si estamos dentro de un resource
                if in_resource:
                    brace_count += line.count('{') - line.count('}')
                    terraform_lines.append(line)
                    
                    # Si cerramos todos los bloques, terminar
                    if brace_count <= 0:
                        break
                    
                    # Detectar código inválido y detener
                    if any(invalid in line.lower() for invalid in ['aws_ebs_instance', 'aws_s3_instance', '```terraform']):
                        # Remover esta línea y todo lo que sigue
                        terraform_lines = terraform_lines[:-1]
                        break
                # Si encontramos otro resource después del primero, detener
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
            
            # Filtrar código claramente inválido
            invalid_patterns = ['aws_ebs_instance', 'aws_s3_instance', 'TODO:', 'github.com', 'dongeran']
            if any(pattern in result.lower() for pattern in invalid_patterns):
                result = "# Error: El modelo generó código inválido.\n# Por favor, entrena el modelo con más datos:\n# python train_model.py --data_dir training_data/ --output_dir models/terraform_generator --epochs 5"
            
            return result.strip()
        except Exception as e:
            print(f"Error al generar código: {e}")
            return f"Error: {str(e)}"
    
    def generate_with_context(self, description: str, context: Optional[str] = None) -> str:
        """
        Genera código Terraform con contexto adicional
        
        Args:
            description: Descripción de lo que se quiere crear
            context: Contexto adicional (ej: otros recursos existentes)
        
        Returns:
            Código Terraform generado
        """
        if context:
            full_description = f"{description}\n\nContexto: {context}"
        else:
            full_description = description
        
        return self.generate(full_description)

