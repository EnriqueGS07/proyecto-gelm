import os
from typing import List, Optional, Dict
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

try:
    from langchain_core.runnables import RunnablePassthrough
    USE_RUNNABLE = True
except ImportError:
    USE_RUNNABLE = False
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
    
    def __init__(
        self,
        model_name: str = "microsoft/CodeGPT-small-py",
        device: str = "auto",
        max_length: int = 512,
        temperature: float = 0.7
    ):
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
            

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            

            effective_temp = 0.3 if self.model_name == "microsoft/CodeGPT-small-py" or "gpt2" in self.model_name.lower() else self.temperature
            
            pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                max_new_tokens=300, 
                max_length=self.max_length,
                temperature=effective_temp,
                do_sample=True,
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2  
            )
            

            try:
                from langchain_huggingface import HuggingFacePipeline as NewHuggingFacePipeline
                self.llm = NewHuggingFacePipeline(pipeline=pipe)
            except ImportError:
                import warnings
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                self.llm = HuggingFacePipeline(pipeline=pipe)
            
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
            
            if USE_RUNNABLE:
                self.chain = prompt_template | self.llm
            else:
                if LLMChain is not None:
                    self.chain = LLMChain(llm=self.llm, prompt=prompt_template)
                else:
                    self.chain = {"prompt": prompt_template, "llm": self.llm}
            
            print("Modelo cargado exitosamente")
            
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            print("Intentando con un modelo alternativo...")
            self.model_name = "gpt2"
            self.load_model()
    
    def load_from_local(self, model_path: str):
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
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        print("Preparando datos de entrenamiento...")
        
        def format_training_example(example):
            text = f"""Genera código Terraform para la siguiente descripción:

Descripción: {example['description']}

Código Terraform:
```terraform
{example['terraform_code']}
```"""
            return {"text": text}
        
        dataset = Dataset.from_list(training_data)
        dataset = dataset.map(format_training_example)
        
        if "text" not in dataset.column_names:
            raise ValueError("El dataset debe tener una columna 'text' después del formateo")
        
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
            remove_columns=dataset.column_names
        )
        
        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                logging_dir=f"{output_dir}/logs",
                save_strategy="epoch",
                eval_strategy="no",
                push_to_hub=False,
                remove_unused_columns=False,
                report_to="none"
            )
        except TypeError:
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
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
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
        if self.chain is None:
            self.load_model()
        
        try:
            if USE_RUNNABLE:
                result = self.chain.invoke({"description": description})
            elif hasattr(self.chain, 'run'):
                result = self.chain.run(description=description)
            else:
                prompt_text = self.chain["prompt"].format(description=description)
                result = self.chain["llm"].invoke(prompt_text)
            
            if isinstance(result, dict):
                result = result.get("text", str(result))
            result = str(result).strip()
            
            if result.startswith("```terraform"):
                result = result[12:].strip()
            if result.startswith("```"):
                result = result[3:].strip()
            if result.endswith("```"):
                result = result[:-3].strip()
            
            lines = result.split('\n')
            terraform_lines = []
            in_resource = False
            brace_count = 0
            found_resource = False
            
            for line in lines:
                line_stripped = line.strip()
                
                if 'resource "' in line and not found_resource:
                    in_resource = True
                    found_resource = True
                    brace_count = line.count('{') - line.count('}')
                    terraform_lines.append(line)
                    continue
                
                if in_resource:
                    brace_count += line.count('{') - line.count('}')
                    terraform_lines.append(line)
                    
                    if brace_count <= 0:
                        break
                    
                    if any(invalid in line.lower() for invalid in ['aws_ebs_instance', 'aws_s3_instance', '```terraform']):
                        terraform_lines = terraform_lines[:-1]
                        break
                elif 'resource "' in line and found_resource:
                    break
            
            if terraform_lines and found_resource:
                result = '\n'.join(terraform_lines)
                if brace_count > 0:
                    result += '\n' + '}' * brace_count
            else:
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
            
            invalid_patterns = [
                'aws_ebs_instance',
                'aws_s3_instance',
                'volume_name',
                'aws_volumes[',
                'instance_name',
                'TODO:',
                'github.com',
                'dongeran',
                '```terraform',
                '```'
            ]
            
            resource_attr_validation = {
                'aws_internet_gateway': {
                    'valid': ['vpc_id', 'tags'],
                    'invalid': ['key', 'source', 'instance_name', 'target_group_arns', 'bucket', 'ami', 'instance_type', 'availability_zone', 'size', 'type', 'kms_id', 'kms_client_side_encryption']
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
            
            result_lower = result.lower()
            has_invalid = any(pattern in result_lower for pattern in invalid_patterns)
            
            for resource_type, validation in resource_attr_validation.items():
                if f'resource "{resource_type}"' in result_lower:
                    for invalid_attr in validation['invalid']:
                        if f'{invalid_attr} =' in result_lower or f'{invalid_attr}=' in result_lower:
                            has_invalid = True
                            break
                    if has_invalid:
                        break
            
            has_valid_resource = 'resource "' in result and '"' in result.split('resource "')[1] if 'resource "' in result else False
            has_valid_syntax = result.count('{') == result.count('}') if result else False
            
            if has_invalid or not has_valid_resource or not has_valid_syntax:
                lines = result.split('\n')
                clean_lines = []
                in_valid_resource = False
                brace_count = 0
                
                for line in lines:
                    line_lower = line.lower()
                    if any(pattern in line_lower for pattern in invalid_patterns):
                        continue
                    
                    if 'resource "' in line and not any(inv in line_lower for inv in ['aws_ebs_instance', 'aws_s3_instance']):
                        in_valid_resource = True
                        brace_count = line.count('{') - line.count('}')
                        clean_lines.append(line)
                        continue
                    
                    if in_valid_resource:
                        brace_count += line.count('{') - line.count('}')
                        clean_lines.append(line)
                        
                        if brace_count <= 0:
                            break
                
                if clean_lines and brace_count == 0:
                    result = '\n'.join(clean_lines)
                else:
                    result = "# Error: El modelo generó código inválido.\n# Por favor, entrena el modelo con más datos:\n# python train_model.py --data_dir training_data/ --output_dir models/terraform_generator --epochs 5"
            
            return result.strip()
        except Exception as e:
            print(f"Error al generar código: {e}")
            return f"Error: {str(e)}"
    
    def generate_with_context(self, description: str, context: Optional[str] = None) -> str:
        if context:
            full_description = f"{description}\n\nContexto: {context}"
        else:
            full_description = description
        
        return self.generate(full_description)

