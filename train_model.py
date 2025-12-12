"""
Script para entrenar el modelo con archivos de datos de entrenamiento.

Este script carga ejemplos de entrenamiento desde archivos JSON o TXT
y entrena el modelo base para especializarlo en generación de código Terraform.

Uso básico:
    python train_model.py --data_dir training_data/ --output_dir models/terraform_generator

Uso avanzado:
    python train_model.py \\
        --data_dir training_data/ \\
        --output_dir models/terraform_generator \\
        --epochs 10 \\
        --batch_size 8 \\
        --learning_rate 3e-5

Formato de datos:
    Los archivos deben estar en formato JSON con estructura:
    [
        {
            "description": "Descripción en lenguaje natural",
            "terraform_code": "resource \"aws_...\" \"name\" { ... }"
        }
    ]

Autor: Proyecto GELM
Fecha: 2024
"""

import os
import json
import argparse
from pathlib import Path
from terraform_generator import TerraformGenerator


def load_training_data(data_dir: str) -> list:
    """
    Carga los datos de entrenamiento desde archivos en el directorio.
    
    Este método busca y carga todos los archivos de entrenamiento válidos
    en el directorio especificado. Soporta múltiples formatos:
    - Archivos JSON: Formato estándar con lista de ejemplos
    - Archivos TXT: Formato simple con DESCRIPTION: y TERRAFORM:
    
    Args:
        data_dir (str): Directorio que contiene los archivos de entrenamiento.
            El método busca recursivamente archivos .json y .txt en este directorio.
    
    Returns:
        list: Lista de diccionarios, cada uno con:
            - "description": Descripción en lenguaje natural
            - "terraform_code": Código Terraform correspondiente
    
    Formato JSON esperado:
        [
            {
                "description": "Crear un bucket S3",
                "terraform_code": "resource \"aws_s3_bucket\" \"example\" { ... }"
            }
        ]
    
    Formato TXT esperado:
        DESCRIPTION: Crear un bucket S3
        TERRAFORM:
        resource "aws_s3_bucket" "example" {
          bucket = "my-bucket"
        }
    
    Raises:
        FileNotFoundError: Si el directorio no existe
        json.JSONDecodeError: Si un archivo JSON está mal formateado
    
    Nota:
        - Los archivos JSON pueden contener una lista o un objeto único
        - Los archivos TXT deben tener el formato exacto mostrado arriba
        - Se ignoran archivos que no se pueden parsear (con mensaje de error)
    """
    training_data = []
    data_path = Path(data_dir)
    
    # Buscar archivos JSON
    json_files = list(data_path.glob("*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    training_data.extend(data)
                elif isinstance(data, dict):
                    training_data.append(data)
        except Exception as e:
            print(f"Error al cargar {json_file}: {e}")
    
    # Buscar archivos de texto con formato específico
    txt_files = list(data_path.glob("*.txt"))
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Intentar parsear como formato simple
                # Formato esperado: DESCRIPTION: ... TERRAFORM: ...
                if "DESCRIPTION:" in content and "TERRAFORM:" in content:
                    parts = content.split("TERRAFORM:")
                    if len(parts) == 2:
                        desc = parts[0].replace("DESCRIPTION:", "").strip()
                        terraform = parts[1].strip()
                        training_data.append({
                            "description": desc,
                            "terraform_code": terraform
                        })
        except Exception as e:
            print(f"Error al cargar {txt_file}: {e}")
    
    print(f"Cargados {len(training_data)} ejemplos de entrenamiento")
    return training_data


def main():
    """
    Función principal que parsea argumentos y ejecuta el entrenamiento.
    
    Este script coordina todo el proceso de entrenamiento:
    1. Carga los datos de entrenamiento desde archivos
    2. Inicializa el generador con el modelo base
    3. Ejecuta el entrenamiento con los parámetros especificados
    4. Guarda el modelo entrenado en el directorio de salida
    
    Argumentos:
        --data_dir: Directorio con archivos de entrenamiento (default: training_data)
        --output_dir: Directorio donde guardar el modelo (default: models/terraform_generator)
        --model_name: Modelo base a usar (default: microsoft/CodeGPT-small-py)
        --epochs: Número de épocas (default: 5)
        --batch_size: Tamaño del batch (default: 4)
        --learning_rate: Tasa de aprendizaje (default: 5e-5)
    
    Tiempo estimado:
        - 5-10 ejemplos: 15-30 minutos (CPU)
        - 50+ ejemplos: 1-3 horas (CPU) o 30-60 minutos (GPU)
        - 100+ ejemplos: 3-5 horas (CPU) o 1-2 horas (GPU)
    
    Ejemplo:
        python train_model.py \\
            --data_dir training_data/ \\
            --output_dir models/terraform_generator \\
            --epochs 5 \\
            --batch_size 4
    """
    parser = argparse.ArgumentParser(
        description="Entrenar modelo para generar código Terraform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Entrenamiento básico
  python train_model.py --data_dir training_data/ --output_dir models/terraform_generator
  
  # Entrenamiento con más épocas
  python train_model.py --data_dir training_data/ --output_dir models/terraform_generator --epochs 10
  
  # Entrenamiento con parámetros personalizados
  python train_model.py \\
      --data_dir training_data/ \\
      --output_dir models/terraform_generator \\
      --epochs 5 \\
      --batch_size 8 \\
      --learning_rate 3e-5 \\
      --model_name gpt2
        """
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="training_data",
        help="Directorio con los archivos de entrenamiento"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/terraform_generator",
        help="Directorio donde guardar el modelo entrenado"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/CodeGPT-small-py",
        help="Modelo base de Hugging Face a usar"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Número de épocas de entrenamiento (default: 5)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Tamaño del batch"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Tasa de aprendizaje"
    )
    
    args = parser.parse_args()
    
    # Crear directorio de salida si no existe
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Cargar datos de entrenamiento
    training_data = load_training_data(args.data_dir)
    
    if not training_data:
        print(f"Error: No se encontraron datos de entrenamiento en {args.data_dir}")
        print("Por favor, asegúrate de tener archivos .json o .txt con el formato correcto")
        return
    
    # Inicializar generador
    generator = TerraformGenerator(model_name=args.model_name)
    
    # Entrenar modelo
    generator.train(
        training_data=training_data,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    print(f"\nModelo entrenado y guardado en {args.output_dir}")
    print("Puedes usar este modelo con: python generate_terraform.py --model_path " + args.output_dir)


if __name__ == "__main__":
    main()

