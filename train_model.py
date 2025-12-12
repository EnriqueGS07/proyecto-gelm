"""
Script para entrenar el modelo con archivos de datos
"""
import os
import json
import argparse
from pathlib import Path
from terraform_generator import TerraformGenerator


def load_training_data(data_dir: str) -> list:
    """
    Carga los datos de entrenamiento desde archivos en el directorio
    
    Args:
        data_dir: Directorio que contiene los archivos de entrenamiento
    
    Returns:
        Lista de diccionarios con 'description' y 'terraform_code'
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
    parser = argparse.ArgumentParser(description="Entrenar modelo para generar código Terraform")
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

