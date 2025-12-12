"""
Script para generar código Terraform desde la línea de comandos.

Este script permite generar código Terraform de forma no interactiva,
útil para automatización, scripts y pipelines CI/CD.

Uso básico:
    python generate_terraform.py --prompt "Crear un bucket S3"

Uso con modelo entrenado:
    python generate_terraform.py --prompt "Crear EC2" --model_path models/terraform_generator

Guardar en archivo:
    python generate_terraform.py --prompt "Crear S3" --output s3.tf

Con contexto:
    python generate_terraform.py --prompt "Crear SG" --context "Para VPC vpc-123"

Autor: Proyecto GELM
Fecha: 2024
"""

import argparse
from terraform_generator import TerraformGenerator


def main():
    """
    Función principal que parsea argumentos y genera código Terraform.
    
    Este script está diseñado para uso no interactivo, ideal para:
    - Scripts de automatización
    - Pipelines CI/CD
    - Integración con otras herramientas
    - Generación batch de código
    
    Argumentos requeridos:
        --prompt: Descripción del recurso a crear (requerido)
    
    Argumentos opcionales:
        --model_path: Ruta al modelo entrenado
        --model_name: Nombre del modelo pre-entrenado
        --output: Archivo donde guardar el código generado
        --context: Contexto adicional para la generación
    
    Ejemplo:
        python generate_terraform.py \\
            --prompt "Crear un bucket S3 con versionado" \\
            --model_path models/terraform_generator \\
            --output s3_bucket.tf
    """
    parser = argparse.ArgumentParser(
        description="Generar código Terraform desde línea de comandos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Uso básico
  python generate_terraform.py --prompt "Crear un bucket S3"
  
  # Con modelo entrenado
  python generate_terraform.py --prompt "Crear EC2" --model_path models/terraform_generator
  
  # Guardar en archivo
  python generate_terraform.py --prompt "Crear S3" --output s3.tf
  
  # Con contexto
  python generate_terraform.py --prompt "Crear grupo de seguridad" \\
      --context "Para una instancia EC2 en la VPC vpc-12345"
        """
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Descripción de lo que se quiere crear en Terraform"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Ruta al modelo entrenado (si no se especifica, usa modelo pre-entrenado)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/CodeGPT-small-py",
        help="Modelo base de Hugging Face a usar (si no se especifica model_path)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Archivo donde guardar el código generado (opcional)"
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Contexto adicional para la generación"
    )
    
    args = parser.parse_args()
    
    # Inicializar generador
    if args.model_path:
        generator = TerraformGenerator()
        generator.load_from_local(args.model_path)
    else:
        generator = TerraformGenerator(model_name=args.model_name)
    
    # Generar código
    print(f"Generando código Terraform para: {args.prompt}")
    print("-" * 60)
    
    if args.context:
        terraform_code = generator.generate_with_context(args.prompt, args.context)
    else:
        terraform_code = generator.generate(args.prompt)
    
    print("\nCódigo Terraform generado:")
    print("=" * 60)
    print(terraform_code)
    print("=" * 60)
    
    # Guardar en archivo si se especifica
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(terraform_code)
        print(f"\nCódigo guardado en {args.output}")


if __name__ == "__main__":
    main()

