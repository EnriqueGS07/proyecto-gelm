"""
Script principal interactivo para generar código Terraform.

Este script proporciona una interfaz de línea de comandos interactiva
para generar código Terraform usando modelos pre-entrenados o entrenados.

Uso:
    python main.py                          # Usa modelo pre-entrenado
    python main.py --model_path models/...  # Usa modelo entrenado
    python main.py --model_name gpt2        # Usa modelo específico

Características:
    - Modo interactivo con bucle continuo
    - Opción de guardar código generado en archivos
    - Soporte para modelos pre-entrenados y entrenados
    - Manejo de errores y excepciones

Autor: Proyecto GELM
Fecha: 2024
"""

import argparse
import os
from terraform_generator import TerraformGenerator


def interactive_mode(model_path: str = None, model_name: str = "microsoft/CodeGPT-small-py"):
    """
    Modo interactivo para generar código Terraform.
    
    Este modo permite al usuario generar múltiples recursos Terraform
    en una sesión interactiva. El usuario puede:
    - Ingresar descripciones en lenguaje natural
    - Ver el código generado inmediatamente
    - Guardar el código en archivos .tf
    - Salir en cualquier momento
    
    Args:
        model_path (str, optional): Ruta al modelo entrenado local.
            Si se proporciona, carga el modelo desde esta ruta.
            Si es None, usa el modelo pre-entrenado especificado.
        
        model_name (str): Nombre del modelo pre-entrenado de Hugging Face.
            Solo se usa si model_path es None.
            Por defecto: "microsoft/CodeGPT-small-py"
    
    Comandos especiales:
        - "salir", "exit", "quit": Termina la sesión
        - Cualquier otra entrada: Se trata como descripción para generar código
    
    Ejemplo de uso:
        >>> interactive_mode()
        Descripción del recurso Terraform: Crear un bucket S3
        [Código generado...]
        ¿Guardar en archivo? (s/n): s
        Nombre del archivo: mi_bucket.tf
    """
    
    # Inicializar generador
    print("Inicializando generador de Terraform...")
    if model_path:
        generator = TerraformGenerator()
        generator.load_from_local(model_path)
    else:
        generator = TerraformGenerator(model_name=model_name)
    
    print("\n" + "=" * 60)
    print("Generador de Código Terraform")
    print("=" * 60)
    print("Escribe 'salir' o 'exit' para terminar")
    print("Escribe 'guardar' después de generar para guardar el código")
    print("-" * 60 + "\n")
    
    while True:
        try:
            # Solicitar descripción
            description = input("Descripción del recurso Terraform: ").strip()
            
            if description.lower() in ['salir', 'exit', 'quit']:
                print("¡Hasta luego!")
                break
            
            if not description:
                continue
            
            # Generar código
            print("\nGenerando código Terraform...")
            terraform_code = generator.generate(description)
            
            print("\n" + "=" * 60)
            print("Código Terraform generado:")
            print("=" * 60)
            print(terraform_code)
            print("=" * 60 + "\n")
            
            # Preguntar si quiere guardar
            save = input("¿Guardar en archivo? (s/n): ").strip().lower()
            if save == 's':
                filename = input("Nombre del archivo (default: output.tf): ").strip()
                if not filename:
                    filename = "output.tf"
                if not filename.endswith('.tf'):
                    filename += '.tf'
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(terraform_code)
                print(f"Código guardado en {filename}\n")
            
        except KeyboardInterrupt:
            print("\n\n¡Hasta luego!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    """
    Función principal que parsea argumentos y ejecuta el modo interactivo.
    
    Esta función configura el parser de argumentos y delega la ejecución
    al modo interactivo con los parámetros proporcionados.
    
    Argumentos de línea de comandos:
        --model_path: Ruta al modelo entrenado (opcional)
        --model_name: Nombre del modelo pre-entrenado (por defecto: microsoft/CodeGPT-small-py)
    
    Ejemplo:
        python main.py --model_path models/terraform_generator
        python main.py --model_name gpt2
    """
    parser = argparse.ArgumentParser(
        description="Generador interactivo de código Terraform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py
  python main.py --model_path models/terraform_generator
  python main.py --model_name gpt2
        """
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
        help="Modelo base de Hugging Face a usar (solo si no se especifica --model_path)"
    )
    
    args = parser.parse_args()
    
    interactive_mode(args.model_path, args.model_name)


if __name__ == "__main__":
    main()

