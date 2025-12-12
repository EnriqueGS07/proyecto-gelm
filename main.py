
import argparse
import os
from terraform_generator import TerraformGenerator


def interactive_mode(model_path: str = None, model_name: str = "microsoft/CodeGPT-small-py"):
    
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
            description = input("Descripción del recurso Terraform: ").strip()
            
            if description.lower() in ['salir', 'exit', 'quit']:
                print("¡Hasta luego!")
                break
            
            if not description:
                continue
            
            print("\nGenerando código Terraform...")
            terraform_code = generator.generate(description)
            
            print("\n" + "=" * 60)
            print("Código Terraform generado:")
            print("=" * 60)
            print(terraform_code)
            print("=" * 60 + "\n")
            
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
    parser = argparse.ArgumentParser(description="Generador interactivo de código Terraform")
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

