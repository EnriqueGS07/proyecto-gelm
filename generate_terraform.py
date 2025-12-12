
import argparse
from terraform_generator import TerraformGenerator


def main():
    parser = argparse.ArgumentParser(description="Generar código Terraform")
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
    

    if args.model_path:
        generator = TerraformGenerator()
        generator.load_from_local(args.model_path)
    else:
        generator = TerraformGenerator(model_name=args.model_name)
    

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
    

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(terraform_code)
        print(f"\nCódigo guardado en {args.output}")


if __name__ == "__main__":
    main()

