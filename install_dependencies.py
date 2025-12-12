"""
Script de instalación que maneja conflictos de dependencias
"""
import subprocess
import sys


def install_package(package, upgrade=False):
    """Instala un paquete con manejo de errores"""
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.append(package)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ {package} instalado correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error al instalar {package}: {e.stderr}")
        return False


def main():
    print("Instalando dependencias del proyecto...")
    print("=" * 60)
    
    # Instalar paquetes básicos primero
    basic_packages = [
        "python-dotenv>=1.0.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0,<5.0.0"
    ]
    
    print("\n1. Instalando paquetes básicos...")
    for package in basic_packages:
        install_package(package)
    
    # Instalar Hugging Face ecosystem
    print("\n2. Instalando ecosistema de Hugging Face...")
    hf_packages = [
        "huggingface-hub>=0.16.0",
        "transformers>=4.30.0,<5.0.0",
        "datasets>=2.12.0",
        "accelerate>=0.20.0"
    ]
    
    for package in hf_packages:
        install_package(package, upgrade=True)
    
    # Instalar LangChain
    print("\n3. Instalando LangChain...")
    langchain_packages = [
        "langchain>=0.1.0,<0.3.0",
        "langchain-community>=0.0.10"
    ]
    
    for package in langchain_packages:
        install_package(package, upgrade=True)
    
    # PyTorch (opcional)
    print("\n4. PyTorch (opcional)...")
    print("   Si tienes GPU NVIDIA, instala PyTorch con CUDA:")
    print("   pip install torch --index-url https://download.pytorch.org/whl/cu118")
    print("   O para CPU solamente:")
    print("   pip install torch")
    
    response = input("\n¿Instalar PyTorch ahora? (s/n): ").strip().lower()
    if response == 's':
        install_package("torch")
    
    # PEFT (opcional, para fine-tuning avanzado)
    print("\n5. PEFT (opcional, para fine-tuning avanzado)...")
    response = input("¿Instalar PEFT? (s/n): ").strip().lower()
    if response == 's':
        install_package("peft>=0.4.0")
    
    print("\n" + "=" * 60)
    print("Instalación completada!")
    print("=" * 60)
    print("\nPuedes verificar la instalación ejecutando:")
    print("  python -c 'from terraform_generator import TerraformGenerator; print(\"OK\")'")


if __name__ == "__main__":
    main()

