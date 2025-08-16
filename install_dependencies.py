#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依赖安装脚本
自动安装房价预测模型所需的Python包
"""

import subprocess
import sys
import os

def install_package(package):
    """安装单个包"""
    try:
        print(f"正在安装 {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} 安装成功")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ {package} 安装失败")
        return False

def check_package(package):
    """检查包是否已安装"""
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("🔧 房价预测模型依赖安装脚本")
    print("=" * 50)
    
    # 需要安装的包
    required_packages = [
        "pandas>=1.3.0",
        "numpy>=1.21.0", 
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0"
    ]
    
    # PyTorch相关包（可选，用于深度学习）
    pytorch_packages = [
        "torch>=1.9.0",
        "torchvision>=0.10.0"
    ]
    
    print("📦 检查已安装的包...")
    
    # 检查基础包
    missing_packages = []
    for package in required_packages:
        package_name = package.split('>=')[0]
        if not check_package(package_name):
            missing_packages.append(package)
        else:
            print(f"✅ {package_name} 已安装")
    
    # 检查PyTorch
    pytorch_installed = check_package('torch')
    if pytorch_installed:
        print("✅ PyTorch 已安装")
    else:
        print("❌ PyTorch 未安装")
    
    # 安装缺失的基础包
    if missing_packages:
        print(f"\n📥 需要安装 {len(missing_packages)} 个包:")
        for package in missing_packages:
            print(f"  - {package}")
        
        print("\n开始安装...")
        success_count = 0
        for package in missing_packages:
            if install_package(package):
                success_count += 1
        
        print(f"\n📊 安装结果: {success_count}/{len(missing_packages)} 个包安装成功")
        
        if success_count < len(missing_packages):
            print("⚠️  部分包安装失败，请手动安装")
    else:
        print("\n✅ 所有基础包都已安装")
    
    # 询问是否安装PyTorch
    if not pytorch_installed:
        print("\n🧠 PyTorch 用于深度学习模型")
        print("注意: PyTorch 安装可能需要较长时间")
        
        install_pytorch = input("是否安装 PyTorch? (y/N): ").strip().lower()
        
        if install_pytorch in ['y', 'yes']:
            print("\n开始安装 PyTorch...")
            pytorch_success = 0
            
            for package in pytorch_packages:
                if install_package(package):
                    pytorch_success += 1
            
            if pytorch_success == len(pytorch_packages):
                print("✅ PyTorch 安装成功！深度学习模型可以使用")
            else:
                print("❌ PyTorch 安装失败，深度学习模型无法使用")
        else:
            print("跳过 PyTorch 安装，深度学习模型将无法使用")
    
    # 最终检查
    print("\n🔍 最终检查...")
    
    all_installed = True
    for package in required_packages:
        package_name = package.split('>=')[0]
        if not check_package(package_name):
            print(f"❌ {package_name} 未正确安装")
            all_installed = False
        else:
            print(f"✅ {package_name} 可用")
    
    if pytorch_installed:
        print("✅ PyTorch 可用")
    else:
        print("❌ PyTorch 不可用")
    
    print("\n" + "=" * 50)
    if all_installed:
        print("🎉 所有依赖安装完成！")
        print("现在可以运行房价预测模型了")
        print("\n运行方式:")
        print("1. 快速开始: python quick_test.py")
        print("2. 完整模型: python house_price_prediction.py")
        print("3. 数据分析: python data_analysis.py")
        print("4. 交互式运行: python run_models.py")
    else:
        print("⚠️  部分依赖安装失败")
        print("请检查错误信息并手动安装")
    
    print("=" * 50)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 安装被中断")
    except Exception as e:
        print(f"\n❌ 安装过程出错: {e}")
        print("请检查错误信息并重试") 