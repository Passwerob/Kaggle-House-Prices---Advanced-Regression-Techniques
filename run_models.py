#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
房价预测模型运行脚本
提供多种运行选项
"""

import os
import sys
import time
from datetime import datetime

def print_banner():
    """打印欢迎横幅"""
    print("=" * 60)
    print("🏠 房价预测深度学习模型 🏠")
    print("=" * 60)
    print("Kaggle竞赛: House Prices - Advanced Regression Techniques")
    print("=" * 60)

def check_data_files():
    """检查必要的数据文件是否存在"""
    required_files = ['train.csv', 'test.csv']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 缺少必要的数据文件: {', '.join(missing_files)}")
        print("请确保以下文件在当前目录中:")
        for file in required_files:
            print(f"  - {file}")
        return False
    
    print("✅ 数据文件检查通过")
    return True

def run_data_analysis():
    """运行数据分析"""
    print("\n📊 运行数据分析...")
    try:
        import data_analysis
        start_time = time.time()
        data_analysis.analyze_house_prices_data()
        end_time = time.time()
        print(f"✅ 数据分析完成，耗时: {end_time - start_time:.2f}秒")
    except Exception as e:
        print(f"❌ 数据分析失败: {e}")

def run_quick_model():
    """运行快速模型"""
    print("\n🚀 运行快速预测模型...")
    try:
        import quick_test
        start_time = time.time()
        submission, model, feature_importance = quick_test.quick_house_price_prediction()
        end_time = time.time()
        print(f"✅ 快速预测完成，耗时: {end_time - start_time:.2f}秒")
        return True
    except Exception as e:
        print(f"❌ 快速预测失败: {e}")
        return False

def run_deep_learning_model():
    """运行深度学习模型"""
    print("\n🧠 运行深度学习模型...")
    print("⚠️  注意: 深度学习模型训练时间较长，请耐心等待...")
    
    try:
        import house_price_prediction
        start_time = time.time()
        predictor = house_price_prediction.HousePricePredictor()
        submission = predictor.run_pipeline()
        end_time = time.time()
        print(f"✅ 深度学习模型完成，耗时: {end_time - start_time:.2f}秒")
        return True
    except Exception as e:
        print(f"❌ 深度学习模型失败: {e}")
        return False

def show_menu():
    """显示主菜单"""
    print("\n请选择要运行的模型:")
    print("1. 📊 数据分析 (推荐先运行)")
    print("2. 🚀 快速预测模型 (随机森林，快速)")
    print("3. 🧠 深度学习模型 (神经网络，较慢但准确)")
    print("4. 🔄 运行所有模型")
    print("5. 📁 查看生成的文件")
    print("6. ❌ 退出")
    
    while True:
        try:
            choice = input("\n请输入选择 (1-6): ").strip()
            if choice in ['1', '2', '3', '4', '5', '6']:
                return choice
            else:
                print("❌ 无效选择，请输入1-6之间的数字")
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            sys.exit(0)

def list_generated_files():
    """列出生成的文件"""
    print("\n📁 当前目录中的文件:")
    
    # 数据文件
    data_files = ['train.csv', 'test.csv', 'data_description.txt', 'sample_submission.csv']
    print("\n📊 数据文件:")
    for file in data_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024  # KB
            print(f"  ✅ {file} ({size:.1f} KB)")
        else:
            print(f"  ❌ {file} (不存在)")
    
    # 生成的文件
    generated_files = [
        'submission.csv', 'quick_submission.csv',
        'target_correlations.csv', 'missing_values_summary.csv'
    ]
    print("\n📤 生成的文件:")
    for file in generated_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024  # KB
            print(f"  ✅ {file} ({size:.1f} KB)")
        else:
            print(f"  ❌ {file} (不存在)")
    
    # 可视化文件
    viz_files = [
        'target_analysis.png', 'missing_values_analysis.png',
        'correlation_heatmap.png', 'important_features_analysis.png',
        'categorical_features_analysis.png', 'training_loss.png',
        'prediction_vs_actual.png'
    ]
    print("\n🖼️  可视化文件:")
    for file in viz_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024  # KB
            print(f"  ✅ {file} ({size:.1f} KB)")
        else:
            print(f"  ❌ {file} (不存在)")

def main():
    """主函数"""
    print_banner()
    
    # 检查数据文件
    if not check_data_files():
        print("\n❌ 请先下载必要的数据文件后再运行模型")
        return
    
    while True:
        choice = show_menu()
        
        if choice == '1':
            # 数据分析
            run_data_analysis()
            
        elif choice == '2':
            # 快速模型
            success = run_quick_model()
            if success:
                print("\n🎉 快速模型运行成功！")
                print("📁 提交文件: quick_submission.csv")
            
        elif choice == '3':
            # 深度学习模型
            print("\n⚠️  深度学习模型需要较长时间训练...")
            confirm = input("确认继续吗？(y/N): ").strip().lower()
            if confirm in ['y', 'yes']:
                success = run_deep_learning_model()
                if success:
                    print("\n🎉 深度学习模型运行成功！")
                    print("📁 提交文件: submission.csv")
            else:
                print("已取消深度学习模型运行")
            
        elif choice == '4':
            # 运行所有模型
            print("\n🔄 运行所有模型...")
            print("这将需要较长时间，请耐心等待...")
            
            # 1. 数据分析
            run_data_analysis()
            
            # 2. 快速模型
            print("\n" + "="*50)
            success1 = run_quick_model()
            
            # 3. 深度学习模型
            print("\n" + "="*50)
            success2 = run_deep_learning_model()
            
            if success1 and success2:
                print("\n🎉 所有模型运行成功！")
                print("📁 提交文件:")
                print("  - quick_submission.csv (快速模型)")
                print("  - submission.csv (深度学习模型)")
            
        elif choice == '5':
            # 查看文件
            list_generated_files()
            
        elif choice == '6':
            # 退出
            print("\n👋 感谢使用房价预测模型！")
            print("祝您在Kaggle竞赛中取得好成绩！")
            break
        
        # 询问是否继续
        if choice != '6':
            input("\n按回车键继续...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 程序被中断，再见！")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        print("请检查错误信息并重试") 