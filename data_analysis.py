import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_house_prices_data():
    """房价数据探索性分析"""
    print("=== 房价数据探索性分析 ===")
    
    # 1. 加载数据
    print("正在加载数据...")
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    
    print(f"训练数据形状: {train_data.shape}")
    print(f"测试数据形状: {test_data.shape}")
    
    # 2. 基本信息
    print("\n=== 数据基本信息 ===")
    print(f"训练数据列数: {len(train_data.columns)}")
    print(f"测试数据列数: {len(test_data.columns)}")
    
    # 数据类型分析
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns
    categorical_cols = train_data.select_dtypes(include=['object']).columns
    
    print(f"\n数值型特征数量: {len(numeric_cols)}")
    print(f"分类型特征数量: {len(categorical_cols)}")
    
    # 3. 目标变量分析
    print("\n=== 目标变量分析 ===")
    target = train_data['SalePrice']
    
    print(f"目标变量统计:")
    print(target.describe())
    
    # 检查目标变量分布
    plt.figure(figsize=(15, 5))
    
    # 直方图
    plt.subplot(1, 3, 1)
    plt.hist(target, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('房价分布直方图')
    plt.xlabel('房价 ($)')
    plt.ylabel('频数')
    plt.grid(True, alpha=0.3)
    
    # 箱线图
    plt.subplot(1, 3, 2)
    plt.boxplot(target)
    plt.title('房价箱线图')
    plt.ylabel('房价 ($)')
    plt.grid(True, alpha=0.3)
    
    # Q-Q图
    plt.subplot(1, 3, 3)
    stats.probplot(target, dist="norm", plot=plt)
    plt.title('房价Q-Q图')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('target_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. 缺失值分析
    print("\n=== 缺失值分析 ===")
    
    missing_train = train_data.isnull().sum()
    missing_test = test_data.isnull().sum()
    
    missing_data = pd.DataFrame({
        '训练集缺失': missing_train,
        '测试集缺失': missing_test,
        '训练集缺失率(%)': (missing_train / len(train_data) * 100).round(2),
        '测试集缺失率(%)': (missing_test / len(test_data) * 100).round(2)
    })
    
    missing_data = missing_data[(missing_data['训练集缺失'] > 0) | (missing_data['测试集缺失'] > 0)]
    missing_data = missing_data.sort_values('训练集缺失', ascending=False)
    
    print("缺失值统计:")
    print(missing_data)
    
    # 可视化缺失值
    plt.figure(figsize=(12, 8))
    
    # 训练集缺失值
    plt.subplot(2, 1, 1)
    missing_train_plot = missing_train[missing_train > 0].sort_values(ascending=False)
    plt.bar(range(len(missing_train_plot)), missing_train_plot.values, color='red', alpha=0.7)
    plt.title('训练集缺失值分布')
    plt.xlabel('特征')
    plt.ylabel('缺失值数量')
    plt.xticks(range(len(missing_train_plot)), missing_train_plot.index, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # 测试集缺失值
    plt.subplot(2, 1, 2)
    missing_test_plot = missing_test[missing_test > 0].sort_values(ascending=False)
    plt.bar(range(len(missing_test_plot)), missing_test_plot.values, color='blue', alpha=0.7)
    plt.title('测试集缺失值分布')
    plt.xlabel('特征')
    plt.ylabel('缺失值数量')
    plt.xticks(range(len(missing_test_plot)), missing_test_plot.index, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('missing_values_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. 数值特征相关性分析
    print("\n=== 数值特征相关性分析 ===")
    
    # 选择数值特征
    numeric_features = train_data.select_dtypes(include=[np.number]).columns
    correlation_matrix = train_data[numeric_features].corr()
    
    # 与目标变量的相关性
    target_correlations = correlation_matrix['SalePrice'].sort_values(ascending=False)
    print("与房价相关性最高的特征:")
    print(target_correlations.head(10))
    
    # 相关性热力图
    plt.figure(figsize=(16, 12))
    
    # 选择相关性较高的特征
    high_corr_features = target_correlations[abs(target_correlations) > 0.3].index
    high_corr_matrix = correlation_matrix.loc[high_corr_features, high_corr_features]
    
    sns.heatmap(high_corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('高相关性特征热力图')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. 重要特征分析
    print("\n=== 重要特征分析 ===")
    
    # 分析最重要的几个特征
    top_features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'YearBuilt', 'GarageCars']
    
    plt.figure(figsize=(20, 12))
    
    for i, feature in enumerate(top_features):
        if feature in train_data.columns:
            plt.subplot(2, 3, i+1)
            
            if train_data[feature].dtype in ['int64', 'float64']:
                plt.scatter(train_data[feature], train_data['SalePrice'], alpha=0.6, color='blue')
                plt.xlabel(feature)
                plt.ylabel('房价 ($)')
                plt.title(f'{feature} vs 房价')
                plt.grid(True, alpha=0.3)
            else:
                # 分类特征
                train_data.boxplot(column='SalePrice', by=feature, ax=plt.gca())
                plt.title(f'{feature} vs 房价')
                plt.suptitle('')  # 移除自动标题
        
        plt.tight_layout()
    
    plt.savefig('important_features_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. 分类特征分析
    print("\n=== 分类特征分析 ===")
    
    # 选择一些重要的分类特征
    important_categorical = ['MSZoning', 'Neighborhood', 'BldgType', 'HouseStyle', 'OverallQual']
    
    plt.figure(figsize=(20, 12))
    
    for i, feature in enumerate(important_categorical):
        if feature in train_data.columns:
            plt.subplot(2, 3, i+1)
            
            # 计算每个类别的平均房价
            avg_prices = train_data.groupby(feature)['SalePrice'].mean().sort_values(ascending=False)
            
            plt.bar(range(len(avg_prices)), avg_prices.values, color='lightcoral', alpha=0.7)
            plt.title(f'{feature} - 平均房价')
            plt.xlabel(feature)
            plt.ylabel('平均房价 ($)')
            plt.xticks(range(len(avg_prices)), avg_prices.index, rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('categorical_features_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 8. 数据质量报告
    print("\n=== 数据质量报告 ===")
    
    print("数据概览:")
    print(f"- 训练样本数: {len(train_data):,}")
    print(f"- 测试样本数: {len(test_data):,}")
    print(f"- 总特征数: {len(train_data.columns)}")
    print(f"- 数值特征数: {len(numeric_cols)}")
    print(f"- 分类特征数: {len(categorical_cols)}")
    
    print(f"\n目标变量统计:")
    print(f"- 平均房价: ${target.mean():,.2f}")
    print(f"- 房价中位数: ${target.median():,.2f}")
    print(f"- 房价标准差: ${target.std():,.2f}")
    print(f"- 最低房价: ${target.min():,.2f}")
    print(f"- 最高房价: ${target.max():,.2f}")
    
    print(f"\n缺失值统计:")
    print(f"- 训练集总缺失值: {missing_train.sum():,}")
    print(f"- 测试集总缺失值: {missing_test.sum():,}")
    print(f"- 有缺失值的特征数: {len(missing_data)}")
    
    # 9. 保存分析结果
    print("\n=== 保存分析结果 ===")
    
    # 保存相关性数据
    target_correlations.to_csv('target_correlations.csv')
    print("目标变量相关性已保存到 'target_correlations.csv'")
    
    # 保存缺失值数据
    missing_data.to_csv('missing_values_summary.csv')
    print("缺失值统计已保存到 'missing_values_summary.csv'")
    
    print("\n数据探索分析完成！")
    print("生成的可视化文件:")
    print("- target_analysis.png: 目标变量分析")
    print("- missing_values_analysis.png: 缺失值分析")
    print("- correlation_heatmap.png: 相关性热力图")
    print("- important_features_analysis.png: 重要特征分析")
    print("- categorical_features_analysis.png: 分类特征分析")

if __name__ == "__main__":
    # 运行数据分析
    analyze_house_prices_data() 