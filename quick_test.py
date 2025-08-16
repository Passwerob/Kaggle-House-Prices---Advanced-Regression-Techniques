import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def quick_house_price_prediction():
    """快速房价预测函数"""
    print("=== 快速房价预测模型 ===")
    
    # 1. 加载数据
    print("正在加载数据...")
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    print(f"训练数据: {train_data.shape}, 测试数据: {test_data.shape}")
    
    # 2. 简单特征工程
    print("正在进行特征工程...")
    
    # 合并数据以统一处理
    combined_data = pd.concat([train_data.drop('SalePrice', axis=1), test_data], ignore_index=True)
    
    # 创建一些重要特征
    combined_data['TotalSF'] = combined_data['TotalBsmtSF'] + combined_data['1stFlrSF'] + combined_data['2ndFlrSF']
    combined_data['TotalBathrooms'] = combined_data['FullBath'] + 0.5 * combined_data['HalfBath']
    combined_data['HouseAge'] = combined_data['YrSold'] - combined_data['YearBuilt']
    combined_data['OverallScore'] = combined_data['OverallQual'] * combined_data['OverallCond']
    
    # 3. 处理缺失值
    print("正在处理缺失值...")
    
    # 数值型特征用中位数填充
    numeric_cols = combined_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if combined_data[col].isnull().sum() > 0:
            median_val = combined_data[col].median()
            combined_data[col].fillna(median_val, inplace=True)
    
    # 分类型特征用众数填充
    categorical_cols = combined_data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if combined_data[col].isnull().sum() > 0:
            mode_val = combined_data[col].mode()[0]
            combined_data[col].fillna(mode_val, inplace=True)
    
    # 4. 编码分类特征
    print("正在编码分类特征...")
    
    for col in categorical_cols:
        if col in combined_data.columns:
            le = LabelEncoder()
            combined_data[col] = combined_data[col].astype(str)
            combined_data[col] = le.fit_transform(combined_data[col])
    
    # 5. 准备特征
    print("正在准备特征...")
    
    # 选择重要特征
    important_features = [
        'TotalSF', 'TotalBathrooms', 'HouseAge', 'OverallScore',
        'OverallQual', 'OverallCond', 'YearBuilt', 'GrLivArea',
        'GarageCars', 'GarageArea', 'FullBath', 'BedroomAbvGr',
        'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'LotArea',
        'Neighborhood', 'MSZoning', 'BldgType', 'HouseStyle'
    ]
    
    # 确保所有特征都存在
    available_features = [col for col in important_features if col in combined_data.columns]
    
    # 分离训练和测试数据
    train_features = combined_data.iloc[:len(train_data)][available_features]
    test_features = combined_data.iloc[len(train_data):][available_features]
    
    # 6. 特征标准化
    print("正在标准化特征...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_features)
    X_test_scaled = scaler.transform(test_features)
    
    # 7. 训练随机森林模型（快速且有效）
    print("正在训练随机森林模型...")
    
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train_scaled, train_data['SalePrice'])
    
    # 8. 评估模型
    print("正在评估模型...")
    
    y_train_pred = rf_model.predict(X_train_scaled)
    mse = mean_squared_error(train_data['SalePrice'], y_train_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(train_data['SalePrice'], y_train_pred)
    
    print(f"模型性能:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    
    # 9. 特征重要性
    print("\n前10个最重要特征:")
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10))
    
    # 10. 进行预测
    print("\n正在进行预测...")
    
    y_test_pred = rf_model.predict(X_test_scaled)
    
    # 创建提交文件
    submission = pd.DataFrame({
        'Id': test_data['Id'],
        'SalePrice': y_test_pred
    })
    
    submission.to_csv('quick_submission.csv', index=False)
    print("预测完成！提交文件已保存为 'quick_submission.csv'")
    
    return submission, rf_model, feature_importance

if __name__ == "__main__":
    # 运行快速预测
    submission, model, feature_importance = quick_house_price_prediction()
    
    print("\n=== 快速预测完成 ===")
    print("您可以使用 'quick_submission.csv' 文件提交到Kaggle竞赛。")
    print("或者运行完整的深度学习模型: python house_price_prediction.py") 