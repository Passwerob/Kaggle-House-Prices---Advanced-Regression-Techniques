import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

class HousePriceDataset(Dataset):
    """房价数据集类"""
    def __init__(self, features, targets=None):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        else:
            return self.features[idx]

class HousePriceNN(nn.Module):
    """房价预测神经网络"""
    def __init__(self, input_size, hidden_sizes=[512, 256, 128, 64], dropout_rate=0.3):
        super(HousePriceNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class HousePricePredictor:
    """房价预测器主类"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        self.feature_names = None
        
    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        self.train_data = pd.read_csv('train.csv')
        self.test_data = pd.read_csv('test.csv')
        print(f"训练数据形状: {self.train_data.shape}")
        print(f"测试数据形状: {self.test_data.shape}")
        
    def explore_data(self):
        """数据探索"""
        print("\n=== 数据探索 ===")
        print(f"训练数据列数: {len(self.train_data.columns)}")
        print(f"目标变量: SalePrice")
        print(f"目标变量统计:\n{self.train_data['SalePrice'].describe()}")
        
        # 检查缺失值
        missing_train = self.train_data.isnull().sum()
        missing_test = self.test_data.isnull().sum()
        
        print(f"\n训练数据缺失值:\n{missing_train[missing_train > 0]}")
        print(f"\n测试数据缺失值:\n{missing_test[missing_test > 0]}")
        
        # 数据类型分析
        numeric_cols = self.train_data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.train_data.select_dtypes(include=['object']).columns
        
        print(f"\n数值型特征数量: {len(numeric_cols)}")
        print(f"分类型特征数量: {len(categorical_cols)}")
        
    def handle_missing_values(self):
        """处理缺失值"""
        print("\n正在处理缺失值...")
        
        # 合并训练和测试数据以统一处理
        combined_data = pd.concat([self.train_data.drop('SalePrice', axis=1), self.test_data], ignore_index=True)
        
        # 数值型特征的缺失值用中位数填充
        numeric_cols = combined_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if combined_data[col].isnull().sum() > 0:
                median_val = combined_data[col].median()
                combined_data[col].fillna(median_val, inplace=True)
                self.train_data[col].fillna(median_val, inplace=True)
                self.test_data[col].fillna(median_val, inplace=True)
        
        # 分类型特征的缺失值用众数填充
        categorical_cols = combined_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if combined_data[col].isnull().sum() > 0:
                mode_val = combined_data[col].mode()[0]
                combined_data[col].fillna(mode_val, inplace=True)
                self.train_data[col].fillna(mode_val, inplace=True)
                self.test_data[col].fillna(mode_val, inplace=True)
        
        print("缺失值处理完成")
        
    def feature_engineering(self):
        """特征工程"""
        print("\n正在进行特征工程...")
        
        # 合并数据以统一处理
        combined_data = pd.concat([self.train_data.drop('SalePrice', axis=1), self.test_data], ignore_index=True)
        
        # 创建新特征
        # 1. 总面积
        combined_data['TotalSF'] = combined_data['TotalBsmtSF'] + combined_data['1stFlrSF'] + combined_data['2ndFlrSF']
        
        # 2. 浴室总数
        combined_data['TotalBathrooms'] = combined_data['FullBath'] + 0.5 * combined_data['HalfBath'] + \
                                        combined_data['BsmtFullBath'] + 0.5 * combined_data['BsmtHalfBath']
        
        # 3. 房屋年龄
        combined_data['HouseAge'] = combined_data['YrSold'] - combined_data['YearBuilt']
        combined_data['RemodelAge'] = combined_data['YrSold'] - combined_data['YearRemodAdd']
        
        # 4. 质量评分组合
        combined_data['OverallScore'] = combined_data['OverallQual'] * combined_data['OverallCond']
        
        # 5. 外部质量评分
        quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
        for col in ['ExterQual', 'ExterCond', 'HeatingQC', 'KitchenQual']:
            if col in combined_data.columns:
                combined_data[col] = combined_data[col].map(quality_map).fillna(3)
        
        # 6. 地下室质量评分
        basement_quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
        for col in ['BsmtQual', 'BsmtCond']:
            if col in combined_data.columns:
                combined_data[col] = combined_data[col].map(basement_quality_map).fillna(0)
        
        # 7. 车库质量评分
        garage_quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
        for col in ['GarageQual', 'GarageCond']:
            if col in combined_data.columns:
                combined_data[col] = combined_data[col].map(garage_quality_map).fillna(0)
        
        # 8. 功能性评分
        functional_map = {'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod': 4, 'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'Sal': 0}
        combined_data['Functional'] = combined_data['Functional'].map(functional_map).fillna(7)
        
        # 9. 销售条件评分
        sale_condition_map = {'Normal': 5, 'Abnorml': 1, 'AdjLand': 2, 'Alloca': 3, 'Family': 4, 'Partial': 6}
        combined_data['SaleCondition'] = combined_data['SaleCondition'].map(sale_condition_map).fillna(5)
        
        # 10. 邻里评分（基于房价中位数）
        neighborhood_prices = self.train_data.groupby('Neighborhood')['SalePrice'].median().sort_values()
        neighborhood_scores = {neigh: i for i, neigh in enumerate(neighborhood_prices.index)}
        combined_data['NeighborhoodScore'] = combined_data['Neighborhood'].map(neighborhood_scores)
        
        # 更新训练和测试数据
        train_features = combined_data.iloc[:len(self.train_data)]
        test_features = combined_data.iloc[len(self.train_data):]
        
        self.train_data = pd.concat([train_features, self.train_data['SalePrice']], axis=1)
        self.test_data = test_features
        
        print("特征工程完成")
        
    def encode_categorical_features(self):
        """编码分类特征"""
        print("\n正在编码分类特征...")
        
        # 合并数据以统一编码
        combined_data = pd.concat([self.train_data.drop('SalePrice', axis=1), self.test_data], ignore_index=True)
        
        categorical_cols = combined_data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col in combined_data.columns:
                # 创建标签编码器
                le = LabelEncoder()
                # 处理训练数据中的未知类别
                combined_data[col] = combined_data[col].astype(str)
                le.fit(combined_data[col])
                
                # 保存编码器
                self.label_encoders[col] = le
                
                # 应用编码
                self.train_data[col] = le.transform(self.train_data[col].astype(str))
                self.test_data[col] = le.transform(self.test_data[col].astype(str))
        
        print("分类特征编码完成")
        
    def prepare_features(self):
        """准备特征"""
        print("\n正在准备特征...")
        
        # 分离特征和目标
        self.X_train = self.train_data.drop(['Id', 'SalePrice'], axis=1)
        self.y_train = self.train_data['SalePrice']
        self.X_test = self.test_data.drop('Id', axis=1)
        
        # 确保训练和测试数据有相同的列
        common_cols = self.X_train.columns.intersection(self.X_test.columns)
        self.X_train = self.X_train[common_cols]
        self.X_test = self.X_test[common_cols]
        
        # 保存特征名称
        self.feature_names = self.X_train.columns.tolist()
        
        print(f"特征数量: {len(self.feature_names)}")
        print(f"训练特征形状: {self.X_train.shape}")
        print(f"测试特征形状: {self.X_test.shape}")
        
    def scale_features(self):
        """特征标准化"""
        print("\n正在标准化特征...")
        
        # 对训练数据进行拟合和转换
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        # 对测试数据进行转换
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("特征标准化完成")
        
    def create_model(self, input_size):
        """创建神经网络模型"""
        print(f"\n正在创建神经网络模型，输入特征数: {input_size}")
        
        self.model = HousePriceNN(input_size=input_size)
        print(self.model)
        
        return self.model
    
    def train_model(self, epochs=100, batch_size=32, learning_rate=0.001):
        """训练模型"""
        print(f"\n开始训练模型...")
        print(f"训练参数: epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}")
        
        # 创建数据集
        train_dataset = HousePriceDataset(self.X_train_scaled, self.y_train.values)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        self.model.to(device)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        # 训练循环
        self.model.train()
        train_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                
                # 前向传播
                outputs = self.model(batch_features).squeeze()
                loss = criterion(outputs, batch_targets)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # 计算平均损失
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            # 学习率调度
            scheduler.step(avg_loss)
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
        
        print("模型训练完成")
        
        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses)
        plt.title('训练损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return train_losses
    
    def evaluate_model(self):
        """评估模型"""
        print("\n正在评估模型...")
        
        # 在训练集上预测
        self.model.eval()
        with torch.no_grad():
            X_train_tensor = torch.FloatTensor(self.X_train_scaled)
            y_train_pred = self.model(X_train_tensor).squeeze().numpy()
        
        # 计算评估指标
        mse = mean_squared_error(self.y_train, y_train_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_train, y_train_pred)
        r2 = r2_score(self.y_train, y_train_pred)
        
        print(f"训练集评估结果:")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R²: {r2:.4f}")
        
        # 绘制预测vs实际值
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_train, y_train_pred, alpha=0.5)
        plt.plot([self.y_train.min(), self.y_train.max()], [self.y_train.min(), self.y_train.max()], 'r--', lw=2)
        plt.xlabel('实际价格')
        plt.ylabel('预测价格')
        plt.title('预测 vs 实际价格')
        plt.grid(True)
        plt.savefig('prediction_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
    
    def make_predictions(self):
        """进行预测"""
        print("\n正在进行预测...")
        
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(self.X_test_scaled)
            y_test_pred = self.model(X_test_tensor).squeeze().numpy()
        
        # 创建提交文件
        submission = pd.DataFrame({
            'Id': self.test_data['Id'],
            'SalePrice': y_test_pred
        })
        
        submission.to_csv('submission.csv', index=False)
        print("预测完成，提交文件已保存为 'submission.csv'")
        
        return submission
    
    def run_pipeline(self):
        """运行完整的预测流程"""
        print("=== 房价预测深度学习模型 ===")
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 数据探索
        self.explore_data()
        
        # 3. 处理缺失值
        self.handle_missing_values()
        
        # 4. 特征工程
        self.feature_engineering()
        
        # 5. 编码分类特征
        self.encode_categorical_features()
        
        # 6. 准备特征
        self.prepare_features()
        
        # 7. 特征标准化
        self.scale_features()
        
        # 8. 创建模型
        input_size = len(self.feature_names)
        self.create_model(input_size)
        
        # 9. 训练模型
        self.train_model(epochs=150, batch_size=64, learning_rate=0.001)
        
        # 10. 评估模型
        metrics = self.evaluate_model()
        
        # 11. 进行预测
        submission = self.make_predictions()
        
        print("\n=== 流程完成 ===")
        print(f"最终模型性能:")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"R²: {metrics['r2']:.4f}")
        
        return submission

if __name__ == "__main__":
    # 创建预测器实例
    predictor = HousePricePredictor()
    
    # 运行完整流程
    submission = predictor.run_pipeline()
    
    print("\n预测完成！请查看 'submission.csv' 文件进行提交。") 