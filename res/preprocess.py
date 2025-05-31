import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def preprocess_telco_data(raw_data_path='res/WA_Fn-UseC_-Telco-Customer-Churn.csv',
                          processed_data_path='res/processed_telco_churn.csv'):

    # 1. 检查是否已经存在处理过的数据文件
    if os.path.exists(processed_data_path):
        print(f"检测到已存在的处理后数据文件: {processed_data_path}。直接加载...")
        df_processed = pd.read_csv(processed_data_path)
        print("数据加载成功！")
        print("处理后数据前5行:")
        print(df_processed.head())
        print("\n处理后数据基本信息:")
        df_processed.info()
        return df_processed
    else:
        print(f"未检测到处理后数据文件: {processed_data_path}。开始进行数据预处理...")

        # 2. 检查原始数据文件是否存在
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(f"错误：原始数据文件 '{raw_data_path}' 未找到。请确保文件在正确路径下。")

        print(f"原始数据文件 '{raw_data_path}' 已找到。")
        # 加载原始数据
        df = pd.read_csv(raw_data_path)
        print("原始数据加载成功！")
        print("原始数据前5行:")
        print(df.head())
        print("\n原始数据基本信息:")
        df.info()

        # --- 预处理步骤开始 ---

        # 3. 处理 'TotalCharges' 列
        # 将 'TotalCharges' 转换为数值型，非数值的转换为 NaN
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        # 填充 'TotalCharges' 的缺失值。观察到缺失值对应的 tenure 为0，合理填充为0。
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
        print("\n'TotalCharges' 列处理完成。")

        # 4. 数据类型转换
        # 将 SeniorCitizen 转换为对象类型（分类变量），因为它代表的是分类意义上的0/1
        df['SeniorCitizen'] = df['SeniorCitizen'].astype(object)
        print("\n'SeniorCitizen' 列类型转换完成。")

        # 5. 特征工程
        # 统一 'No phone service' 和 'No internet service' 为 'No'
        df.replace('No phone service', 'No', inplace=True)
        df.replace('No internet service', 'No', inplace=True)

        # 创建新的特征 'ServicesCount'
        service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                        'StreamingTV', 'StreamingMovies', 'MultipleLines']
        df['ServicesCount'] = df[service_cols].apply(lambda row: sum(row == 'Yes'), axis=1)

        # 创建 'HasInternetService' 和 'HasPhoneService'
        df['HasInternetService'] = df['InternetService'].apply(lambda x: 1 if x != 'No' else 0)
        df['HasPhoneService'] = df['PhoneService'].apply(lambda x: 1 if x != 'No' else 0)
        print("\n特征工程完成。")

        # 6. 分类变量编码
        # 确定要进行编码的分类列
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        categorical_cols.remove('customerID') # 移除客户ID，不进行独热编码
        categorical_cols.remove('Churn')      # 移除目标变量

        # 对 'Churn' 列进行标签编码 ('No' -> 0, 'Yes' -> 1)
        df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

        # 对其他分类列进行独热编码
        # drop_first=True 可以避免多重共线性，并减少一列特征
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
        print("\n分类变量编码完成。")


        # 7. 保存处理后的数据
        df_encoded.to_csv(processed_data_path, index=False)
        print(f"\n处理后的数据已保存到: {processed_data_path}")

        print("\n最终预处理后的数据概览 (customerID 保留):")
        print(df_encoded.head())
        # 打印customerID列以确认
        if 'customerID' in df_encoded.columns:
            print("\n'customerID' 列的前5个值:")
            print(df_encoded['customerID'].head())
        df_encoded.info()

        return df_encoded

if __name__ == "__main__":
    try:
        processed_data = preprocess_telco_data(
            raw_data_path='WA_Fn-UseC_-Telco-Customer-Churn.csv',
            processed_data_path='processed_telco_churn.csv'
        )
        print("\n数据预处理流程（保留customerID和gender）成功完成！")

    except FileNotFoundError as e:
        print(e)
        print("请确保 'WA_Fn-UseC_-Telco-Customer-Churn.csv' 文件存在于脚本运行的同一目录下。")
    except Exception as e:
        print(f"数据预处理过程中发生未知错误: {e}")