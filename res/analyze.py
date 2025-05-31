from pyecharts import options as opts
from pyecharts.charts import Bar, Pie, Line, Boxplot, Scatter, HeatMap
from pyecharts.globals import ThemeType
import pandas as pd
import numpy as np
import json

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --- 仪表盘1: 探索性分析 (8 图) ---

def create_pie_churn_overview(df):
    counts = df['Churn'].value_counts()
    data = [("未流失", int(counts.get(0, 0))), ("已流失", int(counts.get(1, 0)))]
    pie = Pie(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
    pie.add("", data, radius=["40%", "75%"], label_opts=opts.LabelOpts(formatter="{b}: {d}% ({c} 人)"))
    pie.set_global_opts(
        legend_opts=opts.LegendOpts(orient="vertical", pos_left="10%", pos_bottom="10%")
    )
    return json.loads(pie.dump_options())


def create_services_heatmap(df):
    # 服务名称映射到中文
    service_names = {
        'OnlineSecurity_Yes': '在线安全',
        'OnlineBackup_Yes': '在线备份',
        'DeviceProtection_Yes': '设备保护',
        'TechSupport_Yes': '技术支持',
        'StreamingTV_Yes': '流媒体电视',
        'StreamingMovies_Yes': '流媒体电影'
    }

    # 使用中文服务名称
    services = list(service_names.values())
    service_keys = list(service_names.keys())

    # 计算每个服务的流失率
    data = []
    print("\n服务流失率分析:")
    for i, service_key in enumerate(service_keys):
        # 检查服务列是否存在且有效
        if service_key not in df.columns:
            print(f"警告: 数据集中缺少服务列 {service_key}")
            continue

        # 计算使用该服务的客户数量
        service_users = df[service_key].sum()
        if service_users == 0:
            print(f"警告: 没有客户使用服务 '{service_names[service_key]}'")
            churn_rate = 0
        else:
            # 计算使用该服务的客户的流失率
            churn_rate = df[df[service_key] == 1]['Churn'].mean() * 100

        # 添加到数据 - 关键修复：每个服务一个独立位置
        data.append([i, 0, round(churn_rate, 2)])

        # 打印调试信息
        print(f"- {service_names[service_key]}: 使用人数={service_users}, 流失率={round(churn_rate, 2)}%")

    # 创建热力图
    heatmap = HeatMap(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))

    # 关键修复：交换X轴和Y轴
    heatmap.add_xaxis(services)  # X轴显示服务名称
    heatmap.add_yaxis('流失率(%)', ['流失率'], data)

    # 设置全局选项
    heatmap.set_global_opts(
        visualmap_opts=opts.VisualMapOpts(
            min_=0,
            max_=100,
            is_calculable=True,
            orient="vertical",
            pos_left="right",
            pos_top="middle",
            range_text=['高流失率', '低流失率']
        )
    )

    return json.loads(heatmap.dump_options())


def create_contract_grouped_bar(df):
    # 确定合同类型
    df['Contract'] = np.select(
        [df['Contract_One year'] == 1, df['Contract_Two year'] == 1],
        ['一年合同', '两年合同'],
        default='月合同'
    )

    # 按合同类型和流失状态分组
    grp = df.groupby(['Contract', 'Churn']).size().unstack(fill_value=0)
    contracts = grp.index.tolist()

    bar = Bar(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
    bar.add_xaxis(contracts)
    bar.add_yaxis('未流失', grp[0].tolist())
    bar.add_yaxis('已流失', grp[1].tolist())

    bar.set_global_opts(
        legend_opts=opts.LegendOpts(pos_top='5%'),
        yaxis_opts=opts.AxisOpts(name='客户数')
    )
    return json.loads(bar.dump_options())


def create_tenure_histogram(df):
    # 创建在网时长分箱
    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = ['0-12月', '13-24月', '25-36月', '37-48月', '49-60月', '60+月']
    df['TenureGroup'] = pd.cut(df['tenure'], bins=bins, labels=labels, right=False)

    # 分组统计
    tenure_counts = df['TenureGroup'].value_counts().sort_index()

    bar = Bar(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
    bar.add_xaxis(tenure_counts.index.tolist())
    bar.add_yaxis('客户数', tenure_counts.values.tolist())

    bar.set_global_opts(
        xaxis_opts=opts.AxisOpts(name='在网时长'),
        yaxis_opts=opts.AxisOpts(name='客户数')
    )
    return json.loads(bar.dump_options())


def create_payment_bar(df):
    # 确定支付方式
    payment_methods = {
        'PaymentMethod_Credit card (automatic)': '信用卡(自动)',
        'PaymentMethod_Electronic check': '电子支票',
        'PaymentMethod_Mailed check': '邮寄支票'
    }

    # 计算每种支付方式的流失率
    data = []
    for col, name in payment_methods.items():
        payment_df = df[df[col] == 1]
        if len(payment_df) > 0:
            churn_rate = payment_df['Churn'].mean() * 100
            data.append((name, round(churn_rate, 2)))

    # 按流失率排序
    data.sort(key=lambda x: x[1], reverse=True)
    methods, rates = zip(*data)

    bar = Bar(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
    bar.add_xaxis(list(methods))
    bar.add_yaxis('流失率 (%)', list(rates))

    bar.set_global_opts(
        yaxis_opts=opts.AxisOpts(name='流失率 (%)')
    )
    return json.loads(bar.dump_options())


def create_churn_trend(df):
    # 按在网时长分组计算流失率
    tenure_groups = df.groupby(pd.cut(df['tenure'], bins=12))
    churn_rates = tenure_groups['Churn'].mean() * 100

    # 准备数据
    x_axis = [f"{int(group.left)}-{int(group.right)}月" for group in churn_rates.index]
    y_axis = churn_rates.values.round(2).tolist()

    line = Line(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
    line.add_xaxis(x_axis)
    line.add_yaxis('流失率', y_axis, is_smooth=True)

    line.set_global_opts(
        xaxis_opts=opts.AxisOpts(name='在网时长(月)'),
        yaxis_opts=opts.AxisOpts(name='流失率 (%)'),
        tooltip_opts=opts.TooltipOpts(trigger="axis")
    )
    return json.loads(line.dump_options())


# --- 仪表盘2: 深度驱动因素分析 (6 图) ---

def create_service_combo_churn_bar(df):
    # 创建服务组合特征
    df['ServiceCombo'] = df.apply(lambda row:
                                  f"安全:{row['OnlineSecurity_Yes']}, 备份:{row['OnlineBackup_Yes']}, 设备:{row['DeviceProtection_Yes']}",
                                  axis=1
                                  )

    # 计算每个组合的流失率
    combo_churn = df.groupby('ServiceCombo')['Churn'].mean().reset_index()
    combo_churn['ChurnRate'] = combo_churn['Churn'] * 100
    combo_churn = combo_churn.sort_values('ChurnRate', ascending=False).head(10)

    bar = Bar(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
    bar.add_xaxis(combo_churn['ServiceCombo'].tolist())
    bar.add_yaxis('流失率 (%)', combo_churn['ChurnRate'].round(2).tolist())

    bar.set_global_opts(
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=30)),
        yaxis_opts=opts.AxisOpts(name='流失率 (%)')
    )
    return json.loads(bar.dump_options())


def create_scatter_tenure_charges(df):
    # 抽样并按在网时长排序
    sample_df = df[['tenure', 'MonthlyCharges', 'Churn']].sample(n=500).sort_values('tenure')

    data = sample_df.values.tolist()

    # 创建散点图
    scatter = Scatter(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))

    # 添加X轴数据（在网时长）
    scatter.add_xaxis([d[0] for d in data])

    # 按流失状态分组添加数据点
    for churn_status in [0, 1]:
        churn_data = [d for d in data if d[2] == churn_status]
        name = '未流失' if churn_status == 0 else '已流失'
        scatter.add_yaxis(
            name,
            [d[1] for d in churn_data],
            symbol_size=8,
            label_opts=opts.LabelOpts(is_show=False)
        )

    # 正确位置：全局选项设置应放在循环外部
    scatter.set_global_opts(
        xaxis_opts=opts.AxisOpts(
            name='在网时长(月)',
            type_='value',  # 使用数值轴而非分类轴
            min_=0,
            max_=sample_df['tenure'].max() * 1.1
        ),
        yaxis_opts=opts.AxisOpts(
            name='月费',
            type_='value',
            min_=0,
            max_=sample_df['MonthlyCharges'].max() * 1.1
        ),
        datazoom_opts=[opts.DataZoomOpts()]  # 添加缩放功能
    )

    # 调试信息
    print("图表配置完成，X轴类型:", scatter.options['xAxis'][0]['type'])

    return json.loads(scatter.dump_options())


def create_senior_bar(df):
    # 按老年人分组计算流失率
    senior_churn = df.groupby('SeniorCitizen_1')['Churn'].mean() * 100
    senior_churn.index = senior_churn.index.map({0: '非老年人', 1: '老年人'})

    bar = Bar(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
    bar.add_xaxis(senior_churn.index.tolist())
    bar.add_yaxis('流失率 (%)', senior_churn.round(2).tolist())

    bar.set_global_opts(
        yaxis_opts=opts.AxisOpts(name='流失率 (%)')
    )
    return json.loads(bar.dump_options())


def create_churn_services_bar(df):
    # 流失客户的服务使用情况
    lost_customers = df[df['Churn'] == 1]
    services = ['OnlineSecurity_Yes', 'OnlineBackup_Yes', 'DeviceProtection_Yes',
                'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes']

    service_usage = {}
    for service in services:
        service_name = service.split('_')[0]
        usage_rate = lost_customers[service].mean() * 100
        service_usage[service_name] = round(usage_rate, 2)

    # 按使用率排序
    sorted_services = sorted(service_usage.items(), key=lambda x: x[1], reverse=True)
    service_names, usage_rates = zip(*sorted_services)

    bar = Bar(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
    bar.add_xaxis(list(service_names))
    bar.add_yaxis('使用率 (%)', list(usage_rates))

    bar.set_global_opts(
        yaxis_opts=opts.AxisOpts(name='使用率 (%)')
    )
    return json.loads(bar.dump_options())


def create_contract_billing_stacked(df):
    # 合同类型
    df['Contract'] = np.select(
        [df['Contract_One year'] == 1, df['Contract_Two year'] == 1],
        ['一年合同', '两年合同'],
        default='月合同'
    )

    # 账单方式
    df['Billing'] = np.where(df['PaperlessBilling_Yes'] == 1, '电子账单', '纸质账单')

    # 分组计算流失率
    grouped = df.groupby(['Contract', 'Billing'])['Churn'].mean().unstack().fillna(0) * 100

    bar = Bar(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
    bar.add_xaxis(grouped.index.tolist())

    for col in grouped.columns:
        bar.add_yaxis(col, grouped[col].round(2).tolist())

    bar.set_global_opts(
        yaxis_opts=opts.AxisOpts(name='流失率 (%)'),
        legend_opts=opts.LegendOpts(pos_top='5%')
    )
    return json.loads(bar.dump_options())


def create_customer_matrix(df):
    # 创建客户价值矩阵 (基于在网时长和总消费)
    df['TenureGroup'] = pd.qcut(df['tenure'], q=3, labels=['低忠诚', '中忠诚', '高忠诚'])
    df['SpendGroup'] = pd.qcut(df['TotalCharges'], q=3, labels=['低消费', '中消费', '高消费'])

    # 计算每个分组的流失率
    matrix = df.groupby(['TenureGroup', 'SpendGroup'])['Churn'].mean().unstack().fillna(0) * 100

    # 准备热力图数据
    data = []
    for i, tenure_group in enumerate(matrix.index):
        for j, spend_group in enumerate(matrix.columns):
            churn_rate = matrix.loc[tenure_group, spend_group]
            data.append([j, i, round(churn_rate, 2)])

    heatmap = HeatMap(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
    heatmap.add_xaxis(matrix.columns.tolist())
    heatmap.add_yaxis('流失率 (%)', matrix.index.tolist(), data)

    heatmap.set_global_opts(
        visualmap_opts=opts.VisualMapOpts(min_=0, max_=100, orient="vertical", pos_left="left", pos_top="middle"),
        xaxis_opts=opts.AxisOpts(name='消费水平'),
        yaxis_opts=opts.AxisOpts(name='忠诚度')
    )
    return json.loads(heatmap.dump_options())


# --- 仪表盘3: 模型与解释 (6 图) ---

# 全局变量用于存储模型和评估结果
model = None
model_metrics = {}
roc_data = {}
feature_importances = []
conf_matrix = None
y_pred_proba = None

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np



def train_churn_model(df):
    """
    使用 Pipeline 对输入 DataFrame 训练随机森林流失预测模型，并计算各类评估指标、ROC 等。
    要求 df 包含列 ['customerID', ..., 'Churn']，其中 'Churn' 为 0/1 标签。
    全局会设置并返回以下内容（可在函数外部直接访问）：
        model                - 训练好的 Pipeline 模型
        model_metrics        - 包含 accuracy、precision、recall、f1 的字典
        roc_data             - {'fpr', 'tpr', 'auc'} 用于绘制 ROC 曲线
        feature_importances  - 按编码后特征顺序排列的特征重要性数组
        feature_names        - 对应 feature_importances 的特征名称列表
        conf_matrix          - 混淆矩阵 ndarray
        y_pred_proba         - 测试集上对 “流失为 1” 的预测概率 ndarray
    返回值：feature_names, feature_importances
    """
    global model, model_metrics, roc_data, feature_importances, feature_names, conf_matrix, y_pred_proba

    # 1. 检查输入 DataFrame 格式
    if not isinstance(df, pd.DataFrame):
        raise TypeError("train_churn_model: 输入 df 必须是 pandas DataFrame。")
    if 'Churn' not in df.columns or 'customerID' not in df.columns:
        raise KeyError("train_churn_model: DataFrame 必须包含 'customerID' 和 'Churn' 列。")

    # 2. 分离特征和目标
    X = df.drop(columns=['Churn', 'customerID'])
    y = df['Churn']

    # 3. 检查缺失值
    if X.isnull().any().any():
        missing = X.columns[X.isnull().any()].tolist()
        raise ValueError(f"train_churn_model: 特征 X 存在缺失值，请先处理。列：{missing}")
    if y.isnull().any():
        raise ValueError("train_churn_model: 目标列 y 存在缺失值，请先处理。")

    # 4. 自动识别分类特征和数值特征
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # 5. 构造 ColumnTransformer：分类列用 OneHotEncoder，数值列 passthrough
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ]
    )

    # 6. 创建 Pipeline，将预处理和模型串联
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # 7. 将数据拆分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 8. 用 Pipeline 训练模型（此时会先对分类变量做 OneHot，再训 RandomForest）
    model_pipeline.fit(X_train, y_train)
    model = model_pipeline  # 将训练好的 Pipeline 赋给全局变量

    # 9. 在测试集上做预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # 流失 (Churn=1) 的概率
    # 保存全局变量 y_pred_proba
    globals()['y_pred_proba'] = y_pred_proba

    # 10. 计算分类评估指标
    model_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }
    globals()['model_metrics'] = model_metrics

    # 11. 计算 ROC 曲线数据
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    roc_data = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
    globals()['roc_data'] = roc_data

    # 12. 提取编码后所有特征名称，以便后续与 feature_importances 对应
    #    先从 Pipeline 中取出 OneHotEncoder 对象
    ohe = model_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    # 如果有分类特征，生成编码后特征名；否则为空列表
    if categorical_features:
        # get_feature_names_out 在新版 sklearn 中可直接使用
        try:
            encoded_cat_names = ohe.get_feature_names_out(categorical_features).tolist()
        except AttributeError:
            # 兼容旧版，使用 get_feature_names
            encoded_cat_names = ohe.get_feature_names(categorical_features).tolist()
    else:
        encoded_cat_names = []

    # 数值特征名称保持不变
    passthrough_names = numerical_features

    # 合并得到完整的特征名称列表
    feature_names = encoded_cat_names + passthrough_names
    globals()['feature_names'] = feature_names

    # 13. 获取训练好模型的特征重要性（基于 RandomForestClassifier）
    clf = model_pipeline.named_steps['classifier']
    feature_importances = clf.feature_importances_
    globals()['feature_importances'] = feature_importances

    # 14. 混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    globals()['conf_matrix'] = conf_matrix

    # 15. 返回特征名称和重要性
    return feature_names, feature_importances, model


def create_model_metrics_bar(df):
    """使用真实模型性能指标"""
    global model_metrics

    if not model_metrics:
        train_churn_model(df)

    metrics = ['准确率', '精确率', '召回率', 'F1分数']
    values = [
        model_metrics['accuracy'] * 100,
        model_metrics['precision'] * 100,
        model_metrics['recall'] * 100,
        model_metrics['f1'] * 100
    ]

    bar = Bar(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
    bar.add_xaxis(metrics)
    bar.add_yaxis('值', [round(v, 1) for v in values])

    bar.set_global_opts(
        yaxis_opts=opts.AxisOpts(name='百分比 (%)', min_=0, max_=100)
    )
    return json.loads(bar.dump_options())


def create_roc_curve(df):
    """使用真实ROC曲线数据"""
    global roc_data

    if not roc_data:
        train_churn_model(df)

    line = Line(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
    line.add_xaxis(roc_data['fpr'].tolist())
    line.add_yaxis('ROC曲线', roc_data['tpr'].tolist(), is_smooth=True,
                   linestyle_opts=opts.LineStyleOpts(width=2),
                   label_opts=opts.LabelOpts(is_show=False))
    line.add_xaxis([0, 1])
    line.add_yaxis('随机猜测', [0, 1], linestyle_opts=opts.LineStyleOpts(type_='dashed'))

    line.set_global_opts(
        title_opts=opts.TitleOpts(title=f'AUC={roc_data["auc"]:.2f}'),
        xaxis_opts=opts.AxisOpts(name='假正率(FPR)'),
        yaxis_opts=opts.AxisOpts(name='真正率(TPR)'),
        tooltip_opts=opts.TooltipOpts(trigger="axis")
    )
    return json.loads(line.dump_options())


def create_feature_importance_bar(df):
    """
    使用训练好的全局特征名称和特征重要性，绘制前 10 个最重要特征的柱状图。
    如果全局变量尚未初始化或长度不匹配，则先调用 train_churn_model(df) 进行训练。
    返回值：pyecharts 生成的 JSON dict，可直接用于前端渲染。
    """
    global feature_names, feature_importances

    # 1. 检查全局特征和权重是否已经存在且长度匹配
    valid_globals = False
    if 'feature_names' in globals() and 'feature_importances' in globals():
        fn = globals().get('feature_names')
        fi = globals().get('feature_importances')
        if isinstance(fn, list) and isinstance(fi, np.ndarray) and len(fn) == fi.shape[0]:
            valid_globals = True

    # 2. 如果不存在或长度不匹配，就先训练模型以填充这两个全局变量
    if not valid_globals:
        # train_churn_model 会更新全局 feature_names、feature_importances 等
        train_churn_model(df)
        # 再次检查
        fn = globals().get('feature_names')
        fi = globals().get('feature_importances')
        if not (isinstance(fn, list) and isinstance(fi, np.ndarray) and len(fn) == fi.shape[0]):
            raise RuntimeError("create_feature_importance_bar: 训练后全球特征名称与重要性长度仍不匹配。")

    # 3. 从全局读取特征名称和重要性
    feature_names = globals()['feature_names']
    feature_importances = globals()['feature_importances']

    # 4. 对特征按重要性进行排序（降序）
    sorted_idx = np.argsort(feature_importances)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importances = feature_importances[sorted_idx].tolist()

    # 5. 取前 10 个，如果不足 10 个就全部显示
    top_n = 10
    if len(sorted_features) > top_n:
        sorted_features = sorted_features[:top_n]
        sorted_importances = sorted_importances[:top_n]

    # 6. 构造 pyecharts 柱状图
    bar = Bar(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
    bar.add_xaxis(sorted_features)
    bar.add_yaxis("重要性", sorted_importances)

    bar.set_global_opts(
        xaxis_opts=opts.AxisOpts(
            axislabel_opts=opts.LabelOpts(rotate=30, interval=0)
        ),
        yaxis_opts=opts.AxisOpts(name="重要性分数")
    )

    # 7. 返回 JSON 配置，前端直接使用即可
    return json.loads(bar.dump_options())


def create_confusion_matrix_heatmap(df):
    """使用真实混淆矩阵数据"""
    global conf_matrix

    if conf_matrix is None:
        train_churn_model(df)

    data = []
    for i in range(2):
        for j in range(2):
            data.append([j, i, int(conf_matrix[i, j])])

    heatmap = HeatMap(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
    heatmap.add_xaxis(['预测未流失', '预测已流失'])
    heatmap.add_yaxis('实际', ['实际未流失', '实际已流失'], data)

    heatmap.set_global_opts(
        visualmap_opts=opts.VisualMapOpts(min_=0, max_=max(conf_matrix.flatten()), orient="vertical", pos_left="right", pos_top="middle"),
        tooltip_opts=opts.TooltipOpts(formatter='{b0}: {c} 人')
    )
    return json.loads(heatmap.dump_options())


def create_probability_dist(df):
    """使用真实预测概率分布"""
    global y_pred_proba

    if y_pred_proba is None:
        train_churn_model(df)

    # 创建预测概率分布
    bins = np.linspace(0, 1, 11)
    hist, bin_edges = np.histogram(y_pred_proba, bins=bins)
    bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}" for i in range(len(bin_edges) - 1)]

    bar = Bar(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
    bar.add_xaxis(bin_labels)
    bar.add_yaxis('客户数', hist.tolist())

    bar.set_global_opts(
        xaxis_opts=opts.AxisOpts(name='流失概率区间'),
        yaxis_opts=opts.AxisOpts(name='客户数')
    )
    return json.loads(bar.dump_options())


def create_churn_probability_trend(df):
    """
    绘制“按在网时长分组的实际流失率”和“基于训练好模型的预测流失率”折线图。
    df 需要包含 ['customerID', 'tenure', 'Churn', 以及其他特征列]。
    """
    global model

    # 1. 先按 tenure 分箱分组，计算每个区间的实际流失率
    #    使用 pd.cut 按 12 个等距区间切分
    tenure_bins = pd.cut(df['tenure'], bins=12)
    tenure_groups = df.groupby(tenure_bins)

    # 2. 计算每个区间的“实际流失率”：只取 'Churn' 列（数值），再取 mean
    #    乘以 100 得到百分比
    churn_rates = tenure_groups['Churn'].mean(numeric_only=True) * 100
    # churn_rates.index 是 IntervalIndex，比如 (0.909, 9.091], (9.091, 17.273], ...

    # 3. 准备绘图的 x 轴标签和 y 轴数值
    x_axis = [f"{int(interval.left)}-{int(interval.right)}月" for interval in churn_rates.index]
    y_actual = churn_rates.values.round(2).tolist()

    # 4. 初始化折线图并添加实际流失率折线
    line = Line(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
    line.add_xaxis(x_axis)
    line.add_yaxis("实际流失率 (%)", y_actual, is_smooth=True, linestyle_opts=opts.LineStyleOpts(width=2))

    # 5. 如果全局模型已训练好，则对每个分组的“平均特征”做一次预测
    if model is not None:
        # 5.1 先按分组计算数值特征平均值（自动忽略非数值列）
        #     比如 df 中除了 tenure、Churn，还有 MonthlyCharges、TotalCharges……这些都会被聚合
        group_means = tenure_groups.mean(numeric_only=True)

        # 5.2 去掉标签列 'Churn'，以及非特征列 'customerID'（mean 结果里不会有 'customerID'，已 numeric_only=True 忽略）
        if 'Churn' in group_means.columns:
            group_features = group_means.drop(columns=['Churn'])
        else:
            group_features = group_means.copy()

        # 5.3 注意：如果你的模型 Pipeline 里需要对分类特征进行 OneHot，
        #      此处需要传入包含分类列的“原始分组平均值”，然后让 Pipeline 自动编码。
        #      考虑到我们只对数值列做 mean，下面的做法需要保证“分类列”在聚合时要么被忽略、
        #      要么你自己补齐分组后的“分类特征平均”方式（通常没有意义）。最常见的做法是：
        #      model.predict_proba(group_features) 会自动通过 ColumnTransformer 处理，
        #      但前提是 group_features 必须含有完整的原始特征列（包括分类列）。
        #
        #      因此，更稳妥的实现：
        #      ① 先用 tenure_bins 给 df 增加一列，例如 df['tenure_bin'] = tenure_bins
        #      ② 对原始 df 做 groupby，选出所有特征列的“平均”方法，对于数值列用 mean，对于分类列用“众数”或“第一个出现值”……
        #      ③ 得到一个“每个分组对应一行、包含所有原始特征列”的 DataFrame group_all_features，
        #      ④ 再把 group_all_features 直接传给 model.predict_proba()。
        #
        #      下面示例中，我们采取一个简化技巧：只让模型对数值特征做预测，而把分类特征“拼一个常数”或跳过。这通常只适用于分类特征对预测影响不大，或者分类特征只有 one-hot 编码之后才用。
        #
        #      如果你的 model.pipeline.preprocessor 中已经指定了对分类列做 OneHot，那么直接传 group_features（只含数值列）会导致管道抛错“找不到分类列”。
        #
        #      正确做法是重建一个“分组后的原始 DataFrame”，示例见下面的“附加说明”。
        #
        # 5.4 假设“分组平均值只包含数值列”正好能让 Pipeline 工作（你的 ColumnTransformer 中只对数值列 passthrough，分类列全部用 OneHotEncoder），则：
        try:
            # Model.predict_proba 接受 DataFrame 或二维 ndarray，两种方式都行
            proba_raw = model.predict_proba(group_features)[:, 1] * 100
            y_pred = np.round(proba_raw, 2).tolist()
            # 将预测折线加到图上，虚线样式
            line.add_yaxis(
                "预测流失率 (%)",
                y_pred,
                is_smooth=True,
                linestyle_opts=opts.LineStyleOpts(type_='dashed', width=2),
                label_opts=opts.LabelOpts(is_show=False)
            )
        except Exception:
            # 如果上述简化方式出错，可改用“原始分组 + 众数”策略，见下方附加说明
            pass

    # 6. 全局配置
    line.set_global_opts(
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        xaxis_opts=opts.AxisOpts(name="在网时长 (月)", axislabel_opts=opts.LabelOpts(rotate=30)),
        yaxis_opts=opts.AxisOpts(name="流失率 (%)"),
        legend_opts=opts.LegendOpts(pos_top="5%")
    )

    return json.loads(line.dump_options())