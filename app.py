import pandas as pd
from flask import Flask, request, jsonify, request, render_template, stream_with_context, Response
from res import preprocess as prep
from res import analyze
import requests

app = Flask(__name__)

# 全局变量
df_processed = None
model = None
feature_names = None
feature_importances = None
AI_SERVER_URL = "http://localhost:2268"  # AI服务器配置

def load_and_preprocess_data():
    global df_processed, model, feature_names, feature_importances
    if df_processed is None:
        try:
            df_processed = prep.preprocess_telco_data()
            print("数据加载与预处理完成。")
        except Exception as e:
            print(f"数据初始化失败: {e}")
            df_processed = pd.DataFrame()

    # 初次训练模型，填充 model、feature_names、feature_importances 等全局
    if df_processed is not None and not df_processed.empty:
        try:
            # 分析模块中的 train_churn_model 会更新全局 model、feature_names、feature_importances 等
            feature_names, feature_importances, model = analyze.train_churn_model(df_processed)
        except Exception as e:
            print(f"模型训练失败: {e}")

with app.app_context():
    load_and_preprocess_data()


# 流式生成接口代理
@app.route('/api/ai/stream_generate', methods=['POST'])
def ai_stream_generate():
    """
    代理请求到AI服务器的流式生成接口
    """
    try:
        # 获取客户端发送的数据
        data = request.get_json()

        data['prompt'] = ("你是一个电信客户流失分析助手，请根据提供的数据回答用户问题。注意：你只能回答与电信客户流失分析相关的问题。如果"
                          "用户的问题与电信客户流失分析无关，请礼貌地拒绝回答。以下是用户提问：" + data['prompt'])

        # 转发请求到AI服务器
        ai_response = requests.post(
            f"{AI_SERVER_URL}/stream_generate",
            json=data,
            stream=True  # 保持流式连接
        )

        # 检查AI服务器响应
        if ai_response.status_code != 200:
            return jsonify({
                "error": f"AI server error: {ai_response.status_code}",
                "message": ai_response.text
            }), 500

        # 创建流式响应
        def generate():
            for chunk in ai_response.iter_content(chunk_size=1024):
                if chunk:  # 过滤空块
                    yield chunk

        return Response(
            stream_with_context(generate()),
            content_type='text/event-stream'
        )

    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": "AI server connection failed",
            "message": "请确保AI服务器正在运行"
        }), 503
    except Exception as e:
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


# 中断生成接口代理
@app.route('/api/ai/interrupt_stream', methods=['POST'])
def ai_interrupt_stream():
    """
    代理中断请求到AI服务器
    """
    try:
        # 转发请求到AI服务器
        response = requests.post(f"{AI_SERVER_URL}/interrupt_stream")
        return jsonify(response.json()), response.status_code
    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": "AI server connection failed",
            "message": "请确保AI服务器正在运行"
        }), 503
    except Exception as e:
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


# 非流式生成接口代理 (可选)
@app.route('/api/ai/generate', methods=['POST'])
def ai_generate():
    """
    代理请求到AI服务器的非流式生成接口
    """
    try:
        # 获取客户端发送的数据
        data = request.get_json()

        # 转发请求到AI服务器
        response = requests.post(f"{AI_SERVER_URL}/generate", json=data)
        return jsonify(response.json()), response.status_code
    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": "AI server connection failed",
            "message": "请确保AI服务器正在运行"
        }), 503
    except Exception as e:
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/test')
def testIndex():
    return render_template('dashboard-test.html')

@app.route('/new')
def newIndex():
    return render_template('dashboard-new.html')


@app.route('/api/dashboard')
def api_dashboard():
    """
    返回仪表盘 1/2/3 所有图表的配置 JSON，字段名称必须与前端 initCharts 中一致
    同时返回 dataCount 供前端显示数据量
    """
    global df_processed

    # 确保全局模型等已初始化
    if df_processed is None or df_processed.empty:
        return jsonify({"error": "数据尚未准备好"}), 500

    try:
        charts = {
            # 仪表盘1
            "churnRatePie": analyze.create_pie_churn_overview(df_processed),
            "servicesHeatmap": analyze.create_services_heatmap(df_processed),
            "contractBar": analyze.create_contract_grouped_bar(df_processed),
            "tenureHist": analyze.create_tenure_histogram(df_processed),
            "paymentBar": analyze.create_payment_bar(df_processed),
            "churnTrend": analyze.create_churn_probability_trend(df_processed),

            # 仪表盘2
            "serviceCombo": analyze.create_service_combo_churn_bar(df_processed),
            "scatterTC": analyze.create_scatter_tenure_charges(df_processed),
            "seniorBar": analyze.create_senior_bar(df_processed),
            "churnServices": analyze.create_churn_services_bar(df_processed),
            "contractBilling": analyze.create_contract_billing_stacked(df_processed),
            "customerMatrix": analyze.create_customer_matrix(df_processed),

            # 仪表盘3
            "modelMetrics": analyze.create_model_metrics_bar(df_processed),
            "rocCurve": analyze.create_roc_curve(df_processed),
            "featureImportance": analyze.create_feature_importance_bar(df_processed),
            "confusionMatrix": analyze.create_confusion_matrix_heatmap(df_processed),
            "probabilityDist": analyze.create_probability_dist(df_processed),
            "churnProbabilityTrend": analyze.create_churn_probability_trend(df_processed)
        }
        charts["dataCount"] = len(df_processed)
        return jsonify(charts)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def compute_loyalty_score(row):
    """
    根据简化的业务规则，为单行（单个客户）计算忠诚度分数，返回 1～10 之间的浮点数，保留一位小数。
    规则示例（仅演示思路，您可自行微调权重与阈值）：
      1. 合同类型：Two year 合同 +2 分；One year +1；Month-to-month +0
      2. 在网时长（tenure）越长得分越高，取 tenure/max_tenure * 3 分
      3. 月消费（MonthlyCharges）越高得分越高，取 MonthlyCharges/max_monthlyCharges * 2 分
      4. 服务数量（ServicesCount）>=3 得 +2 分；1～2 个得 +1 分；0 个得 0 分
      5. 最终将总分映射到 1～10 之间
    """
    # 先提取常用字段
    contract = row['Contract']                # 合同类型：’Month-to-month‘ / ’One year‘ / ’Two year‘
    tenure = float(row['tenure'])             # 在网时长（月）
    monthly_charges = float(row['MonthlyCharges'])  # 月消费
    services_count = int(row['ServicesCount'])      # 服务数量

    # 1. 合同得分
    contract_score = 0
    if contract == 'Two year':
        contract_score = 2
    elif contract == 'One year':
        contract_score = 1
    # Month-to-month 不加分

    # 2. tenure 得分：tenure 相对于数据集最大值，归一化到 0～3 分
    #    （假设 df_processed 中最大 tenure 不超过 72 月；可根据实际数据集最大值微调）
    max_tenure = df_processed['tenure'].max() if df_processed is not None else 72
    tenure_score = (tenure / max_tenure) * 3  # 归一到 0～3

    # 3. monthly_charges 得分：相对于该字段最大值，归一化到 0～2 分
    max_montly = df_processed['MonthlyCharges'].max() if df_processed is not None else 120.0
    monthly_score = (monthly_charges / max_montly) * 2  # 归一到 0～2

    # 4. 服务数量得分
    if services_count >= 3:
        services_score = 2
    elif services_count >= 1:
        services_score = 1
    else:
        services_score = 0

    # 汇总原始总分
    raw_score = contract_score + tenure_score + monthly_score + services_score
    # raw_score 最大约为：2 + 3 + 2 + 2 = 9；最小为 0

    # 将 raw_score （范围 0～9）线性映射到 1～10
    #   formula: loyalty = raw_score / 9 * 9 + 1 = raw_score + 1
    #   但由于 raw_score 含小数，因此最后取 1 位小数
    loyalty = raw_score + 1.0

    # 保证边界在 [1, 10]
    if loyalty < 1.0:
        loyalty = 1.0
    if loyalty > 10.0:
        loyalty = 10.0

    return round(loyalty, 1)


@app.route('/api/customer/<customerId>')
def api_customer(customerId):
    """
    根据 customerID 返回该客户的基本信息、流失预测概率以及忠诚度评分
    """
    global df_processed, model

    # 1. 判断数据与模型是否已准备好
    if df_processed is None or df_processed.empty or model is None:
        return jsonify({"error": "数据或模型尚未准备好"}), 500

    try:
        # 2. 根据 customerId 查找对应行
        row = df_processed[df_processed['customerID'] == customerId]
        if row.empty:
            return jsonify({"error": "客户ID不存在"}), 404
        row = row.iloc[0]  # 取出 Series

        # 3. 预测流失概率
        X_single = df_processed[df_processed['customerID'] == customerId].drop(columns=['customerID', 'Churn'])
        proba = float(model.predict_proba(X_single)[0][1] * 100)

        # 4. 计算忠诚度评分
        loyalty_score = compute_loyalty_score(row)

        # 5. 构造返回 JSON
        info = {
            "customerID": customerId,
            "churnProbability": round(proba, 2),
            "tenure": int(row['tenure']),
            "monthlyCharges": float(row['MonthlyCharges']),
            "contractType": row['Contract'],
            "servicesCount": int(row['ServicesCount']),
            "loyaltyScore": loyalty_score  # 新增：忠诚度评分
        }
        return jsonify(info)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=9999)
