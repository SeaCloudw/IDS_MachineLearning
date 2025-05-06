import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify

app = Flask(__name__)

# 加载已训练好的模型和 LabelEncoder
model = joblib.load('./model/model_DecisionTreeClassifier.joblib')
labelencoder = joblib.load('./model/label_encoder.joblib')

def hash_string(s):
    return hash(s)

def preprocess_single_json(json_data, labelencoder):
    df = pd.DataFrame([json_data])
    # print("处理前特征数量:", len(df.columns))
    # print("原始特征名称:", df.columns)

    if not pd.api.types.is_numeric_dtype(df['rectimestamp']):
        try:
            df['rectimestamp'] = pd.to_numeric(df['rectimestamp'])
        except ValueError:
            non_numeric_rows = df[pd.to_numeric(df['rectimestamp'], errors='coerce').isna()]
            assert False, f"发现非数值类型的 'rectimestamp':\n{non_numeric_rows.head()}"

    try:
        df['ID'] = df['ID'].apply(lambda x: hash_string(x) if isinstance(x, str) else x)
    except Exception as e:
        print(f"无法转换的值: {e}")
        non_numeric_rows = df[df['ID'].apply(lambda x: not isinstance(x, (int, float)))]
        print("无法转换的行:\n", non_numeric_rows)
        assert False, "存在无法转换为数值的 'ID' 值"

    # print("所有 'rectimestamp' 值均为数值类型，可以继续处理数据。")

    numeric_features = df.dtypes[df.dtypes != 'object'].index
    df = df.fillna(0)

    X = df.drop(['Type', 'Modulation', 'AID'], axis=1).values
    # X = df.values
    # print("处理后特征数量:", len(df.drop(['Type', 'Modulation', 'AID'], axis=1).columns))

    return X

def predict_with_model(model, X, labelencoder):
    y_pred = model.predict(X)
    y_pred_labels = labelencoder.inverse_transform(y_pred)
    return y_pred_labels

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取 POST 请求中的 JSON 数据
        data = request.get_json()

        # 预处理 JSON 数据
        X = preprocess_single_json(data, labelencoder)

        # 进行预测
        predictions = predict_with_model(model, X, labelencoder)

        # 返回预测结果作为 JSON 响应
        return jsonify({'prediction': predictions[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=8080)