# 模型调用API

## 简介交互流程
后端使用Flask,开放8080端口，路径指定/predict

前端访问http://localhost:8080/predict, POST请求，发送的29个标签需要json序列化

后端调用/model下的模型，返回预测结果,json序列化返回前端

## 所需环境与包
- `json`：处理 JSON 数据格式，进行数据的序列化和反序列化。
- `joblib`：加载已保存的模型和 `LabelEncoder` 对象。
- `pandas`：进行数据处理和分析，例如把 JSON 数据转换为 `DataFrame` 格式。
- `numpy`：用于数值计算，在数据预处理等环节会用到。
- `sklearn`：具体使用 `sklearn.preprocessing.LabelEncoder` 对标签进行编码和解码。
- `flask`：用于创建应用、处理请求和返回 JSON 响应。

可以使用如下命令安装这些包：
```bash
pip install joblib pandas numpy scikit-learn flask
```


## 开放端口
监听 `8080` 端口,在(port=8080)中可以自定义
```python 
if __name__ == '__main__':
    app.run(debug=True, port=8080)
```

## 路径
API 的访问路径为 `/predict`，完整的请求地址为 `http://localhost:8080/predict`。

## 输入
JSON 格式序列化后的输入数据，数据需包含 29 个特征，具体的特征的对应名称如下：
1. `SerialNumber`（序号）
2. `rectimestamp`（记录时间戳）
3. `sendtimestamp`（发送时间戳）
4. `DelayMilliseconds`（时延毫秒）
5. `Latitude`（纬度）
6. `Longitude`（经度）
7. `Altitude`（海拔）
8. `SpeedKilometersPerHour`（速度 km/h）
9. `SatelliteCount`（锁星数）
10. `GroundHeadingDegree`（地面航向度）
11. `ID`（ID 号，当该值为空时，可以直接填充 `null`）
12. `msgCnt`（消息计数）
13. `TotalRcvMsgCount`（总接收消息数）
14. `TotalLostMsgCount`（总丢失消息数）
15. `AID`（AID 号）
16. `Source_Layer2-ID`（源 Layer2 ID）
17. `DestMACID`（目标 MAC ID）
18. `Priority`（优先级）
19. `Power`（功率，单位 dBm）
20. `RSSI`（接收信号强度指示，单位 dBm）
21. `SINR`（信号与干扰加噪声比，单位 dB）
22. `CBR`（恒定比特率，单位 %）
23. `MCS`（调制与编码策略）
24. `RSRP`（参考信号接收功率，单位 dBm）
25. `EARFCN`（绝对频点号）
26. `Band`（频段）
27. `TBSize`（传输块大小，单位 bit）
28. `Modulation`（调制方式）
29. `Type`（类型）


## 输出
API 处理完输入数据并调用模型进行预测后，将返回一个 JSON 格式的响应，内容为预测的标签值，标签值可能为以下几种之一：`Normal`（正常）、`Replay`（重放攻击）、`DoS`（拒绝服务攻击）、`DDoS`（分布式拒绝服务攻击）、`Spoof`（欺骗攻击）。

输出格式也是json序列化后的结果：
```python
jsonify({'prediction': predictions[0]})
```

输出示例：
```json
{
    "prediction": "Normal"
}
``` 
### 简单测试
#### 运行python后端服务器
python app.py
#### ubuntu使用curl
```bash
curl -X POST "http://localhost:8080/predict" \
-H "Content-Type: application/json" \
-d '{
    "SerialNumber": 2320,
    "rectimestamp": 1699952571352,
    "sendtimestamp": 1699952571601,
    "DelayMilliseconds": -249.2808072596588,
    "Latitude": 116.493275,
    "Longitude": 39.72909,
    "Altitude": 23.2,
    "SpeedKilometersPerHour": 9.13,
    "SatelliteCount": 50,
    "GroundHeadingDegree": 342.97,
    "ID": null,
    "msgCnt": 71,
    "TotalRcvMsgCount": 3257,
    "TotalLostMsgCount": 2322,
    "AID": 3619,
    "Source_Layer2-ID": "8000001",
    "DestMACID": 7,
    "Priority": 8,
    "Power": -100,
    "RSSI": -73,
    "SINR": 13,
    "CBR": 0,
    "MCS": 5,
    "RSRP": -100,
    "EARFCN": 55140,
    "Band": 47,
    "TBSize": 4264,
    "Modulation": "QPSK",
    "Type": "SPAT"
}'
```
#### windows使用powershell发送
```powershell
$json = @'
{
    "SerialNumber": 2320,
    "rectimestamp": 1699952571352,
    "sendtimestamp": 1699952571601,
    "DelayMilliseconds": -249.2808072596588,
    "Latitude": 116.493275,
    "Longitude": 39.72909,
    "Altitude": 23.2,
    "SpeedKilometersPerHour": 9.13,
    "SatelliteCount": 50,
    "GroundHeadingDegree": 342.97,
    "ID": null,
    "msgCnt": 71,
    "TotalRcvMsgCount": 3257,
    "TotalLostMsgCount": 2322,
    "AID": 3619,
    "Source_Layer2-ID": "8000001",
    "DestMACID": 7,
    "Priority": 8,
    "Power": -100,
    "RSSI": -73,
    "SINR": 13,
    "CBR": 0,
    "MCS": 5,
    "RSRP": -100,
    "EARFCN": 55140,
    "Band": 47,
    "TBSize": 4264,
    "Modulation": "QPSK",
    "Type": "SPAT"
}
'@

$headers = @{
    "Content-Type" = "application/json"
}

$response = Invoke-RestMethod -Uri "http://localhost:8080/predict" -Method Post -Headers $headers -Body $json
$response
```
#### 返回值
```powershell
prediction                   
----------                   
Replay
```