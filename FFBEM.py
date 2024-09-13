# 导入所需的库
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from lightgbm import early_stopping, LGBMRegressor
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def ffbem_model(data_id, train_begin_time, train_end_time, begin_time, end_time):
    data = pd.read_csv(f'数据集/{data_id}.csv', parse_dates=['DATATIME'], dayfirst=False)
    data = data.drop_duplicates(subset='DATATIME', keep='first')
    data['DATATIME'] = pd.to_datetime(data['DATATIME'], format='%d/%m/%Y %H:%M:%S').dt.strftime('%Y-%m-%d %H:%M')
    for index, row in data.iterrows():
        if np.isnan(row['A_WS']):
            data.at[index, 'A_WS'] = data.at[index, 'WINDSPEED']
    data = data.interpolate(method='linear', limit_direction='both').reset_index()
    data['DATATIME'] = pd.to_datetime(data['DATATIME'])
    columns_to_float = ['WINDSPEED', 'PREPOWER', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE',
                        'A_WS', 'A_POWER', 'YD15']
    data[columns_to_float] = data[columns_to_float].astype(float)
    # 异常值处理
    for index, row in data.iterrows():
        if row['YD15'] < -800:
            data.at[index, 'YD15'] = 0
        # if row['A_WS'] < 0:
        #     data.at[index, 'A_WS'] = 0
        if row['WINDSPEED'] < 1 and row['YD15'] > 800:
            data.at[index, 'YD15'] = 0
    for col in data.columns.tolist():
        if col != 'DATATIME' and col != 'A_POWER' and col != 'YD15':
            data.drop(data[(data[col] > (data[col].mean() + 4 * data[col].std())) | (
                    data[col] < (data[col].mean() - 4 * data[col].std()))].index, inplace=True)
    # 使数据更平稳
    mean_power = dict(data.groupby('WINDSPEED')['YD15'].mean())
    for index, row in data.iterrows():
        speed = row['WINDSPEED']
        power = row['YD15']
        if row['YD15'] == 0:
            data.at[index, 'YD15'] = mean_power[speed]
        if power > mean_power[speed] * 1.5 or power < mean_power[speed] * 0.5:
            data.at[index, 'YD15'] = float((mean_power[speed] + power) / 2)
        if 10 > row['WINDSPEED'] > 5 and row['YD15'] < 0:
            data.at[index, 'YD15'] = float(mean_power[speed])
    # 特征交叉
    data['WINDSPEED_A_WS'] = data['WINDSPEED'] * data['A_WS']
    data['WINDSPEED_B_WS'] = data['WINDSPEED'] * data['A_WS'] * data['TEMPERATURE'] * data['WINDDIRECTION']
    data['TEMPERATURE_HUMIDITY'] = data['TEMPERATURE'] * data['HUMIDITY']
    data['TEMPERATURE_PRESSURE'] = data['TEMPERATURE'] * data['PRESSURE']
    # 使用训练好的模型预测下一天的实际功率YD15
    next_day_data = data.loc[
        (data.DATATIME >= begin_time) & (data.DATATIME <= end_time), ['DATATIME', 'WINDSPEED',
                                                                      'WINDDIRECTION',
                                                                      'TEMPERATURE', 'HUMIDITY',
                                                                      'PRESSURE', 'PREPOWER', 'A_WS',
                                                                      'WINDSPEED_A_WS',
                                                                      'WINDSPEED_B_WS',
                                                                      'TEMPERATURE_HUMIDITY',
                                                                      'TEMPERATURE_PRESSURE',
                                                                      'A_POWER',
                                                                      'YD15']]
    # 将DATATIME字段转换为日期时间格式
    data['DATATIME'] = pd.to_datetime(data['DATATIME'])
    data.set_index('DATATIME', inplace=True)
    data = data.drop(data.columns[0], axis=1)
    train_data = data[train_begin_time: train_end_time]
    test_data = data[begin_time: end_time]
    # 选择特征和目标
    features = ['WINDSPEED', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE', 'PREPOWER', 'A_WS',
                'WINDSPEED_A_WS', 'WINDSPEED_B_WS', 'TEMPERATURE_HUMIDITY', 'TEMPERATURE_PRESSURE']
    # 数据预处理
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data[features])
    test_scaled = scaler.fit_transform(test_data[features])
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 构建Transformer
    transformer_model = torch.load(f'transformer/transformer_model_{data_id}.pth')
    transformer_train = predict_transformer(transformer_model, train_scaled, device).reshape(-1, 1)
    transformer_train = transformer_train.reshape(-1)
    transformer_train = np.array(transformer_train)
    transformer_train = pd.DataFrame(transformer_train, columns=['transformer'])
    print(transformer_train)
    transformer_test = predict_transformer(transformer_model, test_scaled, device).reshape(-1, 1)
    transformer_test = transformer_test.reshape(-1)
    transformer_test_array = np.array(transformer_test)
    transformer_test_df = pd.DataFrame(transformer_test_array, columns=['transformer'])
    print(transformer_test_df)
    # 训练 LightGBM
    mse, _ = mutiTrain_GBM(train_data, transformer_train, data_id)
    # 额外特征输入
    test_data['transformer'] = transformer_test_df['transformer']
    train_data1 = next_day_data.copy()
    models = load_models(data_id)
    frame0 = weightPredict(models, test_data, next_day_data, train_data1)
    frame = frame0.loc[:, ['DATATIME', 'A_POWER', 'YD15']]
    ffbem = np.array(frame['YD15'])
    ffbem = divide_by_thousand(ffbem)
    ffbem = np.array(ffbem)
    # 分析 FFBEM
    true = np.array(next_day_data['YD15'])
    true = divide_by_thousand(true)
    true = np.array(true)
    print(true)
    rmse_ffbem_result = calculate_rmse(ffbem, true)
    mae_ffbem_result = calculate_mae(ffbem, true)
    fa_ffbem_result = calculate_fa(ffbem, true)
    dc_ffbem_result = calculate_dc(ffbem, true)
    print('RMSE-FFBEM:', rmse_ffbem_result)
    print('MAE-FFBEM:', mae_ffbem_result)
    print('FA-FFBEM:', fa_ffbem_result)
    print('DC-FFBEM:', dc_ffbem_result)
    # 分析 Transformer
    transformer_test_array = divide_by_thousand(transformer_test_array)
    transformer_test_array = np.array(transformer_test_array)
    rmse_transformer_result = calculate_rmse(transformer_test_array, true)
    mae_transformer_result = calculate_mae(transformer_test_array, true)
    fa_transformer_result = calculate_fa(transformer_test_array, true)
    dc_transformer_result = calculate_dc(transformer_test_array, true)
    print(transformer_test_array)
    print('RMSE-transformer:', rmse_transformer_result)
    print('MAE-transformer:', mae_transformer_result)
    print('FA-transformer:', fa_transformer_result)
    print('DC-transformer:', dc_transformer_result)
    return ffbem


# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(input_dim, model_dim)
        self.output_layer = nn.Linear(model_dim, 1)

    def forward(self, src):
        src = self.linear(src)
        output = self.transformer_encoder(src)
        output = self.output_layer(output)
        return output


# Transformer模型预测
def predict_transformer(model, data, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(len(data)):
            input_seq = torch.tensor(data[i], dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(input_seq.unsqueeze(1)).cpu().numpy().flatten()[0]
            predictions.append(pred)
    return np.array(predictions)


def divide_by_thousand(arr):
    return [x / 1000 for x in arr]


def calculate_rmse(predictions, true_values):
    # 计算预测结果与真实结果之间的差值数组
    residuals = predictions - true_values
    # 计算差值数组的平方
    squared_residuals = residuals ** 2
    # 计算均方误差（MSE）
    mse = np.mean(squared_residuals)
    # 计算均方根误差（RMSE）
    rmse = np.sqrt(mse)
    return rmse


def calculate_mae(predictions, true_values):
    # 计算预测结果与真实结果之间的差值数组
    residuals = np.abs(predictions - true_values)
    mae = np.mean(residuals)
    return mae


def calculate_fa(predictions, true_values):
    # 计算预测结果与真实结果之间的差值数组
    residuals = np.abs(np.abs(predictions - true_values) / true_values)
    all = np.mean(residuals)
    fa = 1 - all
    return fa


def calculate_dc(predictions, true_values):
    # 计算残差平方和（SSE）
    residuals = true_values - predictions
    sse = np.sum(residuals ** 2)
    # 计算总平方和（SST）
    mean_actual = np.mean(true_values)
    sst = np.sum((true_values - mean_actual) ** 2)
    # 计算决定系数（DC）
    dc = 1 - sse / sst
    return dc


# lightGBM模型训练
def mutiTrain_GBM(dataTrain, transformer_train, id):
    # models = joblib.load('templates/src/model/gbm' + id + '.pkl')
    new_dataTrainGBM = dataTrain.copy()
    # train = new_dataTrainGBM[
    #     ['WINDSPEED', 'PREPOWER', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE', 'A_WS',
    #      'Year', 'Month', 'Day', 'Hour', 'Minute',
    #      'WINDSPEED_WINDDIRECTION', 'WINDSPEED_TEMPERATURE', 'WINDSPEED_HUMIDITY', 'WINDSPEED_PRESSURE']]
    # train = new_dataTrainGBM[['WINDSPEED', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE', 'PREPOWER', 'A_WS',
    #                           'WINDSPEED_A_WS', 'WINDSPEED_B_WS', 'TEMPERATURE_HUMIDITY',
    #                           'TEMPERATURE_PRESSURE']]
    print(new_dataTrainGBM)
    print(new_dataTrainGBM.shape)
    new_dataTrainGBM['transformer'] = transformer_train['transformer']
    # train = new_dataTrainGBM[['WINDSPEED', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE', 'PREPOWER', 'A_WS']]
    # print(train.shape)
    target = new_dataTrainGBM[['YD15']].values
    x_train, x_test, y_train, y_test = train_test_split(new_dataTrainGBM.values, target, test_size=0.1)
    gbm = LGBMRegressor(objective='regression', learning_rate=0.01, n_estimators=400, n_jobs=-1)
    gbm = gbm.fit(x_train, y_train.ravel(), eval_set=[(x_test, y_test.ravel())], eval_metric='l1',
                  callbacks=[early_stopping(stopping_rounds=200)])
    y_pre_gbm = gbm.predict(x_test)
    mse = mean_squared_error(y_test, y_pre_gbm)
    print(f'gbm {id} mean_squared_error:{mse}')
    score = gbm.score(x_test, y_test)
    print(f'gbm {id} score:{score}')
    joblib.dump(gbm, 'model/FFBEM_' + id + '.pkl')
    return mse, score


def load_models(id):
    models = joblib.load('model/FFBEM_' + id + '.pkl')
    return models


def weightPredict(models, train, new_data, new_data1):
    output = models.predict(train)
    datas = {'DATATIME': new_data1['DATATIME'], 'WINDSPEED': new_data['WINDSPEED'],
             'PREPOWER': new_data['PREPOWER'], 'WINDDIRECTION': new_data['WINDDIRECTION'],
             'TEMPERATURE': new_data['TEMPERATURE'], 'HUMIDITY': new_data['HUMIDITY'],
             'PRESSURE': new_data['PRESSURE'], 'A_WS': new_data['A_WS'],
             'A_POWER': new_data['A_POWER'], 'YD15': output.flatten()}
    frame = pd.DataFrame(datas)
    return frame


data_id = '06'
train_begin_time = '2021-01-02 00:00'
train_end_time = '2022-05-29 23:45'
begin_time = '2022-05-30 00:00'
end_time = '2022-05-30 23:45'
ffbem_model = ffbem_model(data_id, train_begin_time, train_end_time, begin_time, end_time)
print('FFBEM: ', ffbem_model)
