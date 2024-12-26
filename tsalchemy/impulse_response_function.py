import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

class IRF:
    def __init__(self, data, lags):
        """
        初始化IRF类
        :param data: 输入的时间序列数据，类型为pandas DataFrame
        :param lags: VAR模型的滞后阶数
        """
        self.data = data
        self.lags = lags
        self.model = VAR(data)
        self.fitted_model = self.model.fit(lags)
        self.irf = None

    def compute_irf(self, steps=10, shock_magnitude=-1.0):
        """
        计算脉冲响应函数
        :param steps: 响应步数
        :param shock_magnitude: 冲击幅度
        """
        self.irf = self.fitted_model.irf(steps)
        self.irf.irfs *= shock_magnitude
        return self.irf

    def plot_irf(self, impulse, responsefigsize=(20, 5)):
        """
        绘制脉冲响应函数
        :param impulse: 脉冲变量的名称
        :param response: 响应变量的名称
        """
        self.irf.plot(impulse=impulse, response=response, figsize=figsize)
        plt.show()

    def get_irf_values(self, impulse, response):
        """
        获取脉冲响应函数的数值
        :param impulse: 脉冲变量的名称
        :param response: 响应变量的名称
        :return: 脉冲响应函数的数值
        """
        return self.irf.irfs[:, self.data.columns.get_loc(impulse), self.data.columns.get_loc(response)]

    def get_all_responses(self, impulse, steps=10, shock_magnitude=-1.0):
        """
        获取一个变量对data中其他变量的冲击响应
        :param impulse: 冲击变量的名称
        :param steps: 响应步数
        :param shock_magnitude: 冲击幅度
        :return: 其他所有变量的响应值，以DataFrame形式返回
        """
        
        impulse_index = self.data.columns.get_loc(impulse)
        responses = {}
        
        for response in self.data.columns:
            response_index = self.data.columns.get_loc(response)
            responses[response] = self.irf.irfs[:, impulse_index, response_index] * shock_magnitude
        
        return pd.DataFrame(responses)