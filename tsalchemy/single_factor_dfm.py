import pandas as pd
from copy import deepcopy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
import statsmodels.api as sm
from collections import OrderedDict


class SingleFactorDFM:
    def __init__(self, df_data, standardize=True, s_ref=None, factor_name=None, plot_title=None, ref_name=None):
        """
        初始化方法
        :param df_data: 输入数据，pandas DataFrame 格式
        :param standardize: 是否标准化数据，默认为 True
        :param s_ref: 参考数据，pandas Series 格式
        :param factor_name: 因子名称，字符串格式
        """
        self.df_data_raw = df_data
        if not standardize:
            self.df_data = df_data
        else:
            self.df_data = self.standardize()
        self.res = None
        self.factor_name = factor_name
        self.s_ref = s_ref
        self.s_factor = pd.Series(dtype='float', name=factor_name)
        self.df_eval = pd.DataFrame()
        self.plot_title = plot_title
        self.ref_name = ref_name
        return

    def standardize(self):
        df = deepcopy(self.df_data_raw)
        variables = df.columns.tolist()
        df_data = pd.DataFrame()
        for var in variables:
            s = pd.DataFrame(StandardScaler().fit_transform(df[[var]]), index=df.index)[0]
            df_data[var] = s
        return df_data

    def build_model(self, k_factors=1, factor_order=2, error_order=1, fit_method='powell'):
        """
        构建动态因子模型
        :param k_factors: 因子数量，默认为 1
        :param factor_order: 因子阶数，默认为 2
        :param error_order: 误差阶数，默认为 1
        :param fit_method: 拟合方法，默认为 'powell'
        """
        model = DynamicFactor(self.df_data, k_factors=k_factors, factor_order=factor_order, error_order=error_order)
        res = model.fit(method=fit_method)
        self.res = res
        s_factor = pd.Series(self.res.factors['filtered'][0], index=self.df_data.index)
        s_factor = pd.DataFrame(MinMaxScaler().fit_transform(pd.DataFrame(s_factor)), index=s_factor.index)[0]
        s_factor.name = self.factor_name
        self.s_factor = s_factor
        self.check_factor()
        self.eval_factor()
        self.plot_factor()
        return

    def check_factor(self):
        """
        检查因子与参考数据的相关性，并根据相关性调整因子正负符号
        """
        if self.s_ref is None:
            return
        df = pd.DataFrame(self.s_factor).merge(self.s_ref, how='left', left_index=True, right_index=True)
        df.columns = ['factor', 'ref']
        corr = df.corr().loc['factor', 'ref']
        if corr < 0:
            self.s_factor = (-1 * self.s_factor) + 1
        return

    def eval_factor(self):
        """
        评估因子与参考数据的相关性和一致性
        """
        if self.s_ref is None:
            return
        s_ref = self.s_ref.resample('Q').last()
        s_factor = self.s_factor.resample('Q').last()
        ref_name = '参照指标'
        s_ref.name = ref_name
        s_factor.name = '因子序列'
        df = pd.DataFrame(s_factor)
        df = df.merge(s_ref, how='inner', left_index=True, right_index=True)

        eval_dict = OrderedDict()

        df_corr = df.corr()
        eval_dict['相关系数'] = round(df_corr.loc['因子序列', ref_name], 2)

        s_diff = df.diff().prod(axis=1)
        eval_dict['变动方向一致性'] = round(len(s_diff[s_diff > 0]) / len(s_diff), 2)

        ols = sm.OLS(df[ref_name], df['因子序列']).fit()
        eval_dict['R方'] = round(ols.rsquared, 2)
        eval_dict['系数P值'] = round(ols.pvalues.loc['因子序列'], 4)

        df_eval = pd.DataFrame(eval_dict.values(), index=eval_dict.keys(), columns=[ref_name]).transpose()
        self.df_eval = df_eval
        return

    def plot_factor(self):
        name = '' if self.factor_name is None else self.factor_name
        ref_name = '' if self.ref_name is None else self.ref_name
        title = self.plot_title if self.plot_title is not None else 'DFM Single Factor %s' % name
        if self.s_ref is None:
            self.s_factor.plot(title=title)
        else:
            df = pd.DataFrame()
            df['factor'] = self.s_factor.rolling(3).mean()
            ref_col = 'reference' if ref_name == '' else ref_name
            df[ref_col] = self.s_ref
            df.dropna().plot(secondary_y=ref_col, title=title)
        return
