import pandas as pd 
import numpy as np 
import datetime as dt 
from copy import deepcopy
from IPython.display import display
from collections import Counter
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import matplotlib.pyplot as plt


def infer_frequency(dt_index):
    # 计算索引中连续日期时间的差异
    if len(dt_index) < 2:
        return None
    
    # 计算所有连续时间点之间的差异
    deltas = [dt_index[i+1] - dt_index[i] for i in range(len(dt_index) - 1)]
    
    # 使用Counter找出最常见的差异
    delta_counts = Counter(deltas)
    most_common_delta, _ = delta_counts.most_common(1)[0]
    
   # 将时间差转换为字符串描述的频率
    delta_days = most_common_delta.days
    
    frequency_map = {
        1: 'D',  # Daily
        7: 'W',  # Weekly
        10: '10D',  # 旬
        30: 'M',  # Monthly
        31: 'M', 
        90: 'Q',  # Quarterly
        91: 'Q', 
        92: 'Q', 
        14: '14D',  # 半月
        360: 'Y',  # 年
        365: 'Y',  # 年
        366: 'Y',  # 年
    }
    return frequency_map.get(delta_days)


def check_time_series(df_raw, print_res=False):
    # 初始化一个空的DataFrame来存储结果
    results = pd.DataFrame(columns=['Start Date', 'End Date', 'Non-Empty Length', 'Frequency'])

    # 遍历每一列
    for column in df_raw.columns:
        # 获取非空数据的索引
        non_empty = df_raw[column].dropna()
        
        if not non_empty.empty:
            # 计算最早的非空开始日期和最晚的非空结束日期
            start_date = non_empty.index.min()
            end_date = non_empty.index.max()
            
            # 计算非空时间序列的长度
            non_empty_length = len(non_empty)
            
            # 识别DateTimeIndex的频率
            frequency = infer_frequency(non_empty.index)
            
            # 将结果存储到DataFrame中
            results.loc[column] = [start_date, end_date, non_empty_length, frequency]
        else:
            # 如果整列都是空的，记录为NaN或适当的标记
            results.loc[column] = [pd.NaT, pd.NaT, 0, None]
    if print_res:
        display(results)
    return results

def process_raw_yoy(s_raw, freq=None, to_monthly=True):
    # 将从Wind下载的原始YOY数据转换为标准YOY数据
    s_raw = deepcopy(s_raw)
    if freq is None:
        freq = infer_frequency(s_raw.index)
    if to_monthly:
        freq = 'M'
        s_raw = s_raw.resample(freq).last()
    s_raw = s_raw.interpolate().dropna() * 0.01
    return s_raw

def current_to_yoy(s_raw, freq=None, to_monthly=True):
    # 将当前值/总量值/年累计值转为YoY
    s_raw = deepcopy(s_raw)
    if freq is None:
        freq = infer_frequency(s_raw.index)
    if to_monthly:
        freq = 'M'
        s_raw = s_raw.resample(freq).last()
    period_dict = {'M': 12, 'Q': 4, 'Y': 1}
    return s_raw.pct_change(period_dict[freq], fill_method=None).interpolate()


def ror_to_index(s_raw, freq=None, to_monthly=True):
    # 环比转总量类指数
    s_raw = deepcopy(s_raw)
    if to_monthly:
        freq = 'M'
        s_raw = s_raw.resample(freq).last()
    s_raw = s_raw.interpolate().dropna() * 0.01
    s_raw.iloc[0] = 0
    s_raw = (s_raw + 1).cumprod()
    return s_raw

def ratio_to_yoy(s_raw, freq=None, to_monthly=True):
    # 将比率值转为YOY 
    s_raw = deepcopy(s_raw)
    if freq is None:
        freq = infer_frequency(s_raw.index)
    if to_monthly:
        freq = 'M'
        s_raw = s_raw.resample(freq).last()
    s_raw = s_raw.interpolate().dropna() * 0.01
    period_dict = {'M': 12, 'Q': 4, 'Y': 1}
    return s_raw.diff(period_dict[freq]).dropna()

def diffusion_to_index(s_raw, freq=None, to_monthly=True):
    s_raw = deepcopy(s_raw)
    if freq is None:
        freq = infer_frequency(s_raw.index)
    if to_monthly:
        freq = 'M'
        s_raw = s_raw.resample(freq).last()
    s = 1 + (s_raw - 50) / 100 
    return s.interpolate().dropna()


def remove_outliers(ts, method='iqr', factor=1.5):
    """
    剔除时间序列中的异常值。

    参数:
    ts (pd.Series): 时间序列数据。
    method (str): 检测异常值的方法，'iqr' 或 'std'。
    factor (float): 异常值检测的因子，默认为1.5。

    返回:
    pd.Series: 剔除异常值后的时间序列。
    """
    if method == 'iqr':
        Q1 = ts.quantile(0.25)
        Q3 = ts.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
    elif method == 'std':
        mean = ts.mean()
        std = ts.std()
        lower_bound = mean - factor * std
        upper_bound = mean + factor * std
    else:
        raise ValueError("method 参数必须是 'iqr' 或 'std'")

    return ts[(ts >= lower_bound) & (ts <= upper_bound)]


def check_seasonality(ts, freq=None):
    """
    判断时间序列是否存在季节性。

    参数:
    ts (pd.Series): 时间序列数据。
    freq (int): 季节性频率。如果为 None，将自动推断。

    返回:
    bool: 如果存在显著季节性，返回 True；否则返回 False。
    """
    if freq is None:
        freq = infer_frequency(ts.index)
        if freq is None:
            raise ValueError("无法推断时间序列的频率，请手动指定频率。")
    freq_num = 12 if freq == 'M' else 4 if freq == 'Q' else 1
    decomposition = seasonal_decompose(ts, model='additive', period=freq_num)
    seasonal = decomposition.seasonal
    
    # 使用 Ljung-Box 检验来判断季节性成分是否显著
    lb_test = acorr_ljungbox(seasonal, lags=[freq_num], return_df=True)
    p_value = lb_test['lb_pvalue'].values[0]
    
    # 如果 p 值小于显著性水平（例如 0.05），则认为存在显著的季节性
    if p_value < 0.05:
        return True
    else:
        return False

def get_stl_components(ts, freq=None, components='trend'):
    if len(ts) != len(ts.dropna()):
        raise ValueError("时间序列中存在NaN值，无法进行STL分解。")
    if freq is None:
        freq = infer_frequency(ts.index)
    if freq is None:
        raise ValueError("无法推断时间序列的频率，请手动指定频率。")
    freq_num = 12 if freq == 'M' else 4 if freq == 'Q' else 1
    decomposition = STL(ts, period=freq_num).fit()
    if components == 'trend':
        new_ts = decomposition.trend
    elif components == 'seasonal':
        new_ts = decomposition.seasonal
    elif components == 'resid':
        new_ts = decomposition.resid
    else:
        raise ValueError("components 参数必须是 'trend'、'seasonal' 或 'resid'")
    new_ts.name = ts.name 
    return new_ts 

def bry_boschan(data, window=13):
    """
    Bry-Boschan算法用于识别时间序列中的峰值和谷值。

    参数:
    data (pd.Series): 时间序列数据。
    window (int): 平滑窗口大小，默认为13。

    返回:
    tuple: 包含两个列表，分别是识别出的峰值和谷值的索引。
    """
    def filter_extremes(extremes, min_cycle_length):
        """
        过滤极值，确保相邻极值之间的最小周期长度。

        参数:
        extremes (list): 初步识别的极值索引列表。
        min_cycle_length (pd.Timedelta): 最小周期长度。

        返回:
        list: 过滤后的极值索引列表。
        """
        filtered = [extremes[0]]
        for extreme in extremes[1:]:
            if extreme - filtered[-1] >= min_cycle_length:
                filtered.append(extreme)
        return filtered

    # 平滑数据
    smoothed_data = data.rolling(window=window, center=True).mean()
    
    # 初步识别峰值和谷值
    peaks = (smoothed_data.shift(1) < smoothed_data) & (smoothed_data.shift(-1) < smoothed_data)
    troughs = (smoothed_data.shift(1) > smoothed_data) & (smoothed_data.shift(-1) > smoothed_data)
    
    # 应用规则
    min_cycle_length = pd.Timedelta(days=window)
    peaks = peaks[peaks].index
    troughs = troughs[troughs].index
    
    if len(peaks) == 0 or len(troughs) == 0:
        return [], []
    
    # 过滤极值
    filtered_peaks = filter_extremes(peaks, min_cycle_length)
    filtered_troughs = filter_extremes(troughs, min_cycle_length)
    
    return filtered_peaks, filtered_troughs

def determine_cycle_phase(data, window=13, plot=False, figsize=None, s_ref=None):
    """
    确定时间序列处于上升周期还是下降周期。

    参数:
    data (pd.Series): 时间序列数据。
    window (int): 平滑窗口大小，默认为12。
    plot (bool): 是否绘制时间序列图，默认为False。
    figsize (tuple): 图形大小，默认为None。

    返回:
    pd.DataFrame: 包含时间序列和周期状态的DataFrame。
    """
    peaks, troughs = bry_boschan(data, window)
    cycle_phase = pd.Series(index=data.index, dtype='object')

    # 合并峰值和谷值并排序
    turning_points = sorted(peaks + troughs)
    
    # 确定每个时间点的周期状态
    for i in range(len(turning_points) - 1):
        start, end = turning_points[i], turning_points[i + 1]
        if start in peaks:
            cycle_phase[start:end] = 'down'
        else:
            cycle_phase[start:end] = 'up'
    
    # 处理最后一个区间
    if turning_points:
        last_point = turning_points[-1]
        if last_point in peaks:
            cycle_phase[last_point:] = 'down'
        else:
            cycle_phase[last_point:] = 'up'
    
    df = pd.DataFrame(data, index=data.index, columns=[data.name])
    df['cycle_phase'] = cycle_phase
    
    if plot:
        plot_cycle_phases(df, figsize, s_ref)
    
    return df

def plot_cycle_phases(df, figsize=None, s_ref=None):
    """
    绘制时间序列图，其中上行区间用绿色阴影覆盖，下行区间用红色阴影覆盖。

    参数:
    df (pd.DataFrame): 包含时间序列和周期状态的DataFrame。
    figsize (tuple): 图形大小，默认为None
    s_ref (pd.Series): 参考序列，默认为None
    """
    data = df.iloc[:, 0]
    cycle_phase = df['cycle_phase']
    turning_points = cycle_phase.dropna().index
    
    if figsize is None:
        figsize = (14, 7)
    plt.figure(figsize=figsize)
    ax1 = plt.gca()
    ax1.plot(data, label=data.name)

    if s_ref is not None:
        ax2 = ax1.twinx()
        ax2.plot(s_ref, label=s_ref.name, color='orange')
        ax2.set_ylabel(s_ref.name)
    for i in range(len(turning_points) - 1):
        start, end = turning_points[i], turning_points[i + 1]
        if cycle_phase[start] == 'down':
            plt.axvspan(start, end, color='red', alpha=0.3)
        else:
            plt.axvspan(start, end, color='green', alpha=0.3)
    
    # 绘制最后一个区间
    if len(turning_points) > 0:
        last_point = turning_points[-1]
        if cycle_phase[last_point] == 'down':
            plt.axvspan(last_point, data.index[-1], color='red', alpha=0.3)
        else:
            plt.axvspan(last_point, data.index[-1], color='green', alpha=0.3)
    
    plt.legend()
    plt.title(f'{df.columns[0].upper()} Factor Time Series with Cycle Phases')
    plt.show()


def max_drawdowns(ts, max_duration=365, plot=True, plot_top_n=3, figsize=(20, 5), direction='down', 
                    change_type='value'):
	"""计算最大回撤或上涨
	Args:
		ts (pd.Series): 时间序列数据
		max_duration (int): 最大持续时间限制(天数)
		plot (bool): 是否绘制图
		plot_top_n (int): 绘制前n个最大波动
		figsize (tuple): 图形大小
		direction (str): 计算方向，'down'表示计算回撤，'up'表示计算上涨，默认为'down'
		change_type (str): 字段change代表的含义，'value'表示数值，'pct'表示百分比，默认为'value'
	Returns:
		pd.DataFrame: 包含波动信息的DataFrame
	"""
	if direction not in ['up', 'down']:
		raise ValueError("direction参数必须是'up'或'down'")
		
	dates = ts.index.tolist()
	n = len(dates)
	values = []
	
	# 使用向量化操作计算波动
	ts_array = ts.values
	for i in range(n-1):
		# 计算从当前点开始到未来所有点的波动
		future_values = ts_array[i+1:]
		future_dates = dates[i+1:]
		
		# 使用广播计算所有可能的波动
		changes = future_values - ts_array[i]
		if direction == 'down':
			changes = -changes  # 对于回撤，我们关注负向变化
			
		durations = [(end_date - dates[i]).days for end_date in future_dates]
		
		# 筛选符合duration限制的波动
		valid_indices = [j for j, d in enumerate(durations) if d <= max_duration]
		
		if valid_indices:
			for j in valid_indices:
				values.append([
					changes[j],
					durations[j],
					dates[i],
					future_dates[j],
					ts_array[i],
					future_values[j]
				])
	
	# 创建DataFrame并排序
	df_changes = pd.DataFrame(
		values,
		columns=['change', 'duration', 'start_date', 'end_date', 'start_value', 'end_value']
	)
	df_changes = df_changes.sort_values(by='change', ascending=False).reset_index(drop=True)
	
	# 使用集合操作优化日期范围重叠检查
	used_ix = [0]
	used_ranges = [(df_changes.loc[0, 'start_date'], df_changes.loc[0, 'end_date'])]
	
	for ix in df_changes.index[1:]:
		start_date = df_changes.loc[ix, 'start_date']
		end_date = df_changes.loc[ix, 'end_date']
		
		# 检查是否与已有区间重叠
		overlap = False
		for used_start, used_end in used_ranges:
			if (used_start <= start_date <= used_end or 
				used_start <= end_date <= used_end or
				(start_date <= used_start and end_date >= used_end)):
				overlap = True
				break
		
		if not overlap:
			used_ranges.append((start_date, end_date))
			used_ix.append(ix)
	
	# 筛选不重叠的波动区间
	df_changes = df_changes.loc[used_ix].reset_index(drop=True)
	
	if change_type == 'pct':
		df_changes['change'] = df_changes['change'] / df_changes['start_value']
	
	# 绘图
	if plot:
		plot_drawdowns(ts, df_changes[:plot_top_n], figsize, direction=direction)
	
	return df_changes

def plot_drawdowns(ts, df_drawdown, figsize=None, direction='down'):
	"""
	绘制时间序列及其波动区间，并在图中标注每个区间的变化比例。

	参数:
	ts (pd.Series): 时间序列数据
	df_drawdown (pd.DataFrame): 包含波动信息的DataFrame
	figsize (tuple): 图形大小，默认为None
	direction (str): 计算方向，'down'表示回撤，'up'表示上涨
	"""
	if figsize is None:
		figsize = (20, 5)
	plt.figure(figsize=figsize)
	plt.plot(ts, label='Time Series')
	
	# 根据方向选择颜色
	color = 'red' if direction == 'down' else 'green'
	
	for _, row in df_drawdown.iterrows():
		plt.axvspan(row['start_date'], row['end_date'], color=color, alpha=0.3)
		change_percentage = (row['end_value'] - row['start_value']) / row['start_value'] * 100
		mid_point = row['start_date'] + (row['end_date'] - row['start_date']) / 2
		
		# Find the nearest index to the mid_point using asof
		nearest_date = ts.index.asof(mid_point)
		
		plt.text(nearest_date, ts.loc[nearest_date], f'{change_percentage:.2f}%', 
				 horizontalalignment='center', verticalalignment='bottom', color='black')
	
	plt.legend()
	title_type = 'Drawdowns' if direction == 'down' else 'Uptrends'
	plt.title(f'{ts.name.upper()} Factor Time Series with Max {title_type}')
	plt.show()

def get_last_quarter_month(current_date=None):
    """
    获取当前日期三个月前之前最近的一个季度月
    :return: pandas.Timestamp, 最近的季度月
    """
    # 取当前日期
    if current_date is None:
        current_date = pd.Timestamp.now().date()

    # 计算三个月前的日期
    three_months_ago = current_date - pd.DateOffset(months=4)

    # 获取最近的季度月
    quarter_months = [3, 6, 9, 12]
    month = three_months_ago.month

    # 找到最近的季度月
    for qm in reversed(quarter_months):
        if month >= qm:
            quarter_month = qm
            break
    else:
        quarter_month = 12  # 如果没有找到，默认取上一年的12月

    # 构造最近的季度月日期
    last_quarter_date = pd.Timestamp(year=three_months_ago.year, month=quarter_month, day=1) + pd.DateOffset(months=1) - pd.DateOffset(days=1)
    return last_quarter_date


def calculate_lagged_correlation(s_factor, s_ref, max_lag, double_sided=False, plot=False):
	"""
	计算s_factor在不同滞后阶数下与s_ref的相关性，并可选择绘图

	:param s_factor: pd.Series, 因子序列
	:param s_ref: pd.Series, 参考序列
	:param max_lag: int, 最大滞后阶数
	:param plot: bool, 是否绘制相关性图
	:return: pd.DataFrame, 每个滞后阶数下的相关性
	"""
	correlations = {}
	if double_sided:
		range_lag = range(-1*max_lag, max_lag+1)
	else:
		range_lag = range(0, max_lag+1)
	for lag in range_lag:
		correlations[lag] = s_factor.shift(lag).corr(s_ref)
	df_correlations = pd.DataFrame.from_dict(correlations, orient='index', columns=['correlation'])

	if plot:
		factor_name = s_factor.name
		ref_name = s_ref.name
		if factor_name is not None and ref_name is not None:
			title = f'{factor_name} and {ref_name} Lagged Correlation'
		else:
			title = 'Lagged Correlation'
		df_correlations.plot(title=title)
		plt.xlabel('Lag')
		plt.ylabel('Correlation')
		plt.show()
	
	return df_correlations


def calculate_lagged_kl_divergence(s_factor, s_ref, max_lag, double_sided=False, plot=False, bins=50):
	"""
	计算s_factor在不同滞后阶数下与s_ref的K-L信息量，并可选择绘图

	:param s_factor: pd.Series, 因子序列
	:param s_ref: pd.Series, 参考序列
	:param max_lag: int, 最大滞后阶数
	:param plot: bool, 是否绘制K-L信息量图
	:param bins: int, 用于直方图的箱数
	:return: pd.DataFrame, 每个滞后阶数下的K-L信息量
	"""
	df = pd.concat([s_factor, s_ref], axis=1).dropna(how='all').interpolate().dropna()
	factor_name = s_factor.name
	ref_name = s_ref.name
	if factor_name is not None and ref_name is not None:
		s_factor = df[factor_name]
		s_ref = df[ref_name]
	else:
		df.columns = ['factor', 'ref']
		s_factor = df['factor']
		s_ref = df['ref']
	# 标准化数据
	scaler = StandardScaler()
	s_factor_scaled = pd.Series(scaler.fit_transform(s_factor.values.reshape(-1, 1)).flatten(), index=s_factor.index)
	s_ref_scaled = pd.Series(scaler.fit_transform(s_ref.values.reshape(-1, 1)).flatten(), index=s_ref.index)
	
	kl_divergences = {}

	if double_sided:
	    range_lag = range(-1*max_lag, max_lag+1)
	else:
	    range_lag = range(0, max_lag+1)

	for lag in range_lag:
		shifted_s_factor = s_factor_scaled.shift(lag).dropna()
		aligned_s_ref = s_ref_scaled.loc[shifted_s_factor.index]
		
		# 转换为概率分布
		p_factor, _ = np.histogram(shifted_s_factor, bins=bins, density=True)
		p_ref, _ = np.histogram(aligned_s_ref, bins=bins, density=True)
		
		# 避免零概率
		p_factor = np.where(p_factor == 0, 1e-10, p_factor)
		p_ref = np.where(p_ref == 0, 1e-10, p_ref)
		
		kl_divergences[lag] = entropy(p_factor, p_ref)
	df_kl_divergences = pd.DataFrame.from_dict(kl_divergences, orient='index', columns=['kl_divergence'])
	
	if plot:
		df_kl_divergences.plot(title='Lagged K-L Divergence')
		plt.xlabel('Lag')
		plt.ylabel('K-L Divergence')
		plt.show()
	
	return df_kl_divergences


def calculate_lagged_dtw_distance(s_factor, s_ref, max_lag, double_sided=False, plot=False):
	"""
	计算s_factor在不同滞后阶数下与s_ref的DTW距离，并可选择绘图

	:param s_factor: pd.Series, 因子序列
	:param s_ref: pd.Series, 参考序列
	:param max_lag: int, 最大滞后阶数
	:param plot: bool, 是否绘制DTW距离图
	:return: pd.DataFrame, 每个滞后阶数下的DTW距离
	"""
	df = pd.concat([s_factor, s_ref], axis=1).dropna(how='all').interpolate().dropna()
	factor_name = s_factor.name
	ref_name = s_ref.name
	if factor_name is not None and ref_name is not None:
		s_factor = df[factor_name]
		s_ref = df[ref_name]
	else:
		df.columns = ['factor', 'ref']
		s_factor = df['factor']
		s_ref = df['ref']
	
	scaler = StandardScaler()
	s_factor_scaled = pd.Series(scaler.fit_transform(s_factor.values.reshape(-1, 1)).flatten(), index=s_factor.index)
	s_ref_scaled = pd.Series(scaler.fit_transform(s_ref.values.reshape(-1, 1)).flatten(), index=s_ref.index)
	
	dtw_distances = {}
	if double_sided:
	    range_lag = range(-1*max_lag, max_lag+1)
	else:
	    range_lag = range(0, max_lag+1)

	for lag in range_lag:
		shifted_s_factor = s_factor_scaled.shift(lag).dropna()
		aligned_s_ref = s_ref_scaled.loc[shifted_s_factor.index]

		shifted_s_factor.index = np.arange(len(shifted_s_factor))
		aligned_s_ref.index = np.arange(len(aligned_s_ref))

		arr_factor = np.array(shifted_s_factor.reset_index())
		arr_ref = np.array(aligned_s_ref.reset_index())
	
		distance, _ = fastdtw(arr_factor, arr_ref, dist=euclidean)
		dtw_distances[lag] = distance
	df_dtw_distances = pd.DataFrame.from_dict(dtw_distances, orient='index', columns=['dtw_distance'])
	
	if plot:
		df_dtw_distances.plot(title='Lagged DTW Distance')
		plt.xlabel('Lag')
		plt.ylabel('DTW Distance')
		plt.show()
	
	return df_dtw_distances


def get_lagged_info(s_factor, s_ref, max_lag, double_sided=False, plot=False):
    """
    获取领先滞后信息，包括最大相关系数、最小KL散度和最小DTW距离

    :param s_factor: pd.Series, 因子序列
    :param s_ref: pd.Series, 参考序列
    :param max_lag: int, 最大滞后阶数
    :param plot: bool, 是否绘制图表 
    :return: pd.DataFrame, 滞后信息
    """
    df_lagged_correlation = calculate_lagged_correlation(s_factor, s_ref, max_lag=max_lag)
    df_lagged_kl = calculate_lagged_kl_divergence(s_factor, s_ref, max_lag=max_lag)
    df_lagged_dtw = calculate_lagged_dtw_distance(s_factor, s_ref, max_lag=max_lag)

    df = pd.DataFrame()
    for df_lagged in [df_lagged_correlation, df_lagged_kl, df_lagged_dtw]:
        df = pd.concat([df, df_lagged], axis=1)

    corr = df_lagged_correlation.loc[0, 'correlation']
    max_corr_lag = df[df['correlation']==df['correlation'].max()].index[0]
    min_kl_lag = df[df['kl_divergence']==df['kl_divergence'].min()].index[0]
    min_dtw_lag = df[df['dtw_distance']==df['dtw_distance'].min()].index[0]

    df_lagged_info = pd.DataFrame({
        '指标': ['同步相关系数', '最大时差相关系数', '最小时差KL散度', '最小时差DTW距离'], 
        '指标值': [corr, df['correlation'].max(), df['kl_divergence'].min(), df['dtw_distance'].min()],
        '对应滞后期': [0, max_corr_lag, min_kl_lag, min_dtw_lag],
    }).set_index('指标')

    print(f'因子名称：{s_factor.name}')
    print(f'参考序列名称：{s_ref.name}')
    print('统计指标：')
    print(f'同步相关系数：{round(corr, 2)}')
    print(f'最大时差相关系数：{round(df["correlation"].max(), 2)}, 领先阶数：{max_corr_lag}')
    print(f'最小时差KL散度：{round(df["kl_divergence"].min(), 2)}, 领先阶数：{min_kl_lag}')
    print(f'最小时差DTW距离：{round(df["dtw_distance"].min(), 2)}, 领先阶数：{min_dtw_lag}')

    if plot:
        fig, axs = plt.subplots(3, 1, figsize=(20, 15))
        for i, col in enumerate(df.columns):
            axs[i].plot(df[col], label=col)
            axs[i].set_title(col)
            axs[i].legend()
        plt.show()

    return df_lagged_info
