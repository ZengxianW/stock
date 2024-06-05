import efinance as ef

ef_df = ef.stock.get_realtime_quotes()  # 获得全部上一个交易日活跃的股票列表
# # ef.stock.get_quote_history("600519", beg="20200101", end="20210101", ) # 获得600519(茅台)，2020-01-01至2021-01-01的全部数据
# print(ef)

# 对于每一行，通过列名name访问对应的元素
for _, row in ef_df.iterrows():
    # print(row['市场类型'], row['股票代码']) # 输出每一行
    if row['市场类型'] == '深A':
        row['股票代码'] = 'sz.'+row['股票代码']
    elif row['市场类型'] == '沪A':
        row['股票代码'] = 'sh.'+row['股票代码']
    else:
        print(row['股票代码'])
