import efinance as ef

ef=ef.stock.get_realtime_quotes() # 获得全部上一个交易日活跃的股票列表
# # ef.stock.get_quote_history("600519", beg="20200101", end="20210101", ) # 获得600519(茅台)，2020-01-01至2021-01-01的全部数据
print(type(ef))

s=ef['股票代码'].tolist()
print(s[0:101])

