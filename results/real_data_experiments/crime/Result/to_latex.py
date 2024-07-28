# import pandas as pd
#
# # 从 Excel 文件读取数据
# df = pd.read_excel('Result.xlsx', sheet_name='Sheet1')
#
# # 将数据四舍五入至小数点后三位
# df_rounded = df.round(3)  # 3表示保留三位小数，根据需要调整
#
# # 将 DataFrame 转换为 LaTeX 格式的表格，并指定数值格式
# latex_table = df_rounded.to_latex(index=False, float_format="%.3f")  # %.3f 表示保留三位小数
#
# # 指定要保存的文件名和路径
# file_path = 'latex code of knockoffBO vs other GLMs.txt'
#
# # 将 LaTeX 代码写入 .txt 文件
# with open(file_path, 'w') as file:
#     file.write(latex_table)
#
# print(f"LaTeX 代码已保存至文件：{file_path}")


import pandas as pd
from tabulate import tabulate

# 读取 Excel 文件
excel_file = 'Result.xlsx'  # 替换成你的 Excel 文件路径
df = pd.read_excel(excel_file)

# 将数值变量转换为科学计数法
df = df.applymap(lambda x: f'{x:.3e}' )




# 将 DataFrame 转换为 LaTeX 表格
latex_table = tabulate(df, headers='keys', tablefmt='latex_raw')

# 打印 LaTeX 表格
print(latex_table)




