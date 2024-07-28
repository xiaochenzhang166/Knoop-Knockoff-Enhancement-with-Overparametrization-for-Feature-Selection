import pandas as pd

# 从 Excel 文件读取数据
df = pd.read_excel('Result_for_FS.xlsx')

# 将数据四舍五入至小数点后三位
df_rounded = df.round(3)  # 3表示保留三位小数，根据需要调整

# 将 DataFrame 转换为 LaTeX 格式的表格，并指定数值格式
latex_table = df_rounded.to_latex(index=False, float_format="%.3f")  # %.3f 表示保留三位小数

# 指定要保存的文件名和路径
file_path = 'FS result.txt'

# 将 LaTeX 代码写入 .txt 文件
with open(file_path, 'w') as file:
    file.write(latex_table)

print(f"LaTeX 代码已保存至文件：{file_path}")




