library(multiknockoffs)
# 从数据中提取 X 和 y
# 生成文件路径
X <- read.csv("E:/_Research/Benign-Overfitting-Research-SDU/May2024/real data/energy data/X_train.csv")
y <- read.csv("E:/_Research/Benign-Overfitting-Research-SDU/May2024/real data/energy data/y_train.csv")

# 将 y 中的数据框转换为数值向量
y <- as.numeric(y[[1]])
  
# 调用 run.pKO 函数
res.pKO <- run.pKO(X, y, pvals = TRUE)
  
# 生成文件路径并写入CSV文件
write.csv(res.pKO$pvals, file = "E:/_Research/Benign-Overfitting-Research-SDU/May2024/real data/energy data/pVal.csv", row.names = FALSE)
