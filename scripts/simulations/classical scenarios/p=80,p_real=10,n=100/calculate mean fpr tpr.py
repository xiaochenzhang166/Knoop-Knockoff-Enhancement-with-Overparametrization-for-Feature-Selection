import numpy as np
import matplotlib.pyplot as plt

# F1 for Knoop ===========================================
fpr_lists_for_Knoop = np.load("Fpr values for knoop.npy")
average_fpr_for_Knoop = np.mean(fpr_lists_for_Knoop, axis=0)
tpr_lists_for_Knoop = np.load("Tpr values for knoop.npy")
average_tpr_for_Knoop = np.mean(tpr_lists_for_Knoop, axis=0)
# F1 for Knockoff ===========================================
fpr_lists_for_Knockoff = np.load("Fpr values for knockoff.npy")
average_fpr_for_Knockoff = np.mean(fpr_lists_for_Knockoff, axis=0)
tpr_lists_for_Knockoff = np.load("Tpr values for knockoff.npy")
average_tpr_for_Knockoff = np.mean(tpr_lists_for_Knockoff, axis=0)
# F1 for ridge ===========================================
fpr_lists_for_ridge = np.load("Fpr values for ridge.npy")
average_fpr_for_ridge = np.mean(fpr_lists_for_ridge, axis=0)
tpr_lists_for_ridge = np.load("Tpr values for ridge.npy")
average_tpr_for_ridge = np.mean(tpr_lists_for_ridge, axis=0)
# F1 for Pearson ===========================================
fpr_lists_for_Pearson = np.load("Fpr values for Pearson.npy")
average_fpr_for_Pearson = np.mean(fpr_lists_for_Pearson, axis=0)
tpr_lists_for_Pearson = np.load("Tpr values for Pearson.npy")
average_tpr_for_Pearson = np.mean(tpr_lists_for_Pearson, axis=0)
# F1 for Pearson ===========================================
fpr_lists_for_Pearson = np.load("Fpr values for Pearson.npy")
average_fpr_for_Pearson = np.mean(fpr_lists_for_Pearson, axis=0)
tpr_lists_for_Pearson = np.load("Tpr values for Pearson.npy")
average_tpr_for_Pearson = np.mean(tpr_lists_for_Pearson, axis=0)
# F1 for lasso ===========================================
fpr_lists_for_lasso = np.load("Fpr values for lasso.npy")
average_fpr_for_lasso = np.mean(fpr_lists_for_lasso, axis=0)
tpr_lists_for_lasso = np.load("Tpr values for lasso.npy")
average_tpr_for_lasso = np.mean(tpr_lists_for_lasso, axis=0)
# F1 for elasticNet ===========================================
fpr_lists_for_elasticNet = np.load("Fpr values for elasticNet.npy")
average_fpr_for_elasticNet = np.mean(fpr_lists_for_elasticNet, axis=0)
tpr_lists_for_elasticNet = np.load("Tpr values for elasticNet.npy")
average_tpr_for_elasticNet = np.mean(tpr_lists_for_elasticNet, axis=0)


plt.rc('font', family='Arial',size=12)
plt.plot(average_fpr_for_Knoop, average_tpr_for_Knoop, 'k-', lw=2, label='Knoop ROC curve')  # Black solid
plt.plot(average_fpr_for_Knockoff, average_tpr_for_Knockoff, 'r', lw=2, label='Knockoff ROC curve')  # Blue solid
plt.plot(average_fpr_for_ridge, average_tpr_for_ridge, 'b', lw=2, label='ridge ROC curve')  # Red dashed
plt.plot(average_fpr_for_lasso, average_tpr_for_lasso, 'orange', lw=2, label='lasso ROC curve', markersize=5)  # Red dotted with smaller 'x' markers
plt.plot(average_fpr_for_elasticNet, average_tpr_for_elasticNet, 'cyan', lw=2, label='elasticNet ROC curve', markersize=5)  # Blue dashed with smaller 'o' markers

plt.plot([0, 1], [0, 1], 'k:', lw=2, linestyle=':')  # Chance level with dotted line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Average False Positive Rate',size=12)
plt.ylabel('Average True Positive Rate',size=12)
plt.savefig('ROC2.pdf',dpi=1600)
plt.show()
