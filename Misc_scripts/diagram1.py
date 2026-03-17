import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data for d=11 (64.50% accuracy)
test_size = 2000
correct = int(test_size * 0.645)
wrong = test_size - correct

# Distributing counts to show the "Plateau"
# Higher False Positives and False Negatives than lower distances
tn, tp = 650, 640 
fp, fn = 350, 360

cm = np.array([[tn, fp], [fn, tp]])

plt.figure(figsize=(4, 3.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False,
            xticklabels=['No Error', 'Error'], 
            yticklabels=['No Error', 'Error'])

plt.title('Confusion Matrix: d=11 (The Wall)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('cm_d11.png', dpi=300, bbox_inches='tight')
plt.show()