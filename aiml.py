import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Create synthetic classification dataset
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, 
                           n_redundant=0, random_state=42)

# 2. Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Initialize models
logreg = LogisticRegression()
tree = DecisionTreeClassifier()

# 4. Train models
logreg.fit(X_train, y_train)
tree.fit(X_train, y_train)

# 5. Predict and evaluate
logreg_pred = logreg.predict(X_test)
tree_pred = tree.predict(X_test)

logreg_acc = accuracy_score(y_test, logreg_pred)
tree_acc = accuracy_score(y_test, tree_pred)

print("📊 Logistic Regression Accuracy:", logreg_acc)
print("🌳 Decision Tree Accuracy:", tree_acc)

# 6. Plot decision boundaries
def plot_decision_boundary(model, X, y, title, ax):
    h = .02
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    ax.set_title(title)

# 7. Visualize both models
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

plot_decision_boundary(logreg, X, y, "Logistic Regression", ax1)
plot_decision_boundary(tree, X, y, "Decision Tree", ax2)

plt.tight_layout()
plt.show()


