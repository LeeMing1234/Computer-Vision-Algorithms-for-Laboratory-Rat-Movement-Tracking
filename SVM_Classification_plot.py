import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

movements = {
    "Eating": "C:/DeepLabCut_Data/Movement_Threshold/Eating_Speed.csv",
    "Standing": "C:/DeepLabCut_Data/Movement_Threshold/Standing_Speed.csv",
    "Inactive": "C:/DeepLabCut_Data/Movement_Threshold/Inactive_Speed.csv",
    "Abnormal": "C:/DeepLabCut_Data/Movement_Threshold/Abnormal_Speed.csv",
    "Drinking": "C:/DeepLabCut_Data/Movement_Threshold/Drinking_Speed.csv",
    "Social": "C:/DeepLabCut_Data/Movement_Threshold/Social_Speed.csv",
    "Walking": "C:/DeepLabCut_Data/Movement_Threshold/Walking_Speed.csv",
}

keypoints = [
    "Right forepaw", "Tail tip", "Tail center", "Tail base", "Left ear", "Right ear",
    "Left hind paw", "Right hind paw", "Left forepaw", "Nose", "Abdomen", "Flank",
    "Lumber", "Shoulder", "Nape", "Left eye", "Mouse", "Right eye"
]

model_path = "C:/DeepLabCut_Data/SVM/svm_model.pkl"

def plot_svm_classification(movements_dict, keypoints, model_path, output_path='C:/DeepLabCut_Data/SVM/Classification_Plot/svm_classification_plot.png'):
    # Load the model and class labels
    model, class_labels = joblib.load(model_path)

    # Recreate the dataset for plotting
    X, y = [], []
    expected_features = None
    for label, (movement_name, csv_path) in enumerate(movements_dict.items()):
        df = pd.read_csv(csv_path)
        speed_cols = [col for col in df.columns if 'Speed' in col and any(kp in col for kp in keypoints)]

        if expected_features is None:
            expected_features = len(speed_cols)

        for i in range(len(df)):
            speeds = df.loc[i, speed_cols].values
            if np.any(pd.isna(speeds)):
                continue
            if len(speeds) != expected_features:
                continue
            X.append(speeds)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Reduce features to 2D using PCA
    pca = PCA(n_components=2)
    X_2D = pca.fit_transform(X)

    # Train a new SVM on the 2D data just for visualization
    viz_model = SVC(kernel='rbf', C=10, gamma='scale')
    viz_model.fit(X_2D, y)

    # Create mesh grid for decision boundary
    x_min, x_max = X_2D[:, 0].min() - 1, X_2D[:, 0].max() + 1
    y_min, y_max = X_2D[:, 1].min() - 1, X_2D[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    Z = viz_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    scatter = plt.scatter(X_2D[:, 0], X_2D[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')

    # Create legend
    handles = [plt.Line2D([], [], marker='o', linestyle='', color=plt.cm.coolwarm(i / (len(class_labels)-1)),
                          label=name) for i, name in enumerate(class_labels)]
    plt.legend(handles=handles, title="Movement Class")

    plt.title("SVM Classification Plot (2D PCA Projection)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path}")

    # Optionally show the plot
    # plt.show()
    plt.close()

plot_svm_classification(movements, keypoints, model_path)
