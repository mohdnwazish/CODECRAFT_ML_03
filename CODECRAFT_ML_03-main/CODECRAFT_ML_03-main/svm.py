import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
import matplotlib.pyplot as plt

dataset_path = r"C:\Users\nwazish\OneDrive\Desktop\CodeCraft3\dogs-vs-cats\train"
# --------------------------------------

image_size = (64, 64)  # Resize all images to 64x64

def load_images_labels(dataset_path, max_images_per_class=2000):
    images = []
    labels = []
    cat_count = 0
    dog_count = 0

    # Ensure dataset path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset folder not found at: {dataset_path}")

    for filename in os.listdir(dataset_path):
        # Load cat images
        if filename.startswith('cat') and cat_count < max_images_per_class:
            img_path = os.path.join(dataset_path, filename)
            img = imread(img_path)
            img_resized = resize(img, image_size)
            images.append(img_resized)
            labels.append(0)  # Cat = 0
            cat_count += 1

        # Load dog images
        elif filename.startswith('dog') and dog_count < max_images_per_class:
            img_path = os.path.join(dataset_path, filename)
            img = imread(img_path)
            img_resized = resize(img, image_size)
            images.append(img_resized)
            labels.append(1)  # Dog = 1
            dog_count += 1

        # Stop when enough images are loaded
        if cat_count >= max_images_per_class and dog_count >= max_images_per_class:
            break

    return np.array(images), np.array(labels)

def extract_hog_features(images):
    hog_features = []
    for image in images:
        # Convert to grayscale
        if image.shape[-1] == 3:
            image_gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            image_gray = image

        features = hog(image_gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        hog_features.append(features)
    return np.array(hog_features)

def main():
    print("Loading images...")
    images, labels = load_images_labels(dataset_path)

    print(f"Loaded {len(images)} images")

    print("Extracting HOG features...")
    features = extract_hog_features(images)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print("Training SVM classifier...")
    svm_clf = SVC(kernel='rbf', C=10, gamma='scale')
    svm_clf.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = svm_clf.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))

    # Show some predictions
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    test_images = images[len(X_train):len(X_train)+9]
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(test_images[i])
        ax.axis('off')
        ax.set_title(f"Pred: {'Dog' if y_pred[i]==1 else 'Cat'}\nTrue: {'Dog' if y_test[i]==1 else 'Cat'}")
    plt.show()

if __name__ == "__main__":
    main()
