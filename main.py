#Project 1 : Kannada MNIST - Classification Problem

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score,confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc




'''
Step 1:Data Collection and Preprocessing:
Extract the dataset from the npz file from the downloaded dataset.
'''

# Load the training data 
train_data = np.load(r"D:\\IT\\GUVI\\Final project\\Project 1\\Dataset\\X_kannada_MNIST_train.npz")
X_train = train_data['arr_0']

#Load the training labels
train_labels = np.load(r"D:\\IT\\GUVI\\Final project\\Project 1\\Dataset\\y_kannada_MNIST_train.npz")
y_train = train_labels['arr_0']

#Load the testing data
test_data = np.load(r"D:\\IT\\GUVI\\Final project\\Project 1\\Dataset\\X_kannada_MNIST_test.npz")
X_test = test_data['arr_0']

#Load the testing labels
test_labels = np.load(r"D:\\IT\\GUVI\\Final project\\Project 1\\Dataset\\y_kannada_MNIST_test.npz")
y_test = test_labels['arr_0']

# Print the shapes of the loaded data
print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Test data shape:", X_test.shape)

#To check the size of each image
# Load the npz file
dataset=np.load(r'D:\\IT\\GUVI\\Final project\\Project 1\\Dataset\\X_kannada_MNIST_train.npz')

# Extract the data
x_train = dataset['arr_0']

# Print the shape of the extracted data
print("Training data shape:", x_train.shape)

#To display the images
some_digit=X_train[90]
some_digit_img=some_digit.reshape(28,28)
plt.imshow(some_digit_img,cmap=matplotlib.cm.binary,interpolation="nearest")
y_train[90]

'''
Step 2:Performing PCA to data
Training and Testing of  images in 10 dimensions instead of 28X28
'''

X_train_2d = X_train.reshape(X_train.shape[0], -1)
X_test_2d = X_test.reshape(X_test.shape[0], -1)

for n in range (10,31,5):


    #Performing PCA to n Components
    print(f"PCA for {n}components")
    pca = PCA(n_components=n)

    X_train_pca = pca.fit_transform(X_train_2d)
    X_test_pca = pca.transform(X_test_2d)

    y_train[1].dtype

    '''Step 3:Applying various Models:
    '''

    dt_classifier = DecisionTreeClassifier()
    rf_classifier = RandomForestClassifier()
    nb_classifier = GaussianNB()
    knn_classifier = KNeighborsClassifier(n_neighbors=8)
    svm_classifier = SVC(probability=True)


    '''Step 4:Evaluate each model
    '''

    def evaluate_model(model, x_train, y_train, x_test, y_test):
        # Fit the model on the training data
        model.fit(x_train, y_train)

        # Predict labels for the test data
        y_pred = model.predict(x_test)

        # Calculate accuracy, f1-score, and recall
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')

        # Calculate ROC-AUC score
        y_pred_proba = model.predict_proba(x_test)
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        
        ## Calculating Confusion Matrix
        conf_matrix=confusion_matrix(y_test,y_pred)

        # Return evaluation metrics
        return accuracy, f1, recall, roc_auc, conf_matrix

    #Evaluation for Decision Tree
    #Decision Tree

    dt_accuracy, dt_f1, dt_recall, dt_roc_auc, dt_confusion_matrix = evaluate_model(dt_classifier, X_train_pca, y_train, X_test_pca, y_test)
    print("Decision Tree Accuracy:", dt_accuracy)
    print("Decision Tree F1-score:", dt_f1)
    print("Decision Tree Recall:", dt_recall)
    print("Decision Tree ROC-AUC:", dt_roc_auc)
    print("Decision Tree Confusion Matrix:")
    print(dt_confusion_matrix)

    #Evaluation for Random Forest
    #Random Forest

    rf_accuracy, rf_f1, rf_recall, rf_roc_auc, rf_confusion_matrix = evaluate_model(rf_classifier, X_train_pca, y_train, X_test_pca, y_test)
    print("Random Forest Accuracy:", rf_accuracy)
    print("Random Forest F1-score:", rf_f1)
    print("Random Forest Recall:", rf_recall)
    print("Random Forest ROC-AUC:", rf_roc_auc)
    print("Random Forest Confusion Matrix",rf_confusion_matrix)


    #Evaluation for Naive Bayes
    #Naive Bayes

    nb_accuracy, nb_f1, nb_recall, nb_roc_auc, nb_confusion_matrix = evaluate_model(nb_classifier, X_train_pca, y_train, X_test_pca, y_test)
    print("Naive Bayes Accuracy:", nb_accuracy)
    print("Naive Bayes F1-score:", nb_f1)
    print("Naive Bayes Recall:", nb_recall)
    print("Naive Bayes ROC-AUC:", nb_roc_auc)
    print("Naive Bayes Confusion Matrix:")
    print(nb_confusion_matrix)


    #Evaluation for K-NN
    #K-Nearest Neighbour

    knn_accuracy, knn_f1, knn_recall, knn_roc_auc, knn_confusion_matrix = evaluate_model(knn_classifier, X_train_pca, y_train, X_test_pca, y_test)
    print("K-NN Accuracy:", knn_accuracy)
    print("K-NN F1-score:", knn_f1)
    print("K-NN Recall:", knn_recall)
    print("K-NN ROC-AUC:", knn_roc_auc)
    print("K-NN Confusion Matrix")
    print(knn_confusion_matrix)


    #Evaluation for SVM
    #Support Vector Machine
    svm_classifier = SVC(probability=True)
    svm_accuracy, svm_f1, svm_recall, svm_roc_auc, svm_confusion_matrix = evaluate_model(svm_classifier, X_train_pca, y_train, X_test_pca, y_test)
    print("SVM Accuracy:", svm_accuracy)
    print("SVM F1-score:", svm_f1)
    print("SVM Recall:", svm_recall)
    print("SVM ROC-AUC:", svm_roc_auc)
    print("SVM confusion Matrix")
    print(svm_confusion_matrix)

    # Convert the true labels to one-hot encoded format
    y_test_bin = label_binarize(y_test, classes=range(10))

    # For Decision Trees
    dt_pred_probs = dt_classifier.predict_proba(X_test_pca)
    dt_fpr, dt_tpr, _ = roc_curve(y_test_bin.ravel(), dt_pred_probs.ravel())
    dt_auc = auc(dt_fpr, dt_tpr)

    # For Random Forest
    rf_pred_probs = rf_classifier.predict_proba(X_test_pca)
    rf_fpr, rf_tpr, _ = roc_curve(y_test_bin.ravel(), rf_pred_probs.ravel())
    rf_auc = auc(rf_fpr, rf_tpr)

    # For Naive Bayes
    nb_pred_probs = nb_classifier.predict_proba(X_test_pca)
    nb_fpr, nb_tpr, _ = roc_curve(y_test_bin.ravel(), nb_pred_probs.ravel())
    nb_auc = auc(nb_fpr, nb_tpr)

    # For K-NN
    knn_pred_probs = knn_classifier.predict_proba(X_test_pca)
    knn_fpr, knn_tpr, _ = roc_curve(y_test_bin.ravel(), knn_pred_probs.ravel())
    knn_auc = auc(knn_fpr, knn_tpr)

    # Plot micro-averaged ROC curves for each classifier
    plt.figure(figsize=(8, 6))
    plt.plot(dt_fpr, dt_tpr, label='Decision Trees (AUC = {:.2f})'.format(dt_auc))
    plt.plot(rf_fpr, rf_tpr, label='Random Forest (AUC = {:.2f})'.format(rf_auc))
    plt.plot(nb_fpr, nb_tpr, label='Naive Bayes (AUC = {:.2f})'.format(nb_auc))
    plt.plot(knn_fpr, knn_tpr, label='K-NN (AUC = {:.2f})'.format(knn_auc))
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Micro-Averaged ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Convert the true labels to one-hot encoded format
    y_test_bin = label_binarize(y_test, classes=range(10))

    # For Decision Trees
    dt_fpr = dict()
    dt_tpr = dict()
    dt_auc = dict()
    for class_idx in range(10):
        dt_pred_probs = dt_classifier.predict_proba(X_test_pca)[:, class_idx]
        dt_fpr[class_idx], dt_tpr[class_idx], _ = roc_curve(y_test_bin[:, class_idx], dt_pred_probs)
        dt_auc[class_idx] = auc(dt_fpr[class_idx], dt_tpr[class_idx])


    # Plot ROC curves for each class for Decision Trees
    plt.figure(figsize=(8, 6))
    for class_idx in range(10):
        plt.plot(dt_fpr[class_idx], dt_tpr[class_idx], label='Class {} (AUC = {:.2f})'.format(class_idx, dt_auc[class_idx]))

    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve for Decision Trees')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # For Random Forest
    rf_fpr = dict()
    rf_tpr = dict()
    rf_auc = dict()
    for class_idx in range(10):
        rf_pred_probs = rf_classifier.predict_proba(X_test_pca)[:, class_idx]
        rf_fpr[class_idx], rf_tpr[class_idx], _ = roc_curve(y_test_bin[:, class_idx], rf_pred_probs)
        rf_auc[class_idx] = auc(rf_fpr[class_idx], rf_tpr[class_idx])


    # Plot ROC curves for each class for Random Forest
    plt.figure(figsize=(8, 6))
    for class_idx in range(10):
        plt.plot(rf_fpr[class_idx], rf_tpr[class_idx], label='Class {} (AUC = {:.2f})'.format(class_idx, rf_auc[class_idx]))

    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve for Random Forest')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # For Naive Bayes
    nb_fpr = dict()
    nb_tpr = dict()
    nb_auc = dict()
    for class_idx in range(10):
        nb_pred_probs = nb_classifier.predict_proba(X_test_pca)[:, class_idx]
        nb_fpr[class_idx], nb_tpr[class_idx], _ = roc_curve(y_test_bin[:, class_idx], nb_pred_probs)
        nb_auc[class_idx] = auc(nb_fpr[class_idx], nb_tpr[class_idx])


    # Plot ROC curves for each class for Naive Bayes
    plt.figure(figsize=(8, 6))
    for class_idx in range(10):
        plt.plot(nb_fpr[class_idx], nb_tpr[class_idx], label='Class {} (AUC = {:.2f})'.format(class_idx, nb_auc[class_idx]))

    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve for Naive Bayes')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


    # For K-NN
    knn_fpr = dict()
    knn_tpr = dict()
    knn_auc = dict()
    for class_idx in range(10):
        knn_pred_probs = knn_classifier.predict_proba(X_test_pca)[:, class_idx]
        knn_fpr[class_idx], knn_tpr[class_idx], _ = roc_curve(y_test_bin[:, class_idx], knn_pred_probs)
        knn_auc[class_idx] = auc(knn_fpr[class_idx], knn_tpr[class_idx])


    # Plot ROC curves for each class for K-NN
    plt.figure(figsize=(8, 6))
    for class_idx in range(10):
        plt.plot(knn_fpr[class_idx], knn_tpr[class_idx], label='Class {} (AUC = {:.2f})'.format(class_idx, knn_auc[class_idx]))

    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve for K-NN')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

