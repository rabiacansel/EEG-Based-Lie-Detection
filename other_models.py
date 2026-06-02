import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, LSTM, Bidirectional, GRU
from tensorflow.keras.utils import to_categorical  
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import seaborn as sns
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.regularizers import l2

import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
import random

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

train_file = r"C:\Users\r4bi4\Desktop\lie_detection\revizyon_dwt\data\normalization\db4_lv5\S2\features_normalization_train.csv"
test_file  = r"C:\Users\r4bi4\Desktop\lie_detection\revizyon_dwt\data\normalization\db4_lv5\S2\features_normalization_test.csv"
label_file = r"C:\Users\r4bi4\Desktop\lie_detection\processing_label.xlsx"

# Verileri oku
train_df = pd.read_csv(train_file)
test_df  = pd.read_csv(test_file)
label_df = pd.read_excel(label_file)

# Label merge
train_data = pd.merge(train_df, label_df, on="Subject", how="inner")
test_data  = pd.merge(test_df, label_df, on="Subject", how="inner")

# Feature ve label ayır
X_train = train_data.loc[:, "Minimum Value":"Relative Energy"].values
X_test  = test_data.loc[:, "Minimum Value":"Relative Energy"].values

y_train = train_data["Label"].values
y_test  = test_data["Label"].values

# --------- NORMALIZATION (SADECE TRAIN'DE FIT!) ----------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, 10)).reshape(-1, 30, 10)
X_test  = scaler.transform(X_test.reshape(-1, 10)).reshape(-1, 30, 10)

# --------- LABEL SHAPE ----------
y_train = y_train.reshape(-1, 30)[:, 0]
y_test  = y_test.reshape(-1, 30)[:, 0]

y_train = to_categorical(y_train, num_classes=2)
y_test  = to_categorical(y_test, num_classes=2)


def save_results(model_name, y_true, y_pred, history=None):
    base_dir = r"C:\Users\r4bi4\Desktop\lie_detection\revizyon_dwt\data\results_db4_lv5"
    
    subject_name = os.path.basename(os.path.dirname(train_file))
    
    save_dir = os.path.join(base_dir, model_name, subject_name)
    os.makedirs(save_dir, exist_ok=True)

    # -------------------- SUBJECT --------------------
    test_subjects = test_data["Subject"].values.reshape(-1, 30)[:, 0]

    # -------------------- METRICS --------------------
    acc = accuracy_score(y_true, y_pred)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )

    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )

    report_dict = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    # -------------------- CONFUSION MATRIX --------------------
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    # -------------------- SUBJECT-BASED --------------------
    subject_results = pd.DataFrame({
        "Subject": test_subjects,
        "True": y_true,
        "Pred": y_pred
    })

    subject_accuracy = subject_results.groupby("Subject").apply(
        lambda x: (x["True"] == x["Pred"]).mean()
    ).reset_index(name="Accuracy")

    # -------------------- SAVE --------------------
    pd.DataFrame({
        "Accuracy": [acc],
        "Macro Precision": [macro_p],
        "Macro Recall": [macro_r],
        "Macro F1-score": [macro_f1]
    }).to_csv(os.path.join(save_dir, "summary_metrics.csv"), index=False)

    pd.DataFrame({
        "Precision": p,
        "Recall": r,
        "F1-score": f1,
        "Support": support
    }).to_csv(os.path.join(save_dir, "classwise_metrics.csv"), index=False)

    report_df.to_csv(os.path.join(save_dir, "classification_report.csv"))
    subject_accuracy.to_csv(os.path.join(save_dir, "subject_accuracy.csv"), index=False)

    # -------------------- TRAINING PLOTS --------------------
    if history is not None:
        plt.figure()
        plt.plot(history.history['accuracy'], label='train_accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.legend()
        plt.title("Accuracy")
        plt.savefig(os.path.join(save_dir, "accuracy_plot.png"))
        plt.close()

        plt.figure()
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.legend()
        plt.title("Loss")
        plt.savefig(os.path.join(save_dir, "loss_plot.png"))
        plt.close()

    print(f"{model_name} kaydedildi -> {save_dir}")
def KNN():
    X_train_knn = X_train.reshape(X_train.shape[0], -1)
    X_test_knn  = X_test.reshape(X_test.shape[0], -1)

    y_train_knn = np.argmax(y_train, axis=1)
    y_test_knn  = np.argmax(y_test, axis=1)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_knn, y_train_knn)
    y_pred = knn.predict(X_test_knn)
    save_results("knn", y_test_knn, y_pred)
    
    acc = accuracy_score(y_test_knn, y_pred)
    print(f"KNN Test Accuracy: {acc:.2f}")
    print(classification_report(y_test_knn, y_pred))
    
    cm = confusion_matrix(y_test_knn, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("KNN Confusion Matrix")
    plt.show()
    
    false_positives = np.where((y_test_knn == 0) & (y_pred == 1))[0]
    false_negatives = np.where((y_test_knn == 1) & (y_pred == 0))[0]
    print("\nYanlış Pozitifler (Gerçek 0, Tahmin 1):")
    print(pd.DataFrame({
        "Index": false_positives,
        "True Label": np.array(y_test_knn)[false_positives],
        "Predicted Label": y_pred[false_positives]
    }))
    print("\nYanlış Negatifler (Gerçek 1, Tahmin 0):")
    print(pd.DataFrame({
        "Index": false_negatives,
        "True Label": np.array(y_test_knn)[false_negatives],
        "Predicted Label": y_pred[false_negatives]
    }))

def SVM():
    X_train_svm = X_train.reshape(X_train.shape[0], -1)
    X_test_svm  = X_test.reshape(X_test.shape[0], -1)

    y_train_svm = np.argmax(y_train, axis=1)
    y_test_svm  = np.argmax(y_test, axis=1)
    
    svm_model = SVC(kernel='rbf')
    svm_model.fit(X_train_svm, y_train_svm)
    y_pred = svm_model.predict(X_test_svm)
    save_results("svm", y_test_svm, y_pred)
    
    acc = accuracy_score(y_test_svm, y_pred)
    print(f"SVM Test Accuracy: {acc:.2f}")
    print(classification_report(y_test_svm, y_pred))
    
    cm = confusion_matrix(y_test_svm, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("SVM Confusion Matrix")
    plt.show()
    
    false_positives = np.where((y_test_svm == 0) & (y_pred == 1))[0]
    false_negatives = np.where((y_test_svm == 1) & (y_pred == 0))[0]
    print("\nYanlış Pozitifler (Gerçek 0, Tahmin 1):")
    print(pd.DataFrame({
        "Index": false_positives,
        "True Label": np.array(y_test_svm)[false_positives],
        "Predicted Label": y_pred[false_positives]
    }))
    print("\nYanlış Negatifler (Gerçek 1, Tahmin 0):")
    print(pd.DataFrame({
        "Index": false_negatives,
        "True Label": np.array(y_test_svm)[false_negatives],
        "Predicted Label": y_pred[false_negatives]
    }))

def RF():
    X_train_rf = X_train.reshape(X_train.shape[0], -1)
    X_test_rf  = X_test.reshape(X_test.shape[0], -1)

    y_train_rf = np.argmax(y_train, axis=1)
    y_test_rf  = np.argmax(y_test, axis=1)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_rf, y_train_rf)
    y_pred = rf.predict(X_test_rf)
    save_results("rf", y_test_rf, y_pred)
    
    acc = accuracy_score(y_test_rf, y_pred)
    print(f"RF Test Accuracy: {acc:.2f}")
    print(classification_report(y_test_rf, y_pred))
    
    cm = confusion_matrix(y_test_rf, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("RF Confusion Matrix")
    plt.show()
    
    false_positives = np.where((y_test_rf == 0) & (y_pred == 1))[0]
    false_negatives = np.where((y_test_rf == 1) & (y_pred == 0))[0]
    print("\nYanlış Pozitifler (Gerçek 0, Tahmin 1):")
    print(pd.DataFrame({
        "Index": false_positives,
        "True Label": np.array(y_test_rf)[false_positives],
        "Predicted Label": y_pred[false_positives]
    }))
    print("\nYanlış Negatifler (Gerçek 1, Tahmin 0):")
    print(pd.DataFrame({
        "Index": false_negatives,
        "True Label": np.array(y_test_rf)[false_negatives],
        "Predicted Label": y_pred[false_negatives]
    }))

def BILSTM():
    X_train_bilstm = X_train
    X_test_bilstm  = X_test
    y_train_bilstm = y_train
    y_test_bilstm  = y_test
    
    model = Sequential([
        Input(shape=(30, 10)),
        Bidirectional(LSTM(256, activation='tanh', dropout=0.2, recurrent_dropout=0.1, return_sequences=True)),
        Bidirectional(LSTM(128, activation='tanh', dropout=0.2, recurrent_dropout=0.1)),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        ModelCheckpoint('BILSTM_DWT_best_model.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-1)
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=100,
        validation_split=0.1,
        callbacks=callbacks
    )

    BILSTM_DWT_best_model = load_model('BILSTM_DWT_best_model.keras')

    loss, accuracy = BILSTM_DWT_best_model.evaluate(X_test, y_test)

    BILSTM_DWT_best_model.save("BILSTM_DWT_best_model.keras")

    print(f"CNN Test Loss: {loss:.2f}")
    print(f"CNN Test Accuracy: {accuracy:.2f}")
    
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    plt.title('Bi-LSTM Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    y_pred = np.argmax(BILSTM_DWT_best_model.predict(X_test_bilstm), axis=1)
    y_true = np.argmax(y_test_bilstm, axis=1)
    save_results("bilstm", y_true, y_pred, history)
    
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Bi-LSTM Confusion Matrix")
    plt.show()
    
    false_positives = np.where((y_true == 0) & (y_pred == 1))[0]
    false_negatives = np.where((y_true == 1) & (y_pred == 0))[0]
    print("\nYanlış Pozitifler (Gerçek 0, Tahmin 1):")
    print(pd.DataFrame({
        "Index": false_positives,
        "True Label": y_true[false_positives],
        "Predicted Label": y_pred[false_positives]
    }))
    print("\nYanlış Negatifler (Gerçek 1, Tahmin 0):")
    print(pd.DataFrame({
        "Index": false_negatives,
        "True Label": y_true[false_negatives],
        "Predicted Label": y_pred[false_negatives]
    }))

def LSTM_model():
    X_train_lstm = X_train
    X_test_lstm  = X_test
    y_train_lstm = y_train
    y_test_lstm  = y_test

    model = Sequential([
        LSTM(512, activation='tanh', dropout=0.2, recurrent_dropout=0.1,
             return_sequences=True, input_shape=(30, 10)),
        LSTM(128, activation='tanh', dropout=0.2, recurrent_dropout=0.1),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        ModelCheckpoint('LSTM_DWT_best_model.keras',
                        monitor='val_loss',
                        save_best_only=True,
                        mode='min',
                        verbose=1),
        ReduceLROnPlateau(monitor='val_loss',
                          factor=0.3,
                          patience=3,
                          verbose=1,
                          min_lr=1e-6)
    ]

    history = model.fit(
        X_train_lstm, y_train_lstm,
        epochs=100,
        batch_size=100,
        validation_split=0.1,
        callbacks=callbacks
    )

    best_model = load_model('LSTM_DWT_best_model.keras')

    loss, accuracy = best_model.evaluate(X_test_lstm, y_test_lstm)
    print(f"LSTM Test Loss: {loss:.2f}")
    print(f"LSTM Test Accuracy: {accuracy:.2f}")

    best_model.save("LSTM_DWT_best_model.keras")

    # -------------------- PREDICTION --------------------
    y_pred = np.argmax(best_model.predict(X_test_lstm), axis=1)
    y_true = np.argmax(y_test_lstm, axis=1)

    # -------------------- SAVE RESULTS --------------------
    save_results("lstm", y_true, y_pred, history)

    # -------------------- REPORT --------------------
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("LSTM Confusion Matrix")
    plt.tight_layout()
    plt.show()



def GRU_model():
    X_train_gru = X_train
    X_test_gru  = X_test
    y_train_gru = y_train
    y_test_gru  = y_test    
    
    model = Sequential([
    Input(shape=(30, 10)),

    GRU(256, return_sequences=True, activation='tanh', recurrent_dropout=0.2),
    BatchNormalization(),
    Dropout(0.3),

    GRU(128, activation='tanh', recurrent_dropout=0.2),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(0.3),

    Dense(2, activation='softmax')
])

    optimizer = Adam(learning_rate=0.0005)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])  
    callbacks = [
        ModelCheckpoint('GRU_DWT_best_model.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    ]
    
    history = model.fit(X_train_gru, y_train_gru, epochs=100, batch_size=100, validation_split=0.1, callbacks=callbacks)
    GRU_DWT_best_model = load_model('GRU_DWT_best_model.keras')
    loss, accuracy = GRU_DWT_best_model.evaluate(X_test_gru, y_test_gru)
    model.save("GRU_DWT_best_model.keras")
    print(f"GRU Test Loss: {loss:.2f}")
    print(f"GRU Test Accuracy: {accuracy:.2f}")
    
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], label='train_accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='val_accuracy', color='red')
    plt.title('GRU Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    y_pred = np.argmax(GRU_DWT_best_model.predict(X_test_gru), axis=1)
    y_true = np.argmax(y_test_gru, axis=1)
    save_results("gru", y_true, y_pred, history)
    
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("GRU Confusion Matrix")
    plt.show()
    
    false_positives = np.where((y_true == 0) & (y_pred == 1))[0]
    false_negatives = np.where((y_true == 1) & (y_pred == 0))[0]
    print("\nYanlış Pozitifler (Gerçek 0, Tahmin 1):")
    print(pd.DataFrame({
        "Index": false_positives,
        "True Label": y_true[false_positives],
        "Predicted Label": y_pred[false_positives]
    }))
    print("\nYanlış Negatifler (Gerçek 1, Tahmin 0):")
    print(pd.DataFrame({
        "Index": false_negatives,
        "True Label": y_true[false_negatives],
        "Predicted Label": y_pred[false_negatives]
    }))
    
KNN()
SVM()
RF()
BILSTM()
LSTM_model()
GRU_model()


