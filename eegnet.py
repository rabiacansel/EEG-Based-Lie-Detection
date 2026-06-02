import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    AveragePooling2D, Dropout, Flatten, Dense,
    DepthwiseConv2D, SeparableConv2D
)
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.constraints import max_norm
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report,
    confusion_matrix
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau
)
from tensorflow.keras.models import load_model

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

train_file = "features_normalization_train.csv"
test_file  = "features_normalization_test.csv"
label_file = "processing_label.xlsx"

# ---------------- READ DATA ----------------
train_df = pd.read_csv(train_file)
test_df  = pd.read_csv(test_file)
label_df = pd.read_excel(label_file)

# ---------------- MERGE LABELS ----------------
train_data = pd.merge(train_df, label_df, on="Subject", how="inner")
test_data  = pd.merge(test_df, label_df, on="Subject", how="inner")

# ---------------- FEATURES / LABELS ----------------
X_train = train_data.loc[:, "Minimum Value":"Relative Energy"].values
X_test  = test_data.loc[:, "Minimum Value":"Relative Energy"].values

y_train = train_data["Label"].values
y_test  = test_data["Label"].values

# ---------------- NORMALIZATION ----------------
scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train.reshape(-1, 10)
).reshape(-1, 5, 60, 1)

X_test = scaler.transform(
    X_test.reshape(-1, 10)
).reshape(-1, 5, 60, 1)

# ---------------- LABEL SHAPE ----------------
y_train = y_train.reshape(-1, 30)[:, 0]
y_test  = y_test.reshape(-1, 30)[:, 0]

y_train = to_categorical(y_train, num_classes=2)
y_test  = to_categorical(y_test, num_classes=2)


def save_results(model_name, y_true, y_pred, history=None):
    base_dir = r"C:\Users\r4bi4\Desktop\lie_detection\revizyon_dwt\data\results_db4_lv5\eegnet"
    
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

def EEGNet(
    nb_classes=2,
    Chans=5,
    Samples=256,
    dropoutRate=0.5,
    kernLength=64,
    F1=8,
    D=2,
    F2=16,
    norm_rate=0.25,
    dropoutType="Dropout"
):
    """
    EEGNet model for EEG classification.

    Parameters
    ----------
    nb_classes : int
        Number of output classes. For lie/truth classification, use 2.
    Chans : int
        Number of EEG channels. LieWaves uses 5 channels.
    Samples : int
        Number of time samples in each EEG window.
    dropoutRate : float
        Dropout ratio.
    kernLength : int
        Temporal convolution kernel length.
    F1 : int
        Number of temporal filters.
    D : int
        Depth multiplier for depthwise convolution.
    F2 : int
        Number of pointwise filters. Usually F2 = F1 * D.
    norm_rate : float
        Max-norm constraint for dense layer.
    dropoutType : str
        "Dropout" or "SpatialDropout2D".
    """

    if dropoutType == "SpatialDropout2D":
        DropoutLayer = tf.keras.layers.SpatialDropout2D
    else:
        DropoutLayer = Dropout

    input1 = Input(shape=(Chans, Samples, 1))

    # Block 1: Temporal convolution + spatial filtering
    block1 = Conv2D(
        F1,
        (1, kernLength),
        padding="same",
        use_bias=False
    )(input1)

    block1 = BatchNormalization()(block1)

    block1 = DepthwiseConv2D(
        (Chans, 1),
        use_bias=False,
        depth_multiplier=D,
        depthwise_constraint=max_norm(1.0)
    )(block1)

    block1 = BatchNormalization()(block1)
    block1 = Activation("elu")(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = DropoutLayer(dropoutRate)(block1)

    # Block 2: Separable convolution
    block2 = SeparableConv2D(
        F2,
        (1, 16),
        use_bias=False,
        padding="same"
    )(block1)

    block2 = BatchNormalization()(block2)
    block2 = Activation("elu")(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = DropoutLayer(dropoutRate)(block2)

    flatten = Flatten(name="flatten")(block2)

    dense = Dense(
        nb_classes,
        name="dense",
        kernel_constraint=max_norm(norm_rate)
    )(flatten)

    softmax = Activation("softmax", name="softmax")(dense)

    return Model(inputs=input1, outputs=softmax)








# ---------------- EEGNet INPUT SHAPE ----------------
X_train_eegnet = X_train.reshape(X_train.shape[0], 5, 60, 1)
X_test_eegnet  = X_test.reshape(X_test.shape[0], 5, 60, 1)

# ---------------- EEGNet MODEL ----------------
model = EEGNet(
    nb_classes=2,
    Chans=5,
    Samples=60,
    dropoutRate=0.5,
    kernLength=32,
    F1=8,
    D=2,
    F2=16
)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------- CALLBACKS ----------------
checkpoint = ModelCheckpoint(
    "best_eegnet.keras",
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1
)

# ---------------- TRAIN ----------------
history = model.fit(
    X_train_eegnet,
    y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test_eegnet, y_test),
    callbacks=[checkpoint, reduce_lr],
    verbose=1
)

# ---------------- LOAD BEST MODEL ----------------
best_model = load_model("best_eegnet.keras")

# ---------------- PREDICT ----------------
y_pred_prob = best_model.predict(X_test_eegnet)

y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

# ---------------- SAVE RESULTS ----------------
save_results(
    model_name="EEGNet",
    y_true=y_true,
    y_pred=y_pred,
    history=history
)
