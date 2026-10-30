import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

base = r'C:\Users\DELL\Documents\arss\arss\real time\AI_Network_Intrusion_Detection'
train = pd.read_csv(os.path.join(base, 'dataset', 'UNSW_NB15_training-set.csv'), low_memory=False)
test = pd.read_csv(os.path.join(base, 'dataset', 'UNSW_NB15_testing-set.csv'), low_memory=False)
for df in (train, test):
    df.columns = df.columns.str.strip().str.lower()

train = train.drop(columns=['id', 'attack_cat'], errors='ignore')
test = test.drop(columns=['id', 'attack_cat'], errors='ignore')
for col in train.select_dtypes(include='object').columns:
    le = LabelEncoder()
    vals = pd.concat([train[col], test[col]], ignore_index=True).astype(str)
    le.fit(vals)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
X_train_df, y_train = train.drop(columns=['label']), train['label']
X_test_df, y_test = test.drop(columns=['label']), test['label']
X_train_df = X_train_df[X_test_df.columns]
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_df.values)
X_test = scaler.transform(X_test_df.values)

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train.values, test_size=0.15, stratify=y_train.values, random_state=42)
selector = ExtraTreesClassifier(n_estimators=250, random_state=42, n_jobs=-1)
selector.fit(X_tr, y_tr)
idx = np.argsort(selector.feature_importances_)[::-1][:8]
Xtr = X_tr[:, idx]
Xva = X_val[:, idx]
Xte = X_test[:, idx]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(Xtr.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
cb = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)]
model.fit(Xtr, y_tr, validation_data=(Xva, y_val), epochs=30, batch_size=128, callbacks=cb, verbose=0)
prob = model.predict(Xte, verbose=0).reshape(-1)
pred = (prob >= 0.5).astype(int)
print('acc', accuracy_score(y_test.values, pred))
print('prec', precision_score(y_test.values, pred, zero_division=0))
print('rec', recall_score(y_test.values, pred, zero_division=0))
print('f1', f1_score(y_test.values, pred, zero_division=0))
print('auc', roc_auc_score(y_test.values, prob))
