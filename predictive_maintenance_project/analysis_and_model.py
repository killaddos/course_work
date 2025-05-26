import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score, classification_report,
    RocCurveDisplay
)


def preprocess_data(data):
    data = data.drop(
        columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'],
        errors='ignore')

    if 'Type' not in data.columns:
        raise ValueError("Колонка 'Type' отсутствует!")

    data['Type'] = LabelEncoder().fit_transform(data['Type'])

    # Очистка названий колонок
    data.columns = [col.replace('[', '').replace(']', '').replace(' ', '_') for
                    col in data.columns]

    numerical_features = [
        'Type',
        'Air_temperature_K',
        'Process_temperature_K',
        'Rotational_speed_rpm',
        'Torque_Nm',
        'Tool_wear_min'
    ]
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    return data, scaler, numerical_features


def train_model(X_train, y_train, model_name):
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Random Forest":
        model = RandomForestClassifier()
    elif model_name == "SVM":
        model = SVC(probability=True)
    elif model_name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError("Неверное имя модели")

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return acc, auc, cm, report, y_proba


def analysis_and_model_page():
    st.title("Предиктивная аналитика и сравнение моделей")

    uploaded_file = st.file_uploader("Загрузите CSV-файл", type="csv")
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            data, scaler, feature_names = preprocess_data(data)

            if 'Machine_failure' not in data.columns:
                st.error("Нет колонки 'Machine_failure'")
                return

            X = data.drop(columns=['Machine_failure'])
            y = data['Machine_failure']
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                test_size=0.2,
                                                                random_state=42)

            model_name = st.selectbox("Выберите модель",
                                      ["Logistic Regression", "Random Forest",
                                       "SVM", "XGBoost"])
            model = train_model(X_train, y_train, model_name)

            acc, auc, cm, report, y_proba = evaluate_model(model, X_test,
                                                           y_test)
            st.metric("Accuracy", f"{acc:.2%}")
            st.metric("AUC", f"{auc:.2%}")

            st.subheader("Матрица ошибок")
            fig_cm, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Предсказано")
            ax.set_ylabel("Факт")
            st.pyplot(fig_cm)
            plt.close(fig_cm)

            st.subheader("ROC кривая")
            fig_roc, ax = plt.subplots()
            RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
            st.pyplot(fig_roc)
            plt.close(fig_roc)

            st.subheader("Классификационный отчет")
            st.dataframe(pd.DataFrame(report).transpose())

            # Интерфейс для предсказания
            st.header("Предсказание в реальном времени")
            with st.form("prediction_form"):
                st.write("Введите параметры оборудования:")

                air_temp = st.number_input("Температура воздуха [K]",
                                           value=300.0)
                process_temp = st.number_input("Температура процесса [K]",
                                               value=310.0)
                rotational_speed = st.number_input("Скорость вращения [rpm]",
                                                   value=1500)
                torque = st.number_input("Крутящий момент [Nm]", value=40.0)
                tool_wear = st.number_input("Износ инструмента [min]",
                                            value=100)

                type_str = st.selectbox("Тип оборудования", ['L', 'M', 'H'])
                submitted = st.form_submit_button("Предсказать")

            if submitted:
                try:
                    type_encoded = {'L': 0, 'M': 1, 'H': 2}[type_str]

                    input_data = pd.DataFrame({
                        'Type': [type_encoded],
                        'Air_temperature_K': [air_temp],
                        'Process_temperature_K': [process_temp],
                        'Rotational_speed_rpm': [rotational_speed],
                        'Torque_Nm': [torque],
                        'Tool_wear_min': [tool_wear]
                    })[X_train.columns.tolist()]

                    input_scaled = scaler.transform(input_data)
                    prediction = model.predict(input_scaled)
                    probability = model.predict_proba(input_scaled)[:, 1]

                    st.success(
                        f"Прогноз: {'Отказ ' if prediction[0] == 1 else 'Хорошо'}")
                    st.metric("Вероятность отказа", f"{probability[0]:.2%}")

                except Exception as e:
                    st.error(f"Ошибка: {str(e)}")

        except Exception as e:
            st.error(f"Ошибка при загрузке данных: {str(e)}")
            st.stop()
