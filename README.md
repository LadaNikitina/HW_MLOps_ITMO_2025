# Домашнее задание 1

## Workflow

Проект ведётся по GitHub Flow, адаптированному под командную работу над исследовательским ML-проектом (2 человека). Вот как мы работаем:
- Всё, что находится в ветке main, считается стабильным и готовым к запуску.
- Чтобы начать работу над новой частью проекта (например, модуль векторизации или классификатор), создаётся новая ветка от main с понятным названием, например: feature/embedding-model или feature/classifier-cb.
- Работа ведётся локально в своей ветке, изменения регулярно коммитятся и пушатся на GitHub в ту же ветку.
- Когда нужен фидбэк или фича готова для слияния, создаётся pull request. Второй участник команды проводит ревью и подтверждает изменения. После ревью ветка вливается в main. После мержа в main изменения считаются готовыми к запуску/тестированию в общем пайплайне.
- Для мержа используем squash commit, чтобы история коммитов не засорялась и содержала только наполненные смыслом сообщения.

# Домашнее задание 2

## Что было сделано

1. Выбрана система версионирования DVC и описан процесс работы с ней в readme репозитория (раздел работа с DVC)
2. Отдельным коммитом добавлены сырые данные в DVC, [коммит](https://github.com/LadaNikitina/hw_mlops_itmo_2025/commit/61cd7d7ef8a0ac44f3bcf3a187f192d7ec89d3c2)
3. Написаны юпитер ноутбуки (находятся в папке research_artifacts) для обработки данных (первый датасет - из fna файлов сырых данных осуществлется парсинг цепочек ДНК и таргета, генерация сплитов данных, второй датасет - на основе первого датасета формирование векторных представлений цепочек). Результат работы добавлен в DVC [коммитом](https://github.com/LadaNikitina/hw_mlops_itmo_2025/commit/30d613d389535da2d796cee965a2518ea9c8e08a) для первого датасета и [коммитом](https://github.com/LadaNikitina/hw_mlops_itmo_2025/commit/476825c3781065d306ad861c7ee08349cf610a5f) для второго датасета.
4. Написан юпитер ноутбук для обучения моделей с использованием данных из папки `data/embeddings`. Было обучено 3 модели - `catboost.CatBoostClassifier`, `xgboost.XGBClassifier`, `lightgbm.LGBMClassifier`. Обученные модели были сохранены в папку `models` и добавлены в DVC [коммитом](https://github.com/LadaNikitina/hw_mlops_itmo_2025/pull/6/commits/0422600b0fcd699b6ac70f17f3cdeb6bc4a5370a)
5. Написан юпитер ноутбук для сравнения моделей на тестовых выборках. Результаты сравнения представлены в отчете ниже

Исправления:
1. Написан юпитер ноутбук для обучения моделей с использованием данных из папки `data/embeddings`. Было обучено 2 модели - `catboost.CatBoostClassifier` с 2 наборами разных гиперпараметров. Обученные модели были сохранены в папку `two_diff_models`, первая версия добавлена в DVC [коммитом](https://github.com/LadaNikitina/hw_mlops_itmo_2025/commit/9bfd9624f37281268d913b9bf128fe7442b1d8f8), вторая версия добавлена в DVC [коммитом](https://github.com/LadaNikitina/hw_mlops_itmo_2025/commit/9ba21c196da72c3856cff79b2ca30554fbfc2e55).
2. Написан юпитер ноутбук для сравнения моделей на тестовых выборках. Результаты сравнения представлены в отчете по результатам v2.

## Отчет по результатам v2

| Dataset            | catboost\_v1 | catboost\_v2 | Δ (v2 − v1) |
| ------------------ | ------------ | ------------ | ----------- |
| enhancers          | 0.4830       | 0.4700       | **−0.0130** |
| promoter\_all      | 0.8579       | 0.8542       | **−0.0037** |
| splice\_sites\_all | 0.3430       | 0.3430       | +0.0000     |
| H3K9me3            | 0.3062       | 0.3012       | **−0.0050** |
| H4K20me1           | 0.5841       | 0.5718       | **−0.0123** |

**Выводы:**
1. Во всех задачах кроме splice_sites_all метрика у catboost_v2 снизилась.
2. Наибольшее падение наблюдается на:
- enhancers (−0.0130)
- H4K20me1 (−0.0123)

Модель catboost_v1 показывает более высокие результаты на этих подзадачах и является лучшей по итогам сравнения.

## Отчет по результатам 

| Dataset          | XGBoost | CatBoost | LightGBM |
|------------------|---------|----------|----------|
| enhancers        | 0.4721  | 0.4758   | 0.4638   |
| promoter_all     | 0.8508  | 0.85     | 0.8557   |
| splice_sites_all | 0.4807  | 0.343    | 0.5437   |
| H3K9me3          | 0.2801  | 0.2896   | 0.2894   |
| H4K20me1         | 0.5742  | 0.5832   | 0.5843   |

На основе результатов сравнения трех моделей для пяти различных датасетов, можно сделать следующие выводы:

1. **promoter_all** - все модели показали наилучшие результаты на этом наборе данных (0.85-0.857), который предназначен для определения промоторных участков ДНК. LightGBM незначительно превосходит остальные модели.

2. **H4K20me1** - второй по результативности набор данных (0.57-0.58), связанный с метилированием гистонов. CatBoost и LightGBM показывают почти идентичные результаты, незначительно превосходя XGBoost.

3. **splice_sites_all** - набор данных для определения сайтов сплайсинга, где LightGBM (0.5437) значительно превосходит XGBoost (0.4807), а CatBoost показывает наихудший результат (0.343).

4. **enhancers** - данные для классификации энхансеров (усилителей транскрипции), где все модели показывают похожие результаты около 0.47.

5. **H3K9me3** - набор данных для определения метилирования гистонов, где все модели показывают наиболее низкие результаты (0.28-0.29).

В результате сравнения были сделаные следующие выводы:

1. Для задач классификации ДНК-последовательностей наиболее универсальным решением является LightGBM, особенно для задач определения сайтов сплайсинга, где его преимущество наиболее заметно.

2. Модели показывают значительно лучшие результаты на задаче классификации промоторов по сравнению с другими задачами, что может свидетельствовать о более выраженных паттернах в данных этого типа или лучшем качестве векторных представлений для этой задачи.

3. Низкие результаты для H3K9me3 (около 0.28-0.29) говорят о том, что данная задача является наиболее сложной для всех моделей, и возможно требует улучшения методов векторизации или архитектуры моделей.

## Работа с DVC

Мы используем [DVC](https://dvc.org/) для версионирования данных и моделей с удаленным хранилищем S3.

##  Установка и инициализация

```bash
uv pip install dvc
dvc init
git commit -m "init DVC"
```
##  Добавление данных

```bash
dvc add data/something.csv
git add data/something.csv.dvc .gitignore
git commit -m "add some data"
dvc push
```

##  Настройка удалённого хранилища S3

```bash
dvc remote add -d -f yandex s3://mlops-bucket-2025/dvc-store
dvc remote modify yandex endpointurl https://storage.yandexcloud.net 
git commit .dvc/config -m "setup remote"
```

## Синхронизация

```bash
dvc push   # залить данные в облако
dvc pull   # скачать данные из облака
```

## При клонировании проекта

```bash
git clone git@github.com:LadaNikitina/hw_mlops_itmo_2025.git
cd hw_mlops_itmo_2025
uv pip install .
dvc remote modify yandex access_key_id <access-key> --local
dvc remote modify yandex secret_access_key <secret-key> --local
dvc pull
```

# Домашнее задание 3
Для выполнения ДЗ весь пайплайн был адаптирован в 3 скрипта в папке `src`:
- `process.py` - предобработка данных
- `train.py` - обучение моделей
- `evaluate.py` - подсчет метрик

Был настроен Airflow для автоматического запуска скриптов. Для сборки Airflow и поднятия необходимых сервисов используется Docker-конфигурация

### Конфигурация Airflow
1. Установка переменных среды: склонировать `airflow.env` в `.env` или задать переменные вручную:
```bash
export AWS_ACCESS_KEY_ID="YANDEX_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="YANDEX_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION="ru-central1"
export DVC_REMOTE_NAME="yandex"
export DVC_REMOTE_URL="s3://mlops-bucket-2025/dvc-store"
export AIRFLOW_UID=50000
export AIRFLOW_PROJ_DIR=.
export _AIRFLOW_WWW_USER_USERNAME=airflow
export _AIRFLOW_WWW_USER_PASSWORD=airflow
```

2. Инициализация и запуск сервисов
```bash 
# Инициализация
docker compose -f docker-compose-airflow.yaml up airflow-init

# Запуск
docker compose -f docker-compose-airflow.yaml up -d
```

3. Теперь доступен Airflow Web UI (http://localhost:8080). 

Данные для входа:
- Username: `airflow`
- Password: `airflow`

### Обзор DAG
Был создан граф `ml_pipeline` со следующими задачами:
1. `check_data_availability`: Проверяет наличие данных
2. `process_data`: Запускает `src/process.py`
3. `train_models`: Запускает `src/train.py`
4. `evaluate_models`: Запускает `src/evaluate.py`

#### Работа с пайплайном
1. Посмотреть доступные для проекта пайплайны
```bash
docker compose -f docker-compose-airflow.yaml exec airflow-webserver python -m airflow dags list
```
2. Вывести `ml_pipeline` из режима "паузы"
```
docker compose -f docker-compose-airflow.yaml exec airflow-webserver python -m airflow dags unpause ml_pipeline
```
3. Запустить `ml_pipeline`
```bash
docker compose -f docker-compose-airflow.yaml exec airflow-webserver python -m airflow dags trigger ml_pipeline
```
4. Посмотреть состояние
```bash
docker compose -f docker-compose-airflow.yaml exec airflow-webserver python -m airflow dags state ml_pipeline <dag_run_id>
```

# Домашнее задание 4

В качестве системы трекинга экспериментов был выбран MLflow.

## Инструкция по подключению:

1. Установить зависимость:

   ```bash
   uv pip install mlflow
   ```
2. Добавить логирование в `src/train.py` и `src/evaluate.py`:

   * `mlflow.log_param(...)` — для гиперпараметров.
   * `mlflow.log_metric(...)` — для метрик.
   * `mlflow.log_artifact(...)` — для модели и json-результатов.
3. Запуск интерфейса:

   ```bash
   mlflow ui
   ```

   Затем открыть [http://localhost:5000](http://localhost:5000)


## Проведение 3 экспериментов + логирование

Для контроля версий данных используется DVC.
Все входные эмбеддинги (data/embeddings/) получаются через dvc pull из облачного хранилища.
Эти данные участвуют в пайплайне process -> train -> evaluate, как указано в dvc.yaml.

Запуск экспериментов реализован через dvc.yaml в 3 разных конфигурациях. Запуск осуществляется при помощи `dvc.repro`.

Проведено 3 эксперимента с разными моделями:

| Модель        | Гиперпараметры     | Папка             |
| ------------- | ------------------ | ----------------- |
| CatBoost      | `depth=4, lr=0.02` | `models/catboost` |
| Random Forest | `depth=6`          | `models/rf`       |
| LightGBM      | `depth=6, lr=0.05` | `models/lgbm`     |

Каждая модель обучалась на 5 датасетах, один датасет - один тип задачи.
Оценка производилась на test-сплите с сохранением метрик `F1`, `MCC`, `Accuracy` в папку `mlflow_metrics/`.

## Что логируется в MLflow:

* Параметры обучения
* Метрика на тесте
* Файл модели
* JSON-файл с метрикой

Логирование артефактов реализовано в скриптах пайплайна в папке src/train.py, src/evaluate.py.
