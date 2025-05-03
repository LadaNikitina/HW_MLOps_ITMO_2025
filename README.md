# Workflow

Проект ведётся по GitHub Flow, адаптированному под командную работу над исследовательским ML-проектом (2 человека). Вот как мы работаем:
- Всё, что находится в ветке main, считается стабильным и готовым к запуску.
- Чтобы начать работу над новой частью проекта (например, модуль векторизации или классификатор), создаётся новая ветка от main с понятным названием, например: feature/embedding-model или feature/classifier-cb.
- Работа ведётся локально в своей ветке, изменения регулярно коммитятся и пушатся на GitHub в ту же ветку.
- Когда нужен фидбэк или фича готова для слияния, создаётся pull request. Второй участник команды проводит ревью и подтверждает изменения. После ревью ветка вливается в main. После мержа в main изменения считаются готовыми к запуску/тестированию в общем пайплайне.
- Для мержа используем squash commit, чтобы история коммитов не засорялась и содержала только наполненные смыслом сообщения.

# Работа с DVC

Мы используем [DVC](https://dvc.org/) для версионирования данных и моделей.

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
dvc pull
```
