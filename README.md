# Target Speech Separation

ДЗ-2 на курсе DLA 23/24.

## Preliminaries

Предварительно необходимо установить используемые зависимости путём вызова команды
```commandline
pip install -r requirements.txt
```

## Генерация миксов

Для генерации миксов достаточно выбрать один из конфигов в директории `configs/mixtures_generator`
или создать свой по аналогии.
Затем запустить скрипт `generate_mixtures.py`:
```commandline
python3 generate_mixtures.py \
    -c {путь до файла конфига} \
    -o {путь до директории, в которой будут находиться миксы}
```

> Все скрипты поддерживают вызов `python3 {script} --help` с описанием своих параметров.

> В миксах для обучения дополнительно ID дикторов из датасетов Librispeech `dev-clean` и
> `train-clean-100`были отображены в отрезок [0, 290] на основе файла `SPEAKERS.TXT` из датасета.
> Файл отображения из исходных ID в новые можно передать в скрипт аргументом `-m/--mapping`.

Сгенерированные на основе Librispeech датасеты расположены на ~~[Google-диске](https://drive.google.com/drive/folders/1QBd2_5cuVlu3QgqMt5IqOzXagQ94nZ_y?usp=drive_link)
и будут загружены автоматически при их использовании через `tss_lib.datasets.LibrispeechDataset`~~ [Яндекс диске](https://disk.yandex.ru/d/EAOv9By429lcyg), для их использования необходимо загрузить и распаковать архив с выбранным набором миксов в некоторую директорию `$dir` и затем в конфиге при использовании `tss_lib.datasets.LibrispeechDataset` в аргументе `data_dir` указать `{$dir}/{название миксов}` (например, `{$dir}/dev-clean-3s`).
Также они доступны на [Kaggle](https://www.kaggle.com/datasets/trickmanoff/librispeech-mixtures).

## Обучение модели

Для обучения модели достаточно выбрать конфиг из `configs/training` или задать свой по аналогии, а
затем запустить команду
```commandline
python3 train.py \
   --config={путь до файла конфига}
```

Для продолжения обучения чекпоинта можно использовать аргумент `-r/--resume`:
```commandline
python3 train.py \
   --config={путь до файла конфига} \
   --resume={путь до сохранённого чекпоинта}
```

## Загрузка предобученной модели

Модель обучалась вызовом команды
```commandline
python3 train.py \
   --config=configs/kaggle_test-with-test.json
```

> Повторить это без дополнительных действий не получится, т.к. в конфиге указано сохранение чекпоинтов на Google Drive, для чего локально использовался отдельный файл с ключом и доп. авторизация в аккаунт.
Достаточно убрать из файла конфигурации запись с "external_storage", чтобы всё заработало без экспортирования чекпоинтов на Google Drive.

Модель обучалась не только на Kaggle, но в Датасфере, для чего использовался
конфиг `configs/datasphere/kaggle_test-with-test.json`.

Загрузить чекпоинт и соответствующий ему конфиг можно вызовом команды
```commandline
python3 model_loader.py \
   --config=gdrive_storage/external_storage.json \
   --run={имя эксперимента}:{имя запуска} \
   config

python3 model_loader.py \
   --config=gdrive_storage/external_storage.json \
   --run={имя эксперимента}:{имя запуска} \
   checkpoint {имя чекпоинта}
```

<details>
<summary>Конкретные команды для загрузки финального чекпоинта</summary>

```commandline
python3 model_loader.py \
   --config=gdrive_storage/external_storage.json \
   --run=kaggle_test:final \
   config
   
python3 model_loader.py \
   --config=gdrive_storage/external_storage.json \
   --run=kaggle_test:final \
   checkpoint model_best
```
</details>

> При запуске скриптов из корневой директории репозитория конфиг `gdrive_storage/external_storage.json` не требуется модифицировать, иначе - нужно обновить в нём путь до файла с ключом

Также с помощью аргумента `-p, --path` можно указать директорию сохранения конфига и чекпоинта (по умолчанию: `saved/models`).

В случае каких-либо проблем со стороны API Google Drive загрузить модель можно вручную по [ссылке](https://drive.google.com/drive/folders/1oizgsefXRnSYi4rCLZFIZq_fN0fq3o3h?usp=drive_link).


## Оценка качества итоговой модели

Для того, чтобы оценить модель, можно использовать скрипт `test.py`.
Для этого нужно задать конфиг по аналогии с конфигами из `configs/eval`.
В этом конфиге можно указать вычисляемые метрики и датасет (как в `configs/eval/dev-clean.json`).
Вместо указания датасета в конфиге можно в качестве аргумента скрипта (`-t`, `--test-data-folder`)
указать путь до директории с данными, имеющей следующую структуру:
```
dir
|-- mix
|    |-- id1-mixed.[wav|mp3|flac|m4a]
|    |-- id2-mixed.[wav|mp3|flac|m4a]
|-- refs
|    |-- id1-ref.[wav|mp3|flac|m4a]
|    |-- id2-ref.[wav|mp3|flac|m4a]
|-- targets
|    |-- id1-target.[wav|mp3|flac|m4a]
|    |-- id2-target.[wav|mp3|flac|m4a]
```

Например, чтобы вычислить метрики PESQ и SI-SDR для финальной предобученной модели,
загруженной с помощью команды выше, по данным из директории, имеющей описанную структуру,
можно вызвать команду:
```commandline
python3 test.py --config=configs/eval/public-small.json \
                --resume=saved/models/kaggle_test/final/model_best.pth \
                -t {путь до директории с данными}
```

Значения метрик будут выведены в терминал.

- Аргумент `-r, --resume`: путь до чекпоинта запускаемой модели
- Аргумент `-c, --config`: путь до дополнительного конфига.
Основной возьмётся из той же директории, что и указанный чекпоинт.
Итоговый конфиг берётся как основной, объединённый с дополнительным и с полями `data`/`postprocessor`/`metrics` из дополнительного, если они в нём определены.

## Автор

Егоров Егор:
- tg: [@TrickmanOff](https://t.me/TrickmanOff)
- e-mail: yegyegorov@gmail.com
