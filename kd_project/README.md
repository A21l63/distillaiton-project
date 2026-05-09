# Проект: Knowledge Distillation на CIFAR-10

## Цель проекта

Цель проекта — сравнить три модели для классификации изображений CIFAR-10:

1. маленькую `StudentModel`, обученную обычным способом по правильным меткам;
2. большую `TeacherModel` на основе pretrained ResNet18;
3. новую `StudentModel`, обученную через knowledge distillation от teacher-модели.

Это учебный каркас: важные части нужно дописать самостоятельно в местах с `TODO` и `...`.

## Идея knowledge distillation

Knowledge distillation — это способ обучить маленькую модель повторять не только правильные ответы, но и распределение вероятностей большой модели. Teacher-модель может показывать, какие классы похожи друг на друга, а student-модель учится использовать эту дополнительную информацию.

## Структура проекта

```text
kd_project/
  README.md
  requirements.txt
  data.py
  models.py
  losses.py
  train.py
  evaluate.py
  utils.py
  main_train_student.py
  main_train_teacher.py
  main_distill_student.py
  main_compare.py
```

## Установка

```bash
pip install -r requirements.txt
```

Желательно использовать виртуальное окружение Python.

## Данные

Используется только CIFAR-10. Датасет содержит 10 классов маленьких цветных изображений размером 32×32. Загрузка данных должна быть реализована в `data.py`.

## Модели

- `ClassificationModel` — базовый класс.
- `StudentModel` — маленькая CNN-модель для CIFAR-10.
- `TeacherModel` — ResNet18 с pretrained ImageNet weights, у которой последний слой заменён на классификацию 10 классов.

## Часть 1: обучение StudentModel без distillation

В файле `main_train_student.py` нужно обучить student-модель обычной cross-entropy loss по hard labels.

## Часть 2: обучение TeacherModel

В файле `main_train_teacher.py` нужно дообучить pretrained ResNet18 на CIFAR-10 и сохранить checkpoint.

## Часть 3: обучение StudentModel через distillation

В файле `main_distill_student.py` нужно:

1. загрузить обученную teacher-модель;
2. заморозить её параметры;
3. создать новую `StudentModel` с нуля;
4. обучить её с distillation loss.

Важно: distilled student не должен продолжать обучение baseline student.

## Что нужно реализовать

Нужно дописать места с `TODO`:

- transforms и dataloaders для CIFAR-10;
- архитектуру `StudentModel`;
- загрузку и адаптацию ResNet18 в `TeacherModel`;
- hard-label loss и distillation loss;
- один общий training loop;
- accuracy, confusion matrix и inference time;
- сохранение, загрузку checkpoint и построение графиков.

## Как запускать

Примерный порядок запуска:

```bash
python main_train_student.py
python main_train_teacher.py
python main_distill_student.py
python main_compare.py
```

Перед запуском нужно заполнить значения гиперпараметров и пути к checkpoint-файлам.

## Что сравнивать

Сравните модели по следующим пунктам:

```text
Model | Training | Accuracy | Params | Inference time
```

Также постройте loss/accuracy curves и confusion matrix или покажите примеры ошибок.

## Что сдавать

Нужно сдать:

1. код с реализованными TODO;
2. checkpoint baseline StudentModel;
3. checkpoint TeacherModel;
4. checkpoint distilled StudentModel;
5. таблицу сравнения;
6. графики обучения;
7. confusion matrix или примеры ошибок;
8. короткий вывод: помогла ли distillation.

## Что важно объяснить на защите

Будьте готовы объяснить:

1. что такое CIFAR-10;
2. что такое logits;
3. что такое hard labels;
4. что такое soft labels;
5. почему вероятности teacher-модели могут содержать больше информации, чем одна правильная метка;
6. что делает temperature;
7. что контролирует alpha;
8. почему teacher должен быть сильнее student;
9. почему distilled student нужно обучать с нуля;
10. как отличаются accuracy, число параметров и inference time у разных моделей.
