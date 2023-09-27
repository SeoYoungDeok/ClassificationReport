# ClassificationReport
> 분류 모델에 대한 평가 지표를 출력하는 코드

## 사용 예제
**Example1**
```python
from classification_report import ClassificationReport
from pprint import pprint

y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ["class 0", "class 1", "class 2"]

matrix = ClassificationReport(class_name=target_names)
matrix.compute(y_true, y_pred)

print(matrix.print_report())
pprint(matrix.report_dict())
```
```sh
         precision    recall  f1_score  n_sample
 class 0     0.500     1.000     0.667         1
 class 1     0.000     0.000     0.000         1
 class 2     1.000     0.667     0.800         3
------------------------------------------------
accuracy     0.000     0.000     0.600         5

{'accuracy': 0.6,
 'class 0': {'f1_score': 0.6666666657777777,
             'n_sample': 1,
             'precision': 0.49999999975,
             'recall': 0.9999999989999999},
 'class 1': {'f1_score': 0.0, 'n_sample': 1, 'precision': 0.0, 'recall': 0.0},
 'class 2': {'f1_score': 0.7999999992,
             'n_sample': 3,
             'precision': 0.9999999995,
             'recall': 0.6666666664444444},
 'total_sample': 5}
```

**Example2**
```python
from classification_report import ClassificationReport

y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ["class 0", "class 1", "class 2"]

matrix = ClassificationReport(class_name=target_names)
matrix.compute(y_true, y_pred)

print(matrix.print_report())

y_true2 = [2, 2, 1, 0, 0]
y_pred2 = [2, 1, 1, 0, 0]
matrix.compute(y_true2, y_pred2)

print(matrix.print_report())
```
```sh
         precision    recall  f1_score  n_sample
 class 0     0.500     1.000     0.667         1
 class 1     0.000     0.000     0.000         1
 class 2     1.000     0.667     0.800         3
------------------------------------------------
accuracy     0.000     0.000     0.600         5

         precision    recall  f1_score  n_sample
 class 0     0.750     1.000     0.857         3
 class 1     0.333     0.500     0.400         2
 class 2     1.000     0.600     0.750         5
------------------------------------------------
accuracy     0.000     0.000     0.700        10
```

**Example3**
```python
from classification_report import ClassificationReport

y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ["class 0", "class 1", "class 2"]

matrix = ClassificationReport(class_name=target_names)
matrix.compute(y_true, y_pred)

print(matrix.print_report())

matrix.reset()

y_true2 = [2, 2, 1, 0, 0]
y_pred2 = [2, 1, 1, 0, 0]
matrix.compute(y_true2, y_pred2)

print(matrix.print_report())
```
```sh
         precision    recall  f1_score  n_sample
 class 0     0.500     1.000     0.667         1
 class 1     0.000     0.000     0.000         1
 class 2     1.000     0.667     0.800         3
------------------------------------------------
accuracy     0.000     0.000     0.600         5

         precision    recall  f1_score  n_sample
 class 0     1.000     1.000     1.000         2
 class 1     0.500     1.000     0.667         1
 class 2     1.000     0.500     0.667         2
------------------------------------------------
accuracy     0.000     0.000     0.800         5
```
