## Задание

Модифицировать программу из [лабораторной работы №2](https://github.com/eeeeagle/PP_2) для параллельной работы по технологии MPI.

## Исследование: сравнение производительности параллельных вычислений по технологии OpenMP и MPI в зависимости от числа потоков

Вычисления выполнялись на системе с 4-х ядерным процессором Intel Core i5-6400 с тактовой частотой 3,3 ГГц и оперативной памятью 8 Гб и тактовой частотой 2133 МГц.

_Таблица 1: зависимость времени выполнения операции умножения матрицы 2500x1500 на матрицу 1500x2000 от числа потоков по технологии OpenMP и MPI_
|**Threads**|**OpenMP time, с**|**MPI time, с**|
|:---------:|:----------------:|:-------------:|
|1          |86,1427		   |0000000        |
|2          |43,5835		   |0000000        |
|3          |28,6283		   |0000000        |
|4          |22,8077		   |0000000        |
|8          |22,6872		   |0000000        |
|16         |22,7157		   |0000000        |

**тут будет описание таблицы**

_График 1: сравнение зависимости времени выполнения операции умножения матрицы 2500x1500 на матрицу 1500x2000 от числа потоков_

**тут будет график**
