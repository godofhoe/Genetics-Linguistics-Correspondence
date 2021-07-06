# Cut-space

Sometimes you don't have the original (**unsegmented**) txt file.This program can help you recovery **segmented** txt file to **unsegmented** one.

For example:
* recovery 'A B' to 'AB' where 'A' and 'B' are words.
* recovery 'ap-ple' to 'apple' where 'ap' and 'ple' are syllables.

## Tutorial
For different symbol of segmentation, you need to change your code.

* The first example is usually seen in Chinese corpora, 
```python
l = line.split()
```
the function split() will view ' ' (space) as symbol of segmentation.

* The second example is usually seen in English corpora, 
```python
l = line.split('-')
```
the function split() will view '-' (dash line) as symbol of segmentation.
