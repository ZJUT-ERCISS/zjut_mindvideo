# unit test (UT)

UT test the main modules including dataset, loss, model, optimizer, scheduler and utils.

To test all modules:

```python
pytest tests/ut/*.py
```

# system test (ST)

ST test the training and validation pipeline.

To test the training process (in graph mode and pynative+ms_function mode) and the validation process, run

```python
pytest tests/st/test_train_infer_segmentation.py
```
