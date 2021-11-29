# Configs

---

## Train config
All the train config should inherit `BaseConfig` and implements all the abstract methods.
The abstract methods can be divided into two parts.

Train configs related:
```python
    @abstractmethod
    def create_model(self, weight=None) -> Model:
        raise NotImplementedError

    @abstractmethod
    def get_optimizer(self):
        raise NotImplementedError

    @abstractmethod
    def get_loss(self):
        raise NotImplementedError

    @abstractmethod
    def get_metrics(self):
        raise NotImplementedError

    @abstractmethod
    def get_lr_scheduler(self):
        raise NotImplementedError

    @abstractmethod
    def get_callbacks(self):
        raise NotImplementedError

```
Dataset related:
```python
    @abstractmethod
    def get_train_loader(self):
        pass

    @abstractmethod
    def get_validate_loader(self):
        pass

    @abstractmethod
    def get_test_loader(self):
        pass
```

### Model1 and Model2
Configs of Model1 and Model2 should inherit `Config` in `core/config/resnet_base.py`.

### WSI Classifier
Configs of WSI Classifier should inherit `Config` in `core/config/rnn_base.py`.

## Inference config
Inference config should inherit `InferenceConfig` in `core/config/inference_base.py`.

The most important part of `InferenceConfig` is the `__ini__` function. One can set all the inference related parameters in it.
