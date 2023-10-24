import importlib.util

from gluefactory.utils.tools import get_class
from gluefactory.datasets.base_dataset import BaseDataset


def get_dataset(name):
    """
    尝试从不同路径导入指定的数据集类，然后返回该类对象
    """
    import_paths = [name, f"{__name__}.{name}"]
    for path in import_paths:
        try:
            spec = importlib.util.find_spec(path)
        except ModuleNotFoundError:
            spec = None
        if spec is not None:
            try:
                return get_class(path, BaseDataset)
            except AssertionError:
                mod = __import__(path, fromlist=[""])
                try:
                    return mod.__main_dataset__
                except AttributeError as exc:
                    print(exc)
                    continue

    raise RuntimeError(f'Dataset {name} not found in any of [{" ".join(import_paths)}]')
