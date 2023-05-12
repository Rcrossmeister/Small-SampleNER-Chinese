import rtoml
from typing import List, Union


def load_config(config_path: str, params: Union[str, List[str]]):
    args = None
    with open(config_path, "r", encoding="utf-8") as file:
        args = rtoml.load(file)["config"]
    if isinstance(params, str):
        return args[params]
    ret = []
    for i in params:
        ret.append(args[i])
    return ret
