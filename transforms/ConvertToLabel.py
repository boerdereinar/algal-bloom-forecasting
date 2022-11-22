from typing import Dict, Any


class ConvertToLabel:
    """
    Convert the output of a dataset to a label.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """
        :param x: Input data
        :return: Labeled data
        """
        data: Dict[str, Any] = dict()
        for key, value in x.items():
            if key == "image":
                data["label"] = value
            else:
                data[f"{key}_label"] = value

        return data
