"""Miscellanous functions"""

from typing import Any, Dict


def merge_dict(
    dict1: Dict[Any, Any], dict2: Dict[Any, Any], dict3: Dict[Any, Any]
) -> Dict[Any, Any]:
    """Function to merge three dictionaries

    Args:
        * dict1 (Dict[Any, Any]): 1st dictionary
        * dict2 (Dict[Any, Any]): 2nd dictionary
        * dict3 (Dict[Any, Any]): 3rd dictionary

    Returns:
        * Dict[Any, Any]: Merged dictionary
    """

    _dict = dict1.copy()
    _dict.update(dict2)
    _dict.update(dict3)

    return _dict
