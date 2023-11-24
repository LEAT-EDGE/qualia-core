from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from typing import Any, TypeVar, Union, cast

logger = logging.getLogger(__name__)

RecursiveConfigUnion = Union[Any, Mapping[str, Any], list[Any]]
RecursiveMapping = Mapping[str, RecursiveConfigUnion]

T = TypeVar('T', str, int)
TRecursiveDict = dict[T, RecursiveConfigUnion]
TRecursiveMapping = Mapping[T, RecursiveConfigUnion]

U = TypeVar('U', bound=RecursiveMapping)

def merge_dict(dict_a: U,
               dict_b: U,
               merge_lists: bool = False) -> U:  # noqa: FBT001, FBT002
    """Merge elements from dict_b into dict_a if they are missing."""
    def merger(table: TRecursiveDict[T],
               template: TRecursiveMapping[T]) -> None:
        for k, template_v in template.items():
            if k not in table:
                table[k] = copy.deepcopy(template_v)
                continue

            table_v = table[k]
            if isinstance(template_v, dict):
                if not isinstance(table_v, dict):
                    logger.error("Incompatible target value type %s for key '%s', expected dict", type(table_v), k)
                    raise TypeError
                merger(table_v, template_v)
            elif isinstance(template_v, list) and merge_lists:
                if not isinstance(table_v, dict):
                    logger.error("Incompatible target value type %s for key '%s', expected dict", type(table_v), k)
                    raise TypeError
                # Convert possible string keys to int
                temp_table = {int(tblk): tblv for tblk, tblv in table_v.items()}
                # Convert template list to dict
                v_dict: TRecursiveDict[int] = dict(enumerate(template_v))
                # Merge converted template dict into table
                merger(temp_table, v_dict)
                # Convert back dest table to list, iterate up to highest index
                table[k] = [temp_table[tbli] for tbli in range(sorted(temp_table.keys())[-1] + 1)]

    # Mapping becomes Dict after copy as it is mutable and modified by merger()
    dict_r = cast(TRecursiveDict[str], copy.deepcopy(dict_a))
    merger(dict_r, dict_b)
    # Merged dict is cast back to original Mapping type to keep type compatibility
    return cast(U, dict_r)

