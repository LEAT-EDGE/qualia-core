from __future__ import annotations

import ast
import logging

from qualia_core.typing import TYPE_CHECKING

from .merge_dict import merge_dict

if TYPE_CHECKING:
    from qualia_core.typing import RecursiveConfigDict

logger = logging.getLogger(__name__)

def parse_args(args: list[str]) -> RecursiveConfigDict:
    def lookup_paths_to_dict(lookup_key_value_pairs: RecursiveConfigDict) -> RecursiveConfigDict:
        result_dict: RecursiveConfigDict = {}

        for k, v in lookup_key_value_pairs.items():
            if len(k) == 1: # Last item of lookup path, assign directly
                result_dict[k[0]] = v
            else: # Visit rest of path and merge into result dict
                result_dict_element = result_dict.get(k[0], {})
                if not isinstance(result_dict_element, dict):
                    logger.error("Conflicting value type for existing result key '%s', expected dict, got: %s",
                                 k[0],
                                 type(result_dict_element))
                    raise TypeError
                result_dict[k[0]] = merge_dict(result_dict_element, lookup_paths_to_dict({k[1:]: v}), merge_lists=True)

        return result_dict

    args_key_value_pairs = {}
    for arg in args:
        if not arg.startswith('--'):
            logger.error('Unknown argument: %s', arg)
            raise ValueError
        arg_expr = arg[2:] # remove '--'
        key, value = arg_expr.split('=', 2)
        key = tuple(key.split('.'))
        args_key_value_pairs[key] = ast.literal_eval(value) # Parse a literal to its correct type

    args_dict = lookup_paths_to_dict(args_key_value_pairs)
    logger.info('Command line parameters: %s', args_dict)
    return args_dict
