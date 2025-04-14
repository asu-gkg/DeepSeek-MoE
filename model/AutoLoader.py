import itertools
import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import torch
import torch.nn as nn
from torch.nn import Parameter

class AutoWeightsLoader:
    """
    Helper class to load weights into a :class:`torch.nn.Module`. It is able
    to automatically detect child modules and parameters while iterating over
    the weights only once.

    The weight loading logic for individual modules can be overridden
    by defining a ``load_weights`` method.

    Similarly, the weight loading logic for individual parameters can be
    overridden by defining a ``weight_loader`` method.
    """
    
    def __init__(self, 
                module: nn.Module,
                *,
                skip_prefixes: Optional[List[str]] = None,
                ignore_unexpected_prefixes: Optional[List[str]] = None,):
        self.module = module
        self.skip_prefixes = skip_prefixes or []
        self.ignore_unexpected_prefixes = ignore_unexpected_prefixes or []

    def load_weights(self, 
                    weights: Iterable[Tuple[str, Any]], 
                    *,
                    mapper = None) -> Set[str]:
        if mapper is not None:
            raise ValueError("mapper is not supported")
        
        autoloaded_weights = set(self._load_module("", self.module, weights))
        return autoloaded_weights
    
    def _add_loadable_non_param_tensors(self, module: nn.Module,
                                        child_params: Dict[str, torch.Tensor]):
        """
        Add tensor names that are not in the model params that may be in the
        safetensors, e.g., batch normalization stats.
        """
        if isinstance(module, (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.LazyBatchNorm1d,
                nn.LazyBatchNorm2d,
                nn.LazyBatchNorm3d,
                nn.SyncBatchNorm,
        )):
            module_state_dict = module.state_dict()
            for stat_name in ("running_mean", "running_var",
                            "num_batches_tracked"):
                child_params[stat_name] = module_state_dict[stat_name]
                
    def _groupby_prefix(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> Iterable[Tuple[str, Iterable[Tuple[str, torch.Tensor]]]]:
        weights_by_parts = ((weight_name.split(".", 1), weight_data)
                            for weight_name, weight_data in weights)

        for prefix, group in itertools.groupby(weights_by_parts,
                                            key=lambda x: x[0][0]):
            yield (
                prefix,
                # Because maxsplit=1 in weight_name.split(...),
                # the length of `parts` must either be 1 or 2
                (("" if len(parts) == 1 else parts[1], weights_data)
                for parts, weights_data in group),
            )

    def _load_param(self, 
                    prefix: str, 
                    param: nn.Parameter,
                    weights: Iterable[Tuple[str, torch.Tensor]]) -> Iterable[str]:
        pass
    
    def _load_module(self,
                    base_prefix: str,
                    module: nn.Module, 
                    weights: Iterable[Tuple[str, torch.Tensor]]) -> Iterable[str]:
        
        # Avoid infinite recursion since this function is typically
        # called inside load_weights of the module itself
        if module != self.module:
            module_load_weights = getattr(module, "load_weights", None)
            if callable(module_load_weights):
                loaded_params = module_load_weights(weights)
                if loaded_params is None:
                    logging.warning(
                        "Unable to collect loaded parameters "
                        "for module %s", module)
                else:
                    yield from map(
                        lambda x: self._get_qualname(base_prefix, x),
                        loaded_params,
                    )

        child_modules = dict(module.named_children())
        child_params = dict(module.named_parameters(recurse=False))

        self._add_loadable_non_param_tensors(module, child_params)
        
        for child_prefix, child_weights in self._groupby_prefix(weights):
            prefix = self._get_qualname(base_prefix, child_prefix)
            print(f"prefix: {prefix}")

            if child_prefix in child_modules:
                if self._can_skip(prefix + "."):
                    logging.debug("Skipping module %s", prefix)

                    continue

                yield from self._load_module(prefix,
                                            child_modules[child_prefix],
                                            child_weights)
            elif child_prefix in child_params:
                if self._can_skip(prefix):
                    logging.debug("Skipping param %s", prefix)

                    continue

                yield from self._load_param(prefix, child_params[child_prefix],
                                            child_weights)
            else:
                can_skip_module = self._can_skip(prefix + ".")
                can_skip_param = self._can_skip(prefix)
                if can_skip_module or can_skip_param:
                    logging.debug("Skipping missing %s", prefix)

                    continue

                can_ignore_module = self._can_ignore_unexpected(prefix + ".")
                can_ignore_param = self._can_ignore_unexpected(prefix)
                if can_ignore_module or can_ignore_param:
                    logging.debug("Ignoring missing %s", prefix)

                    continue

                msg = (f"There is no module or parameter named '{prefix}' "
                        f"in {type(self.module).__name__}")
                raise ValueError(msg)
    
    def _get_qualname(self, prefix: str, rest: str) -> str:
        if prefix == "":
            return rest
        if rest == "":
            return prefix

        return ".".join((prefix, rest))

    def _can_skip(self, qualname: str) -> bool:
        return any(qualname.startswith(p) for p in self.skip_prefixes)

    def _can_ignore_unexpected(self, qualname: str) -> bool:
        return any(
            qualname.startswith(p) for p in self.ignore_unexpected_prefixes)
