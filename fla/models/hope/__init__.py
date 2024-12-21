# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.hope.configuration_hope import hopeConfig
from fla.models.hope.modeling_hope import (
    hopeForCausalLM, hopeModel)

AutoConfig.register(hopeConfig.model_type, hopeConfig)
AutoModel.register(hopeConfig, hopeModel)
AutoModelForCausalLM.register(hopeConfig, hopeForCausalLM)


__all__ = ['hopeConfig', 'hopeForCausalLM', 'hopeModel']
