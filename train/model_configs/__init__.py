"""Model configuration builders for different architectures."""

from .deepseek_config import create_deepseek_config
from .diffusion_configs import create_llada_config, create_sedd_config
from .mla_configs import (
    create_mla_config, create_mla_model,
    create_mla_selective_config, create_mla_selective_model,
    create_parscale_mla_config, create_parscale_mla_model,
    create_mla_llada_config, create_mla_llada_model
)
from .moe_configs import create_moe_mla_config, create_moe_mla_model
from .gpt_config import create_gpt_config
from .mdm_config import create_mdm_config, create_mdm_model
from .swan_configs import create_swan_config, create_swa_mla_config
from .specialized_configs import (
    create_slm_config, create_slm_model,
    create_nsa_config, create_nsa_model,
    create_hse_config, create_hse_model,
    create_hrm_config, create_hrm_model
)

__all__ = [
    # DeepSeek
    'create_deepseek_config',
    # Diffusion models
    'create_llada_config',
    'create_sedd_config',
    # MLA variants
    'create_mla_config',
    'create_mla_model',
    'create_mla_selective_config',
    'create_mla_selective_model',
    'create_parscale_mla_config',
    'create_parscale_mla_model',
    'create_mla_llada_config',
    'create_mla_llada_model',
    # MoE
    'create_moe_mla_config',
    'create_moe_mla_model',
    # GPT
    'create_gpt_config',
    # MDM
    'create_mdm_config',
    'create_mdm_model',
    # SWAN and SWA_MLA
    'create_swan_config',
    'create_swa_mla_config',
    # Specialized models
    'create_slm_config',
    'create_slm_model',
    'create_nsa_config',
    'create_nsa_model',
    'create_hse_config',
    'create_hse_model',
    'create_hrm_config',
    'create_hrm_model',
]