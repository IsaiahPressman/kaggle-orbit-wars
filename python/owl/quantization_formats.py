from __future__ import annotations

from typing import Literal, TypeAlias

QuantizationFormat: TypeAlias = Literal[
    "fp8_e4m3fn",
    "fp4_e2m1fn_x2_scaled_block16",
    "nf5_g128_lsq_policy_last_fp8",
    "nf4_g128_lsq",
    "nf3_nf4_structured_3p5",
    "nf3_g128_lsq",
]

FP8_E4M3FN: QuantizationFormat = "fp8_e4m3fn"
FP4_E2M1FN_X2_SCALED_BLOCK16: QuantizationFormat = "fp4_e2m1fn_x2_scaled_block16"
NF5_G128_LSQ_POLICY_LAST_FP8: QuantizationFormat = "nf5_g128_lsq_policy_last_fp8"
NF4_G128_LSQ: QuantizationFormat = "nf4_g128_lsq"
NF3_NF4_STRUCTURED_3P5: QuantizationFormat = "nf3_nf4_structured_3p5"
NF3_G128_LSQ: QuantizationFormat = "nf3_g128_lsq"
SUPPORTED_QUANTIZATION_FORMATS: tuple[QuantizationFormat, ...] = (
    FP8_E4M3FN,
    FP4_E2M1FN_X2_SCALED_BLOCK16,
    NF5_G128_LSQ_POLICY_LAST_FP8,
    NF4_G128_LSQ,
    NF3_NF4_STRUCTURED_3P5,
    NF3_G128_LSQ,
)
