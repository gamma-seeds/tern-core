======================================================================
  TinyLlama-1.1B Weight Analysis
======================================================================

LAYER TYPE PROFILES
----------------------------------------------------------------------
Type         | Count |   Mean|W| |       Std | Spars@0.7 |   QE@0.7 | Kurtosis | EffRank
----------------------------------------------------------------------------------------
up_proj      |    22 |  0.014067 |  0.017735 |    42.7% |   0.4444 |     0.44 |   205.0
down_proj    |    22 |  0.013831 |  0.017468 |    42.8% |   0.4469 |     1.07 |   204.4
gate_proj    |    22 |  0.016071 |  0.020443 |    43.1% |   0.4560 |     1.32 |   202.0
v_proj       |    22 |  0.011335 |  0.014606 |    44.2% |   0.4627 |     0.75 |   200.6
o_proj       |    22 |  0.011879 |  0.015236 |    43.7% |   0.4683 |     4.45 |   199.6
lm_head      |     1 |  0.019269 |  0.024723 |    43.3% |   0.4705 |     1.67 |   199.5
q_proj       |    22 |  0.019225 |  0.025857 |    46.9% |   0.5066 |     8.55 |   183.5
k_proj       |    22 |  0.033193 |  0.044910 |    47.3% |   0.5100 |    15.41 |   173.1

Type homogeneity (are all layers of a type similar?):
  down_proj   : QE std=0.0041  sparsity std=0.0022  → YES
  gate_proj   : QE std=0.0082  sparsity std=0.0038  → YES
  k_proj      : QE std=0.0551  sparsity std=0.0379  → NO
  lm_head     : QE std=0.0000  sparsity std=0.0000  → YES
  o_proj      : QE std=0.0318  sparsity std=0.0153  → NO
  q_proj      : QE std=0.0444  sparsity std=0.0269  → NO
  up_proj     : QE std=0.0040  sparsity std=0.0014  → YES
  v_proj      : QE std=0.0131  sparsity std=0.0082  → NO

BLOCK DEPTH ANALYSIS
----------------------------------------------------------------------
  blocks_0_5      : avg QE=0.4806  avg sens=1202.72x  avg |W|=0.016150  (42 layers)
  blocks_6_10     : avg QE=0.4658  avg sens=1.39x  avg |W|=0.017066  (35 layers)
  blocks_11_15    : avg QE=0.4691  avg sens=1.00x  avg |W|=0.017084  (35 layers)
  blocks_16_21    : avg QE=0.4662  avg sens=1.00x  avg |W|=0.018041  (42 layers)

QUANT ERROR vs SENSITIVITY CORRELATION
----------------------------------------------------------------------
  Pearson r (all data):        -0.3067  (n=20)
  Pearson r (log sensitivity):  -0.1816
  Pearson r (excl. outlier):    0.6662
  Outlier: model.layers.2.mlp.down_proj (ratio 9609.3x)
  Interpretation: MODERATE predictor — some predictive value

TERNARY FRIENDLINESS RANKING
----------------------------------------------------------------------
Top 10 most ternary-friendly:
   1. model.layers.1.mlp.gate_proj                       score=0.5539 (gate_proj)
   2. model.layers.3.self_attn.v_proj                    score=0.5520 (v_proj)
   3. model.layers.9.self_attn.o_proj                    score=0.5475 (o_proj)
   4. model.layers.13.self_attn.v_proj                   score=0.5472 (v_proj)
   5. model.layers.4.self_attn.o_proj                    score=0.5471 (o_proj)
   6. model.layers.20.self_attn.v_proj                   score=0.5451 (v_proj)
   7. model.layers.10.self_attn.o_proj                   score=0.5438 (o_proj)
   8. model.layers.12.self_attn.o_proj                   score=0.5392 (o_proj)
   9. model.layers.14.self_attn.v_proj                   score=0.5368 (v_proj)
  10. model.layers.17.self_attn.v_proj                   score=0.5251 (v_proj)

Bottom 5 least ternary-friendly:
      model.layers.4.self_attn.q_proj                    score=0.2530 (q_proj)
      model.layers.4.self_attn.k_proj                    score=0.2180 (k_proj)
      model.layers.5.self_attn.k_proj                    score=0.2080 (k_proj)
      model.layers.5.self_attn.q_proj                    score=0.2007 (q_proj)
      model.layers.2.mlp.down_proj                       score=0.0001 (down_proj)

BIMODAL/TRIMODAL DISTRIBUTION DETECTION
----------------------------------------------------------------------
  No bimodal distributions detected (all unimodal)

OUTLIER WEIGHT DETECTION (>5 std from mean)
----------------------------------------------------------------------
  Found 155 layers with extreme outlier weights:
    model.layers.0.self_attn.k_proj                    max_z=37.1  min_z=97.9  range=[-3.1094, 1.1797]
    model.layers.0.self_attn.q_proj                    max_z=62.9  min_z=96.7  range=[-1.5859, 1.0312]
    model.layers.2.mlp.gate_proj                       max_z=54.2  min_z=63.1  range=[-1.1641, 1.0000]
    model.layers.7.mlp.down_proj                       max_z=60.7  min_z=40.6  range=[-0.6797, 1.0156]
    model.layers.2.mlp.down_proj                       max_z=30.2  min_z=47.9  range=[-0.8359, 0.5273]
    model.layers.8.self_attn.o_proj                    max_z=47.6  min_z=30.1  range=[-0.4375, 0.6914]
    model.layers.12.mlp.down_proj                      max_z=47.4  min_z=24.5  range=[-0.4160, 0.8047]
    model.layers.13.mlp.down_proj                      max_z=28.6  min_z=46.4  range=[-0.7969, 0.4902]
    model.layers.0.self_attn.o_proj                    max_z=45.9  min_z=41.2  range=[-0.3418, 0.3809]
    model.layers.2.self_attn.o_proj                    max_z=42.5  min_z=41.9  range=[-0.5898, 0.5977]

======================================================================
  TOP DISCOVERIES
======================================================================

  1. Quant error correlates with sensitivity (r=0.666 excl. outlier). Can be used as a cheap proxy for sensitivity analysis.

======================================================================
