# Experiment Configuration

exp_01: reconstruction loss + perception loss (only shading stage)
exp_02: reconstruction loss (only shading stage)
exp_03: reconstruction loss (cross stage training)'
exp_04: reconstruction loss (cross stage training), sine activation function
exp_05: reconstruction loss (cross stage training), relu activation function, only use rgb to estimate shadow map
exp_06: reconstruction loss (cross stage training), relu activation function, use gfft as positional embedding
exp_07: reconstruction loss (cross stage training), using CosineAnnealingLR scheduler