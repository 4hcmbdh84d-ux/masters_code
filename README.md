Use the run_... .py file with:

cd ~/Downloads
caffeinate -i -s python3 -u run_hexatic_weekend_matrix.py \
  --seeds __ __ __ \ (numbers separated with spaces; doesn't need to be three)
  --lambdas __ __ __ \ (numbers separated with spaces; doesn't need to be three)
  --betas __ __ \ (numbers separated with spaces; doesn't need to be two)
  --epochs __ \ (one number; usually, in my case, 15000)
  --cooldown-minutes __ (one number; can be zero)

Based on work found here: https://github.com/fausto-martelli/Disordered-Hyperuniformity-Generator
