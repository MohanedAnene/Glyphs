# Loop over learning rates and decay values
for lr in 0.0001 0.0002 0.0005 0.00001 0.00002 0.00005; do
  for lrd in 0.7 0.75 0.8 0.85 0.9; do
    echo "Running: python experiment/pairwise_training_experiment.py learning_rate=$lr learning_decay=$lrd"
    python experiment/pairwise_training_experiment.py learning_rate=$lr learning_decay=$lrd
  done
done
