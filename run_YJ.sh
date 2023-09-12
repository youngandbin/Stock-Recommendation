#!/bin/bash

# Initialize a counter to limit parallel jobs
counter=0
max_jobs=5 # Maximum number of parallel jobs

for GNN in graph_attention graph_NGCF graph_LightGCN graph_UltraGCN graph_sum identity; do
    for RNN in gru rnn lstm transformer; do
        echo "Running with GNN=$GNN and RNN=$RNN"

        # Run the command in the background
        python train_self_supervised.py --use_memory --memory_updater $RNN --embedding_module $GNN --prefix ${GNN}_$RNN --data transaction &

        # Increment counter
        ((counter++))

        # Wait for all background jobs to complete if counter reaches max_jobs
        if [ $counter -eq $max_jobs ]; then
            wait
            counter=0
        fi
    done
done

# Wait for any remaining background jobs to complete
wait
