#!/bin/bash

# Domains to fine-tune (excluding 'cs')
DOMAINS=("eess" "math" "physics" "q-bio" "stat")

echo "Starting fine-tuning for domains: ${DOMAINS[*]}"

for domain in "${DOMAINS[@]}"; do
    echo "============================================================"
    echo "Starting fine-tuning for domain: $domain"
    echo "============================================================"
    
    python3 finetune_lora.py --domain "$domain"
    
    if [ $? -ne 0 ]; then
        echo "Error: Fine-tuning failed for domain '$domain'. Exiting."
        exit 1
    fi
    
    echo "Successfully completed fine-tuning for domain: $domain"
done

echo "All specified domains have been fine-tuned successfully."
