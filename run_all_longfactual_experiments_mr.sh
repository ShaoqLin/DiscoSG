#!/bin/bash

export PYTHONPATH="."
export CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER='PCI_BUS_ID'

# Create log directory
LOG_DIR="logs/discosg_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

run_script() {
    local script_path="$1"
    local args="$2"
    
    local script_name=$(basename "$script_path" .py)
    local log_file="${LOG_DIR}/${script_name}_$(date +%H%M%S).log"
    
    echo "====================================" | tee -a "$log_file"
    echo "Executing: python $script_path $args" | tee -a "$log_file"
    echo "Start time: $(date)" | tee -a "$log_file"
    echo "====================================" | tee -a "$log_file"
    
    python "$script_path" $args 2>&1 | tee -a "$log_file"
    
    echo "====================================" | tee -a "$log_file"
    echo "End time: $(date)" | tee -a "$log_file"
    echo "Script execution completed" | tee -a "$log_file"
    echo "====================================" | tee -a "$log_file"
    
    echo "Script execution completed, log saved at: $log_file"
    echo ""
}

echo "Batch execution start time: $(date)" > "${LOG_DIR}/00_summary.log"
echo "Environment variables:" >> "${LOG_DIR}/00_summary.log"
echo "PYTHONPATH=$PYTHONPATH" >> "${LOG_DIR}/00_summary.log"
echo "NLTK_DATA=$NLTK_DATA" >> "${LOG_DIR}/00_summary.log"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >> "${LOG_DIR}/00_summary.log"
echo "HF_HOME=$HF_HOME" >> "${LOG_DIR}/00_summary.log"
echo "" >> "${LOG_DIR}/00_summary.log"

run_script "detailcap_discosg_mr.py" \
"--dataset discosg \
--max_input_length 512 \
--max_output_length 512 \
--num_beams 5 \
--model_path <Model Path from cloud drive> \
--capture \
--save_folder eval_res/eval_res_run_all_discosg \
--round 2 \
--bs_scale 2"

# Record script end time
echo "Batch execution end time: $(date)" >> "${LOG_DIR}/00_summary.log"
echo "All scripts executed, log files saved in directory: $LOG_DIR"
