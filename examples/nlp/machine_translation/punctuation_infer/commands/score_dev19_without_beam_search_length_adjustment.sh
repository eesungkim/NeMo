work_dir=/media/apeganov/DATA/punctuation_and_capitalization/simplest/3_128/wiki_wmt_17.01.2022
output_dir="${work_dir}/inference_on_IWSLT_tst2019_results"
model_name=nmt_wmt_base_bs800000_steps300000_lr2e-4
python compute_metrics.py \
  --hyp "${output_dir}/${model_name}_without_adjustment_labels.txt" \
  --ref "${work_dir}/IWSLT_tst2019/autoregressive_labels.txt" \
  --output "${output_dir}/${model_name}_without_adjustment_scores.json"