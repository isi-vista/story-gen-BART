#!/usr/bin/env bash

# Path to WritingPrompts
data_dir=$1
# Path to output directory
out_dir=$2
# Path to co-reference model
coref_model=$3
# Path to SRL model
srl_model=$4

types="test valid train"

echo "Cleaning WritingPrompts..."
for type in ${types}; do
    echo "\tcleaning $type"
    python convert_wp.py \
      --input-prompts ${data_dir}/${type}.wp_source \
      --input-stories ${data_dir}/${type}.wp_target \
      --output ${out_dir}/
done

## Generate storylines by SRL
echo "Extracting SRL plots..."
for type in ${types}; do
    echo "\tprocessing $type"
    python srl_to_storyline.py \
      --input_file ${out_dir}/${type}.clean.txt \
      --output_file ${out_dir}/WP.storyline_dic.${type}.json \
      --save_coref_srl ${out_dir}/WP.srl_coref.${type}.json \
      --label_story ${out_dir}/WP.label_story.${type} \
      --coref_model ${coref_model} \
      --title ${out_dir}/WP.title.${type} \
      --srl_model ${srl_model} \
      --batch 8 \
      --cuda -1
done

echo "Assembling final file format..."
#change storyline format from dictionary to string
for type in ${types}; do
    echo "\tprocessing $type"
    python prepare_SRL_storyline_format.py \
      --input_file ${out_dir}/WP.storyline_dic.${type}.json \
      --output_file ${out_dir}/WP.storyline.${type}
done

# Concat titles and storylines
echo "Concatenating titles and storylines..."
for type in ${types}; do
    echo "\tprocessing $type"
    python concat.py  \
      --title_file ${out_dir}/WP.title.${type} \
      --kw_file ${out_dir}/WP.storyline.${type} \
      --story_file ${out_dir}/WP.label_story.${type} \
      --data_type ${type} \
      --target_dir ${out_dir} \
      --title_key
done

echo "Preparing final directories..."
mkdir ${out_dir}/plot
mkdir ${out_dir}/story
for type in ${types}; do
  cp ${out_dir}/${type}.titles ${out_dir}/plot/${type}.source
  cp ${out_dir}/WP.storyline.${type} ${out_dir}/plot/${type}.target
  cp ${out_dir}/WP.titlesepkey.${type} ${out_dir}/story/${type}.source
  cp ${out_dir}/${type}.stories ${out_dir}/story/${type}.target
done
