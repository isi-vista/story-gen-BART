# Processing Storyline

These scripts are used for processing storylines based on semantic role labels and co-reference resolution.

## 1. Clean up WritingPrompts
Use `convert_wp.py` to clean up the *WritingPrompts* data.

## 2. Co-reference resolution and Semantic Role Labelling
`srl_to_storyline.py` contains code to run SRL and co-ref to generate storylines. The input is story, and output is a dictionary of storyline. It will save all parsed co-reference and SRL information for future use.

## 3. Convert to final file format

`prepare_SRL_storyline_format.py` changes storyline format and deletes "{}", extracts all useful information from the dictionary to a string and uses "#" to separate.

## Automation

`generate_WP_training_files_srl.sh` can run all the steps automatically. You can change parameters here.

# Requirements
To do storyline processing based on SRL+Coref, you need to meet these requirements.

## 1. Packages
Use the [pip](https://pip.pypa.io/en/stable/) to install the packages in `requirements.txt`.

## 2. Data
 Download WritingPrompts and put the six files into:
```
data/writingPrompts/ready_srl/
```
otherwise, you need to modify the parent_dir to your own path in line 6 in
```
/generate_WP_training_files_srl.sh
```


## Usage
With all done, you just need to run this command:

```bash
bash generate_WP_training_files_srl.sh data_dir out_dir coref_path srl_path
```
For the batch size and CUDA device, look at  `generate_WP_training_files_srl.sh`

PS: According to the AllenNLP documentation, the argument --cuda refers to the cuda device ID. The type of this parameter is int, different from pytorch which is a boolean, so if we want to use GPU, we need to set --cuda 0, otherwise, CPU is --cuda -1.
