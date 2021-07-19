This is the official code accompanying our paper __"Rethinking the Objectives of Extractive Question Answering"__.  
If you use this code in your publication, please cite
```
@article{fajcik2020rethinking,
  title={Rethinking the objectives of extractive question answering},
  author={Fajcik, Martin and Jon, Josef and Kesiraju, Santosh and Smrz, Pavel},
  journal={arXiv preprint arXiv:2008.12804},
  year={2020}
}
```

### Resources
* Results of all individual experiments are available [here](https://docs.google.com/spreadsheets/d/1yYCWWLc40WlB-jUNf91oRIom6Pt57Iz73QMhpbwUyCY/edit?usp=sharing).
* Results of manual analysis from the paper are available [here](https://docs.google.com/spreadsheets/d/1_X1hrzrx31PKV-rIKqPlJJfqeH9I5KWeEQvLRzY9nXU/edit?usp=sharing).



### Running the code

Install dependencies from the `requirements.txt` into your python3.6 environment.
Before running any experiment, you will need to set your python's home directory to projects root folder, for instance in linux shell and make sure you have an en_us locale setting:
```
cd JointSpanExtraction
export PYTHONPATH=$(pwd)
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
```

To replicate the results, explore the directory `src/scripts/run_files`. The directory contains _hit&run_ scripts, set to run on SQuADv1.1 by default. For instance, you can run compound objective training by executing:
```
python src/scripts/run_files/run_transformer_reader_compound.py
```

To play around with the models, you can adjust the configuration in the run file, e.g. by changing dataset to squad2 as:
```
"dataset": "squad2_transformers"
```

Similarly, you can change the transformer type by adjusting attributes `"tokenizer_type"` and `"model_type"` to e.g. `"albert-xxlarge-v1"`, or change similarity function by 
adjusting `"similarity_function"` attribute.


