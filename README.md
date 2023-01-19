This is the official code accompanying our paper __"[Rethinking the Objectives of Extractive Question Answering](https://arxiv.org/abs/2008.12804)"__.  
If you use this code in your publication, please cite
```
@inproceedings{fajcik-etal-2021-rethinking,
    title = "Rethinking the Objectives of Extractive Question Answering",
    author = "Fajcik, Martin  and
      Jon, Josef  and
      Smrz, Pavel",
    booktitle = "Proceedings of the 3rd Workshop on Machine Reading for Question Answering",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.mrqa-1.2",
    doi = "10.18653/v1/2021.mrqa-1.2",
    pages = "14--27",
}
```

### Resources
* Results of all individual experiments are available [here](https://docs.google.com/spreadsheets/d/1_X1hrzrx31PKV-rIKqPlJJfqeH9I5KWeEQvLRzY9nXU/edit?usp=sharing).
* Results of manual analysis from the paper are available [here](https://docs.google.com/spreadsheets/d/1yYCWWLc40WlB-jUNf91oRIom6Pt57Iz73QMhpbwUyCY/edit?usp=sharing).



### Running the code

Install dependencies from the `requirements.txt` into your python3.6 environment.
Before running any experiment, you will need to set your python's home directory to project's root folder.Always make sure you have an en_us locale setting.
```
# if you are using conda:
# conda create --name jointqa python=3.6
cd JointSpanExtraction
python -m pip install -r requirements.txt
export PYTHONPATH=$(pwd)
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
```

To replicate our results, explore the directory `src/scripts/run_files`. The directory contains _hit&run_ scripts, set to run on SQuADv1.1 by default. For instance, you can run compound objective training by executing:
```
python src/scripts/run_files/run_transformer_reader_compound.py
```

To play around with the models, you can adjust the configuration in the run file, e.g. by changing dataset to squad2 as:
```
"dataset": "squad2_transformers"
```

Similarly, you can change the transformer type by adjusting attributes `"tokenizer_type"` and `"model_type"` to e.g. `"albert-xxlarge-v1"`, or change similarity function by 
adjusting `"similarity_function"` attribute.


