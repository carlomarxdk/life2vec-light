[![DOI](https://zenodo.org/badge/747774269.svg)](https://zenodo.org/doi/10.5281/zenodo.10680995)

# life2vec-light
Basic implementation of the `life2vec` model with the dummy data. 

This repository contains basic code for the [Using Sequences of Life-events to Predict Human Lives](https://www.nature.com/articles/s43588-023-00573-5) (life2vec) paper. The [SocialComplexityLab/life2vec](https://github.com/SocialComplexityLab/life2vec) depends (in large) on the specific structure of the data and the version of the packages available.

Here, we provide a code with the simple dummy data (+ using the latest versions of Python packages) - the code contains only the backbone of the model.
Thus, you can easily extend it for your specific use. 

Open the `simple_workflow.ipynb` notebook to try a simple step-by-step workflow. The specifications for the Anaconda environment are located in `Environment.yml`; I run the code on the Windows Subsystem for Linux (WS2).

## What is missing?

Compared to the [original implementation](https://github.com/SocialComplexityLab/life2vec) of the `life2vec` model, this repository does not include the following:
1. Bookkeeping and logging functionality,
2. Data visualisation functionality,
3. Code for experiments, including robustness tests,
4. Implementation of loss and metric functions mentioned in the paper.


## Data

You can generate dummy data (that we use here as an example) using the Jupyter Notebook in `misc/synthetic_data.ipynb`:
1. Generate the dummy *user database*, aka `users.csv`,
2. Generate the dummy *labor dataset*, aka `synth_labor.csv`,
3. Move both to the `data\rawdata` folder.

## To Do

Note on 16th FEB 2024: **Due to some package updates, the process the takes a bit more time than I expected, I still work on the code**. 
- [x] Add code for the pretraining 
- [ ] More detailed annotation
- [x] Add code with working data pipeline (by the 20th Feb)
- [x] Add Data Example (misc/synthetic_data.ipynb)
- [x] Create a Source file for the Synthetic Labor Data (src/sources/synth_labor.py)
- [ ] Add Logging support
- [ ] Add finetuning example with the specialised decoder


## Citations
### How to cite THIS code
```bibtex
@misc{https://doi.org/10.5281/zenodo.10680995,
      doi = {10.5281/ZENODO.10680995},
      url = {https://zenodo.org/doi/10.5281/zenodo.10680995},
      author = {Germans Savcisens},
      title = {Github Repository for carlomarxdk/life2vec-light},
      publisher = {Zenodo},
      year = {2024},
      copyright = {Creative Commons Attribution 4.0 International}
}
```
If you want to cite a *specific* release, check the DOI number at the top of the README file (or see details of the release).

### Connected Materials
```bibtex
@article{savcisens2024using,
      author={Savcisens, Germans and Eliassi-Rad, Tina and Hansen, Lars Kai and Mortensen, Laust Hvas and Lilleholt, Lau and Rogers, Anna and Zettler, Ingo and Lehmann, Sune},
      title={Using sequences of life-events to predict human lives},
      journal={Nature Computational Science},
      year={2024},
      month={Jan},
      day={01},
      volume={4},
      number={1},
      pages={43-56},
      issn={2662-8457},
      doi={10.1038/s43588-023-00573-5},
      url={https://doi.org/10.1038/s43588-023-00573-5}
}
```

```bibtex
@misc{life2vec_code,
  author = {Germans Savcisens},
  title = {Official code for the "Using Sequences of Life-events to Predict Human Lives" paper},
  note = {GitHub: SocialComplexityLab/life2vec},
  year = {2023},
  howpublished = {\url{https://doi.org/10.5281/zenodo.10118621}},
}
```
