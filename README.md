# life2vec-light
Basic implementation of the life2vec model with the dummy data. 

This repository contains basic code for the [Using Sequences of Life-events to Predict Human Lives](https://www.nature.com/articles/s43588-023-00573-5) (life2vec) paper. The [SocialComplexityLab/life2vec](https://github.com/SocialComplexityLab/life2vec) depends (in large) on the specific structure of the data, and the version of the packages available.

Here, we provide a code used with the simple dummy data (+ using the latest versions of Python packages). Thus, you can easily extended it for your specific use.

## TO-DO

Note on 16th FEB 2024: **Due to some package updates, the process the takes a bit more time than I expected, I still work on the code**. 
- [x] Add code for the pretraining (12FEB2024)
- [ ] More detailed annotation
- [ ] Add code with working data pipeline (by the 15th Feb)
- [ ] Add Data Example (by 15th Feb)
- [ ] Add Logging support
- [ ] Add finetuning example with the specialised decoder
- [ ] Make a package (?)
- [ ] Add `Hydra` Support

## Citations
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
