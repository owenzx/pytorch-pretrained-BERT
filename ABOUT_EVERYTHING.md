## TODO
<!--- - [x] MNLI different data size
- [x] Debug and finish coding general framework for sentence classification (multi-domain/single domain)
- [x] Add most basic data augmentation and check the results
- [x] Make every result a json file 
-->
- [x] Check what will be changed because of tuning BertAdam. Also Check if BertAdam is helpful for other models
- [ ] Make the subsampled dataset balanced
- [ ] Feature-based fine-tuning and cluster idea
- [ ] Check how the predictions improved in person
- [ ] LM fine-tuning
- [ ] Write script for running exp multiple times (in parallel)
- [x] Try more datasets
- [ ] Start doing different ways of Bert fine-tuning
- [ ] Set up Bert for QA



## Target
Generalizable Fine-tuning for Bert  (GFT)


Data-augmentation for low-resource sentence classification (DaLore)
    
    Not doing QA for now since data-augmentation for QA seems much more complicated than others

## Metrics
Number of training samples/portion of the original training set

GFT: Performance on both in-domain dev set and out-of-domain dev set

DaLore: 


## Datasets
### NLI

* MNLI
* SNLI
* SICK
* RTE
* WNLI
* SciTail

### QA
* SQuAD
* NewsQA
* TriviaQA
* SearchQA
* HotpotQA
* NaturalQuestions

* BioASQ
* DROP
* DuoRC
* RACE
* RelationExtraction
* TextbookQA

If a question has more than one answers, then only using the first one.

## Method
### Methods for generating augmentation
#### Rule-Based Minor Perturbation

#### Semantic-Preserving Paraphrase
* Seq2seq paraphrase model
* Paraphrase by back-translation

#### Other Augmentation Methods
* Interpolation
* Mask




## Experiment Plans
* Run Bert on MNLI for different portion of data and see how it is doing on MNLI/other datasets
* Try simple augmentation rules such as Tong's, masking, etc.



## Papers
* [mixup: BEYOND EMPIRICAL RISK MINIMIZATION](https://arxiv.org/pdf/1710.09412.pdf)
* [Unsupervised Data Augmentation](https://arxiv.org/pdf/1904.12848.pdf)