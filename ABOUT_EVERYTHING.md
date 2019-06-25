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
* First learn policy on large datasets and then transfer to small datasets

    Training set: Leave one small portion untouched in training to train when doing validation. 
    
    
    Training: Each time, sample a small training set and a large validation set (large enough for a stable evaluation). Apply augmentation, train model, get reward, update policy.
    Validate: Train on the remaining part and get acc on the validation set
    
    Testing: Apply policy on other datasets and train/test 
    
    Alternative: keep training policy just on a small portion of training data, but have the risk of overfitting, but it may also stablize the training, so should try both.
    
    Subpolicy: 
    
    Strongest baseline: First fine-tune Bert on the large training set and then apply Bert to the smaller training set (both w/wo data augmentation) Should beat wo/wo, hopefully beat w/wo (but may need to transfer to distant domain)
    
    
    Weak setting: assume no high resource, train on a small set and test on a small set
    
    



## Papers
* [mixup: BEYOND EMPIRICAL RISK MINIMIZATION](https://arxiv.org/pdf/1710.09412.pdf)
* [Unsupervised Data Augmentation](https://arxiv.org/pdf/1904.12848.pdf)