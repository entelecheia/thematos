<!--next-version-placeholder-->

## v0.14.5 (2024-10-21)

### Fix

* **dependencies:** Update tomotopy to ^0.13.0 ([`f4f545b`](https://github.com/entelecheia/thematos/commit/f4f545baa89c63c499693c63e525de5b575a7726))
* **book:** Update jupyter-book version in requirements.txt ([`cfb6764`](https://github.com/entelecheia/thematos/commit/cfb6764836226cbd6add2ac7a8838cc2e22733cd))

## v0.14.4 (2024-03-27)

### Fix

* **dependencies:** Update Python hyfi, lexikanon, black versions ([`f83ffdb`](https://github.com/entelecheia/thematos/commit/f83ffdb2c327507959a369a4b27a07a94189812d))

## v0.14.3 (2023-08-24)

### Fix

* **thematos/runners:** Replace BaseRunner with Runner in TopicRunner ([`3533197`](https://github.com/entelecheia/thematos/commit/35331970711a5ab8093f6654695c9cdb8c58bdfc))
* **thematos:** Change TopicModel parent from BatchTaskConfig to BatchTask ([`8c0a71d`](https://github.com/entelecheia/thematos/commit/8c0a71d88da870892b62a4392a9bc1abd980f9c2))
* **thematos/datasets:** Change from BatchTaskConfig to BatchTask and RunConfig to Run ([`6c65558`](https://github.com/entelecheia/thematos/commit/6c655582f4bfb43b41a0cf87786da195d8bd6a97))
* **init_project:** Rename to initialize and update parameters ([`c855dbe`](https://github.com/entelecheia/thematos/commit/c855dbe8ef32c70a3585cab616f430439fb143e7))
* **Makefile:** Add trust flag to copier copy command ([`4904341`](https://github.com/entelecheia/thematos/commit/4904341c3961449d2b496aed3c9e317024988e2c))
* **dependencies:** Upgrade hyfi to 1.32.1 ([`adfd8cf`](https://github.com/entelecheia/thematos/commit/adfd8cf358c5fab2c37e5dcef317dea91120762d))

## v0.14.2 (2023-08-18)

### Fix

* **dependencies:** Upgrade hyfi to 1.29.8 (fix plugin issues) ([`1cd28a5`](https://github.com/entelecheia/thematos/commit/1cd28a5e99dd2303dfecf5ad9c0fa2afb6ac5aec))

## v0.14.1 (2023-08-15)

### Fix

* **book:** Update introduction information and links ([`acbeb30`](https://github.com/entelecheia/thematos/commit/acbeb30b980e0f20f71632f17cdc0799404d8b82))

### Documentation

* **README:** Update badges and project description ([`8c957b9`](https://github.com/entelecheia/thematos/commit/8c957b97ccea21b9b70f4bac8396e4bf3e8fd75c))

## v0.14.0 (2023-08-14)

### Feature

* **tests:** Add workflow test in thematos ([`8ae381f`](https://github.com/entelecheia/thematos/commit/8ae381f6a61a7402d066b2871c98313f9d32b881))
* **topic:** Add model loading and inference methods ([`fe625f6`](https://github.com/entelecheia/thematos/commit/fe625f6060437f260cfa9e26be143c8f4f542ada))
* **thematos/runners/config:** Add InferConfig class ([`7e1aa0c`](https://github.com/entelecheia/thematos/commit/7e1aa0cc7b3ae0f113da38203de9dfa29725be7a))
* **runners:** Add InferConfig to the module exports ([`701aab3`](https://github.com/entelecheia/thematos/commit/701aab33e75f7bf30eb7b28e7193bd20bc06a2cd))
* **thematos/config:** Add model_config_file in infer_topics.yaml ([`8af15a3`](https://github.com/entelecheia/thematos/commit/8af15a3bfd74e51ed8ebca973ccbf1f2d99c4c72))
* **runner:** Add infer topics configuration in topic runner ([`130128c`](https://github.com/entelecheia/thematos/commit/130128c1eed3282c5106c3f94d6b9bb11c4bb7e4))
* **thematos/config:** Add infer_topics configuration ([`7ec0eca`](https://github.com/entelecheia/thematos/commit/7ec0eca33d45525ff6b0240033cdc574c4e514be))
* **thematos:** Add lexikanon to plugins ([`edef2e0`](https://github.com/entelecheia/thematos/commit/edef2e06fe78817c34dd903274a5da5f7ba60055))
* **runner:** Add train calls to topic.yaml ([`5a3756d`](https://github.com/entelecheia/thematos/commit/5a3756d2cecdf2e8020a7bbcd9cefcc94a6de589))
* **tests:** Add sample_data files for testing ([`aeeb33e`](https://github.com/entelecheia/thematos/commit/aeeb33ee8ba62cb1b5e2a29ef32a70debee14de9))
* **tests:** Add new word_prior assets ([`f4cdb4c`](https://github.com/entelecheia/thematos/commit/f4cdb4c39cefef397bf4caa22c4524d179b652d0))

## v0.13.0 (2023-08-14)

### Feature

* **TopicModel:** Add topic top words saving functionality ([`463c12f`](https://github.com/entelecheia/thematos/commit/463c12fd36f51802b551a92d7955fb3b17bcd164))

### Fix

* **corpus:** Improve logging and data access ([`3bf526c`](https://github.com/entelecheia/thematos/commit/3bf526c63e2e3882248d909462f4d0e0fa73e479))
* **thematos/models:** Add functionality to extend topic_words_freq_tuple to topic_top_words_dists ([`f2e84e2`](https://github.com/entelecheia/thematos/commit/f2e84e29135e5faa8c80377b271401d4dad8e085))

## v0.12.0 (2023-08-13)

### Feature

* **datasets/corpus:** Add Stopwords class and integrate with tokenizer ([`2d6de3b`](https://github.com/entelecheia/thematos/commit/2d6de3bface8f8f41050f575b6ea17465b92818f))
* **thematos:** Add stopwords initialization in topic_corpus configuration ([`714dc89`](https://github.com/entelecheia/thematos/commit/714dc89a5777241449509f83f1e8a96db3ecdd4d))

### Fix

* **thematos/models:** Add verbose logging for word prior setting in TopicModel ([`7ab1029`](https://github.com/entelecheia/thematos/commit/7ab10293085a4947392563c8910598fc93ff0c37))

## v0.11.0 (2023-08-13)

### Feature

* **thematos/models:** Add TrainSummaryConfig in config.py ([`bfe7517`](https://github.com/entelecheia/thematos/commit/bfe75176dc94ba6b0f5e586dda95f940df47ff96))
* **thematos/models:** Add TrainSummaryConfig in imports and TopicModel, modify the method of calling model.summary, assign default titles if not set in wc_args ([`ef40b3f`](https://github.com/entelecheia/thematos/commit/ef40b3f6ebda13ecc80e2d786c5e0c1ffcea4689))
* **thematos/models:** Include TrainSummaryConfig in imports and __all__ list. ([`111ea9f`](https://github.com/entelecheia/thematos/commit/111ea9f1527d7a1dfbd4f2b55a09eb73dee1f57d))
* **thematos/conf/model:** Add new file for topic train configuration ([`2e19572`](https://github.com/entelecheia/thematos/commit/2e195726ab41ae52c2d55f0327e2d8fbb2a4b100))
* **thematos:** Add topic_train to topic model configuration ([`c819175`](https://github.com/entelecheia/thematos/commit/c819175500a145fb4c713413e6c079d4b4853e25))
* **thematos/model:** Add topic_train to lda.yaml training args ([`291791c`](https://github.com/entelecheia/thematos/commit/291791cb351cdbbc601e553ef2b51cba5a9fca65))

### Fix

* **thematos:** Add directory creation for train summary file ([`acbce7d`](https://github.com/entelecheia/thematos/commit/acbce7d29985d3e627b05c9b7277dc629d41a082))
* **thematos:** Reset model in train method ([`aaea9bb`](https://github.com/entelecheia/thematos/commit/aaea9bbc58b6aa0c5f995e166db0c9f97bf58d37))
* **thematos/models:** Add save_train_summary method ([`fd57a66`](https://github.com/entelecheia/thematos/commit/fd57a66984a830a0cf0b476bc6144aa427bfc7d3))
* **thematos:** Replace 'k' with 'num_topics' in TopicModel class ([`8a14d3c`](https://github.com/entelecheia/thematos/commit/8a14d3c89ee38c61bf1e6df5471278221a45975f))
* **thematos/datasets:** Add document processing logging in 'corpus' ([`5226cd6`](https://github.com/entelecheia/thematos/commit/5226cd657f9281d5f95ccbd8fb9dda0a3305c31b))
* **lda:** Add num_workers to training log message ([`d39d2d8`](https://github.com/entelecheia/thematos/commit/d39d2d87af852199a0eb4f5d85469f26d35a5f78))
* **dependencies:** Upgrade lexikanon to 0.6.0 ([`2983fed`](https://github.com/entelecheia/thematos/commit/2983feda4028fdc07f561a85e19bada1d30a2705))

## v0.10.0 (2023-08-12)

### Feature

* **thematos/datasets:** Add delimiter field in NgramConfig ([`6c0ff28`](https://github.com/entelecheia/thematos/commit/6c0ff28996e44925fd22159f33a2104c2d0c323c))
* **thematos/datasets:** Add ngram config and logging in corpus.py ([`c2ec34b`](https://github.com/entelecheia/thematos/commit/c2ec34bb989f6ba9533622a15723f445fe817eaf))
* **datasets:** Add NgramConfig to __all__ ([`787cdbc`](https://github.com/entelecheia/thematos/commit/787cdbc2cb06d6f4b017c87ba7e1e9266f4847b3))
* **thematos:** Add new configuration for ngrams ([`96708f3`](https://github.com/entelecheia/thematos/commit/96708f354d30a5c1014a22d13542d14ff5ef982c))
* **thematos/conf/dataset:** Add ngrams and timestamp_col fields to topic_corpus.yaml ([`828adbd`](https://github.com/entelecheia/thematos/commit/828adbd99529a21393035f3f365fc1561da13df0))

## v0.9.0 (2023-08-12)

### Feature

* **thematos/models:** Add wordclouds generation method ([`4e866cc`](https://github.com/entelecheia/thematos/commit/4e866ccccbf64d34f60ab079663838e8b036b19c))
* **corpus:** Add function to concatenate n-grams ([`a78edc6`](https://github.com/entelecheia/thematos/commit/a78edc6d6ccf418287ad46322bb0665480f564a9))
* **thematos/datasets:** Add NgramConfig class in ngrams.py ([`10a2e2e`](https://github.com/entelecheia/thematos/commit/10a2e2e6dc199a05809196f9045784192c9306fb))
* **thematos-models:** Add image collage feature for word clouds in TopicModel ([`6953b9f`](https://github.com/entelecheia/thematos/commit/6953b9f03d8cf58eab50e19bd55000f274b811ff))
* **thematos/models/config:** Add new config variables for WordcloudConfig class ([`d88d76e`](https://github.com/entelecheia/thematos/commit/d88d76e322a15ff133cc5cca03a4252624739165))
* **thematos:** Add make_collage and related configs ([`231cebb`](https://github.com/entelecheia/thematos/commit/231cebb61b56c0d871bdb601e49c41f174f4e32b))

## v0.8.0 (2023-08-12)

### Feature

* **thematos/models:** Add WordcloudConfig ([`9919ce9`](https://github.com/entelecheia/thematos/commit/9919ce972e6bfdd2794edef8f6c1795fba927541))
* **thematos/models:** Add WordcloudConfig and generate_wordclouds method ([`118e7e6`](https://github.com/entelecheia/thematos/commit/118e7e6f5f191e4e34cada1e2b903d27744b4549))
* **thematos/models:** Add WordcloudConfig class in config.py ([`198053c`](https://github.com/entelecheia/thematos/commit/198053cb8aee11a60f32539875880941a0e7e54a))
* **thematos/plots:** Add WordCloud module ([`ff24048`](https://github.com/entelecheia/thematos/commit/ff24048040af0089cb03e68574c1276b27e8cc33))
* **tests:** Add wordcloud notebook in tests/notebook ([`b213e41`](https://github.com/entelecheia/thematos/commit/b213e41666b10b197fcb0e0997733afca182de3e))
* **thematos/plots:** Add WordCloud class ([`ecc71e3`](https://github.com/entelecheia/thematos/commit/ecc71e3658c7d017d7c2a0916af3c1e954935616))
* **dependencies:** Add wordcloud version 1.9.2 ([`2c21075`](https://github.com/entelecheia/thematos/commit/2c21075014e5ab98d2574e7f3b6f5b6eee125dad))
* **thematos:** Add defaults for wordcloud in config ([`47ee61f`](https://github.com/entelecheia/thematos/commit/47ee61f36cdfdcd486549ee964c30111404daf34))
* **thematos:** Add wordcloud to topic model configuration ([`66600e2`](https://github.com/entelecheia/thematos/commit/66600e2768c319e05fa597411c3f50dc6be6e9a6))
* **thematos/conf/model:** Add wordcloud to lda model configuration ([`8edb40a`](https://github.com/entelecheia/thematos/commit/8edb40a533514057d39340c3dfce0714055c9183))
* **thematos:** Add wordcloud configuration model ([`b59dade`](https://github.com/entelecheia/thematos/commit/b59dade629bb4771cfd3a1e5562e56a52cc05e47))
* **thematos:** Add wordcloud configuration ([`ca5944e`](https://github.com/entelecheia/thematos/commit/ca5944ec4b20039522772d6d397561c5aab4cbf1))

### Fix

* **thematos/models:** Add loading of topic term distributions ([`50c8bb5`](https://github.com/entelecheia/thematos/commit/50c8bb526f7c9aa5cc5eb2812e92d3a33d13fe28))

## v0.7.0 (2023-08-11)

### Feature

* **thematos:** Implement method for lda visualization saving ([`91322ca`](https://github.com/entelecheia/thematos/commit/91322ca682d1258b1e36576a1e9debebc3534187))

### Fix

* **runner:** Correct typo in configuration file ([`0b23775`](https://github.com/entelecheia/thematos/commit/0b237750bcfeb2ba5363d7c2598d8a39130aeb4a))
* **thematos/runners/topic:** Add save method and implement batch increment ([`b7dfb53`](https://github.com/entelecheia/thematos/commit/b7dfb53be3bbfc2f100aeee97fec4b8a22fe76d8))

## v0.6.0 (2023-08-11)

### Feature

* **thematos/models:** Add update_model_args method ([`1fd68c0`](https://github.com/entelecheia/thematos/commit/1fd68c0e5ae9ad29294558d4532e2178d548c796))
* **runners/config:** Add TopicRunnerResult class ([`8d2b8d4`](https://github.com/entelecheia/thematos/commit/8d2b8d4b2bf39deb295a0553af85e26bf6068baa))
* **thematos/runners:** Add TopicRunnerResult to __all__ ([`1c70d61`](https://github.com/entelecheia/thematos/commit/1c70d61b45b10ac0f7f69076b5c3e55ac93a217c))

### Fix

* **lda:** Add multi-worker support for model training ([`ac386c0`](https://github.com/entelecheia/thematos/commit/ac386c01d7f6dab17b37833baf49d86ab700a991))
* **thematos:** Enable eval_coherence in topic model configuration ([`992f069`](https://github.com/entelecheia/thematos/commit/992f06958b102093cbbe337fad27d6650ddb9d1c))
* **thematos/model:** Enable eval_coherence in lda.yaml ([`8ba0f13`](https://github.com/entelecheia/thematos/commit/8ba0f13e98817207329084e293d1b7a147c85938))

## v0.5.0 (2023-08-11)

### Feature

* **config:** Add new test configuration files for dataset, model, project, runner, words, and workflow ([`8d7b07b`](https://github.com/entelecheia/thematos/commit/8d7b07b09392e37c7d34fa173111d746756fc460))
* **tests/scripts:** Add TopicRunner and LdaRunConfig generate_config calls ([`cfb92e0`](https://github.com/entelecheia/thematos/commit/cfb92e0787ed8a616885cc4902dd9824aac0beab))
* **topic-runner:** Add new topic runner class ([`7ca6987`](https://github.com/entelecheia/thematos/commit/7ca69877d5199b64a324dce406609fdaab4ecd49))
* **runners/config:** Add new lda run configuration ([`4e647be`](https://github.com/entelecheia/thematos/commit/4e647be19204f51f4b1449624c7e74c17074dad6))
* **runners:** Add new files for topic handling and configuration ([`b9ba47f`](https://github.com/entelecheia/thematos/commit/b9ba47f7ca7afc20108b54dc31c3029e4aae1174))
* **runner/config:** Add lda.yaml for thematos.runners.config.LdaRunConfig ([`dc9b733`](https://github.com/entelecheia/thematos/commit/dc9b7330929dc7dba79c2c2cd4c6274b6eefe3ed))
* **thematos:** Add new configuration for topic runner ([`1e872ca`](https://github.com/entelecheia/thematos/commit/1e872cab170842a5c3ef8096a61c5640b5da73d3))

## v0.4.0 (2023-08-11)

### Feature

* **thematos:** Implement loading functions for ll_per_words and document_topic_dists in TopicModel ([`d252b29`](https://github.com/entelecheia/thematos/commit/d252b29776521655e2c9cbff2d3d432a02344027))
* **lda:** Add model loading functionality ([`6f128bd`](https://github.com/entelecheia/thematos/commit/6f128bdc011249add52dda2e486e6e8690651922))

### Fix

* **thematos/models:** Add seed to ModelSummary ([`5c00df2`](https://github.com/entelecheia/thematos/commit/5c00df2da77e458fa01eaabed9d2a1f8d9ca50f3))
* **models/config:** Change IDF value to int, remove seed from TrainConfig ([`f800375`](https://github.com/entelecheia/thematos/commit/f80037573711e194cbc3fb89a613a703dce75067))

## v0.3.0 (2023-08-10)

### Feature

* **thematos/models:** Add config.py ([`af87d17`](https://github.com/entelecheia/thematos/commit/af87d17a65e82d47be8d40577a01b73b07af6f05))
* **thematos/models:** Add methods to save config and dataframes ([`e8579e1`](https://github.com/entelecheia/thematos/commit/e8579e174b2f2cfaa0a4fc9c1077027d23fbd2b4))
* **corpus:** Add task_name attribute ([`cd38432`](https://github.com/entelecheia/thematos/commit/cd384322319b570e09d9a767b9bc023c9453b3cd))
* **tests/scripts:** Add gen_models.py to generate and print model configurations ([`1f3d2ee`](https://github.com/entelecheia/thematos/commit/1f3d2ee9dee90aec5917fa8f93138ee920160922))
* **thematos/models:** Add additional models and ([`d78390c`](https://github.com/entelecheia/thematos/commit/d78390cb3c078395459b676e65894019bdef7975))
* **model:** Add lda, topic and train configurations ([`597d11a`](https://github.com/entelecheia/thematos/commit/597d11a59f1cd93c91d6ccc386f5e6647df0a8c9))
* **thematos/models/base:** Add model_summary property and save_model_summary method ([`22da5ed`](https://github.com/entelecheia/thematos/commit/22da5edffe96609cdc96da3b4c00288d5f631228))
* **thematos/models:** Add min and max prior weight fields in WordPrior class ([`f0cc4c0`](https://github.com/entelecheia/thematos/commit/f0cc4c00def9fa5ed9d9e1df62ebf7531e0f774e))
* **thematos:** Add min_prior_weight and max_prior_weight configurations in wordprior ([`1d76999`](https://github.com/entelecheia/thematos/commit/1d76999b948996cf8ec817c81c8af8317dc53ae8))
* **tests/notebook:** Add wordprior notebook ([`7bd613b`](https://github.com/entelecheia/thematos/commit/7bd613ba2787ecc754f6dd7afde6f95ecb8a620a))
* **tests/notebook:** Add corpus.ipynb ([`88ebc1a`](https://github.com/entelecheia/thematos/commit/88ebc1a029be0283be72c6065028d79d55d19f97))
* **tests:** Add test for WordPrior model in thematos ([`ae3ddd6`](https://github.com/entelecheia/thematos/commit/ae3ddd6b45b02cb4b9bde1c4a9f089eae79aa253))
* **thematos:** Add _topic_corpus_ yaml config in dataset module ([`2454094`](https://github.com/entelecheia/thematos/commit/24540946fdf756414cddb47e995c888010310141))
* **thematos:** Add _wordprior_ configuration for WordPrior model ([`b84a637`](https://github.com/entelecheia/thematos/commit/b84a637cf895098b00ce82c6da17f46f75f11e2b))
* **tests:** Add corpus tests in thematos/datasets ([`b7382cb`](https://github.com/entelecheia/thematos/commit/b7382cb9c3334875394fedc962c68d99b4739cfb))
* **thematos/tasks:** Add topic functionality ([`d671cd4`](https://github.com/entelecheia/thematos/commit/d671cd45118a73446c5f731d3c451e5d52c848f0))
* **tasks:** Add TopicTask ([`6e44b9b`](https://github.com/entelecheia/thematos/commit/6e44b9b5ee66c43cfc47416c45d30837fb349e23))
* **thematos/models:** Add types.py with CoherenceMetrics and ModelSummary classes ([`4f725fb`](https://github.com/entelecheia/thematos/commit/4f725fbcf03df2292c720801505a101d8ba61174))
* **thematos/models:** Add WordPrior class ([`62c579a`](https://github.com/entelecheia/thematos/commit/62c579ab218fad0a685792715e23767b908837c7))
* **thematos/models:** Add lda model ([`cae5427`](https://github.com/entelecheia/thematos/commit/cae54276517c1ba66a79ac8b32b2fa13280cc32a))
* **thematos/models:** Add TopicModel in base.py ([`0e52722`](https://github.com/entelecheia/thematos/commit/0e52722eb324a6afb97db251defd053d003054a1))
* **thematos/models:** Add LdaModel and WordPrior classes ([`f11966b`](https://github.com/entelecheia/thematos/commit/f11966bfe260eb1a9cd0ea67361c9a8fd1d8afd6))
* **datasets:** Add Corpus class in corpus.py module ([`f6a4bc3`](https://github.com/entelecheia/thematos/commit/f6a4bc31ce3701a126e5d4b8860df3733bfdc62c))
* **datasets:** Add Corpus import in __init__.py ([`053d372`](https://github.com/entelecheia/thematos/commit/053d37200d4e9dc53a5d380b6d2a3e0d002333bf))
* **topic:** Initialize module ([`58d2b44`](https://github.com/entelecheia/thematos/commit/58d2b44db644654e13dec9043a161cbe49ce00b1))
* **topic-model:** Add base and lda models ([`f376220`](https://github.com/entelecheia/thematos/commit/f376220aefd0051d51514358f3a1b0a080ce28cc))
* **thematos/topic/types:** Add new types for CoherenceMetrics and ModelSummary ([`bd5a3ae`](https://github.com/entelecheia/thematos/commit/bd5a3ae2427e1c7fe43521abfbe0798cb71d59c7))
* **thematos/topic:** Add a new class WordPrior for managing words and their priors ([`43805d4`](https://github.com/entelecheia/thematos/commit/43805d42af09a436df662979b3215dcbf968b832))
* **thematos/topic:** Add corpus.py ([`d96175d`](https://github.com/entelecheia/thematos/commit/d96175dad644bbbdc9eb1872e85473cb03cc5285))
* **topic:** Add topic task class ([`10297f8`](https://github.com/entelecheia/thematos/commit/10297f8b4a8514c8766ab2c41ec6bde006be0942))
* **pyproject.toml:** Add tomotopy and pyldavis dependencies ([`ec7b3f5`](https://github.com/entelecheia/thematos/commit/ec7b3f5f80de3f6aad01d8334437855c6f3c129f))

### Fix

* **thematos:** Correct 'll_per_words' property type hint ([`27afd5a`](https://github.com/entelecheia/thematos/commit/27afd5acf3a47c8b2b8889d9995d7fe7f9c47b71))
* **thematos:** Rename evaluate_coherence to eval_coherence_value in TopicModel ([`6084550`](https://github.com/entelecheia/thematos/commit/6084550d8b62be3c498d033575b2d4a9b3d73e92))

## v0.2.3 (2023-08-03)

### Fix

* **dependencies:** Upgrade hyfi to 1.17.2 ([`ae81a43`](https://github.com/entelecheia/thematos/commit/ae81a4344ab3eaa54aeaeeeb82b229dd511333b2))

## v0.2.2 (2023-07-30)

### Fix

* **dependencies:** Upgrade hyfi to 1.12.5 ([`aebc82c`](https://github.com/entelecheia/thematos/commit/aebc82cd67fee99ec74eb90378b8cf2464ba10c6))

## v0.2.1 (2023-07-28)

### Fix

* **pyproject.toml:** Upgrade hyfi to ^1.12.0 ([`a3d6c93`](https://github.com/entelecheia/thematos/commit/a3d6c931d4ce3c0b39cff37d5905ee4c3bdba000))

## v0.2.0 (2023-07-27)

### Feature

* **book:** Add new documentation for API reference ([`9ec0cc9`](https://github.com/entelecheia/thematos/commit/9ec0cc90add63e32f6ba3376864882a85039e583))
* **book:** Add sphinx-carousel and sphinxcontrib-lastupdate ([`0b5461f`](https://github.com/entelecheia/thematos/commit/0b5461fb5b42bab507a3df8b869b9f5d65c8b925))
* **book:** Add new features to _config.yml and _toc.yml ([`2a89341`](https://github.com/entelecheia/thematos/commit/2a89341e8234afc62ab2cea84e174e4164a1d3da))
* **.copier-config.yaml:** Update commit version and add new configurations ([`9f57524`](https://github.com/entelecheia/thematos/commit/9f5752429f51225fecb4be0247eb128b24b2f7d8))
* **tasks-extra:** Add new tasks for managing hyfi dependency and configuration, add a variety of tooling tasks for linting, testing, cleaning, building, and more ([`f088483`](https://github.com/entelecheia/thematos/commit/f08848312e6da408e717ae053f79f70a7c9660be))
* **codecov:** Add new codecov.yml configuration file ([`882c2be`](https://github.com/entelecheia/thematos/commit/882c2beefdb4c4358248fd0c255273640ac0dada))
* **github-workflows:** Add source code installation in deploy-docs.yaml ([`bd82a84`](https://github.com/entelecheia/thematos/commit/bd82a84315870c09a7cbe8b56e4d8b7b47a808b3))
* **thematos:** Add about section in thematos.yaml ([`d74728d`](https://github.com/entelecheia/thematos/commit/d74728d94b0383339b066f9bc06b64e71f3afea0))
* **.envrc:** Add new environment file and set up virtual environment ([`f37d07f`](https://github.com/entelecheia/thematos/commit/f37d07f54457db4951b8b6586cc163ba3231220d))

## v0.1.4 (2023-04-25)
### Fix
* Apply updated template ([`a400c5a`](https://github.com/entelecheia/thematos/commit/a400c5a6b5861a651e027b6a4511264a5cfd04bc))

## v0.1.3 (2023-04-21)
### Fix
* **version:** Disable scm-version ([`07058b6`](https://github.com/entelecheia/thematos/commit/07058b67ae77d92a81918f2c0add4ad025b15208))

## v0.1.2 (2023-04-21)
### Fix
* **version:** Add pre-commit command to make scm-version ([`427e4ad`](https://github.com/entelecheia/thematos/commit/427e4ad98c1069895514dbf85a70643468a513f8))

## v0.1.1 (2023-04-21)
### Fix
* Add PYPI TOKEN ([`d57b6c9`](https://github.com/entelecheia/thematos/commit/d57b6c9c0746027ae76a47d6398916ba222bcbde))

## v0.1.0 (2023-04-21)
### Feature
* Initial version ([`6d2d9ee`](https://github.com/entelecheia/thematos/commit/6d2d9eeff97fb2a3efca19784d7cb1f3d77309bb))
