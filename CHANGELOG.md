<!--next-version-placeholder-->

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
