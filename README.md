# MORF Replication of Xing et al. (2016)

This repository contains the code, dockerfile, and analysis scripts used to execute and evaluate an end-to-end replication of Xing, Chen, Stein, and Marcinowski (2016), *Temporal predication of dropouts in MOOCs: Reaching the low hanging fruit through stacking generalization*.

This experiment was executed using an early version of the MOOC Replication Framework (MORF). The experiment is described in Gardner, Brooks, Andres, and Baker, Replicating MOOC Predictive Models at Scale, *Proceedings of the Fifth Annual Meeting of the ACM Conference on Learning@Scale*; June 2018; London, UK.

Because of the immense computational load of running this replication (over 1,588 models, including structure-learning of large Bayesian networks), we do not currently support the execution of the complete replication on the MORF platform. However, individual components of the pipeline (e.g. individual feature sets and models) can be reconstructed using the code provided in this repository. The authors are also available for collaboration or consultation on this code; you can find information on how to contact us on the [MORF website](https://educational-technology-collective.github.io/morf/).

## Repository contents

* `config.properties`: a configuration file to execute the job on MORF.
* `feature_extraction`: Python scripts to extract set of weekly features from raw data, according to three methods (week-only, appended, and summed) described in Xing et al. (2016).
* `modeling`: R scripts to generate Bayesian networks, classification trees, and a logistic regression meta-learner which ensembles the two models.
* `analysis`: various scripts to evaluate results, as well as session-level prediction results from the experiment.

