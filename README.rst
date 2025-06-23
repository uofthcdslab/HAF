
Human-Aligned Faithfulness in Toxicity Explanations of LLMs
===========================================================

.. image:: https://github.com/uofthcdslab/HAF/blob/main/utils/haf_intro.png
  :align: center
  :width: 400px

The discourse around toxicity and LLMs in NLP largely revolves around detection tasks. This work shifts the focus to evaluating LLMs' *reasoning* about toxicity---from their explanations that justify a stance---to enhance their trustworthiness in downstream tasks. In our recent `paper <arxiv.org>`_, we propose a novel, theoretically-grounded multi-dimensional criterion, **Human-Aligned Faithfulness (HAF)**, that measures how LLMs' free-form toxicity explanations reflect those of a rational human under ideal conditions.
We develop six metrics, based on uncertainty quantification, to comprehensively evaluate HAF of LLMs' toxicity explanations with no human involvement, and highlight how “non-ideal” the explanations are. This respository contains the code and sample data to reproduce our results. 

The complete LLM-generated toxicity explanations and our HAF scores are available in `Hugging Face <https://huggingface.co/collections/uofthcdslab/haf-6857895ac09959da821bd015>`_. The complete LLM output tokens and entropy scores are available upon request.


Requirements:
=============

``pip install -r requirements.txt``


Pipeline:
=========

Quick Demo (with sample data):
------------------------------

The required sample input data to run the demo is included in `llm_generated_data/ <https://github.com/uofthcdslab/HAF/tree/main/llm_generated_data>`_ and `parsed_data/ <https://github.com/uofthcdslab/HAF/tree/main/parsed_data>`_ directories. To compute HAF metrics on this sample data, run the following command:

``python haf.py``

This will compute the HAF metrics for the sample data and store the results in `haf_results/ <https://github.com/uofthcdslab/HAF/tree/main/haf_results>`_ directory. The results include HAF scores for different models and datasets.


Reproducing Full Pipeline:
--------------------------

**Using an existing or a new dataset:**

1. Add the dataset name and path in `utils/data_path_map.json <https://github.com/uofthcdslab/HAF/blob/main/utils/data_path_map.json>`_.
2. Include the main processing function for the dataset in `utils/data_processor.py <https://github.com/uofthcdslab/HAF/blob/main/utils/data_processor.py>`_ and give it the exact same name as the dataset.
3. Access shared parameters and methods defined in the `DataLoader <https://github.com/uofthcdslab/HAF/blob/main/data_loader.py#L8>`_ class in `data_loader.py <https://github.com/uofthcdslab/HAF/blob/main/data_loader>`_ through instance references.


**LLM explanation generation and parsing:**

In the paper, we describe a three-stage pipeline to compute **HAF** metrics. The pipeline consists of:

1. Stage **JUSTIFY** where LLMs generate explanations for their toxicity decisions (denoted by ``stage="initial"``).
2. Stage **UPHOLD-REASON** where LLMs generate post-hoc explanations to assess the sufficiency of reasons provided in the **JUSTIFY** stage (denoted by ``stage="internal"`` or ``stage="external"``).
3. Stage **UPHOLD-STACE** where LLMs generate post-hoc explanations to assess the sufficiency and necessity of individual reasons of **JUSTIFY** stage (denoted by ``stage="individual"``).

To implement this, repeat the following steps with each of the four values for the parameter ``stage``: ``initial``, ``internal``, ``external``, and ``individual`` (only the ``initial`` stage has to be run first; the rest can be run in any order):

1. Run `generate.py <https://github.com/uofthcdslab/HAF/blob/main/generate.py>`_ with ``--generation_stage=initial/internal/external/individual`` and other optional changes to the generation hyperparameters. 
2. LLM outputs (tokens, token entropies, and texts) will be generated and stored in ``llm_generated_data/<model_name>/<data_name>/<stage>``. 
3. Run `parse.py <https://github.com/uofthcdslab/HAF/blob/main/parse.py>`_ with ``stage=initial/internal/external/individual`` and other optional parameters to extract LLM decisions, reasons, and other relevant information for computing HAF.
4. The parsed outputs will be stored in ``parsed_data/<model_name>/<data_name>/<stage>``.


**Computing HAF metrics:**

1. Run `haf.py <https://github.com/uofthcdslab/HAF/blob/main/haf.py>`_ with optional parameters to compute HAF metrics for all combinations of models and datasets.
2. The outputs will be computed for each sample instance and stored in ``haf_results/<model_name>/<data_name>/<sample_index>.pkl``.


Roadmap:
========
1. We are working on updating the parser files to support more datasets and models. We will soon integrate the results of Microsoft Phi-4 reasoning model.
2. We will include the results of naive prompting without explicit reasoning instructions.


Citing:
=======
Bibtex::

	@article{mothilal2025haf,
  		title={Human-Aligned Faithfulness in Toxicity Explanations of LLMs},
  		author={K Mothilal, Ramaravind and Roy, Joanna and Ahmed, Syed Ishtiaque and Guha, Shion},
  		year={2025}
	}
