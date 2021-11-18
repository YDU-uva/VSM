Learning to Learn Variational Semantic Memory
====================================================



The main components of the repository are:

* ``run_classifier.py``: script to run classification experiments on miniImageNet
* ``baseline.py``: script to run classification experiments on Variational prototype network
* ``features.py``: deep neural networks for feature extraction and image generation
* ``inference.py``: amortized inference networks for various versions of Versa
* ``utilities.py``: assorted functions to support the repository

Dependencies
------------
This code requires the following:

*  python 3
* TensorFlow v1.0+

Data
----
For miniImagenet, see the usage instructions in  ``data/save_mini_imagenet_data.py``

Usage
-----

* To run few-shot classification, see the usage instructions at the top of ``run_classifier.py``.



Extending the Model
-------------------

There are a number of ways the repository can be extended:

* **Data**: to use alternative datasets, a class must be implemented to handle the new dataset. The necessary methods for the class are: ``__init__``, ``get_batch``, ``get_image_height``, ``get_image_width``, and ``get_image_channels``. For example signatures see  ``mini_imagenet.py``. Note that the code currently handles only image data. Finally, add the initialization of the class to the file ``data.py``.

* **Feature extractors**: to use alternative feature extractors, simply implement a desired feature extractor in ``features.py`` and change the function call in ``run_classifier.py``. For the required signature of a feature extractor see the function ``extract_features`` in ``features.py``.
