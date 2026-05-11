Installation
============

First, clone the repository of spora-bench and install its requirements. This
will automatically install spora-io as well:

.. code-block:: bash

   git clone https://github.com/bunnelab/spora-bench.git
   pip install -r requirements.txt

Finally, make sure the foundation models you want to benchmark are set up in
those repositories as well. For installation instructions, refer to the
model-specific projects:

1. VirTues: https://github.com/bunnelab/virtues
2. KRONOS: https://github.com/mahmoodlab/KRONOS

spora [bench] can also be used to evaluate new foundation models. If you want
that, follow the later section in the README for the model integration notes.
