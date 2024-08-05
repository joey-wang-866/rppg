# Sphinx documentation

See this [presentation](https://www.dropbox.com/s/33hdyvduvqsdxzq/Presentation_Nordling_Hedbrant_software_documentation_220526.pptx?dl=0).
Write the documentation of the code following [PEP 8](https://peps.python.org/pep-0008/), [PEP 257](https://peps.python.org/pep-0257/), [PEP 287](https://peps.python.org/pep-0287/) and `reStructuredText` for the docstring format.

## Install and setting up Sphinx in Conda

1.  Open your terminal.

1.  Activate the conda environment:

        conda activate rppg

1.  Install Sphinx using conda:

        conda install sphinx

1.  Create a new Sphinx docs in project:

        sphinx-quickstart

1.  Choose `n` if you are asked whether you want to separate build and source folder.

## Install and using Sphinx theme

1.  Activate environment and install `sphinx_rtd_theme`

        conda activate rppg
        conda install sphinx_rtd_theme

1.  To use the theme, open `conf.py` and set `html_theme` to `sphinx_rtd_theme`

        html_theme = 'sphinx_rtd_theme'

## Update/Build documentation html

_Make sure you have installed sphinx theme. If you found errors, fix any errors and repeat build until there is no error._

In `./docs` directory open terminal and execute

    make html

## Open documentation contents

Navigate to `./docs/_build/html/index.html`
