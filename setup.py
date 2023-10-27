
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='ruletransform',
    version='1.0.0',
    author='Omar Bahri',
    author_email='bahri.o@outlook.com',
    description='Rule Transform algorithm for time series temporal rule mining and classification',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/omarbahri/RuleTransform',
  #  project_urls = {
  #      "Bug Tracker": "https://github.com/omarbahri/RuleTransform/issues"
  #  },
    license='MIT',
    packages=['ruletransform'],
    install_requires=['scikit-learn',
    			'sktime==0.20.0',
    			'numpy==1.24.3',
    			'pandas==1.4.1',
    			'numba==0.58.0',
                'tslearn==0.6.1',
    			'skfeature @ git+https://github.com/jundongl/scikit-feature'],
)

