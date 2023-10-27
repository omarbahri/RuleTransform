
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
    install_requires=['scikit-learn==0.24.2',
    			'sktime==0.9.0',
    			'numpy==1.19.3',
    			'pandas==1.1.5',
    			'numba==0.53.1',
                'tslearn==0.5.2',
                'scipy==1.5.4',
    			'skfeature @ git+https://github.com/jundongl/scikit-feature'],
)

