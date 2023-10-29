# Rule Transform
Rule Transform (RT) mines temporal rules from time series datasets for visualization and classification purposes. <br />
### Original Paper: <br />
O. Bahri, P. Li, S. F. Boubrahimi and S. M. Hamdi, "Shapelet-based Temporal Association Rule Mining for Multivariate Time Series Classification," 2022 IEEE International Conference on Big Data (Big Data), Osaka, Japan, 2022, pp. 242-251, doi: 10.1109/BigData55660.2022.10020478.
### Project website: <br />
https://sites.google.com/view/ruletransform/
### Installation: <br />
python3.6.13 is required. I suggest using conda to create the virtual environment:
```
conda create -n rt python=3.6.13
```
or alternatively, install python3.6.13 and: <br />
```
python3.6.13 -m venv ./rt
source venv/bin/activate
```
Then:<br />
```
pip install git+https://github.com/omarbahri/RuleTransform
```
### Instructions: <br />
The BasicMotions dataset is uploaded to the `data` directory. The other UEA datasets can be downloaded [here](https://timeseriesclassification.com/dataset.php).
The `examples/rule_transform` notebook provides a tutorial on how to run mine rules using RuleTransform, use the rules for classification, and visualize them.
### Citation: <br />
```
@inproceedings{bahri2022,
   title = {Shapelet-based Temporal Association Rule Mining for Multivariate Time Series Classification},
   author = {Omar Bahri and Peiyu Li and Soukaina Filali Boubrahimi and Shah Muhammad Hamdi},
   doi = {10.1109/BigData55660.2022.10020478},
   isbn = {9781665480451},
   journal = {Proceedings of the 2022 IEEE International Conference on Big Data},
   keywords = {multivariate time series mining,shapelets,temporal association rules,time series mining},
   pages = {242-251},
   publisher = {Institute of Electrical and Electronics Engineers Inc.},
   year = {2022},
}
```
