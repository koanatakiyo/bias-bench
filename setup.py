from setuptools import setup

setup(
    name="bias-bench",
    version="0.1.0",
    description="An Empirical Survey of the Effectiveness of Debiasing Techniques for Pre-trained Language Models",
    url="https://github.com/mcgill-nlp/bias-bench",
    packages=["bias_bench"],
    install_requires=[
        # "numpy==1.22.4"
        # "scipy==1.7.3",
        # "scikit-learn==1.0.2",
        # "nltk==3.7.0",
        # "datasets==1.18.3",
        # "accelerate==0.20.3",
 
    ],
    include_package_data=True,
    zip_safe=False,
)
# pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
