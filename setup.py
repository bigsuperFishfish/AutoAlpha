from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='autoalpha',
    version='0.1.0',
    author='AutoAlpha Contributors',
    author_email='',
    description='Efficient Hierarchical Evolutionary Algorithm for Mining Alpha Factors',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/bigsuperFishfish/AutoAlpha',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
        'lightgbm>=3.3.0',
        'xgboost>=1.5.0',
        'deap>=1.3.1',
        'numexpr>=2.8.0',
        'joblib>=1.1.0',
        'tqdm>=4.62.0',
        'pydantic>=1.8.0',
    ],
)
