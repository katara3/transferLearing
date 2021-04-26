from setuptools import setup, find_packages

setup(
    name="mdv",
    version="0.1",
    description="Transfer learning for deep image classification",
    packages=find_packages(),

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=['absl-py==0.11.0',
    'astor==0.8.1',
    'cached-property==1.5.2',
    'click==7.1.2',
    'Flask==1.1.2',
    'gast==0.2.2',
    'google-pasta==0.2.0',
    'gpuinfo==1.0.0a7',
    'grpcio==1.33.2',
    'h5py==2.10.0',
    'importlib-metadata==2.0.0',
    'itsdangerous==1.1.0',
    'Jinja2==2.11.2',
    'Keras==2.2.4',
    'Keras-Applications==1.0.8',
    'Keras-Preprocessing==1.1.2',
    'Markdown==3.3.3',
    'MarkupSafe==1.1.1',
    'numpy==1.19.3',
    'opt-einsum==3.3.0',
    'Pillow==6.2.2',
    'protobuf==3.13.0',
    'PyYAML==5.3.1',
    'scipy==1.5.3',
    'six==1.15.0',
    'tensorboard==1.15.0',
    'tensorflow-estimator==1.15.1',
    'tensorflow-gpu==1.15.0',
    'termcolor==1.1.0',
    'Werkzeug==1.0.1',
    'wrapt==1.12.1',
    'zipp==3.4.0',
    'scikit-learn==0.23.2',
    'matplotlib==3.3.3'],
    python_requires='>=3',
    author="DDE",
    # could also include long_description, download_url, classifiers, etc.
    entry_points = {
        'console_scripts': [
            'mdv = mdv.__main__:main'
        ]
    }
)
