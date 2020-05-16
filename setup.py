from setuptools import find_packages, setup

dependencies = [
    'tensorflow==2.2.0',
    'numpy==1.18.4',
    'lxml==4.2.6',
    'scikit-image==0.16.2',
    'imgaug==0.2.9'
]

test_dependencies = [
    'pytest==5.4.2'
]

lint_dependencies = [
    'flake8==3.8.1',
    'flake8-import-order==0.18.1',
    'flake8-quotes==3.2.0'
]

setup(
    name='yolo',
    version='0.0.1',
    packages=find_packages(exclude=['test', 'test.*']),
    url='https://github.com/FlorisHoogenboom/keras-yolo',
    license='MIT',
    author='Floris Hoogenboom',
    author_email='floris@digitaldreamworks.nl',
    description='An implementation of YOLOv2 in Keras',
    install_requires=dependencies,
    test_dependencies=test_dependencies,
    extras_require={
        'test': test_dependencies,
        'lint': lint_dependencies
    }
)
