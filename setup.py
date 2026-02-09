from setuptools import setup, find_packages

setup(
    name="delta_belief_rl",
    version="0.1",
    package_dir={"": "."},
    packages=find_packages(include=["delta_belief_rl"]),
    author='Bethgelab Team',
    author_email='',
    acknowledgements='',
    description='',
    install_requires=[], 
    package_data={'ragen': ['*/*.md']},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
    ]
)
