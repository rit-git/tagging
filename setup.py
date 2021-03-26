from setuptools import setup, find_packages

setup(
    name='factmine',
    version='0.1',
    author='megagon labs',
    author_mail='jinfeng@megagon.ai',
    url='https://megagon.ai/',
    packages=find_packages(),
    include_package_data=True,
    py_modules=['third/pyfunctor'],
    install_requires=[
        'click', 
        'sklearn',
        'transformers == 4.2.2',
    ],
    python_requires='>=3.6',
    entry_points='''
        [console_scripts]
        tagging=main:cli
     ''',
)

