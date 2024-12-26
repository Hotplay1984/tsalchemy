from setuptools import setup, find_packages

setup(
	name="tsalchemy",
	version="0.1.0",
	packages=find_packages(),
	install_requires=[
		'pandas',
		'numpy',
		'matplotlib',
		'scikit-learn',
		'statsmodels',
		'scipy',
		'fastdtw'
	],
	author="Weiyao Sun",
	description="A time series analysis toolkit",
	long_description=open('README.md').read(),
	long_description_content_type="text/markdown",
	python_requires='>=3.7',
) 