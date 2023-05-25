from setuptools import setup
import os

def read_version():
    version_file = os.path.join(
        os.path.dirname(__file__),
        'isoCycle',
        'version.py'
    )
    with open(version_file, 'r') as f:
        exec(f.read())
    return locals()['__version__']


setup(
  use_incremental=True,
  name = 'isoCycle',         
  packages = ['isoCycle'],   
  version = read_version(),#'0.0.31',      
  license='CC-BY-NC-SA-4.0',        
  description = 'isolating single cycles of oscillatory activity in neuronal spiking',   
  author = 'Ehsan Sabri',                   
  author_email = 'ehsan.sabri@gmail.com',      
  url = 'https://github.com/esiabri/isoCycle',  
  download_url = 'https://github.com/esiabri/isoCycle/archive/refs/tags/0.0.61.tar.gz',   
  keywords = ['Neuroscience', 'Oscillation', 'Cycle'],  
  setup_requires=['setuptools_scm',
                  # 'incremental'
                  ],
  include_package_data=True,
  install_requires=[            
          'numpy',
          'tensorflow',
          'scipy',
          'ipympl',
          'ipywidgets',
          'matplotlib'
      ],
  # package_data={
  #       # If any package contains *.ipynb files, include them:
  #       'isoCycle': ['*.ipynb', 'example/*.', 'model/*.h5', 'example/*.npy']
  #   },
  classifiers=[
    'Development Status :: 2 - Pre-Alpha',     
    'Intended Audience :: Science/Research', 
    'Programming Language :: Python',   
  ],
  long_description=open('README.md').read(),
  long_description_content_type='text/markdown',
)