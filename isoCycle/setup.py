from distutils.core import setup
setup(
  name = 'isoCycle',         
  packages = ['isoCycle'],   
  version = '0.0.1',      
  license='CC-BY-NC-SA-4.0',        
  description = 'isolating single cycles in neuronal spiking',   
  author = 'Ehsan Sabri',                   
  author_email = 'ehsan.sabri@gmail.com',      
  url = 'https://github.com/esiabri/isoCycle',  
  download_url = 'https://github.com/esiabri/isoCycle.git',   
  keywords = ['Neuroscience', 'Oscillation', 'Cycle'],  
  install_requires=[            
          'validators',
          'beautifulsoup4',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',     
    'Intended Audience :: Science/Research', 
    'Programming Language :: Python',   
  ],
)