import setuptools 
  
with open("README.md", "r") as fh: 
    description = fh.read() 
  
setuptools.setup( 
    name="t-product", 
    version="0.0.1", 
    author="Ashwin Prabu", 
    packages=["t-product"], 
    description="A package that implements the different operations used and needed for operations involving the t-product.", 
    long_description=description, 
    long_description_content_type="text/markdown", 
    url="https://github.com/Ashwin-Prabu/t-product", 
    license='MIT', 
    install_requires=['numpy', 'scipy'] 
) 