"""
The setup.py file is an essential part of packaging and distributing Python Projects.
It is used by setuptools(or distutils in older python versions) to define the configuration 
of our project, such as its meta-data, dependencies, and more
"""

from setuptools import find_packages,setup
from typing import List

def get_requirements()->List[str]:
    """
    This function will return list of requirements
    """
    requirement_list:List[str] = []
    try:
        with open('requirements.txt','r') as file:
            #Read line from the file
            lines = file.readlines()

            ## Process each line
            for line in lines:
                requirement = line.strip()
                ## ignore empty line and -e .
                ## -e . will refer to setup.py file
                if requirement and requirement != '-e .':
                    requirement_list.append(requirement)
    except FileNotFoundError:
        print('requirements.txt file not found')

    return requirement_list

setup(
    name= "Network Security",
    version= "0.0.1",
    author= "Ashish Kumar",
    author_email= "ashish28430@gmail.com",
    packages= find_packages(),
    install_requires= get_requirements()
)