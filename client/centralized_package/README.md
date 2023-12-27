# Client Centralized code modules

This is a package with the client centralized learning code modules.   
The modules here should be imported in the Federated-Learning code.   

It is required to install the libraries that are listed in the requirements.txt file

        sh-4.2$ cd SageMaker/
        sh-4.2$ pwd
        /home/ec2-user/SageMaker
        sh-4.2$ ls
        client  client-1  client-2  client-3  data  flwr_server.ipynb
        sh-4.2$ cd client
        sh-4.2$ ls
        centralized_package  client_centralized.ipynb  data
        sh-4.2$ cd centralized_package/
        sh-4.2$ ls
        client_centralized.ipynb  client_centralized.py  __init__.py  requirements.txt

        sh-4.2$ which python
        ~/anaconda3/envs/JupyterSystemEnv/bin/python

        # INSTALL THE REQUIREMENTS:
        sh-4.2$ pip install -r requirements.txt
        
        # To run the client_centralized code as from 'main':
        sh-4.2$ python client_centralized.py

To import custom modules in notebook:   
You can add a __init__.py file to your package directory to make it a Python package.    
Then you will be import the modules from the package inside your Jupyter notebook.   

        /home/ec2-user/SageMaker
            -- Notebook.ipynb 
            -- mypackage
                -- __init__.py
                -- mymodule.py

Contents of Notebook.ipynb   

        from mypackage.mymodule import SomeClass, SomeOtherClass      
        
For more details, see https://docs.python.org/3/tutorial/modules.html#packages      




