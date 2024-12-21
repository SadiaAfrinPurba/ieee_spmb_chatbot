IEEE SPMB Chat Bot

Follow the steps below to install, configure, and run the application.

Requirements:

Ensure the following prerequisites are met before proceeding:

1. Install Ollama and Llama3.1  
Install the required dependencies by executing the following commands:  

    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull llama3.1

These commands will install the Ollama environment into nedc_130 server and
download the Llama3.1 model (4GB memory is required). This step can be done by the help of the IT team.

Setup Instructions:

1. Activate the Virtual Environment  

Ensure the Python virtual environment is active by running:  
    source venv/bin/activate

If the virtual environment is not created yet, create one using:  

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    

2. Start the Ollama Server  

Make sure the Ollama server is running by starting it with:  
    ollama server

If the server fails to start, check if the Ollama installation was successful or
any other process is running on port 11434.

3. Run the Main Script  

Run the main application script using Streamlit with the following command:  
    streamlit run main.py

This command will start the Streamlit server and open the application in the default web browser.

For issues or support, consult the official Ollama documentation at https://ollama.com/docs.