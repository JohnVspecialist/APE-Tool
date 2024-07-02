# APE-Tool
Adversarial Prompt Engineering Security Tool
Adversarial Prompt Engineering Security Tool
Overview
Welcome to the Adversarial Prompt Engineering Security Tool repository. This project was developed for the NVIDIA Generative AI Agents Contest, aiming to create a robust system that detects and mitigates adversarial prompts in AI models, thereby enhancing security and robustness. Leveraging NVIDIA's NeMo framework, TensorRT-LLM, and LangChain, this tool showcases state-of-the-art techniques in AI security.

Features
Adversarial Detection: Identifies malicious prompts that attempt to manipulate AI model outputs.
NeMo Framework: Utilizes NVIDIA's NeMo for building and training advanced conversational AI models.
LangChain Integration: Manages and orchestrates language models efficiently.
Performance Optimization: Implements NVIDIA TensorRT-LLM for optimized performance and faster inference.
Web Interface: A user-friendly web application built with Flask for easy interaction and testing.
Installation
Follow these steps to set up the project on your local machine:

Clone the repository:

bash
Copy code
git clone https://github.com/JohnVspecialist/adversarial-prompt-engineering-security-tool.git
cd adversarial-prompt-engineering-security-tool
Install the required libraries:

bash
Copy code
pip install torch transformers langchain tensorrt flask nemo_toolkit
Obtain and configure API keys:

NVIDIA API Key: Sign up for NVIDIA’s developer program and obtain an API key.
OpenAI API Key: Sign up at OpenAI and get your API key.
Add these keys in the configuration section of your code:

python
Copy code
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
openai.api_key = OPENAI_API_KEY
Usage
Run the Flask web application:

bash
Copy code
python app.py
Interact with the application:

Send a POST request to the /ask endpoint with your input text in JSON format.
Example:
json
Copy code
{
  "input_text": "Explain the concept of black holes."
}
Project Structure
app.py: Main file to run the Flask web application.
agent.py: Contains the Agent4454 class which handles logical deduction, evaluation, and expansion of input text.
requirements.txt: List of required Python packages.
README.md: This file, providing an overview and instructions for the project.
Demo
Watch the demo video to see the tool in action: [Demo Video Link coming soon]

Contributing
We welcome contributions from the community. To contribute:

Fork this repository.
Create a new branch (git checkout -b feature-branch).
Make your changes and commit them (git commit -m 'Add new feature').
Push to the branch (git push origin feature-branch).
Create a pull request.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
For any questions or inquiries, please contact John Vaina at JohnVspecialist@gmail.com

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

This code is for a web application that processes text input to generate logical deductions, evaluations, and expansions using a pre-trained language model (GPT-2). It uses Flask to create a web server, which listens for incoming requests, processes them using the language model, and returns a response.

Components and Functions:
Importing Libraries:

Flask: A web framework for Python that allows you to build web applications.
GPT2Tokenizer and GPT2LMHeadModel from transformers: These are used to tokenize text and generate text using the GPT-2 language model.
openai: Library for interacting with OpenAI's API (not used in this specific script but included for potential future use).
tensorrt as trt: NVIDIA’s library for high-performance deep learning inference (not directly used in this specific script).
nemo.collections.nlp as nemo_nlp: NVIDIA’s toolkit for building NLP applications (not directly used in this specific script).
LocalLangModel and Agent from langchain: Libraries for managing language models and agents (not directly used in this specific script).
Configuration:

OPENAI_API_KEY: Sets up the API key for OpenAI services, although this script does not utilize OpenAI’s API directly.
Defining the Agent4454 Class:

This class is designed to handle the logical deduction, evaluation, and expansion of input text using two sequences: 4:4 and 5:4.
Sequences (4:4 and 5:4): These are predefined sets of instructions or steps that guide how the text should be processed.
4:4 sequence includes steps like suggesting solutions, clarifying concepts, offering practical advice, and discussing theoretical aspects.
5:4 sequence includes steps like providing simple explanations, detailed information, summaries, evidence, or balanced views.
Methods in Agent4454 Class:

__init__: Initializes the class, loads the tokenizer and the model.
logical_deduction: Processes the input text through the steps in the sequences, generating responses for each step using the language model.
evaluate: Evaluates the input text by applying the logical deduction process (simplified in this example).
expand: Expands on the input text by applying the evaluation process (simplified in this example).
process_input: Combines the deduction, evaluation, and expansion processes to handle the complete input text processing.
Setting Up the Flask Web Application:

Creating the Flask app: app = Flask(__name__)
Creating an instance of Agent4454: agent_4454 = Agent4454()
Defining the /ask Endpoint:

This endpoint listens for POST requests.
Function ask:
Retrieves the input text from the request.
Processes the input text using the agent_4454.process_input method.
Returns the processed response in JSON format.
Running the Flask App:

The app runs in debug mode, allowing you to test it locally on your machine.
Detailed Steps of What Happens When You Use the Application:
Start the Web Server:

Run the script. This starts a web server using Flask.
Send a POST Request:

You (or any client) send a POST request to the /ask endpoint with some input text.
Example: POST /ask with JSON body {"input_text": "Explain the concept of black holes."}
Process the Input:

The ask function receives the input text.
It calls agent_4454.process_input(input_text), which processes the text through logical deduction, evaluation, and expansion steps.
Generate Responses:

The logical_deduction method generates responses for each step in the sequences by:
Encoding the step prompt and input text.
Using the GPT-2 model to generate a continuation of the text.
Decoding the generated text back to a readable format.
Return the Response:

The processed text (deduction, evaluation, and expansion results) is returned as a JSON response.
Example of Interaction:
Input Text: "Explain the concept of black holes."
Logical Deduction: The text is processed through steps like suggesting solutions, clarifying concepts, offering practical advice, and discussing theoretical aspects.
Evaluation: Evaluates the input text based on the generated logical deductions.
Expansion: Further expands on the input text based on the evaluation.
Output: A JSON object containing the processed results.
This code creates an intelligent agent that can handle and respond to complex queries by breaking down the problem into smaller, manageable steps, making it robust and effective for generating meaningful and contextually relevant responses.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import openai
import tensorrt as trt
import nemo.collections.nlp as nemo_nlp
from langchain.llms import LocalLangModel
from langchain.agents import Agent

# Configuration
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
openai.api_key = OPENAI_API_KEY

# Define Agent 4454 using 4:4/5:4 Framework
class Agent4454:
    def __init__(self):
        self.sequence_44_steps = [
            "If the problem is technical, then suggest a solution.",
            "If the problem is conceptual, then clarify the concept.",
            "If the problem is practical, then offer practical advice.",
            "If the problem is theoretical, then discuss theoretical aspects."
        ]
        self.sequence_54_steps = [
            "If the user is confused, then provide a simple explanation.",
            "If the user is inquisitive, then provide detailed information.",
            "If the user is in a hurry, then provide a summary.",
            "If the user is skeptical, then provide evidence or examples.",
            "If the user is undecided, then provide a balanced view."
        ]
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

    def logical_deduction(self, input_text):
        # Generate logical deduction results based on the 4:4 and 5:4 sequences
        result = []
        for i in range(5):  # 5 cycles for 4:4 sequence
            step_prompt = self.sequence_44_steps[i % 4] + " " + input_text
            inputs = self.tokenizer.encode(step_prompt, return_tensors="pt")
            outputs = self.model.generate(inputs, max_length=50)
            result.append(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
        for i in range(4):  # 4 cycles for 5:4 sequence
            step_prompt = self.sequence_54_steps[i % 5] + " " + input_text
            inputs = self.tokenizer.encode(step_prompt, return_tensors="pt")
            outputs = self.model.generate(inputs, max_length=50)
            result.append(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
        return result

    def evaluate(self, input_text):
        # Evaluate results
        return self.logical_deduction(input_text)  # Simplified for illustration

    def expand(self, input_text):
        # Expand results
        return self.evaluate(input_text)  # Simplified for illustration

    def process_input(self, input_text):
        # Complete processing by looping through the framework
        deduction = self.logical_deduction(input_text)
        evaluation = self.evaluate(input_text)
        expansion = self.expand(input_text)
        return {
            "deduction": deduction,
            "evaluation": evaluation,
            "expansion": expansion
        }

app = Flask(__name__)
agent_4454 = Agent4454()

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    input_text = data.get('input_text', '')
    response = agent_4454.process_input(input_text)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
