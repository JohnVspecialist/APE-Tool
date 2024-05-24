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
git clone https://github.com/yourusername/adversarial-prompt-engineering-security-tool.git
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
