# LLM required libraries
import boto3
import json
import os
import sys
from botocore.exceptions import ClientError
from .llm_prompts import LLMPrompt

# Set up the module path
module_path = ".."
sys.path.append(os.path.abspath(module_path))

DEFAULT_LLM = "claude-v3.5-sonnet"


class LLMService:
    def __init__(self, 
                 model=DEFAULT_LLM, 
                 version="bedrock-2023-05-31", 
                 max_tokens=8192, #2048, #1024, 
                 temperature=0.5):
        
        self.model_id = self.get_model_id(model)
        self.version = version
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

    # modelId = "anthropic.claude-3-sonnet-20240229-v1:0" i know this one work 
    def get_model_id(self, model) -> str:
        model_mapping = {
            "claude-v2": "anthropic.claude-v2:1",
            "claude-instant-v1": "anthropic.claude-instant-v1",
            "claude-v3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
            "claude-v3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
            "claude-v3-opus": "anthropic.claude-3-opus-20240229-v1:0",
            "claude-v3.5-sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            #"claude-v3.5-sonnet":  "anthropic.claude-3-5-sonnet-20241022-v2:0",# on demand error msg
            "mistral-7b-instruct": "mistral.mistral-7b-instruct-v0:2",
            "mixtral-8x7b-instruct": "mistral.mixtral-8x7b-instruct-v0:1",
            "mistral-large": "mistral.mistral-large-2402-v1:0"
        }
        if model not in model_mapping:
            raise NotImplementedError(f"Model '{model}' not implemented.")
        return model_mapping[model]

    def invoke_model(self, prompt) -> str:

        conversation = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }]
        
        native_request = {
            "anthropic_version": self.version,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": conversation,
        }

        #inference_profile_arn = ""
        try:
            response = self.bedrock_client.invoke_model(modelId=self.model_id, 
                                                        body=json.dumps(native_request) 
                                                        #, inferenceProfileArn=inference_profile_arn 
                                                        )
            model_response = json.loads(response["body"].read())
            return model_response["content"][0]["text"]

        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke '{self.model_id}'. Reason: {e}")
            return ""


    def analyze_results(self, csv_file_path) -> None:
        # 1) Extract sub-issues from given issue
        prompt = LLMPrompt.generate_nfl_analysis_prompt(csv_data=csv_file_path, 
                                     motion_column='_any_motion', 
                                     play_type_column='_is_pass', 
                                     yardage_column='yardsGained', 
                                     time_column='_time_bucket')
        gen_analysis_text = self.invoke_model(prompt)
        gen_analysis_text = gen_analysis_text.strip()
        print(f"Generated Analysis:")
        print(gen_analysis_text)
        print("----------------------------------------")

        return gen_analysis_text
        # 3) Store them in Jira 
        
    
# test code 

# llmservice = LLMService()

# team_name="New England"
# prompt = LLMPrompt.generate_simple_analysis_prompt(team_name)
# result = llmservice.invoke_model(prompt)
# print(f"Analysis of {team_name} team")
# print(result)
# print("-------------------------------------")

        
